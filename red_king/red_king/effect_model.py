"""red_king alternative: a SUPERVISED treatment-effect model.

Instead of relying on the RSSM's implicit action-conditioning, directly learn
effect(state, arm) -> incremental GP from the randomized windows (IPW-weighted),
then pick the per-customer best arm. Validate against the STORED true optimal_arm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[2]
CFM = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"


def build():
	import duckdb
	pc = duckdb.connect(str(CFM), read_only=True)
	try:
		anch = pc.execute("SELECT customer_key, anchor_epoch, embedding FROM anchor_embeddings "
						  "ORDER BY customer_key, anchor_epoch").pl()
		splits = pc.execute("SELECT customer_key, split FROM encoder_samples").pl()
	finally:
		pc.close()
	B = set(splits.filter(splits["split"] == "B")["customer_key"].to_list())
	anch = anch.filter(anch["customer_key"].is_in(list(B)))
	sc = duckdb.connect(str(STREAM), read_only=True)
	try:
		snd = sc.execute("SELECT customer_id k, epoch(CAST(send_ts AS TIMESTAMPTZ)) t, arm, propensity "
						 "FROM email_sends WHERE arm IS NOT NULL").pl()
		inc = sc.execute("""
			SELECT s.customer_id k, epoch(CAST(o.order_ts AS TIMESTAMPTZ)) t, o.gross_margin gm
			FROM email_sends s JOIN orders o
			  ON o.customer_id=s.customer_id AND o.session_id=s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked=1""").pl()
		opt = sc.execute("SELECT customer_id k, optimal_arm FROM email_arm").pl()
	finally:
		sc.close()
	optmap = dict(zip(opt["k"].to_list(), opt["optimal_arm"].to_list()))
	sby = {}
	for r in snd.iter_rows(named=True):
		sby.setdefault(r["k"], []).append((float(r["t"]), int(r["arm"]), float(r["propensity"])))
	iby = {}
	for r in inc.iter_rows(named=True):
		iby.setdefault(r["k"], []).append((float(r["t"]), float(r["gm"])))
	X, Y, Ax, W, keys = [], [], [], [], []
	rows = list(anch.iter_rows(named=True))
	i, n = 0, len(rows)
	while i < n:
		k = rows[i]["customer_key"]; j = i
		while j < n and rows[j]["customer_key"] == k:
			j += 1
		grp = rows[i:j]
		io = iby.get(k, []); it = np.array([x[0] for x in io]); ig = np.array([x[1] for x in io])
		if len(grp) >= 2:
			for m in range(len(grp) - 1):
				t0 = float(grp[m]["anchor_epoch"]); t1 = float(grp[m + 1]["anchor_epoch"])
				win = [(a, b, c) for (a, b, c) in sby.get(k, []) if t0 < a <= t1]
				if not win:
					continue
				arm = max(set(x[1] for x in win), key=[x[1] for x in win].count)
				prop = max(sum(x[2] for x in win) / len(win), 1e-3)
				r = float(ig[(it > t0) & (it <= t1)].sum()) if len(io) else 0.0
				X.append(grp[m]["embedding"]); Ax.append(arm); Y.append(r); W.append(1.0 / prop)
				keys.append(k)
		i = j
	return (np.array(X, np.float32), np.array(Ax, np.int64), np.array(Y, np.float32),
			np.array(W, np.float32), keys, optmap)


def run(seed=0, K=5, steps=3000):
	import torch
	import torch.nn as nn
	X, A, Y, W, keys, optmap = build()
	nA = int(A.max()) + 1
	dim = X.shape[1]
	dev = "cuda" if torch.cuda.is_available() else "cpu"
	Xtr = torch.tensor(X, device=dev); Atr = torch.tensor(A, device=dev)
	Ytr = torch.tensor(Y / (Y.std() + 1e-6), device=dev)
	Wtr = torch.tensor(W / W.mean(), device=dev)
	class M(nn.Module):
		def __init__(self):
			super().__init__()
			self.net = nn.Sequential(nn.Linear(dim + nA, 512), nn.ReLU(),
									 nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, nA))
		def out(self, s, oh):
			return self.net(torch.cat([s, oh], -1))
	ens = [M().to(dev) for _ in range(K)]
	opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in ens]
	for m, opt in zip(ens, opts):
		torch.manual_seed(seed)
		for _ in range(steps):
			b = torch.randint(0, len(X), (512,), device=dev)
			oh = nn.functional.one_hot(Atr[b], nA).float()
			pred = (m.out(Xtr[b], oh) * oh).sum(-1)
			loss = (Wtr[b] * (pred - Ytr[b]) ** 2).mean()
			opt.zero_grad(); loss.backward(); opt.step()
	# per-customer: use the LAST window's state; predict per-arm, argmax
	last = {}
	for i, k in enumerate(keys):
		last[k] = i
	ks = list(last)
	sidx = np.array([last[k] for k in ks])
	S = torch.tensor(X[sidx], device=dev)
	with torch.no_grad():
		grids = []
		for m in ens:
			rows = []
			for a in range(nA):
				oh = nn.functional.one_hot(torch.full((len(ks),), a, device=dev, dtype=torch.long), nA).float()
				rows.append((m.out(S, oh) * oh).sum(-1))
			grids.append(torch.stack(rows, 0))
		V = torch.stack(grids).mean(0).cpu().numpy()      # (nA, n_cust)
	pred = V.argmax(0)
	truth = np.array([optmap.get(k, -1) for k in ks])
	mask = truth >= 0
	acc = float((pred[mask] == truth[mask]).mean())
	from collections import Counter
	maj = Counter(truth[mask]).most_common(1)[0][0]
	maj_acc = float((np.full(mask.sum(), maj) == truth[mask]).mean())
	return {"windows": len(X), "customers": len(ks), "nA": nA,
			"per_customer_rank_acc": round(acc, 4),
			"majority_baseline": round(maj_acc, 4),
			"beats_majority": bool(acc > maj_acc)}


if __name__ == "__main__":
	print("== red_king EFFECT MODEL (supervised treatment effect) ==")
	for k, v in run().items():
		print(f"  {k:24s}: {v}")
