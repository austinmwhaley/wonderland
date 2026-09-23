"""Same-estimand test: does red_king's model-based estimate beat the confounded
baseline at recovering the TRUE causal arm effect?

Estimand (identical for all methods): E[ incremental gross margin | do(arm=a) ].
Ground truth: IPW using the LOGGED propensity (valid because arms are randomized).

  WITHOUT red_king : naive observed mean  E[g | arm=a]        (confounded)
  WITH    red_king : ensemble model of (donor state, arm) -> g, averaged over the
                     whole population  E_s[ f(s, a) ]          (deconfounded)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[2]
CFM_PRODUCTS = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
STREAM_DB = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"


def load():
	import duckdb
	import polars as pl
	sc = duckdb.connect(str(STREAM_DB), read_only=True)
	try:
		arm = sc.execute("SELECT customer_id, arm, propensity FROM email_arm").pl()
		inc = sc.execute("""
			SELECT s.customer_id, SUM(o.gross_margin) AS g
			FROM email_sends s JOIN orders o
			  ON o.customer_id = s.customer_id AND o.session_id = s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked = 1 GROUP BY 1
		""").pl()
	finally:
		sc.close()
	pc = duckdb.connect(str(CFM_PRODUCTS), read_only=True)
	try:
		emb = pc.execute("""
			SELECT customer_key, embedding FROM (
				SELECT customer_key, anchor_epoch, embedding,
				       row_number() OVER (PARTITION BY customer_key ORDER BY anchor_epoch DESC) rn
				FROM anchor_embeddings) WHERE rn = 1
		""").pl()
	finally:
		pc.close()
	df = (emb.rename({"customer_key": "customer_id"})
			 .join(arm, on="customer_id", how="inner")
			 .join(inc, on="customer_id", how="left")
			 .with_columns(pl.col("g").fill_null(0.0)))
	return df


def run(seed=0, K=5, steps=2000):
	import torch
	import torch.nn as nn
	df = load()
	ei = df["embedding"].to_list()
	n = len(ei)
	dim = len(ei[0])
	S = np.zeros((n, dim), np.float32)
	for i, e in enumerate(ei):
		S[i] = e
	A = df["arm"].to_numpy().astype(np.int64)
	E = df["propensity"].to_numpy().astype(np.float64)
	g = df["g"].to_numpy().astype(np.float64)
	nA = int(A.max()) + 1
	# ground truth: IPW per arm
	ipw = {}
	naive = {}
	for a in range(nA):
		m = A == a
		ipw[a] = float((g[m] / E[m]).sum() / np.maximum((1.0 / E[m]).sum(), 1e-9))
		naive[a] = float(g[m].mean())
	# model-based (deconfounded by averaging predictions over the population)
	R = g / (g.std() + 1e-6)
	dev = "cuda" if torch.cuda.is_available() else "cpu"
	St = torch.tensor(S, device=dev); At = torch.tensor(A, device=dev)
	Rt = torch.tensor(R, dtype=torch.float32, device=dev)
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
			b = torch.randint(0, n, (512,), device=dev)
			oh = nn.functional.one_hot(At[b], nA).float()
			pred = (m.out(St[b], oh) * oh).sum(-1)
			loss = nn.functional.mse_loss(pred, Rt[b])
			opt.zero_grad(); loss.backward(); opt.step()
	gstd = g.std() + 1e-6
	P = {}
	with torch.no_grad():
		for a in range(nA):
			oh = nn.functional.one_hot(torch.full((n,), a, device=dev), nA).float()
			vals = torch.stack([(m.out(St, oh) * oh).sum(-1) for m in ens]).mean(0)
			P[a] = vals.cpu().numpy() * gstd                      # per-row f(s,a)
	P_obs = np.array([P[int(A[i])][i] for i in range(n)])          # f(s_i, a_i)
	mb = {a: float(P[a].mean()) for a in range(nA)}               # model-based avg
	# doubly-robust: MB + IPW correction (unbiased, model reduces variance)
	dr = {a: float(np.mean(P[a] + (A == a) / E * (g - P_obs))) for a in range(nA)}
	err = lambda d: float(np.sqrt(np.mean([(d[a] - ipw[a]) ** 2 for a in range(nA)])))
	print("== SAME-ESTIMAND CAUSAL RECOVERY (E[incremental GP | do(arm)]) ==")
	print(f"{'arm':>4} {'IPW(truth)':>11} {'naive':>10} {'MB':>10} {'DR':>10}")
	for a in range(nA):
		print(f"{a:>4} {ipw[a]:>11.2f} {naive[a]:>10.2f} {mb[a]:>10.2f} {dr[a]:>10.2f}")
	print(f"\nRMSE vs truth:  naive={err(naive):.2f}  MB(red_king)={err(mb):.2f}  DR={err(dr):.2f}")
	for nm, d in (("truth", ipw), ("naive", naive), ("MB", mb), ("DR", dr)):
		print(f"  best arm [{nm:>5}]: {int(max(d, key=d.get))}")
	return {"ipw": ipw, "naive": naive, "mb": mb, "dr": dr,
			"rmse_naive": err(naive), "rmse_mb": err(mb), "rmse_dr": err(dr)}


if __name__ == "__main__":
	run()
