"""red_king — RSSM latent world model on frozen donor states (module-level, reusable).

Trains on sample-B anchor trajectories (Layer B state table) with action = the
randomized email arm (causal treatment) and reward = incremental gross margin.
Exposes `rollout_arm_values`: imagine discounted incremental GP under each arm
from a state, with ensemble uncertainty -> a calibrated LOWER BOUND for red_queen
and white_queen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[2]
CFM = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "red_king_rssm.pt"
GAMMA_DAY = 0.999


def build_sequences():
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
		arm = sc.execute("SELECT customer_id AS customer_key, epoch(CAST(send_ts AS TIMESTAMPTZ)) t, "
						 "arm, propensity FROM email_sends WHERE arm IS NOT NULL").pl()
		inc = sc.execute("""
			SELECT s.customer_id AS customer_key, epoch(CAST(o.order_ts AS TIMESTAMPTZ)) t, o.gross_margin gm
			FROM email_sends s JOIN orders o
			  ON o.customer_id=s.customer_id AND o.session_id=s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked=1""").pl()
	finally:
		sc.close()
	sby = {}
	for r in arm.iter_rows(named=True):
		sby.setdefault(r["customer_key"], []).append(
			(float(r["t"]), int(r["arm"]), float(r["propensity"])))
	iby = {}
	for r in inc.iter_rows(named=True):
		iby.setdefault(r["customer_key"], []).append((float(r["t"]), float(r["gm"])))
	rows = list(anch.iter_rows(named=True))
	i, n = 0, len(rows); seqs = []
	while i < n:
		k = rows[i]["customer_key"]; j = i
		while j < n and rows[j]["customer_key"] == k:
			j += 1
		grp = rows[i:j]
		io = iby.get(k, []); it = np.array([x[0] for x in io]); ig = np.array([x[1] for x in io])
		if len(grp) >= 2:
			S = [grp[m]["embedding"] for m in range(len(grp))]
			A, R, DT, D, P = [], [], [], [], []
			for m in range(len(grp)):
				t0 = float(grp[m]["anchor_epoch"])
				t1 = float(grp[m + 1]["anchor_epoch"]) if m < len(grp) - 1 else t0
				win = [(tt, aa, pp) for (tt, aa, pp) in sby.get(k, []) if t0 < tt <= t1]
				if win:
					aa_ = [x[1] for x in win]
					A.append(int(max(set(aa_), key=aa_.count)))
					P.append(max(sum(x[2] for x in win) / len(win), 1e-3))
				else:
					A.append(0); P.append(1.0)
				R.append(float(ig[(it > t0) & (it <= t1)].sum()) if len(io) else 0.0)
				DT.append((t1 - t0) / 86400.0)
				D.append(1.0 if m == len(grp) - 1 else 0.0)
			seqs.append((S, A, R, DT, D, P))
		i = j
	T = max(len(s[0]) for s in seqs); dim = len(seqs[0][0][0]); N = len(seqs)
	S = np.zeros((N, T, dim), np.float32); A = np.zeros((N, T), np.int64)
	R = np.zeros((N, T), np.float32); DT = np.zeros((N, T), np.float32)
	D = np.zeros((N, T), np.float32); M = np.zeros((N, T), np.float32)
	Pp = np.ones((N, T), np.float32)
	for b, (s, a, r, dt, d, p) in enumerate(seqs):
		L = len(s)
		S[b, :L] = np.asarray(s, np.float32); A[b, :L] = a
		R[b, :L] = r; DT[b, :L] = dt; D[b, :L] = d; M[b, :L] = 1.0
		Pp[b, :L] = p
	return S, A, R, DT, D, M, 4, Pp


def make_rssm(nA, dim, zdim=32, hdim=256):
	import torch
	import torch.nn as nn

	class RSSM(nn.Module):
		def __init__(self):
			super().__init__()
			self.emb_a = nn.Embedding(nA, 32)
			self.cell = nn.GRUCell(zdim + 32 + 1, hdim)
			self.prior = nn.Linear(hdim, 2 * zdim)
			self.post = nn.Linear(hdim + dim, 2 * zdim)
			self.dec = nn.Sequential(nn.Linear(hdim + zdim, 512), nn.ReLU(), nn.Linear(512, dim))
			self.rew = nn.Sequential(nn.Linear(hdim + zdim, 256), nn.ReLU(), nn.Linear(256, 1))
			self.con = nn.Sequential(nn.Linear(hdim + zdim, 256), nn.ReLU(), nn.Linear(256, 1))

		def split(self, o):
			mu, lv = o.chunk(2, -1)
			return mu, lv.clamp(-6, 4)

		def step(self, h, z, a, dt, s):
			h = self.cell(torch.cat([z, self.emb_a(a), dt], -1), h)
			pmu, plv = self.split(self.prior(h))
			qmu, qlv = self.split(self.post(torch.cat([h, s], -1)))
			return h, (pmu, plv), (qmu, qlv)

		def obs(self, h, z):
			hz = torch.cat([h, z], -1)
			return self.dec(hz), self.rew(hz), self.con(hz).squeeze(-1)
	return RSSM(), zdim, hdim


def train(seed=0, K=3, zdim=32, hdim=256, steps=4000):
	import torch
	S, A, R, DT, D, M, nA, Pp = build_sequences()
	N, T, dim = S.shape
	dev = "cuda" if torch.cuda.is_available() else "cpu"
	rstd = R.std() + 1e-6
	St = torch.tensor(S, device=dev); At = torch.tensor(A, device=dev)
	Rt = torch.tensor(R / rstd, dtype=torch.float32, device=dev)
	DTt = torch.tensor(DT, dtype=torch.float32, device=dev).unsqueeze(-1)
	Dt = torch.tensor(D, dtype=torch.float32, device=dev); Mt = torch.tensor(M, device=dev)
	# IPW weights, clipped to stabilise under confounding (control variance)
	_frac = 1.0 / np.clip(Pp, 1e-3, None)
	_cap = float(np.quantile(_frac, 0.95))
	Pt = torch.tensor(np.clip(_frac, None, _cap), dtype=torch.float32, device=dev)
	ens = []
	for _ in range(K):
		m, zd, hd = make_rssm(nA, dim, zdim, hdim); ens.append(m.to(dev))
	opts = [torch.optim.Adam(m.parameters(), lr=2e-3) for m in ens]
	rs = np.random.default_rng(seed); free = 0.5
	for m, opt in zip(ens, opts):
		torch.manual_seed(seed)
		for _ in range(steps):
			b = torch.tensor(rs.integers(0, N, 256), device=dev)
			Ls = min(int(Mt[b].sum(1).max().item()), T)
			h = torch.zeros(len(b), hdim, device=dev); z = torch.zeros(len(b), zdim, device=dev)
			rec = rew = kl = ce = 0.0
			for t in range(Ls):
				s_t = St[b, t]
				h, (pmu, plv), (qmu, qlv) = m.step(h, z, At[b, t], DTt[b, t], s_t)
				qz = qmu + torch.exp(0.5 * qlv) * torch.randn_like(qmu)
				s_hat, r_hat, c_hat = m.obs(h, qz)
				mt = Mt[b, t]
				rec = rec + (mt * ((s_hat - s_t) ** 2).mean(-1)).sum()
				wt = Pt[b, t] / (Pt[b].mean() + 1e-6)          # IPW: deconfound
				rew = rew + (mt * wt * (r_hat.squeeze(-1) - Rt[b, t]) ** 2).sum()
				kl_t = 0.5 * (torch.exp(qlv - plv) + (pmu - qmu) ** 2 / torch.exp(plv)
							  - 1 + plv - qlv).sum(-1)
				kl = kl + (mt * torch.clamp(kl_t - free, min=0.0)).sum()
				ce = ce + (mt * torch.nn.functional.binary_cross_entropy_with_logits(
					c_hat, Dt[b, t], reduction="none")).sum()
				z = qz
			denom = Mt[b].sum() + 1e-6
			loss = (rec + rew + 0.5 * kl + ce) / denom
			opt.zero_grad(); loss.backward()
			torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
	with torch.no_grad():
		evt = torch.tensor(np.arange(0, N, 5), device=dev)
		h = torch.zeros(len(evt), hdim, device=dev); z = torch.zeros(len(evt), zdim, device=dev)
		pr, tr, cc = [], [], []
		for t in range(T):
			s_t = St[evt, t]
			h, (pmu, plv), (qmu, qlv) = ens[0].step(h, z, At[evt, t], DTt[evt, t], s_t)
			s_hat, r_hat, _ = ens[0].obs(h, qmu)
			mk = Mt[evt, t] > 0
			pr.append((r_hat.squeeze(-1) * rstd)[mk]); tr.append(Rt[evt, t][mk] * rstd)
			cc.append(torch.nn.functional.cosine_similarity(s_hat, s_t, dim=-1)[mk]); z = qmu
		PR = torch.cat(pr).cpu().numpy(); TRt = torch.cat(tr).cpu().numpy()
		r2 = 1 - ((PR - TRt) ** 2).sum() / (((TRt - TRt.mean()) ** 2).sum() + 1e-9)
		cosm = float(torch.cat(cc).mean())
	OUT.parent.mkdir(parents=True, exist_ok=True)
	torch.save({"ens": [m.state_dict() for m in ens], "hdim": hdim, "zdim": zdim,
				"nA": nA, "dim": dim, "rstd": float(rstd)}, OUT)
	return {"sequences": int(N), "T": int(T), "dim": int(dim),
			"reward_r2": float(r2), "next_state_cos": cosm, "out": str(OUT)}


def rollout_arm_values(states, horizon=6, gamma=GAMMA_DAY, path=OUT, batch=512):
	"""Imagine discounted incremental GP under each arm; return (mean, std) per arm."""
	import torch
	blob = torch.load(path, map_location="cpu", weights_only=False)
	nA, dim, zdim, hdim = blob["nA"], blob["dim"], blob["zdim"], blob["hdim"]
	rstd = blob.get("rstd", 1.0)
	ens = []
	for sd in blob["ens"]:
		m, _, _ = make_rssm(nA, dim, zdim, hdim); m.load_state_dict(sd); m.eval(); ens.append(m)
	states = np.asarray(states, np.float32)
	N = len(states)
	V = np.zeros((N, nA), np.float32); SD = np.zeros((N, nA), np.float32)
	with torch.no_grad():
		for a in range(nA):
			acc = np.zeros((N, len(ens)), np.float32)
			for bi in range(0, N, batch):
				sb = states[bi:bi + batch]
				Bt = len(sb)
				h = torch.zeros(Bt, hdim); z = torch.zeros(Bt, zdim)
				s = torch.tensor(sb)
				disc = 1.0; ret = [torch.zeros(Bt) for _ in ens]
				for t in range(horizon):
					for ei, m in enumerate(ens):
						hh, (pmu, _), _ = m.step(h, z, torch.full((Bt,), a, dtype=torch.long),
												 torch.ones(Bt, 1) * 30.0, s)
						_, r_hat, _ = m.obs(hh, pmu)
						ret[ei] = ret[ei] + disc * r_hat.squeeze(-1) * rstd
						h, z = hh, pmu
					disc *= gamma ** 30.0
				for ei in range(len(ens)):
					acc[bi:bi + batch, ei] = ret[ei].numpy()
			V[:, a] = acc.mean(1); SD[:, a] = acc.std(1)
	return V, SD


def main(argv=None):
	ap = argparse.ArgumentParser(description="red_king RSSM")
	ap.parse_args(argv)
	print("== RED_KING RSSM ==")
	for k, v in train().items():
		print(f"  {k:16s}: {v}")


if __name__ == "__main__":
	main()
