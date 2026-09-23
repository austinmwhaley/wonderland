"""Multi-step red_king + with/without long-horizon value test.

Trains an ensemble one-step model (s, a, dt) -> (delta_s, reward) on the sequential
email dataset, then evaluates LONG-HORIZON discounted return by autoregressive
rollout with uncertainty. Compares, on held-out trajectories:

  WITHOUT red_king : direct model-free regression  s -> discounted return
  WITH    red_king : multi-step rollout using observed actions (dynamics+r)

Ground truth = the actual discounted incremental return from each trajectory start.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "seq_email.npz"
GAMMA_DAY = 0.999


def run(seed=0, K=5, steps=2500):
	import torch
	import torch.nn as nn
	from sklearn.linear_model import Ridge
	from sklearn.preprocessing import StandardScaler
	from scipy.stats import spearmanr
	z = np.load(DATA)
	S, A, R, S2, D, DT, TID, G = (z["S"], z["A"], z["R"], z["S2"], z["D"],
								  z["DT"], z["TID"], z["G"])
	dim, nA = S.shape[1], int(A.max()) + 1
	tids = np.unique(TID)
	rng = np.random.default_rng(seed)
	test_tids = set(rng.choice(tids, size=int(0.3 * len(tids)), replace=False).tolist())
	te_mask = np.isin(TID, list(test_tids))
	tr = np.where(~te_mask)[0]
	dev = "cuda" if torch.cuda.is_available() else "cpu"
	rstd = R[tr].std() + 1e-6
	class M(nn.Module):
		def __init__(self):
			super().__init__()
			self.net = nn.Sequential(nn.Linear(dim + nA + 1, 512), nn.ReLU(),
									 nn.Linear(512, 256), nn.ReLU(),
									 nn.Linear(256, dim + 1))
		def forward(self, s, oh, dt):
			o = self.net(torch.cat([s, oh, dt], -1))
			return o[:, :dim], o[:, dim]
	St = torch.tensor(S, device=dev); At = torch.tensor(A, device=dev)
	DTt = torch.tensor(DT, dtype=torch.float32, device=dev)
	Rt = torch.tensor(R / rstd, dtype=torch.float32, device=dev)
	DS = torch.tensor(S2 - S, device=dev)
	ens = [M().to(dev) for _ in range(K)]
	opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in ens]
	tri = torch.tensor(tr, device=dev)
	for m, opt in zip(ens, opts):
		torch.manual_seed(seed)
		for _ in range(steps):
			b = tri[torch.randint(0, len(tri), (512,), device=dev)]
			oh = nn.functional.one_hot(At[b], nA).float()
			ds, r = m(St[b], oh, DTt[b].unsqueeze(-1))
			loss = nn.functional.mse_loss(ds, DS[b]) + nn.functional.mse_loss(r, Rt[b])
			opt.zero_grad(); loss.backward(); opt.step()
	# model-free baseline: s -> discounted return
	sc = StandardScaler().fit(S[tr])
	mf = Ridge(alpha=1.0).fit(sc.transform(S[tr]), G[tr])
	# evaluate per held-out trajectory start
	tids_sorted = {t: np.where(TID == t)[0] for t in test_tids}
	mb_pred, mf_pred, actual = [], [], []
	with torch.no_grad():
		for t, idx in tids_sorted.items():
			if len(idx) < 2:
				continue
			# forward rollout from start with observed actions
			s = torch.tensor(S[idx[0]], device=dev).unsqueeze(0)
			acc = 0.0
			for k in idx:
				oh = nn.functional.one_hot(torch.tensor([int(A[k])], device=dev), nA).float()
				dt = torch.tensor([DT[k]], dtype=torch.float32, device=dev).unsqueeze(-1)
				members = [m(s, oh, dt) for m in ens]
				ds = torch.stack([x[0] for x in members]).mean(0)
				r = torch.stack([x[1] for x in members]).mean(0) * rstd
				acc = float(r.item()) + (GAMMA_DAY ** float(DT[k])) * acc
				s = s + ds
			mb_pred.append(acc)
			mf_pred.append(float(mf.predict(sc.transform(S[idx[0]:idx[0] + 1]))[0]))
			actual.append(float(G[idx[0]]))
	mb_pred = np.array(mb_pred); mf_pred = np.array(mf_pred); actual = np.array(actual)
	rho = lambda p: float(spearmanr(p, actual).statistic)
	mae = lambda p: float(np.mean(np.abs(p - actual)))
	print("== MULTI-STEP RED_KING (long-horizon value, held-out trajectories) ==")
	print(f"  trajectories          : {len(actual)}")
	print(f"  WITHOUT (model-free)  : spearman={rho(mf_pred):+.3f}  mae={mae(mf_pred):.1f}")
	print(f"  WITH    (multi-step)  : spearman={rho(mb_pred):+.3f}  mae={mae(mb_pred):.1f}")
	return {"n": len(actual), "rho_mf": rho(mf_pred), "rho_mb": rho(mb_pred),
			"mae_mf": mae(mf_pred), "mae_mb": mae(mb_pred)}


if __name__ == "__main__":
	run()
