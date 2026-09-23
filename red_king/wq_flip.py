"""Decisive flip test: red_king mb vs white_queen mb on a white_queen-TRAINED
candidate (iql). Does the conservative world-model witness change DEPLOY/HOLD?"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[0] / "data" / "seq_email.npz"
GAMMA = 0.99


def rk_value(S, A, S2, R, DT, TID, nA, policy, seed=0, K=3, steps=1500):
	import torch
	import torch.nn as nn
	dim = S.shape[1]
	dev = "cuda" if torch.cuda.is_available() else "cpu"
	dstd = (S2 - S).std() + 1e-6; rstd = R.std() + 1e-6
	class M(nn.Module):
		def __init__(self):
			super().__init__()
			self.net = nn.Sequential(nn.Linear(dim + nA + 1, 512), nn.ReLU(),
									 nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, dim + 1))
		def forward(self, s, oh, dt):
			o = self.net(torch.cat([s, oh, dt], -1))
			return o[:, :dim], o[:, dim]
	St = torch.tensor(S, device=dev); At = torch.tensor(A, device=dev)
	DTt = torch.tensor(DT, dtype=torch.float32, device=dev).unsqueeze(-1)
	Rt = torch.tensor(R / rstd, dtype=torch.float32, device=dev)
	DS = torch.tensor((S2 - S) / dstd, device=dev)
	ens = [M().to(dev) for _ in range(K)]
	opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in ens]
	for m, opt in zip(ens, opts):
		torch.manual_seed(seed)
		for _ in range(steps):
			ix = torch.randint(0, len(S), (512,), device=dev)
			oh = nn.functional.one_hot(At[ix], nA).float()
			ds, r = m(St[ix], oh, DTt[ix])
			loss = nn.functional.mse_loss(ds, DS[ix]) + nn.functional.mse_loss(r, Rt[ix])
			opt.zero_grad(); loss.backward(); opt.step()
	vals = []
	with torch.no_grad():
		for t in np.unique(TID):
			idx = np.where(TID == t)[0]
			s = S[idx[0]:idx[0] + 1]
			acc = 0.0
			for k in idx:
				probs = np.asarray(policy.action_probs(s))
				a = int(np.argmax(probs[0]))
				oh = nn.functional.one_hot(torch.tensor([a], device=dev), nA).float()
				st = torch.tensor(s, device=dev)
				dt = torch.tensor([DT[k]], dtype=torch.float32, device=dev).unsqueeze(-1)
				mem = [m(st, oh, dt) for m in ens]
				ds = torch.stack([x[0] for x in mem]).mean(0)
				r = torch.stack([x[1] for x in mem]).mean(0) * rstd
				acc = float(r.item()) + (GAMMA ** float(DT[k])) * acc
				s = s + (ds.cpu().numpy() * dstd)
			vals.append(acc)
	vals = np.array(vals)
	return float(vals.mean()), float(vals.std() / np.sqrt(len(vals)))


def main(seed=0):
	from white_queen.tribunal.ope import data as _data, estimators as _E, gate as _gate, judge as _judge
	from white_queen.tribunal.ope.pipeline import train_candidates
	from white_queen.tribunal.ope.receipts import behavior_stats
	z = np.load(DATA)
	S, A, R, S2, D, DT, TID = (z["S"], z["A"], z["R"], z["S2"], z["D"], z["DT"], z["TID"])
	nA = int(A.max()) + 1
	diet = _data.to_canonical({"obs": S, "act": A, "rew": R, "next_obs": S2, "done": D}, nA=nA)
	b = behavior_stats(diet, GAMMA)
	handles = train_candidates(diet, algorithms=("iql",), gamma=GAMMA,
							   offline_steps=1500, seed=seed)
	pol = handles["iql"]
	panel = _E.panel(diet, pol, GAMMA, fast=True, ensemble_K=2)
	ne = len(np.unique(diet["episode"]))
	rows = _gate.adjudicate({"iql": panel}, b["mean"], b["std"], None, None, n_episodes=ne)
	rk_val, rk_se = rk_value(S, A, S2, R, DT, TID, nA, pol, seed=seed)
	panel2 = dict(panel); panel2["mb"] = {"mb": rk_val, "se": rk_se, "sims": 3}
	rows2 = _gate.adjudicate({"iql": panel2}, b["mean"], b["std"], None, None, n_episodes=ne)
	print("== FLIP TEST: iql candidate ==")
	print(f"  WHITE_QUEEN mb: {panel['mb'].get('mb')} | RED_KING mb: {rk_val:.2f} (se {rk_se:.2f})")
	print(f"  {'risk_av':>8} {'WITHOUT':>18} {'WITH red_king':>18}")
	res = {}
	for ra in (0.5, 0.8, 1.0):
		d0 = _judge.judge_diet(rows, b["mean"], b["std"], None, risk_aversion=ra)["decisions"]["iql"]
		d1 = _judge.judge_diet(rows2, b["mean"], b["std"], None, risk_aversion=ra)["decisions"]["iql"]
		print(f"  {ra:>8} {str(d0['deploy'])+' w='+str(d0.get('witnesses')):>18} "
			  f"{str(d1['deploy'])+' w='+str(d1.get('witnesses')):>18}")
		res[ra] = (d0["deploy"], d1["deploy"])
	return res


if __name__ == "__main__":
	main()
