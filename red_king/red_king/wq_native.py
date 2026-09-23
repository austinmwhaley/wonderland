"""Native white_queen witness: red_king rollouts feed `panel["mb"]`.

Replicates white_queen's own flow (to_canonical -> panel -> adjudicate -> judge),
then replaces the model-based entry with red_king's multi-step rollout value and
re-decides. WITH vs WITHOUT red_king, same candidate, same estimand
(discounted incremental gross margin).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "seq_email.npz"
GAMMA = 0.99


def main():
	import torch
	import torch.nn as nn
	from white_queen.tribunal.ope import data as _data, estimators as _E, gate as _gate, judge as _judge
	from white_queen.tribunal.ope.receipts import behavior_stats
	z = np.load(DATA)
	S, A, R, S2, D, DT, TID, G = (z["S"], z["A"], z["R"], z["S2"], z["D"],
								  z["DT"], z["TID"], z["G"])
	nA = int(A.max()) + 1
	means = np.array([R[A == a].mean() if (A == a).any() else -1e9 for a in range(nA)])
	astar = int(means.argmax())

	class ConstPolicy:
		def act(self, state, eval=True):
			return astar
		def action_probs(self, obs, temperature=1.0):
			o = np.asarray(obs)
			n = len(o) if o.ndim > 1 else 1
			p = np.zeros((n, nA), np.float32)
			p[:, astar] = 1.0
			return p

	diet = _data.to_canonical({"obs": S, "act": A, "rew": R, "next_obs": S2, "done": D},
							  nA=nA)
	b = behavior_stats(diet, GAMMA)
	cand = ConstPolicy()
	panel = _E.panel(diet, cand, GAMMA, fast=True, ensemble_K=2)
	rows = _gate.adjudicate({"const": panel}, b["mean"], b["std"], None, None, n_episodes=None)
	v = _judge.judge_diet(rows, b["mean"], b["std"], None)
	dec0 = v["decisions"]["const"]

	# red_king multi-step rollout value of the candidate (same diet)
	dev = "cuda" if torch.cuda.is_available() else "cpu"
	dim = S.shape[1]; K = 3; steps = 1500
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
		torch.manual_seed(0)
		for _ in range(steps):
			bix = torch.randint(0, len(S), (512,), device=dev)
			oh = nn.functional.one_hot(At[bix], nA).float()
			ds, r = m(St[bix], oh, DTt[bix])
			loss = nn.functional.mse_loss(ds, DS[bix]) + nn.functional.mse_loss(r, Rt[bix])
			opt.zero_grad(); loss.backward(); opt.step()
	vals = []
	with torch.no_grad():
		for t in np.unique(TID):
			idx = np.where(TID == t)[0]
			s = torch.tensor(S[idx[0]], device=dev).unsqueeze(0)
			acc = 0.0
			for k in idx:
				oh = nn.functional.one_hot(torch.tensor([astar], device=dev), nA).float()
				dt = torch.tensor([DT[k]], dtype=torch.float32, device=dev).unsqueeze(-1)
				mem = [m(s, oh, dt) for m in ens]
				ds = torch.stack([x[0] for x in mem]).mean(0)
				r = torch.stack([x[1] for x in mem]).mean(0) * rstd
				acc = float(r.item()) + (GAMMA ** float(DT[k])) * acc
				s = s + ds * dstd
			vals.append(acc)
	vals = np.array(vals)
	rk_val, rk_se = float(vals.mean()), float(vals.std() / np.sqrt(len(vals)))
	panel2 = dict(panel); panel2["mb"] = {"mb": rk_val, "se": rk_se, "sims": K}
	rows2 = _gate.adjudicate({"const": panel2}, b["mean"], b["std"], None, None, n_episodes=None)
	v2 = _judge.judge_diet(rows2, b["mean"], b["std"], None)
	dec1 = v2["decisions"]["const"]
	print("== NATIVE WITNESS: red_king feeds panel['mb'] ==")
	print(f"  candidate arm        : {astar}")
	print(f"  WHITE_QUEEN mb (without): {panel['mb'].get('mb')}")
	print(f"  RED_KING     mb (with)  : {rk_val:.2f} (se {rk_se:.2f})")
	print(f"  decision WITHOUT : deploy={dec0['deploy']} witnesses={dec0.get('witnesses')}")
	print(f"  decision WITH    : deploy={dec1['deploy']} witnesses={dec1.get('witnesses')}")
	return {"astar": astar, "dec0": dec0["deploy"], "dec1": dec1["deploy"]}


if __name__ == "__main__":
	main()
