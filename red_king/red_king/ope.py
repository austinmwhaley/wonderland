"""white_queen integration: does red_king improve the policy-value estimate?

Runs on the REAL email data and compares, on held-out customers:
  WITHOUT red_king : behavior value (logged) and a raw-action reward model
                     (no deconfounding, no pessimism) -> a naive model-based value
  WITH    red_king : deconfounded (residual exposure) ensemble reward model,
                     greedy policy scored with ensemble pessimism (mean - std)

This is the optional path white_queen's plugin calls when use_world_model=True.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .world_model import build_transitions


def _fit(s, a, r, nA, tr, seed=0, steps=1500, K=5, deconfound=True):
	import torch.nn as nn
	from sklearn.linear_model import Ridge
	dev = "cuda" if torch.cuda.is_available() else "cpu"
	dim = s.shape[1]
	ctrl = Ridge(alpha=1.0).fit(s[tr], a[tr]) if deconfound else None
	if deconfound:
		resid = a - ctrl.predict(s)
		resid = (resid - resid[tr].mean()) / (resid[tr].std() + 1e-6)
		X = torch.tensor(resid, dtype=torch.float32, device=dev)
	else:
		X = torch.zeros(len(s), device=dev)
	class M(nn.Module):
		def __init__(self):
			super().__init__()
			self.net = nn.Sequential(nn.Linear(dim + nA + (1 if deconfound else 0), 256),
									 nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
									 nn.Linear(256, nA))
		def out(self, s, oh, x):
			inp = torch.cat([s, oh, x[:, None]], -1) if deconfound else torch.cat([s, oh], -1)
			return self.net(inp)
	ens = [M().to(dev) for _ in range(K)]
	opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in ens]
	S = torch.tensor(s, device=dev); A = torch.tensor(a, device=dev); R = torch.tensor(r, device=dev)
	tri = torch.tensor(np.where(tr)[0], device=dev)
	for m, opt in zip(ens, opts):
		torch.manual_seed(seed)
		for _ in range(steps):
			b = tri[torch.randint(0, len(tri), (256,), device=dev)]
			oh = torch.nn.functional.one_hot(A[b], nA).float()
			pred = (m.out(S[b], oh, X[b]) * oh).sum(-1)
			loss = torch.nn.functional.mse_loss(pred, R[b])
			opt.zero_grad(); loss.backward(); opt.step()
	return ens, ctrl


def _greedy_value(ens, ctrl, s, nA, deconfound, rscale, pessim=1.0):
	dev = next(ens[0].parameters()).device
	with torch.no_grad():
		S = torch.tensor(s, device=dev)
		grids = []
		for m in ens:
			rows = []
			for ac in range(nA):
				acv = torch.full((S.shape[0],), ac, device=dev, dtype=torch.long)
				oh = torch.nn.functional.one_hot(acv, nA).float()
				if deconfound:
					base = ctrl.predict(s)
					rx = (ac - base) / (rscale + 1e-6)
					x = torch.tensor(rx, dtype=torch.float32, device=dev)
				else:
					x = torch.zeros(S.shape[0], device=dev)
				rows.append((m.out(S, oh, x) * oh).sum(-1))
			grids.append(torch.stack(rows, 0))
		g = torch.stack(grids, 0)
		mean, std = g.mean(0), g.std(0)
	return float(mean.max(0).values.mean()), float((mean - pessim * std).max(0).values.mean())


def run(nA=4, seed=0):
	S, A, R, S2, D, nAb, C = build_transitions(nA)
	trm = np.zeros(len(S), bool)
	from sklearn.model_selection import GroupShuffleSplit
	tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
				  .split(np.zeros(len(S)), groups=C))
	trm[tr] = True
	rs = R[tr].std() + 1e-6
	behavior = float(R[te].mean())
	# WITHOUT: raw-action reward model (no deconfound, no pessimism)
	ens0, _ = _fit(S, A, R, nAb, tr, seed, deconfound=False)
	wm0, _ = _greedy_value(ens0, None, S[te], nAb, False, rs, pessim=0.0)
	# WITH: deconfounded + pessimistic
	ens1, ctrl = _fit(S, A, R, nAb, tr, seed, deconfound=True)
	wm1m, wm1p = _greedy_value(ens1, ctrl, S[te], nAb, True, rs, pessim=1.0)
	return {"n": int(len(S)), "nA": int(nAb), "behavior": behavior,
			"without_wm_mean": wm0, "with_wm_mean": wm1m, "with_wm_pessimistic": wm1p}


def main(argv=None):
	ap = argparse.ArgumentParser(description="red_king x white_queen with/without")
	a = ap.parse_args(argv)
	res = run()
	print("== WITH vs WITHOUT red_king (real email data, held-out customers) ==")
	print(f"  transitions                  : {res['n']}")
	print(f"  actions                      : {res['nA']}")
	print(f"  WITHOUT  behavior value      : {res['behavior']:.3f}")
	print(f"  WITHOUT  wm greedy (raw act) : {res['without_wm_mean']:.3f}")
	print(f"  WITH     wm greedy (deconf)  : {res['with_wm_mean']:.3f}")
	print(f"  WITH     wm greedy pessimist : {res['with_wm_pessimistic']:.3f}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
