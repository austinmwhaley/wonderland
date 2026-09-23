"""Per-CUSTOMER value model (heterogeneous treatment effects).

Goal: estimate V_i(a) = E[reward | state_i, do(a)] PER CUSTOMER, so red_queen can
choose the best action per customer.

Method: a linear model with STATE x ACTION INTERACTION (a low-variance HTE model),
fit on randomized/within-customer windows with CLIPPED IPW weights to deconfound.
Validated per-customer against the stored true `optimal_arm`, vs the majority
baseline.
"""
from __future__ import annotations

import numpy as np

from red_king.effect_model import build


def _features(S, A, nA):
	oh = np.zeros((len(S), nA), np.float32)
	oh[np.arange(len(S)), A] = 1.0
	inter = np.hstack([S * oh[:, a:a + 1] for a in range(nA)])   # state × action
	return np.hstack([S, inter, oh]).astype(np.float32)


def run(seed=0):
	from sklearn.linear_model import Ridge
	from sklearn.preprocessing import StandardScaler
	X, A, Y, W, keys, optmap = build()
	nA = int(A.max()) + 1
	# clip IPW weights (stabilise) and fit a low-variance interaction model
	w = np.clip(W, None, np.quantile(W, 0.95))
	F = _features(X, A, nA)
	sc = StandardScaler().fit(F)
	m = Ridge(alpha=1.0).fit(sc.transform(F), Y, sample_weight=w)
	# per-customer: last window state -> predict per arm -> argmax
	last = {}
	for i, k in enumerate(keys):
		last[k] = i
	ks = list(last)
	sidx = np.array([last[k] for k in ks])
	S = X[sidx]
	V = np.zeros((len(ks), nA), np.float32)
	for a in range(nA):
		aa = np.full(len(ks), a, np.int64)
		V[:, a] = m.predict(sc.transform(_features(S, aa, nA)))
	pred = V.argmax(1)
	truth = np.array([optmap.get(k, -1) for k in ks])
	mask = truth >= 0
	acc = float((pred[mask] == truth[mask]).mean())
	from collections import Counter
	maj = Counter(truth[mask]).most_common(1)[0][0]
	maj_acc = float((np.full(mask.sum(), maj) == truth[mask]).mean())
	# SWITCHBACK subset: customers observed under ALL arms (identification exists)
	arms_of_k = {}
	for i, k in enumerate(keys):
		arms_of_k.setdefault(k, set()).add(int(A[i]))
	switch = np.array([len(arms_of_k.get(k, set())) >= nA for k in ks])
	sm = mask & switch
	acc_s = float((pred[sm] == truth[sm]).mean()) if sm.sum() else float("nan")
	maj_s = Counter(truth[sm]).most_common(1)[0][0] if sm.sum() else -1
	maj_s_acc = float((np.full(sm.sum(), maj_s) == truth[sm]).mean()) if sm.sum() else float("nan")
	# value of the learned PERSONALIZED policy vs the best CONSTANT arm (IPS, holdout-able)
	return {"windows": len(X), "customers": len(ks), "nA": nA,
			"per_customer_rank_acc": round(acc, 4),
			"majority_baseline": round(maj_acc, 4),
			"beats_majority": bool(acc > maj_acc + 1e-9),
			"switchback_customers": int(switch.sum()),
			"switch_per_customer_acc": round(acc_s, 4) if sm.sum() else None,
			"switch_majority": round(maj_s_acc, 4) if sm.sum() else None,
			"switch_beats_majority": bool(sm.sum() and acc_s > maj_s_acc + 1e-9)}


if __name__ == "__main__":
	print("== PER-CUSTOMER VALUE MODEL (HTE, state x action) ==")
	for k, v in run().items():
		print(f"  {k:24s}: {v}")
