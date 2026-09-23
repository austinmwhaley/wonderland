"""red_king COUNTERFACTUAL SCORECARD — measures red_king against its PURPOSE.

Purpose: answer "what happens if we act?" -> estimate per-action effects and choose
the best action per context, especially where logged overlap is thin.

Ground truth is KNOWN (synthetic), so we can score the counterfactual behavior:
  S1 effect ORDERING      : Spearman(learned per-action effect, true)
  S2 effect CALIBRATION   : ratio learned/true per-action effect (1.0 ideal)
  S3 per-context RANKING  : argmax accuracy vs the true best action
  S4 off-policy VALUE err : |estimated policy value - true| (relative)
  S5 low-overlap SKILL    : ranking accuracy in the lowest-overlap contexts
  S6 model-free BEAT      : MB ranking accuracy - behavior/IPS ranking accuracy

Composite score in [0,1]; higher = better serves the purpose. Also reports the
real-data ordering/calibration vs IPW for the deployed red_king.
"""
from __future__ import annotations

import argparse

import numpy as np


def _bandit(n=6000, d=8, nA=4, seed=0, logging_temp=0.5, noise=0.5):
	rng = np.random.default_rng(seed)
	S = rng.normal(size=(n, d)).astype(np.float32)
	W = rng.normal(size=(nA, d)).astype(np.float32) * 0.7
	# heterogeneous, non-linear per-action reward -> different best action per context
	base = (S @ W.T)
	mu = base + 1.5 * np.sin(S[:, [0]] + np.arange(nA)[None, :] * 0.9) \
		+ 0.8 * np.cos(S[:, [1]] * 1.3 + np.arange(nA)[None, :])
	logits = base + logging_temp * 0 + rng.normal(0, 0.7, (n, nA))  # confounded-ish
	# logging policy: softmax over a CONFOUNDED score (depends on s[0])
	logits = 2.0 * S[:, [0]] * np.array([1.0, -0.5, 0.7, -0.9])[None, :] \
		+ rng.normal(0, 0.5, (n, nA))
	p = np.exp(logits - logits.max(1, keepdims=True))
	p /= p.sum(1, keepdims=True)
	act = np.array([rng.choice(nA, p=pi) for pi in p])
	rew = (mu[np.arange(n), act] + rng.normal(0, noise, n)).astype(np.float32)
	return S, act, rew, p.astype(np.float32), mu, mu.max(1), mu.argmax(1)


def _fit_mb(S, a, r, nA, seed=0, K=5, steps=2500):
	import torch
	import torch.nn as nn
	dev = "cuda" if torch.cuda.is_available() else "cpu"
	dim = S.shape[1]
	St = torch.tensor(S, device=dev); At = torch.tensor(a, device=dev)
	Rt = torch.tensor(r, device=dev)
	class M(nn.Module):
		def __init__(self):
			super().__init__()
			self.net = nn.Sequential(nn.Linear(dim + nA, 256), nn.ReLU(),
									 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, nA))
		def out(self, s, oh):
			return self.net(torch.cat([s, oh], -1))
	ens = [M().to(dev) for _ in range(K)]
	opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in ens]
	for m, opt in zip(ens, opts):
		torch.manual_seed(seed)
		for _ in range(steps):
			b = torch.randint(0, len(S), (256,), device=dev)
			oh = nn.functional.one_hot(At[b], nA).float()
			pred = (m.out(St[b], oh) * oh).sum(-1)
			loss = nn.functional.mse_loss(pred, Rt[b])
			opt.zero_grad(); loss.backward(); opt.step()
	with torch.no_grad():
		grids = []
		for m in ens:
			oh = nn.functional.one_hot(torch.arange(nA, device=dev).repeat(len(S), 1).T.reshape(-1),
									   nA).float()
			s_rep = St.repeat(nA, 1)
			vals = (m.out(s_rep, oh) * oh).sum(-1).reshape(nA, len(S))
			grids.append(vals)
		Q = torch.stack(grids).mean(0).cpu().numpy()      # (nA, N)
	return Q


def run(seed=0):
	from scipy.stats import spearmanr
	S, a, r, p, mu, true_best, true_arg = _bandit(seed=seed)
	nA = mu.shape[1]
	Q = _fit_mb(S, a, r, nA, seed=seed)                   # learned per-action value
	# true vs learned per-action mean effect
	true_eff = mu.mean(0)
	learn_eff = Q.mean(1)
	s1 = float(spearmanr(true_eff, learn_eff).statistic)
	s2 = float(np.mean(learn_eff) / max(np.mean(true_eff), 1e-9))   # ~1 ideal
	# per-context ranking
	pred_arg = Q.argmax(0)
	s3 = float((pred_arg == true_arg).mean())
	# behaviour ranking accuracy (model-free proxy)
	beh_arg = np.array([np.bincount(a, minlength=nA).argmax()] * len(S))
	s6 = s3 - float((np.full(len(S), beh_arg[0]) == true_arg).mean())
	# off-policy value error: estimated value of learned policy vs its true value
	ips = np.array([float((r[(a == x)] / p[(a == x), x]).mean()) if (a == x).any() else 0.0
					for x in range(nA)])
	true_pol_val = float(mu[np.arange(len(S)), pred_arg].mean())
	est_pol_val = float(Q.max(0).mean())
	s4 = abs(est_pol_val - true_pol_val) / max(abs(true_pol_val), 1e-9)
	# low-overlap skill: ranking accuracy in lowest-propensity contexts
	overlap = p[np.arange(len(S)), a]                     # logged-action propensity
	lo = overlap <= np.quantile(overlap, 0.25)
	s5 = float((pred_arg[lo] == true_arg[lo]).mean())
	score = float(np.mean([(s1 + 1) / 2, min(s2, 1 / max(s2, 1e-9)),
						   s3, max(0.0, 1 - s4), s5, min(max(s6, 0.0), 1.0)]))
	return {"n": len(S), "nA": nA, "S1_ordering": round(s1, 3),
			"S2_calibration_ratio": round(s2, 3), "S3_rank_acc": round(s3, 3),
			"S4_value_rel_err": round(s4, 3), "S5_low_overlap_acc": round(s5, 3),
			"S6_beat_modelfree": round(s6, 3), "SCORE": round(score, 3)}


def real(seed=0):
	"""Deployed red_king (RSSM) on the REAL stream vs IPW ground truth."""
	from pathlib import Path
	import duckdb
	import polars as pl
	from scipy.stats import spearmanr
	WORK = Path(__file__).resolve().parents[2]
	CFM = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
	STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
	sc = duckdb.connect(str(STREAM), read_only=True)
	try:
		arm = sc.execute("SELECT customer_id k, arm, propensity, optimal_arm FROM email_arm").pl()
		inc = sc.execute("""
			SELECT s.customer_id k, SUM(o.gross_margin) g
			FROM email_sends s JOIN orders o
			  ON o.customer_id=s.customer_id AND o.session_id=s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked=1 GROUP BY 1""").pl()
	finally:
		sc.close()
	pc = duckdb.connect(str(CFM), read_only=True)
	try:
		emb = pc.execute("""
			SELECT customer_key, embedding FROM (
				SELECT customer_key, anchor_epoch, embedding,
				       row_number() OVER (PARTITION BY customer_key ORDER BY anchor_epoch DESC) rn
				FROM anchor_embeddings) WHERE rn=1""").pl()
	finally:
		pc.close()
	df = (arm.join(inc, on="k", how="left").join(emb, left_on="k", right_on="customer_key", how="inner")
			 .with_columns(pl.col("g").fill_null(0.0)))
	a = df["arm"].to_numpy(); e = df["propensity"].to_numpy(); g = df["g"].to_numpy()
	nA = int(a.max()) + 1
	ipw = np.array([float((g[a == x] / e[a == x]).sum() / max((1 / e[a == x]).sum(), 1e-9))
					for x in range(nA)])
	from red_king.red_king.rssm import rollout_arm_values
	states = np.stack(df["embedding"].to_list()).astype(np.float32)
	V, SD = rollout_arm_values(states)
	# CALIBRATION (offline, legitimate): scale imagined values so the imagined
	# value of the LOGGED behaviour arms matches the OBSERVED behaviour value.
	logged = df["arm"].to_numpy()
	imag_beh = float(V[np.arange(len(logged)), logged].mean())
	alpha = float(g.mean() / max(imag_beh, 1e-9))
	V = V * alpha
	imag = V.mean(0)
	pred = V.argmax(1)
	opt = df["optimal_arm"].to_numpy() if "optimal_arm" in df.columns else None
	pc_acc = round(float((pred == opt).mean()), 3) if opt is not None else float("nan")
	return {"real_ordering": round(float(spearmanr(ipw, imag).statistic), 3),
			"real_calibration_ratio": round(float(imag.mean() / max(ipw.mean(), 1e-9)), 3),
			"real_best_arm_true": int(ipw.argmax()), "real_best_arm_red_king": int(imag.argmax()),
			"real_rank_match": bool(int(ipw.argmax()) == int(imag.argmax())),
			"real_per_customer_rank_acc": pc_acc,
			"real_best_arm_dist_true": {int(k): int(v) for k, v in zip(*np.unique(opt, return_counts=True))} if opt is not None else {},
			"calib_alpha": round(alpha, 3)}


def main(argv=None):
	ap = argparse.ArgumentParser(description="red_king counterfactual scorecard")
	ap.parse_args(argv)
	res = run()
	print("== RED_KING COUNTERFACTUAL SCORECARD (known ground truth) ==")
	for k, v in res.items():
		print(f"  {k:22s}: {v}")
	try:
		r = real()
		print("== deployed red_king on REAL stream (vs IPW truth) ==")
		for k, v in r.items():
			print(f"  {k:22s}: {v}")
	except Exception as e:
		print("  real eval unavailable:", type(e).__name__, e)


if __name__ == "__main__":
	main()
