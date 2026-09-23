"""Per-CUSTOMER response model (uplift) + management incrementality report.

Offline, no acting. The persistent holdout (randomized control) makes the
treatment effect identifiable, so we can estimate, PER CUSTOMER, how much
marketing lifts their outcome, and produce a management-facing incrementality
report (population ATE + CI + decile calibration + targeting gain).

  outcome  y  = gross margin in the period
  treatment t = 1 if not held out (marketed) else 0   (randomized 5%)
  state    s  = frozen donor embedding (frozen looking_glass representation)

Model: y ~ s + t + s:t  -> uplift(s) = b_t + s·b_{t:x}   (heterogeneous effect).
Validation: held-out customers; predicted vs realised uplift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
CFM = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
OUT = Path(__file__).resolve().parents[0] / "artifacts" / "incrementality_report.json"
N_PERIODS = 12


def build():
	import duckdb
	import polars as pl
	con = duckdb.connect(str(STREAM), read_only=True)
	try:
		min_t, max_t = con.execute(
			"SELECT min(epoch(CAST(event_ts AS TIMESTAMPTZ))), "
			"max(epoch(CAST(event_ts AS TIMESTAMPTZ))) FROM customer_events").fetchone()
		pdays = (max_t - min_t) / N_PERIODS
		orders = con.execute(f"""
			SELECT customer_id k,
			       least({N_PERIODS-1}, greatest(0,
			         floor((epoch(CAST(order_ts AS TIMESTAMPTZ)) - {min_t}) / {pdays})::int)) p,
			       SUM(gross_margin) gm
			FROM orders GROUP BY 1,2""").pl()
		hold = con.execute("SELECT customer_id k, period p, holdout FROM email_holdout").pl()
	finally:
		con.close()
	cp = duckdb.connect(str(CFM), read_only=True)
	try:
		emb = cp.execute("""
			SELECT customer_key, embedding FROM (
				SELECT customer_key, anchor_epoch, embedding,
				       row_number() OVER (PARTITION BY customer_key ORDER BY anchor_epoch DESC) rn
				FROM anchor_embeddings) WHERE rn=1""").pl()
	finally:
		cp.close()
	grid = hold.join(orders, on=["k", "p"], how="left").with_columns(pl.col("gm").fill_null(0.0))
	grid = grid.join(emb, left_on="k", right_on="customer_key", how="inner")
	return grid


def run(seed=0):
	from sklearn.linear_model import Ridge
	from sklearn.preprocessing import StandardScaler
	from scipy.stats import spearmanr
	gr = build()
	S = np.stack(gr["embedding"].to_list()).astype(np.float32)
	t = (1.0 - gr["holdout"].to_numpy().astype(np.float64))     # treatment
	y = gr["gm"].to_numpy().astype(np.float64)
	k = gr["k"].to_numpy()
	# customer split
	uniq = np.unique(k)
	rng = np.random.default_rng(seed)
	test_c = set(rng.choice(uniq, int(0.3 * len(uniq)), replace=False).tolist())
	te = np.array([kk in test_c for kk in k]); tr = ~te

	def feats(S_, t_):
		return np.hstack([S_, t_[:, None], S_ * t_[:, None]]).astype(np.float32)
	sc = StandardScaler().fit(feats(S[tr], t[tr]))
	m = Ridge(alpha=1.0).fit(sc.transform(feats(S[tr], t[tr])), y[tr])

	def uplift(S_):
		f1 = feats(S_, np.ones(len(S_))); f0 = feats(S_, np.zeros(len(S_)))
		return m.predict(sc.transform(f1)) - m.predict(sc.transform(f0))
	# population ATE (on held-out customer-periods)
	ate = float(uplift(S[te]).mean())
	boot = [float(uplift(S[te])[j].mean()) for j in
			(rng.integers(0, te.sum(), te.sum()) for _ in range(300))]
	lo, hi = np.percentile(boot, [2.5, 97.5])
	# per-customer observed vs predicted uplift (customers with both states)
	up = {}
	for kk in uniq:
		mk = k == kk
		tt = mk & (t == 1); hh = mk & (t == 0)
		if tt.any() and hh.any():
			up[kk] = float(y[tt].mean() - y[hh].mean())
	ks = [kk for kk in up if kk in test_c]
	if len(ks) > 10:
		# predicted uplift per customer (use their first state)
		idx = {kk: np.where(k == kk)[0][0] for kk in ks}
		pred = uplift(S[[idx[kk] for kk in ks]])
		obs = np.array([up[kk] for kk in ks])
		rho = float(spearmanr(pred, obs).statistic)
		# decile calibration
		order = np.argsort(pred); dec = np.array_split(order, 5)
		calib = [round(float(obs[d].mean()), 1) for d in dec]
		# targeting gain: treat only the top-uplift customers
		top = order[int(0.8 * len(order)):]
		gain = float(obs[top].mean() - obs.mean())
	else:
		rho, calib, gain = float("nan"), [], float("nan")
	res = {"customer_periods": int(len(y)), "treated": int((t == 1).sum()),
		   "heldout": int((t == 0).sum()), "customers": int(len(uniq)),
		   "ATE_incremental_margin": round(ate, 2), "ate_ci95": [round(lo, 2), round(hi, 2)],
		   "ate_significant": bool(lo > 0),
		   "predictable_rho_holdout": round(rho, 3) if np.isfinite(rho) else None,
		   "uplift_quintile_realized": calib,
		   "targeting_top20pct_gain": round(gain, 2) if np.isfinite(gain) else None}
	OUT.parent.mkdir(parents=True, exist_ok=True)
	OUT.write_text(json.dumps({"summary": res}, indent=1))
	return res


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.parse_args()
	print("== PER-CUSTOMER RESPONSE MODEL + INCREMENTALITY REPORT ==")
	for kk, v in run().items():
		print(f"  {kk:28s}: {v}")


# ---------------------------------------------------------------------------
# TARGETING: use the per-customer uplift to choose WHOM to market
# ---------------------------------------------------------------------------
def fit_uplift(seed=0):
	"""Fit the uplift model and return (per-customer predicted uplift, best arm)."""
	from sklearn.linear_model import Ridge
	from sklearn.preprocessing import StandardScaler
	gr = build()
	S = np.stack(gr["embedding"].to_list()).astype(np.float32)
	t = (1.0 - gr["holdout"].to_numpy().astype(np.float64))
	y = gr["gm"].to_numpy().astype(np.float64)
	k = gr["k"].to_numpy()

	def feats(S_, t_):
		return np.hstack([S_, t_[:, None], S_ * t_[:, None]]).astype(np.float32)
	sc = StandardScaler().fit(feats(S, t))
	m = Ridge(alpha=1.0).fit(sc.transform(feats(S, t)), y)

	def uplift(S_):
		f1 = feats(S_, np.ones(len(S_))); f0 = feats(S_, np.zeros(len(S_)))
		return m.predict(sc.transform(f1)) - m.predict(sc.transform(f0))
	# one representative state per customer (first occurrence)
	seen, idx, keys = set(), [], []
	for i, kk in enumerate(k):
		if kk not in seen:
			seen.add(kk); idx.append(i); keys.append(kk)
	up = uplift(S[idx])
	best_arm = 0
	try:
		from red_queen.engine import _validated_arm_effects
		best_arm = int(np.argmax(_validated_arm_effects()))
	except Exception:
		pass
	return keys, up, best_arm


def target_plan(budget=None, min_uplift=0.0):
	"""Rank customers by predicted uplift; allocate marketing budget to responders.
	Non-responders (uplift <= min_uplift) get NO action (fail-safe)."""
	keys, up, best_arm = fit_uplift()
	order = np.argsort(-up)
	pos = up[order] > min_uplift
	if budget is None:
		budget = float(pos.sum())          # one weekly touch per responding customer
	chosen = []
	for j in order:
		if up[j] <= min_uplift or budget <= 0:
			break
		chosen.append(j); budget -= 1.0
	plan = {"customer_key": [keys[j] for j in chosen],
			"uplift": [round(float(up[j]), 2) for j in chosen],
			"arm": [best_arm] * len(chosen)}
	report = {"customers_total": len(keys),
			  "responders": int((up > min_uplift).sum()),
			  "targeted": len(chosen),
			  "expected_incremental_margin": round(float(up[chosen].sum()), 1) if chosen else 0.0,
			  "population_fallback": "non-responders -> no action (fail-safe)",
			  "best_arm": best_arm}
	OUT.parent.mkdir(parents=True, exist_ok=True)
	(Path(OUT).parent / "target_plan.json").write_text(json.dumps({"report": report, "plan": plan}))
	return report


if __name__ == "__main__" and False:
	pass
