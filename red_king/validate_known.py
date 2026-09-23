"""Independent validation: does red_king's IMAGINED arm value match the KNOWN
causal effect E[incremental GP | do(arm)]?

Ground truth: IPW over the randomized arms (valid because assignment is randomized
with logged propensity). red_king: averaged rollout value per arm from sample-B
customer states.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
CFM = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"


def _facts():
	import duckdb
	sc = duckdb.connect(str(STREAM), read_only=True)
	try:
		arm = sc.execute("SELECT customer_id AS customer_key, arm, propensity FROM email_arm").pl()
		inc = sc.execute("""
			SELECT s.customer_id AS customer_key, SUM(o.gross_margin) g
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
	df = (arm.join(inc, on="customer_key", how="left")
			 .join(emb, on="customer_key", how="inner")
			 .with_columns(__import__("polars").col("g").fill_null(0.0)))
	return df


def main():
	import polars as pl
	from red_king.rssm import rollout_arm_values
	df = _facts()
	arm = df["arm"].to_numpy(); prop = df["propensity"].to_numpy(); g = df["g"].to_numpy()
	states = np.stack(df["embedding"].to_list()).astype(np.float32)
	nA = int(arm.max()) + 1
	# ground truth: IPW per arm
	ipw = []
	for a in range(nA):
		m = arm == a
		ipw.append(float((g[m] / prop[m]).sum() / max((1 / prop[m]).sum(), 1e-9)))
	V, SD = rollout_arm_values(states)
	imagined = [float(V[:, a].mean()) for a in range(nA)]
	print("== RED_KING IMAGINED ARM VALUE vs KNOWN CAUSAL EFFECT ==")
	print(f"{'arm':>4} {'IPW truth':>12} {'red_king':>12} {'ens SD':>8}")
	for a in range(nA):
		print(f"{a:>4} {ipw[a]:>12.2f} {imagined[a]:>12.2f} {SD[:,a].mean():>8.2f}")
	from scipy.stats import spearmanr, pearsonr
	print("ordering truth   :", list(np.argsort(ipw)))
	print("ordering rk      :", list(np.argsort(imagined)))
	print(f"spearman(true,rk): {spearmanr(ipw, imagined).statistic:.3f}")
	print(f"pearson (true,rk): {pearsonr(ipw, imagined).statistic:.3f}")


if __name__ == "__main__":
	main()
