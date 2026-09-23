"""caterpillar — read-only interpretability. Answers "why" from ARTIFACTS only.

Given a customer, it reports (grounded, no invention):
  * the red_queen recommended action + expected incremental GP (from the NBA plan)
  * the nearest customers in frozen-donor space (the representation's own view)
  * what those similar customers were recommended and their expected value
  * provenance: which artifacts/versions produced the answer

Changes nothing; reads everything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[2]
CFM_PRODUCTS = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
NBA_PLAN = WORK / "red_queen" / "artifacts" / "nba_plan.json"


def _load():
	import duckdb
	con = duckdb.connect(str(CFM_PRODUCTS), read_only=True)
	try:
		key = con.execute("SELECT DISTINCT version FROM anchor_embeddings").fetchall()[0][0]
		df = con.execute("""
			SELECT customer_key, embedding FROM (
				SELECT customer_key, anchor_epoch, embedding,
				       row_number() OVER (PARTITION BY customer_key ORDER BY anchor_epoch DESC) rn
				FROM anchor_embeddings) WHERE rn = 1""").pl()
	finally:
		con.close()
	return key, df["customer_key"].to_list(), np.stack(df["embedding"].to_list()).astype(np.float32)


def explain(customer: str, k: int = 5):
	version, keys, E = _load()
	plan = json.loads(NBA_PLAN.read_text())
	idx = {c: i for i, c in enumerate(plan["customer_key"])}
	if customer not in idx:
		return {"error": "customer not in plan", "customer": customer}
	i = idx[customer]
	# nearest neighbours in the frozen donor space
	En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
	ki = keys.index(customer)
	q = En[ki]
	sims = En @ q
	nn = np.argsort(-sims)[1:k + 1]
	rec = {
		"customer": customer,
		"donor_version": version,
		"recommended_arm": int(plan["arm"][i]),
		"weekly_sends": float(plan["weekly_sends"][i]),
		"expected_weekly_incremental_gp": float(plan["expected_gp"][i]),
		"similar_customers": [
			{"customer": keys[int(j)], "similarity": round(float(sims[int(j)]), 3),
			 "recommended_arm": int(plan["arm"][plan["customer_key"].index(keys[int(j)])]),
			 "expected_gp": float(plan["expected_gp"][plan["customer_key"].index(keys[int(j)])])}
			for j in nn if keys[int(j)] in idx],
		"provenance": {"nba_plan": str(NBA_PLAN), "embeddings": "anchor_embeddings",
					   "cfm_version": version},
	}
	return rec


def _render(rec):
	if "error" in rec:
		return f"caterpillar: {rec}"
	lines = [
		f"Customer {rec['customer']}  (donor {rec['donor_version']})",
		f"  Recommended action : arm {rec['recommended_arm']} "
		f"({rec['weekly_sends']:.1f} sends/week)",
		f"  Expected incremental GP/week : {rec['expected_weekly_incremental_gp']:.1f}",
		f"  Why (similar customers in frozen-donor space):",
	]
	for s in rec["similar_customers"]:
		lines.append(f"    - {s['customer']} (sim {s['similarity']}) -> arm "
					 f"{s['recommended_arm']}, E[GP] {s['expected_gp']:.1f}")
	lines.append(f"  Provenance : {rec['provenance']}")
	return "\n".join(lines)


def main(argv=None):
	ap = argparse.ArgumentParser(description="caterpillar explain")
	ap.add_argument("--customer", required=True)
	ap.add_argument("--k", type=int, default=5)
	a = ap.parse_args(argv)
	print(_render(explain(a.customer, a.k)))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
