"""Per-channel frequency optimization + discount optimization (white_queen).

For each marketing channel (email/sms/push) and for discount, build logs from
rabbit_hole's unified `contact_sends`, then run white_queen's offline-RL pipeline
to LEARN and CERTIFY a policy with a DEPLOY/HOLD verdict.

  state  = frozen donor embedding (customer)
  action = cadence arm (frequency) OR discount bucket
  reward = incremental gross margin (order linked to the contact, else 0)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
CFM = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"


def _state():
	import duckdb
	con = duckdb.connect(str(CFM), read_only=True)
	try:
		df = con.execute("""
			SELECT customer_key, embedding FROM (
				SELECT customer_key, anchor_epoch, embedding,
				       row_number() OVER (PARTITION BY customer_key ORDER BY anchor_epoch DESC) rn
				FROM anchor_embeddings) WHERE rn=1""").pl()
	finally:
		con.close()
	return df


def _logs(channel, action):
	import duckdb
	con = duckdb.connect(str(STREAM), read_only=True)
	try:
		if action == "arm":
			df = con.execute("""
				SELECT cs.customer_id k, cs.arm a, COALESCE(o.gross_margin,0) r
				FROM contact_sends cs LEFT JOIN orders o ON o.transaction_id = cs.converted_order_id
				WHERE cs.channel = ? AND cs.arm IS NOT NULL""", [channel]).pl()
		else:  # discount
			df = con.execute("""
				SELECT cs.customer_id k, cs.discount_pct a, COALESCE(o.gross_margin,0) r
				FROM contact_sends cs LEFT JOIN orders o ON o.transaction_id = cs.converted_order_id
				WHERE cs.channel = ? AND cs.discount_pct IS NOT NULL""", [channel]).pl()
	finally:
		con.close()
	emb = _state()
	return df.join(emb, left_on="k", right_on="customer_key", how="inner")


def optimize(channel, action="arm", max_rows=15000, seed=0):
	import polars as pl
	df = _logs(channel, action)
	# action -> discrete bucket
	a = df["a"].to_numpy().astype(np.float64)
	if action == "discount":
		edges = np.unique(np.quantile(a, np.linspace(0, 1, 5)[1:-1]))
		act = np.digitize(a, edges).astype(np.int64); nA = int(act.max()) + 1
	else:
		act = a.astype(np.int64); nA = int(act.max()) + 1
	obs = np.stack(df["embedding"].to_list()).astype(np.float32)
	rew = df["r"].to_numpy().astype(np.float32)
	n = len(obs)
	if n > max_rows:
		idx = np.random.default_rng(seed).choice(n, max_rows, replace=False)
		obs, act, rew = obs[idx], act[idx], rew[idx]
	from white_queen.tribunal.ope.pipeline import run as wq_run
	rep = wq_run({"obs": obs, "act": act, "rew": rew, "next_obs": obs,
				  "done": np.ones(len(obs), dtype=np.float32)},
				 algorithms=("iql", "cql", "bc"), nA=nA, fast=True,
				 ensemble_K=2, offline_steps=1500, seed=seed)
	return {"channel": channel, "action": action, "rows": int(n) if n <= max_rows else max_rows,
			"nA": nA, "deployed": rep.get("deployed"), "behavior": rep.get("behavior_mean"),
			"bar": rep.get("bar"), "n_candidates": rep.get("n_candidates")}


def main(argv=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("--max_rows", type=int, default=15000)
	a = ap.parse_args(argv)
	print("== CHANNEL / DISCOUNT OPTIMIZATION (white_queen) ==")
	for chan in ("email", "sms", "push"):
		print("  ", optimize(chan, "arm", a.max_rows))
	print("  ", optimize("email", "discount", a.max_rows))


if __name__ == "__main__":
	main()
