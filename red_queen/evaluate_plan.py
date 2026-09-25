"""Ground-truth evaluation of red_queen's plan.

Does acting on red_queen's next-best-action plan beat the logging behavior on
REALIZED incremental gross margin? Arms are randomized with logged propensity, so
IPS gives an UNBIASED estimate of any plan's value from logged rewards.

  V(pi) = (1/n) sum_i  r_i * 1[a_i == pi(s_i)] / e_i        (IPS, unbiased)
  V(beh)= mean_i r_i                                        (observed behavior)
Also reports the best CONSTANT arm plan for reference, and a bootstrap CI on the lift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
PLAN = Path(__file__).resolve().parents[0] / "artifacts" / "nba_plan.json"


def _load():
    import duckdb

    con = duckdb.connect(str(STREAM), read_only=True)
    try:
        arm = con.execute("SELECT customer_id AS customer_key, arm, propensity FROM email_arm").pl()
        inc = con.execute("""
			SELECT s.customer_id AS customer_key, SUM(o.gross_margin) g
			FROM email_sends s JOIN orders o
			  ON o.customer_id=s.customer_id AND o.session_id=s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked=1 GROUP BY 1""").pl()
    finally:
        con.close()
    import polars as pl

    df = arm.join(inc, on="customer_key", how="left").with_columns(pl.col("g").fill_null(0.0))
    return df


def evaluate(risk_aversion_z=1.96, seed=0):
    df = _load()
    plan = json.loads(PLAN.read_text())
    pmap = dict(zip(plan["customer_key"], plan["arm"]))
    keys = df["customer_key"].to_list()
    a = df["arm"].to_numpy()
    e = df["propensity"].to_numpy()
    r = df["g"].to_numpy()
    pi = np.array([pmap.get(k, -1) for k in keys])
    served = pi >= 0
    # estimate per-arm ground truth E[r|do(arm)] via IPW (for best-constant reference)
    nA = int(a.max()) + 1
    gt = np.array(
        [float((r[a == x] / e[a == x]).sum() / max((1 / e[a == x]).sum(), 1e-9)) for x in range(nA)]
    )
    best_arm = int(gt.argmax())
    # evaluate over served customers
    ix = np.where(served)[0]
    rr, aa, ee, pp = r[ix], a[ix], e[ix], pi[ix]

    def ips(plan_arm):
        return float(np.mean(rr * (aa == plan_arm) / ee))

    v_plan = ips(pp)
    v_beh = float(np.mean(rr))
    v_best = ips(np.full(len(ix), best_arm))
    # bootstrap CI on lift (plan - behavior) over served customers
    rng = np.random.default_rng(seed)
    B = 500
    d = []
    for _ in range(B):
        j = rng.integers(0, len(ix), len(ix))
        d.append(np.mean(rr[j] * (aa[j] == pp[j]) / ee[j]) - np.mean(rr[j]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {
        "served": int(len(ix)),
        "n_total": int(len(keys)),
        "nA": nA,
        "gt_per_arm": [round(x, 2) for x in gt],
        "best_arm_by_gt": best_arm,
        "behavior_value": round(v_beh, 2),
        "plan_value_ips": round(v_plan, 2),
        "best_constant_value_ips": round(v_best, 2),
        "lift": round(v_plan - v_beh, 2),
        "lift_ci95": [round(lo, 2), round(hi, 2)],
        "beats_behavior": bool(lo > 0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    res = evaluate()
    print("== RED_QUEEN GROUND-TRUTH EVAL (IPS on randomized data) ==")
    for k, v in res.items():
        print(f"  {k:24s}: {v}")


if __name__ == "__main__":
    main()
