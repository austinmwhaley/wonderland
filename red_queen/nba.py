"""red_queen — Next-Best-Action engine.

Consumes the frozen donor + a learned action-value model and emits, per customer,
the BEST action under hard constraints:
  * objective  = predicted discounted INCREMENTAL gross margin (AGENTS.md reward)
  * action     = email cadence arm (0..3), with weekly send intensity
  * constraint = a global weekly send BUDGET (capacity)
  * cadence    = sends spread across weekly buckets

Output: a per-customer action plan + a receipt (arm mix, sends used, expected value).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "red_king" / "data" / "seq_email.npz"
CFM_PRODUCTS = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
CADENCE = np.array([0.2, 0.6, 1.2, 2.0])  # sends/week per arm (known schedule)


def _latest_embeddings():
    import duckdb

    con = duckdb.connect(str(CFM_PRODUCTS), read_only=True)
    try:
        df = con.execute("""
			SELECT customer_key, embedding FROM (
				SELECT customer_key, anchor_epoch, embedding,
				       row_number() OVER (PARTITION BY customer_key ORDER BY anchor_epoch DESC) rn
				FROM anchor_embeddings) WHERE rn = 1""").pl()
    finally:
        con.close()
    return df["customer_key"].to_list(), np.stack(df["embedding"].to_list()).astype(np.float32)


def _value_model(seed=0):
    """Per-arm ridge value model E[reward | s, arm] on the sequential dataset."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    z = np.load(DATA)
    S, A, R = z["S"], z["A"], z["R"]
    nA = int(A.max()) + 1
    sc = StandardScaler().fit(S)
    Xs = sc.transform(S)
    models = {}
    for a in range(nA):
        m = A == a
        models[a] = (
            Ridge(alpha=1.0).fit(Xs[m], R[m]) if m.sum() > 50 else Ridge(alpha=1.0).fit(Xs, R)
        )
    return sc, models, nA


def run(weekly_send_budget=None, seed=0):
    sc, models, nA = _value_model(seed)
    keys, E = _latest_embeddings()
    Xs = sc.transform(E)
    V = np.stack([models[a].predict(Xs) for a in range(nA)], 1)  # (N, nA)
    vps = V / CADENCE[None, :]  # value per send
    if weekly_send_budget is None:
        weekly_send_budget = float(len(keys) * CADENCE.mean())  # capacity = behaviour
    # Greedy allocation under the budget: pick the best value-per-send, decrement.
    order_best = vps.max(1)
    best_arm = vps.argmax(1)
    cost = CADENCE[best_arm]
    idx = np.argsort(-(order_best / np.maximum(cost, 1e-9)))
    budget = weekly_send_budget
    arm = np.full(len(keys), -1, dtype=int)
    for i in idx:
        if cost[i] <= budget:
            arm[i] = best_arm[i]
            budget -= cost[i]
        else:
            # fall back to the cheapest arm that fits
            for a in np.argsort(CADENCE):
                if CADENCE[a] <= budget:
                    arm[i] = a
                    budget -= CADENCE[a]
                    break
    unserved = int((arm < 0).sum())
    served = arm >= 0
    arm_clean = np.where(served, arm, 0)
    weekly = np.where(served, CADENCE[arm_clean], 0.0)
    exp_value = np.where(served, V[np.arange(len(keys)), arm_clean], 0.0)
    plan = {
        "customer_key": keys,
        "arm": np.where(served, arm, -1).tolist(),
        "weekly_sends": weekly.tolist(),
        "expected_gp": exp_value.tolist(),
    }
    import json

    out = Path(__file__).resolve().parents[0] / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "nba_plan.json").write_text(json.dumps(plan))
    rec = {
        "customers": len(keys),
        "budget_sends_wk": round(weekly_send_budget, 1),
        "sends_used_wk": round(float(weekly.sum()), 1),
        "arm_mix": np.bincount(arm[served], minlength=nA).tolist(),
        "unserved": unserved,
        "expected_weekly_gp": round(float(exp_value.sum()), 1),
    }
    print("== RED_QUEEN NEXT-BEST-ACTION ==")
    for k, v in rec.items():
        print(f"  {k:20s}: {v}")
    print(f"  plan -> {out / 'nba_plan.json'}")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    run(a.budget)
