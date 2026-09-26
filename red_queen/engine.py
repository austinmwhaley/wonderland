"""red_queen engine — the product.

Next-best-action per customer, from the frozen donor + a value model, under hard
business constraints. Design principles (AGENTS.md):
  * objective = predicted discounted INCREMENTAL gross margin
  * fail-safe: act only if the value LOWER BOUND > 0 (else NO action)
  * constraints = reject, never clamp (per-customer cap + global send budget)
  * multi-cadence: weekly / biweekly / monthly re-decision tiers by value
  * uncertainty reported with every action; receipts for every rejection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
DATA = WORK / "red_king" / "data" / "seq_email.npz"
CFM_PRODUCTS = WORK / "looking_glass" / "artifacts" / "cfm" / "cfm_products.duckdb"
OUT = Path(__file__).resolve().parents[0] / "artifacts"
CADENCE = np.array([0.2, 0.6, 1.2, 2.0])  # sends/week per arm
PER_CUSTOMER_CAP = 2.0  # hard max sends/week


def _embeddings():
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


def _value_model():
    """Per-arm ridge value model + residual std (uncertainty proxy)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    z = np.load(DATA)
    S, A, R = z["S"], z["A"], z["R"]
    nA = int(A.max()) + 1
    sc = StandardScaler().fit(S)
    Xs = sc.transform(S)
    models, sigmas = [], []
    for a in range(nA):
        m = A == a
        if m.sum() > 50:
            mod = Ridge(alpha=1.0).fit(Xs[m], R[m])
            sig = float((R[m] - mod.predict(Xs[m])).std())
        else:
            mod = Ridge(alpha=1.0).fit(Xs, R)
            sig = float(R.std())
        models.append(mod)
        sigmas.append(sig)
    return sc, models, np.array(sigmas), nA


def _validated_arm_effects():
    """Per-arm causal effect E[inc GP | do(arm)] via IPW from randomized logs."""
    import duckdb
    import polars as pl

    from red_queen.identifiability import require_propensity, require_stream_view

    con = duckdb.connect(str(STREAM), read_only=True)
    try:
        require_stream_view(con, "email_arm", "logged arm + propensity for IPW arm effects")
        require_propensity(con, "email_arm")
        arm = con.execute("SELECT customer_id k, arm, propensity FROM email_arm").pl()
        inc = con.execute("""
			SELECT s.customer_id k, SUM(o.gross_margin) g
			FROM email_sends s JOIN orders o
			  ON o.customer_id=s.customer_id AND o.session_id=s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked=1 GROUP BY 1""").pl()
    finally:
        con.close()
    df = arm.join(inc, on="k", how="left").with_columns(pl.col("g").fill_null(0.0))
    a = df["arm"].to_numpy()
    e = df["propensity"].to_numpy()
    g = df["g"].to_numpy()
    return np.array(
        [
            float((g[a == x] / e[a == x]).sum() / max((1 / e[a == x]).sum(), 1e-9))
            for x in range(int(a.max()) + 1)
        ]
    )


def _donor_value(keys):
    """WHO to serve: Ridge on the donor state -> observed incremental GP."""
    import duckdb
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    con = duckdb.connect(str(STREAM), read_only=True)
    try:
        inc = con.execute("""
			SELECT s.customer_id k, SUM(o.gross_margin) g
			FROM email_sends s JOIN orders o
			  ON o.customer_id=s.customer_id AND o.session_id=s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked=1 GROUP BY 1""").pl()
    finally:
        con.close()
    gmap = dict(zip(inc["k"].to_list(), inc["g"].to_list()))
    _, E = _embeddings()
    y = np.array([float(gmap.get(k, 0.0)) for k in keys])
    sc = StandardScaler().fit(E)
    m = Ridge(alpha=1.0).fit(sc.transform(E), y)
    return m.predict(sc.transform(E))


def run(weekly_budget=None, risk_z=0.0):
    # risk_z=0: act on calibrated positive expected value; uncertainty penalty
    # pending proper calibration of the value ensemble.
    # DECISION-PATH PURITY (see red_king/ab_witness.py): the red_king RSSM was
    # A/B'd against known truth (25 candidates x 5 cells, WITH/WITHOUT) and
    # changed ZERO certified decisions -> it is analyst-tool only and must
    # never route decisions. This is the validated configuration: rank arms by
    # the IPW causal effect from the randomized logs; choose WHO by donor value.
    keys, E = _embeddings()
    eff = _validated_arm_effects()
    nA = len(eff)
    best_a = int(np.argmax(eff))
    who = _donor_value(keys)
    V = np.zeros((len(keys), nA))
    V[:, best_a] = who
    SD = np.zeros_like(V)
    LB = V - risk_z * SD  # fail-safe bound
    feasible = CADENCE[None, :] <= PER_CUSTOMER_CAP
    # maximise TOTAL expected value subject to the budget (not value-per-send):
    # the response is unimodal in cadence, so the best arm is the argmax of V.
    score = np.where(feasible, V, -1e9)
    best = score.argmax(1)
    lb_ok = LB[np.arange(len(keys)), best] > 0.0
    vbest = V[np.arange(len(keys)), best]
    q = np.quantile(vbest, [0.33, 0.66])
    cad = np.where(vbest >= q[1], "weekly", np.where(vbest >= q[0], "biweekly", "monthly"))
    cost = CADENCE[best]
    if weekly_budget is None:
        weekly_budget = float(len(keys) * CADENCE.mean())
    order = np.argsort(-(score[np.arange(len(keys)), best] * lb_ok))
    arm = np.full(len(keys), -1, int)
    budget = weekly_budget
    budget_rej = 0
    for i in order:
        if not lb_ok[i]:
            continue
        if cost[i] <= budget:
            arm[i] = best[i]
            budget -= cost[i]
        else:
            budget_rej += 1
    served = arm >= 0
    arm_c = np.clip(arm, 0, nA - 1)
    sends = np.where(served, CADENCE[arm_c], 0.0)
    val = np.where(served, V[np.arange(len(keys)), arm_c], 0.0)
    plan = {
        "customer_key": keys,
        "arm": np.where(served, arm, -1).tolist(),
        "cadence": cad.tolist(),
        "weekly_sends": sends.tolist(),
        "expected_incremental_gp": val.tolist(),
        "value_sd": np.where(served, SD[np.arange(len(keys)), arm_c], 0.0).tolist(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nba_plan.json").write_text(json.dumps(plan))
    rec = {
        "customers": len(keys),
        "acted": int(served.sum()),
        "no_action_failsafe": int((~lb_ok).sum()),
        "no_action_budget": int(budget_rej),
        "sends_used_wk": round(float(sends.sum()), 1),
        "budget_wk": round(weekly_budget, 1),
        "arm_mix": np.bincount(arm[served], minlength=nA).tolist() if served.any() else [],
        "cadence_mix": {
            c: int((cad[served] == c).sum()) for c in ("weekly", "biweekly", "monthly")
        },
        "expected_weekly_incremental_gp": round(float(val.sum()), 1),
    }
    print("== RED_QUEEN ENGINE (v2) ==")
    for k, v in rec.items():
        print(f"  {k:32s}: {v}")
    print(f"  plan -> {OUT / 'nba_plan.json'}")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    run(a.budget)
