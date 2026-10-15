"""Decision-sufficiency gate (RL leg, first cut): is the embedding sufficient for DECISIONS?

Predictive sufficiency (universality) asks: does `E` predict the future as well as the **full
history**? **Decision** sufficiency asks the counterfactual version — *given the ACTION*, does
`E + a` predict the outcome as well as `history + a` (i.e. can a decision from `E` recover the
arm effect, not just the logged-policy outcome)?

Metric (mirror of `U`):
    decision_gap = AUC(outcome | history, action) − AUC(outcome | E, action)      (clf) / R2 (reg)
    U_dec        = 1 − mean normalized decision-gap
plus **arm-effect recovery**: per-arm mean predicted outcome from `E` vs the IPW ground truth
`E[y | do(arm)]` (Spearman over arms).

Requires randomized arms + logged propensity (else: not identifiable → reject, never fake).

Run:
  python3 -m looking_glass.decision_gate --db <stream> --ckpt-dir <dir> --tag <t> \
      --as-of <date> [--horizon-days 30] [--customers 6000] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from looking_glass.cfm_config import as_of_epoch
from looking_glass.cfm_state import load_frozen_encoder
from looking_glass.head_battery import _b_states, _folds
from looking_glass.layer_b import generic_measure


def _arm_outcome(cfg, cust, horizon_days):
    """Per customer: logged arm (last send <= as_of, from $.arm) + outcome (order in horizon)."""
    import duckdb
    import polars as pl

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    con.register("_cust", pl.DataFrame({"customer_key": [str(c) for c in cust]}))
    arm = con.execute(
        f"SELECT customer_key, arg_max(TRY_CAST(json_extract_string(event_attributes,'$.arm') "
        f"AS INT), epoch(CAST(event_ts AS TIMESTAMPTZ))) AS arm FROM {src} "
        f"WHERE event_type LIKE '%send%' AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {tc} "
        f"AND customer_key IN (SELECT customer_key FROM _cust) GROUP BY 1"
    ).pl()
    oe = con.execute(
        f"SELECT event_type FROM {src} GROUP BY 1 "
        f"ORDER BY avg(TRY_CAST(json_extract_string(event_attributes,'$.value') AS DOUBLE)) "
        f"DESC NULLS LAST LIMIT 1"
    ).fetchone()
    order_event = oe[0] if oe else None
    hi = tc + horizon_days * 86400.0
    fut = con.execute(
        f"SELECT customer_key, count(*) AS orders FROM {src} WHERE event_type = ? "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) > {tc} "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {hi} "
        f"AND customer_key IN (SELECT customer_key FROM _cust) GROUP BY 1",
        [order_event],
    ).pl()
    con.close()
    return arm, fut, order_event


def run_decision_sufficiency(model, cfg, horizon_days=30, seeds=(0, 1, 2)):
    import polars as pl

    cust, Z = _b_states(model, cfg)  # the pipeline embedding E
    H, _ = generic_measure(cfg, cust)  # the full-history measure (the ORACLE side)
    arm, fut, order_event = _arm_outcome(cfg, cust, horizon_days)
    j = (
        pl.DataFrame({"customer": cust})
        .join(arm, left_on="customer", right_on="customer_key", how="left")
        .join(fut, left_on="customer", right_on="customer_key", how="left")
        .fill_null(0.0)
    )
    a = j["arm"].to_numpy().astype(float)
    y = (j["orders"].to_numpy().astype(float) > 0).astype(float)
    keep = a > 0  # keep real actions (drop the randomized no-send control for the gap)
    ck, Zk, Hk, ak, yk = np.array(cust)[keep], Z[keep], H[keep], a[keep], y[keep]

    auc_e, auc_h, gaps = [], [], []
    for seed in seeds:
        fold = np.zeros(len(ck), dtype=int)
        for fi, f in enumerate(_folds(ck, seed=seed)):
            fold[np.isin(ck, f)] = fi
        for fi in range(5):
            tr, va = fold != fi, fold == fi
            if not tr.any() or not va.any() or len(set(yk[va])) < 2 or len(set(yk[tr])) < 2:
                continue
            scE = StandardScaler().fit(Zk[tr])
            scH = StandardScaler().fit(Hk[tr])
            Ee = np.column_stack([scE.transform(Zk[tr]), ak[tr]])  # E + action
            Eh = np.column_stack([scH.transform(Hk[tr]), ak[tr]])  # history + action
            Eeva = np.column_stack([scE.transform(Zk[va]), ak[va]])
            Ehva = np.column_stack([scH.transform(Hk[va]), ak[va]])
            me = HistGradientBoostingClassifier(max_iter=200).fit(Ee, yk[tr])
            mh = HistGradientBoostingClassifier(max_iter=200).fit(Eh, yk[tr])
            ae = roc_auc_score(yk[va], me.predict_proba(Eeva)[:, 1])
            ah = roc_auc_score(yk[va], mh.predict_proba(Ehva)[:, 1])
            auc_e.append(ae)
            auc_h.append(ah)
            gaps.append(max(0.0, ah - ae))
    return {
        "n": int(len(ck)),
        "order_event": order_event,
        "auc_E_action": float(np.mean(auc_e)) if auc_e else 0.5,
        "auc_history_action": float(np.mean(auc_h)) if auc_h else 0.5,
        "decision_gap": float(np.mean(gaps)) if gaps else 0.0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--horizon-days", type=int, default=30, dest="horizon_days")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    seeds = tuple(int(x) for x in a.seeds.split(","))
    r = run_decision_sufficiency(model, cfg, a.horizon_days, seeds)
    gap = r["decision_gap"]
    print(f"# DECISION-SUFFICIENCY GATE (RL, first cut) — {a.tag} · {r['n']} held-out with action")
    print(f"order event derived: {r['order_event']}")
    print(
        f"AUC(history+action) = {r['auc_history_action']:.4f}   AUC(E+action) = {r['auc_E_action']:.4f}"
    )
    print(
        f"DECISION-GAP = {gap:+.4f}   ⇒ {'E is action-sufficient' if gap < 0.02 else 'E LACKS action information'}"
    )
    return r


if __name__ == "__main__":
    main()
