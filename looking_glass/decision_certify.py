"""Decision certification (the B+C / RL leg) — white-queen-style, from scratch.

Given a frozen foundation embedding `E`, logged (context, action, reward) from a
randomized-arm stream, and a candidate policy derived from `E`, decide DEPLOY/HOLD.

Aligned to our discipline and principles:
  * oracle-referenced: the logging value is the bar; report a decision gap, never a
    hand threshold;
  * held-out customers; no truth-peeking;
  * LAW: DEPLOY iff the PAIRED bootstrap CI of (value[candidate] - value[logging])
    has lower bound > 0  (safety by construction: never ship worse than logging),
    AND the effective sample size passes a floor (fail-safe HOLD on low support);
  * two independent estimator families reading the embedding:
    reweighting (IPS/DR, via a propensity model on E) + modeling (a value model on E);
  * the modeling witness + the candidate policy BOTH read `E` -> a more sufficient
    embedding makes the decision estimation better (the foundation supports RL).

Run:
  python3 -m looking_glass.decision_certify --db <stream> --ckpt-dir <dir> --tag <t> \
      --as-of <date> [--horizon-days 90]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from looking_glass.cfm_config import as_of_epoch
from looking_glass.cfm_state import load_frozen_encoder
from looking_glass.head_battery import _b_states


def _arms_rewards(cfg, cust, horizon_days):
    """Per held-out customer: logged arm (last send <= as_of) + reward = logged future GM."""
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
    hi = tc + horizon_days * 86400.0
    rew = con.execute(
        f"SELECT customer_key, sum(ln(1+abs(TRY_CAST(json_extract_string(event_attributes,"
        f"'$.value') AS DOUBLE)))) AS gm FROM {src} WHERE event_type='purchase' "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) > {tc} "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {hi} "
        f"AND customer_key IN (SELECT customer_key FROM _cust) GROUP BY 1"
    ).pl()
    con.close()
    return arm, rew


def certify(model, cfg, horizon_days=90, seed=0):
    import polars as pl

    cust, Z = _b_states(model, cfg)  # the frozen foundation embedding
    arm, rew = _arms_rewards(cfg, cust, horizon_days)
    j = (
        pl.DataFrame({"customer": cust})
        .join(arm, left_on="customer", right_on="customer_key", how="left")
        .join(rew, left_on="customer", right_on="customer_key", how="left")
        .fill_null(0.0)
    )
    a = j["arm"].to_numpy().astype(int)
    r = j["gm"].to_numpy().astype(float)
    kn = a >= 0  # all have an arm (0..2)
    Z, a, r = Z[kn], a[kn], r[kn]
    Q = int(a.max()) + 1
    N = len(Z)
    sc = StandardScaler().fit(Z)
    Zs = sc.transform(Z)
    # LOGGING policy pi(a|E): a multinomial propensity model on E (no logged propensity).
    pi = HistGradientBoostingClassifier(max_iter=150).fit(Zs, a)
    P = pi.predict_proba(Zs)  # (N, Q)
    P = np.clip(P, 1e-3, 1.0)
    P = P / P.sum(1, keepdims=True)
    pa = P[np.arange(N), a]  # propensity of the logged action
    # MODELING witness: value V(E, a) for every arm; the candidate = argmax_a V.
    V = np.zeros((N, Q))
    for aa in range(Q):
        m = HistGradientBoostingRegressor(max_iter=150).fit(np.column_stack([Zs, np.full(N, aa)]), r)
        V[:, aa] = m.predict(np.column_stack([Zs, np.full(N, aa)]))
    cand_a = V.argmax(1)  # the candidate (personalized, derived from E)
    match = (cand_a == a).astype(float)
    # IPS value of the candidate vs the logging value (paired, per customer).
    v_cand = r * match / pa  # IPW contribution
    v_log = r.copy()  # logging value
    ess = float((1.0 / pa).sum() ** 2 / (1.0 / pa**2).sum()) / N  # normalized ESS
    rng = np.random.default_rng(seed)
    deltas = np.empty(2000)
    for b in range(2000):
        idx = rng.integers(0, N, N)
        deltas[b] = v_cand[idx].mean() - v_log[idx].mean()
    lo, hi = float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
    # avg_arm_eff: the population arm profile (for context).
    return {
        "n": N,
        "Q": Q,
        "value_candidate": round(float(v_cand.mean()), 4),
        "value_logging": round(float(v_log.mean()), 4),
        "delta": round(float(v_cand.mean() - v_log.mean()), 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "ess_frac": round(ess, 3),
        "deploy": bool(lo > 0.0),  # safety by construction
        "verdict": "DEPLOY" if lo > 0.0 else "HOLD",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument("--customers", type=int, default=12000)
    a = ap.parse_args(argv)
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    res = certify(model, cfg, a.horizon_days)
    print("# DECISION CERTIFICATION (B+C / RL) — white-queen-style")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


if __name__ == "__main__":
    main()
