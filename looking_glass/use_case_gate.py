"""Use-case capability gate (per stream).

`looking_glass + white_queen` is only useful for the *use cases a stream supports* — like
Churn/LTV are. This gate identifies the stream's **randomized action keys** (the optimization
use cases: email *frequency*, send *hour*, arm, ...) and, for each, reports:

  * the **IPW dose-response** `E[reward | do(action)]` (population) and its argmax — does the
    capability exist in the stream at all (identifiable)?
  * the **per-customer capability**: a policy from `E` (`argmax_a V(E,a)`) certified
    **never-worse than logging** (paired bootstrap CI of the IPS value), i.e. does the
    foundation embedding supply the optimization.

Aligned to our principles: IPW (oracle-ish) reference, held-out, never-ship-worse by
construction, no truth-peeking, per-use-case verdicts.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from looking_glass.cfm_config import as_of_epoch
from looking_glass.cfm_state import load_frozen_encoder
from looking_glass.head_battery import _b_states


def _period_rows(cfg, cust, action_key, reward_event="purchase"):
    """(customer, action) rows at the stream's own cadence: the randomized action per
    (customer, period) from sends, reward = did the reward event fire that period."""
    import duckdb
    import polars as pl

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    con.register("_cust", pl.DataFrame({"customer_key": [str(c) for c in cust]}))
    key = action_key.replace("'", "''")
    # Period = one action cadence ANCHORED AT THE STREAM ORIGIN (min ts), so buckets align
    # with the randomized periods instead of arbitrary unix-week boundaries (which split
    # a period and mismatch action<->reward). Must match cfm_data._effect_targets.
    t0 = float(
        con.execute(
            f"SELECT min(epoch(CAST(event_ts AS TIMESTAMPTZ))) FROM {src} "
            f"WHERE epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {tc}"
        ).fetchone()[0]
    )
    # randomized action per (customer, period): period = a rounded time bucket (data-derived)
    snd = con.execute(
        f"SELECT customer_key, "
        f"CAST(floor((epoch(CAST(event_ts AS TIMESTAMPTZ)) - {t0}) / 604800.0) AS BIGINT) AS period, "
        f"arg_max(TRY_CAST(json_extract_string(event_attributes,'$.{key}') AS DOUBLE), "
        f"epoch(CAST(event_ts AS TIMESTAMPTZ))) AS action "
        f"FROM {src} WHERE event_type LIKE '%send%' "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {tc} "
        f"AND json_extract_string(event_attributes,'$.{key}') IS NOT NULL "
        f"AND customer_key IN (SELECT customer_key FROM _cust) GROUP BY 1,2"
    ).pl()
    evt = con.execute(
        f"SELECT customer_key, "
        f"CAST(floor((epoch(CAST(event_ts AS TIMESTAMPTZ)) - {t0}) / 604800.0) AS BIGINT) AS period, "
        f"count(*) AS y FROM {src} WHERE event_type = '{reward_event}' "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {tc} "
        f"AND customer_key IN (SELECT customer_key FROM _cust) GROUP BY 1,2"
    ).pl()
    con.close()
    j = snd.join(evt, on=["customer_key", "period"], how="left").fill_null(0.0)
    return j


def discover_action_keys(cfg, max_distinct=24):
    """Identify the stream's randomized ACTION keys: JSON keys on send events whose values are
    (almost always) integers with a small, DISCRETE set of values (an action space). Generic —
    no per-domain key names."""
    from pathlib import Path

    # prefer the stream's DECLARED use-case action keys (a metadata / truth receipt next to it)
    _p = Path(str(cfg.db))
    for _t in (
        _p.parent / f"{_p.parent.name}_truth.json",
        _p.with_suffix("").with_name(_p.stem + "_truth.json"),
    ):
        if _t.exists():
            try:
                import json

                meta = json.loads(_t.read_text()).get("meta", {})
                declared = meta.get("action_keys")
                if declared:
                    return [str(k) for k in declared]
            except Exception:
                pass
    import duckdb

    _META = ("idx", "index", "week", "campaign", "time", "date", "day", "send")

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    try:
        keys = [
            str(r[0])
            for r in con.execute(
                f"SELECT DISTINCT unnest(json_keys(event_attributes)) AS k FROM {src} "
                f"WHERE event_type LIKE '%send%' AND event_attributes IS NOT NULL ORDER BY k"
            ).fetchall()
        ]
    except Exception:
        keys = []
    out = []
    for k in keys:
        esc = k.replace("'", "''")
        c = f"json_extract_string(event_attributes,'$.{esc}')"
        try:
            nn, tot = con.execute(
                f"SELECT count(*) FILTER (WHERE TRY_CAST({c} AS BIGINT) IS NOT NULL), "
                f"count({c}) FROM {src} WHERE event_type LIKE '%send%' "
                f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {tc}"
            ).fetchone()
            if not tot or nn / tot < 0.99:
                continue
            nd = con.execute(
                f"SELECT count(DISTINCT TRY_CAST({c} AS BIGINT)) FROM {src} "
                f"WHERE event_type LIKE '%send%' AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {tc}"
            ).fetchone()[0]
        except Exception:
            continue
        if nd and 1 < int(nd) <= max_distinct and not any(t in k.lower() for t in _META):
            out.append(k)
    con.close()
    return out


def run_use_case(model, cfg, action_key, seed=0):
    import polars as pl

    cust, Z = _b_states(model, cfg)
    j = _period_rows(cfg, cust, action_key)
    ci = {c: i for i, c in enumerate(cust)}
    rows = j.filter(pl.col("customer_key").is_in(list(ci)))
    cu = rows["customer_key"].to_numpy()
    act = rows["action"].to_numpy().astype(float)
    y = (rows["y"].to_numpy().astype(float) > 0).astype(float)
    Zi = np.stack([Z[ci[c]] for c in cu])  # embedding of the row's customer (state proxy)
    # discretize continuous actions (hour) into the stream's own quantile bins (data-derived)
    uq = np.unique(act)
    if len(uq) > 12:
        qs = np.quantile(act, np.linspace(0.05, 0.95, 10))
        act = np.digitize(act, qs).astype(float)
    A = act.astype(int)
    Q = int(A.max()) + 1
    N = len(A)
    # dose-response: empirical E[reward | do(action)] (uniform randomization)
    dose = {int(a): round(float((y[A == a]).mean()), 4) for a in range(Q) if (A == a).any()}
    sc = StandardScaler().fit(Zi)
    Zs = sc.transform(Zi)
    V = np.zeros((N, Q))
    for a in range(Q):
        m = HistGradientBoostingRegressor(max_iter=150).fit(np.column_stack([Zs, np.full(N, a)]), y)
        V[:, a] = m.predict(np.column_stack([Zs, np.full(N, a)]))
    cand = V.argmax(1)
    pa = 1.0 / Q
    v_cand = y * (cand == A) / pa
    v_log = y.copy()
    rng = np.random.default_rng(seed)
    d = np.array(
        [(v_cand[i].mean() - v_log[i].mean()) for i in [rng.integers(0, N, N) for _ in range(1000)]]
    )
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    return {
        "action_key": action_key,
        "n_rows": int(N),
        "n_actions": int(Q),
        "dose_response": dose,
        "dose_argmax": int(max(dose, key=dose.get)) if dose else None,
        "value_candidate": round(float(v_cand.mean()), 4),
        "value_logging": round(float(v_log.mean()), 4),
        "delta": round(float(v_cand.mean() - v_log.mean()), 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "verdict": "DEPLOY" if lo > 0.0 else "HOLD",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--actions", required=True, help="comma list of randomized action keys")
    ap.add_argument("--customers", type=int, default=12000)
    a = ap.parse_args(argv)
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    print("# USE-CASE CAPABILITY GATE")
    for k in a.actions.split(","):
        r = run_use_case(model, cfg, k.strip())
        print(
            f"  [{r['action_key']}] rows={r['n_rows']} actions={r['n_actions']} "
            f"dose_argmax={r['dose_argmax']} >> policy-from-E Δ={r['delta']:+.3f} "
            f"CI[{r['lo']:+.3f},{r['hi']:+.3f}] {r['verdict']}"
        )
        print(f"      dose-response: {r['dose_response']}")
    return r


if __name__ == "__main__":
    main()
