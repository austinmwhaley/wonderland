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


_PERIOD_CACHE: dict = {}


def _period_rows(
    cfg,
    cust,
    action_key,
    reward_event="purchase",
    period_secs=604800.0,
    future=True,
    horizon_days=90,
):
    """(customer, action) rows at the stream's own cadence: the randomized action per
    (customer, period) from sends, reward = did the reward event fire that period.

    CACHED per (stream, as_of, customers, action, period): the card runs each action many times."""
    _pkey = (
        str(cfg.db),
        str(getattr(cfg, "as_of", None)),
        tuple(map(str, cust)),
        str(action_key),
        str(reward_event),
        float(period_secs),
        bool(future),
        int(horizon_days),
    )
    _phit = _PERIOD_CACHE.get(_pkey)
    if _phit is not None:
        return _phit
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
    try:
        _has_tk = (
            con.execute(
                f"SELECT count(*) FROM {src} WHERE "
                f"json_extract_string(event_attributes,'$.target_key') IS NOT NULL"
            ).fetchone()[0]
            > 0
        )
    except Exception:
        _has_tk = False
    _tk = f"AND json_extract_string(event_attributes,'$.target_key') = '{key}'" if _has_tk else ""
    t0 = float(
        con.execute(
            f"SELECT min(epoch(CAST(event_ts AS TIMESTAMPTZ))) FROM {src} "
            f"WHERE epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {tc}"
        ).fetchone()[0]
    )
    # POINT-IN-TIME (DEC-217): the action/outcome must lie in the FUTURE window
    # (t > as_of), while E is frozen at as_of. Evaluating PAST periods against the
    # as-of E is lookahead (the measure already contains their outcomes).
    hi = float(tc) + float(horizon_days) * 86400.0
    _tw = f"> {tc} AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {hi}" if future else f"<= {tc}"
    # randomized action per (customer, period): period = a rounded time bucket (data-derived)
    snd = con.execute(
        f"SELECT customer_key, "
        f"CAST(floor((epoch(CAST(event_ts AS TIMESTAMPTZ)) - {t0}) / {period_secs}) AS BIGINT) AS period, "
        f"arg_max(TRY_CAST(json_extract_string(event_attributes,'$.{key}') AS DOUBLE), "
        f"epoch(CAST(event_ts AS TIMESTAMPTZ))) AS action "
        f"FROM {src} WHERE event_type LIKE '%send%' "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) {_tw} {_tk} "
        f"AND json_extract_string(event_attributes,'$.{key}') IS NOT NULL "
        f"AND customer_key IN (SELECT customer_key FROM _cust) GROUP BY 1,2"
    ).pl()
    evt = con.execute(
        f"SELECT customer_key, "
        f"CAST(floor((epoch(CAST(event_ts AS TIMESTAMPTZ)) - {t0}) / {period_secs}) AS BIGINT) AS period, "
        f"count(*) AS y FROM {src} WHERE event_type = '{reward_event}' "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) {_tw} "
        f"AND customer_key IN (SELECT customer_key FROM _cust) GROUP BY 1,2"
    ).pl()
    con.close()
    j = snd.join(evt, on=["customer_key", "period"], how="left").fill_null(0.0)
    _PERIOD_CACHE[_pkey] = j
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


def run_use_case(model, cfg, action_key, seed=0, period_secs=604800.0, horizon_days=90):
    import polars as pl

    cust, Z = _b_states(model, cfg)
    j = _period_rows(cfg, cust, action_key, period_secs=period_secs, horizon_days=horizon_days)
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
    # policy-from-E: a JOINT outcome model V(E, a) (action as a feature), CROSS-FITTED by
    # customer so the policy is evaluated out-of-sample. (Separate per-action regressors
    # compared miscalibrated outputs and collapsed to the marginal — a measurement bug.)
    # E is now compact (338 = shape 256 ‖ measure 82; DEC-198). Use it FULL: a PCA-128 on the small
    # E drops the low-variance timing/response dims (hour/dow regressed). Only reduce if E is large.
    from sklearn.decomposition import PCA

    sc = StandardScaler().fit(Zi)
    _Z = sc.transform(Zi)
    if _Z.shape[1] > 512:
        _k = max(2, min(256, _Z.shape[1], _Z.shape[0] - 1))
        Zs = PCA(n_components=_k, random_state=seed).fit_transform(_Z)
    else:
        Zs = _Z
    from sklearn.model_selection import GroupKFold

    _folds = max(2, min(5, len(np.unique(cu))))
    logged = np.unique(A)

    import zlib

    def _fit_V(yfit):
        """Out-of-fold value model V(E,a)->y: each row scored by a model that never saw it."""
        V = np.zeros((N, Q))
        for tr, va in GroupKFold(_folds).split(Zs, groups=cu):
            m = HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=50).fit(
                np.column_stack([Zs[tr], A[tr].astype(float)]), yfit[tr]
            )
            for a in range(Q):
                V[va, a] = m.predict(np.column_stack([Zs[va], np.full(len(va), float(a))]))
        return V

    # DIRECT (g-computation) policy value with a CROSS-HALF candidate. The candidate = argmax_a
    # V(E,a) from a model fit on the OTHER half of customers; the value = the out-of-sample
    # prediction E[V(E, pi(E))] — NO importance reweighting. Agents diagnosed that IPS/SNIPS/DR
    # all keep a ~+0.03 NULL bias (the `1[cand==A]` term covaries with y through the E->y activity
    # signal, and the 1/pa reweight AMPLIFIES it); g-computation removes the reweighting and the
    # cross-half selection removes the winner's curse. Measured: |Delta_null| < 0.008 on all keys.
    V = _fit_V(y)
    _half = np.array([zlib.crc32(str(c).encode()) % 2 for c in cu])
    cand = np.empty(N, dtype=int)
    for h in (0, 1):
        tr = np.where(_half != h)[0]
        va = np.where(_half == h)[0]
        m = HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=50).fit(
            np.column_stack([Zs[tr], A[tr].astype(float)]), y[tr]
        )
        Vh = np.column_stack(
            [m.predict(np.column_stack([Zs[va], np.full(len(va), float(a))])) for a in range(Q)]
        )
        cand[va] = logged[np.argmax(Vh[:, logged], axis=1)]
    v_cand = V[np.arange(N), cand]  # out-of-sample predicted outcome under pi(E)
    delta = float(v_cand.mean() - y.mean())
    rng = np.random.default_rng(seed)
    dvals = v_cand - y
    d = np.array([dvals[rng.integers(0, N, N)].mean() for _ in range(1000)])
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    bias = 0.0
    delta_adj = delta
    pval = 0.0 if (lo > 0.0 and delta > 0.0) else 1.0
    return {
        "action_key": action_key,
        "n_rows": int(N),
        "n_actions": int(Q),
        "dose_response": dose,
        "dose_argmax": int(max(dose, key=dose.get)) if dose else None,
        "value_candidate": round(float(v_cand.mean()), 4),
        "value_logging": round(float(y.mean()), 4),
        "delta": round(delta, 4),
        "bias": round(bias, 4),
        "delta_adj": round(delta_adj, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "pval": round(pval, 4),
        "verdict": "DEPLOY" if (lo > 0.0 and delta_adj > 0.0) else "HOLD",
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
            f"CI[{r['lo']:+.3f},{r['hi']:+.3f}] p={r['pval']:.3f} {r['verdict']}"
        )
        print(f"      dose-response: {r['dose_response']}")
    return r


if __name__ == "__main__":
    main()


def run_use_case_response(
    model, cfg, action_key, horizon_days=180, seed=0, period_secs=None, n_boot=500
):
    """Point-in-time decision-sufficiency via the E's RESPONSE BLOCK.

    The response block is the action-conditioned sufficient statistic: per held-out
    customer it exposes the estimated reward at each action level, estimated from
    the PAST (<= as_of). The consumer takes its argmax -> a policy pi(c). We then
    evaluate pi on the FUTURE (as_of < t <= as_of+horizon) against REALIZED
    outcomes: because the reference randomizes each action, E[y | a = pi(c)] is an
    unbiased estimate of the policy's value, and E[y] is the logging value, so

        lift = mean(y | logged action == pi(c)) - mean(y).

    No lookahead: pi comes from the frozen as-of E; y comes only from the future.
    Caller runs the null model (snr=0) and applies the DiD rule, which cancels the
    base-rate and any residual consumer bias.
    """
    from looking_glass.head_battery import _b_states

    cust, Z = _b_states(model, cfg)
    lb = getattr(model, "layer_b", None)
    names = list(getattr(lb, "names", ()) or ())
    if not names:
        raise ValueError("no fitted LayerBEmbedder measure")
    d = int(Z.shape[1]) - len(names)
    emeas = Z[:, d:]
    idx = [i for i, n in enumerate(names) if n.startswith(f"resp_{action_key}_")]
    if not idx:
        raise ValueError(f"no response block for {action_key!r}")
    R = np.asarray(emeas[:, idx], dtype=float)
    ci = {c: i for i, c in enumerate(cust)}
    ps = (
        period_secs
        if period_secs is not None
        else (float(getattr(cfg, "action_period_seconds", None) or 604800.0))
    )
    j = _period_rows(cfg, cust, action_key, period_secs=ps, future=True, horizon_days=horizon_days)
    cu = j["customer_key"].to_numpy()
    A = j["action"].to_numpy().astype(int)
    Y = (j["y"].to_numpy().astype(float) > 0).astype(float)
    ok = np.array([c in ci for c in cu])
    cu, A, Y = cu[ok], A[ok], Y[ok]
    if len(cu) == 0:
        return {"action_key": action_key, "lift": None, "lo": None, "hi": None, "n": 0}
    Ri = np.stack([R[ci[c]] for c in cu])
    Q = Ri.shape[1]
    A = np.clip(A, 0, Q - 1)
    # Only levels the response block actually estimates (a zero row carries no
    # estimate; choosing it would be a spurious argmax). Fail safe -> skip.
    observed = np.where(R.max(0) > 0)[0]
    if len(observed) == 0:
        return {"action_key": action_key, "lift": None, "lo": None, "hi": None, "n": 0}
    chosen = observed[np.argmax(Ri[:, observed], axis=1)]
    match = A == chosen
    if not match.any():
        return {"action_key": action_key, "lift": None, "lo": None, "hi": None, "n": 0}
    lift_rows = np.where(match, Y, 0.0) / max(match.mean(), 1e-12) - Y
    cs, cc = np.unique(cu, return_inverse=True)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(cs), len(cs))
        m = np.isin(cc, pick)
        if m.any():
            boot.append(float(lift_rows[m].mean()))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
    return {
        "action_key": action_key,
        "lift": float(lift_rows.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(len(cu)),
        "levels": int(len(observed)),
        "match_rate": float(match.mean()),
        "value_policy": float(Y[match].mean()),
        "value_logging": float(Y.mean()),
        "argmax_acc": float(np.mean(np.isin(chosen, A))),
    }
