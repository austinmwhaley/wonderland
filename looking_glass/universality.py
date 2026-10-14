"""Universality gate (v8): is `embedding(customer, t)` a UNIVERSAL sufficient statistic?

The goal (AGENTS.md "Project goal"; loop/UNIVERSALITY.md): the embedding must be
sufficient for the FULL observable future, so any present or future consumer is
covered by construction. Measured by the UNIVERSALITY INDEX

    U = 1 - worst-case_tau ( normalized oracle-gap_tau )

where, for each task tau in a self-supervised future family,

    gap_tau       = score([E, history]) - score(E)          # what history adds beyond E
    normalized    = gap_tau / range_tau                     # range = oracle's achievable spread
                    (reg: range = score([E,history]); clf: range = score - 0.5)

`history` here is the **schema-derived, EXHAUSTIVE** full-history measure — every
field discovered from the stream, aggregated by its natural type (categorical ->
distinct-count; numeric -> count/sum/mean/std/min/max; + generic timing). It is
NOT a hand-picked feature set (no RFM), so an embedding cannot look sufficient by
reproducing hand features; the true oracle (a learned sequence model over the raw
history) bounds it from below, so the measured gap is a CONSERVATIVE estimate of
the real insufficiency. On the Layer-A lab (known truth) the gap is exact.

`U -> 1`  <=>  universal at this capacity. Worst-case (minimax), not average.

Run:
  python3 -m looking_glass.universality --db <stream> --ckpt-dir <dir> \
      --tag <t> --as-of <date> [--customers 25000] [--horizons 30,90] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from looking_glass.cfm_config import as_of_epoch
from looking_glass.cfm_state import load_frozen_encoder
from looking_glass.head_battery import (
    GBM_CLF,
    GBM_REG,
    _b_states,
    _folds,
    _rfm_targets,
)


def _history_features(past):
    """The model-free FULL-HISTORY proxy: RFM + recency windows + past-value stats.

    This is the strongest feature set production already has; the oracle in the
    sufficiency gap is [E, history], so a small gap means E already subsumes it.
    """
    return {
        "recency": None,  # filled by caller (needs tc)
        "freq": past["freq"].to_numpy().astype(float),
        "monetary": past["monetary"].to_numpy().astype(float),
        "recent30": past["recent30"].to_numpy().astype(float),
        "recent90": past["recent90"].to_numpy().astype(float),
        "past_vmean": past["past_vmean"].to_numpy().astype(float),
        "past_vlast": past["past_vlast"].to_numpy().astype(float),
    }


def _full_history_features(cfg, cust):
    """Schema-derived, EXHAUSTIVE full-history features — the ORACLE's input.

    GENERIC, not hand-picked: discover EVERY field in the canonical stream and
    aggregate it by its natural type — every categorical field -> distinct-count;
    every numeric field -> count/sum/mean/std/min/max; plus generic timing
    (span, recency, timestamp spread). This is the empirical measure of the full
    history (bounded only by the fields the stream carries), so an embedding
    cannot look sufficient merely by reproducing a hand-picked set like RFM.
    Returns (X:(N,D) float32, names) aligned to `cust` order."""
    import duckdb
    import polars as pl

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    keys = [str(c) for c in list(cust)]
    if not keys:
        return np.zeros((0, 1), dtype=np.float32), []
    con.register("_cust", pl.DataFrame({"customer_key": keys}))
    con.execute(
        f"CREATE TEMP TABLE base AS SELECT e.customer_key AS ck, "
        f"epoch(CAST(e.event_ts AS TIMESTAMPTZ)) AS t, e.event_type AS et, "
        f"e.event_attributes AS attrs FROM {src} e "
        f"JOIN _cust c ON e.customer_key = c.customer_key WHERE "
        f"epoch(CAST(e.event_ts AS TIMESTAMPTZ)) <= {tc}"
    )
    jkeys = [
        str(r[0])
        for r in con.execute(
            "SELECT DISTINCT unnest(json_keys(attrs)) FROM base WHERE attrs IS NOT NULL"
        ).fetchall()
    ]
    numeric = []
    for k in jkeys:
        esc = k.replace("'", "''")
        cast = f"TRY_CAST(json_extract_string(attrs,'$.{esc}') AS DOUBLE)"
        nn, tot = con.execute(
            f"SELECT count(*) FILTER (WHERE {cast} IS NOT NULL), "
            f"count(json_extract_string(attrs,'$.{esc}')) FROM base"
        ).fetchone()
        if tot and nn / tot > 0.9:
            numeric.append(k)
    aggs = [
        "count(*) AS n_events",
        "count(DISTINCT et) AS n_etype",
        "max(t) - min(t) AS tspan",
        "stddev(t) AS tspread",
        "max(t) AS tlast",
    ]
    cols = ["n_events", "n_etype", "tspan", "tspread", "tlast"]
    for k in jkeys:
        esc = k.replace("'", "''")
        safe = "".join(ch if ch.isalnum() else "_" for ch in k)
        aggs.append(f"count(DISTINCT json_extract_string(attrs,'$.{esc}')) AS cat_{safe}_nuniq")
        cols.append(f"cat_{safe}_nuniq")
    for k in numeric:
        esc = k.replace("'", "''")
        safe = "".join(ch if ch.isalnum() else "_" for ch in k)
        c = f"TRY_CAST(json_extract_string(attrs,'$.{esc}') AS DOUBLE)"
        for fn, s in (
            ("cnt", f"count({c})"),
            ("sum", f"sum({c})"),
            ("mean", f"avg({c})"),
            ("std", f"stddev({c})"),
            ("min", f"min({c})"),
            ("max", f"max({c})"),
        ):
            aggs.append(f"{s} AS num_{safe}_{fn}")
            cols.append(f"num_{safe}_{fn}")
    df = con.execute(f"SELECT ck, {', '.join(aggs)} FROM base GROUP BY ck").pl()
    con.close()
    full = pl.DataFrame({"ck": keys}).join(df, on="ck", how="left").fill_null(0.0)
    full = full.with_columns((pl.lit(tc) - pl.col("tlast")).alias("recency"))
    cols = cols + ["recency"]
    X = np.column_stack([full[c].to_numpy().astype(np.float32) for c in cols])
    return X.astype(np.float32), cols


def _future_family(j):
    """Self-supervised future targets (derived from the stream; no domain literal).

    All are functions of the FUTURE observables, so they are task-agnostic probes
    of predictive sufficiency (the exact opposite of our known business tasks).
    """
    orders = j["orders"].to_numpy().astype(float)
    return {
        "LTV (future log-margin)": (j["ltv"].to_numpy().astype(float), "reg"),
        "order_count (horizon)": (orders, "reg"),
        "propensity (order in horizon)": ((orders > 0).astype(float), "clf"),
        "churn (no order in horizon)": ((orders == 0).astype(float), "clf"),
        "next-order value": (j["next_v"].to_numpy().astype(float), "reg"),
    }


def _score(kind, y, s):
    from sklearn.metrics import r2_score, roc_auc_score

    if kind == "clf":
        if len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, s))
    return float(r2_score(y, s))


def _normalized_gap(donor, e_alone, kind, eps=1e-9):
    """(gap, normalized) where gap = donor - E = what history adds beyond E.

    Range = the oracle/donor's own achievable spread (reg) or above-chance spread
    (clf), so the gap is scale-free and dimensionless."""
    gap = donor - e_alone
    denom = donor if kind == "reg" else (donor - 0.5)
    norm = 0.0 if denom <= eps else float(np.clip(gap / denom, 0.0, 1.0))
    return float(gap), norm


def _oracle_gap(Z, Xhist, yy, kind, cust, seeds=(0, 1, 2), eps=1e-9):
    """Mean oracle-gap over held-out folds for one target.

    E-alone = model on E; history-alone = model on the history proxy; oracle
    [E,history] = model on both. gap = oracle - E-alone (what history adds beyond
    E), normalized to the oracle's own range. Returns (e_alone, hist_alone,
    oracle, gap, norm) means, or None if no usable fold.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    Cls = HistGradientBoostingClassifier if kind == "clf" else HistGradientBoostingRegressor
    Conf = GBM_CLF if kind == "clf" else GBM_REG
    e_al, raw_al, don_al, gaps, norms, nvs = [], [], [], [], [], []
    for seed in seeds:
        fold = np.zeros(len(cust), dtype=int)
        for fi, f in enumerate(_folds(cust, seed=seed)):
            fold[np.isin(cust, f)] = fi
        for fi in range(5):
            tr, va = fold != fi, fold == fi
            if (
                not tr.any()
                or not va.any()
                or (kind == "clf" and (len(set(yy[tr])) < 2 or len(set(yy[va])) < 2))
            ):
                continue
            scZ = StandardScaler().fit(Z[tr])
            Ztr, Zva = scZ.transform(Z[tr]), scZ.transform(Z[va])
            scH = StandardScaler().fit(Xhist[tr])
            Htr, Hva = scH.transform(Xhist[tr]), scH.transform(Xhist[va])
            Etra, Etva = np.hstack([Ztr, Htr]), np.hstack([Zva, Hva])  # oracle = [E, history]
            hE = Cls(**Conf).fit(Ztr, yy[tr])
            hH = Cls(**Conf).fit(Htr, yy[tr])
            hEH = Cls(**Conf).fit(Etra, yy[tr])
            if kind == "clf":
                se = hE.predict_proba(Zva)[:, 1]
                sh = hH.predict_proba(Hva)[:, 1]
                shh = hEH.predict_proba(Etva)[:, 1]
            else:
                se = hE.predict(Zva)
                sh = hH.predict(Hva)
                shh = hEH.predict(Etva)
            ya = yy[va]
            onE, onH, onEH = _score(kind, ya, se), _score(kind, ya, sh), _score(kind, ya, shh)
            if onE is None or onH is None or onEH is None:
                continue
            g, nrm = _normalized_gap(onEH, onE, kind, eps=eps)
            gaps.append(g)
            norms.append(nrm)
            e_al.append(onE)
            raw_al.append(onH)
            don_al.append(onEH)
            nvs.append(int(va.sum()))
    if not norms:
        return None
    # NOISE FLOOR (derived, not a magic number): the oracle's fold-to-fold SE,
    # floored by the finite-sample rate 1/sqrt(n_val). "Solved" for this member
    # = E is within this noise of the full-history oracle (gap indistinguishable
    # from 0), i.e. the embedding is as good as the raw history can make it.
    nval = max(int(np.mean(nvs)), 2)
    noise = max(float(np.std(don_al)), float(np.std(e_al)), 1.0 / np.sqrt(nval))
    gap_mean = float(np.mean(gaps))
    solved = abs(gap_mean) <= noise
    return (
        float(np.mean(e_al)),
        float(np.mean(raw_al)),
        float(np.mean(don_al)),
        gap_mean,
        float(np.mean(norms)),
        noise,
        bool(solved),
    )


def synthetic_machinery_check(seed=0, n=4000, kind="reg", seeds=(0, 1, 2)):
    """Ground-truth verification of the oracle-gap / U machinery.

    A KNOWN sufficient statistic S* = [direction S, log-magnitude], a future
    target that IS a function of both, and a history proxy that carries the
    magnitude. Then, by construction:
      * E = S*            (sufficient)      -> history adds nothing  -> U ~ 1
      * E = direction only (lossy)           -> history adds magnitude -> U < 1
      * E = [S*, noise]   (over-capacity)    -> no improvement       -> U ~ U(sufficient)
    Verifies the metric separates lossy from sufficient and SATURATES on extra
    capacity (no false reward for more dims)."""
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, 3))
    logmag = rng.normal(0.0, 1.0, size=n)  # the extensive axis (unbounded, additive)
    direction = S / np.linalg.norm(S, axis=1, keepdims=True)
    cust = np.array([f"c{i:07d}" for i in range(n)])
    # target depends on the DIRECTION (in E) AND the log-magnitude (in history):
    y = 2.0 * direction[:, 0] + 1.5 * logmag + rng.normal(0.0, 0.3, size=n)
    Xhist = np.column_stack([logmag, rng.normal(size=(n, 4))])  # history proxy carries magnitude
    reps = {
        "sufficient [S, logmag]": np.column_stack([S, logmag]),
        "lossy [direction]": direction,
        "over [S, logmag, noise]": np.column_stack([S, logmag, rng.normal(size=(n, 8))]),
    }
    out = {}
    for name, Z in reps.items():
        r = _oracle_gap(Z, Xhist, y, kind, cust, seeds=seeds)
        if r is not None:
            out[name] = {"norm_gap": round(r[4], 4), "gap": round(r[3], 4)}
    return out


def run_universality(model, cfg, horizons=(30, 90), seeds=(0, 1, 2)):
    import polars as pl

    cust, Z = _b_states(model, cfg)
    # ORACLE input: the schema-derived, exhaustive full-history measure (not RFM).
    Xhist, _hnames = _full_history_features(cfg, cust)
    rows = []
    for h in horizons:
        fut, _past, _tc = _rfm_targets(cfg, h)
        j = (
            pl.DataFrame({"customer": cust})
            .join(fut, left_on="customer", right_on="customer_key", how="left")
            .fill_null(0.0)
        )
        for name, (yy, kind) in _future_family(j).items():
            r = _oracle_gap(Z, Xhist, yy, kind, cust, seeds=seeds)
            if r is None:
                continue
            e_al, raw_al, don_al, gap, norm, noise, solved = r
            rows.append(
                {
                    "target": f"{name} @ {h}d",
                    "E_alone": round(e_al, 4),
                    "history_alone": round(raw_al, 4),
                    "oracle [E,hist]": round(don_al, 4),
                    "gap (hist adds)": round(gap, 4),
                    "norm_gap": round(norm, 4),
                    "noise": round(noise, 4),
                    "solved": solved,
                }
            )
    u = 1.0 - max((r["norm_gap"] for r in rows), default=0.0)
    solved = bool(rows) and all(r["solved"] for r in rows)
    return rows, u, len(cust), solved


def _card(tag, rows, u, n, solved):
    worst = max(rows, key=lambda r: r["norm_gap"]) if rows else None
    lines = [
        f"# UNIVERSALITY GATE — {tag}  [loop/UNIVERSALITY.md]",
        f"{n} HELD-OUT (B) customers · oracle = [E, exhaustive full-history measure] · worst-case (minimax)",
        "",
        f"## U (universality index) = {u:.4f}   (1.0 = universal)",
        f"## SOLVED = {solved}   (E within the oracle's own noise on EVERY member; gap <= noise)",
    ]
    if worst is not None:
        lines.append(
            f"worst-case member: `{worst['target']}`  norm_gap {worst['norm_gap']:.4f}  "
            f"gap {worst['gap (hist adds)']:+.4f}  noise {worst['noise']:.4f}  "
            f"solved={worst['solved']}"
        )
    lines += [
        "",
        "| target | E alone | history alone | oracle [E,hist] | gap (hist adds) | noise | "
        "norm_gap | solved |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['E_alone']:.4f} | {r['history_alone']:.4f} | "
            f"{r['oracle [E,hist]']:.4f} | {r['gap (hist adds)']:+.4f} | {r['noise']:.4f} | "
            f"{r['norm_gap']:.4f} | {'✓' if r['solved'] else '✗'} |"
        )
    lines.append("")
    lines.append(
        "> gap = score([E,history]) - score(E): what the EXHAUSTIVE full-history measure adds "
        "beyond E. SOLVED for a member when |gap| <= noise (the oracle's fold SE, floored by "
        "1/sqrt(n)); SOLVED overall when every member is. NOT hand-RFM."
    )
    return "\n".join(lines) + "\n", u, solved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=False)
    ap.add_argument("--as-of", required=False, dest="as_of")
    ap.add_argument("--ckpt-dir", required=False, dest="ckpt_dir")
    ap.add_argument("--tag", required=False)
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--horizons", default="30,90")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--no-accumulator",
        action="store_true",
        dest="no_acc",
        help="disable the cumulative-accumulator measure pathway (matched control for U)",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="ground-truth machinery check (known sufficient statistic; no checkpoint)",
    )
    ap.add_argument(
        "--require-solved",
        action="store_true",
        dest="require_solved",
        help="exit non-zero unless SOLVED (E within oracle noise on every member) — loop predicate",
    )
    a = ap.parse_args(argv)
    if a.synthetic:
        res = synthetic_machinery_check()
        print("# UNIVERSALITY MACHINERY CHECK (ground truth: known sufficient statistic)")
        for name, r in res.items():
            print(f"  {name:28s} norm_gap {r['norm_gap']:.4f}  (gap {r['gap']:+.4f})")
        return res
    if not (a.db and a.as_of and a.ckpt_dir and a.tag):
        ap.error("--db/--as-of/--ckpt-dir/--tag are required unless --synthetic")
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    if a.no_acc:
        model.accumulator_on = False  # matched control: shape-only (no measure pathway)
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    horizons = tuple(int(x) for x in a.horizons.split(","))
    seeds = tuple(int(x) for x in a.seeds.split(","))
    rows, u, n, solved = run_universality(model, cfg, horizons, seeds)
    card, u, solved = _card(a.tag, rows, u, n, solved)
    print(card)
    if a.out:
        Path(a.out).write_text(card)
    if a.require_solved and not solved:
        raise SystemExit(1)
    return {"U": u, "solved": solved, "rows": rows}


if __name__ == "__main__":
    main()
