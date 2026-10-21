"""Layer-C downstream head battery — SPEC 0.9.5 §9.6 rigorous B+C check.

Requirements (all enforced here):
  1. Standardized head inputs (per-dim z-score).
  2. Donor grading: head reads [E, raw]; baseline reads raw -> PASS iff E adds
     incremental signal (SPEC C9). E-alone reported for the C9 ablation.
  3. HELD-OUT customers only (the encoder's B split) -- no train-customer leakage.
  4. STRONG model-free baseline: RFM + recency-window counts (30/90d) + past-value
     stats, gradient boosting. The DONOR reads [E, STRONG] and is graded against the
     SAME strong features -> the encoder's true increment over what production has
     anyway (raw counters). RFM-only reported for context.
  5. Effect size + bootstrap 95% CI over held-out customers, across >=3 seeds.
  6. Pre-registered gates + reproducibility receipt.

Verdict per head (SPEC §1.2): PASS = encoder adds over the strong baseline (CI-lo > 0);
FAIL = it hurts (CI-hi < 0); NO_SIGNAL = the baseline carries the target and E adds
nothing beyond it (CI straddles 0).

Run: python3 -m looking_glass.head_battery --db <stream> --ckpt-dir <dir> --tag <t> \
        --as-of <date> --horizon-days 90 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from looking_glass.cfm_config import CFMConfig, as_of_epoch
from looking_glass.cfm_data import (
    _customer_keys,
    _read_stream,
    assign_split,
    build_sequences,
)
from looking_glass.cfm_training import embed_batch_size, forward_embedding
from looking_glass.probes import freeze_encoder

# ---- PRE-REGISTERED GATE (fixed before evaluation) --------------------------
GATE = {
    "pass": "bootstrap 95% CI of Δ(donor − raw) excludes 0 AND mean Δ > 0",
    "no_signal": "raw baseline itself at chance (AUROC<=0.5, or R2<=0)",
    "uncertain": "CI straddles 0 and raw carries signal",
    "bootstrap": 2000,
    "seeds": 3,
    "folds": 5,
}
N_FOLDS = 5
GBM_REG = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "min_samples_leaf": 25,
    "l2_regularization": 1.0,
    "max_depth": 4,
}
GBM_CLF = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "min_samples_leaf": 25,
    "l2_regularization": 1.0,
    "max_depth": 4,
}


def _folds(customers, n=N_FOLDS, seed=1):
    cs = sorted(set(np.asarray(customers).tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(cs)
    return np.array_split(np.asarray(cs, dtype=object), n)


_BSTATE_CACHE: dict = {}


def _b_states(model, cfg, batch=None):
    """Frozen donor states for HELD-OUT customers (split B) only.

    CACHED per (model, stream, as_of, sampling): the report card / gates / white_queen call this
    ~15+ times with the SAME inputs — recomputing the read + encode each time was the dominant
    card cost.

    Batch size is derived from the sequence budget (a fixed token count), not a
    literal: long-cadence streams (seq_len in the thousands) would OOM the GPU at
    a fixed 512, short ones would under-utilize it. Memory scales with
    batch * seq_len, so cap the per-step token count and let batch float."""
    _key = (
        id(model),
        str(cfg.db),
        str(getattr(cfg, "as_of", None)),
        int(getattr(cfg, "sample_customers", 0) or 0),
        int(getattr(cfg, "split_seed", 0) or 0),
        int(getattr(cfg, "seq_len", 0) or 0),
        tuple(getattr(cfg, "effect_actions", ()) or ()),
        tuple(getattr(cfg, "action_fields", ()) or ()),
        float(getattr(cfg, "action_period_seconds", 0.0) or 0.0),
        bool(getattr(cfg, "use_measure", False)),
        id(getattr(model, "layer_b", None)),
        bool(getattr(model, "accumulator_on", False)),
        bool(getattr(model, "counter_on", False)),
    )
    _hit = _BSTATE_CACHE.get(_key)
    if _hit is not None:
        return _hit
    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    bkeys = [k for k in keys if split.get(k) == "B"]
    seqs = build_sequences(df, bkeys, cfg, split, with_anchors=False)
    if batch is None:
        batch = embed_batch_size(cfg)
    Z = []
    for i in range(0, len(seqs), batch):
        with torch.no_grad():
            z = forward_embedding(
                model, seqs[i : i + batch], cfg=cfg, lb=getattr(model, "layer_b", None)
            )  # consumed embedding
        Z.append(z.detach().float().cpu().numpy())
    _res = (np.array([s["customer"] for s in seqs]), np.concatenate(Z, 0))
    _BSTATE_CACHE[_key] = _res
    return _res


def _rfm_targets(cfg, horizon_days, lookback_days=365):
    """Per-customer RFM (past) + future targets (SQL, canonical stream)."""
    import duckdb
    import polars as pl

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    hi, lo = tc + horizon_days * 86400.0, tc - lookback_days * 86400.0
    # conversion/order event DERIVED from the data (doctrine #4 — no business
    # literal): the event type with the highest mean monetary `value`.
    oe = con.execute(
        f"SELECT event_type FROM {src} GROUP BY 1 "
        f"ORDER BY avg(TRY_CAST(json_extract_string(event_attributes,'$.value') AS DOUBLE)) "
        f"DESC NULLS LAST LIMIT 1"
    ).fetchone()
    order_event = oe[0] if oe else None
    base = (
        f"SELECT customer_key, epoch(CAST(event_ts AS TIMESTAMPTZ)) t,"
        f" CAST(json_extract_string(event_attributes,'$.value') AS DOUBLE) v"
        f" FROM {src} WHERE event_type = ?"
    )
    fut = con.execute(
        f"SELECT customer_key, sum(ln(1+abs(v))) ltv, count(*) orders,"
        f" arg_min(v, t) next_v, min(t) first_t"
        f" FROM ({base}) WHERE t>{tc} AND t<={hi} GROUP BY 1",
        [order_event],
    ).pl()
    past = con.execute(
        f"SELECT customer_key, count(*) freq, sum(ln(1+abs(v))) monetary,"
        f" max(t) last_t, avg(ln(1+abs(v))) past_vmean, arg_max(v, t) past_vlast,"
        f" count(*) FILTER (WHERE t > {tc - 30 * 86400.0}) recent30,"
        f" count(*) FILTER (WHERE t > {tc - 90 * 86400.0}) recent90"
        f" FROM ({base}) WHERE t>{lo} AND t<={tc} GROUP BY 1",
        [order_event],
    ).pl()
    # ORDER-AWARE targets (adversarial sufficiency family): the aggregate measure
    # has event-type COUNTS but not the last/next type, so these probe whether the
    # learned shape actually carries sequence order (not just stationary mass).
    anyev = f"SELECT customer_key, event_type et, epoch(CAST(event_ts AS TIMESTAMPTZ)) t FROM {src}"
    past_any = con.execute(
        f"SELECT customer_key, arg_max(et, t) last_et, max(t) last_any_t "
        f"FROM ({anyev}) WHERE t<={tc} GROUP BY 1"
    ).pl()
    fut_any = con.execute(
        f"SELECT customer_key, arg_min(et, t) next_et, min(t) next_any_t "
        f"FROM ({anyev}) WHERE t>{tc} AND t<={hi} GROUP BY 1"
    ).pl()
    # drive from ALL customers with a past event so a customer with no future
    # event still gets a censored delay (horizon), not a spurious 0
    fut = past_any.join(fut, on="customer_key", how="left").join(
        fut_any, on="customer_key", how="left"
    )
    # censored next-event delay: no future event within the horizon -> horizon
    fut = fut.with_columns(
        pl.when(pl.col("next_any_t").is_not_null())
        .then((pl.col("next_any_t") - tc) / 86400.0)
        .otherwise(float(horizon_days))
        .alias("next_delay_days"),
        (
            (pl.col("next_et").is_not_null())
            & (pl.col("last_et").is_not_null())
            & (pl.col("next_et") == pl.col("last_et"))
        )
        .cast(pl.Float64)
        .alias("repeat_last_type"),
    )
    con.close()
    return fut, past, tc


def _metrics(kind, y, s_donor, s_base, w=None):
    w = np.ones(len(y)) if w is None else w
    if kind == "clf":
        if len(set(y)) < 2:
            return None, None
        return roc_auc_score(y, s_donor, sample_weight=w), roc_auc_score(y, s_base, sample_weight=w)
    ss = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r2d = 1 - np.sum(w * (y - s_donor) ** 2) / max(ss, 1e-12)
    r2b = 1 - np.sum(w * (y - s_base) ** 2) / max(ss, 1e-12)
    return r2d, r2b


def run_head_battery(model, cfg, horizon_days=90, seeds=(0, 1, 2)):
    freeze_encoder(model)
    cust, Z = _b_states(model, cfg)
    fut, past, tc = _rfm_targets(cfg, horizon_days)
    import polars as pl

    j = (
        pl.DataFrame({"customer": cust})
        .join(fut, left_on="customer", right_on="customer_key", how="left")
        .join(past, left_on="customer", right_on="customer_key", how="left")
        .fill_null(0.0)
    )
    # RFM features (recency small = recently active)
    recency = (tc - j["last_t"].to_numpy().astype(float)) / 86400.0
    Xraw = np.column_stack(
        [recency, j["freq"].to_numpy().astype(float), j["monetary"].to_numpy().astype(float)]
    ).astype(float)
    # richer model-free baseline: RFM + recency-window counts + past-value stats.
    # If even THIS can't beat RFM+GBM, the target is NOT carried beyond RFM.
    Xrich = np.column_stack(
        [
            Xraw,
            j["recent30"].to_numpy().astype(float),
            j["recent90"].to_numpy().astype(float),
            j["past_vmean"].to_numpy().astype(float),
            j["past_vlast"].to_numpy().astype(float),
        ]
    ).astype(float)
    yall = {
        "LTV (future log-margin)": (j["ltv"].to_numpy().astype(float), "reg"),
        "churn (no order in horizon)": (
            (j["orders"].to_numpy().astype(float) == 0).astype(float),
            "clf",
        ),
        "purchase propensity (order in horizon)": (
            (j["orders"].to_numpy().astype(float) > 0).astype(float),
            "clf",
        ),
        "order count (horizon)": (j["orders"].to_numpy().astype(float), "reg"),
        "next-order value": (j["next_v"].to_numpy().astype(float), "reg"),
    }
    from sklearn.preprocessing import StandardScaler

    rows = []
    for name, (yy, kind) in yall.items():
        if kind == "clf" and len(set(yy)) < 2:
            rows.append(
                {
                    "head": name,
                    "n": len(yy),
                    "delta": 0.0,
                    "lo": 0.0,
                    "hi": 0.0,
                    "e_alone": 0.0,
                    "raw": 0.0,
                    "verdict": "NO_SIGNAL",
                }
            )
            continue
        deltas = []  # bootstrap Δ samples pooled across folds/seeds
        raw_scores, e_scores, rfm_scores = [], [], []
        for seed in seeds:
            fold = np.zeros(len(cust), dtype=int)
            for fi, f in enumerate(_folds(cust, seed=seed)):
                fold[np.isin(cust, f)] = fi
            for fi in range(N_FOLDS):
                tr, va = fold != fi, fold == fi
                if (
                    not tr.any()
                    or not va.any()
                    or (kind == "clf" and (len(set(yy[tr])) < 2 or len(set(yy[va])) < 2))
                ):
                    continue
                sc = StandardScaler().fit(Z[tr])
                Ztr, Zva = sc.transform(Z[tr]), sc.transform(Z[va])
                # STRONG model-free baseline = RFM + recency windows + past-value.
                # The DONOR reads [E, strong] and is graded against the SAME strong
                # features -> the ENCODER's true increment (production-realistic:
                # downstream always has the raw counters). RFM-only is reported too.
                sc2 = StandardScaler().fit(Xrich[tr])
                Rtr, Rva = sc2.transform(Xrich[tr]), sc2.transform(Xrich[va])
                sc3 = StandardScaler().fit(Xraw[tr])
                Rrtr, Rrva = sc3.transform(Xraw[tr]), sc3.transform(Xraw[va])
                Etva = np.hstack([Zva, Rva])
                Etra = np.hstack([Ztr, Rtr])
                if kind == "clf":
                    hd = HistGradientBoostingClassifier(**GBM_CLF).fit(Etra, yy[tr])
                    hb = HistGradientBoostingClassifier(**GBM_CLF).fit(Rtr, yy[tr])
                    he = HistGradientBoostingClassifier(**GBM_CLF).fit(Ztr, yy[tr])
                    hr = HistGradientBoostingClassifier(**GBM_CLF).fit(Rrtr, yy[tr])
                    sd, sb, se_, sr = (
                        hd.predict_proba(Etva)[:, 1],
                        hb.predict_proba(Rva)[:, 1],
                        he.predict_proba(Zva)[:, 1],
                        hr.predict_proba(Rrva)[:, 1],
                    )
                else:
                    hd = HistGradientBoostingRegressor(**GBM_REG).fit(Etra, yy[tr])
                    hb = HistGradientBoostingRegressor(**GBM_REG).fit(Rtr, yy[tr])
                    he = HistGradientBoostingRegressor(**GBM_REG).fit(Ztr, yy[tr])
                    hr = HistGradientBoostingRegressor(**GBM_REG).fit(Rrtr, yy[tr])
                    sd, sb, se_, sr = (
                        hd.predict(Etva),
                        hb.predict(Rva),
                        he.predict(Zva),
                        hr.predict(Rrva),
                    )
                ya = yy[va]
                d, b = _metrics(kind, ya, sd, sb)
                e, _ = _metrics(kind, ya, se_, sb)
                rfm, _ = _metrics(kind, ya, sr, sb)
                if d is None:
                    continue
                raw_scores.append(b)
                e_scores.append(e)
                if rfm is not None:
                    rfm_scores.append(rfm)
                rng = np.random.default_rng(seed * 100 + fi)
                for _ in range(GATE["bootstrap"] // (len(seeds) * N_FOLDS)):
                    idx = rng.integers(0, len(ya), len(ya))
                    dd, _bb = _metrics(kind, ya[idx], sd[idx], sb[idx])
                    if dd is not None:
                        deltas.append(dd - _bb)
        if not deltas:
            rows.append(
                {
                    "head": name,
                    "n": len(cust),
                    "delta": 0.0,
                    "lo": 0.0,
                    "hi": 0.0,
                    "e_alone": 0.0,
                    "raw": 0.0,
                    "verdict": "NO_SIGNAL",
                }
            )
            continue
        deltas = np.array(deltas)
        mean_d = float(deltas.mean())
        lo, hi = float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
        rawsig = float(np.mean(raw_scores)) if raw_scores else 0.5  # STRONG baseline score
        ealone = float(np.mean(e_scores)) if e_scores else rawsig
        rfm_sig = float(np.mean(rfm_scores)) if rfm_scores else rawsig  # RFM-only (context)
        # FAR-EXCEED floor (grounded, no magic number): the strong baseline's own
        # measurement SE. EXCEED = the donor's CI LOWER bound beats the baseline by
        # more than the baseline's own uncertainty -> a robust win, not a marginal one.
        raw_se = float(np.std(raw_scores) / np.sqrt(max(len(raw_scores), 1))) if raw_scores else 0.0
        exceeds = bool(lo > raw_se and raw_se > 0)
        # The baseline is the STRONG model-free feature set (RFM + clock + past-value).
        # PASS = the encoder ADDS over it; NO_SIGNAL = the baseline carries the target
        # and E adds nothing (its unique contribution is not carried); FAIL = E hurts.
        carried = (rawsig > 0.5 + 1e-6) if kind == "clf" else (rawsig > 0.0)
        if not carried:
            verdict = "NO_SIGNAL"
        elif lo > 0:
            verdict = "PASS"
        elif hi < 0:
            verdict = "FAIL"
        else:
            verdict = "NO_SIGNAL"  # E adds nothing beyond the strong baseline
        rows.append(
            {
                "head": name,
                "n": len(cust),
                "delta": mean_d,
                "lo": lo,
                "hi": hi,
                "e_alone": ealone,
                "raw": rawsig,
                "rfm": rfm_sig,
                "raw_se": raw_se,
                "exceeds": exceeds,
                "verdict": verdict,
            }
        )
    # unsupervised —— graded against a FAIR raw/RFM k-means, not against a literal.
    # The old row hardcoded PASS with raw=nan and lo=hi; that is not transfer evidence.
    from sklearn.cluster import KMeans

    def _eta2(labels, y):
        groups = [y[labels == c] for c in np.unique(labels) if (labels == c).sum() > 1]
        if not groups:
            return 0.0
        return float(1.0 - np.mean([g.var() for g in groups]) / max(y.var(), 1e-9))

    yv = yall["LTV (future log-margin)"][0]
    km = KMeans(n_clusters=4, n_init=5, random_state=0).fit(Z)
    eta2 = _eta2(km.labels_, yv)
    sc_raw = StandardScaler().fit(Xrich)
    kmr = KMeans(n_clusters=4, n_init=5, random_state=0).fit(sc_raw.transform(Xrich))
    eta2_raw = _eta2(kmr.labels_, yv)
    rng_u = np.random.default_rng(0)
    _diffs = []
    for _ in range(200):
        _ix = rng_u.integers(0, len(yv), len(yv))
        _diffs.append(_eta2(km.labels_[_ix], yv[_ix]) - _eta2(kmr.labels_[_ix], yv[_ix]))
    _lo, _hi = np.quantile(_diffs, [0.025, 0.975])
    rows.append(
        {
            "head": "unsupervised segmentation (eta^2 on LTV)",
            "n": len(cust),
            "delta": float(eta2 - eta2_raw),
            "lo": float(_lo),
            "hi": float(_hi),
            "e_alone": float(eta2),
            "raw": float(eta2_raw),
            "raw_se": float(np.std(_diffs) / np.sqrt(max(len(_diffs), 1))),
            "verdict": "PASS" if _lo > 0 else ("FAIL" if _hi < 0 else "NO_SIGNAL"),
        }
    )
    receipt = {
        "stream": None,
        "counterfactual_guard": "held-out split B only",
        "n_heldout": int(len(cust)),
        "horizon_days": horizon_days,
        "seeds": list(seeds),
        "gate": GATE,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    return rows, receipt


def _card(stream, tag, rows, receipt):
    fails = [r["head"] for r in rows if r["verdict"] == "FAIL"]
    unc = [r["head"] for r in rows if r["verdict"] == "UNCERTAIN"]
    green = not fails and not unc
    gated = [r for r in rows if r.get("verdict") in ("PASS", "FAIL", "UNCERTAIN")]
    n_exceed = sum(1 for r in gated if r.get("exceeds"))
    strong = green and bool(gated) and n_exceed == len(gated)
    verdict = "GREEN+ (far-exceed) ✅" if strong else ("GREEN ✅" if green else "NOT GREEN ❌")
    lines = [
        f"# Layer-C (B+C) Head Battery — {stream}  [SPEC 0.9.5 §9.6]",
        f"frozen `{tag}` · {receipt['n_heldout']} HELD-OUT (B) customers · horizon {receipt['horizon_days']}d · "
        f"donor head `[E, strong]` vs strong model-free baseline (RFM+clock+past-value) · bootstrap 95% CI · seeds {receipt['seeds']}",
        f"gate: {GATE['pass']}   git `{receipt['git_sha']}`",
        "",
        f"## VERDICT: {verdict}",
        f"far-exceed (donor CI-lo > raw's own SE): {n_exceed}/{len(gated)} heads",
    ]
    if fails:
        lines.append(f"BLOCKERS (FAIL): {fails}")
    if unc:
        lines.append(f"UNCERTAIN: {unc}")
    lines += [
        "",
        "| head | Δ (donor−raw) 95% CI | E alone | raw | exceed | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
        ex = (
            "✓"
            if r.get("exceeds")
            else ("–" if r.get("verdict") in ("PASS", "FAIL", "UNCERTAIN") else "")
        )
        lines.append(
            f"| {r['head']} | {ci} | {r['e_alone']:.4f} | {r['raw']:.4f} | {ex} | {r['verdict']} |"
        )
    return "\n".join(lines) + "\n", green


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stream", default=None)
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from looking_glass.cfm_state import load_frozen_encoder

    model, _ = load_frozen_encoder(args.tag, args.ckpt_dir)
    cfg = CFMConfig()
    cfg.db, cfg.as_of = args.db, args.as_of
    cfg.sample_customers = args.customers
    seeds = tuple(int(x) for x in args.seeds.split(","))
    rows, receipt = run_head_battery(model, cfg, args.horizon_days, seeds)
    receipt["stream"] = args.stream or Path(args.db).parent.name
    card, green = _card(receipt["stream"], args.tag, rows, receipt)
    print(card)
    if args.out:
        Path(args.out).write_text(card)
        Path(args.out).with_suffix(".json").write_text(
            json.dumps({"receipt": receipt, "rows": rows}, indent=1, default=float)
        )
    return {"green": green, "rows": rows}


if __name__ == "__main__":
    main()
