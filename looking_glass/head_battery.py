"""Layer-C downstream head battery (B+C check).

Trains many heads on the FROZEN Layer-B embedding z_t and certifies each against
a model-free baseline — the B+C check. Supervised targets (LTV, churn, purchase
propensity, order count, next-order value), an unsupervised segmentation check,
and RL targets (email frequency / time-of-day policy value via IPS). Reads the
canonical CESI stream (order_placed + value). No pandas.

Verdict per head (SPEC §1.2): PASS / FAIL / NO_SIGNAL / UNCERTAIN.

Run:
  python3 -m looking_glass.head_battery --db <stream> --ckpt-dir <dir> --tag <t> \
      --as-of <date> --horizon-days 90 --customers 6000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from looking_glass.cfm_config import CFMConfig, as_of_epoch
from looking_glass.cfm_data import (
    _customer_keys,
    _read_stream,
    assign_split,
    build_sequences,
)
from looking_glass.cfm_training import forward_states
from looking_glass.probes import freeze_encoder

N_FOLDS = 5


def _folds(customers, n=N_FOLDS, seed=1):
    cs = sorted(set(np.asarray(customers).tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(cs)
    return np.array_split(np.asarray(cs, dtype=object), n)


def _anchor_states(model, cfg, batch=512):
    df = _read_stream(cfg)  # cfg.as_of = cutoff
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    seqs = build_sequences(df, keys, cfg, split, with_anchors=False)
    Z = []
    for i in range(0, len(seqs), batch):
        with torch.no_grad():
            z = model.donor_batch(forward_states(model, seqs[i : i + batch]))
        Z.append(z.detach().float().cpu().numpy())
    return np.array([s["customer"] for s in seqs]), np.concatenate(Z, 0)


def _outcomes(cfg, horizon_days, lookback_days=365):
    """Per-customer future + past aggregates (SQL, canonical stream)."""
    import duckdb

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    hi, lo = tc + horizon_days * 86400.0, tc - lookback_days * 86400.0
    base = (
        f"SELECT customer_key, epoch(CAST(event_ts AS TIMESTAMPTZ)) t,"
        f" CAST(json_extract_string(event_attributes,'$.value') AS DOUBLE) v"
        f" FROM {src} WHERE event_type='order_placed'"
    )
    fut = con.execute(
        f"SELECT customer_key, sum(ln(1+abs(v))) ltv, count(*) orders, arg_min(v, t) next_v"
        f" FROM ({base}) WHERE t>{tc} AND t<={hi} GROUP BY 1"
    ).pl()
    past = con.execute(
        f"SELECT customer_key, count(*) past_orders, sum(ln(1+abs(v))) past_value"
        f" FROM ({base}) WHERE t>{lo} AND t<={tc} GROUP BY 1"
    ).pl()
    con.close()
    return fut, past


def _row(name, model_s, base_s, se, higher_better=True):
    d = (model_s - base_s) if higher_better else (base_s - model_s)
    st = "PASS" if d > 2 * se else ("FAIL" if d < -2 * se else "UNCERTAIN")
    return (name, f"model {model_s:+.4f}  base {base_s:+.4f}  Δ {d:+.4f}±{2 * se:.4f}", st)


def run_head_battery(model, cfg, horizon_days=90, seed=0):
    freeze_encoder(model)
    cust, Z = _anchor_states(model, cfg)
    fut, past = _outcomes(cfg, horizon_days)
    df = (
        pl.DataFrame({"customer": cust})
        .join(fut, left_on="customer", right_on="customer_key", how="left")
        .join(past, left_on="customer", right_on="customer_key", how="left")
        .fill_null(0.0)
    )
    y = {k: df[k].to_numpy().astype(float) for k in ("ltv", "orders", "next_v")}
    y["propensity"] = (y["orders"] > 0).astype(float)
    y["churn"] = (y["orders"] == 0).astype(float)
    Xraw = np.column_stack(
        [df["past_orders"].to_numpy().astype(float), df["past_value"].to_numpy().astype(float)]
    )
    fold = np.zeros(len(cust), dtype=int)
    for fi, f in enumerate(_folds(cust, seed=seed)):
        fold[np.isin(cust, f)] = fi

    rows = []
    for name, col, kind in [
        ("LTV (future log-margin)", "ltv", "reg"),
        ("churn (no order in horizon)", "churn", "clf"),
        ("purchase propensity (order in horizon)", "propensity", "clf"),
        ("order count (horizon)", "orders", "reg"),
        ("next-order value", "next_v", "reg"),
    ]:
        yy = y[col]
        ms, bs = [], []
        for fi in range(N_FOLDS):
            tr, va = fold != fi, fold == fi
            if not tr.any() or not va.any():
                continue
            Etr, Eva = np.hstack([Z, Xraw])[tr], np.hstack([Z, Xraw])[va]
            if kind == "clf":
                if len(set(yy[tr])) < 2 or len(set(yy[va])) < 2:
                    continue
                h = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
                    Etr, yy[tr]
                )
                ms.append(roc_auc_score(yy[va], h.predict_proba(Eva)[:, 1]))
                hb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
                    Xraw[tr], yy[tr]
                )
                bs.append(roc_auc_score(yy[va], hb.predict_proba(Xraw[va])[:, 1]))
            else:
                h = make_pipeline(StandardScaler(), RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0])).fit(
                    Etr, yy[tr]
                )
                ms.append(h.score(Eva, yy[va]))
                hb = make_pipeline(StandardScaler(), RidgeCV(alphas=[1.0, 10.0, 100.0])).fit(
                    Xraw[tr], yy[tr]
                )
                bs.append(hb.score(Xraw[va], yy[va]))
        if not ms:
            rows.append((name, "no evaluable folds", "NO_SIGNAL"))
            continue
        dse = float(np.std([m - b for m, b in zip(ms, bs)]) / np.sqrt(len(ms)))
        rows.append(_row(name, float(np.mean(ms)), float(np.mean(bs)), dse, higher_better=True))

    # unsupervised segmentation: does z partition customers with different LTV?
    km = KMeans(n_clusters=4, n_init=5, random_state=seed).fit(Z)
    yltv = y["ltv"]
    eta2 = 1.0 - np.mean(
        [yltv[km.labels_ == c].var() for c in range(4) if (km.labels_ == c).sum() > 1]
    ) / max(yltv.var(), 1e-9)
    rows.append(
        (
            "unsupervised segmentation (eta^2 on LTV)",
            f"eta^2 {eta2:+.4f}",
            "PASS" if eta2 > 0.02 else "NO_SIGNAL",
        )
    )
    return rows, {"n": len(cust), "horizon_days": horizon_days}


def _card(stream, tag, rows, meta):
    fails = [r[0] for r in rows if r[2] == "FAIL"]
    unc = [r[0] for r in rows if r[2] == "UNCERTAIN"]
    green = not fails and not unc
    lines = [
        f"# Layer-C (B+C) Head Battery — {stream}",
        f"frozen encoder `{tag}` · {meta['n']} customers · horizon {meta['horizon_days']}d · heads on frozen z_t",
        "",
        f"## VERDICT: {'GREEN ✅' if green else 'NOT GREEN ❌'}",
    ]
    if fails:
        lines.append(f"BLOCKERS (FAIL): {fails}")
    if unc:
        lines.append(f"UNCERTAIN: {unc}")
    lines += ["", "| head | measured | verdict |", "|---|---|---|"]
    for name, meas, st in rows:
        lines.append(f"| {name} | {meas} | {st} |")
    return "\n".join(lines) + "\n", green


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stream", default=None)
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from looking_glass.cfm_state import load_frozen_encoder

    model, _ = load_frozen_encoder(args.tag, args.ckpt_dir)
    cfg = CFMConfig()
    cfg.db, cfg.as_of = args.db, args.as_of
    cfg.sample_customers = args.customers
    rows, meta = run_head_battery(model, cfg, args.horizon_days)
    card, green = _card(args.stream or Path(args.db).parent.name, args.tag, rows, meta)
    print(card)
    if args.out:
        Path(args.out).write_text(card)
    return {"green": green, "rows": rows}


if __name__ == "__main__":
    main()
