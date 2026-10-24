"""Aggregate-invisibility probe for a relational reference build.

Predicts the future from PREFIX-only features and compares:
  (a) AGGREGATE features   — the PATH A wide-table columns (counts/sums/recency/cat-hist)
  (b) ORDER-AWARE features — aggregates + recent k-gram one-hots, last gap, sequence-shape
Targets: y_value_90 (regression), y_churn_90 (classification), and the next event type after
a cut (classification). Order-aware must beat aggregate substantially, else the reference is
not testing sequence models.

Usage: order_probe.py --dir <build> [--out order_probe.json] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold


def _attrs(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


def _cv(X, y, kind, n_splits=5, seed=0):
    y = np.asarray(y, float)
    oof = np.zeros_like(y)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        if kind == "clf":
            m = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        elif kind == "multi":
            m = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        else:
            m = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
    return oof


def _dummies(df, cols, drop_first=True):
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    return df.to_dummies(columns=present, drop_first=drop_first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.dir)
    man = json.loads((d / "manifest.json").read_text())
    as_of = datetime.fromisoformat(man["as_of"])
    A = pl.read_parquet(d / "path_a/path_a_wide_table.parquet")
    B = (
        pl.read_parquet(d / "path_b/path_b_customer_event_stream.parquet")
        .with_columns(pl.col("event_ts").str.to_datetime().alias("ts"))
        .sort(["customer_key", "ts"])
    )
    cats = [
        c for c in ("segment", "region", "acquisition_channel", "loyalty_tier") if c in A.columns
    ]
    agg_cols = [
        c for c in A.columns if not c.startswith("y_") and c not in ("customer_key", "signup_ts")
    ]
    Xa = _dummies(A.select(agg_cols), cats).to_numpy().astype(float)

    # order-aware sequence features from the prefix (<= as_of)
    pre = B.filter(pl.col("ts") <= as_of)
    p = pre.with_columns(
        pl.col("event_type").shift(1).over("customer_key").alias("prev_et"),
        pl.col("event_type").shift(2).over("customer_key").alias("prev2_et"),
        (pl.col("ts") - pl.col("ts").shift(1).over("customer_key"))
        .dt.total_seconds()
        .alias("gap_s"),
    )
    last = p.group_by("customer_key", maintain_order=True).agg(
        pl.col("event_type").last().alias("last_et"),
        pl.col("prev_et").last().alias("prev_et"),
        pl.col("prev2_et").last().alias("prev2_et"),
        pl.col("event_attributes").last().alias("last_attrs"),
        pl.col("gap_s").last().alias("last_gap_s"),
        pl.col("gap_s").mean().alias("mean_gap_s"),
        pl.col("gap_s").std().alias("std_gap_s"),
        pl.len().alias("seq_len"),
    )
    at = [_attrs(s) for s in last["last_attrs"].to_list()]
    last = last.with_columns(
        pl.Series("last_category", [x.get("category_code") for x in at]),
        pl.Series("last_touch", [x.get("touch") or x.get("channel") for x in at]),
    ).drop("last_attrs")
    seq_cat = ["last_et", "prev_et", "prev2_et", "last_category", "last_touch"]
    S = A.select(["customer_key"] + agg_cols).join(last, on="customer_key", how="left")
    Xo = (
        _dummies(S.select([c for c in S.columns if c != "customer_key"]), cats + seq_cat)
        .to_numpy()
        .astype(float)
    )

    yv = np.asarray(A["y_value_90"], float)
    yc = np.asarray(A["y_churn_90"], float)
    Xa = np.nan_to_num(Xa)
    Xo = np.nan_to_num(Xo)
    out = {
        "n": int(A.height),
        "value90_r2_aggregate": round(r2_score(yv, _cv(Xa, yv, "reg", seed=a.seed)), 4),
        "value90_r2_order": round(r2_score(yv, _cv(Xo, yv, "reg", seed=a.seed)), 4),
        "churn90_auc_aggregate": round(roc_auc_score(yc, _cv(Xa, yc, "clf", seed=a.seed)), 4),
        "churn90_auc_order": round(roc_auc_score(yc, _cv(Xo, yc, "clf", seed=a.seed)), 4),
    }
    out["value90_delta_r2"] = round(out["value90_r2_order"] - out["value90_r2_aggregate"], 4)
    out["churn90_delta_auc"] = round(out["churn90_auc_order"] - out["churn90_auc_aggregate"], 4)

    # next-event-after-a-cut: target = event type of first event after as_of
    fut = (
        B.filter(pl.col("ts") > as_of)
        .group_by("customer_key", maintain_order=True)
        .agg(pl.col("event_type").first().alias("next_et"))
    )
    J = A.select(["customer_key", "y_"] if False else ["customer_key"]).join(
        fut, on="customer_key", how="left"
    )
    mask = J["next_et"].is_not_null().to_numpy()
    classes = sorted(fut["next_et"].unique().to_list())
    cmap = {c: i for i, c in enumerate(classes)}
    yn = np.array([cmap.get(c, -1) for c in J["next_et"].to_list()])
    sel = mask & (yn >= 0)
    Xa_n = Xa[sel]
    # rebuild order-aware aligned to sel index
    So = S.join(fut, on="customer_key", how="left")
    Xo_n = (
        _dummies(
            So.select([c for c in So.columns if c not in ("customer_key", "next_et")]),
            cats + seq_cat,
        )
        .to_numpy()
        .astype(float)[sel]
    )
    Xa_n = np.nan_to_num(Xa_n)
    Xo_n = np.nan_to_num(Xo_n)
    yn = yn[sel]
    out["next_event_classes"] = classes
    out["next_event_n"] = int(sel.sum())
    out["next_event_acc_aggregate"] = round(
        accuracy_score(yn, _cv(Xa_n, yn, "multi", seed=a.seed)), 4
    )
    out["next_event_acc_order"] = round(accuracy_score(yn, _cv(Xo_n, yn, "multi", seed=a.seed)), 4)
    out["next_event_delta_acc"] = round(
        out["next_event_acc_order"] - out["next_event_acc_aggregate"], 4
    )

    dest = Path(a.out) if a.out else d / "order_probe.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
