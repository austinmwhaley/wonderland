#!/usr/bin/env python3
"""DEC-590 value block diagnostic (B6.2): does an explicit per-customer level/quantile
block close the 90-day value/orders gap that the frozen E misses?

History features only (<= as_of): purchase counts/value/quantiles, 30/90-day windows,
recency, tenure, return rate. Fit A -> B ridge on E, E+block, block alone, and the PATH-A
table; the frozen target is y_value_90 / y_orders_90. DIAGNOSTIC — no gate edit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_battery_law import _load_all, _ridge_head  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from measure_judge_law import _table_matrix  # noqa: E402
from safe_run import ensure_boundary  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "value_block.json"


def _round(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def _boot_r2_diff(y, p1, p2, n=2000, seed=0):
    sse1 = (y - p1) ** 2
    sse2 = (y - p2) ** 2
    sst = (y - y.mean()) ** 2
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(
            (1.0 - sse1[i].sum() / max(sst[i].sum(), 1e-12))
            - (1.0 - sse2[i].sum() / max(sst[i].sum(), 1e-12))
        )
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _block(stream: pl.DataFrame) -> pl.DataFrame:
    df = stream
    if "ts" not in df.columns:
        from looking_glass.stream_adapter import to_ts_expr

        df = df.with_columns(to_ts_expr(pl.col("event_ts")).alias("ts"))
    tmax = df["ts"].max()
    pur = df.filter(pl.col("event_type") == "purchase")
    ret = df.filter(pl.col("event_type") == "return")
    g = pur.group_by("customer_key").agg(
        pl.len().alias("n_orders"),
        pl.col("value").sum().alias("v_sum"),
        pl.col("value").mean().alias("v_mean"),
        pl.col("value").std().fill_null(0.0).alias("v_std"),
        pl.col("value").quantile(0.25).alias("q25"),
        pl.col("value").quantile(0.5).alias("q50"),
        pl.col("value").quantile(0.75).alias("q75"),
        pl.col("value").quantile(0.9).alias("q90"),
        pl.col("ts").max().alias("last_ts"),
        pl.col("ts").min().alias("first_ts"),
    )
    for days in (30, 90):
        w = (
            pur.filter(pl.col("ts") > tmax - pl.duration(days=days))
            .group_by("customer_key")
            .agg(pl.len().alias(f"n_{days}"), pl.col("value").sum().alias(f"v_{days}"))
        )
        g = g.join(w, on="customer_key", how="left")
    r = ret.group_by("customer_key").agg(pl.len().alias("n_returns"))
    g = g.join(r, on="customer_key", how="left")
    g = g.with_columns(
        (pl.col("last_ts") - pl.col("first_ts"))
        .dt.total_seconds()
        .truediv(86400)
        .alias("tenure_d"),
        (pl.lit(tmax) - pl.col("last_ts")).dt.total_seconds().truediv(86400).alias("recency_d"),
        (pl.col("n_returns").fill_null(0) / pl.col("n_orders").clip(1)).alias("return_rate"),
    )
    cols = [
        "n_orders",
        "v_sum",
        "v_mean",
        "v_std",
        "q25",
        "q50",
        "q75",
        "q90",
        "n_30",
        "v_30",
        "n_90",
        "v_90",
        "n_returns",
        "tenure_d",
        "recency_d",
        "return_rate",
    ]
    return g.select(["customer_key", *cols]).with_columns([pl.col(c).fill_null(0.0) for c in cols])


def main(argv=None) -> int:
    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/cfm_v2k")
    ap.add_argument("--demo", default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/demo_v2")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    demo = Path(a.demo)
    products = Path(a.products)
    _reg, truth, emb, _raw, stream, bkeys, akeys, _asof = _load_all(
        str(products / "cfm_products.duckdb"), demo
    )
    keys = [k for k in emb if k in set(akeys) | set(bkeys)]
    A = set(akeys)
    labels = pl.read_parquet(demo / "path_a" / "customer_labels.parquet")
    lab = {r["customer_key"]: r for r in labels.iter_rows(named=True)}
    blk = _block(stream)
    bmap = {r["customer_key"]: r for r in blk.iter_rows(named=True)}
    cols = [c for c in blk.columns if c != "customer_key"]
    B = np.asarray([[float(bmap.get(k, {}).get(c, 0.0) or 0.0) for c in cols] for k in keys])
    X = np.vstack([emb[k] for k in keys])
    table, _ = _table_matrix(demo, keys)
    T = np.vstack([table[k] for k in keys])
    amask = np.asarray([k in A for k in keys])
    bmask = ~amask
    out = {}
    for tgt in ("y_value_90", "y_orders_90"):
        y = np.asarray([float(lab[k][tgt]) for k in keys], dtype=float)
        row = {"n_A": int(amask.sum()), "n_B": int(bmask.sum()), "dims": int(B.shape[1])}
        preds = {}
        for name, F in (("E", X), ("E+block", np.hstack([X, B])), ("block", B), ("table", T)):
            m = _ridge_head()
            m.fit(F[amask], y[amask])
            preds[name] = m.predict(F[bmask])
            row[f"{name}_r2"] = _round(r2_score(y[bmask], preds[name]))
        lo, hi = _boot_r2_diff(y[bmask], preds["E+block"], preds["table"])
        row["Eplusblock_minus_table_ci"] = [_round(lo), _round(hi)]
        out[tgt] = row
    body = {
        "kind": "value_block_diagnostic",
        "protocol": (
            "DEC-590 value block: per-customer history level/quantile block appended to E; "
            "A->B ridge on y_value_90 / y_orders_90; E vs E+block vs block vs PATH-A table. "
            "DIAGNOSTIC — no gate edit."
        ),
        "products_identity": products_identity(products),
        "results": out,
        "complete": bool(out),
        "seconds": 0.0,
    }
    body["seconds"] = round(time.perf_counter() - t0, 1)
    stamp_merge_write(
        Path(a.out), body, " ".join([Path(__file__).name] + sys.argv[1:]), stamp_blocks=False
    )
    print(f"value block -> {a.out} ({body['seconds']}s)")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
