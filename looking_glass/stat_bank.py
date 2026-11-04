"""DEC-590 statistic bank — served blocks.

The first block is the **level/value block**: per-customer extensive statistics of the
valued events in the observed stream. It is generic (no event-name literal): a "valued
event" is any event with a non-zero primary numeric field, and the trailing windows are
derived from the stream's own inter-event gap distribution.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from looking_glass.stream_adapter import to_ts_expr

LEVEL_COLS = (
    "n_valued",
    "v_sum",
    "v_mean",
    "v_std",
    "q25",
    "q50",
    "q75",
    "q90",
    "n_w1",
    "v_w1",
    "n_w2",
    "v_w2",
    "n_neg",
    "tenure_d",
    "recency_d",
    "neg_rate",
)


def level_block(stream: pl.DataFrame, keys: list[str], as_of_epoch: float) -> np.ndarray:
    """(len(keys), 16) level block. Windows are derived from the median gap between valued
    events (w1 = 4 gaps, w2 = 12 gaps); no hand-picked 30/90-day literal."""
    df = stream
    if "ts" not in df.columns:
        df = df.with_columns(to_ts_expr(pl.col("event_ts")).alias("ts"))
    if "value" not in df.columns:
        return np.zeros((len(keys), len(LEVEL_COLS)), dtype=np.float64)
    valued = df.filter(pl.col("value").abs() > 0).sort(["customer_key", "ts"])
    if valued.height == 0:
        return np.zeros((len(keys), len(LEVEL_COLS)), dtype=np.float64)
    gaps = valued.with_columns(
        pl.col("ts").diff().over("customer_key").dt.total_seconds().truediv(86400).alias("gap")
    ).filter(pl.col("gap") > 0)
    med = float(gaps["gap"].median() or 1.0)
    w1, w2 = 4.0 * med, 12.0 * med
    tmax = df["ts"].max()  # the stream is already cut at as_of
    pos = valued.filter(pl.col("value") > 0)
    g = pos.group_by("customer_key").agg(
        pl.len().alias("n_valued"),
        pl.col("value").sum().alias("v_sum"),
        pl.col("value").mean().alias("v_mean"),
        pl.col("value").std().fill_null(0.0).alias("v_std"),
        pl.col("value").quantile(0.25).alias("q25"),
        pl.col("value").quantile(0.5).alias("q50"),
        pl.col("value").quantile(0.75).alias("q75"),
        pl.col("value").quantile(0.9).alias("q90"),
        pl.col("ts").min().alias("first_ts"),
        pl.col("ts").max().alias("last_ts"),
    )
    for tag, w in (("w1", w1), ("w2", w2)):
        sub = (
            pos.filter(((pl.lit(tmax) - pl.col("ts")).dt.total_seconds().truediv(86400)) <= w)
            .group_by("customer_key")
            .agg(pl.len().alias(f"n_{tag}"), pl.col("value").sum().alias(f"v_{tag}"))
        )
        g = g.join(sub, on="customer_key", how="left")
    neg = valued.filter(pl.col("value") < 0).group_by("customer_key").agg(pl.len().alias("n_neg"))
    g = g.join(neg, on="customer_key", how="left")
    g = g.with_columns(
        (pl.lit(tmax) - pl.col("first_ts")).dt.total_seconds().truediv(86400).alias("tenure_d"),
        (pl.lit(tmax) - pl.col("last_ts")).dt.total_seconds().truediv(86400).alias("recency_d"),
        (pl.col("n_neg").fill_null(0) / pl.col("n_valued").clip(1)).alias("neg_rate"),
    )
    key_df = pl.DataFrame({"customer_key": keys})
    m = key_df.join(g, on="customer_key", how="left")
    m = m.with_columns([pl.col(c).fill_null(0.0) for c in LEVEL_COLS if c in m.columns])
    for c in LEVEL_COLS:
        if c not in m.columns:
            m = m.with_columns(pl.lit(0.0).alias(c))
    return m.select(list(LEVEL_COLS)).to_numpy().astype(np.float64)


def level_block_seq(seq, med: float | None = None) -> np.ndarray:
    """The SAME level block computed from one sequence (the one composition path).

    `med` is the STREAM's global median gap (days); every path passes the same value.
    """
    _v = seq.get("value")
    _t = seq.get("ts")
    vals = np.asarray(_v, dtype=np.float64) if _v is not None else np.array([])
    ts = np.asarray(_t, dtype=np.float64) if _t is not None else np.array([])
    out = np.zeros(len(LEVEL_COLS), dtype=np.float64)
    if vals.size == 0 or ts.size != vals.size:
        return out
    pos = vals > 0
    neg = vals < 0
    pv, pt = vals[pos], ts[pos]
    if pv.size == 0:
        return out
    ref = float(seq.get("ref_epoch") or seq.get("anchor_epoch") or ts[-1])
    if med is None:
        gaps = np.diff(np.sort(pt))
        gaps = gaps[gaps > 0]
        med = float(np.median(gaps)) if gaps.size else 1.0
    w1, w2 = 4.0 * med, 12.0 * med
    age = (ref - pt) / 86400.0
    q = np.quantile(pv, [0.25, 0.5, 0.75, 0.9], method="nearest")
    out[:] = [
        float(pv.size),
        float(pv.sum()),
        float(pv.mean()),
        float(pv.std(ddof=1)) if pv.size > 1 else 0.0,
        q[0],
        q[1],
        q[2],
        q[3],
        float((age <= w1).sum()),
        float(pv[age <= w1].sum()),
        float((age <= w2).sum()),
        float(pv[age <= w2].sum()),
        float(neg.sum()),
        float((ref - ts.min()) / 86400.0),
        float((ref - ts.max()) / 86400.0),
        float(neg.sum()) / max(float(pv.size), 1.0),
    ]
    return out


def median_gap_seqs(seqs) -> float:
    gaps = []
    for s in seqs:
        v = s.get("value")
        t = s.get("ts")
        if v is None or t is None:
            continue
        vals = np.asarray(v, dtype=np.float64)
        ts = np.asarray(t, dtype=np.float64)
        pt = np.sort(ts[vals != 0])
        if pt.size > 1:
            g = np.diff(pt)
            gaps.append(g[g > 0])
    if not gaps:
        return 1.0
    allg = np.concatenate(gaps)
    return float(np.median(allg)) if allg.size else 1.0


def median_gap_stream(stream) -> float:
    from looking_glass.stream_adapter import to_ts_expr

    df = (
        stream
        if "ts" in stream.columns
        else stream.with_columns(to_ts_expr(pl.col("event_ts")).alias("ts"))
    )
    valued = df.filter(pl.col("value").abs() > 0).sort(["customer_key", "ts"])
    g = valued.with_columns(
        pl.col("ts").diff().over("customer_key").dt.total_seconds().truediv(86400).alias("gap")
    ).filter(pl.col("gap") > 0)
    return float(g["gap"].median() or 1.0)


def level_blocks_from_seqs(seqs, med: float | None = None) -> np.ndarray:
    if not seqs:
        return np.zeros((0, len(LEVEL_COLS)))
    med = median_gap_seqs(seqs) if med is None else med
    return np.vstack([level_block_seq(s, med) for s in seqs])
