"""Canonical stream adapter: apply a FROZEN schema to raw events.

Training discovers the schema (through `scripts/demo/stream_loader.py`). Serving must reuse
that same schema — vocabulary, typing, periods — and never rediscover it from the new batch,
because a serving batch may contain only a subset of values or fields.

Input contract (canonical stream):

    customer_key, event_ts, event_type, event_attributes

`transform` returns prepared per-entity sequences matching what the frozen encoder consumes:
(event_type, categorical, numeric, time, gaps, timestamps, raw).
"""

from __future__ import annotations

import datetime
import math
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

REQUIRED = ("customer_key", "event_ts", "event_type", "event_attributes")
SECONDS_PER_DAY = 86400.0  # principle-ok: unit conversion seconds->days, not a cadence


@dataclass
class Schema:
    """Everything needed to encode new events exactly as training did."""

    et_v: dict = field(default_factory=dict)
    catspec: list = field(default_factory=list)
    num: list = field(default_factory=list)
    rawspec: list = field(default_factory=list)
    periods: list = field(default_factory=list)
    as_of_ts: float = 0.0
    action_keys: list = field(default_factory=list)
    version: str = "1.0.0"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(
            et_v=dict(d.get("et_v") or {}),
            catspec=[dict(g) for g in (d.get("catspec") or [])],
            num=[tuple(x) for x in (d.get("num") or [])],
            rawspec=[tuple(x) for x in (d.get("rawspec") or [])],
            periods=list(d.get("periods") or []),
            as_of_ts=float(d.get("as_of_ts", 0.0)),
            action_keys=list(d.get("action_keys") or []),
            version=d.get("version", "1.0.0"),
        )

    @classmethod
    def from_prepared(cls, vs, et_v, catspec, num, periods, as_of_ts, action_keys=()):
        """Capture the training-time schema for later serving."""
        num_pairs = [tuple(x) for x in num]
        stored = vs.get("rawspec")
        rawspec = (
            [tuple(x) for x in stored]
            if stored
            else num_pairs
            + [(g["field"], "int") for g in catspec if g["field"] not in {f for f, _ in num_pairs}]
        )
        return cls(
            et_v=dict(et_v),
            catspec=[dict(g) for g in catspec],
            num=num_pairs,
            rawspec=rawspec,
            periods=list(periods),
            as_of_ts=float(as_of_ts),
            action_keys=list(action_keys or []),
        )


def _local_tz():
    import os

    try:
        return os.path.realpath("/etc/localtime").split("/zoneinfo/")[-1] or "UTC"
    except Exception:  # noqa: BLE001
        return "UTC"


def _validate(df: pl.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"canonical stream missing required columns: {missing}")
    if df.height and df["customer_key"].null_count():
        raise ValueError("customer_key contains nulls")
    if df.height and df["event_ts"].null_count():
        raise ValueError("event_ts contains nulls")


def _apply_categorical(df: pl.DataFrame, g: dict) -> pl.Expr:
    """Map one field with the frozen vocabulary (or frozen hash buckets)."""
    name = f"_gc_{g['field']}"
    field_name = g["field"]
    if field_name not in df.columns:
        return pl.lit(0, dtype=pl.Int64).alias(name)
    raw = df[field_name].cast(pl.String)
    if g.get("mode") == "hash":
        buckets = int(g["buckets"])

        def _bucket(value):
            if value is None or value == "":
                return 0
            return 1 + zlib.crc32(f"{field_name}={value}".encode()) % buckets

        return raw.map_elements(_bucket, return_dtype=pl.Int64).alias(name)
    return (
        raw.replace_strict(g.get("map") or {}, default=0, return_dtype=pl.Int64)
        .fill_null(0)
        .alias(name)
    )


def transform(df: pl.DataFrame, schema: Schema, as_of_ts: float | None = None):
    """Encode raw canonical events with a frozen schema.

    Returns `(keys, seqs, receipt)`. Schema discovery never happens here.
    """
    _validate(df)
    as_of = float(schema.as_of_ts if as_of_ts is None else as_of_ts)
    if as_of <= 0:
        raise ValueError("transform requires a positive as_of timestamp")
    if not schema.et_v and not schema.catspec:
        raise ValueError("schema is empty; cannot encode")

    base = (
        df.select([pl.col(c) for c in REQUIRED])
        .with_columns(pl.col("event_ts").cast(pl.String).str.to_datetime().alias("ts"))
        .filter(pl.col("ts") <= pl.lit(datetime.datetime.fromtimestamp(as_of)))
        .sort(["customer_key", "ts"])
        .with_columns(
            pl.col("ts").dt.epoch("us").alias("_nus"),
            pl.col("event_type").cast(pl.String).alias("_et"),
        )
    )
    if base.height == 0:
        return [], {}, {"rows": 0, "entities": 0, "as_of": as_of}

    tz = _local_tz()
    naive = pl.col("ts").dt.epoch("s")
    tzep = (
        pl.col("ts")
        .dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
        .dt.epoch("s")
    )
    base = base.with_columns((tzep - naive).fill_null(strategy="forward").alias("_offs"))
    base = base.with_columns(
        ((pl.col("_nus") + pl.col("_offs") * 1_000_000) / 1_000_000.0).alias("_tss")
    )
    n = base.height

    attrs = base["event_attributes"].cast(pl.String).str.json_decode(infer_schema_length=None)
    frame = base.with_columns(attrs.alias("_j")).unnest("_j")

    etA = (
        base["_et"]
        .replace_strict(schema.et_v, default=0, return_dtype=pl.Int64)
        .to_numpy()
        .astype(np.int64)
    )

    cat_exprs = [_apply_categorical(frame, g) for g in schema.catspec]
    if cat_exprs:
        frame = frame.with_columns(cat_exprs)
    if schema.catspec:
        mats = [
            frame[f"_gc_{g['field']}"].cast(pl.Int64).fill_null(0).to_numpy()
            for g in schema.catspec
        ]
        gcatA = np.column_stack(mats).astype(np.int64)
    else:
        gcatA = np.zeros((n, 0), np.int64)

    def _numeric(field_name):
        if field_name in frame.columns:
            return frame[field_name].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
        return np.zeros(n, np.float64)

    gnumA = np.zeros((n, len(schema.num)), np.float32)
    grawA = np.zeros((n, len(schema.rawspec)), np.float32)
    for j, (f, kind) in enumerate(schema.num):
        if kind == "ts" and f in frame.columns:
            # (event time - attribute timestamp) in days, matching training
            delta = (
                frame.select(
                    (pl.col("ts") - pl.col(f).cast(pl.String).str.to_datetime(strict=False))
                    .dt.total_microseconds()
                    .alias("_d")
                )["_d"]
                .to_numpy()
                .astype(np.float64)
            )
            x = np.nan_to_num(delta / 1_000_000.0 / SECONDS_PER_DAY, nan=0.0)
        else:
            x = _numeric(f)
        grawA[:, j] = x
        gnumA[:, j] = np.where(x != 0, np.sign(x) * np.log1p(np.abs(x)), 0.0)
    for j, (f, _kind) in enumerate(schema.rawspec[len(schema.num) :]):
        grawA[:, len(schema.num) + j] = _numeric(f)

    g_us = (
        base.select(pl.col("ts").diff().over("customer_key").dt.total_microseconds())
        .to_series()
        .to_numpy()
    )
    gapsA = np.maximum(np.nan_to_num((g_us / 1_000_000.0) / SECONDS_PER_DAY, nan=0.0), 0.0)
    tssA = base["_tss"].to_numpy().astype(np.float64)
    timeA = np.zeros((n, 1 + 2 * len(schema.periods)), np.float32)
    timeA[:, 0] = np.log1p(gapsA)
    for j, period in enumerate(schema.periods):
        if not period:
            continue
        ph = 2 * math.pi * (tssA / 3600.0) / float(period)
        timeA[:, 1 + 2 * j] = np.sin(ph)
        timeA[:, 2 + 2 * j] = np.cos(ph)

    ck = base["customer_key"].to_numpy()
    b = np.flatnonzero(ck[1:] != ck[:-1]) + 1
    starts = np.concatenate(([0], b))
    ends = np.append(b, n)
    seqs = {
        str(ck[s]): (
            etA[s:e],
            gcatA[s:e],
            gnumA[s:e],
            timeA[s:e],
            gapsA[s:e],
            tssA[s:e],
            grawA[s:e],
        )
        for s, e in zip(starts, ends)
    }
    receipt = {
        "rows": int(n),
        "entities": int(len(seqs)),
        "fields": len(schema.catspec),
        "numeric": len(schema.num),
        "periods": list(schema.periods),
        "as_of": as_of,
    }
    return sorted(seqs), seqs, receipt
