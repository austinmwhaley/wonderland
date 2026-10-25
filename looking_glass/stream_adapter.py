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


# ---------------------------------------------------------------------------
# Discovery: the canonical way to derive a Schema from a raw stream.
# Sequence construction then goes through `transform`, so training and serving
# cannot drift apart.
# ---------------------------------------------------------------------------
_NUM_DTYPES = (
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
)
_INT_DTYPES = ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64")


def _is_iso(value) -> bool:
    try:
        datetime.datetime.fromisoformat(str(value))
        return True
    except Exception:  # noqa: BLE001
        return False


def discover(
    df: pl.DataFrame,
    as_of_ts: float,
    action_keys=(),
    coverage: float = 0.99,
    periods: int = 2,
):
    """Derive a Schema (and prepared sequences) from a raw canonical stream.

    This is the *training-time* entry point. Everything it derives is frozen into the
    artifact; `transform` then reuses it for serving, so the two paths share one mapping.
    """
    _validate(df)
    as_of = float(as_of_ts)
    if as_of <= 0:
        raise ValueError("discover requires a positive as_of timestamp")
    raw = df.select([pl.col(c) for c in REQUIRED])
    if raw.height == 0:
        raise ValueError("cannot discover a schema from an empty stream")

    parsed = (
        raw.with_columns(pl.col("event_ts").cast(pl.String).str.to_datetime().alias("ts"))
        .filter(pl.col("ts") <= pl.lit(datetime.datetime.fromtimestamp(as_of)))
        .sort(["customer_key", "ts"])
    )
    if parsed.height == 0:
        raise ValueError("no events at or before as_of")
    attrs = parsed["event_attributes"].cast(pl.String).str.json_decode(infer_schema_length=None)
    frame = parsed.with_columns(attrs.alias("_j")).unnest("_j")
    n = frame.height

    reserved = {
        "customer_key",
        "event_ts",
        "event_type",
        "event_attributes",
        "ts",
        "_j",
        "target_key",
    }
    fields = sorted(set(frame.columns) - reserved)

    # ---- type inference, mirroring the loader exactly (documented, behavioural)
    cat, num, cat_int = [], [], []
    for f in fields:
        col = frame[f]
        as_str = col.cast(pl.String)
        sample = as_str.filter(as_str.is_not_null() & (as_str != "")).head(200).to_list()
        if sample and all(_is_iso(v) for v in sample):
            num.append((f, "ts"))
            continue
        dtype = str(col.dtype)
        if dtype in _NUM_DTYPES:
            uniq = as_str.filter(as_str.is_not_null() & (as_str != "")).n_unique()
            if dtype in _INT_DTYPES and uniq <= 2 * int(math.sqrt(n)):
                cat.append(f)
                cat_int.append(f)
            else:
                num.append((f, "num"))
            continue
        cat.append(f)

    catspec = []
    for f in cat:
        as_str = frame[f].cast(pl.String)
        vals = as_str.filter(as_str.is_not_null() & (as_str != ""))
        vc = vals.value_counts(sort=True)
        cnt = dict(zip(vc[vc.columns[0]].to_list(), vc["count"].to_list()))
        if len(cnt) > 2 * int(math.sqrt(n)):
            buckets = 1 << max(4, int(math.ceil(math.log2(2 * math.sqrt(n)))))
            catspec.append({"field": f, "mode": "hash", "buckets": buckets, "size": buckets + 1})
        else:
            total = sum(cnt.values())
            first = {v: i for i, v in enumerate(vals.unique(maintain_order=True).to_list())}
            most = sorted(cnt, key=lambda v: (-cnt[v], first[v]))
            mapping, acc, i = {}, 0, 1
            for v in most:
                if total and acc / total >= coverage:
                    break
                mapping[v] = i
                i += 1
                acc += cnt[v]
            catspec.append({"field": f, "mode": "vocab", "map": mapping, "size": i})

    # ---- event types
    et_v = {
        v: i + 1
        for i, v in enumerate(
            sorted({v for v in frame["event_type"].cast(pl.String).to_list() if v})
        )
    }
    et_v[""] = 0

    # ---- exogenous event types from declared action keys
    ak = [f for f in (action_keys or []) if f in frame.columns]
    exo_idx = []
    if ak:
        is_act = pl.any_horizontal([pl.col(f).is_not_null() for f in ak])
    elif "target_key" in frame.columns:
        is_act = pl.col("target_key").is_not_null()
    else:
        is_act = None
    if is_act is not None:
        flag = (
            frame.select(pl.col("event_type").alias("_et"), is_act.alias("_a"))
            .group_by("_et")
            .agg(pl.col("_a").mean())
        )
        exo_idx = sorted(
            {
                et_v[r["_et"]]
                for r in flag.iter_rows(named=True)
                if r["_a"] > 0.5 and r["_et"] in et_v
            }
        )

    # ---- data-derived periods from hourly event counts
    # Periods are derived on LOCAL-time timestamps, matching the training loader: a
    # constant tz shift changes histogram binning and therefore the spectral ranking.
    _tz = _local_tz()
    _naive = pl.col("ts").dt.epoch("s")
    _tzep = (
        pl.col("ts")
        .dt.replace_time_zone(_tz, ambiguous="earliest", non_existent="null")
        .dt.epoch("s")
    )
    tss = (
        frame.select(
            (
                (
                    pl.col("ts").dt.epoch("us")
                    + (_tzep - _naive).fill_null(strategy="forward") * 1_000_000
                )
                / 1_000_000.0
            ).alias("_t")
        )["_t"]
        .to_numpy()
        .astype(float)
    )
    h0 = float(tss.min())
    hn = int(math.ceil((float(tss.max()) - h0) / 3600.0)) + 1
    counts, _ = np.histogram(tss, bins=hn, range=(h0, h0 + hn * 3600.0))
    spec = np.abs(np.fft.rfft(counts - counts.mean())) ** 2
    freqs = np.fft.rfftfreq(hn, d=1.0)
    spec[0] = 0.0
    peaks = []
    for i in range(1, len(spec) - 1):
        if (
            spec[i] > spec[i - 1]
            and spec[i] >= spec[i + 1]
            and spec[i] > np.percentile(spec[1:], 90)
        ):
            peaks.append((spec[i], freqs[i]))
    peaks = sorted(peaks, reverse=True)[:periods]
    periods_found = [1.0 / f for _, f in peaks if f > 0]

    schema = Schema(
        et_v=et_v,
        catspec=catspec,
        num=[tuple(x) for x in num],
        rawspec=[tuple(x) for x in num] + [(f, "int") for f in cat_int],
        periods=periods_found,
        as_of_ts=as_of,
        action_keys=list(action_keys or []),
    )

    # sequences come from the SAME mapping serving uses
    keys, seqs, rcpt = transform(raw, schema, as_of)

    # ---- temporal configuration from observed gaps/spans
    gaps = np.concatenate([seqs[k][4] for k in keys])
    pos = gaps[gaps > 0]
    if not len(pos):
        pos = np.array([1.0])
    e = max(2, min(4, len(periods_found) + 1))
    taus = np.quantile(pos, np.linspace(0.25, 0.95, e))
    spans = np.array([(seqs[k][5][-1] - seqs[k][5][0]) / SECONDS_PER_DAY for k in keys])
    sp = np.quantile(spans, [0.05, 0.25, 0.5, 0.75, 0.95])
    ladder = np.unique(np.geomspace(max(taus.min(), 1e-3), max(sp[0], 1e-3), 8))
    horizons = np.unique(np.concatenate([taus, ladder, sp]))

    miss = []
    for j in range(len(catspec)):
        z = np.concatenate([(seqs[k][1][:, j] == 0).astype(float) for k in keys])
        miss.append(float(z.mean()))
    p_mask = float(np.clip(np.mean(miss) if miss else 0.1, 0.05, 0.4))

    gidx = np.repeat(np.arange(len(keys), dtype=np.int64), [len(seqs[k][0]) for k in keys])
    gcat_all = np.concatenate([seqs[k][1] for k in keys])
    id_idx = [
        j
        for j, g in enumerate(catspec)
        if _identity_like(gcat_all[:, j], gidx, len(keys), int(g["size"]))
    ]
    rel_names = [f for f in ("campaign_id", "product_id") if f in {g["field"] for g in catspec}]
    rel_idx = [j for j, g in enumerate(catspec) if g["field"] in rel_names]

    raw_dim = len(schema.rawspec)
    vs = {
        "et": len(et_v),
        "gcat": [g["size"] for g in catspec],
        "gnum": len(num),
        "time_dim": 1 + 2 * len(periods_found),
        "horizons": horizons.tolist(),
        "id_idx": id_idx,
        "id_dims": [catspec[i]["size"] for i in id_idx],
        "rel_idx": rel_idx,
        "unc_dim": len(catspec) + len(num),
        "ident_dims": [min(int(sz), 32) for sz in [g["size"] for g in catspec]],
        "raw": raw_dim,
        "mass_dim": raw_dim + len(et_v),
        "past_dim": len(et_v) + len(catspec) + 6 * raw_dim + 7,
        "fut_dim": len(et_v) + len(catspec) + 2 * raw_dim,
        "as_of_ts": as_of,
        "rawspec": [[f, k] for f, k in schema.rawspec],
        "et_v": dict(et_v),
        "catspec": catspec,
        "num": [list(x) for x in num],
        "periods": list(periods_found),
        "action_keys": list(action_keys or []),
        "exo_idx": exo_idx,
        "exo_source": "declared action_keys" if ak else "structural (target_key)",
        "id_source": "schema-derived (per-entity constant)",
        "rel_source": "declared configuration",
    }
    vs["ident_dim"] = int(sum(vs["ident_dims"]))
    return {
        "keys": keys,
        "seqs": seqs,
        "schema": schema,
        "vs": vs,
        "et_v": et_v,
        "catspec": catspec,
        "num": [tuple(x) for x in num],
        "taus": taus,
        "p_mask": p_mask,
        "periods": periods_found,
        "horizons": horizons,
        "as_of_ts": as_of,
        "receipt": rcpt,
    }


def _identity_like(col, group_index, n_entities, size, threshold=0.95):
    keep = col > 0
    if not keep.any():
        return False
    npairs = np.unique(group_index[keep] * size + col[keep])
    per_entity = np.bincount(npairs // size, minlength=int(n_entities))
    return float((per_entity <= 1).mean()) >= threshold
