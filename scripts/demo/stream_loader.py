"""Stream loader: event parquet -> per-customer sequence arrays (polars/numpy; no torch).

Single source of truth for the data path. engine=None -> CPU collects. engine="gpu"
-> cudf-polars GPU engine with per-node CPU fallback (raise_on_fail=False), run inside
the dedicated .venv-cudf process boundary (RAPIDS cu12 vs torch cu13 conflict).
"""

import datetime
import json
import math
import os as _os
import zlib
from pathlib import Path

import numpy as np
import polars as pl


def _collect(lf, engine):
    if engine != "gpu":
        return lf.collect()
    try:
        import cudf.polars  # noqa: F401

        return lf.collect(engine=pl.GPUEngine(raise_on_fail=False))
    except Exception:
        return lf.collect()


def _local_tz():
    try:
        return _os.path.realpath("/etc/localtime").split("/zoneinfo/")[-1] or "UTC"
    except Exception:
        return "UTC"


def _is_iso(v):
    try:
        datetime.datetime.fromisoformat(v)
        return True
    except Exception:
        return False


def _vocab(vals):
    u = sorted({str(v) for v in vals if v is not None and str(v) != ""})
    m = {v: i + 1 for i, v in enumerate(u)}
    m[""] = 0
    return m


def _prep_rev(data):
    d = Path(data)
    pb = d / "path_b/path_b_customer_event_stream.parquet"
    h = __import__("hashlib").sha256()
    h.update((d / "manifest.json").read_bytes()[:4096])
    h.update(str(pb.stat().st_mtime_ns).encode())
    h.update(str(pb.stat().st_size).encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:12]


_NUM_D = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)
_INT_D = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def derive_identity_fields(gcat, group_index, n_entities, catspec, single_threshold=0.95):
    """Schema-derived identity/static fields: essentially constant within an entity.

    A customer-level attribute (segment, region, ...) takes one value per entity;
    an event attribute varies within the entity.  No field name is consulted.
    `single_threshold` is a documented conservative fallback.
    """
    out = []
    for j, g in enumerate(catspec):
        col = np.asarray(gcat[:, j])
        keep = col > 0
        if not keep.any():
            continue
        size = int(g["size"])
        npairs = np.unique(np.asarray(group_index)[keep] * size + col[keep])
        per_entity = np.bincount(npairs // size, minlength=int(n_entities))
        if float((per_entity <= 1).mean()) >= float(single_threshold):
            out.append(j)
    return out


def load_data(data, engine=None):
    d = Path(data)
    manifest = json.loads((d / "manifest.json").read_text())
    as_of = manifest["as_of"]
    as_of_ts = datetime.datetime.fromisoformat(as_of).timestamp()
    A = pl.read_parquet(d / "path_a/path_a_wide_table.parquet")
    _naive = pl.col("ts").dt.epoch("s")
    _tzep = (
        pl.col("ts")
        .dt.replace_time_zone(_local_tz(), ambiguous="earliest", non_existent="null")
        .dt.epoch("s")
    )
    _off = (_tzep - _naive).fill_null(strategy="forward")
    B = _collect(
        pl.scan_parquet(d / "path_b/path_b_customer_event_stream.parquet")
        .with_columns(pl.col("event_ts").str.to_datetime().alias("ts"))
        .filter(pl.col("ts") <= pl.lit(as_of).str.to_datetime())
        .sort(["customer_key", "ts"])
        .select(
            pl.col("customer_key"),
            pl.col("ts"),
            _off.alias("_offs"),
            pl.col("ts").dt.epoch("us").alias("_nus"),
            pl.col("event_type").alias("_et"),
            pl.col("event_attributes"),
        ),
        engine,
    )
    keys = sorted(B["customer_key"].unique().to_list())
    _j = B["event_attributes"].str.json_decode(infer_schema_length=None)
    DF = B.drop("event_attributes").with_columns(_j.alias("_j")).unnest("_j")
    fields = sorted(
        set(DF.columns)
        - {"customer_key", "ts", "_nus", "_offs", "_et", "event_attributes", "target_key"}
    )
    n = DF.height
    cat, num, cat_int = [], [], []
    for f in fields:
        col = DF[f]
        sc = col.cast(pl.String)
        sv = sc.filter(sc.is_not_null() & (sc != ""))
        head = sv.head(200).to_list()
        if head and all(isinstance(v, str) for v in head) and all(_is_iso(v) for v in head):
            num.append((f, "ts"))
            continue
        if col.dtype in _NUM_D:
            u = sv.n_unique()
            if col.dtype in _INT_D and u <= 2 * int(math.sqrt(n)):
                cat.append(f)
                cat_int.append(f)
            else:
                num.append((f, "num"))
        else:
            cat.append(f)
    rawspec = list(num) + [(f, "int") for f in cat_int]
    catspec = []
    for f in cat:
        sc = DF[f].cast(pl.String)
        sv = sc.filter(sc.is_not_null() & (sc != ""))
        vc = sv.value_counts(sort=True)
        cnt = dict(zip(vc[vc.columns[0]].to_list(), vc["count"].to_list()))
        if len(cnt) > 2 * int(math.sqrt(n)):
            buckets = 1 << max(4, int(math.ceil(math.log2(2 * math.sqrt(n)))))
            catspec.append(dict(field=f, mode="hash", buckets=buckets, size=buckets + 1))
        else:
            total = sum(cnt.values())
            first = {v: i for i, v in enumerate(sv.unique(maintain_order=True).to_list())}
            most = sorted(cnt, key=lambda v: (-cnt[v], first[v]))
            m = {}
            acc = 0
            i = 1
            for v in most:
                if total and acc / total >= 0.99:
                    break
                m[v] = i
                i += 1
                acc += cnt[v]
            catspec.append(dict(field=f, mode="vocab", map=m, size=i))
    et_v = _vocab(B["_et"].to_list())
    exprs = [pl.col("_et").replace_strict(et_v, default=0, return_dtype=pl.Int64).alias("_et_i")]
    for j, g in enumerate(catspec):
        if g["mode"] == "hash":
            vals = DF[g["field"]].cast(pl.String).drop_nulls().unique().to_list()
            mp = {
                v: 1 + zlib.crc32((g["field"] + "=" + v).encode()) % g["buckets"]
                for v in vals
                if v != ""
            }
        else:
            mp = g["map"]
        if mp:
            exprs.append(
                pl.col(g["field"])
                .cast(pl.String)
                .replace_strict(mp, default=0, return_dtype=pl.Int64)
                .fill_null(0)
                .alias(f"_gc{j}")
            )
        else:
            exprs.append(pl.lit(0, dtype=pl.Int64).alias(f"_gc{j}"))
    for j, (f, kind) in enumerate(num):
        if kind == "ts":
            _dts = (pl.col("ts") - pl.col(f).str.to_datetime(strict=False)).dt.total_microseconds()
            exprs.append(_dts.alias(f"_gts{j}"))
        else:
            x = pl.col(f).cast(pl.Float64, strict=False).fill_null(0.0)
            exprs.append(x.alias(f"_gr{j}"))
            exprs.append(
                pl.when(x != 0)
                .then(x.sign() * (x.abs() + 1.0).log())
                .otherwise(0.0)
                .alias(f"_gn{j}")
            )
    for j, (f, kind) in enumerate(rawspec[len(num) :]):
        exprs.append(pl.col(f).cast(pl.Float64, strict=False).fill_null(0.0).alias(f"_gri{j}"))
    _d = pl.col("ts").diff().over("customer_key").dt.total_microseconds()
    exprs.append(_d.alias("_g_us"))
    out_cols = ["customer_key", "_nus", "_offs", "_et_i"] + [f"_gc{j}" for j in range(len(catspec))]
    for j, (f, kind) in enumerate(num):
        out_cols += [f"_gts{j}"] if kind == "ts" else [f"_gr{j}", f"_gn{j}"]
    out_cols += [f"_gri{j}" for j in range(len(rawspec) - len(num))] + ["_g_us"]
    DF2 = _collect(DF.lazy().with_columns(exprs).select(out_cols), engine)
    tssA = (DF2["_nus"].to_numpy() + DF2["_offs"].to_numpy() * 1000000) / 1.0e6
    etA = DF2["_et_i"].to_numpy()
    gcatA = np.column_stack([DF2[f"_gc{j}"].to_numpy() for j in range(len(catspec))]).astype(
        np.int64
    )
    gnumA = np.zeros((n, len(num)), np.float32)
    grawA = np.zeros((n, len(rawspec)), np.float32)
    for j, (f, kind) in enumerate(num):
        if kind == "ts":
            dts = DF2[f"_gts{j}"].to_numpy().astype(np.float64)
            x = np.nan_to_num((dts / 1.0e6) / 86400.0, nan=0.0)
            grawA[:, j] = x
            gnumA[:, j] = np.where(x != 0, np.sign(x) * np.log1p(np.abs(x)), 0.0)
        else:
            grawA[:, j] = DF2[f"_gr{j}"].to_numpy()
            gnumA[:, j] = DF2[f"_gn{j}"].to_numpy()
    for j, (f, kind) in enumerate(rawspec[len(num) :]):
        grawA[:, len(num) + j] = DF2[f"_gri{j}"].to_numpy()
    g_us = DF2["_g_us"].to_numpy().astype(np.float64)
    gapsA = np.maximum(np.nan_to_num((g_us / 1.0e6) / 86400.0, nan=0.0), 0.0)
    allt = tssA
    h0 = allt.min()
    hn = int(np.ceil((allt.max() - h0) / 3600.0)) + 1
    counts, _ = np.histogram(allt, bins=hn, range=(h0, h0 + hn * 3600.0))
    spec = np.abs(np.fft.rfft(counts - counts.mean())) ** 2
    freqs = np.fft.rfftfreq(hn, d=1.0)
    spec[0] = 0
    peaks = []
    for i in range(1, len(spec) - 1):
        if (
            spec[i] > spec[i - 1]
            and spec[i] >= spec[i + 1]
            and spec[i] > np.percentile(spec[1:], 90)
        ):
            peaks.append((spec[i], freqs[i]))
    peaks = sorted(peaks, reverse=True)[:2]
    periods = [1.0 / f for _, f in peaks if f > 0]
    time_dim = 1 + 2 * len(periods)
    timeA = np.zeros((n, time_dim), np.float32)
    timeA[:, 0] = np.log1p(gapsA)
    for j, P in enumerate(periods):
        ph = 2 * math.pi * (tssA / 3600.0) / P
        timeA[:, 1 + 2 * j] = np.sin(ph)
        timeA[:, 2 + 2 * j] = np.cos(ph)
    ck = DF2["customer_key"].to_numpy()
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
    spans = (tssA[ends - 1] - tssA[starts]) / 86400.0
    gaps_all = gapsA[gapsA > 0]
    if not len(gaps_all):
        gaps_all = np.array([1.0])
    e = max(2, min(4, len(peaks) + 1))
    taus = np.quantile(gaps_all, np.linspace(0.25, 0.95, e))
    _sp = np.quantile(spans, [0.05, 0.25, 0.5, 0.75, 0.95])
    _ladder = np.unique(np.geomspace(max(taus.min(), 1e-3), max(_sp[0], 1e-3), 8))
    horizons = np.unique(np.concatenate([taus, _ladder, _sp]))
    lens = (ends - starts).astype(np.float64)
    miss = []
    for j in range(len(catspec)):
        z = (gcatA[:, j] == 0).astype(np.float64)
        miss.extend((np.add.reduceat(z, starts) / lens).tolist())
    p_mask = float(np.clip(np.mean(miss) if miss else 0.1, 0.05, 0.4))
    vs = dict(et=len(et_v), gcat=[g["size"] for g in catspec], gnum=len(num), time_dim=time_dim)
    vs["horizons"] = horizons.tolist()
    # ---- schema-derived field roles (no business field names) ----------------
    # Identity/static: the field is essentially constant within an entity
    # (a customer-level attribute rather than an event attribute).  Derived from
    # the data; a field name is never consulted.
    _gidx = np.repeat(np.arange(len(keys), dtype=np.int64), lens.astype(np.int64))
    id_idx = derive_identity_fields(gcatA, _gidx, len(keys), catspec)
    vs["id_idx"] = id_idx
    vs["id_dims"] = [catspec[i]["size"] for i in vs["id_idx"]]
    # Relational/context: declared configuration, NOT a universal schema rule.
    # The reference's relational pair is not recoverable from a principled
    # data-derived test (verified: cardinality and per-entity distinct patterns
    # do not separate it from ordinary behavioural attributes), so it is kept as
    # an explicit declaration and flagged for a later loop.
    _rel = _os.environ.get("WQ_REL_FIELDS", "campaign_id,product_id")
    rel_names = {x for x in _rel.split(",") if x}
    vs["rel_idx"] = [i for i, g in enumerate(catspec) if g["field"] in rel_names]
    vs["rel_source"] = "declared (WQ_REL_FIELDS); not schema-derived"
    vs["id_source"] = "schema-derived (per-entity constant, >=95% single-valued)"
    vs["unc_dim"] = len(catspec) + len(num)
    vs["ident_dims"] = [min(int(sz), 32) for sz in vs["gcat"]]
    vs["ident_dim"] = int(sum(vs["ident_dims"]))
    vs["actions"] = manifest.get("actions", [])
    vs["raw"] = len(rawspec)
    vs["mass_dim"] = len(rawspec) + len(et_v)
    vs["past_dim"] = len(et_v) + len(catspec) + 6 * len(rawspec) + 7
    vs["fut_dim"] = len(et_v) + len(catspec) + 2 * len(rawspec)
    return B, A, keys, seqs, vs, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons
