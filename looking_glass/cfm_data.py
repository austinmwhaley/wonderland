"""CFM data prep: stream reads, splits, covariates, and sequence building."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

from looking_glass.cfm_config import AT, CFMConfig, _h, as_of_epoch

# The randomized-action cadence (seconds). One action "period" = one week — the
# convention the use-case gate's `_period_rows` also buckets by, so the training
# objective and the decision gate measure the SAME dose-response.
ACTION_PERIOD_S = 604800.0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
_BASE_COLS = ("customer_key", "event_ts", "event_type", "event_attributes")
# Canonical metadata columns that are NOT event attributes.
_META_COLS = {"source_table"}


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _json_expr(key: str) -> str:
    esc = str(key).replace("'", "''")
    return f"json_extract_string(event_attributes, '$.\"{esc}\"') AS {_quote_ident(key)}"


def _nested_value(v) -> bool:
    s = str(v).lstrip()
    return bool(s) and s[0] in "{["


def _infer_field_types(df, fields, cfg: CFMConfig):
    """Classify discovered attribute fields from the data (no fixed threshold).

    LAW: a field is NUMERIC iff **every** observed non-null value parses as a
    finite float. Any non-numeric observation means the field is not a pure
    number, so it is treated as CATEGORICAL (fail safe — never silently coerce a
    mixed column to a number). Fields whose values are mostly nested objects/
    arrays are skipped in v1 (recorded, never crashed on). The per-field decision
    and the measured parse fraction are emitted in ``cfg._field_receipt``.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    skipped: list[str] = []
    receipt: dict[str, dict] = {}
    if not fields or df.height == 0:
        return numeric, categorical, skipped, receipt
    for f in fields:
        s = df[f].drop_nulls()
        if s.len() == 0:
            categorical.append(f)
            receipt[f] = {"kind": "categorical", "reason": "all-null", "finite_frac": 0.0}
            continue
        vals = s.cast(pl.Utf8).to_list()
        nested = sum(_nested_value(v) for v in vals)
        if nested and nested * 2 >= s.len():
            skipped.append(f)
            receipt[f] = {"kind": "skipped_nested", "nested_frac": nested / s.len()}
            continue
        parsed = s.cast(pl.Float64, strict=False)
        finite = int(parsed.is_finite().fill_null(False).sum())
        ratio = finite / max(s.len(), 1)
        is_num = ratio >= 1.0
        (numeric if is_num else categorical).append(f)
        receipt[f] = {
            "kind": "numeric" if is_num else "categorical",
            "finite_frac": round(float(ratio), 6),
            # a mixed column is kept categorical (fail safe); flag it for diagnosis.
            "dirty": bool(0.0 < ratio < 1.0),
        }
    return sorted(numeric), sorted(categorical), sorted(skipped), receipt


def _read_stream(
    cfg: CFMConfig,
    lo_epoch: float | None = None,
    hi_epoch: float | None = None,
    update_fields: bool = True,
):
    """Read the canonical stream and DISCOVER every event-attribute field.

    No attribute name is hardcoded: top-level JSON keys (and legacy wide
    columns) are discovered, typed from a seeded sample, and emitted as
    columns. Inferred numeric fields go to ``cfg.numeric_fields``; categorical
    fields to ``cfg.attr_fields`` (both sorted/deterministic). Nested values are
    skipped and recorded. ``read_stream_window`` reuses this with
    ``update_fields=False`` so inference keeps the checkpoint's field contract.
    """
    import duckdb

    db = str(cfg.db)
    is_ipc = db.endswith((".arrow", ".feather", ".ipc"))
    con = None
    try:
        if is_ipc:
            raw = pl.read_ipc(cfg.db, memory_map=True)
            con = duckdb.connect()
            con.register("_stream", raw)
            src = "_stream"
            tcols = set(raw.columns)
        else:
            is_file = db.endswith((".duckdb", ".ddb"))
            con = duckdb.connect(db, read_only=True) if is_file else duckdb.connect()
            # Parquet/other files are queried in place via read_parquet (the
            # canonical compressed form); DuckDB files use their table.
            src = cfg.table if is_file else f"read_parquet('{db}')"
            tcols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
        # GENERIC exhaustive consumption: discover EVERY attribute key from the
        # JSON attributes (plus legacy top-level columns), so the encoder can
        # consume any schema with no code change.
        json_keys: list[str] = []
        if "event_attributes" in tcols:
            try:
                json_keys = sorted(
                    {
                        str(r[0])
                        for r in con.execute(
                            f"SELECT DISTINCT unnest(json_keys(event_attributes)) AS k "
                            f"FROM {src} WHERE event_attributes IS NOT NULL"
                        ).fetchall()
                        if r[0]
                    }
                )
            except Exception:  # noqa: BLE001 - not JSON: fall through to columns
                json_keys = []
        candidates = sorted(
            (set(json_keys) | {c for c in tcols if c not in _BASE_COLS and c not in _META_COLS})
            - _META_COLS
        )
        base_present = [c for c in _BASE_COLS if c in tcols]
        select_parts = [f"{_quote_ident(c)}" for c in base_present]
        select_parts += [
            f"CAST({_quote_ident(c)} AS VARCHAR) AS {_quote_ident(c)}"
            if c in tcols
            else _json_expr(c)
            for c in candidates
        ]
        where = []
        if getattr(cfg, "as_of", None):
            try:
                where.append(f"epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {as_of_epoch(cfg.as_of)}")
            except ValueError as e:
                raise ValueError(f"--as-of must be an ISO date/datetime, got {cfg.as_of!r}") from e
        if hi_epoch is not None:
            where.append(f"epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {float(hi_epoch)}")
        if lo_epoch is not None:
            where.append(f"epoch(CAST(event_ts AS TIMESTAMPTZ)) > {float(lo_epoch)}")
        # Scale guard (speed/memory): a large stream (100M+ rows) cannot be
        # materialized. Push a DETERMINISTIC customer-hash sample into SQL so
        # ~read_mult x the training sample is read — the full stream lives in
        # DuckDB, only a bounded, seeded subset is loaded into memory.
        _sc = getattr(cfg, "sample_customers", None)
        wsql = (" WHERE " + " AND ".join(where)) if where else ""
        if _sc:
            try:
                _pop = con.execute(
                    f"SELECT count(distinct customer_key) FROM {src}{wsql}"
                ).fetchone()[0]
                if _pop and _pop > 3 * int(_sc):
                    _thr = max(1, int(round(3 * int(_sc) / _pop * 1_000_000)))
                    wsql = (
                        wsql + " AND " if wsql.strip() else " WHERE "
                    ) + f"hash(customer_key) % 1000000 < {_thr}"
                    print(
                        f"[read] stream has {_pop} customers -> sampled ~{3 * int(_sc)} "
                        f"(hash threshold {_thr}/1e6)",
                        flush=True,
                    )
            except Exception:
                pass  # fall through to a full read (small streams)
        df = con.execute(
            f"SELECT {', '.join(select_parts)} FROM {src}{wsql} ORDER BY customer_key, event_ts"
        ).pl()
    finally:
        if con is not None:
            con.close()

    numeric, categorical, skipped, field_types = _infer_field_types(df, candidates, cfg)
    if not update_fields:
        # Inference must honor the checkpoint's field types even when this narrow
        # window lacks enough non-null values to infer them.
        numeric = sorted(
            set(numeric) | (set(getattr(cfg, "numeric_fields", ()) or ()) & set(candidates))
        )
        categorical = sorted(
            (set(categorical) | (set(getattr(cfg, "attr_fields", ()) or ()) & set(candidates)))
            - set(numeric)
        )
    if skipped:
        df = df.drop([c for c in skipped if c not in numeric])
    if numeric:
        df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric])
    if update_fields:
        # Preserve fields pinned by a checkpoint/CLI that are absent from this
        # read (e.g. a narrow inference window) in their original bucket.
        prev_numeric = set(getattr(cfg, "numeric_fields", ()) or ())
        prev_categorical = set(getattr(cfg, "attr_fields", ()) or ())
        numeric = sorted(set(numeric) | (prev_numeric - set(categorical) - set(skipped)))
        categorical = sorted(set(categorical) | (prev_categorical - set(numeric) - set(skipped)))
        cfg.numeric_fields = tuple(numeric)
        cfg.attr_fields = tuple(categorical)
        cfg._field_receipt = {
            "numeric_fields": list(numeric),
            "categorical_fields": list(categorical),
            "skipped_nested_fields": list(skipped),
            "fields": field_types,
            "rule": "numeric iff every non-null value parses as a finite float",
        }
        print(
            f"[fields] numeric={len(numeric)} categorical={len(categorical)} "
            f"skipped_nested={len(skipped)}",
            flush=True,
        )
    return _cut_as_of(df, cfg)


def _cut_as_of(df, cfg: CFMConfig):
    """Point-in-time cutoff: keep only events with ts <= cfg.as_of (ISO).

    Applied to the RAW stream so every downstream step (keys, A/B split,
    anchors, sequences, products) is leak-free with respect to the cutoff.
    """
    if not getattr(cfg, "as_of", None):
        return df
    import polars as pl

    try:
        cut = as_of_epoch(cfg.as_of)
    except ValueError as e:
        raise ValueError(f"--as-of must be an ISO date/datetime, got {cfg.as_of!r}") from e
    return (
        df.with_columns(
            pl.col("event_ts")
            .str.to_datetime(time_zone="UTC", strict=False)
            .dt.epoch("s")
            .alias("_asof")
        )
        .filter(pl.col("_asof") <= cut)
        .drop("_asof")
    )


def read_stream_window(cfg: CFMConfig, lo_epoch: float | None, hi_epoch: float | None):
    """Read ONLY events in ``(lo_epoch, hi_epoch]`` (the inference window).

    Uses the same generic field discovery/selection as ``_read_stream`` (the
    time filter is pushed into SQL for DuckDB sources), but does NOT re-derive
    the field contract — inference must use the checkpoint's fields.
    """
    return _read_stream(cfg, lo_epoch=lo_epoch, hi_epoch=hi_epoch, update_fields=False)


def _customer_keys(df, cfg):
    keys = df["customer_key"].unique(maintain_order=True).to_list()
    if cfg.sample_customers is not None:
        keys = keys[: cfg.sample_customers]
    return keys


def assign_split(keys, cfg: CFMConfig) -> dict[str, str]:
    return {
        k: ("A" if (_h(k, cfg.split_seed) % 1000) < int(cfg.split_a_frac * 1000) else "B")
        for k in keys
    }


def draw_sample(keys, split, side, n, seed):
    """Deterministic SAMPLE drawn from Population {side} of the monthly split.

    The populations (assign_split) are disjoint by construction; a sample of A
    and a sample of B can therefore never overlap. Sample size is a compute/signal
    knob (the ladders pick it): as small as possible for speed, large enough for
    the training signal. Seeded, so the draw is reproducible and re-rolls with the
    month's split_seed.
    """
    pop = [k for k in keys if split.get(k) == side]
    if n is None or n >= len(pop):
        return pop
    rng = np.random.default_rng(_h(f"sample:{side}", int(seed)))
    idx = np.arange(len(pop))
    rng.shuffle(idx)
    keep = set(idx[: int(n)].tolist())
    return [k for i, k in enumerate(pop) if i in keep]


def _apply_data_revision(cfg: CFMConfig, df) -> str:
    cfg.revision, sig = AT.data_revision(df)
    cfg._data_signature = sig
    return sig


def _numeric_columns(df) -> list[str]:
    if df is None:
        return []
    return [
        c
        for c in df.columns
        if c not in _BASE_COLS and c not in _META_COLS and df[c].dtype.is_numeric()
    ]


def _primary_numeric_field(df, cfg: CFMConfig | None = None) -> str | None:
    """The stream's dominant extensive numeric field — DERIVED, never named.

    Used where a single monetary/quantity signal is needed (order event,
    counterfactual reward, successor-feature value). Chooses the numeric field
    with the largest total absolute mass; ties are broken by name (stable).
    """
    if df is None or df.height == 0:
        return None
    exclude = set(getattr(cfg, "action_fields", ()) or ()) | set(
        getattr(cfg, "effect_actions", ()) or ()
    )
    cands = [
        c
        for c in (getattr(cfg, "numeric_fields", ()) or ())
        if c in df.columns and c not in exclude
    ]
    cands = cands or [c for c in _numeric_columns(df) if c not in exclude]
    if not cands:
        return None
    score = {
        c: float(df[c].cast(pl.Float64, strict=False).fill_null(0.0).abs().sum()) for c in cands
    }
    return max(sorted(score), key=lambda c: score[c])


def _covariates(et, ts, company, is_action=None):
    """Vectorized exogenous action covariates per event:
    [is_action, log1p(seconds since previous action)].
    Actions are inputs the model must not predict; ``is_action`` may be supplied
    (schema-declared action fields) or defaults to event-type membership."""
    is_co = (
        np.asarray(is_action, dtype=bool) if is_action is not None else np.isin(et, list(company))
    )
    co_ts = np.where(is_co, ts, -np.inf)
    acc = np.maximum.accumulate(co_ts) if co_ts.size else co_ts
    last_prev = np.concatenate([[-np.inf], acc[:-1]]) if co_ts.size else np.zeros(0)
    gap = np.where(np.isfinite(last_prev), ts - last_prev, 0.0)
    return np.stack([is_co.astype(np.float64), np.log1p(np.clip(gap, 0, None))], axis=1)


def _random_anchor_epochs(ts, data_end, cfg, key):
    """n uniform-random anchor days over [history_start, data_end - 1d] —
    agnostic to any plugin's target window."""
    lo = float(ts[0])
    hi = float(data_end) - 86400.0
    if hi <= lo:
        return []
    n = max(1, int(cfg.n_anchors))
    rng = np.random.default_rng(_h(key, cfg.split_seed))
    return sorted(float(x) for x in rng.uniform(lo, hi, size=n))


def derive_order_event(df, cfg: CFMConfig | None = None) -> str | None:
    """The conversion/order event, DERIVED from the data (v4.6, #1): the event
    type with the highest TOTAL mass of the stream's dominant numeric field
    (sum, not mean — a rare reversal like a cancellation can have a high MEAN
    but is not the conversion). No field/business name is hardcoded; a stream
    with no numeric field returns None."""
    field = _primary_numeric_field(df, cfg)
    if field is None or "event_type" not in df.columns:
        return None
    agg = (
        df.group_by("event_type")
        .agg(pl.col(field).cast(pl.Float64, strict=False).fill_null(0.0).sum().alias("mv"))
        .sort("mv")
    )
    if agg.height == 0:
        return None
    return str(agg["event_type"][-1])


def _effect_targets(d, cfg):
    """Per-customer PERIOD-level counterfactual data: for each action period, the
    randomized action level per declared key and the realized reward.

    SELF-SUPERVISED (targets from the stream). This is the direct-method/outcome-model
    view of decision-sufficiency: a head `f(state, action)->reward` fitted on the
    randomized (action, reward) pairs must encode the customer's RESPONSE in `state`,
    else it cannot fit. Lower-variance and stronger than a per-action MEAN target.
    Generic: keys from the schema, levels/cadence from the data, reward = the stream's
    own dominant numeric signal (derived). Returns `(per_cust, levels)` where
    `per_cust[c] = (act (p,K) f32, rew (p,) f32)`, or `(None, None)` when off.
    """
    acts = [k for k in (getattr(cfg, "effect_actions", ()) or ()) if k in d.columns]
    # the internal time column must never be the primary numeric field (it was: the
    # seq's `value` became epochs, so the value head/level block saw timestamps).
    _d_no_ts = d.drop("_ts") if "_ts" in d.columns else d
    value_col = getattr(cfg, "primary_numeric", None) or _primary_numeric_field(_d_no_ts, cfg)
    if not acts or d.height == 0 or value_col is None:
        return None, None
    _ts = pl.col("event_ts").str.to_datetime(time_zone="UTC", strict=False).dt.epoch("s")
    d = d.with_columns(_ts.alias("_ts"))
    # Period = one action cadence ANCHORED AT THE STREAM ORIGIN (min ts). Anchoring at
    # unix-epoch week boundaries arbitrarily SPLITS the randomized periods (a send and
    # its reward land in different buckets), destroying the dose-response. Origin-
    # anchored buckets align with the stream's own cadence.
    _t0 = float(d["_ts"].min())
    d = d.with_columns(((_ts - _t0) // ACTION_PERIOD_S).cast(pl.Int64).alias("_period"))
    v = pl.col(value_col).cast(pl.Float64, strict=False).fill_null(0.0)
    rew = (
        d.filter(v != 0.0)
        .group_by(["customer_key", "_period"])
        .agg(pl.lit(1.0, dtype=pl.Float64).alias("_y"))
    )
    snd = (
        d.filter(pl.all_horizontal([pl.col(k).is_not_null() for k in acts]))
        .group_by(["customer_key", "_period"])
        .agg([pl.col(k).cast(pl.Float64, strict=False).last().alias(k) for k in acts])
    )
    j = snd.join(rew, on=["customer_key", "_period"], how="left").fill_null(0.0)
    # Canonical period order. Polars' group-by/join row order is NOT stable across
    # calls in one process (measured: the same build permuted a customer's action
    # rows), and these rows are consumed positionally as padded (P,K)/(P,) arrays.
    # Sorting by period makes the object a function of its inputs; the effect loss
    # sums BCE over periods, so the order is loss-neutral.
    j = j.sort(["customer_key", "_period"])
    levels = [int(float(j[k].max() or 0)) + 1 for k in acts]
    per_cust = {}
    for g in j.partition_by("customer_key", maintain_order=True):
        per_cust[g["customer_key"][0]] = (
            g.select(acts).to_numpy().astype(np.float32),
            g["_y"].to_numpy().astype(np.float32),
        )
    return per_cust, levels


def _action_flags(g, cfg: CFMConfig) -> np.ndarray:
    """Per-event boolean: is this an EXOGENOUS ACTION?

    True iff the event type is declared exogenous OR it carries any declared
    action field (non-null). Data-driven; no domain literals.
    """
    company = [str(x) for x in (getattr(cfg, "exogenous_events", ()) or ())]
    expr = pl.col("event_type").is_in(company)
    for f in getattr(cfg, "action_fields", ()) or ():
        if f in g.columns:
            expr = expr | pl.col(f).is_not_null()
    return g.select(expr.fill_null(False).alias("_is_action"))["_is_action"].to_numpy()


def _survival_horizons(d) -> list[float]:
    """Data-derived survival probe horizons (DEC-457), seconds, strictly increasing.

    An OCTAVE grid over the stream's own inter-event gap support (robust edges p1..p99.5):
    the number of horizons is the gap range's octaves (clamped to a documented sanity range
    3..9), so the grid probes both the ordinary-gap bulk and the multi-day SILENCE tail the
    churn judgment lives in — a rank-quantile grid clustered on the bulk and missed the tail
    (measured on the demo: every horizon < 1.2 days). Every number is a fact about the stream.
    """
    import numpy as np

    pairs = d.select("customer_key", "_ts").sort(["customer_key", "_ts"])
    gaps = (
        pairs.with_columns(g=(pl.col("_ts") - pl.col("_ts").shift(1)).over("customer_key"))
        .filter(pl.col("g") > 0)["g"]
        .to_numpy()
    )
    if gaps.size < 32:  # principle-ok: below this the quantiles are not statistics
        return []
    lo, hi = float(np.percentile(gaps, 1)), float(np.percentile(gaps, 99.5))
    if lo <= 0 or hi <= lo:
        return []
    K = int(min(9, max(3, round(np.log2(hi / lo)) + 1)))  # octaves; clamp documented above
    return [float(x) for x in np.geomspace(lo, hi, K)]


def _build_sequences_impl(df, keys, cfg: CFMConfig, split, with_anchors: bool, min_events: int = 3):
    """Polars-first: partition once in Rust, then slice per group.

    ALL events are tokens (including exogenous action events — their identity is
    an INPUT channel). ``is_action`` rides along per event so the self-supervised
    losses can avoid predicting action targets. ``min_events`` gates who gets a
    sequence at all (training needs >= 3 events to learn from; STATE rows pass 1).
    """
    # sorted, not set-ordered: today `want` only feeds `is_in` (order-independent), but a
    # set's iteration order is per-process hash randomization — any future consumer reading
    # it positionally would inherit nondeterminism. (DEC-446; the behavior-hash cost of the
    # landmine DEC-438 recorded.)
    want = sorted(set(keys))
    d = df.filter(pl.col("customer_key").is_in(want)).with_columns(
        pl.col("event_ts").str.to_datetime(time_zone="UTC", strict=False).dt.epoch("s").alias("_ts")
    )
    data_end = float(d["_ts"].max()) if d.height else 0.0
    company = set(map(str, getattr(cfg, "exogenous_events", ())))
    min_events = max(1, int(min_events))
    # the internal time column must never be the primary numeric field (it was: the
    # seq's `value` became epochs, so the value head/level block saw timestamps).
    _d_no_ts = d.drop("_ts") if "_ts" in d.columns else d
    value_col = getattr(cfg, "primary_numeric", None) or _primary_numeric_field(_d_no_ts, cfg)
    # per-customer counterfactual period data (cached: cheap if off, computed once)
    if getattr(cfg, "_effect_cache", None) is None:
        _per, _levels = _effect_targets(df, cfg)
        cfg._effect_cache = (_per, _levels)
        cfg._effect_levels = list(_levels or [])
        cfg._effect_dim = int(sum(_levels or []))
    # DEC-457: survival probe horizons, derived from the stream's own gaps (cached once)
    if getattr(cfg, "survival", False) and getattr(cfg, "_survival_cache", None) is None:
        cfg._survival_cache = _survival_horizons(d)
        cfg._survival_horizons_s = list(cfg._survival_cache)
    # DEC-465: per-field level counts for the action-profile objective (cached once). Levels
    # are facts about the stream (max observed level + 1), never typed.
    if (
        getattr(cfg, "action_profile", False)
        and getattr(cfg, "action_profile_fields", ())
        and getattr(cfg, "_profile_cache", None) is None
    ):
        levels: dict[str, int] = {}
        for f in cfg.action_profile_fields:
            if f in d.columns:
                try:
                    mx = d[f].cast(pl.Float64, strict=False).max()
                except Exception:  # noqa: BLE001 - non-numeric field: not profile-able
                    continue
                if mx is not None and float(mx) >= 0:
                    levels[str(f)] = int(float(mx)) + 1
        cfg._profile_cache = levels
        cfg._action_profile_levels = levels
        cfg._action_profile_dims = [levels[f] for f in cfg.action_profile_fields if f in levels]
    eff_map, _ = cfg._effect_cache
    attr_fields = tuple(getattr(cfg, "attr_fields", ()) or ())
    numeric_fields = tuple(getattr(cfg, "numeric_fields", ()) or ())
    _af = [f for f in (getattr(cfg, "action_fields", ()) or ())]
    _n_events = 0
    _n_actions = 0
    _n_ambiguous = 0
    seqs = []
    for g in d.partition_by("customer_key", maintain_order=True):
        k = g["customer_key"][0]
        ts_full = g["_ts"].to_numpy()
        # numpy arrays (Polars->numpy is vectorized in Rust); no per-row Python.
        et_arr = g["event_type"].to_numpy()
        is_action_full = _action_flags(g, cfg)
        _n_events += int(et_arr.size)
        _n_actions += int(is_action_full.sum())
        if _af:
            _exprs = [pl.col(f).is_not_null() for f in _af if f in g.columns]
            if _exprs:
                _any = pl.any_horizontal(_exprs)
                _all = pl.all_horizontal(_exprs)
                _n_ambiguous += int(g.select((_any & ~_all).alias("_amb"))["_amb"].sum())
        # actions are INPUT covariates too, but never the only representation.
        co_full = _covariates(et_arr, ts_full, company, is_action=is_action_full)
        n_full = et_arr.size
        if n_full < min_events:
            continue

        if value_col is not None and value_col in g.columns:
            val_full = g[value_col].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
        else:
            val_full = np.zeros(n_full)
        attrs = {
            c: (g[c].to_numpy() if c in g.columns else np.full(n_full, None, dtype=object))
            for c in attr_fields
        }
        # generic NUMERIC fields -> raw scalars (+ a null mask so missing != 0)
        numeric = {
            c: (
                g[c].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
                if c in g.columns
                else np.zeros(n_full)
            )
            for c in numeric_fields
        }
        numeric_mask = {
            c: (g[c].is_not_null().to_numpy() if c in g.columns else np.zeros(n_full, dtype=bool))
            for c in numeric_fields
        }
        ets = g["event_ts"].to_numpy()
        ts = ts_full
        n = n_full
        spans: list[tuple[int, float | None]] = []
        if with_anchors and split.get(k) == "B":
            for a in _random_anchor_epochs(ts, data_end, cfg, k):
                end = int(np.searchsorted(ts, a, side="right"))
                if end >= min_events:
                    spans.append((end, a))
        else:
            spans.append((n, None))
        for end, anchor_epoch in spans:
            start = max(0, end - cfg.seq_len)
            if end - start >= min_events:
                seqs.append(
                    {
                        "customer": k,
                        "group": split.get(k, "A"),
                        "anchor_epoch": anchor_epoch,
                        "ref_epoch": (
                            float(anchor_epoch) if anchor_epoch is not None else float(data_end)
                        ),
                        "event_type": et_arr[start:end],
                        "attrs": {c: v[start:end] for c, v in attrs.items()},
                        "numeric": {c: v[start:end] for c, v in numeric.items()},
                        "numeric_mask": {c: v[start:end] for c, v in numeric_mask.items()},
                        "is_action": is_action_full[start:end],
                        "value": val_full[start:end],
                        "primary_numeric": value_col,
                        "event_ts": ets[start:end],
                        "ts": ts[start:end].tolist(),
                        "co": co_full[start:end].tolist(),
                        "eff_act": (eff_map[k][0] if eff_map and k in eff_map else None),
                        "eff_rew": (eff_map[k][1] if eff_map and k in eff_map else None),
                    }
                )
    cfg._action_receipt = {
        "exogenous_events": sorted(company),
        "action_fields": list(_af),
        "rule": "event_type in exogenous_events OR any action_field non-null",
        "n_events": int(_n_events),
        "n_actions": int(_n_actions),
        "action_frac": round(_n_actions / max(_n_events, 1), 6),
        # partial action-field presence: detected as an action (fail safe), flagged.
        "n_ambiguous": int(_n_ambiguous),
    }
    return seqs


# ---------------------------------------------------------------------------
# opt-in content-addressed cache for sequence construction (DEC-669 G(iii))
# ---------------------------------------------------------------------------
# Admissible ONLY on a byte-identical proof: a hit returns the exact object a
# rebuild would produce (including the cfg-side derived state), and any changed
# input yields a different key. Default OFF — with WONDERLAND_SEQ_CACHE unset the
# public build_sequences executes the historical path and touches no cache code.
_SEQ_CACHE_ENV = "WONDERLAND_SEQ_CACHE"
_SEQ_CACHE_DIR_ENV = "WONDERLAND_SEQ_CACHE_DIR"
# Output-shape contract: bump when the sequence dict's shape changes. Edits to the
# constructors below invalidate automatically via the source digest (no hand-bump).
_SEQ_CACHE_SCHEMA = "build_sequences/v1"
_SEQ_CACHE_SOURCE_FNS = (
    "_build_sequences_impl",
    "_random_anchor_epochs",
    "_covariates",
    "_action_flags",
    "_primary_numeric_field",
    "_numeric_columns",
    "_effect_targets",
    "_h",
)
# cfg attributes build_sequences derives and downstream steps read; a hit restores
# them so the cached call is indistinguishable from a rebuild (not just in `seqs`).
_SEQ_CFG_STATE = (
    "_effect_cache",
    "_effect_levels",
    "_effect_dim",
    "_survival_cache",
    "_survival_horizons_s",
    "_profile_cache",
    "_action_profile_levels",
    "_action_profile_dims",
    "_action_receipt",
)
# The last cache decision (a receipt): {"status": ..., "key": ..., ...}. Empty on
# the default path — no cache code has run.
_SEQ_CACHE_LAST: dict = {}


def _seq_cache_enabled() -> bool:
    return os.environ.get(_SEQ_CACHE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _seq_cache_receipt(**kw) -> None:
    _SEQ_CACHE_LAST.clear()
    _SEQ_CACHE_LAST.update(kw)


def _data_digest(df) -> str | None:
    """Content digest of the frame build_sequences consumes (order-sensitive).

    Two independent 64-bit row-hash seeds plus shape/column/dtype identity: a
    collision would require ~2^128 agreement. Unhashable frames return None, which
    disables the cache for that call (fail safe, never a guess).
    """
    try:
        h = hashlib.sha256()
        h.update(
            repr(
                (
                    int(df.height),
                    int(df.width),
                    tuple(df.columns),
                    tuple(str(d) for d in df.dtypes),
                )
            ).encode()
        )
        for seed in (0, 1):
            rows = np.ascontiguousarray(df.hash_rows(seed=seed).to_numpy())
            h.update(rows.tobytes())
        return h.hexdigest()
    except Exception:  # noqa: BLE001 - cannot fingerprint: run uncached
        return None


def _seq_source_digest() -> str | None:
    """Digest of the constructors' source — a code edit invalidates every entry."""
    import inspect

    try:
        h = hashlib.sha256()
        for name in _SEQ_CACHE_SOURCE_FNS:
            h.update(inspect.getsource(globals()[name]).encode())
        return h.hexdigest()
    except Exception:  # noqa: BLE001 - no source (frozen/exec'd): run uncached
        return None


def _seq_cache_key(df, want, cfg: CFMConfig, split, with_anchors, min_events):
    """Content address for one build_sequences output: ``(key, fields)`` or None.

    Every construction input is in the key: the frame's content digest, the source
    stream's path/size/mtime (a file swap changes the address even if a caller
    passed an identical frame), ``as_of``, the sorted customer set and its split,
    ``seq_len``, ``n_anchors``, ``split_seed``, ``with_anchors``, ``min_events``,
    every cfg field the constructors read, the module constants they reference,
    and the numpy/polars versions. The function schema version and the
    constructors' source digest pin the builder's identity.
    """
    data_digest = _data_digest(df)
    source_digest = _seq_source_digest()
    if data_digest is None or source_digest is None:
        return None
    stream = {"path": str(getattr(cfg, "db", "") or "")}
    try:
        st = os.stat(stream["path"])
        stream.update({"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    except OSError:
        stream["stat"] = "missing"
    names = [str(k) for k in want]
    key_fields = {
        "schema": _SEQ_CACHE_SCHEMA,
        "source": source_digest,
        # Module constants the constructors read at call time are not in the source
        # digest (a function body references the name, not its value): key them.
        "constants": {
            "ACTION_PERIOD_S": float(ACTION_PERIOD_S),
            "BASE_COLS": list(_BASE_COLS),
            "META_COLS": sorted(_META_COLS),
        },
        # RNG stream (anchors) and row-hash semantics (the data digest) are library
        # properties; a version change invalidates entries rather than risk a drift.
        "runtime": {"numpy": np.__version__, "polars": pl.__version__},
        "data": data_digest,
        "stream": stream,
        "as_of": None if getattr(cfg, "as_of", None) is None else str(cfg.as_of),
        "keys": hashlib.sha256("\x1f".join(names).encode()).hexdigest(),
        "n_keys": len(names),
        "split": hashlib.sha256(
            "\x1f".join(f"{k}\x1e{split.get(k, '')}" for k in want).encode()
        ).hexdigest(),
        "seq_len": int(cfg.seq_len),
        "n_anchors": int(cfg.n_anchors),
        "seed": int(cfg.split_seed),
        "with_anchors": bool(with_anchors),
        "min_events": int(min_events),
        "primary_numeric": None
        if getattr(cfg, "primary_numeric", None) is None
        else str(cfg.primary_numeric),
        "attr_fields": [str(x) for x in (getattr(cfg, "attr_fields", ()) or ())],
        "numeric_fields": [str(x) for x in (getattr(cfg, "numeric_fields", ()) or ())],
        "action_fields": [str(x) for x in (getattr(cfg, "action_fields", ()) or ())],
        "effect_actions": [str(x) for x in (getattr(cfg, "effect_actions", ()) or ())],
        "exogenous_events": sorted(str(x) for x in (getattr(cfg, "exogenous_events", ()) or ())),
        "survival": bool(getattr(cfg, "survival", False)),
        "action_profile": bool(getattr(cfg, "action_profile", False)),
        "action_profile_fields": [
            str(x) for x in (getattr(cfg, "action_profile_fields", ()) or ())
        ],
    }
    blob = json.dumps(key_fields, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest(), key_fields


def _seq_cache_dir() -> Path | None:
    """The per-uid cache dir, or None when it cannot be trusted.

    A pickled object is code execution, so the dir must be owned by this uid and
    not group/world-writable. Untrusted/unusable location = cache OFF (fail safe).
    """
    raw = os.environ.get(_SEQ_CACHE_DIR_ENV)
    if raw:
        d = Path(raw).expanduser()
    else:
        uid = str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()
        d = Path(tempfile.gettempdir()) / f"wonderland-seq-cache-{uid}"
    try:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        st = d.stat()
    except OSError:
        return None
    if not d.is_dir():
        return None
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        return None
    if st.st_mode & 0o022:
        return None
    return d


def _seq_cache_load(path: Path, key: str):
    """``(seqs, cfg_state)`` for a valid entry; None on any problem (a MISS)."""
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _SEQ_CACHE_SCHEMA
            or payload.get("key") != key
        ):
            return None
        seqs = payload.get("seqs")
        state = payload.get("cfg_state")
        if not isinstance(seqs, list) or not isinstance(state, dict):
            return None
        return seqs, state
    except Exception:  # noqa: BLE001 - corrupt/unreadable entry: rebuild, never raise
        return None


def _seq_cache_store(path: Path, key: str, seqs, cfg: CFMConfig) -> str | None:
    """Atomically persist one entry. Returns an error string, or None on success."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        state = {a: getattr(cfg, a) for a in _SEQ_CFG_STATE if hasattr(cfg, a)}
        payload = {"schema": _SEQ_CACHE_SCHEMA, "key": key, "seqs": seqs, "cfg_state": state}
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return None
    except Exception as e:  # noqa: BLE001 - a cache write failure must not fail a run
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return f"{type(e).__name__}: {e}"


def _seq_restore_cfg(cfg: CFMConfig, state: dict) -> None:
    for a in _SEQ_CFG_STATE:
        if a in state:
            setattr(cfg, a, state[a])
        elif hasattr(cfg, a):
            try:
                delattr(cfg, a)
            except AttributeError:
                pass


def _seq_preset_matches(cfg: CFMConfig, state: dict) -> bool:
    """True iff every derived attr already on ``cfg`` equals the cached cold value.

    A caller-supplied derived cache changes the output, so it is not enough to key
    the frame; the object is reusable only when the caller's state IS the state the
    cold build would have written. Byte comparison is order-sensitive (conservative).
    """
    for a in _SEQ_CFG_STATE:
        if not hasattr(cfg, a):
            continue
        if a not in state:
            return False
        try:
            if pickle.dumps(getattr(cfg, a), protocol=5) != pickle.dumps(state[a], protocol=5):
                return False
        except Exception:  # noqa: BLE001 - uncomparable state: miss, never guess
            return False
    return True


def _build_sequences_cached(df, keys, cfg: CFMConfig, split, with_anchors, min_events):
    t0 = time.perf_counter()

    def _ms() -> float:
        return round((time.perf_counter() - t0) * 1000.0, 1)

    keys = list(keys)
    want = sorted(set(keys))
    preset = [a for a in _SEQ_CFG_STATE if hasattr(cfg, a)]
    keyed = _seq_cache_key(df, want, cfg, split, with_anchors, min_events)
    if keyed is None:
        _seq_cache_receipt(status="bypass_unkeyable", elapsed_ms=_ms())
        return _build_sequences_impl(df, keys, cfg, split, with_anchors, min_events)
    key, key_fields = keyed
    d = _seq_cache_dir()
    if d is None:
        _seq_cache_receipt(
            status="bypass_unsafe_dir", key=key, key_fields=key_fields, elapsed_ms=_ms()
        )
        return _build_sequences_impl(df, keys, cfg, split, with_anchors, min_events)
    path = d / f"{key}.pkl"
    existed = path.exists()
    loaded = _seq_cache_load(path, key)
    if loaded is not None:
        seqs, state = loaded
        if preset and not _seq_preset_matches(cfg, state):
            # The caller's derived state is not the cold one this entry encodes:
            # rebuild from its state, but do not overwrite the cold entry.
            _seq_cache_receipt(status="bypass_preset", preset=preset, elapsed_ms=_ms())
            return _build_sequences_impl(df, keys, cfg, split, with_anchors, min_events)
        _seq_restore_cfg(cfg, state)
        _seq_cache_receipt(
            status="hit",
            key=key,
            path=str(path),
            n_seqs=len(seqs),
            key_fields=key_fields,
            elapsed_ms=_ms(),
        )
        print(f"[seq-cache] hit {key[:12]} ({len(seqs)} seqs) <- {path}", flush=True)
        return seqs
    if preset:
        # A cold entry for this frame does not exist yet, and this call's output
        # depends on un-keyed caller state: rebuild without storing (a later cold
        # call owns the entry).
        _seq_cache_receipt(status="bypass_preset", preset=preset, elapsed_ms=_ms())
        return _build_sequences_impl(df, keys, cfg, split, with_anchors, min_events)
    seqs = _build_sequences_impl(df, keys, cfg, split, with_anchors, min_events)
    err = _seq_cache_store(path, key, seqs, cfg)
    status = "miss_corrupt" if existed else "miss"
    _seq_cache_receipt(
        status=status,
        key=key,
        path=str(path),
        n_seqs=len(seqs),
        stored=err is None,
        store_error=err,
        key_fields=key_fields,
        elapsed_ms=_ms(),
    )
    print(
        f"[seq-cache] {status} {key[:12]} ({len(seqs)} seqs) -> {path}"
        + (f" [store failed: {err}]" if err else ""),
        flush=True,
    )
    return seqs


def build_sequences(df, keys, cfg: CFMConfig, split, with_anchors: bool, min_events: int = 3):
    """Polars-first: partition once in Rust, then slice per group.

    ALL events are tokens (including exogenous action events — their identity is
    an INPUT channel). ``is_action`` rides along per event so the self-supervised
    losses can avoid predicting action targets. ``min_events`` gates who gets a
    sequence at all (training needs >= 3 events to learn from; STATE rows pass 1).

    OPT-IN cache (DEC-669, default OFF): ``WONDERLAND_SEQ_CACHE=1`` serves a
    content-addressed, byte-identical pickled result under
    ``WONDERLAND_SEQ_CACHE_DIR`` (default: a per-uid dir under the system temp).
    Unset, this function is the historical path and executes no cache code.
    """
    if _seq_cache_enabled():
        return _build_sequences_cached(df, keys, cfg, split, with_anchors, min_events)
    return _build_sequences_impl(df, keys, cfg, split, with_anchors, min_events)
