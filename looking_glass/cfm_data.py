"""CFM data prep: stream reads, splits, covariates, and sequence building."""

from __future__ import annotations

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
    value_col = _primary_numeric_field(d, cfg)
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


def build_sequences(df, keys, cfg: CFMConfig, split, with_anchors: bool, min_events: int = 3):
    """Polars-first: partition once in Rust, then slice per group.

    ALL events are tokens (including exogenous action events — their identity is
    an INPUT channel). ``is_action`` rides along per event so the self-supervised
    losses can avoid predicting action targets. ``min_events`` gates who gets a
    sequence at all (training needs >= 3 events to learn from; STATE rows pass 1).
    """
    want = list(set(keys))
    d = df.filter(pl.col("customer_key").is_in(want)).with_columns(
        pl.col("event_ts").str.to_datetime(time_zone="UTC", strict=False).dt.epoch("s").alias("_ts")
    )
    data_end = float(d["_ts"].max()) if d.height else 0.0
    company = set(map(str, getattr(cfg, "exogenous_events", ())))
    min_events = max(1, int(min_events))
    value_col = _primary_numeric_field(d, cfg)
    # per-customer counterfactual period data (cached: cheap if off, computed once)
    if getattr(cfg, "_effect_cache", None) is None:
        _per, _levels = _effect_targets(df, cfg)
        cfg._effect_cache = (_per, _levels)
        cfg._effect_levels = list(_levels or [])
        cfg._effect_dim = int(sum(_levels or []))
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
