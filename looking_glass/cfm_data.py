"""CFM data prep: stream reads, splits, covariates, and sequence building."""

from __future__ import annotations

import numpy as np
import polars as pl

from looking_glass.cfm_config import AT, CFMConfig, _h, as_of_epoch


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _read_stream(cfg: CFMConfig):
    cols = (
        "customer_key",
        "event_ts",
        "brand",
        "event_type",
        "event_attributes",
        "entity_type",
        "entity_id",
        "value",
    )
    if str(cfg.db).endswith((".arrow", ".feather", ".ipc")):
        import polars as pl

        df = (
            pl.read_ipc(cfg.db, memory_map=True)
            .select(list(cols))
            .sort(["customer_key", "event_ts"])
        )
    else:
        import duckdb

        con = duckdb.connect(cfg.db, read_only=True)
        try:
            # push the point-in-time cut into SQL (before materializing): the
            # polars filter below is now a no-op for duckdb sources, and loads
            # stop growing with the future tail of the stream.
            where = ""
            if getattr(cfg, "as_of", None):
                try:
                    where = (
                        f" WHERE epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {as_of_epoch(cfg.as_of)}"
                    )
                except ValueError as e:
                    raise ValueError(
                        f"--as-of must be an ISO date/datetime, got {cfg.as_of!r}"
                    ) from e
            df = con.execute(
                f"SELECT {', '.join(cols)} FROM {cfg.table}{where} ORDER BY customer_key, event_ts"
            ).pl()
        finally:
            con.close()
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

    DuckDB sources push the time filter into SQL so live inference never
    materialises the full stream; IPC sources read then filter in Polars.
    """
    import polars as pl

    cols = (
        "customer_key",
        "event_ts",
        "brand",
        "event_type",
        "event_attributes",
        "entity_type",
        "entity_id",
        "value",
    )
    if str(cfg.db).endswith((".arrow", ".feather", ".ipc")):
        df = _read_stream(cfg)
        if hi_epoch is not None:
            df = df.filter(
                pl.col("event_ts").str.to_datetime(time_zone="UTC", strict=False).dt.epoch("s")
                <= float(hi_epoch)
            )
        if lo_epoch is not None:
            df = df.filter(
                pl.col("event_ts").str.to_datetime(time_zone="UTC", strict=False).dt.epoch("s")
                > float(lo_epoch)
            )
        return df
    import duckdb

    where = []
    if hi_epoch is not None:
        where.append(f"CAST(event_ts AS TIMESTAMPTZ) <= to_timestamp({float(hi_epoch)})")
    if lo_epoch is not None:
        where.append(f"CAST(event_ts AS TIMESTAMPTZ) > to_timestamp({float(lo_epoch)})")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    con = duckdb.connect(cfg.db, read_only=True)
    try:
        return con.execute(
            f"SELECT {', '.join(cols)} FROM {cfg.table} {w} ORDER BY customer_key, event_ts"
        ).pl()
    finally:
        con.close()


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


def _covariates(et, ts, company):
    """Vectorized exogenous company-action covariates per event:
    [is_company_action, log1p(seconds since previous company action)].
    Company actions are inputs the model must not predict, not tokens."""
    is_co = np.isin(et, list(company))
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


def build_sequences(df, keys, cfg: CFMConfig, split, with_anchors: bool, min_events: int = 3):
    """Polars-first: partition once in Rust, then slice per group. Anchors are
    random uniform days; company actions ride along as exogenous covariates.

    ``min_events`` gates who gets a sequence at all: training (self-supervised
    spans, anchor spans) needs >= 3 events to learn from; STATE rows pass 1 —
    the blanket 3 silently left every 1-2 event customer with no state (20% of
    a real long-tail stream; rabbit_hole/Instacart are dense enough to hide it).
    """
    want = list(set(keys))
    d = df.filter(pl.col("customer_key").is_in(want)).with_columns(
        pl.col("event_ts").str.to_datetime(time_zone="UTC", strict=False).dt.epoch("s").alias("_ts")
    )
    data_end = float(d["_ts"].max()) if d.height else 0.0
    company = set(map(str, cfg.company_actions))
    min_events = max(1, int(min_events))
    seqs = []
    for g in d.partition_by("customer_key", maintain_order=True):
        k = g["customer_key"][0]
        ts_full = g["_ts"].to_numpy()
        # numpy arrays (Polars->numpy is vectorized in Rust); no per-row Python.
        et_arr = g["event_type"].to_numpy()
        val_full = g["value"].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
        # Company actions are exogenous: NOT tokens. Covariates are computed on
        # the full stream (sends visible) while tokens keep only customer events.
        co_full = _covariates(et_arr, ts_full, company)
        ki = np.flatnonzero(~np.isin(et_arr, list(company)))
        if ki.size < min_events:
            continue
        ts = ts_full[ki]
        et = et_arr[ki]
        co = co_full[ki]
        val = val_full[ki]
        brand = g["brand"].to_numpy()[ki]
        ent = g["entity_type"].to_numpy()[ki]
        eid = g["entity_id"].to_numpy()[ki]
        ets = g["event_ts"].to_numpy()[ki]
        n = ts.size
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
                        "event_type": et[start:end],
                        "brand": brand[start:end],
                        "entity_type": ent[start:end],
                        "entity_id": eid[start:end],
                        "value": val[start:end],
                        "event_ts": ets[start:end],
                        "ts": ts[start:end].tolist(),
                        "co": co[start:end].tolist(),
                    }
                )
    return seqs
