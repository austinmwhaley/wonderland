"""PATH A (traditional supervised wide table) builder — single source of truth.

The generator and the GPU process boundary both call `build_path_a` so the feature
and label definitions live in exactly one place. `engine="gpu"` routes the
aggregation through the polars-CUDF `pl.GPUEngine` (imported inside the dedicated
`.venv-cudf` process); `engine=None` is CPU polars. The caller records which engine
actually ran in the runtime receipt.

Provenance (DEC-446): the feature/label columns are built from the expression lists
below, so the SAME lists answer "which cut of the stream produced this column" —
`path_a_provenance(as_of, customer_cols)` derives `{column: {source, cut_at}}` from
the builders themselves (never a hand-typed inventory, principle 6). `build_path_a`
refuses to return a table with an unstamped column: a fact about the comparator that
is decoration is worse than one that is missing (principles 7 and 11).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

CATEGORIES = ["apparel", "footwear", "accessories", "beauty", "home", "outdoor"]
CHANNEL_TOUCH = ["web", "app", "store"]
HORIZONS = (30, 90)


def _collect(lf, engine):
    if engine == "gpu":
        import cudf_polars  # noqa: F401  (registers the CUDA engine)

        return lf.collect(engine=pl.GPUEngine(raise_on_fail=False))
    return lf.collect()


# ---- expression lists: THE definition of every builder-made column (principle 6) ----------


def _hist_aggs(as_of: datetime) -> list[pl.Expr]:
    """Per-customer aggregates over `hist` only (events with ts <= as_of)."""
    purchase = pl.col("event_type") == "purchase"
    cat_feats = [(pl.col("category_code") == c).sum().alias(f"hist_cat_{c}") for c in CATEGORIES]
    chan_feats = [(pl.col("channel") == c).sum().alias(f"hist_channel_{c}") for c in CHANNEL_TOUCH]
    return [
        pl.len().alias("n_events"),
        purchase.sum().alias("n_orders"),
        (pl.col("event_type") == "return").sum().alias("n_returns"),
        (pl.col("event_type") == "support_ticket").sum().alias("n_support"),
        (pl.col("event_type") == "email_send").sum().alias("n_sends"),
        (pl.col("event_type") == "view").sum().alias("n_views"),
        (pl.col("event_type") == "add_to_cart").sum().alias("n_carts"),
        (pl.col("event_type") == "search").sum().alias("n_searches"),
        pl.col("value").filter(purchase).sum().alias("purchase_value"),
        pl.col("margin").filter(purchase).sum().alias("purchase_margin"),
        pl.col("value").filter(pl.col("event_type") == "return").sum().alias("return_value"),
        pl.col("value")
        .filter(purchase & (pl.col("ts_dt") > pl.lit(as_of - timedelta(days=30))))
        .sum()
        .alias("value_30d"),
        pl.col("value")
        .filter(purchase & (pl.col("ts_dt") > pl.lit(as_of - timedelta(days=90))))
        .sum()
        .alias("value_90d"),
        pl.col("product_id").n_unique().alias("n_products"),
        pl.col("channel").n_unique().alias("n_channels"),
        pl.col("campaign_id").n_unique().alias("n_campaigns"),
        pl.col("discount").mean().alias("discount_mean"),
        pl.col("qty").sum().alias("qty_sum"),
        pl.col("ts_dt").max().alias("last_event_ts"),
        pl.col("ts_dt").min().alias("first_event_ts"),
        pl.col("ts_dt").filter(purchase).max().alias("last_purchase_ts"),
        *cat_feats,
        *chan_feats,
    ]


def _hist_derived(as_of: datetime) -> list[pl.Expr]:
    """Ratios/recency formed from `hist` aggregates (still pre-`as_of` by construction)."""
    return [
        ((pl.lit(as_of) - pl.col("last_event_ts")).dt.total_days()).alias("recency_days"),
        ((pl.lit(as_of) - pl.col("last_purchase_ts")).dt.total_days()).alias(
            "days_since_last_purchase"
        ),
        (pl.col("purchase_value") / pl.col("n_orders").clip(1, None)).alias("avg_order_value"),
        (pl.col("n_returns") / pl.col("n_orders").clip(1, None)).alias("return_rate"),
        (pl.col("purchase_value") + pl.col("return_value")).alias("net_value"),
    ]


def _label_aggs(h: int) -> list[pl.Expr]:
    purchase = pl.col("event_type") == "purchase"
    return [
        purchase.sum().alias(f"y_orders_{h}"),
        pl.col("value").filter(purchase).sum().alias(f"y_value_{h}"),
        pl.col("margin").filter(purchase).sum().alias(f"y_margin_{h}"),
        pl.col("ts_dt").filter(purchase).min().alias("_first"),
    ]


def _label_first_expr(h: int, as_of: datetime) -> list[pl.Expr]:
    return [((pl.col("_first") - pl.lit(as_of)).dt.total_days()).alias(f"y_next_days_{h}")]


def _label_finalize(h: int) -> list[pl.Expr]:
    """Applied to the JOINED labels frame: customers with no horizon events are absent
    from the agg and must still fill (fill before the join would silently skip them)."""
    return [
        pl.col(f"y_orders_{h}").fill_null(0),
        pl.col(f"y_value_{h}").fill_null(0.0),
        pl.col(f"y_margin_{h}").fill_null(0.0),
        pl.when(pl.col(f"y_orders_{h}").fill_null(0) == 0)
        .then(h + 1)
        .otherwise(pl.col(f"y_next_days_{h}"))
        .alias(f"y_next_days_{h}"),
        (pl.col(f"y_orders_{h}").fill_null(0) == 0).cast(pl.Int8).alias(f"y_churn_{h}"),
    ]


def _out_names(exprs: list[pl.Expr]) -> list[str]:
    return [e.meta.output_name() for e in exprs]


def path_a_provenance(as_of: datetime, customer_cols=()) -> dict:
    """`{column: {source, cut_at}}` for every column the builder makes, derived from
    the expression lists themselves — the producer's own knowledge, not an observer's guess.

    `source` is `hist` (aggregated from events with ts <= as_of), `fut` (the label horizon
    window after as_of), or `customers` (static table joined in, stamped by the caller's
    schema). The audit (scripts/measure_path_a_provenance.py) turns this into the
    fair-comparator fact: every column `table_features` selects must be `hist|customers`.
    """
    prov = {}
    cut = as_of.isoformat()
    for site in (_hist_aggs(as_of), _hist_derived(as_of)):
        for name in _out_names(site):
            prov[name] = {"source": "hist", "cut_at": cut}
    for h in HORIZONS:
        future_cut = (as_of + timedelta(days=h)).isoformat()
        names = (
            _out_names(_label_aggs(h))
            + _out_names(_label_first_expr(h, as_of))
            + _out_names(_label_finalize(h))
        )
        for name in names:
            if name.startswith("_"):
                continue  # internal helper column, dropped before output
            prov[name] = {"source": "fut", "cut_at": future_cut}
    # columns formed after the expression lists (signup math + a numpy pass over hist counts)
    prov["tenure_days"] = {"source": "customers", "cut_at": cut}
    prov["purchase_rate"] = {"source": "hist", "cut_at": cut}
    prov["category_entropy"] = {"source": "hist", "cut_at": cut}
    for c in customer_cols:
        if c != "customer_key":
            prov[c] = {"source": "customers", "cut_at": None}
    return prov


def build_path_a(base: pl.DataFrame, customers: pl.DataFrame, as_of: datetime, engine=None):
    """Return (features, labels, wide, engine_state) for the point-in-time split."""
    engine_state = {"requested": engine, "used": "cpu", "note": ""}
    if engine == "gpu":
        try:
            import cudf_polars  # noqa: F401

            engine_state["used"] = "gpu"
        except Exception as exc:  # noqa: BLE001 - environment lacks RAPIDS
            engine_state["used"] = "cpu"
            engine_state["note"] = f"cudf_polars unavailable: {type(exc).__name__}: {exc}"
            engine = None

    prov = path_a_provenance(as_of, customer_cols=customers.columns)

    ev = base.with_columns(pl.col("event_ts").str.to_datetime().alias("ts_dt"))
    hist = ev.filter(pl.col("ts_dt") <= pl.lit(as_of))
    fut = ev.filter(pl.col("ts_dt") > pl.lit(as_of))

    per_cust_lf = (
        hist.lazy()
        .group_by("customer_key")
        .agg(_hist_aggs(as_of))
        .with_columns(_hist_derived(as_of))
        .drop(["last_event_ts", "first_event_ts", "last_purchase_ts"])
        .with_columns(pl.col("days_since_last_purchase").fill_null(999))
    )
    per_cust = _collect(per_cust_lf, engine)

    features = customers.join(per_cust, on="customer_key", how="left").fill_null(0.0)
    features = (
        features.with_columns(pl.col("signup_ts").str.to_datetime().alias("_signup_dt"))
        .with_columns(
            ((pl.lit(as_of) - pl.col("_signup_dt")).dt.total_days()).alias("tenure_days"),
        )
        .drop("_signup_dt")
        .with_columns(
            (pl.col("n_orders") / pl.col("tenure_days").clip(1, None)).alias("purchase_rate"),
        )
    )
    _cat_cols = [f"hist_cat_{c}" for c in CATEGORIES]
    _cat = features.select(_cat_cols).to_numpy()
    _tot = _cat.sum(1, keepdims=True)
    _p = np.divide(_cat, np.maximum(_tot, 1.0))
    _ent = -np.sum(np.where(_p > 0, _p * np.log(_p), 0.0), axis=1)
    features = features.with_columns(pl.Series("category_entropy", _ent))

    labels = customers.select("customer_key")
    for h in HORIZONS:
        fh = fut.filter(pl.col("ts_dt") <= pl.lit(as_of + timedelta(days=h)))
        agg_lf = (
            fh.lazy()
            .group_by("customer_key")
            .agg(_label_aggs(h))
            .with_columns(_label_first_expr(h, as_of))
            .drop("_first")
        )
        agg = _collect(agg_lf, engine)
        labels = labels.join(agg, on="customer_key", how="left")
        labels = labels.with_columns(_label_finalize(h))

    wide = features.join(labels, on="customer_key", how="left")

    # Fail safe (principle 7): a column the builder made but cannot attribute to a cut of
    # the stream is a leak waiting to happen — refuse, never stamp it silently as clean.
    unexplained = (set(features.columns) | set(labels.columns)) - {"customer_key"} - set(prov)
    if unexplained:
        raise ValueError(
            f"build_path_a: {len(unexplained)} column(s) built without provenance: "
            f"{sorted(unexplained)} — add them to the expression lists in path_a_build.py "
            "(the stamp is derived from construction, so an unstamped column means the "
            "construction and the claim have drifted apart)"
        )
    engine_state["provenance"] = prov
    return features, labels, wide, engine_state
