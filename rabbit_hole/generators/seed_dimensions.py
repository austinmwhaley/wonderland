"""Dimension + inventory seeding for the synthetic commerce world (vectorized)."""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import numpy as np
import polars as pl

from rabbit_hole.generators.business_tables import _bulk_insert
from rabbit_hole.generators.generate_support import (
    ProgressReporter,
    iso_expr,
    sample_categorical,
    seasonal_wave,
)

_PROMO_CHANNELS = ["email", "onsite", "paid_search", "social"]
_MIN_CARTS = np.array([0.0, 20.0, 35.0, 50.0, 75.0])
_PRICE_CENTERS = np.array([18.0, 240.0, 58.0, 86.0, 34.0, 72.0])
_BASE_UNITS = np.array([180, 60, 120, 90, 140, 100], dtype=np.int64)
_GEOS = [
    ("US", "CA", "San Francisco", "94105"),
    ("US", "NY", "New York", "10001"),
    ("US", "TX", "Austin", "78701"),
    ("US", "WA", "Seattle", "98101"),
    ("CA", "ON", "Toronto", "M5H"),
    ("GB", "LND", "London", "EC1A"),
]
_GENDERS = ["female", "male", "other"]
_GENDER_WEIGHTS = [0.49, 0.48, 0.03]
_LOYALTY_TIERS = ["none", "silver", "gold", "platinum"]
_LOYALTY_WEIGHTS = [0.48, 0.30, 0.17, 0.05]
_INCOME_BANDS = ["low", "lower_mid", "upper_mid", "high"]
_INCOME_WEIGHTS = [0.23, 0.35, 0.29, 0.13]
_LIFECYCLE_STAGES = ["new", "active", "at_risk", "dormant"]
_LIFECYCLE_WEIGHTS = [0.24, 0.43, 0.22, 0.11]
_CUSTOMER_CHANNELS = ["organic", "paid_search", "email", "affiliate", "social"]
_CHANNEL_WEIGHTS = [0.34, 0.26, 0.14, 0.08, 0.18]
_ACTIVITY_BASELINE = np.array([0.72, 1.0, 0.53, 0.22])


def _seed_dimensions(
    conn: duckdb.DuckDBPyConnection,
    rng: np.random.Generator,
    now: datetime,
    start_ts: datetime,
    total_days: int,
    years: int,
    num_customers: int,
    num_products: int,
    reporter: ProgressReporter | None,
):
    """Generate promotions, products, customers, and daily inventory rows."""
    # Domain taxonomies and weighted priors make simulated behavior coherent.
    categories = ["grocery", "electronics", "apparel", "home", "beauty", "sports"]
    brands = {
        "grocery": ["FreshCraft", "HarvestField", "GreenMile"],
        "electronics": ["Voltix", "PixelForge", "CircuitNest"],
        "apparel": ["Threadline", "NorthVale", "WearHouse"],
        "home": ["Hearthlane", "Oakline", "CopperRoom"],
        "beauty": ["BloomTheory", "NovaSkin", "PureNest"],
        "sports": ["PeakForm", "SprintCore", "Athletica"],
    }
    n_cats = len(categories)

    def _anchor(ts: datetime) -> np.datetime64:
        return np.datetime64(ts.astimezone(timezone.utc).replace(tzinfo=None), "ms")

    start_day = _anchor(start_ts)
    now64 = _anchor(now)

    # ---------------------------------------------------------------- promotions
    promotion_count = max(18, years * 18)
    if reporter is not None:
        reporter.start("generate promotions", promotion_count)
    duration = rng.integers(5, 22, promotion_count)  # randint(5, 21) inclusive
    start_offset = rng.integers(0, total_days - duration + 1)
    p_start = start_day + start_offset.astype("timedelta64[D]")
    p_end = p_start + duration.astype("timedelta64[D]")
    p_channel = rng.integers(0, len(_PROMO_CHANNELS), promotion_count)
    p_discount = np.round(rng.uniform(0.05, 0.35, promotion_count), 3)
    p_min_cart = _MIN_CARTS[rng.integers(0, len(_MIN_CARTS), promotion_count)]
    iso_start = np.datetime_as_string(p_start, unit="s")
    iso_end = np.datetime_as_string(p_end, unit="s")
    promotion_rows = [
        (
            f"PROMO_{idx:05d}",
            f"Promo {idx}",
            _PROMO_CHANNELS[p_channel[idx]],
            iso_start[idx] + "+00:00",
            iso_end[idx] + "+00:00",
            float(p_discount[idx]),
            float(p_min_cart[idx]),
        )
        for idx in range(promotion_count)
    ]
    if reporter is not None:
        reporter.finish(f"rows={len(promotion_rows):,}")

    # ----------------------------------------------------------------- products
    if reporter is not None:
        reporter.start("generate products", num_products)
    p_idx = np.arange(num_products, dtype=np.int64)
    cat_idx = p_idx % n_cats
    brand_flat = [b for c in categories for b in brands[c]]
    brand_pick = cat_idx * 3 + rng.integers(0, 3, num_products)
    product_brand = np.array(brand_flat, dtype=object)[brand_pick]
    launch_dt = start_day + rng.integers(0, total_days // 2 + 1, num_products).astype(
        "timedelta64[D]"
    )
    discontinued = rng.random(num_products) < 0.08
    discont_offset = rng.integers(120, total_days + 1, num_products)
    discont_dt = np.where(
        discontinued,
        launch_dt + discont_offset.astype("timedelta64[D]"),
        np.datetime64("NaT", "ms"),
    ).astype("datetime64[ms]")
    discont_dt = np.where(discont_dt > now64, np.datetime64("NaT", "ms"), discont_dt).astype(
        "datetime64[ms]"
    )
    base_price = np.maximum(
        np.round(rng.lognormal(0.0, 0.42, num_products) * _PRICE_CENTERS[cat_idx], 2), 2.5
    )
    unit_cost = np.round(base_price * rng.uniform(0.42, 0.77, num_products), 2)

    category_arr = np.array(categories, dtype=object)
    titled_arr = np.array([c.title() for c in categories], dtype=object)
    product_df = pl.DataFrame(
        {
            "idx": p_idx,
            "category": category_arr[cat_idx],
            "titled": titled_arr[cat_idx],
            "brand": product_brand,
            "base_price": base_price,
            "unit_cost": unit_cost,
            "launch_dt": launch_dt,
            "discontinue_dt": discont_dt,
        }
    ).with_columns(
        (pl.lit("PROD_") + pl.col("idx").cast(pl.String).str.zfill(7)).alias("product_id"),
        (pl.lit("SKU-") + pl.col("idx").cast(pl.String).str.zfill(8)).alias("sku"),
        (
            pl.col("brand")
            + pl.lit(" ")
            + pl.col("titled")
            + pl.lit(" ")
            + pl.col("idx").cast(pl.String)
        ).alias("product_name"),
        iso_expr("launch_dt").alias("launch_ts"),
        iso_expr("discontinue_dt").alias("discontinue_ts"),
    )
    product_ids = product_df["product_id"].to_list()
    products_by_category: dict[str, list[str]] = {category: [] for category in categories}
    for i, cat_i in enumerate(cat_idx.tolist()):
        products_by_category[categories[cat_i]].append(product_ids[i])
    product_price = dict(zip(product_ids, base_price.tolist()))
    product_cost = dict(zip(product_ids, unit_cost.tolist()))
    _bulk_insert(
        conn,
        """
        INSERT INTO products (
            product_id, sku, product_name, category, brand,
            base_price, unit_cost, launch_ts, discontinue_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        product_df.select(
            "product_id",
            "sku",
            "product_name",
            "category",
            "brand",
            "base_price",
            "unit_cost",
            "launch_ts",
            "discontinue_ts",
        ),
    )
    if reporter is not None:
        reporter.finish(f"rows={len(product_ids):,}")

    # --------------------------------------------------------------- customers
    if reporter is not None:
        reporter.start("generate customers", num_customers)
    c_idx = np.arange(num_customers, dtype=np.int64)
    signup = (
        start_day
        + rng.integers(0, total_days, num_customers).astype("timedelta64[D]")
        + rng.integers(0, 24, num_customers).astype("timedelta64[h]")
    )
    birth_year = rng.integers(1945, 2008, num_customers)
    gender = np.array(_GENDERS, dtype=object)[
        sample_categorical(rng, _GENDER_WEIGHTS, num_customers)
    ]
    geo_pick = rng.integers(0, len(_GEOS), num_customers)
    country = np.array([g[0] for g in _GEOS], dtype=object)[geo_pick]
    state_region = np.array([g[1] for g in _GEOS], dtype=object)[geo_pick]
    city = np.array([g[2] for g in _GEOS], dtype=object)[geo_pick]
    postal = np.array([g[3] for g in _GEOS], dtype=object)[geo_pick]
    loyalty_tier = np.array(_LOYALTY_TIERS, dtype=object)[
        sample_categorical(rng, _LOYALTY_WEIGHTS, num_customers)
    ]
    cardholder = (rng.random(num_customers) < 0.41).astype(np.int64)
    income_band = np.array(_INCOME_BANDS, dtype=object)[
        sample_categorical(rng, _INCOME_WEIGHTS, num_customers)
    ]
    acquisition_channel = np.array(_CUSTOMER_CHANNELS, dtype=object)[
        sample_categorical(rng, _CHANNEL_WEIGHTS, num_customers)
    ]
    lifecycle = sample_categorical(rng, _LIFECYCLE_WEIGHTS, num_customers)
    lifecycle_stage = np.array(_LIFECYCLE_STAGES, dtype=object)[lifecycle]
    pref_category = rng.integers(0, n_cats, num_customers)
    customer_activity_arr = _ACTIVITY_BASELINE[lifecycle] * rng.uniform(0.75, 1.30, num_customers)

    signup_str = iso_expr("signup_dt")
    customer_df = pl.DataFrame(
        {
            "idx": c_idx,
            "signup_dt": signup.astype("datetime64[ms]"),
            "birth_year": birth_year,
            "gender": gender,
            "country": country,
            "state_region": state_region,
            "city": city,
            "postal_code": postal,
            "loyalty_tier": loyalty_tier,
            "cardholder_status": cardholder,
            "income_band": income_band,
            "acquisition_channel": acquisition_channel,
            "lifecycle_stage": lifecycle_stage,
        }
    ).with_columns(
        (pl.lit("CUST_") + pl.col("idx").cast(pl.String).str.zfill(8)).alias("customer_id"),
        signup_str,
    )

    customer_ids = customer_df["customer_id"].to_list()
    customer_country_state = dict(zip(customer_ids, zip(country.tolist(), state_region.tolist())))
    customer_pref_category = dict(zip(customer_ids, category_arr[pref_category].tolist()))
    customer_activity = dict(zip(customer_ids, customer_activity_arr.tolist()))
    _bulk_insert(
        conn,
        """
        INSERT INTO customers (
            customer_id, signup_ts, birth_year, gender, country, state_region,
            city, postal_code, loyalty_tier, cardholder_status, income_band,
            acquisition_channel, lifecycle_stage
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        customer_df.select(
            pl.col("customer_id"),
            pl.col("signup_dt").alias("signup_ts"),
            pl.col("birth_year"),
            pl.col("gender"),
            pl.col("country"),
            pl.col("state_region"),
            pl.col("city"),
            pl.col("postal_code"),
            pl.col("loyalty_tier"),
            pl.col("cardholder_status"),
            pl.col("income_band"),
            pl.col("acquisition_channel"),
            pl.col("lifecycle_stage"),
        ),
    )
    if reporter is not None:
        reporter.finish(f"rows={len(customer_ids):,}")

    _bulk_insert(
        conn,
        """
        INSERT INTO promotions (
            promotion_id, promotion_name, channel, start_ts, end_ts,
            discount_pct, min_cart_value
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        promotion_rows,
    )

    # --------------------------------------------------------------- inventory
    total_inventory_rows = total_days * num_products
    if reporter is not None:
        reporter.start("generate inventory", total_inventory_rows)
    season = 1.0 + 0.25 * seasonal_wave(np.arange(total_days, dtype=np.float64) / total_days)
    product_base = _BASE_UNITS[cat_idx]
    mu = season[:, None] * product_base[None, :]
    on_hand = np.maximum(rng.normal(mu, product_base[None, :] * 0.2, mu.shape).astype(np.int64), 0)
    reserved = np.minimum(
        on_hand,
        np.maximum((on_hand * rng.uniform(0.0, 0.28, on_hand.shape)).astype(np.int64), 0),
    )
    date_strs = np.datetime_as_string(
        (start_day + np.arange(total_days).astype("timedelta64[D]")).astype("datetime64[D]"),
        unit="D",
    )
    product_col = np.array(product_ids, dtype=object)
    inventory_written = 0
    days_per_chunk = max(1, 20_000 // max(num_products, 1))
    for d0 in range(0, total_days, days_per_chunk):
        d1 = min(d0 + days_per_chunk, total_days)
        rows_in_chunk = (d1 - d0) * num_products
        _bulk_insert(
            conn,
            """
            INSERT INTO inventory_daily (
                inventory_date, product_id, on_hand_units, reserved_units, unit_cost
            ) VALUES (?, ?, ?, ?, ?)
            """,
            {
                "inventory_date": np.repeat(date_strs[d0:d1], num_products),
                "product_id": np.tile(product_col, d1 - d0),
                "on_hand_units": on_hand[d0:d1].ravel(),
                "reserved_units": reserved[d0:d1].ravel(),
                "unit_cost": np.tile(unit_cost, d1 - d0),
            },
        )
        inventory_written += rows_in_chunk
        if reporter is not None:
            reporter.advance(rows_in_chunk)
    if reporter is not None:
        reporter.finish(f"rows={inventory_written:,}")
    return (
        promotion_rows,
        products_by_category,
        product_price,
        product_cost,
        customer_ids,
        customer_country_state,
        customer_pref_category,
        customer_activity,
        categories,
    )
