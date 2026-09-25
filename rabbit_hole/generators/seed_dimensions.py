"""Dimension + inventory seeding for the synthetic commerce world."""

from __future__ import annotations

from datetime import datetime, timedelta
import random

import duckdb

from rabbit_hole.generators.business_tables import _bulk_insert
from rabbit_hole.generators.generate_support import (
    ProgressReporter,
    _seasonal_wave,
    _weighted_choice,
)


def _seed_dimensions(
    conn: duckdb.DuckDBPyConnection,
    rng: random.Random,
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

    customer_channels = ["organic", "paid_search", "email", "affiliate", "social"]
    channel_weights = [0.34, 0.26, 0.14, 0.08, 0.18]
    loyalty_tiers = ["none", "silver", "gold", "platinum"]
    loyalty_weights = [0.48, 0.30, 0.17, 0.05]
    income_bands = ["low", "lower_mid", "upper_mid", "high"]
    income_weights = [0.23, 0.35, 0.29, 0.13]
    lifecycle_stages = ["new", "active", "at_risk", "dormant"]
    lifecycle_weights = [0.24, 0.43, 0.22, 0.11]
    genders = ["female", "male", "other"]
    gender_weights = [0.49, 0.48, 0.03]
    geos = [
        ("US", "CA", "San Francisco", "94105"),
        ("US", "NY", "New York", "10001"),
        ("US", "TX", "Austin", "78701"),
        ("US", "WA", "Seattle", "98101"),
        ("CA", "ON", "Toronto", "M5H"),
        ("GB", "LND", "London", "EC1A"),
    ]

    promotion_rows: list[tuple[str, str, str, str, str, float, float]] = []
    promotion_count = max(18, years * 18)
    promo_channels = ["email", "onsite", "paid_search", "social"]
    if reporter is not None:
        reporter.start("generate promotions", promotion_count)
    for idx in range(promotion_count):
        promo_id = f"PROMO_{idx:05d}"
        duration_days = rng.randint(5, 21)
        promo_start = start_ts + timedelta(days=rng.randint(0, total_days - duration_days))
        promo_end = promo_start + timedelta(days=duration_days)
        discount_pct = round(rng.uniform(0.05, 0.35), 3)
        min_cart = round(rng.choice([0.0, 20.0, 35.0, 50.0, 75.0]), 2)
        promotion_rows.append(
            (
                promo_id,
                f"Promo {idx}",
                rng.choice(promo_channels),
                promo_start.isoformat(),
                promo_end.isoformat(),
                discount_pct,
                min_cart,
            )
        )
        if reporter is not None:
            reporter.advance(1)
    if reporter is not None:
        reporter.finish(f"rows={len(promotion_rows):,}")

    product_rows: list[tuple[str, str, str, str, str, float, float, str, str | None]] = []
    products_by_category: dict[str, list[str]] = {category: [] for category in categories}
    product_price: dict[str, float] = {}
    product_cost: dict[str, float] = {}

    if reporter is not None:
        reporter.start("generate products", num_products)
    for idx in range(num_products):
        product_id = f"PROD_{idx:07d}"
        category = categories[idx % len(categories)]
        brand = rng.choice(brands[category])
        launch_dt = start_ts + timedelta(days=rng.randint(0, total_days // 2))
        discontinue_dt = None
        if rng.random() < 0.08:
            discontinue_dt = launch_dt + timedelta(days=rng.randint(120, total_days))
            if discontinue_dt > now:
                discontinue_dt = None

        price_center = {
            "grocery": 18.0,
            "electronics": 240.0,
            "apparel": 58.0,
            "home": 86.0,
            "beauty": 34.0,
            "sports": 72.0,
        }[category]
        base_price = max(round(rng.lognormvariate(mu=0.0, sigma=0.42) * price_center, 2), 2.5)
        unit_cost = round(base_price * rng.uniform(0.42, 0.77), 2)

        product_rows.append(
            (
                product_id,
                f"SKU-{idx:08d}",
                f"{brand} {category.title()} {idx}",
                category,
                brand,
                base_price,
                unit_cost,
                launch_dt.isoformat(),
                discontinue_dt.isoformat() if discontinue_dt else None,
            )
        )
        products_by_category[category].append(product_id)
        product_price[product_id] = base_price
        product_cost[product_id] = unit_cost
        if reporter is not None:
            reporter.advance(1)
    if reporter is not None:
        reporter.finish(f"rows={len(product_rows):,}")

    customer_rows: list[tuple[str, str, int, str, str, str, str, str, str, int, str, str, str]] = []
    customer_ids: list[str] = []
    customer_country_state: dict[str, tuple[str, str]] = {}
    customer_pref_category: dict[str, str] = {}
    customer_activity: dict[str, float] = {}

    if reporter is not None:
        reporter.start("generate customers", num_customers)
    for idx in range(num_customers):
        customer_id = f"CUST_{idx:08d}"
        signup_dt = start_ts + timedelta(
            days=rng.randint(0, total_days - 1), hours=rng.randint(0, 23)
        )
        birth_year = rng.randint(1945, 2007)
        gender = _weighted_choice(rng, genders, gender_weights)
        country, state_region, city, postal = geos[rng.randint(0, len(geos) - 1)]
        loyalty_tier = _weighted_choice(rng, loyalty_tiers, loyalty_weights)
        cardholder_status = 1 if rng.random() < 0.41 else 0
        income_band = _weighted_choice(rng, income_bands, income_weights)
        acquisition_channel = _weighted_choice(rng, customer_channels, channel_weights)
        lifecycle_stage = _weighted_choice(rng, lifecycle_stages, lifecycle_weights)

        customer_rows.append(
            (
                customer_id,
                signup_dt.isoformat(),
                birth_year,
                gender,
                country,
                state_region,
                city,
                postal,
                loyalty_tier,
                cardholder_status,
                income_band,
                acquisition_channel,
                lifecycle_stage,
            )
        )
        customer_ids.append(customer_id)
        customer_country_state[customer_id] = (country, state_region)
        customer_pref_category[customer_id] = categories[rng.randint(0, len(categories) - 1)]

        baseline = {
            "new": 0.72,
            "active": 1.0,
            "at_risk": 0.53,
            "dormant": 0.22,
        }[lifecycle_stage]
        customer_activity[customer_id] = baseline * rng.uniform(0.75, 1.30)
        if reporter is not None:
            reporter.advance(1)
    if reporter is not None:
        reporter.finish(f"rows={len(customer_rows):,}")

    if reporter is not None:
        reporter.log("Writing dimension tables to DuckDB")

    _bulk_insert(
        conn,
        """
        INSERT INTO customers (
            customer_id,
            signup_ts,
            birth_year,
            gender,
            country,
            state_region,
            city,
            postal_code,
            loyalty_tier,
            cardholder_status,
            income_band,
            acquisition_channel,
            lifecycle_stage
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        customer_rows,
    )
    _bulk_insert(
        conn,
        """
        INSERT INTO products (
            product_id,
            sku,
            product_name,
            category,
            brand,
            base_price,
            unit_cost,
            launch_ts,
            discontinue_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        product_rows,
    )
    _bulk_insert(
        conn,
        """
        INSERT INTO promotions (
            promotion_id,
            promotion_name,
            channel,
            start_ts,
            end_ts,
            discount_pct,
            min_cart_value
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        promotion_rows,
    )

    inventory_rows: list[tuple[str, str, int, int, float]] = []
    total_inventory_rows = total_days * len(product_rows)
    inventory_written = 0
    if reporter is not None:
        reporter.start("generate inventory", total_inventory_rows)
    for day in range(total_days):
        inventory_date = (start_ts + timedelta(days=day)).date().isoformat()
        for product_id, _, _, category, _, _, unit_cost, _, _ in product_rows:
            base_units = {
                "grocery": 180,
                "electronics": 60,
                "apparel": 120,
                "home": 90,
                "beauty": 140,
                "sports": 100,
            }[category]
            seasonality = 1.0 + 0.25 * _seasonal_wave(day / total_days)
            on_hand = max(int(rng.gauss(base_units * seasonality, base_units * 0.2)), 0)
            reserved = min(on_hand, max(int(on_hand * rng.uniform(0.0, 0.28)), 0))
            inventory_rows.append((inventory_date, product_id, on_hand, reserved, float(unit_cost)))
            if len(inventory_rows) >= 20_000:
                _bulk_insert(
                    conn,
                    """
                    INSERT INTO inventory_daily (
                        inventory_date,
                        product_id,
                        on_hand_units,
                        reserved_units,
                        unit_cost
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    inventory_rows,
                )
                inventory_written += len(inventory_rows)
                if reporter is not None:
                    reporter.advance(len(inventory_rows))
                inventory_rows.clear()
    if inventory_rows:
        _bulk_insert(
            conn,
            """
            INSERT INTO inventory_daily (
                inventory_date,
                product_id,
                on_hand_units,
                reserved_units,
                unit_cost
            ) VALUES (?, ?, ?, ?, ?)
            """,
            inventory_rows,
        )
        inventory_written += len(inventory_rows)
        if reporter is not None:
            reporter.advance(len(inventory_rows))
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
