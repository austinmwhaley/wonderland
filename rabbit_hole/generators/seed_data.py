"""Synthetic commerce world seeding: dimensions, activity, and contacts."""

from __future__ import annotations

from datetime import timedelta
import random

import duckdb

from rabbit_hole.generators.business_tables import _exec_script
from rabbit_hole.generators.generate_support import ProgressReporter, _REFERENCE_NOW
from rabbit_hole.generators.seed_contacts import _seed_contacts
from rabbit_hole.generators.seed_dimensions import _seed_dimensions
from rabbit_hole.generators.seed_events import _seed_events


def seed_business_data(
    conn: duckdb.DuckDBPyConnection,
    num_customers: int = 120_000,
    num_products: int = 1_200,
    seed: int = 17,
    years: int = 3,
    event_count: int = 10_000_000,
    order_ratio: float = 0.08,
    min_orders_per_customer: int = 4,
    activity_sampling_power: float = 1.35,
    reporter: ProgressReporter | None = None,
) -> None:
    """Generate a coherent synthetic commerce world and persist it to DuckDB.

    Generation strategy:
    - Create dimensions first (products, customers, promotions).
    - Build daily inventory snapshots for realism.
    - Simulate browse and order activity from customer activity profiles.
    - Preserve temporal consistency per customer.
    """

    rng = random.Random(seed)
    order_ratio = min(max(float(order_ratio), 0.01), 0.50)
    min_orders_per_customer = max(int(min_orders_per_customer), 1)
    activity_sampling_power = max(float(activity_sampling_power), 0.10)

    now = _REFERENCE_NOW
    start_ts = now - timedelta(days=365 * years)
    total_days = max(years * 365, 30)

    _exec_script(
        conn,
        """
        DELETE FROM customer_events;
        DELETE FROM orders;
        DELETE FROM website_browse;
        DELETE FROM inventory_daily;
        DELETE FROM promotions;
        DELETE FROM products;
        DELETE FROM customers;
        """,
    )

    if reporter is not None:
        reporter.log(
            "Preparing events-first dataset "
            f"customers={num_customers:,} products={num_products:,} years={years} "
            f"target_events={event_count:,} order_ratio={order_ratio:.3f} "
            f"min_orders_per_customer={min_orders_per_customer} "
            f"activity_sampling_power={activity_sampling_power:.2f}"
        )

    (
        promotion_rows,
        products_by_category,
        product_price,
        product_cost,
        customer_ids,
        customer_country_state,
        customer_pref_category,
        customer_activity,
        categories,
    ) = _seed_dimensions(
        conn,
        rng,
        now,
        start_ts,
        total_days,
        years,
        num_customers,
        num_products,
        reporter,
    )
    order_counter = _seed_events(
        conn,
        rng,
        start_ts,
        total_days,
        reporter,
        activity_sampling_power,
        event_count,
        order_ratio,
        min_orders_per_customer,
        num_customers,
        promotion_rows,
        products_by_category,
        product_price,
        product_cost,
        customer_ids,
        customer_activity,
        customer_pref_category,
        customer_country_state,
        categories,
    )
    _seed_contacts(
        conn,
        rng,
        start_ts,
        total_days,
        num_customers,
        order_counter,
        reporter,
        promotion_rows,
        products_by_category,
        product_price,
        product_cost,
        customer_ids,
        customer_activity,
        customer_country_state,
        customer_pref_category,
    )

    # DuckDB autocommits.
    if reporter is not None:
        reporter.log("DuckDB seeding complete")
