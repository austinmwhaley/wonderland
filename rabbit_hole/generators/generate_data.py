"""Events-first business data generation and customer event-stream materialization.

This is application/demo code showing HOW to generate realistic business data.
It's not a reusable library component—it's specific to this smoke test scenario.
"""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import random
import duckdb
from pathlib import Path
import shutil
import sys
import time

import torch
from torch import Tensor


@dataclass(frozen=True)
class CustomerEventRow:
    """Normalized customer event row used for smoke testing."""

    event_id: int
    customer_key: str
    event_ts: str
    brand: str | None
    event_type: str
    event_attributes: str
    entity_type: str | None
    entity_id: str | None
    source_table: str | None
    value: float | None


from datetime import timezone as _tz
_REFERENCE_NOW = datetime(2026, 1, 1, tzinfo=_tz.utc)  # fixed data window end (deterministic)


class ProgressReporter:
    """Minimal terminal progress bar with optional timestamped log file."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path
        self._current_label: str | None = None
        self._current_total: int = 0
        self._current_count: int = 0
        self._last_percent: int = -1
        self._last_render_time: float = 0.0
        self._log_handle = None

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("w", encoding="utf-8")

    def close(self) -> None:
        """Close active progress rendering and any open log handle."""

        if self._current_label is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._current_label = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def log(self, message: str) -> None:
        """Write a timestamped message to stdout and optional log file."""

        timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        if self._current_label is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
        print(line, flush=True)
        if self._log_handle is not None:
            self._log_handle.write(line + "\n")
            self._log_handle.flush()

    def start(self, label: str, total: int) -> None:
        """Start a named progress scope with total work units."""

        self._current_label = label
        self._current_total = max(total, 1)
        self._current_count = 0
        self._last_percent = -1
        self._last_render_time = 0.0
        self.log(f"START {label} total={total}")
        self._render(force=True)

    def advance(self, count: int) -> None:
        """Advance current progress by ``count`` work units."""

        self._current_count = min(self._current_count + count, self._current_total)
        self._render()

    def finish(self, detail: str | None = None) -> None:
        """Finalize current progress scope and emit completion message."""

        if self._current_label is None:
            if detail:
                self.log(detail)
            return
        self._current_count = self._current_total
        label = self._current_label
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._current_label = None
        message = f"DONE {label}"
        if detail:
            message = f"{message} {detail}"
        self.log(message)

    def _render(self, force: bool = False) -> None:
        """Render progress bar with throttling to reduce terminal/log churn."""

        if self._current_label is None:
            return
        ratio = self._current_count / max(self._current_total, 1)
        percent = int(ratio * 100)
        now = time.monotonic()
        if not force and percent == self._last_percent and (now - self._last_render_time) < 1.0:
            return

        self._last_percent = percent
        self._last_render_time = now
        filled = int(ratio * 28)
        bar = "#" * filled + "-" * (28 - filled)
        line = (
            f"\r[{bar}] {percent:3d}% "
            f"{self._current_label} {self._current_count:,}/{self._current_total:,}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

        if self._log_handle is not None:
            timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
            self._log_handle.write(
                f"[{timestamp}] PROGRESS {self._current_label} {self._current_count}/{self._current_total} ({percent}%)\n"
            )
            self._log_handle.flush()


def _exec_script(conn, script: str) -> None:
    """Run a multi-statement SQL script (DuckDB execute() is one statement)."""
    for stmt in script.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


def _bulk_insert(conn, insert_sql: str, rows) -> int:
    """Bulk-load rows for an INSERT statement via a Polars/Arrow frame.

    DuckDB's executemany runs row-by-row and is extremely slow; loading a
    registered DataFrame in one shot is seconds instead of minutes."""
    import re
    import polars as pl
    if not rows:
        return 0
    m = re.search(r"INSERT\s+INTO\s+(\w+)\s*\(([^)]*)\)", insert_sql, re.S | re.I)
    if m is None:
        raise ValueError("cannot parse INSERT statement")
    table, cols = m.group(1), [c.strip() for c in m.group(2).split(",")]
    df = pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})
    conn.register("_rh_bulk", df)
    try:
        conn.execute(
            f"INSERT INTO {table} ({', '.join(cols)}) "
            f"SELECT {', '.join(cols)} FROM _rh_bulk"
        )
    finally:
        conn.unregister("_rh_bulk")
    return len(rows)


def _flush_order_items(conn, rows) -> int:
    """Bulk-insert order line items (no-op when empty)."""
    return _bulk_insert(
        conn,
        """
        INSERT INTO order_items (
            transaction_id, product_id, quantity,
            unit_price, unit_cost, line_revenue, line_cogs
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _flush_email(conn, rows) -> int:
    """Bulk-insert email sends (no-op when empty)."""
    return _bulk_insert(
        conn,
        """
        INSERT INTO email_sends (
            send_id, customer_id, campaign_id, send_ts,
            opened, clicked, open_ts, click_ts, click_session_id, converted_order_id,
            arm, propensity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def create_business_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Recreate all base DuckDB tables used by generation and smoke scripts."""

    _exec_script(conn, 
        """
        DROP TABLE IF EXISTS customer_sequence_embeddings;
        DROP TABLE IF EXISTS customer_targets;
        DROP TABLE IF EXISTS product_embeddings;
        DROP TABLE IF EXISTS customer_events;
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS website_events;
        DROP TABLE IF EXISTS website_sessions;
        DROP TABLE IF EXISTS website_browse;
        DROP TABLE IF EXISTS inventory_daily;
        DROP TABLE IF EXISTS promotions;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            signup_ts TEXT NOT NULL,
            birth_year INTEGER NOT NULL,
            gender TEXT NOT NULL,
            country TEXT NOT NULL,
            state_region TEXT NOT NULL,
            city TEXT NOT NULL,
            postal_code TEXT NOT NULL,
            loyalty_tier TEXT NOT NULL,
            cardholder_status INTEGER NOT NULL,
            income_band TEXT NOT NULL,
            acquisition_channel TEXT NOT NULL,
            lifecycle_stage TEXT NOT NULL
        );

        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            sku TEXT NOT NULL,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            brand TEXT NOT NULL,
            base_price REAL NOT NULL,
            unit_cost REAL NOT NULL,
            launch_ts TEXT NOT NULL,
            discontinue_ts TEXT
        );

        CREATE TABLE promotions (
            promotion_id TEXT PRIMARY KEY,
            promotion_name TEXT NOT NULL,
            channel TEXT NOT NULL,
            start_ts TEXT NOT NULL,
            end_ts TEXT NOT NULL,
            discount_pct REAL NOT NULL,
            min_cart_value REAL NOT NULL
        );

        CREATE TABLE inventory_daily (
            inventory_date TEXT NOT NULL,
            product_id TEXT NOT NULL,
            on_hand_units INTEGER NOT NULL,
            reserved_units INTEGER NOT NULL,
            unit_cost REAL NOT NULL,
            PRIMARY KEY (inventory_date, product_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE website_browse (
            browse_event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            event_ts TEXT NOT NULL,
            event_name TEXT NOT NULL,
            page_type TEXT NOT NULL,
            product_id TEXT,
            promotion_id TEXT,
            device_type TEXT NOT NULL,
            traffic_source TEXT NOT NULL,
            dwell_seconds INTEGER NOT NULL,
            quantity INTEGER,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id),
            FOREIGN KEY (promotion_id) REFERENCES promotions(promotion_id)
        );

        CREATE TABLE orders (
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            session_id TEXT,
            order_ts TEXT NOT NULL,
            order_status TEXT NOT NULL,
            payment_method TEXT NOT NULL,
            shipping_country TEXT NOT NULL,
            shipping_state TEXT NOT NULL,
            promotion_id TEXT,
            subtotal REAL NOT NULL,
            tax REAL NOT NULL,
            shipping_fee REAL NOT NULL,
            discount_amount REAL NOT NULL,
            order_total REAL NOT NULL,
            revenue REAL NOT NULL DEFAULT 0,       -- net merchandise = subtotal - discount - returns
            cogs REAL NOT NULL DEFAULT 0,           -- realized cost of goods sold
            gross_margin REAL NOT NULL DEFAULT 0,   -- revenue - cogs
            return_flag INTEGER NOT NULL,
            return_ts TEXT,
            return_amount REAL NOT NULL,
            cancelled_ts TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (promotion_id) REFERENCES promotions(promotion_id)
        );

        CREATE TABLE order_items (
            transaction_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            unit_cost REAL NOT NULL,
            line_revenue REAL NOT NULL,   -- unit_price * quantity
            line_cogs REAL NOT NULL,      -- unit_cost * quantity
            FOREIGN KEY (transaction_id) REFERENCES orders(transaction_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE email_sends (
            send_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            campaign_id TEXT,
            send_ts TEXT NOT NULL,
            opened INTEGER NOT NULL DEFAULT 0,
            clicked INTEGER NOT NULL DEFAULT 0,
            open_ts TEXT,
            click_ts TEXT,
            click_session_id TEXT,        -- web session the click landed in
            converted_order_id TEXT,      -- order in that session, if any
            arm INTEGER,                  -- randomized cadence arm active at send time
            propensity REAL,              -- logged P(arm | activity) for that period
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE email_holdout (
            customer_id TEXT NOT NULL,
            period INTEGER NOT NULL,
            holdout INTEGER NOT NULL,     -- 1 = held out (NO marketing this period)
            PRIMARY KEY (customer_id, period)
        );

        CREATE TABLE email_arm (
            customer_id TEXT PRIMARY KEY,
            arm INTEGER NOT NULL,         -- randomized cadence arm (0..3)
            propensity REAL NOT NULL,     -- P(arm | activity): logged, with overlap
            optimal_arm INTEGER,          -- TRUE best arm (ground truth for scoring)
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE SEQUENCE IF NOT EXISTS customer_events_seq START 1;
        CREATE TABLE customer_events (
            event_id BIGINT DEFAULT nextval('customer_events_seq'),
            customer_key TEXT NOT NULL,       -- canonical (1/5)
            event_ts TEXT NOT NULL,           -- canonical (2/5)
            brand TEXT,                       -- canonical (3/5)
            event_type TEXT NOT NULL,         -- canonical (4/5)
            event_attributes TEXT NOT NULL,   -- canonical (5/5)
            entity_type TEXT,                 -- extra
            entity_id TEXT,                   -- extra
            source_table TEXT,                -- extra
            value REAL                        -- extra
        );

        CREATE INDEX idx_browse_customer_ts ON website_browse (customer_id, event_ts);
        CREATE INDEX idx_orders_customer_ts ON orders (customer_id, order_ts);
        CREATE INDEX idx_customer_events_customer_ts ON customer_events (customer_key, event_ts);
        """
    )
    # DuckDB autocommits.


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

    _exec_script(conn, 
        """
        DELETE FROM customer_events;
        DELETE FROM orders;
        DELETE FROM website_browse;
        DELETE FROM inventory_daily;
        DELETE FROM promotions;
        DELETE FROM products;
        DELETE FROM customers;
        """
    )

    if reporter is not None:
        reporter.log(
            "Preparing events-first dataset "
            f"customers={num_customers:,} products={num_products:,} years={years} "
            f"target_events={event_count:,} order_ratio={order_ratio:.3f} "
            f"min_orders_per_customer={min_orders_per_customer} "
            f"activity_sampling_power={activity_sampling_power:.2f}"
        )

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
        signup_dt = start_ts + timedelta(days=rng.randint(0, total_days - 1), hours=rng.randint(0, 23))
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

    _bulk_insert(conn, 
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
    _bulk_insert(conn, 
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
    _bulk_insert(conn, 
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
                _bulk_insert(conn, 
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
        _bulk_insert(conn, 
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

    customer_sampling_weights = [
        max(customer_activity[customer_id], 0.05) ** activity_sampling_power
        for customer_id in customer_ids
    ]
    cumulative_customer_weights: list[float] = []
    running_weight = 0.0
    for weight in customer_sampling_weights:
        running_weight += weight
        cumulative_customer_weights.append(running_weight)

    # Enforce enough order activity to support downstream outcome labels.
    target_orders = max(int(event_count * order_ratio), num_customers * min_orders_per_customer, 1)
    target_browse = max(event_count - target_orders, num_customers)

    browse_event_rows: list[tuple[str, str, str, str, str, str, str | None, str | None, str, str, int, int | None]] = []
    order_rows: list[tuple[str, str, str, str, str, str, str, str, str | None, float, float, float, float, float, float, float, float, int, str | None, float, str | None]] = []
    order_item_rows: list[tuple[str, str, int, float, float, float, float]] = []

    event_counter = 0
    session_counter = 0
    order_counter = 0
    last_session_for_customer: dict[str, str] = {}
    last_ts_for_customer: dict[str, datetime] = {}
    first_ts_for_customer: dict[str, datetime] = {}

    traffic_sources = ["direct", "organic", "paid_search", "email", "social", "affiliate"]
    traffic_weights = [0.17, 0.31, 0.18, 0.12, 0.15, 0.07]
    devices = ["mobile", "desktop", "tablet"]
    device_weights = [0.58, 0.36, 0.06]

    browse_written = 0
    if reporter is not None:
        reporter.start("generate browse events", target_browse)
    for _ in range(target_browse):
        customer_id = customer_ids[_sample_weighted_index(rng, cumulative_customer_weights)]
        activity = customer_activity[customer_id]

        new_session_prob = max(0.08, min(0.34, 0.22 + (1.0 - activity) * 0.08))
        if customer_id not in last_session_for_customer or rng.random() < new_session_prob:
            session_id = f"SESS_{session_counter:010d}"
            session_counter += 1
            last_session_for_customer[customer_id] = session_id
        else:
            session_id = last_session_for_customer[customer_id]

        event_ts = _sample_event_ts(rng=rng, start_ts=start_ts, total_days=total_days)
        prev_ts = last_ts_for_customer.get(customer_id)
        if prev_ts is not None and event_ts < prev_ts:
            event_ts = prev_ts + timedelta(minutes=rng.randint(1, 90))
        last_ts_for_customer[customer_id] = event_ts
        first_ts_for_customer.setdefault(customer_id, event_ts)

        browse_probs = _browse_distribution(activity)
        event_name = _weighted_choice(
            rng,
            ["page_view", "search", "product_view", "add_to_cart"],
            browse_probs,
        )

        if event_name == "page_view":
            page_type = rng.choice(["home", "category", "campaign", "help"])
            product_id = None
            quantity = None
        elif event_name == "search":
            page_type = "search"
            product_id = None
            quantity = None
        elif event_name == "product_view":
            page_type = "product"
            preferred = customer_pref_category[customer_id]
            category = preferred if rng.random() < 0.62 else categories[rng.randrange(0, len(categories))]
            product_id = products_by_category[category][rng.randrange(0, len(products_by_category[category]))]
            quantity = None
        else:
            page_type = "cart"
            preferred = customer_pref_category[customer_id]
            category = preferred if rng.random() < 0.74 else categories[rng.randrange(0, len(categories))]
            product_id = products_by_category[category][rng.randrange(0, len(products_by_category[category]))]
            quantity = rng.randint(1, 3)

        promotion_id = None
        if rng.random() < 0.09:
            promotion_id = promotion_rows[rng.randrange(0, len(promotion_rows))][0]

        browse_event_rows.append(
            (
                f"BROWSE_{event_counter:012d}",
                session_id,
                customer_id,
                event_ts.isoformat(),
                event_name,
                page_type,
                product_id,
                promotion_id,
                _weighted_choice(rng, devices, device_weights),
                _weighted_choice(rng, traffic_sources, traffic_weights),
                max(int(rng.gammavariate(alpha=2.3, beta=18.0)), 2),
                quantity,
            )
        )
        event_counter += 1

        if len(browse_event_rows) >= 20_000:
            _bulk_insert(conn, 
                """
                INSERT INTO website_browse (
                    browse_event_id,
                    session_id,
                    customer_id,
                    event_ts,
                    event_name,
                    page_type,
                    product_id,
                    promotion_id,
                    device_type,
                    traffic_source,
                    dwell_seconds,
                    quantity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                browse_event_rows,
            )
            browse_written += len(browse_event_rows)
            if reporter is not None:
                reporter.advance(len(browse_event_rows))
            browse_event_rows.clear()
    if browse_event_rows:
        _bulk_insert(conn, 
            """
            INSERT INTO website_browse (
                browse_event_id,
                session_id,
                customer_id,
                event_ts,
                event_name,
                page_type,
                product_id,
                promotion_id,
                device_type,
                traffic_source,
                dwell_seconds,
                quantity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            browse_event_rows,
        )
        browse_written += len(browse_event_rows)
        if reporter is not None:
            reporter.advance(len(browse_event_rows))
    if reporter is not None:
        reporter.finish(f"rows={browse_written:,} sessions={session_counter:,}")

    order_written = 0
    if reporter is not None:
        reporter.start("generate orders", target_orders)
    for _ in range(target_orders):
        customer_id = customer_ids[_sample_weighted_index(rng, cumulative_customer_weights)]
        session_id = last_session_for_customer.get(customer_id)
        if session_id is None:
            session_id = f"SESS_{session_counter:010d}"
            session_counter += 1
            last_session_for_customer[customer_id] = session_id

        # Orders are sampled across the customer's active history (not appended
        # after their last browse event), so value is realized throughout time.
        order_ts = _sample_event_ts(rng=rng, start_ts=start_ts, total_days=total_days)
        first = first_ts_for_customer.get(customer_id)
        if first is not None and order_ts < first:
            order_ts = first + timedelta(minutes=rng.randint(1, 1440))
        order_ts = order_ts + timedelta(minutes=rng.randint(5, 420))

        line_count = 1 if rng.random() < 0.65 else 2 if rng.random() < 0.9 else 3
        subtotal = 0.0
        line_cogs = 0.0
        basket_categories: list[str] = []
        line_items: list[tuple[str, int, float, float]] = []  # (product_id, units, price, cost)
        for _line in range(line_count):
            pref_category = customer_pref_category[customer_id]
            category = pref_category if rng.random() < 0.58 else categories[rng.randrange(0, len(categories))]
            basket_categories.append(category)
            product_id = products_by_category[category][rng.randrange(0, len(products_by_category[category]))]
            units = 1 if rng.random() < 0.74 else rng.randint(2, 4)
            unit_price = product_price[product_id]
            unit_cost = product_cost[product_id]
            subtotal += unit_price * units
            line_cogs += unit_cost * units
            line_items.append((product_id, units, unit_price, unit_cost))

        promo_id = None
        discount = 0.0
        if rng.random() < 0.19:
            promo = promotion_rows[rng.randrange(0, len(promotion_rows))]
            promo_id = promo[0]
            promo_discount_pct = float(promo[5])
            promo_min_value = float(promo[6])
            if subtotal >= promo_min_value:
                discount = round(subtotal * promo_discount_pct, 2)

        tax = round(max(subtotal - discount, 0.0) * rng.uniform(0.06, 0.095), 2)
        shipping_fee = 0.0 if subtotal >= 75.0 else round(rng.uniform(3.49, 8.99), 2)
        gross_total = round(max(subtotal - discount, 0.0) + tax + shipping_fee, 2)

        order_status = _weighted_choice(
            rng,
            ["completed", "cancelled", "refunded"],
            [0.935, 0.038, 0.027],
        )

        return_flag = 0
        return_ts = None
        return_amount = 0.0
        cancelled_ts = None

        if order_status == "cancelled":
            cancelled_ts = (order_ts + timedelta(minutes=rng.randint(2, 90))).isoformat()
        elif order_status in {"completed", "refunded"}:
            category_return_bias = 0.08
            if "apparel" in basket_categories:
                category_return_bias += 0.10
            if "electronics" in basket_categories:
                category_return_bias += 0.03
            if rng.random() < min(0.35, category_return_bias):
                return_flag = 1
                return_ts_dt = order_ts + timedelta(days=rng.randint(2, 65))
                return_ts = return_ts_dt.isoformat()
                if rng.random() < 0.64:
                    return_amount = round(gross_total, 2)
                    order_status = "refunded"
                else:
                    return_amount = round(gross_total * rng.uniform(0.15, 0.55), 2)

        payment_method = _weighted_choice(
            rng,
            ["credit_card", "debit_card", "paypal", "wallet", "gift_card"],
            [0.49, 0.23, 0.16, 0.09, 0.03],
        )
        ship_country, ship_state = customer_country_state[customer_id]

        # Revenue = net merchandise (subtotal - discount), realized net of
        # returns; COGS reversed proportionally with the return. Cancelled
        # orders contribute nothing.
        gross_rev = max(subtotal - discount, 0.0)
        eff_return = min(return_amount, gross_rev) if return_amount > 0 else 0.0
        if order_status == "cancelled":
            revenue, cogs = 0.0, 0.0
        else:
            revenue = round(gross_rev - eff_return, 2)
            cogs = round(line_cogs * (1 - eff_return / gross_rev), 2) if gross_rev > 0 else round(line_cogs, 2)
        gross_margin = round(revenue - cogs, 2)
        txn_id = f"TXN_{order_counter:012d}"

        order_rows.append(
            (
                txn_id,
                customer_id,
                session_id,
                order_ts.isoformat(),
                order_status,
                payment_method,
                ship_country,
                ship_state,
                promo_id,
                round(subtotal, 2),
                tax,
                shipping_fee,
                round(discount, 2),
                gross_total,
                revenue,
                cogs,
                gross_margin,
                return_flag,
                return_ts,
                return_amount,
                cancelled_ts,
            )
        )
        for product_id, units, unit_price, unit_cost in line_items:
            order_item_rows.append(
                (
                    txn_id,
                    product_id,
                    units,
                    round(unit_price, 2),
                    round(unit_cost, 2),
                    round(unit_price * units, 2),
                    round(unit_cost * units, 2),
                )
            )
        order_counter += 1

        if len(order_rows) >= 15_000:
            _bulk_insert(conn, 
                """
                INSERT INTO orders (
                    transaction_id,
                    customer_id,
                    session_id,
                    order_ts,
                    order_status,
                    payment_method,
                    shipping_country,
                    shipping_state,
                    promotion_id,
                    subtotal,
                    tax,
                    shipping_fee,
                    discount_amount,
                    order_total,
                    revenue,
                    cogs,
                    gross_margin,
                    return_flag,
                    return_ts,
                    return_amount,
                    cancelled_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                order_rows,
            )
            _flush_order_items(conn, order_item_rows)
            order_written += len(order_rows)
            if reporter is not None:
                reporter.advance(len(order_rows))
            order_rows.clear()
            order_item_rows.clear()

    if order_rows:
        _bulk_insert(conn, 
            """
            INSERT INTO orders (
                transaction_id,
                customer_id,
                session_id,
                order_ts,
                order_status,
                payment_method,
                shipping_country,
                shipping_state,
                promotion_id,
                subtotal,
                tax,
                shipping_fee,
                discount_amount,
                order_total,
                revenue,
                cogs,
                gross_margin,
                return_flag,
                return_ts,
                return_amount,
                cancelled_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            order_rows,
        )
        _flush_order_items(conn, order_item_rows)
        order_written += len(order_rows)
        if reporter is not None:
            reporter.advance(len(order_rows))

    # ------------------------------------------------------------------
    # email interactions: RANDOMIZED arm (cadence) with logged propensity and a
    # KNOWN causal effect on orders. Overlap identifies the counterfactual; the
    # known effect forecasts ground truth so red_king can be validated.
    # ------------------------------------------------------------------
    CADENCE = (0.2, 0.6, 1.2, 2.0)      # sends per week by arm (known schedule)
    # Per-customer UNIMODAL response: each customer has an optimal cadence x0
    # tied to activity (so the optimal arm DIFFERS by customer and is learnable
    # from the donor state). Per-click incremental-order rate peaks at x=x0.
    PEAK_CONV = 0.10
    sessions_by_customer: dict[str, list[str]] = {}
    for cid, sid in conn.execute(
        "SELECT customer_id, session_id FROM website_browse GROUP BY 1, 2"
    ).fetchall():
        sessions_by_customer.setdefault(cid, []).append(sid)
    converted: dict[tuple[str, str], str] = {}
    for cid, sid, txn in conn.execute(
        "SELECT customer_id, session_id, transaction_id FROM orders "
        "WHERE session_id IS NOT NULL"
    ).fetchall():
        converted[(cid, sid)] = txn

    def _softmax(v):
        m = max(v)
        e = [math.exp(x - m) for x in v]
        s = sum(e)
        return [x / s for x in e]

    # Per-PERIOD randomized cadence: within-customer action variation makes the
    # per-customer dose-response identifiable (enables personalization).
    N_PERIODS = 12
    period_days = max(total_days / N_PERIODS, 1.0)
    period_weeks = period_days / 7.0
    HOLD = 0.0                               # NO experiments: purely observational
    intents: dict[str, float] = {}           # LATENT unobserved confounder
    parms: dict[str, list[tuple[int, float]]] = {}
    arm_rows: list[tuple[str, int, float]] = []
    hold_rows: list[tuple[str, int, int]] = []
    for cid in customer_ids:
        activity = customer_activity[cid]
        intent = rng.gauss(0.0, 1.0)         # drives BOTH targeting and outcomes
        intents[cid] = intent
        seq = []
        # Purely OBSERVATIONAL: no experiments. Actions are the company's logged
        # choices, confounded by the latent intent. (No switchback, no holdout.)
        # PERSISTENT HOLDOUT: 5% of customers per period are held out of ALL
        # marketing (realistic; the randomized control that measures incrementality).
        HOLD_FRAC = 0.05
        for _p in range(N_PERIODS):
            if rng.random() < HOLD_FRAC:
                seq.append((-1, 0.0)); hold_rows.append((cid, _p, 1))
                continue
            hold_rows.append((cid, _p, 0))
            util = [(activity + 0.6 * intent) * k + rng.gauss(0.0, 0.8) for k in range(4)]
            pp = _softmax(util)
            a = int(rng.choices(range(4), weights=pp)[0]); pr = float(pp[a])
            seq.append((a, pr))
        parms[cid] = seq
        x0c = CADENCE[0] + (CADENCE[-1] - CADENCE[0]) * max(0.0, min(1.0, activity))
        opt = max(range(4), key=lambda a: CADENCE[a] * math.exp(-CADENCE[a] / x0c))
        arm_rows.append((cid, seq[0][0], seq[0][1], int(opt)))
    if arm_rows:
        _bulk_insert(conn, "INSERT INTO email_arm (customer_id, arm, propensity, optimal_arm) "
                           "VALUES (?, ?, ?, ?)", arm_rows)
    if hold_rows:
        _bulk_insert(conn, "INSERT INTO email_holdout (customer_id, period, holdout) "
                           "VALUES (?, ?, ?)", hold_rows)

    email_rows: list[tuple[str, str, str | None, str, int, int, str | None, str | None, str | None, str | None]] = []
    email_order_rows: list[tuple] = []
    email_item_rows: list[tuple] = []
    send_counter = 0
    weeks = max(total_days / 7.0, 1.0)
    payments = ["credit_card", "debit_card", "paypal", "wallet", "gift_card"]
    payment_w = [0.49, 0.23, 0.16, 0.09, 0.03]
    if reporter is not None:
        reporter.start("generate email", num_customers)
    for cid in customer_ids:
        activity = customer_activity[cid]
        act = max(0.0, min(1.0, activity))
        x0 = CADENCE[0] + (CADENCE[-1] - CADENCE[0]) * act   # optimal cadence
        seq = parms[cid]
        wts = [(0.0 if a < 0 else CADENCE[a] * period_weeks) for a, _ in seq]
        tot = sum(wts)
        n_sends = max(0, min(int(rng.gauss(tot, max(tot, 1.0) ** 0.5)), 1500))
        sess_list = sessions_by_customer.get(cid, [])
        for _ in range(n_sends):
            p = rng.choices(range(N_PERIODS), weights=wts)[0]
            arm, prop = seq[p]
            send_ts = start_ts + timedelta(days=p * period_days + rng.uniform(0.0, period_days))
            campaign = promotion_rows[rng.randrange(0, len(promotion_rows))][0] if promotion_rows else None
            opened = 1 if rng.random() < min(0.9, 0.12 + 0.55 * activity) else 0
            open_ts = click_ts = click_sid = conv_txn = None
            clicked = 0
            if opened:
                open_dt = send_ts + timedelta(minutes=rng.randint(3, 2880))
                open_ts = open_dt.isoformat()
                if rng.random() < min(0.8, 0.06 + 0.45 * activity):
                    clicked = 1
                    click_ts = (open_dt + timedelta(minutes=rng.randint(1, 120))).isoformat()
                    if sess_list:
                        weights = [1.0 + 2.0 * activity * ((cid, s) in converted) for s in sess_list]
                        running = 0.0
                        cum: list[float] = []
                        for w in weights:
                            running += w
                            cum.append(running)
                        click_sid = sess_list[_sample_weighted_index(rng, cum)]
                        conv_txn = converted.get((cid, click_sid))
                    # KNOWN causal effect: unimodal in cadence, peaked at the
                    # customer's x0 -> heterogeneous optimal arm (personalization).
                    x = CADENCE[arm]
                    season = 1.0 + 0.25 * math.sin(2 * math.pi * (send_ts.timetuple().tm_yday / 365.0))
                    p_conv = (PEAK_CONV * math.exp(1.0 - x / x0)
                              * math.exp(0.5 * intents.get(cid, 0.0)) * season)
                    if rng.random() < p_conv:
                        pref = customer_pref_category[cid]
                        prod = products_by_category[pref][rng.randrange(0, len(products_by_category[pref]))]
                        price = product_price[prod]; cost = product_cost[prod]
                        txn = f"TXN_{order_counter:012d}"; order_counter += 1
                        ots = datetime.fromisoformat(click_ts) + timedelta(minutes=rng.randint(5, 180))
                        subtotal = round(price, 2)
                        tax = round(subtotal * rng.uniform(0.06, 0.095), 2)
                        ship = 0.0 if subtotal >= 75.0 else round(rng.uniform(3.49, 8.99), 2)
                        total = round(subtotal + tax + ship, 2)
                        ship_c, ship_s = customer_country_state[cid]
                        email_order_rows.append((
                            txn, cid, click_sid, ots.isoformat(), "completed",
                            _weighted_choice(rng, payments, payment_w), ship_c, ship_s, None,
                            subtotal, tax, ship, 0.0, total,
                            subtotal, round(cost, 2), round(subtotal - cost, 2),
                            0, None, 0.0, None))
                        email_item_rows.append((txn, prod, 1, round(price, 2),
                                                round(cost, 2), round(price, 2), round(cost, 2)))
                        conv_txn = txn
            email_rows.append((f"EMAIL_{send_counter:012d}", cid, campaign, send_ts.isoformat(),
                               opened, clicked, open_ts, click_ts, click_sid, conv_txn,
                               arm, prop))
            send_counter += 1
            if len(email_rows) >= 20_000:
                _flush_email(conn, email_rows)
                email_rows.clear()
        if reporter is not None:
            reporter.advance(1)
    if email_rows:
        _flush_email(conn, email_rows)
    if email_order_rows:
        _bulk_insert(conn, """INSERT INTO orders (transaction_id, customer_id, session_id,
            order_ts, order_status, payment_method, shipping_country, shipping_state,
            promotion_id, subtotal, tax, shipping_fee, discount_amount, order_total,
            revenue, cogs, gross_margin, return_flag, return_ts, return_amount, cancelled_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            email_order_rows)
        _flush_order_items(conn, email_item_rows)
    if reporter is not None:
        reporter.finish(f"rows={send_counter:,}")

    # DuckDB autocommits.
    if reporter is not None:
        reporter.log("DuckDB seeding complete")


def materialize_customer_event_stream(
    conn: duckdb.DuckDBPyConnection,
    reporter: ProgressReporter | None = None,
) -> int:
    """Build unified ``customer_events`` from dimensions, browse, and order facts.

    The resulting stream is intentionally heterogeneous: signup, browse,
    placement, cancellation, and return events share one sequence table so the
    temporal core can learn cross-domain dependencies.
    """

    conn.execute("DELETE FROM customer_events")

    total_stages = 5
    if reporter is not None:
        reporter.start("materialize customer events", total_stages)

    conn.execute(
        """
        INSERT INTO customer_events (
            customer_key,
            event_ts,
            event_type,
            entity_type,
            entity_id,
            source_table,
            value,
            event_attributes
        )
        SELECT
            c.customer_id,
            c.signup_ts,
            'customer_signup',
            'customer',
            c.customer_id,
            'customers',
            0.0,
            json_object(
                'acquisition_channel', c.acquisition_channel,
                'loyalty_tier', c.loyalty_tier,
                'cardholder_status', c.cardholder_status,
                'income_band', c.income_band,
                'lifecycle_stage', c.lifecycle_stage
            )
        FROM customers c
        """
    )
    if reporter is not None:
        reporter.advance(1)

    conn.execute(
        """
        INSERT INTO customer_events (
            customer_key,
            event_ts,
            event_type,
            entity_type,
            entity_id,
            source_table,
            value,
            event_attributes
        )
        SELECT
            wb.customer_id,
            wb.event_ts,
            wb.event_name,
            CASE WHEN wb.product_id IS NULL THEN 'session' ELSE 'product' END,
            COALESCE(wb.product_id, wb.session_id),
            'website_browse',
            0.0,
            json_object(
                'browse_event_id', wb.browse_event_id,
                'session_id', wb.session_id,
                'page_type', wb.page_type,
                'promotion_id', wb.promotion_id,
                'device_type', wb.device_type,
                'traffic_source', wb.traffic_source,
                'dwell_seconds', wb.dwell_seconds,
                'quantity', wb.quantity
            )
        FROM website_browse wb
        """
    )
    if reporter is not None:
        reporter.advance(1)

    conn.execute(
        """
        INSERT INTO customer_events (
            customer_key,
            event_ts,
            event_type,
            entity_type,
            entity_id,
            source_table,
            value,
            event_attributes
        )
        SELECT
            o.customer_id,
            o.order_ts,
            'order_placed',
            'order',
            o.transaction_id,
            'orders',
            o.order_total,
            json_object(
                'session_id', o.session_id,
                'order_status', o.order_status,
                'promotion_id', o.promotion_id,
                'payment_method', o.payment_method,
                'subtotal', o.subtotal,
                'tax', o.tax,
                'shipping_fee', o.shipping_fee,
                'discount_amount', o.discount_amount,
                'revenue', o.revenue,
                'cogs', o.cogs,
                'gross_margin', o.gross_margin
            )
        FROM orders o
        """
    )
    if reporter is not None:
        reporter.advance(1)

    conn.execute(
        """
        INSERT INTO customer_events (
            customer_key,
            event_ts,
            event_type,
            entity_type,
            entity_id,
            source_table,
            value,
            event_attributes
        )
        SELECT
            o.customer_id,
            o.cancelled_ts,
            'order_cancelled',
            'order',
            o.transaction_id,
            'orders',
            o.order_total,
            json_object('order_status', o.order_status)
        FROM orders o
        WHERE o.cancelled_ts IS NOT NULL
        """
    )
    if reporter is not None:
        reporter.advance(1)

    conn.execute(
        """
        INSERT INTO customer_events (
            customer_key,
            event_ts,
            event_type,
            entity_type,
            entity_id,
            source_table,
            value,
            event_attributes
        )
        SELECT
            o.customer_id,
            o.return_ts,
            'order_returned',
            'order',
            o.transaction_id,
            'orders',
            o.return_amount,
            json_object('return_flag', o.return_flag, 'order_status', o.order_status)
        FROM orders o
        WHERE o.return_flag = 1 AND o.return_ts IS NOT NULL
        """
    )
    if reporter is not None:
        reporter.advance(1)

    # --- email action events: send -> open -> click ---
    conn.execute(
        """
        INSERT INTO customer_events (
            customer_key, event_ts, event_type, entity_type, entity_id,
            source_table, value, event_attributes
        )
        SELECT
            customer_id, send_ts, 'email_send', 'email', send_id, 'email_sends', 0,
            json_object('campaign_id', campaign_id, 'opened', opened, 'clicked', clicked)
        FROM email_sends
        """
    )
    if reporter is not None:
        reporter.advance(1)
    conn.execute(
        """
        INSERT INTO customer_events (
            customer_key, event_ts, event_type, entity_type, entity_id,
            source_table, value, event_attributes
        )
        SELECT
            customer_id, open_ts, 'email_open', 'email', send_id, 'email_sends', 0,
            json_object('campaign_id', campaign_id)
        FROM email_sends
        WHERE opened = 1 AND open_ts IS NOT NULL
        """
    )
    if reporter is not None:
        reporter.advance(1)
    conn.execute(
        """
        INSERT INTO customer_events (
            customer_key, event_ts, event_type, entity_type, entity_id,
            source_table, value, event_attributes
        )
        SELECT
            customer_id, click_ts, 'email_click', 'session',
            COALESCE(click_session_id, send_id), 'email_sends', 0,
            json_object(
                'campaign_id', campaign_id,
                'click_session_id', click_session_id,
                'converted_order_id', converted_order_id
            )
        FROM email_sends
        WHERE clicked = 1 AND click_ts IS NOT NULL
        """
    )
    if reporter is not None:
        reporter.advance(1)

    # Fill the canonical `brand` field: product events inherit the product's
    # brand; everything else gets an empty brand (contract: brand present).
    conn.execute(
        """
        UPDATE customer_events
        SET brand = COALESCE(
            (SELECT p.brand FROM products p
             WHERE p.product_id = customer_events.entity_id), '')
        WHERE entity_type = 'product'
        """
    )
    conn.execute("UPDATE customer_events SET brand = '' WHERE brand IS NULL")

    # DuckDB autocommits.
    event_count = int(conn.execute("SELECT COUNT(*) FROM customer_events").fetchone()[0])
    if reporter is not None:
        reporter.finish(f"rows={event_count:,}")
    return event_count


def load_customer_event_stream(
    conn: duckdb.DuckDBPyConnection,
    customer_ids: list[str] | None = None,
) -> list[CustomerEventRow]:
    """Load event stream rows ordered for deterministic per-customer playback."""

    if customer_ids is not None:
        placeholders = ",".join("?" * len(customer_ids))
        rows = conn.execute(
            f"""
            SELECT
                event_id,
                customer_key,
                event_ts,
                brand,
                event_type,
                event_attributes,
                entity_type,
                entity_id,
                source_table,
                value
            FROM customer_events
            WHERE customer_key IN ({placeholders})
            ORDER BY customer_key, event_ts, event_id
            """,
            customer_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
                event_id,
                customer_key,
                event_ts,
                brand,
                event_type,
                event_attributes,
                entity_type,
                entity_id,
                source_table,
                value
            FROM customer_events
            ORDER BY customer_key, event_ts, event_id
            """
        ).fetchall()
    return [CustomerEventRow(*row) for row in rows]


def _weighted_choice(rng: random.Random, items: list[str], weights: list[float]) -> str:
    """Sample a single item from ``items`` proportionally to ``weights``."""

    threshold = rng.random() * sum(weights)
    running = 0.0
    for item, weight in zip(items, weights):
        running += weight
        if running >= threshold:
            return item
    return items[-1]


def _sample_weighted_index(rng: random.Random, cumulative_weights: list[float]) -> int:
    """Sample an index from precomputed cumulative weights."""

    if not cumulative_weights:
        raise ValueError("cumulative_weights must not be empty")
    target = rng.random() * cumulative_weights[-1]
    return int(bisect.bisect_left(cumulative_weights, target))


def _seasonal_wave(position: float) -> float:
    """Seasonality helper with two harmonics for smoother demand variation."""

    # Position expected in [0, 1]. Two yearly demand peaks with different amplitudes.
    return (
        torch.sin(torch.tensor(position * 6.283185307179586 * 2.0)).item() * 0.7
        + torch.sin(torch.tensor(position * 6.283185307179586 * 4.0)).item() * 0.3
    )


def _browse_distribution(activity: float) -> list[float]:
    """Map customer activity to browse-event type probabilities."""

    page_view = max(0.44, 0.60 - activity * 0.10)
    search = max(0.08, 0.14 - activity * 0.02)
    product_view = min(0.30, 0.16 + activity * 0.07)
    add_to_cart = min(0.24, 0.08 + activity * 0.09)
    total = page_view + search + product_view + add_to_cart
    return [
        page_view / total,
        search / total,
        product_view / total,
        add_to_cart / total,
    ]


def _sample_event_ts(rng: random.Random, start_ts: datetime, total_days: int) -> datetime:
    """Sample a timestamp with recency and intra-day activity bias."""

    day_offset = int(rng.triangular(0, total_days - 1, total_days * 0.72))
    hour = _weighted_choice(
        rng,
        [str(h) for h in range(24)],
        [
            0.02,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.02,
            0.03,
            0.05,
            0.06,
            0.06,
            0.06,
            0.05,
            0.05,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.09,
            0.07,
            0.05,
            0.03,
            0.02,
        ],
    )
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return start_ts + timedelta(days=day_offset, hours=int(hour), minutes=minute, seconds=second)


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for data generation scale and output locations."""

    parser = argparse.ArgumentParser(description="Build business raw data and customer event stream")
    parser.add_argument("--duckdb-path", type=Path, default=Path("data/duckdb/customer_event_stream.duckdb"))
    parser.add_argument("--arrow-path", type=Path, default=Path("data/arrow/customer_event_stream.feather"))
    parser.add_argument("--num-customers", type=int, default=120_000)
    parser.add_argument("--num-products", type=int, default=1_200)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--event-count", type=int, default=10_000_000)
    parser.add_argument("--order-ratio", type=float, default=0.08)
    parser.add_argument("--min-orders-per-customer", type=int, default=4)
    parser.add_argument("--activity-sampling-power", type=float, default=1.35)
    parser.add_argument("--lancedb-dir", type=Path, default=Path("scripts/data/lancedb"))
    parser.add_argument("--log-path", type=Path, default=Path("scripts/data/logs/generate_data.log"))
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for full synthetic data generation workflow."""

    args = _parse_args()
    args.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(log_path=args.log_path)

    try:
        reporter.log(f"Build log file: {args.log_path}")

        if args.duckdb_path.exists():
            reporter.log(f"Removing existing DuckDB: {args.duckdb_path}")
            args.duckdb_path.unlink()
        if args.lancedb_dir.exists():
            reporter.log(f"Removing existing LanceDB dir: {args.lancedb_dir}")
            shutil.rmtree(args.lancedb_dir)

        conn = duckdb.connect(str(args.duckdb_path))
        try:
            reporter.log("Creating DuckDB schema")
            create_business_tables(conn)
            seed_business_data(
                conn,
                num_customers=args.num_customers,
                num_products=args.num_products,
                years=args.years,
                event_count=args.event_count,
                order_ratio=args.order_ratio,
                min_orders_per_customer=args.min_orders_per_customer,
                activity_sampling_power=args.activity_sampling_power,
                seed=args.seed,
                reporter=reporter,
            )
            event_count = materialize_customer_event_stream(conn, reporter=reporter)
            customer_count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            web_count = conn.execute("SELECT COUNT(*) FROM website_browse").fetchone()[0]
            promotion_count = conn.execute("SELECT COUNT(*) FROM promotions").fetchone()[0]
            inventory_count = conn.execute("SELECT COUNT(*) FROM inventory_daily").fetchone()[0]
            first_rows = load_customer_event_stream(conn)[:5]
        finally:
            conn.close()
        # Primary data layer: Arrow IPC (zero-copy, memory-mappable).
        from rabbit_hole import stream as _stream
        reporter.log(f"Writing Arrow IPC stream -> {args.arrow_path}")
        _stream.to_arrow(str(args.duckdb_path), str(args.arrow_path))
    finally:
        reporter.close()

    print("Business event stream built")
    print(f"DuckDB: {args.duckdb_path}")
    print(f"Arrow: {args.arrow_path}")
    print(f"Customers: {customer_count}")
    print(f"Orders: {order_count}")
    print(f"Website browse events: {web_count}")
    print(f"Promotions: {promotion_count}")
    print(f"Inventory snapshots: {inventory_count}")
    print(f"Customer events: {event_count}")
    print("First customer events:")
    for row in first_rows:
        print(row)


if __name__ == "__main__":
    main()
