"""Wide-table DuckDB DDL and bulk INSERT helpers."""

from __future__ import annotations

import duckdb


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
            f"INSERT INTO {table} ({', '.join(cols)}) SELECT {', '.join(cols)} FROM _rh_bulk"
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


def _flush_contact(conn, rows) -> int:
    """Bulk-insert multi-channel contacts (no-op when empty)."""
    return _bulk_insert(
        conn,
        """
        INSERT INTO contact_sends (channel, send_id, customer_id, period, campaign_id,
            send_ts, opened, clicked, open_ts, click_ts, click_session_id,
            converted_order_id, arm, propensity, discount_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    _exec_script(
        conn,
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

        CREATE TABLE contact_sends (
            channel TEXT NOT NULL,
            send_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            period INTEGER,
            campaign_id TEXT,
            send_ts TEXT NOT NULL,
            opened INTEGER NOT NULL DEFAULT 0,
            clicked INTEGER NOT NULL DEFAULT 0,
            open_ts TEXT,
            click_ts TEXT,
            click_session_id TEXT,
            converted_order_id TEXT,
            arm INTEGER,
            propensity REAL,
            discount_pct REAL DEFAULT 0,
            PRIMARY KEY (channel, send_id)
        );

        CREATE TABLE contact_holdout (
            channel TEXT NOT NULL, customer_id TEXT NOT NULL, period INTEGER NOT NULL,
            holdout INTEGER NOT NULL, PRIMARY KEY (channel, customer_id, period)
        );

        CREATE TABLE contact_arm (
            channel TEXT NOT NULL, customer_id TEXT NOT NULL, period INTEGER NOT NULL,
            arm INTEGER NOT NULL, propensity REAL NOT NULL, optimal_arm INTEGER,
            PRIMARY KEY (channel, customer_id, period)
        );

        CREATE VIEW email_sends AS SELECT send_id, customer_id, campaign_id, send_ts,
            opened, clicked, open_ts, click_ts, click_session_id, converted_order_id,
            arm, propensity, discount_pct FROM contact_sends WHERE channel = 'email';
        CREATE VIEW email_holdout AS SELECT customer_id, period, holdout
            FROM contact_holdout WHERE channel = 'email';
        CREATE VIEW email_arm AS SELECT customer_id, period, arm, propensity, optimal_arm
            FROM contact_arm WHERE channel = 'email';

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
        """,
    )
