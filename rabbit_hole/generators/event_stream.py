"""Customer event stream materialization and deterministic playback."""

from __future__ import annotations

import duckdb

from rabbit_hole.generators.generate_support import CustomerEventRow, ProgressReporter

# Columns of the canonical stream insert (brand is filled inline per source so
# no post-hoc full-table UPDATE pass is needed over tens of millions of rows).
_INSERT_COLS = """
    customer_key,
    event_ts,
    event_type,
    entity_type,
    entity_id,
    source_table,
    value,
    event_attributes,
    brand
"""


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

    total_stages = 8
    if reporter is not None:
        reporter.start("materialize customer events", total_stages)

    conn.execute(
        f"""
        INSERT INTO customer_events ({_INSERT_COLS})
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
            ),
            ''
        FROM customers c
        """
    )
    if reporter is not None:
        reporter.advance(1)

    conn.execute(
        f"""
        INSERT INTO customer_events ({_INSERT_COLS})
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
            ),
            COALESCE(p.brand, '')
        FROM website_browse wb
        LEFT JOIN products p ON p.product_id = wb.product_id
        """
    )
    if reporter is not None:
        reporter.advance(1)

    conn.execute(
        f"""
        INSERT INTO customer_events ({_INSERT_COLS})
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
            ),
            ''
        FROM orders o
        """
    )
    if reporter is not None:
        reporter.advance(1)

    conn.execute(
        f"""
        INSERT INTO customer_events ({_INSERT_COLS})
        SELECT
            o.customer_id,
            o.cancelled_ts,
            'order_cancelled',
            'order',
            o.transaction_id,
            'orders',
            o.order_total,
            json_object('order_status', o.order_status),
            ''
        FROM orders o
        WHERE o.cancelled_ts IS NOT NULL
        """
    )
    if reporter is not None:
        reporter.advance(1)

    conn.execute(
        f"""
        INSERT INTO customer_events ({_INSERT_COLS})
        SELECT
            o.customer_id,
            o.return_ts,
            'order_returned',
            'order',
            o.transaction_id,
            'orders',
            o.return_amount,
            json_object('return_flag', o.return_flag, 'order_status', o.order_status),
            ''
        FROM orders o
        WHERE o.return_flag = 1 AND o.return_ts IS NOT NULL
        """
    )
    if reporter is not None:
        reporter.advance(1)

    # --- email action events: send -> open -> click ---
    conn.execute(
        f"""
        INSERT INTO customer_events ({_INSERT_COLS})
        SELECT
            customer_id, send_ts, channel || '_send', channel, send_id, 'contact_sends', 0,
            json_object('campaign_id', campaign_id, 'opened', opened, 'clicked', clicked),
            ''
        FROM contact_sends
        """
    )
    if reporter is not None:
        reporter.advance(1)
    conn.execute(
        f"""
        INSERT INTO customer_events ({_INSERT_COLS})
        SELECT
            customer_id, open_ts, channel || '_open', channel, send_id, 'contact_sends', 0,
            json_object('campaign_id', campaign_id),
            ''
        FROM contact_sends
        WHERE opened = 1 AND open_ts IS NOT NULL
        """
    )
    if reporter is not None:
        reporter.advance(1)
    conn.execute(
        f"""
        INSERT INTO customer_events ({_INSERT_COLS})
        SELECT
            customer_id, click_ts, channel || '_click', 'session',
            COALESCE(click_session_id, send_id), 'contact_sends', 0,
            json_object(
                'campaign_id', campaign_id,
                'click_session_id', click_session_id,
                'converted_order_id', converted_order_id
            ),
            ''
        FROM contact_sends
        WHERE clicked = 1 AND click_ts IS NOT NULL
        """
    )
    if reporter is not None:
        reporter.advance(1)

    # DuckDB autocommits.
    event_count = int(conn.execute("SELECT COUNT(*) FROM customer_events").fetchone()[0])
    if reporter is not None:
        reporter.finish(f"rows={event_count:,}")
    return event_count


def load_customer_event_stream(
    conn: duckdb.DuckDBPyConnection,
    customer_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[CustomerEventRow]:
    """Load event stream rows ordered for deterministic per-customer playback.

    ``limit`` bounds the scan (used by the CLI receipt, which only prints the
    first few rows); ``None`` loads the full stream.
    """

    limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
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
            ORDER BY customer_key, event_ts, event_id{limit_sql}
            """,
            customer_ids,
        ).fetchall()
    else:
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
            ORDER BY customer_key, event_ts, event_id{limit_sql}
            """
        ).fetchall()
    return [CustomerEventRow(*row) for row in rows]
