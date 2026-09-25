"""Browse and order activity generation for the synthetic commerce world."""

from __future__ import annotations

from datetime import datetime, timedelta
import random

import duckdb

from rabbit_hole.generators.business_tables import _bulk_insert, _flush_order_items
from rabbit_hole.generators.generate_support import (
    ProgressReporter,
    _browse_distribution,
    _sample_event_ts,
    _sample_weighted_index,
    _weighted_choice,
)


def _seed_events(
    conn: duckdb.DuckDBPyConnection,
    rng: random.Random,
    start_ts: datetime,
    total_days: int,
    reporter: ProgressReporter | None,
    activity_sampling_power: float,
    event_count: int,
    order_ratio: float,
    min_orders_per_customer: int,
    num_customers: int,
    promotion_rows: list,
    products_by_category: dict,
    product_price: dict,
    product_cost: dict,
    customer_ids: list,
    customer_activity: dict,
    customer_pref_category: dict,
    customer_country_state: dict,
    categories: list,
):
    """Generate website browse events and orders; return the running order counter."""
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

    browse_event_rows: list[
        tuple[str, str, str, str, str, str, str | None, str | None, str, str, int, int | None]
    ] = []
    order_rows: list[
        tuple[
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            str | None,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            int,
            str | None,
            float,
            str | None,
        ]
    ] = []
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
            category = (
                preferred if rng.random() < 0.62 else categories[rng.randrange(0, len(categories))]
            )
            product_id = products_by_category[category][
                rng.randrange(0, len(products_by_category[category]))
            ]
            quantity = None
        else:
            page_type = "cart"
            preferred = customer_pref_category[customer_id]
            category = (
                preferred if rng.random() < 0.74 else categories[rng.randrange(0, len(categories))]
            )
            product_id = products_by_category[category][
                rng.randrange(0, len(products_by_category[category]))
            ]
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
            _bulk_insert(
                conn,
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
        _bulk_insert(
            conn,
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
            category = (
                pref_category
                if rng.random() < 0.58
                else categories[rng.randrange(0, len(categories))]
            )
            basket_categories.append(category)
            product_id = products_by_category[category][
                rng.randrange(0, len(products_by_category[category]))
            ]
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
            cogs = (
                round(line_cogs * (1 - eff_return / gross_rev), 2)
                if gross_rev > 0
                else round(line_cogs, 2)
            )
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
            _bulk_insert(
                conn,
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
        _bulk_insert(
            conn,
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

    return order_counter
