"""Browse and order activity generation for the synthetic commerce world (vectorized)."""

from __future__ import annotations

from datetime import datetime

import duckdb
import numpy as np
import polars as pl

from rabbit_hole.generators.business_tables import _bulk_insert, _flush_order_items
from rabbit_hole.generators.generate_support import (
    ProgressReporter,
    browse_probs,
    customer_arrays,
    iso_expr,
    product_arrays,
    sample_categorical,
    sample_event_ts,
    sample_from_cum,
)

_DEVICE_TYPES = ["mobile", "desktop", "tablet"]
_DEVICE_WEIGHTS = [0.58, 0.36, 0.06]
_TRAFFIC_SOURCES = ["direct", "organic", "paid_search", "email", "social", "affiliate"]
_TRAFFIC_WEIGHTS = [0.17, 0.31, 0.18, 0.12, 0.15, 0.07]
_PAGE_TYPES = ["home", "category", "campaign", "help"]  # page_view destinations
_EVENT_NAMES = ["page_view", "search", "product_view", "add_to_cart"]
_PAGE_TYPE_CODES = {
    0: "home",
    1: "category",
    2: "campaign",
    3: "help",
    4: "search",
    5: "product",
    6: "cart",
}
_ORDER_STATUS = ["completed", "cancelled", "refunded"]
_ORDER_STATUS_WEIGHTS = [0.935, 0.038, 0.027]
_PAYMENTS = ["credit_card", "debit_card", "paypal", "wallet", "gift_card"]
_PAYMENT_WEIGHTS = [0.49, 0.23, 0.16, 0.09, 0.03]

_BROWSE_CHUNK = 4_000_000


def _seed_events(
    conn: duckdb.DuckDBPyConnection,
    rng: np.random.Generator,
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
    activity, pref_idx, ship_country, ship_state = customer_arrays(
        customer_ids, customer_activity, customer_pref_category, customer_country_state, categories
    )
    flat_products, prod_offsets, prod_prices, prod_costs = product_arrays(
        products_by_category, categories, product_price, product_cost
    )
    prod_sizes = np.diff(prod_offsets)
    customer_col = np.array(customer_ids, dtype=object)
    promo_ids = np.array([r[0] for r in promotion_rows], dtype=object)
    n_promo = len(promo_ids)

    # Enforce enough order activity to support downstream outcome labels.
    target_orders = max(int(event_count * order_ratio), num_customers * min_orders_per_customer, 1)
    target_browse = max(event_count - target_orders, num_customers)

    sampling_weights = np.maximum(activity, 0.05) ** activity_sampling_power
    cum_weights = np.cumsum(sampling_weights)

    n_cats = len(categories)
    apparel = categories.index("apparel") if "apparel" in categories else -1
    electronics = categories.index("electronics") if "electronics" in categories else -1

    # ---------------------------------------------------------------- browse
    b_total = target_browse
    if reporter is not None:
        reporter.start("generate browse events", b_total)
    cust = sample_from_cum(rng, cum_weights, b_total)
    base_ts = sample_event_ts(rng, start_ts, total_days, b_total)

    # Per-customer generation-order segments: first event of a customer is a new
    # session, later events open a new session with an activity-dependent chance.
    sidx = np.argsort(cust, kind="stable")
    sorted_cust = cust[sidx]
    change = np.empty(b_total, dtype=bool)
    change[0] = True
    change[1:] = sorted_cust[1:] != sorted_cust[:-1]
    seg_end_change = np.empty(b_total, dtype=bool)
    seg_end_change[:-1] = sorted_cust[1:] != sorted_cust[:-1]
    seg_end_change[-1] = True
    seg_cust = sorted_cust[change]

    is_first = np.zeros(b_total, dtype=bool)
    is_first[sidx[change]] = True
    p_new = np.clip(0.22 + (1.0 - activity[cust]) * 0.08, 0.08, 0.34)
    new_session = is_first | (rng.random(b_total) < p_new)
    # Globally unique session labels: sessions belong to exactly one customer
    # (segmented cumsum), never shared across interleaved customers.
    csum_sorted = np.cumsum(new_session[sidx], dtype=np.int64)
    sess_num = np.empty(b_total, dtype=np.int64)
    sess_num[sidx] = csum_sorted - 1
    total_sessions = int(csum_sorted[-1])

    # Per-customer first timestamp and last session (generation order).
    first_ts = np.full(num_customers, np.datetime64("NaT", "ms"))
    first_ts[seg_cust] = base_ts[sidx[change]]
    last_sess = np.full(num_customers, -1, dtype=np.int64)
    last_sess[seg_cust] = sess_num[sidx[seg_end_change]]

    # Event type from the activity-driven browse distribution.
    event_code = sample_categorical(rng, browse_probs(activity[cust]))
    page_draw = rng.integers(0, len(_PAGE_TYPES), b_total)
    page_code = np.where(
        event_code == 0, page_draw, np.where(event_code == 1, 4, np.where(event_code == 2, 5, 6))
    )
    want_pref = np.where(
        event_code == 2,
        rng.random(b_total) < 0.62,
        np.where(event_code == 3, rng.random(b_total) < 0.74, False),
    )
    cat_sel = np.where(want_pref, pref_idx[cust], rng.integers(0, n_cats, b_total))
    prod_pos = prod_offsets[cat_sel] + np.minimum(
        (rng.random(b_total) * prod_sizes[cat_sel]).astype(np.int64), prod_sizes[cat_sel] - 1
    )
    product_col = flat_products[prod_pos]
    has_product = (event_code == 2) | (event_code == 3)
    quantity = rng.integers(1, 4, b_total)  # add_to_cart basket size
    has_promo = rng.random(b_total) < 0.09
    promo_sel = rng.integers(0, max(n_promo, 1), b_total)
    device = np.array(_DEVICE_TYPES, dtype=object)[
        sample_categorical(rng, _DEVICE_WEIGHTS, b_total)
    ]
    traffic = np.array(_TRAFFIC_SOURCES, dtype=object)[
        sample_categorical(rng, _TRAFFIC_WEIGHTS, b_total)
    ]
    dwell = np.maximum(rng.gamma(2.3, 18.0, b_total).astype(np.int64), 2)

    browse_insert = """
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
    """
    browse_written = 0
    for i0 in range(0, b_total, _BROWSE_CHUNK):
        i1 = min(i0 + _BROWSE_CHUNK, b_total)
        df = (
            pl.DataFrame(
                {
                    "browse_idx": np.arange(i0, i1, dtype=np.int64),
                    "sess_num": sess_num[i0:i1],
                    "customer_id": customer_col[cust[i0:i1]],
                    "event_ts": base_ts[i0:i1],
                    "event_code": event_code[i0:i1],
                    "page_code": page_code[i0:i1],
                    "product_id": product_col[i0:i1],
                    "has_product": has_product[i0:i1],
                    "promotion_id": promo_ids[promo_sel[i0:i1]],
                    "has_promo": has_promo[i0:i1],
                    "device": device[i0:i1],
                    "traffic": traffic[i0:i1],
                    "dwell": dwell[i0:i1],
                    "quantity": quantity[i0:i1],
                }
            )
            .with_columns(
                (pl.lit("BROWSE_") + pl.col("browse_idx").cast(pl.String).str.zfill(12)).alias(
                    "browse_event_id"
                ),
                (pl.lit("SESS_") + pl.col("sess_num").cast(pl.String).str.zfill(10)).alias(
                    "session_id"
                ),
                iso_expr("event_ts"),
                pl.col("event_code")
                .replace_strict({i: n for i, n in enumerate(_EVENT_NAMES)}, default=None)
                .alias("event_name"),
                pl.col("page_code")
                .replace_strict(_PAGE_TYPE_CODES, default=None)
                .alias("page_type"),
                pl.when(pl.col("has_product"))
                .then(pl.col("product_id"))
                .otherwise(None)
                .alias("product_id"),
                pl.when(pl.col("has_promo"))
                .then(pl.col("promotion_id"))
                .otherwise(None)
                .alias("promotion_id"),
                pl.when(pl.col("event_code") == 3)
                .then(pl.col("quantity"))
                .otherwise(None)
                .alias("quantity"),
            )
            .select(
                "browse_event_id",
                "session_id",
                "customer_id",
                "event_ts",
                "event_name",
                "page_type",
                "product_id",
                "promotion_id",
                "device",
                "traffic",
                "dwell",
                "quantity",
            )
        )
        df = df.rename(
            {"device": "device_type", "traffic": "traffic_source", "dwell": "dwell_seconds"}
        )
        _bulk_insert(conn, browse_insert, df)
        browse_written += i1 - i0
        if reporter is not None:
            reporter.advance(i1 - i0)
    if reporter is not None:
        reporter.finish(f"rows={browse_written:,} sessions={total_sessions:,}")

    # --------------------------------------------------------------- orders
    o_total = target_orders
    if reporter is not None:
        reporter.start("generate orders", o_total)
    ocust = sample_from_cum(rng, cum_weights, o_total)

    # Orders reuse the customer's last browse session; customers without browse
    # history get a fresh session.
    last = last_sess[ocust]
    need_session = last < 0
    if need_session.any():
        miss_cust = np.unique(ocust[need_session])
        new_ids = total_sessions + np.arange(len(miss_cust), dtype=np.int64)
        last = np.where(
            need_session, new_ids[np.searchsorted(miss_cust, ocust[need_session])], last
        )
        total_sessions += len(miss_cust)

    base_order_ts = sample_event_ts(rng, start_ts, total_days, o_total)
    first = first_ts[ocust]
    need_floor = (~np.isnat(first)) & (base_order_ts < first)
    floored = first + rng.integers(1, 1441, o_total).astype("timedelta64[m]")
    order_ts = np.where(need_floor, floored, base_order_ts)
    order_ts = order_ts + rng.integers(5, 421, o_total).astype("timedelta64[m]")

    u1 = rng.random(o_total)
    u2 = rng.random(o_total)
    n_lines = 1 + (u1 >= 0.65) + ((u1 >= 0.65) & (u2 >= 0.9))
    line_order = np.repeat(np.arange(o_total, dtype=np.int64), n_lines)
    n_line_total = int(n_lines.sum())
    line_cust = ocust[line_order]
    line_pref = np.where(
        rng.random(n_line_total) < 0.58,
        pref_idx[line_cust],
        rng.integers(0, n_cats, n_line_total),
    )
    line_pos = prod_offsets[line_pref] + np.minimum(
        (rng.random(n_line_total) * prod_sizes[line_pref]).astype(np.int64),
        prod_sizes[line_pref] - 1,
    )
    line_pid = flat_products[line_pos]
    line_price = prod_prices[line_pos]
    line_cost = prod_costs[line_pos]
    line_units = np.where(rng.random(n_line_total) < 0.74, 1, rng.integers(2, 5, n_line_total))
    line_rev = line_price * line_units
    line_cogs_raw = line_cost * line_units

    starts = np.zeros(o_total, dtype=np.int64)
    np.cumsum(n_lines[:-1], out=starts[1:])
    subtotal = np.add.reduceat(line_rev, starts)
    line_cogs = np.add.reduceat(line_cogs_raw, starts)
    has_apparel = np.maximum.reduceat((line_pref == apparel).astype(np.int8), starts).astype(bool)
    has_electronics = np.maximum.reduceat(
        (line_pref == electronics).astype(np.int8), starts
    ).astype(bool)

    has_promo_o = rng.random(o_total) < 0.19
    promo_sel_o = rng.integers(0, max(n_promo, 1), o_total)
    promo_pct = np.array([float(r[5]) for r in promotion_rows])
    promo_min = np.array([float(r[6]) for r in promotion_rows])
    pct = promo_pct[promo_sel_o]
    min_cart = promo_min[promo_sel_o]
    discount = np.where(has_promo_o & (subtotal >= min_cart), np.round(subtotal * pct, 2), 0.0)
    tax = np.round(np.maximum(subtotal - discount, 0.0) * rng.uniform(0.06, 0.095, o_total), 2)
    shipping_fee = np.where(subtotal >= 75.0, 0.0, np.round(rng.uniform(3.49, 8.99, o_total), 2))
    order_total = np.round(np.maximum(subtotal - discount, 0.0) + tax + shipping_fee, 2)

    order_status = np.array(_ORDER_STATUS, dtype=object)[
        sample_categorical(rng, _ORDER_STATUS_WEIGHTS, o_total)
    ]
    is_cancelled = order_status == "cancelled"
    cancelled_ts = np.where(
        is_cancelled,
        order_ts + rng.integers(2, 91, o_total).astype("timedelta64[m]"),
        np.datetime64("NaT", "ms"),
    ).astype("datetime64[ms]")

    in_return_window = (order_status == "completed") | (order_status == "refunded")
    bias = 0.08 + 0.10 * has_apparel + 0.03 * has_electronics
    return_flag = in_return_window & (rng.random(o_total) < np.minimum(0.35, bias))
    return_ts = np.where(
        return_flag,
        order_ts + rng.integers(2, 66, o_total).astype("timedelta64[D]"),
        np.datetime64("NaT", "ms"),
    ).astype("datetime64[ms]")
    full_refund = rng.random(o_total) < 0.64
    return_amount = np.where(
        return_flag & full_refund,
        order_total,
        np.where(return_flag, np.round(order_total * rng.uniform(0.15, 0.55, o_total), 2), 0.0),
    )
    order_status = np.where(return_flag & full_refund, "refunded", order_status)

    payment = np.array(_PAYMENTS, dtype=object)[sample_categorical(rng, _PAYMENT_WEIGHTS, o_total)]
    gross_rev = np.maximum(subtotal - discount, 0.0)
    eff_return = np.where(return_amount > 0, np.minimum(return_amount, gross_rev), 0.0)
    revenue = np.where(is_cancelled, 0.0, np.round(gross_rev - eff_return, 2))
    eff_ratio = np.divide(eff_return, gross_rev, out=np.zeros(o_total), where=gross_rev > 0)
    cogs_full = np.round(line_cogs * (1.0 - eff_ratio), 2)
    cogs = np.where(is_cancelled, 0.0, np.where(gross_rev > 0, cogs_full, np.round(line_cogs, 2)))
    gross_margin = np.round(revenue - cogs, 2)

    txn = np.arange(o_total, dtype=np.int64)  # first order counter value is 0
    orders_df = pl.DataFrame(
        {
            "txn_idx": txn,
            "customer_id": customer_col[ocust],
            "session_idx": last,
            "order_ts": order_ts,
            "order_status": order_status,
            "payment_method": payment,
            "shipping_country": ship_country[ocust],
            "shipping_state": ship_state[ocust],
            "promotion_id": np.where(has_promo_o, promo_ids[promo_sel_o], None).tolist(),
            "subtotal": np.round(subtotal, 2),
            "tax": tax,
            "shipping_fee": shipping_fee,
            "discount_amount": np.round(discount, 2),
            "order_total": order_total,
            "revenue": revenue,
            "cogs": cogs,
            "gross_margin": gross_margin,
            "return_flag": return_flag.astype(np.int64),
            "return_ts": return_ts,
            "return_amount": return_amount,
            "cancelled_ts": cancelled_ts,
        }
    ).with_columns(
        (pl.lit("TXN_") + pl.col("txn_idx").cast(pl.String).str.zfill(12)).alias("transaction_id"),
        (pl.lit("SESS_") + pl.col("session_idx").cast(pl.String).str.zfill(10)).alias("session_id"),
        iso_expr("order_ts"),
        iso_expr("return_ts"),
        iso_expr("cancelled_ts"),
    )
    txn_col = orders_df["transaction_id"].to_numpy()
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
        orders_df,
    )
    _flush_order_items(
        conn,
        {
            "transaction_id": txn_col[line_order],
            "product_id": line_pid,
            "quantity": line_units,
            "unit_price": np.round(line_price, 2),
            "unit_cost": np.round(line_cost, 2),
            "line_revenue": np.round(line_rev, 2),
            "line_cogs": np.round(line_cogs_raw, 2),
        },
    )
    if reporter is not None:
        reporter.advance(o_total)
        reporter.finish(f"rows={o_total:,}")

    return o_total
