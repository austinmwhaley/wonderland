"""Multi-channel contact seeding (email/sms/push) with logged randomized arms.

All per-customer/per-send Python loops are replaced by batched numpy draws and
polars column expressions; the generative parameters (holdout rate, arm softmax,
cadence -> sends, open/click/conversion probabilities) are unchanged.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import numpy as np
import polars as pl

from rabbit_hole.generators.business_tables import (
    _bulk_insert,
    _flush_contact,
    _flush_order_items,
)
from rabbit_hole.generators.generate_support import (
    ISO_US,
    ProgressReporter,
    customer_arrays,
    day_of_year,
    iso_expr,
    product_arrays,
    sample_categorical,
    softmax,
    utc_naive,
)

_N_PERIODS = 12
_HOLD_FRAC = 0.05
_ARMS = 4
_SEND_CHUNK = 4_000_000

# MULTI-CHANNEL contacts (email/sms/push): per-period cadence arm (observational,
# confounded by latent `intent`), 5% persistent holdout, and a DISCOUNT action.
_CHANNELS = {
    "email": dict(cadence=(0.2, 0.6, 1.2, 2.0), peak=0.10, o0=0.12, oa=0.55, c0=0.06, ca=0.45),
    "sms": dict(cadence=(0.1, 0.3, 0.8, 1.5), peak=0.07, o0=0.25, oa=0.35, c0=0.12, ca=0.35),
    "push": dict(cadence=(0.5, 1.5, 3.0, 5.0), peak=0.05, o0=0.10, oa=0.40, c0=0.05, ca=0.30),
}
_DISCOUNTS = np.array([0.0, 5.0, 10.0, 15.0])
_PAYMENTS = ["credit_card", "debit_card", "paypal", "wallet", "gift_card"]
_PAYMENT_WEIGHTS = [0.49, 0.23, 0.16, 0.09, 0.03]

_ORDERS_INSERT = """INSERT INTO orders (transaction_id, customer_id, session_id,
    order_ts, order_status, payment_method, shipping_country, shipping_state,
    promotion_id, subtotal, tax, shipping_fee, discount_amount, order_total,
    revenue, cogs, gross_margin, return_flag, return_ts, return_amount, cancelled_ts)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def _session_state(conn, customer_ids, activity):
    """Flat per-customer session index + conversion lookup for click attribution.

    Sessions are grouped by customer (``starts``/``ends`` offsets); ``cum`` is
    the global cumulative of the 1 + 2*activity*converted session sampling
    weights; ``txn_arr`` carries a pre-existing order per session, if any.
    """
    rows = conn.execute(
        "SELECT customer_id, session_id FROM website_browse GROUP BY 1, 2 ORDER BY 1, 2"
    ).fetchall()
    cid_to_idx = {cid: i for i, cid in enumerate(customer_ids)}
    nc = len(customer_ids)
    sess_cust = np.array([cid_to_idx[c] for c, _ in rows], dtype=np.int64)
    sid_list = [s for _, s in rows]
    n_sess = len(sid_list)
    starts = np.searchsorted(sess_cust, np.arange(nc, dtype=np.int64), side="left")
    ends = np.searchsorted(sess_cust, np.arange(nc, dtype=np.int64), side="right")
    sid_arr = np.array(sid_list, dtype=object)

    if not n_sess:
        return starts, ends, sid_arr, np.empty(0), np.empty(0, dtype=object)

    conv_rows = conn.execute(
        "SELECT customer_id, session_id, transaction_id FROM orders "
        "WHERE session_id IS NOT NULL ORDER BY transaction_id"
    ).fetchall()
    if conv_rows:
        conv_df = pl.DataFrame(
            {
                "ci": np.array([cid_to_idx[c] for c, _, _ in conv_rows], dtype=np.int64),
                "sid": [s for _, s, _ in conv_rows],
                "txn": [t for _, _, t in conv_rows],
            }
        ).unique(subset=["ci", "sid"], keep="last", maintain_order=True)
        sess_df = pl.DataFrame({"ci": sess_cust, "sid": sid_list})
        joined = sess_df.join(conv_df, on=["ci", "sid"], how="left", maintain_order="left")
        txn_arr = joined["txn"].to_numpy()
        converted = joined["txn"].is_not_null().to_numpy()
    else:
        txn_arr = np.full(n_sess, None, dtype=object)
        converted = np.zeros(n_sess, dtype=bool)

    w_flat = np.where(converted, 1.0 + 2.0 * activity[sess_cust], 1.0)
    cum = np.cumsum(w_flat)
    return starts, ends, sid_arr, cum, txn_arr


def _seed_contacts(
    conn: duckdb.DuckDBPyConnection,
    rng: np.random.Generator,
    start_ts: datetime,
    total_days: int,
    num_customers: int,
    order_counter: int,
    reporter: ProgressReporter | None,
    promotion_rows: list,
    products_by_category: dict,
    product_price: dict,
    product_cost: dict,
    customer_ids: list,
    customer_activity: dict,
    customer_country_state: dict,
    customer_pref_category: dict,
):
    """Generate contact sends, holdouts, and arm assignments across channels."""
    # ------------------------------------------------------------------
    # email interactions: RANDOMIZED arm (cadence) with logged propensity and a
    # KNOWN causal effect on orders. Overlap identifies the counterfactual; the
    # known effect forecasts ground truth so red_king can be validated.
    # ------------------------------------------------------------------
    # Per-customer UNIMODAL response: each customer has an optimal cadence x0
    # tied to activity (so the optimal arm DIFFERS by customer and is learnable
    # from the donor state). Per-click incremental-order rate peaks at x=x0.
    nc = num_customers
    categories = list(products_by_category.keys())
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

    starts, ends, sid_arr, cum, sess_txn = _session_state(conn, customer_ids, activity)

    period_days = max(total_days / _N_PERIODS, 1.0)
    period_weeks = period_days / 7.0
    start_anchor = utc_naive(start_ts)
    intents = rng.normal(0.0, 1.0, nc)
    arm_axis = np.arange(_ARMS, dtype=np.float64)

    for chan, ccfg in _CHANNELS.items():
        cadence = np.asarray(ccfg["cadence"], dtype=np.float64)

        # ---------------- arm assignment + holdout (per customer x period) ----
        hold = rng.random((nc, _N_PERIODS)) < _HOLD_FRAC
        noise = rng.normal(0.0, 0.8, (nc, _N_PERIODS, _ARMS))
        base = (activity + 0.6 * intents)[:, None] * arm_axis[None, :]
        pp = softmax(base[:, None, :] + noise, axis=-1)
        arm = np.where(hold, -1, sample_categorical(rng, pp)).astype(np.int64)
        prop = np.take_along_axis(pp, arm[..., None].clip(min=0), axis=-1)[..., 0]
        x0c = cadence[0] + (cadence[-1] - cadence[0]) * np.clip(activity, 0.0, 1.0)
        opt = np.argmax(cadence[None, :] * np.exp(-cadence[None, :] / x0c[:, None]), axis=-1)

        ci_rep = np.repeat(np.arange(nc, dtype=np.int64), _N_PERIODS)
        per_rep = np.tile(np.arange(_N_PERIODS, dtype=np.int64), nc)
        hold_flat = hold.ravel()
        arm_flat = arm.ravel()
        prop_flat = prop.ravel()
        _bulk_insert(
            conn,
            "INSERT INTO contact_holdout (channel, customer_id, period, holdout) VALUES (?,?,?,?)",
            {
                "channel": np.repeat(chan, hold_flat.size),
                "customer_id": customer_col[ci_rep],
                "period": per_rep,
                "holdout": hold_flat.astype(np.int64),
            },
        )
        arm_mask = ~hold_flat
        if arm_mask.any():
            _bulk_insert(
                conn,
                "INSERT INTO contact_arm (channel, customer_id, period, arm, propensity, "
                "optimal_arm) VALUES (?,?,?,?,?,?)",
                {
                    "channel": np.repeat(chan, int(arm_mask.sum())),
                    "customer_id": customer_col[ci_rep[arm_mask]],
                    "period": per_rep[arm_mask],
                    "arm": arm_flat[arm_mask],
                    "propensity": prop_flat[arm_mask],
                    "optimal_arm": opt[ci_rep[arm_mask]],
                },
            )

        # ---------------- sends ------------------------------------------------
        wts = np.where(arm >= 0, cadence[arm.clip(min=0)] * period_weeks, 0.0)
        tot = wts.sum(axis=1)
        n_sends = np.clip(rng.normal(tot, np.sqrt(np.maximum(tot, 1.0))).astype(np.int64), 0, 1500)
        n_sends = np.where(tot > 0, n_sends, 0)
        s_total = int(n_sends.sum())
        send_cust = np.repeat(np.arange(nc, dtype=np.int64), n_sends)
        cw = np.cumsum(wts, axis=1)

        send_counter = 0
        if reporter is not None:
            reporter.start(f"generate {chan}", max(s_total, 1))
        for j0 in range(0, s_total, _SEND_CHUNK):
            j1 = min(j0 + _SEND_CHUNK, s_total)
            cs = send_cust[j0:j1]
            m = j1 - j0

            t = rng.random(m) * tot[cs]
            pidx = np.minimum((cw[cs] <= t[:, None]).sum(axis=1), _N_PERIODS - 1)
            arm_cs = arm[cs, pidx]
            prop_cs = prop[cs, pidx]

            days = pidx * period_days + rng.random(m) * period_days
            send_dt = start_anchor + (days * 86_400_000_000.0).astype("timedelta64[us]")

            du = (activity[cs] + 0.3 * intents[cs])[:, None] * (_DISCOUNTS / 15.0)[None, :]
            du = du + rng.normal(0.0, 0.7, (m, _ARMS))
            disc = _DISCOUNTS[sample_categorical(rng, softmax(du, axis=-1))]
            campaign = promo_ids[rng.integers(0, max(n_promo, 1), m)]

            opened = rng.random(m) < np.minimum(0.9, ccfg["o0"] + ccfg["oa"] * activity[cs])
            open_dt = np.full(m, np.datetime64("NaT", "us"), dtype="datetime64[us]")
            clicked = np.zeros(m, dtype=bool)
            opened_idx = np.flatnonzero(opened)
            if opened_idx.size:
                open_dt[opened_idx] = send_dt[opened_idx] + rng.integers(
                    3, 2881, opened_idx.size
                ).astype("timedelta64[m]")
                p_click = np.minimum(0.8, ccfg["c0"] + ccfg["ca"] * activity[cs[opened_idx]])
                clicked[opened_idx[rng.random(opened_idx.size) < p_click]] = True
            clicked_idx = np.flatnonzero(clicked)
            click_dt = np.full(m, np.datetime64("NaT", "us"), dtype="datetime64[us]")
            if clicked_idx.size:
                click_dt[clicked_idx] = open_dt[clicked_idx] + rng.integers(
                    1, 121, clicked_idx.size
                ).astype("timedelta64[m]")

            click_sid = np.full(m, None, dtype=object)
            conv_txn = np.full(m, None, dtype=object)
            if clicked_idx.size and cum.size:
                cust_k = cs[clicked_idx]
                s0 = starts[cust_k]
                s1 = ends[cust_k]
                has_sess = s1 > s0
                if has_sess.any():
                    s0s = s0[has_sess]
                    s1s = s1[has_sess]
                    u_s = rng.random(int(has_sess.sum()))
                    base_w = np.where(s0s > 0, cum[s0s - 1], 0.0)
                    target = base_w + u_s * (cum[s1s - 1] - base_w)
                    j = np.searchsorted(cum, target, side="left")
                    j = np.clip(j, s0s, s1s - 1)
                    sub = clicked_idx[has_sess]
                    click_sid[sub] = sid_arr[j]
                    conv_txn[sub] = sess_txn[j]

            conv_mask = np.zeros(m, dtype=bool)
            if clicked_idx.size:
                ck = cs[clicked_idx]
                x = cadence[arm_cs[clicked_idx].clip(min=0)]
                x0 = cadence[0] + (cadence[-1] - cadence[0]) * np.clip(activity[ck], 0.0, 1.0)
                season = 1.0 + 0.25 * np.sin(
                    2.0 * np.pi * day_of_year(send_dt[clicked_idx]) / 365.0
                )
                p_conv = (
                    ccfg["peak"]
                    * np.exp(1.0 - x / x0)
                    * np.exp(0.5 * intents[ck])
                    * season
                    * (1.0 + 0.04 * disc[clicked_idx])
                )
                conv_mask[clicked_idx[rng.random(clicked_idx.size) < p_conv]] = True

            conv_pos = np.flatnonzero(conv_mask)
            if conv_pos.size:
                cc = cs[conv_pos]
                pcat = pref_idx[cc]
                ppos = prod_offsets[pcat] + np.minimum(
                    (rng.random(conv_pos.size) * prod_sizes[pcat]).astype(np.int64),
                    prod_sizes[pcat] - 1,
                )
                prod = flat_products[ppos]
                price = prod_prices[ppos]
                cost = prod_costs[ppos]
                txns = order_counter + np.arange(conv_pos.size, dtype=np.int64)
                ots = click_dt[conv_pos] + rng.integers(5, 181, conv_pos.size).astype(
                    "timedelta64[m]"
                )
                disc_amt = np.round(price * disc[conv_pos] / 100.0, 2)
                subtotal = np.round(price - disc_amt, 2)
                tax = np.round(subtotal * rng.uniform(0.06, 0.095, conv_pos.size), 2)
                ship = np.where(
                    subtotal >= 75.0, 0.0, np.round(rng.uniform(3.49, 8.99, conv_pos.size), 2)
                )
                total = np.round(subtotal + tax + ship, 2)
                payment = np.array(_PAYMENTS, dtype=object)[
                    sample_categorical(rng, _PAYMENT_WEIGHTS, conv_pos.size)
                ]
                conv_df = pl.DataFrame(
                    {
                        "txn_idx": txns,
                        "customer_id": customer_col[cc],
                        "session_id": click_sid[conv_pos].tolist(),
                        "order_dt": ots,
                        "payment_method": payment,
                        "shipping_country": ship_country[cc],
                        "shipping_state": ship_state[cc],
                        "subtotal": subtotal,
                        "tax": tax,
                        "shipping_fee": ship,
                        "discount_amount": disc_amt,
                        "order_total": total,
                        "revenue": subtotal,
                        "cogs": np.round(cost, 2),
                        "gross_margin": np.round(subtotal - cost, 2),
                    }
                ).with_columns(
                    (pl.lit("TXN_") + pl.col("txn_idx").cast(pl.String).str.zfill(12)).alias(
                        "transaction_id"
                    ),
                    iso_expr("order_dt").alias("order_ts"),
                    pl.lit("completed").alias("order_status"),
                    pl.lit(None, dtype=pl.String).alias("promotion_id"),
                    pl.lit(0, dtype=pl.Int64).alias("return_flag"),
                    pl.lit(None, dtype=pl.String).alias("return_ts"),
                    pl.lit(0.0).alias("return_amount"),
                    pl.lit(None, dtype=pl.String).alias("cancelled_ts"),
                )
                txn_list = conv_df["transaction_id"].to_list()
                conv_txn[conv_pos] = txn_list
                _bulk_insert(conn, _ORDERS_INSERT, conv_df)
                _flush_order_items(
                    conn,
                    {
                        "transaction_id": txn_list,
                        "product_id": prod,
                        "quantity": np.ones(conv_pos.size, dtype=np.int64),
                        "unit_price": np.round(price, 2),
                        "unit_cost": np.round(cost, 2),
                        "line_revenue": np.round(price, 2),
                        "line_cogs": np.round(cost, 2),
                    },
                )
                order_counter += conv_pos.size

            send_df = pl.DataFrame(
                {
                    "channel": np.repeat(chan, m),
                    "send_offset": np.arange(send_counter, send_counter + m, dtype=np.int64),
                    "customer_id": customer_col[cs],
                    "period": pidx,
                    "campaign_id": campaign,
                    "send_dt": send_dt,
                    "opened": opened.astype(np.int64),
                    "clicked": clicked.astype(np.int64),
                    "open_dt": open_dt,
                    "click_dt": click_dt,
                    "click_session_id": click_sid.tolist(),
                    "converted_order_id": conv_txn.tolist(),
                    "arm": arm_cs,
                    "propensity": prop_cs,
                    "discount_pct": disc,
                }
            ).with_columns(
                (
                    pl.lit(f"{chan.upper()}_") + pl.col("send_offset").cast(pl.String).str.zfill(12)
                ).alias("send_id"),
                iso_expr("send_dt", ISO_US).alias("send_ts"),
                iso_expr("open_dt", ISO_US).alias("open_ts"),
                iso_expr("click_dt", ISO_US).alias("click_ts"),
            )
            _flush_contact(conn, send_df)
            send_counter += m
            if reporter is not None:
                reporter.advance(m)
        if reporter is not None:
            reporter.finish(f"rows={send_counter:,}")

    return order_counter
