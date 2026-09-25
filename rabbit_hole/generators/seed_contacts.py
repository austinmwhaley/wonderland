"""Multi-channel contact seeding (email/sms/push) with logged randomized arms."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
import random

import duckdb

from rabbit_hole.generators.business_tables import (
    _bulk_insert,
    _flush_contact,
    _flush_order_items,
)
from rabbit_hole.generators.generate_support import (
    ProgressReporter,
    _sample_weighted_index,
    _weighted_choice,
)


def _seed_contacts(
    conn: duckdb.DuckDBPyConnection,
    rng: random.Random,
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
    sessions_by_customer: dict[str, list[str]] = {}
    for cid, sid in conn.execute(
        "SELECT customer_id, session_id FROM website_browse GROUP BY 1, 2"
    ).fetchall():
        sessions_by_customer.setdefault(cid, []).append(sid)
    converted: dict[tuple[str, str], str] = {}
    for cid, sid, txn in conn.execute(
        "SELECT customer_id, session_id, transaction_id FROM orders WHERE session_id IS NOT NULL"
    ).fetchall():
        converted[(cid, sid)] = txn

    def _softmax(v):
        m = max(v)
        e = [math.exp(x - m) for x in v]
        s = sum(e)
        return [x / s for x in e]

    # MULTI-CHANNEL contacts (email/sms/push): per-period cadence arm (observational,
    # confounded by latent `intent`), 5% persistent holdout, and a DISCOUNT action.
    N_PERIODS = 12
    period_days = max(total_days / N_PERIODS, 1.0)
    period_weeks = period_days / 7.0
    HOLD_FRAC = 0.05
    CHANNELS = {
        "email": dict(cadence=(0.2, 0.6, 1.2, 2.0), peak=0.10, o0=0.12, oa=0.55, c0=0.06, ca=0.45),
        "sms": dict(cadence=(0.1, 0.3, 0.8, 1.5), peak=0.07, o0=0.25, oa=0.35, c0=0.12, ca=0.35),
        "push": dict(cadence=(0.5, 1.5, 3.0, 5.0), peak=0.05, o0=0.10, oa=0.40, c0=0.05, ca=0.30),
    }
    DISCOUNTS = (0.0, 5.0, 10.0, 15.0)
    payments = ["credit_card", "debit_card", "paypal", "wallet", "gift_card"]
    payment_w = [0.49, 0.23, 0.16, 0.09, 0.03]
    intents: dict[str, float] = {cid: rng.gauss(0.0, 1.0) for cid in customer_ids}

    def _gen_channel(chan, ccfg):
        nonlocal order_counter
        CAD = ccfg["cadence"]
        arm_rows, hold_rows, parms = [], [], {}
        for cid in customer_ids:
            activity = customer_activity[cid]
            intent = intents[cid]
            seq = []
            for _p in range(N_PERIODS):
                if rng.random() < HOLD_FRAC:
                    seq.append((-1, 0.0))
                    hold_rows.append((chan, cid, _p, 1))
                    continue
                hold_rows.append((chan, cid, _p, 0))
                util = [(activity + 0.6 * intent) * k + rng.gauss(0.0, 0.8) for k in range(4)]
                pp = _softmax(util)
                a = int(rng.choices(range(4), weights=pp)[0])
                seq.append((a, float(pp[a])))
            parms[cid] = seq
            x0c = CAD[0] + (CAD[-1] - CAD[0]) * max(0.0, min(1.0, activity))
            opt = max(range(4), key=lambda a: CAD[a] * math.exp(-CAD[a] / x0c))
            for _p, (a, pr) in enumerate(seq):
                if a >= 0:
                    arm_rows.append((chan, cid, _p, a, pr, int(opt)))
        if arm_rows:
            _bulk_insert(
                conn,
                "INSERT INTO contact_arm (channel, customer_id, period, arm, propensity, optimal_arm) VALUES (?,?,?,?,?,?)",
                arm_rows,
            )
        if hold_rows:
            _bulk_insert(
                conn,
                "INSERT INTO contact_holdout (channel, customer_id, period, holdout) VALUES (?,?,?,?)",
                hold_rows,
            )
        send_rows, order_rows, item_rows = [], [], []
        send_counter = 0
        if reporter is not None:
            reporter.start(f"generate {chan}", num_customers)
        for cid in customer_ids:
            activity = customer_activity[cid]
            intent = intents[cid]
            act = max(0.0, min(1.0, activity))
            x0 = CAD[0] + (CAD[-1] - CAD[0]) * act
            seq = parms[cid]
            wts = [(0.0 if a < 0 else CAD[a] * period_weeks) for a, _ in seq]
            tot = sum(wts)
            n_sends = max(0, min(int(rng.gauss(tot, max(tot, 1.0) ** 0.5)), 1500))
            sess_list = sessions_by_customer.get(cid, [])
            for _ in range(n_sends):
                pidx = rng.choices(range(N_PERIODS), weights=wts)[0]
                arm, prop = seq[pidx]
                send_ts = start_ts + timedelta(
                    days=pidx * period_days + rng.uniform(0.0, period_days)
                )
                du = [
                    (activity + 0.3 * intent) * (d / 15.0) + rng.gauss(0.0, 0.7) for d in DISCOUNTS
                ]
                dp = _softmax(du)
                disc = float(rng.choices(DISCOUNTS, weights=dp)[0])
                campaign = (
                    promotion_rows[rng.randrange(0, len(promotion_rows))][0]
                    if promotion_rows
                    else None
                )
                opened = 1 if rng.random() < min(0.9, ccfg["o0"] + ccfg["oa"] * activity) else 0
                open_ts = click_ts = click_sid = conv_txn = None
                clicked = 0
                if opened:
                    open_dt = send_ts + timedelta(minutes=rng.randint(3, 2880))
                    open_ts = open_dt.isoformat()
                    if rng.random() < min(0.8, ccfg["c0"] + ccfg["ca"] * activity):
                        clicked = 1
                        click_ts = (open_dt + timedelta(minutes=rng.randint(1, 120))).isoformat()
                        if sess_list:
                            weights = [
                                1.0 + 2.0 * activity * ((cid, s) in converted) for s in sess_list
                            ]
                            running = 0.0
                            cum = []
                            for w in weights:
                                running += w
                                cum.append(running)
                            click_sid = sess_list[_sample_weighted_index(rng, cum)]
                            conv_txn = converted.get((cid, click_sid))
                        x = CAD[arm]
                        season = 1.0 + 0.25 * math.sin(
                            2 * math.pi * (send_ts.timetuple().tm_yday / 365.0)
                        )
                        p_conv = (
                            ccfg["peak"]
                            * math.exp(1.0 - x / x0)
                            * math.exp(0.5 * intent)
                            * season
                            * (1.0 + 0.04 * disc)
                        )
                        if rng.random() < p_conv:
                            pref = customer_pref_category[cid]
                            prod = products_by_category[pref][
                                rng.randrange(0, len(products_by_category[pref]))
                            ]
                            price = product_price[prod]
                            cost = product_cost[prod]
                            txn = f"TXN_{order_counter:012d}"
                            order_counter += 1
                            ots = datetime.fromisoformat(click_ts) + timedelta(
                                minutes=rng.randint(5, 180)
                            )
                            disc_amt = round(price * disc / 100.0, 2)
                            subtotal = round(price - disc_amt, 2)
                            tax = round(subtotal * rng.uniform(0.06, 0.095), 2)
                            ship = 0.0 if subtotal >= 75.0 else round(rng.uniform(3.49, 8.99), 2)
                            total = round(subtotal + tax + ship, 2)
                            ship_c, ship_s = customer_country_state[cid]
                            order_rows.append(
                                (
                                    txn,
                                    cid,
                                    click_sid,
                                    ots.isoformat(),
                                    "completed",
                                    _weighted_choice(rng, payments, payment_w),
                                    ship_c,
                                    ship_s,
                                    None,
                                    subtotal,
                                    tax,
                                    ship,
                                    disc_amt,
                                    total,
                                    subtotal,
                                    round(cost, 2),
                                    round(subtotal - cost, 2),
                                    0,
                                    None,
                                    0.0,
                                    None,
                                )
                            )
                            item_rows.append(
                                (
                                    txn,
                                    prod,
                                    1,
                                    round(price, 2),
                                    round(cost, 2),
                                    round(price, 2),
                                    round(cost, 2),
                                )
                            )
                            conv_txn = txn
                send_rows.append(
                    (
                        chan,
                        f"{chan.upper()}_{send_counter:012d}",
                        cid,
                        pidx,
                        campaign,
                        send_ts.isoformat(),
                        opened,
                        clicked,
                        open_ts,
                        click_ts,
                        click_sid,
                        conv_txn,
                        arm,
                        prop,
                        disc,
                    )
                )
                send_counter += 1
                if len(send_rows) >= 20_000:
                    _flush_contact(conn, send_rows)
                    send_rows.clear()
            if reporter is not None:
                reporter.advance(1)
        if send_rows:
            _flush_contact(conn, send_rows)
        if reporter is not None:
            reporter.finish(f"rows={send_counter:,}")
        return order_rows, item_rows

    _all_orders, _all_items = [], []
    for _chan, _ccfg in CHANNELS.items():
        _o, _i = _gen_channel(_chan, _ccfg)
        _all_orders += _o
        _all_items += _i
    if _all_orders:
        _bulk_insert(
            conn,
            """INSERT INTO orders (transaction_id, customer_id, session_id,
            order_ts, order_status, payment_method, shipping_country, shipping_state,
            promotion_id, subtotal, tax, shipping_fee, discount_amount, order_total,
            revenue, cogs, gross_margin, return_flag, return_ts, return_amount, cancelled_ts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            _all_orders,
        )
        _flush_order_items(conn, _all_items)
