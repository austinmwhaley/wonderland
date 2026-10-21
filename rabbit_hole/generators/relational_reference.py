"""Relational REFERENCE generator: normal business tables -> one customer event stream.

Why this exists (DEC-236): the flat reference was additive/measure-like, so a wide table was
sufficient by construction and could not test a sequence encoder. This generator builds the
tables a real business has — customers, products, orders, order_items, events, campaigns,
returns, support — with latent regimes, path dependence, interactions, long memory, and
missing logs, then unifies them into the canonical 4-column event stream. It also emits the
traditional wide customer table (the baseline), so the same source data supports:
    table + GBM   vs   raw events + sequence encoder   vs   E + table.

Writes:
  <out>/customer_event_stream.parquet     canonical stream (customer_key, event_ts,
                                          event_type, event_attributes, ...)
  <out>/customer_wide_table.parquet       traditional per-customer aggregates
  <out>/relational/customers.parquet      customers, products, orders, order_items, events,
                                          campaigns, returns, support
  <out>/relational_truth.json             known-truth receipt (latent optima/regimes)
  <out>/reference_truth.json              alias for the certification pipeline

The null (`--snr 0`) keeps the baseline process but flattens regimes/action/path effects.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

CATEGORIES = ["apparel", "footwear", "accessories", "beauty", "home", "outdoor"]
BRANDS = [f"brand_{i}" for i in range(10)]
CHANNELS = ["email", "sms", "push", "direct_mail"]
TOUCH = ["web", "app", "store"]
REGIMES = ["browse", "intent", "loyal", "at_risk"]
LOYALTY = ["none", "silver", "gold", "platinum"]
ACQ = ["organic", "paid_search", "social", "referral"]
ORG_EVENTS = ["view", "search", "add_to_cart", "remove_from_cart", "support_ticket", "review"]
ACTS = ("freq", "hour", "dow", "channel", "offer")


@dataclass
class RelationalSpec:
    n_customers: int = 2000
    window_days: int = 540
    seed: int = 17
    snr: float = 1.0
    start_date: str = "2024-01-01"
    as_of: str = "2025-01-01"
    missing_rate: float = 0.15
    n_products: int = 80
    base_activity: float = 0.55
    base_purchase: float = 0.10


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def build_relational(spec: RelationalSpec):
    rng = np.random.default_rng(spec.seed)
    s = float(spec.snr)
    start = datetime.fromisoformat(spec.start_date)
    T = int(spec.window_days)

    # ---- products ---------------------------------------------------------
    prod_cat = rng.integers(0, len(CATEGORIES), spec.n_products)
    prod_brand = rng.integers(0, len(BRANDS), spec.n_products)
    prod_price = np.round(np.exp(rng.normal(np.log(55.0), 0.55, spec.n_products)), 2)
    prod_margin = np.clip(rng.normal(0.38, 0.10, spec.n_products), 0.10, 0.65)
    prod_pop = rng.dirichlet(np.full(spec.n_products, 0.8))
    products = pl.DataFrame(
        {
            "product_id": [f"p{i:03d}" for i in range(spec.n_products)],
            "category_code": [CATEGORIES[int(x)] for x in prod_cat],
            "brand": [BRANDS[int(x)] for x in prod_brand],
            "price": prod_price,
            "margin": np.round(prod_margin, 4),
        }
    )

    # ---- campaigns --------------------------------------------------------
    campaigns = pl.DataFrame(
        {
            "campaign_id": [f"cmp{i:03d}" for i in range(24)],
            "channel": [CHANNELS[i % len(CHANNELS)] for i in range(24)],
            "discount": np.round(rng.uniform(0.05, 0.35, 24), 2),
            "start_day": rng.integers(0, T, 24),
            "duration_days": rng.integers(5, 30, 24),
        }
    )

    rows = {
        k: []
        for k in (
            "customer_key",
            "event_ts",
            "event_type",
            "brand",
            "entity_type",
            "entity_id",
            "source_table",
            "value",
        )
    }
    attrs: list[str] = []
    base_events: list[dict] = []
    customers_truth = []
    orders_rows, items_rows, return_rows, support_rows = [], [], [], []
    order_id = 0
    event_id = 0
    customer_rows = []

    def emit(key, day, etype, *, brand="", ent_type="", ent_id="", value=0.0, keep=True, **a):
        nonlocal event_id
        ts = start + timedelta(days=float(day))
        av = dict(a)
        if brand:
            av.setdefault("brand", brand)
        if ent_type == "order" and ent_id:
            av.setdefault("order_id", ent_id)
        if value:
            av["value"] = round(float(value), 2)
        event_id += 1
        if keep:
            rows["customer_key"].append(key)
            rows["event_ts"].append(ts.isoformat())
            rows["event_type"].append(etype)
            rows["brand"].append(brand)
            rows["entity_type"].append(ent_type)
            rows["entity_id"].append(ent_id)
            rows["source_table"].append("relational")
            rows["value"].append(float(value))
            attrs.append(json.dumps(av, separators=(",", ":")))
        if not keep:
            return ts
        base_events.append(
            {
                "event_id": f"e{event_id:09d}",
                "customer_key": key,
                "event_ts": ts.isoformat(),
                "event_type": etype,
                "product_id": a.get("product_id"),
                "category_code": a.get("category_code"),
                "brand": brand,
                "channel": a.get("touch", a.get("channel")),
                "campaign_id": a.get("campaign_id"),
                "price": a.get("price"),
                "qty": a.get("qty"),
                "discount": a.get("discount"),
                "value": float(value),
                "margin": a.get("margin"),
                "session_id": a.get("session_id"),
                "attrs_json": json.dumps(av, separators=(",", ":")),
            }
        )
        return ts

    for c in range(spec.n_customers):
        key = f"rc{c:07d}"
        cust_cat = rng.dirichlet(np.full(len(CATEGORIES), 0.7))
        price_sens = float(rng.beta(2.0, 2.5))
        chan_pref = int(rng.integers(0, len(CHANNELS)))
        touch_pref = rng.dirichlet(np.array([2.0, 1.5, 1.0]))
        loyal = int(rng.integers(0, len(LOYALTY)))
        acq = int(rng.integers(0, len(ACQ)))
        segment = int(rng.integers(0, 5))
        return_prone = _clip01(float(rng.beta(1.5, 6.0)))
        churn_day = float(rng.uniform(0.45 * T, T)) if rng.random() < 0.18 else float(T)
        x0 = float(rng.uniform(0, 3))
        h0 = float(rng.uniform(0, 24))
        d0 = float(rng.uniform(0, 7))
        offer_opt = int(rng.integers(0, 4))
        # latent regime chain: 4 states, sticky, customer-specific transition
        P = rng.dirichlet(np.full(len(REGIMES), 1.2), size=len(REGIMES))
        P = 0.93 * np.eye(len(REGIMES)) + 0.07 * P
        regime = int(rng.integers(0, len(REGIMES)))
        # long-memory promo shock: starts on a random day, decays
        shock_day = float(rng.uniform(0.1 * T, 0.8 * T))
        shock_amp = float(rng.uniform(0.3, 1.2))
        shock_tau = float(rng.uniform(10, 45))
        region = ["north", "south", "east", "west"][int(rng.integers(0, 4))]
        customer_rows.append(
            {
                "customer_key": key,
                "signup_ts": (start - timedelta(days=float(rng.integers(1, 400)))).isoformat(),
                "segment": segment,
                "region": region,
                "acquisition_channel": ACQ[acq],
                "loyalty_tier": LOYALTY[loyal],
            }
        )
        emit(
            key,
            0.0,
            "customer_signup",
            ent_type="customer",
            ent_id=key,
            acquisition_channel=ACQ[acq],
            region=region,
        )

        last_event = "signup"
        for day in range(T):
            if day >= churn_day:
                break
            # regime switch + long-memory shock
            if rng.random() < 0.035:
                regime = int(rng.choice(len(REGIMES), p=P[regime]))
            shock = 0.0
            if day >= shock_day:
                shock = shock_amp * np.exp(-(day - shock_day) / max(shock_tau, 1.0))
            regime_mult = {
                0: 0.75,  # browse
                1: 1.55,  # intent
                2: 1.25,  # loyal
                3: 0.45,  # at_risk
            }[regime]
            regime_mult = 1.0 + s * (regime_mult - 1.0)
            # one-factor daily action (others at the customer optimum)
            akey = ACTS[int(rng.integers(0, len(ACTS)))]
            acts = {
                "freq": int(np.clip(round(x0), 0, 3)),
                "hour": int(np.clip(round(h0), 0, 23)),
                "dow": int(np.clip(round(d0), 0, 6)),
                "channel": chan_pref,
                "offer": int(offer_opt),
            }
            if akey == "freq":
                acts["freq"] = int(rng.integers(0, 4))
            elif akey == "hour":
                acts["hour"] = int(rng.integers(0, 24))
            elif akey == "dow":
                acts["dow"] = int(rng.integers(0, 7))
            elif akey == "channel":
                acts["channel"] = int(rng.integers(0, len(CHANNELS)))
            else:
                acts["offer"] = int(rng.integers(0, 4))
            q = 1.0
            q *= np.exp(-0.5 * ((acts["freq"] - x0) / 1.1) ** 2)
            q *= np.exp(-0.5 * ((acts["hour"] - h0) / 3.0) ** 2)
            q *= np.exp(-0.5 * ((acts["dow"] - d0) / 1.6) ** 2)
            q *= 1.0 if acts["channel"] == chan_pref else 0.35
            q *= 1.0 - 0.22 * abs(acts["offer"] - offer_opt)
            q = float(np.clip(1.0 + s * (q - 1.0), 0.05, 1.0))
            tkey = akey
            ts = acts["hour"] + 0.25  # action hour, within the day; purchase follows at +0.75
            emit(
                key,
                day + ts / 24.0,
                "email_send",
                ent_type="customer",
                ent_id=key,
                touch=CHANNELS[acts["channel"]],
                target_key=tkey,
                **acts,
            )
            # path factor: last observed event changes the purchase propensity
            path = {"add_to_cart": 1.6, "view": 1.15, "return": 0.65, "purchase": 0.85}.get(
                last_event, 1.0
            )
            path = 1.0 + s * (path - 1.0)
            # interaction: channel effect is regime-dependent; discount interacts with price sensitivity
            chan_regime = 1.0 + s * (
                0.25 if (acts["channel"] == chan_pref) == (regime in (1, 2)) else -0.15
            )
            discount = 0.0 if acts["offer"] == 0 else 0.05 + 0.06 * acts["offer"]
            price_fit = 1.0 + s * (0.35 * (discount * price_sens * 4.0) - 0.15 * price_sens)
            w = cust_cat[prod_cat] * (0.5 + prod_pop)
            w = w / w.sum()
            pid = int(rng.choice(spec.n_products, p=w))
            p_buy = (
                spec.base_purchase
                * regime_mult
                * (0.55 + 0.45 * q)
                * path
                * chan_regime
                * price_fit
                * (1.0 + 0.9 * shock)
                * (1.0 + 0.35 * cust_cat[int(prod_cat[pid])])
            )
            p_buy = _clip01(p_buy)
            made_purchase = False
            if rng.random() < p_buy:
                qty = int(rng.integers(1, 4))
                unit = float(prod_price[pid])
                value = round(unit * qty * (1.0 - discount), 2)
                margin = round(value * float(prod_margin[pid]), 2)
                order_id += 1
                oid = f"o{order_id:09d}"
                day_ts = day + (acts["hour"] + 0.75) / 24.0
                emit(
                    key,
                    day_ts,
                    "purchase",
                    brand=BRANDS[int(prod_brand[pid])],
                    ent_type="order",
                    ent_id=oid,
                    value=value,
                    product_id=f"p{pid:03d}",
                    category_code=CATEGORIES[int(prod_cat[pid])],
                    price=unit,
                    qty=qty,
                    discount=round(discount, 3),
                    channel=TOUCH[int(rng.choice(len(TOUCH), p=touch_pref))],
                    margin=margin,
                )
                orders_rows.append(
                    {
                        "order_id": oid,
                        "customer_key": key,
                        "event_ts": (start + timedelta(days=day_ts)).isoformat(),
                        "channel": CHANNELS[acts["channel"]],
                        "discount": round(discount, 3),
                        "total": value,
                        "margin": margin,
                        "status": "completed",
                    }
                )
                items_rows.append(
                    {
                        "order_id": oid,
                        "product_id": f"p{pid:03d}",
                        "qty": qty,
                        "unit_price": unit,
                        "discount": round(discount, 3),
                        "value": value,
                    }
                )
                last_event = "purchase"
                if rng.random() < s * return_prone * 0.6:
                    rday = day + float(rng.uniform(1, 6))
                    if rday < T:
                        emit(
                            key,
                            rday,
                            "return",
                            brand=BRANDS[int(prod_brand[pid])],
                            ent_type="order",
                            ent_id=oid,
                            value=-value,
                            product_id=f"p{pid:03d}",
                            category_code=CATEGORIES[int(prod_cat[pid])],
                            reason=["size", "fit", "damaged", "changed_mind"][
                                int(rng.integers(0, 4))
                            ],
                        )
                        return_rows.append(
                            {
                                "return_id": f"r{len(return_rows) + 1:08d}",
                                "order_id": oid,
                                "customer_key": key,
                                "event_ts": (start + timedelta(days=rday)).isoformat(),
                                "reason": "size",
                            }
                        )
                        last_event = "return"
            if not made_purchase and rng.random() < s * spec.base_activity * regime_mult:
                # organic funnel: path-dependent event choice
                if last_event == "add_to_cart":
                    et = "purchase" if rng.random() < 0.35 else "remove_from_cart"
                elif last_event == "view":
                    et = ["search", "add_to_cart", "purchase"][int(rng.integers(0, 3))]
                else:
                    et = ORG_EVENTS[int(rng.integers(0, len(ORG_EVENTS)))]
                pid = int(rng.choice(spec.n_products, p=prod_pop))
                keep = not (
                    et in ("view", "search", "add_to_cart", "remove_from_cart")
                    and rng.random() < spec.missing_rate
                )
                if et == "purchase":
                    continue
                emit(
                    key,
                    day + float(rng.integers(0, 86400)) / 86400.0,
                    et,
                    brand=BRANDS[int(prod_brand[pid])],
                    ent_type="product",
                    ent_id=f"p{pid:03d}",
                    product_id=f"p{pid:03d}",
                    category_code=CATEGORIES[int(prod_cat[pid])],
                    channel=TOUCH[int(rng.choice(len(TOUCH), p=touch_pref))],
                    keep=keep,
                )
                last_event = et
                if et == "support_ticket":
                    support_rows.append(
                        {
                            "ticket_id": f"t{len(support_rows) + 1:08d}",
                            "customer_key": key,
                            "event_ts": (start + timedelta(days=day)).isoformat(),
                            "reason": "delivery",
                        }
                    )
        customers_truth.append(
            {
                "customer": key,
                "x0": round(x0, 3),
                "h0": round(h0, 3),
                "d0": round(d0, 3),
                "offer_opt": offer_opt,
                "channel_pref": CHANNELS[chan_pref],
                "segment": segment,
                "churn_day": round(churn_day, 2),
                "regime0": REGIMES[regime],
                "category_pref": CATEGORIES[int(np.argmax(cust_cat))],
            }
        )

    stream = (
        pl.DataFrame({**rows, "event_attributes": attrs})
        .select(["customer_key", "event_ts", "event_type", "event_attributes"])
        .sort(["customer_key", "event_ts"])
    )
    base = pl.DataFrame(base_events).sort(["customer_key", "event_ts"])
    customers = pl.DataFrame(customer_rows)

    # ---- traditional wide table (the baseline view) ----------------------
    orders = (
        pl.DataFrame(orders_rows)
        if orders_rows
        else pl.DataFrame(
            schema={
                "order_id": pl.Utf8,
                "customer_key": pl.Utf8,
                "event_ts": pl.Utf8,
                "channel": pl.Utf8,
                "discount": pl.Float64,
                "total": pl.Float64,
                "margin": pl.Float64,
                "status": pl.Utf8,
            }
        )
    )
    items = (
        pl.DataFrame(items_rows)
        if items_rows
        else pl.DataFrame(
            schema={
                "order_id": pl.Utf8,
                "product_id": pl.Utf8,
                "qty": pl.Int64,
                "unit_price": pl.Float64,
                "discount": pl.Float64,
                "value": pl.Float64,
            }
        )
    )
    returns = (
        pl.DataFrame(return_rows)
        if return_rows
        else pl.DataFrame(
            schema={
                "return_id": pl.Utf8,
                "order_id": pl.Utf8,
                "customer_key": pl.Utf8,
                "event_ts": pl.Utf8,
                "reason": pl.Utf8,
            }
        )
    )
    support = (
        pl.DataFrame(support_rows)
        if support_rows
        else pl.DataFrame(
            schema={
                "ticket_id": pl.Utf8,
                "customer_key": pl.Utf8,
                "event_ts": pl.Utf8,
                "reason": pl.Utf8,
            }
        )
    )
    # ---- PATH A / PATH B split (point-in-time) ---------------------------
    as_of = datetime.fromisoformat(spec.as_of)
    as_of_day = (as_of - start).days
    if not (0 < as_of_day < T):
        raise ValueError(f"as_of {spec.as_of} must be inside the {T}-day window")
    ev = base.with_columns(pl.col("event_ts").str.to_datetime().alias("ts_dt"))
    hist = ev.filter(pl.col("ts_dt") <= as_of)
    fut = ev.filter(pl.col("ts_dt") > as_of)
    _purchase = pl.col("event_type") == "purchase"
    cat_feats = [(pl.col("category_code") == c).sum().alias(f"hist_cat_{c}") for c in CATEGORIES]
    chan_feats = [(pl.col("channel") == c).sum().alias(f"hist_channel_{c}") for c in TOUCH]
    per_cust = (
        hist.group_by("customer_key")
        .agg(
            pl.len().alias("n_events"),
            _purchase.sum().alias("n_orders"),
            (pl.col("event_type") == "return").sum().alias("n_returns"),
            (pl.col("event_type") == "support_ticket").sum().alias("n_support"),
            (pl.col("event_type") == "email_send").sum().alias("n_sends"),
            (pl.col("event_type") == "view").sum().alias("n_views"),
            (pl.col("event_type") == "add_to_cart").sum().alias("n_carts"),
            (pl.col("event_type") == "search").sum().alias("n_searches"),
            pl.col("value").filter(_purchase).sum().alias("purchase_value"),
            pl.col("margin").filter(_purchase).sum().alias("purchase_margin"),
            pl.col("value").filter(pl.col("event_type") == "return").sum().alias("return_value"),
            pl.col("value")
            .filter(_purchase & (pl.col("ts_dt") > pl.lit(as_of - timedelta(days=30))))
            .sum()
            .alias("value_30d"),
            pl.col("value")
            .filter(_purchase & (pl.col("ts_dt") > pl.lit(as_of - timedelta(days=90))))
            .sum()
            .alias("value_90d"),
            pl.col("product_id").n_unique().alias("n_products"),
            pl.col("category_code").n_unique().alias("n_categories"),
            pl.col("channel").n_unique().alias("n_channels"),
            pl.col("campaign_id").n_unique().alias("n_campaigns"),
            pl.col("discount").mean().alias("discount_mean"),
            pl.col("qty").sum().alias("qty_sum"),
            pl.col("ts_dt").max().alias("last_event_ts"),
            pl.col("ts_dt").min().alias("first_event_ts"),
            pl.col("ts_dt").filter(_purchase).max().alias("last_purchase_ts"),
            *cat_feats,
            *chan_feats,
        )
        .with_columns(
            ((pl.lit(as_of) - pl.col("last_event_ts")).dt.total_days()).alias("recency_days"),
            ((pl.lit(as_of) - pl.col("first_event_ts")).dt.total_days()).alias("tenure_days"),
            ((pl.lit(as_of) - pl.col("last_purchase_ts")).dt.total_days()).alias(
                "days_since_last_purchase"
            ),
            (pl.col("purchase_value") / pl.col("n_orders").clip(1, None)).alias("avg_order_value"),
            (pl.col("n_returns") / pl.col("n_orders").clip(1, None)).alias("return_rate"),
            (pl.col("purchase_value") + pl.col("return_value")).alias("net_value"),
        )
        .with_columns(
            (pl.col("n_orders") / pl.col("tenure_days").clip(1, None)).alias("purchase_rate"),
        )
        .drop(["last_event_ts", "first_event_ts", "last_purchase_ts"])
        .with_columns(pl.col("days_since_last_purchase").fill_null(999))
    )
    features = customers.join(per_cust, on="customer_key", how="left").fill_null(0.0)
    labels = customers.select("customer_key")
    for h in (30, 90):
        fh = fut.filter(pl.col("ts_dt") <= as_of + timedelta(days=h))
        agg = (
            fh.group_by("customer_key")
            .agg(
                _purchase.sum().alias(f"y_orders_{h}"),
                pl.col("value").filter(_purchase).sum().alias(f"y_value_{h}"),
                pl.col("margin").filter(_purchase).sum().alias(f"y_margin_{h}"),
                pl.col("ts_dt").filter(_purchase).min().alias("_first"),
            )
            .with_columns(
                ((pl.col("_first") - pl.lit(as_of)).dt.total_days()).alias(f"y_next_days_{h}"),
            )
            .drop("_first")
        )
        labels = labels.join(agg, on="customer_key", how="left")
        labels = labels.with_columns(
            pl.col(f"y_orders_{h}").fill_null(0),
            pl.col(f"y_value_{h}").fill_null(0.0),
            pl.col(f"y_margin_{h}").fill_null(0.0),
            pl.when(pl.col(f"y_orders_{h}").fill_null(0) == 0)
            .then(h + 1)
            .otherwise(pl.col(f"y_next_days_{h}"))
            .alias(f"y_next_days_{h}"),
            (pl.col(f"y_orders_{h}").fill_null(0) == 0).cast(pl.Int8).alias(f"y_churn_{h}"),
        )
    wide = features.join(labels, on="customer_key", how="left")

    meta = {
        "name": "relational_reference",
        "spec": asdict(spec),
        "n_customers": spec.n_customers,
        "window_days": spec.window_days,
        "seed": spec.seed,
        "as_of": as_of.date().isoformat(),
        "as_of_day": as_of_day,
        "paths": {
            "A": "traditional supervised wide table: features <= as_of, labels > as_of (30/90d)",
            "B": "customer event stream (full window; the encoder must use <= as_of)",
        },
        "categories": CATEGORIES,
        "brands": BRANDS,
        "contact_channels": CHANNELS,
        "event_types": [
            "customer_signup",
            "view",
            "search",
            "add_to_cart",
            "remove_from_cart",
            "purchase",
            "return",
            "email_send",
            "support_ticket",
            "review",
            "loyalty_change",
        ],
        "action_keys": list(ACTS),
        "arm_key": "channel",
        "action": {
            "action_period_seconds": 86400.0,
            "action_design": "one_factor_daily",
            "note": "daily one-factor randomized action; other keys at the customer optimum; "
            "purchase depends on action quality x regime x path x interactions x long-memory shock.",
        },
    }
    truth = {"meta": meta, "customers": customers_truth}
    path_a = {"features": features, "labels": labels, "wide": wide}
    tables = {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": items,
        "events": base,
        "campaigns": campaigns,
        "returns": returns,
        "support": support,
    }
    return stream, path_a, tables, truth


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=2000, dest="n_customers")
    ap.add_argument("--window-days", type=int, default=540, dest="window_days")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--snr", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="relational_reference")
    ap.add_argument("--start-date", default="2024-01-01", dest="start_date")
    ap.add_argument("--as-of", default="2025-01-01", dest="as_of")
    a = ap.parse_args(argv)
    spec = RelationalSpec(
        n_customers=a.n_customers,
        window_days=a.window_days,
        seed=a.seed,
        snr=a.snr,
        start_date=a.start_date,
        as_of=a.as_of,
    )
    stream, path_a, tables, truth = build_relational(spec)
    d = Path(a.out)
    (d / "relational").mkdir(parents=True, exist_ok=True)
    (d / "path_a").mkdir(parents=True, exist_ok=True)
    (d / "path_b").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(d / "relational" / f"{name}.parquet", compression="zstd")
    path_a["features"].write_parquet(d / "path_a" / "customer_features.parquet", compression="zstd")
    path_a["labels"].write_parquet(d / "path_a" / "customer_labels.parquet", compression="zstd")
    path_a["wide"].write_parquet(d / "path_a" / "path_a_wide_table.parquet", compression="zstd")
    stream.write_parquet(d / "path_b" / "path_b_customer_event_stream.parquet", compression="zstd")
    # compatibility aliases for the certification pipeline
    stream.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    path_a["wide"].write_parquet(d / "customer_wide_table.parquet", compression="zstd")
    (d / f"{a.name}_truth.json").write_text(json.dumps(truth, indent=1))
    (d / "reference_truth.json").write_text(json.dumps(truth, indent=1))
    manifest = {
        "name": a.name,
        "join_key": "customer_key",
        "as_of": spec.as_of,
        "window": {"start": spec.start_date, "days": spec.window_days},
        "paths": {
            "A": {
                "description": "traditional supervised wide table",
                "features": "path_a/customer_features.parquet",
                "labels": "path_a/customer_labels.parquet",
                "wide": "path_a/path_a_wide_table.parquet",
                "rows": path_a["wide"].height,
                "columns": path_a["wide"].width,
            },
            "B": {
                "description": "customer event stream (full window; encoder uses <= as_of)",
                "stream": "path_b/path_b_customer_event_stream.parquet",
                "rows": stream.height,
                "columns": stream.width,
            },
        },
        "relational": {k: f"relational/{k}.parquet" for k in tables},
        "counts": {
            "customers": spec.n_customers,
            "events": stream.height,
            "orders": tables["orders"].height,
            "returns": tables["returns"].height,
        },
        "truth": f"{a.name}_truth.json",
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"] | {"wide_columns": path_a["wide"].width}, indent=1))


if __name__ == "__main__":
    main()
