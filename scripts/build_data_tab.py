"""Export the payload for the Data tab.

One example customer appears in both data shapes:

  * Legacy  - the traditional wide table: one row per customer, 51 columns,
              history aggregates plus 30/90-day outcome labels.
  * Stream  - the canonical event stream: one row per event, four columns,
              ordered by time and stripped of relational joins.

The export reads the canonical relational reference under
`/tmp/opencode/t0_loop/ref_v3b_6k` and writes `demo/data_tab.json`:

    python3 scripts/build_data_tab.py
    python3 scripts/build_data_tab.py --customer rc0000208

Every count and value in the tab is computed here from the same source tables.
The only authored text is the field grouping and the plain-language derivation
rule shown next to each group.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/tmp/opencode/t0_loop/ref_v3b_6k")
OUT = ROOT / "demo/data_tab.json"
DEFAULT_CUSTOMER = "rc0000208"

RAW_TABLES = [
    ("customers", "customer_key"),
    ("orders", "order_id"),
    ("order_items", "order_id"),
    ("events", "event_id"),
    ("returns", "return_id"),
    ("support", "ticket_id"),
    ("products", "product_id"),
    ("campaigns", "campaign_id"),
]

EVENT_TYPES = [
    ("customer_signup", "signup", "signup"),
    ("view", "view", "browse"),
    ("search", "search", "browse"),
    ("add_to_cart", "cart +", "browse"),
    ("remove_from_cart", "cart −", "browse"),
    ("purchase", "purchase", "commerce"),
    ("return", "return", "commerce"),
    ("review", "review", "service"),
    ("support_ticket", "support", "service"),
    ("email_send", "send", "marketing"),
]

WIDE_GROUPS = [
    {
        "id": "profile",
        "label": "Profile",
        "source": "customers",
        "rule": (
            "Copied from the customer dimension at customer_key. These five fields are "
            "attributes, not behaviour."
        ),
        "fields": [
            ("signup_ts", "Signup timestamp", "date"),
            ("segment", "Segment", "int"),
            ("region", "Region", "text"),
            ("acquisition_channel", "Acquisition channel", "text"),
            ("loyalty_tier", "Loyalty tier", "text"),
        ],
    },
    {
        "id": "engagement",
        "label": "Engagement",
        "source": "events",
        "rule": (
            "One COUNT per event type over history rows only (event_ts ≤ as_of). "
            "A purchase is an event type, not an order-table lookup."
        ),
        "fields": [
            ("n_events", "All events", "int"),
            ("n_orders", "Purchases", "int"),
            ("n_returns", "Returns", "int"),
            ("n_support", "Support tickets", "int"),
            ("n_sends", "Marketing sends", "int"),
            ("n_views", "Product views", "int"),
            ("n_carts", "Add to cart", "int"),
            ("n_searches", "Searches", "int"),
        ],
    },
    {
        "id": "commerce",
        "label": "Commerce",
        "source": "events",
        "rule": (
            "SUM(value) on purchase and return rows, distinct counts over product, channel "
            "and campaign identifiers, mean discount and SUM(qty). All over history rows."
        ),
        "fields": [
            ("purchase_value", "Purchase value", "money"),
            ("purchase_margin", "Purchase margin", "money"),
            ("return_value", "Return value", "money"),
            ("value_30d", "Purchase value, last 30d", "money"),
            ("value_90d", "Purchase value, last 90d", "money"),
            ("n_products", "Distinct products", "int"),
            ("n_channels", "Distinct channels", "int"),
            ("n_campaigns", "Distinct campaigns", "int"),
            ("discount_mean", "Mean discount", "ratio4"),
            ("qty_sum", "Units purchased", "int"),
        ],
    },
    {
        "id": "preference",
        "label": "Preference",
        "source": "events",
        "rule": (
            "COUNT history rows per product category and per touch channel. These are "
            "behavioural histograms, not model outputs."
        ),
        "fields": [
            ("hist_cat_apparel", "Apparel events", "int"),
            ("hist_cat_footwear", "Footwear events", "int"),
            ("hist_cat_accessories", "Accessories events", "int"),
            ("hist_cat_beauty", "Beauty events", "int"),
            ("hist_cat_home", "Home events", "int"),
            ("hist_cat_outdoor", "Outdoor events", "int"),
            ("hist_channel_web", "Web events", "int"),
            ("hist_channel_app", "App events", "int"),
            ("hist_channel_store", "Store events", "int"),
        ],
    },
    {
        "id": "recency",
        "label": "Recency",
        "source": "events · customers",
        "rule": (
            "Days between as_of and the last event / last purchase; ratios over tenure and "
            "orders; category entropy over the six category counts."
        ),
        "fields": [
            ("recency_days", "Days since any event", "int"),
            ("days_since_last_purchase", "Days since purchase", "int"),
            ("avg_order_value", "Mean order value", "money"),
            ("return_rate", "Return rate", "ratio4"),
            ("net_value", "Purchase − returns", "money"),
            ("tenure_days", "Tenure at as_of", "int"),
            ("purchase_rate", "Purchases per day", "ratio4"),
            ("category_entropy", "Category entropy", "num3"),
        ],
    },
    {
        "id": "outcomes",
        "label": "Outcomes",
        "source": "events > as_of",
        "rule": (
            "Computed from the first 30 and 90 days after as_of. Churn is 1 when no purchase "
            "occurs inside that horizon. The stream keeps these rows too."
        ),
        "fields": [
            ("y_orders_30", "Orders, 30d", "int"),
            ("y_value_30", "Value, 30d", "money"),
            ("y_margin_30", "Margin, 30d", "money"),
            ("y_next_days_30", "Days to next order, 30d", "int"),
            ("y_churn_30", "Churn, 30d", "int"),
            ("y_orders_90", "Orders, 90d", "int"),
            ("y_value_90", "Value, 90d", "money"),
            ("y_margin_90", "Margin, 90d", "money"),
            ("y_next_days_90", "Days to next order, 90d", "int"),
            ("y_churn_90", "Churn, 90d", "int"),
        ],
    },
]


def _schema(path: Path) -> list[str]:
    return list(pl.scan_parquet(path).collect_schema().names())


def _customer(raw: pl.DataFrame, key: str) -> dict:
    row = raw.filter(pl.col("customer_key") == key).to_dicts()[0]
    return {
        "key": row["customer_key"],
        "signup_ts": row["signup_ts"],
        "segment": int(row["segment"]),
        "region": row["region"],
        "acquisition_channel": row["acquisition_channel"],
        "loyalty_tier": row["loyalty_tier"],
    }


def _raw_counts(data: Path, key: str) -> dict:
    raw: dict[str, int | None] = {}
    paths = {name: data / "relational" / f"{name}.parquet" for name, _k in RAW_TABLES}
    raw["customers"] = (
        pl.scan_parquet(paths["customers"])
        .filter(pl.col("customer_key") == key)
        .select(pl.len())
        .collect()
        .item()
    )
    orders = pl.scan_parquet(paths["orders"]).filter(pl.col("customer_key") == key)
    order_ids = orders.select("order_id").collect()["order_id"].to_list()
    raw["orders"] = len(order_ids)
    raw["order_items"] = (
        pl.scan_parquet(paths["order_items"])
        .filter(pl.col("order_id").is_in(order_ids))
        .select(pl.len())
        .collect()
        .item()
        if order_ids
        else 0
    )
    raw["events"] = (
        pl.scan_parquet(paths["events"])
        .filter(pl.col("customer_key") == key)
        .select(pl.len())
        .collect()
        .item()
    )
    raw["returns"] = (
        pl.scan_parquet(paths["returns"])
        .filter(pl.col("customer_key") == key)
        .select(pl.len())
        .collect()
        .item()
    )
    raw["support"] = (
        pl.scan_parquet(paths["support"])
        .filter(pl.col("customer_key") == key)
        .select(pl.len())
        .collect()
        .item()
    )
    # Shared dimensions have no customer key; they are only counted globally.
    raw["products"] = None
    raw["campaigns"] = None
    return {
        "rows": raw,
        "columns": {name: len(_schema(path)) for name, path in paths.items()},
    }


def _sum_attr(rows: list[dict], event_type: str, field: str) -> float:
    vals = [float(r["attributes"][field]) for r in rows if r["type"] == event_type]
    return round(sum(vals), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--customer", default=DEFAULT_CUSTOMER)
    ap.add_argument("--as-of", default="2025-01-01", dest="as_of")
    ap.add_argument("--window-start", default="2024-01-01", dest="window_start")
    ap.add_argument("--window-days", type=int, default=540, dest="window_days")
    ap.add_argument("--label-days", type=int, default=90, dest="label_days")
    args = ap.parse_args()

    data = Path(args.data)
    wide_path = data / "path_a/path_a_wide_table.parquet"
    stream_path = data / "path_b/path_b_customer_event_stream.parquet"
    required = [
        wide_path,
        stream_path,
        data / "relational/customers.parquet",
        data / "relational/orders.parquet",
        data / "relational/order_items.parquet",
        data / "relational/events.parquet",
        data / "relational/returns.parquet",
        data / "relational/support.parquet",
        data / "relational/products.parquet",
        data / "relational/campaigns.parquet",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("missing source files:\n  " + "\n  ".join(missing), file=sys.stderr)
        raise SystemExit(1)

    key = args.customer
    wide_row = pl.read_parquet(wide_path).filter(pl.col("customer_key") == key).to_dicts()
    if not wide_row:
        print(f"customer {key} not found in {wide_path}", file=sys.stderr)
        raise SystemExit(1)
    wide = wide_row[0]

    raw_customers = pl.read_parquet(data / "relational/customers.parquet")
    if (raw_customers["customer_key"] == key).sum() == 0:
        print(f"customer {key} not found in relational/customers.parquet", file=sys.stderr)
        raise SystemExit(1)

    stream = (
        pl.read_parquet(stream_path)
        .filter(pl.col("customer_key") == key)
        .with_columns(pl.col("event_ts").str.to_datetime().alias("_ts"))
        .sort("_ts")
    )

    as_of = datetime.fromisoformat(args.as_of)
    start = datetime.fromisoformat(args.window_start)
    label_end = as_of + timedelta(days=args.label_days)
    as_of_s = as_of.isoformat()
    label_end_s = label_end.isoformat()

    events = []
    for row in stream.iter_rows(named=True):
        ts: datetime = row["_ts"]
        attrs = json.loads(row["event_attributes"] or "{}")
        phase = "history" if ts <= as_of else ("labels" if ts <= label_end else "later")
        events.append(
            {
                "ts": row["event_ts"],
                "type": row["event_type"],
                "phase": phase,
                "day": round((ts - start).total_seconds() / 86400.0, 3),
                "attributes": attrs,
            }
        )

    history = [e for e in events if e["phase"] == "history"]
    label_rows = [e for e in events if e["phase"] == "labels"]

    def count(rows, event_type):
        return sum(1 for e in rows if e["type"] == event_type)

    purchase_value = _sum_attr(history, "purchase", "value")
    return_value = _sum_attr(history, "return", "value")
    y_value_90 = _sum_attr(label_rows, "purchase", "value")

    raw = _raw_counts(data, key)
    customer = _customer(raw_customers, key)

    groups = [
        {
            "id": g["id"],
            "label": g["label"],
            "source": g["source"],
            "rule": g["rule"],
            "fields": [
                {
                    "name": name,
                    "label": label,
                    "kind": kind,
                    "value": wide.get(name),
                }
                for name, label, kind in g["fields"]
            ],
        }
        for g in WIDE_GROUPS
    ]

    counts = {
        "history": len(history),
        "labels": len(label_rows),
        "later": len(events) - len(history) - len(label_rows),
        "total": len(events),
    }

    reconcile = [
        {
            "wide": "n_events",
            "label": "Events",
            "wide_value": wide.get("n_events"),
            "event_value": counts["history"],
            "window": "history",
            "from": "count of event rows ≤ as_of",
        },
        {
            "wide": "n_orders",
            "label": "Purchases",
            "wide_value": wide.get("n_orders"),
            "event_value": count(history, "purchase"),
            "window": "history",
            "from": "event_type = purchase, ≤ as_of",
        },
        {
            "wide": "purchase_value",
            "label": "Purchase value",
            "wide_value": wide.get("purchase_value"),
            "event_value": purchase_value,
            "window": "history",
            "from": "Σ value over purchase rows",
        },
        {
            "wide": "n_returns",
            "label": "Returns",
            "wide_value": wide.get("n_returns"),
            "event_value": count(history, "return"),
            "window": "history",
            "from": "event_type = return, ≤ as_of",
        },
        {
            "wide": "return_value",
            "label": "Return value",
            "wide_value": wide.get("return_value"),
            "event_value": return_value,
            "window": "history",
            "from": "Σ value over return rows",
        },
        {
            "wide": "n_support",
            "label": "Support tickets",
            "wide_value": wide.get("n_support"),
            "event_value": count(history, "support_ticket"),
            "window": "history",
            "from": "event_type = support_ticket, ≤ as_of",
        },
        {
            "wide": "net_value",
            "label": "Net value",
            "wide_value": wide.get("net_value"),
            "event_value": round(purchase_value + return_value, 4),
            "window": "history",
            "from": "purchase value + return value",
        },
        {
            "wide": "y_orders_90",
            "label": "Orders, next 90d",
            "wide_value": wide.get("y_orders_90"),
            "event_value": count(label_rows, "purchase"),
            "window": "label90",
            "from": "purchase rows in the 90d after as_of",
        },
        {
            "wide": "y_value_90",
            "label": "Value, next 90d",
            "wide_value": wide.get("y_value_90"),
            "event_value": y_value_90,
            "window": "label90",
            "from": "Σ value over those purchase rows",
        },
    ]

    for r in reconcile:
        r["kind"] = (
            "money"
            if r["wide"] in ("purchase_value", "return_value", "net_value", "y_value_90")
            else "int"
        )

    manifest = json.loads((data / "manifest.json").read_text())
    global_counts = manifest.get("counts") or {}
    raw_global = {
        name: (
            pl.scan_parquet(data / "relational" / f"{name}.parquet")
            .select(pl.len())
            .collect()
            .item()
        )
        for name, _key in RAW_TABLES
    }

    out = {
        "customer": customer,
        "as_of": as_of_s,
        "label_end": label_end_s,
        "window_start": args.window_start,
        "window_days": args.window_days,
        "label_days": args.label_days,
        "global": {
            "customers": global_counts.get("customers", 6000),
            "events": global_counts.get("events", 3355797),
            "wide_rows": global_counts.get("customers", 6000),
            "wide_cols": len(wide),
            "stream_cols": len(stream.columns) - 1,  # exclude helper _ts
            "raw_tables": len(RAW_TABLES),
        },
        "local": {
            "wide_rows": 1,
            "stream_rows": len(events),
            **counts,
            "raw": [
                {
                    "name": name,
                    "key": key_col,
                    "rows": raw["rows"][name],
                    "global_rows": raw_global[name],
                    "columns": raw["columns"][name],
                    "lookup": name in ("products", "campaigns"),
                }
                for name, key_col in RAW_TABLES
            ],
        },
        "wide": {
            "columns": len(wide),
            "key": "customer_key",
            "key_value": key,
            "groups": groups,
        },
        "stream": {
            "columns": [
                {"name": "customer_key", "kind": "string", "note": "repeated on every row"},
                {"name": "event_ts", "kind": "timestamp", "note": "ordered, full window"},
                {"name": "event_type", "kind": "string", "note": "ten event types"},
                {"name": "event_attributes", "kind": "json", "note": "type-specific payload"},
            ],
            "types": [
                {"id": t, "label": label, "family": family} for t, label, family in EVENT_TYPES
            ],
            "events": events,
        },
        "reconcile": reconcile,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(
        f"WROTE {args.out} | {key} | {counts['history']} history + "
        f"{counts['labels']} labels + {counts['later']} later | "
        f"{len(events)} stream rows | {len(wide)} wide columns | "
        f"{len(groups)} groups"
    )


if __name__ == "__main__":
    main()
