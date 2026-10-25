"""Export the payload for the legacy-first Data tab.

The legacy table shows one row per customer. Selecting a row opens that same
customer's event timeline and ledger underneath. This script reads the wide
table and the canonical stream, and writes `demo/data_tab.json`:

    python3 scripts/build_data_tab.py
    python3 scripts/build_data_tab.py --rows 10
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
DEFAULT_ROWS = 8

EVENT_TYPES = [
    ("customer_signup", "signup"),
    ("view", "view"),
    ("search", "search"),
    ("add_to_cart", "cart +"),
    ("remove_from_cart", "cart −"),
    ("purchase", "purchase"),
    ("return", "return"),
    ("review", "review"),
    ("support_ticket", "support"),
    ("email_send", "send"),
]

LEGACY_COLUMNS = [
    ("n_events", "int"),
    ("n_orders", "int"),
    ("purchase_value", "money"),
    ("y_orders_90", "int"),
    ("y_value_90", "money"),
    ("y_churn_90", "int"),
]


def _detail(event_type: str, attrs: dict) -> str:
    if event_type in ("purchase", "return"):
        value = float(attrs.get("value", 0.0))
        money = ("-" if value < 0 else "") + f"${abs(value):,.2f}"
        category = attrs.get("category_code")
        return money + (f" · {category}" if category else "")
    if event_type == "email_send":
        touch = attrs.get("touch") or ""
        target = attrs.get("target_key")
        return touch + (f" · {target}" if target else "")
    if event_type == "customer_signup":
        return "via " + str(attrs.get("acquisition_channel") or "")
    category = attrs.get("category_code")
    brand = attrs.get("brand")
    if category:
        return str(category) + (f" · {brand}" if brand else "")
    return ""


def _customer_rows(
    key: str,
    wide: pl.DataFrame,
    stream: pl.DataFrame,
    as_of: datetime,
    label_end: datetime,
    start: datetime,
) -> dict:
    row = wide.filter(pl.col("customer_key") == key).to_dicts()[0]
    events = []
    for event in stream.filter(pl.col("customer_key") == key).iter_rows(named=True):
        ts: datetime = event["_ts"]
        phase = "history" if ts <= as_of else ("labels" if ts <= label_end else "later")
        attrs = json.loads(event["event_attributes"] or "{}")
        events.append(
            [
                event["event_ts"],
                event["event_type"],
                phase,
                round((ts - start).total_seconds() / 86400.0, 3),
                _detail(event["event_type"], attrs),
            ]
        )

    counts = {
        "total": len(events),
        "history": sum(1 for e in events if e[2] == "history"),
        "labels": sum(1 for e in events if e[2] == "labels"),
        "later": sum(1 for e in events if e[2] == "later"),
    }
    legacy = {}
    for name, kind in LEGACY_COLUMNS:
        value = row.get(name)
        if value is None:
            legacy[name] = None
        elif kind == "int":
            legacy[name] = int(value)
        else:
            legacy[name] = round(float(value), 2)

    return {
        "key": key,
        "legacy": legacy,
        "counts": counts,
        "events": events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--customer", default=DEFAULT_CUSTOMER)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--as-of", default="2025-01-01", dest="as_of")
    ap.add_argument("--window-start", default="2024-01-01", dest="window_start")
    ap.add_argument("--label-days", type=int, default=90, dest="label_days")
    ap.add_argument("--window-days", type=int, default=540, dest="window_days")
    args = ap.parse_args()

    data = Path(args.data)
    stream_path = data / "path_b/path_b_customer_event_stream.parquet"
    wide_path = data / "path_a/path_a_wide_table.parquet"
    missing = [str(p) for p in (stream_path, wide_path) if not p.exists()]
    if missing:
        print("missing source files:\n  " + "\n  ".join(missing), file=sys.stderr)
        raise SystemExit(1)

    wide = pl.read_parquet(wide_path)
    if args.customer not in set(wide["customer_key"].to_list()):
        print(f"customer {args.customer} not found in {wide_path}", file=sys.stderr)
        raise SystemExit(1)

    # Illustrative sample, stratified so the table shows both outcome classes.
    n_extra = max(0, args.rows - 1)
    n_churn = min(len(wide.filter(pl.col("y_churn_90") == 1)), n_extra // 2)
    n_retain = n_extra - n_churn
    churned = (
        wide.filter(pl.col("y_churn_90") == 1).sample(n=n_churn, seed=17)["customer_key"].to_list()
        if n_churn
        else []
    )
    retained = (
        wide.filter((pl.col("y_churn_90") == 0) & (pl.col("customer_key") != args.customer))
        .sample(n=n_retain, seed=17)["customer_key"]
        .to_list()
        if n_retain
        else []
    )
    keys = sorted(set([args.customer, *churned, *retained]))

    stream = (
        pl.read_parquet(stream_path)
        .filter(pl.col("customer_key").is_in(keys))
        .with_columns(pl.col("event_ts").str.to_datetime().alias("_ts"))
        .sort(["customer_key", "_ts"])
    )

    as_of = datetime.fromisoformat(args.as_of)
    start = datetime.fromisoformat(args.window_start)
    label_end = as_of + timedelta(days=args.label_days)

    customers = [_customer_rows(key, wide, stream, as_of, label_end, start) for key in keys]

    out = {
        "as_of": as_of.isoformat(),
        "label_end": label_end.isoformat(),
        "window_start": args.window_start,
        "window_days": args.window_days,
        "label_days": args.label_days,
        "wide": {"rows": wide.height, "columns": wide.width},
        "types": [{"id": t, "label": label} for t, label in EVENT_TYPES],
        "customers": customers,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(
        f"WROTE {args.out} | {len(customers)} legacy rows | "
        f"{sum(c['counts']['total'] for c in customers)} events | "
        f"{wide.width} wide columns"
    )


if __name__ == "__main__":
    main()
