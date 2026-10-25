"""Export the compact payload for the Data tab.

The Data tab is deliberately one view: a single customer's event timeline and
event ledger. This script reads the canonical stream and writes the minimum
data that view needs to `demo/data_tab.json`.

    python3 scripts/build_data_tab.py
    python3 scripts/build_data_tab.py --customer rc0000208
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--customer", default=DEFAULT_CUSTOMER)
    ap.add_argument("--as-of", default="2025-01-01", dest="as_of")
    ap.add_argument("--window-start", default="2024-01-01", dest="window_start")
    ap.add_argument("--label-days", type=int, default=90, dest="label_days")
    ap.add_argument("--window-days", type=int, default=540, dest="window_days")
    args = ap.parse_args()

    data = Path(args.data)
    stream_path = data / "path_b/path_b_customer_event_stream.parquet"
    customers_path = data / "relational/customers.parquet"
    missing = [str(p) for p in (stream_path, customers_path) if not p.exists()]
    if missing:
        print("missing source files:\n  " + "\n  ".join(missing), file=sys.stderr)
        raise SystemExit(1)

    key = args.customer
    customers = pl.read_parquet(customers_path).filter(pl.col("customer_key") == key)
    if customers.height == 0:
        print(f"customer {key} not found in {customers_path}", file=sys.stderr)
        raise SystemExit(1)
    row = customers.to_dicts()[0]

    stream = (
        pl.read_parquet(stream_path)
        .filter(pl.col("customer_key") == key)
        .with_columns(pl.col("event_ts").str.to_datetime().alias("_ts"))
        .sort("_ts")
    )
    if stream.height == 0:
        print(f"customer {key} has no events in {stream_path}", file=sys.stderr)
        raise SystemExit(1)

    as_of = datetime.fromisoformat(args.as_of)
    start = datetime.fromisoformat(args.window_start)
    label_end = as_of + timedelta(days=args.label_days)

    events = []
    for event in stream.iter_rows(named=True):
        ts: datetime = event["_ts"]
        phase = "history" if ts <= as_of else ("labels" if ts <= label_end else "later")
        attrs = json.loads(event["event_attributes"] or "{}")
        events.append(
            {
                "ts": event["event_ts"],
                "type": event["event_type"],
                "phase": phase,
                "day": round((ts - start).total_seconds() / 86400.0, 3),
                "detail": _detail(event["event_type"], attrs),
            }
        )

    counts = {
        "total": len(events),
        "history": sum(1 for e in events if e["phase"] == "history"),
        "labels": sum(1 for e in events if e["phase"] == "labels"),
        "later": sum(1 for e in events if e["phase"] == "later"),
    }

    out = {
        "customer": {
            "key": row["customer_key"],
            "signup_ts": row["signup_ts"],
            "segment": int(row["segment"]),
            "region": row["region"],
            "acquisition_channel": row["acquisition_channel"],
            "loyalty_tier": row["loyalty_tier"],
        },
        "as_of": as_of.isoformat(),
        "label_end": label_end.isoformat(),
        "window_start": args.window_start,
        "label_days": args.label_days,
        "window_days": args.window_days,
        "counts": counts,
        "types": [{"id": t, "label": label} for t, label in EVENT_TYPES],
        "events": events,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(
        f"WROTE {args.out} | {key} | {counts['total']} events | "
        f"{counts['history']} history + {counts['labels']} labels + {counts['later']} later"
    )


if __name__ == "__main__":
    main()
