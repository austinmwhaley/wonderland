"""Export the payload for the raw-data view.

The Data tab shows a legacy wide-table sample: one row per customer, historical
counts only. Selecting a row opens that customer's raw event timeline and
ledger over the full window.

    python3 scripts/build_data_tab.py
    python3 scripts/build_data_tab.py --rows 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
]


def _customer_rows(key: str, wide: pl.DataFrame, stream: pl.DataFrame, start: datetime) -> dict:
    row = wide.filter(pl.col("customer_key") == key).to_dicts()[0]
    events = []
    for event in stream.filter(pl.col("customer_key") == key).iter_rows(named=True):
        ts: datetime = event["_ts"]
        events.append(
            [
                event["event_ts"],
                event["event_type"],
                round((ts - start).total_seconds() / 86400.0, 3),
                event["event_attributes"] or "{}",
            ]
        )

    legacy = {}
    for name, kind in LEGACY_COLUMNS:
        value = row.get(name)
        if value is None:
            legacy[name] = None
        elif kind == "int":
            legacy[name] = int(value)
        else:
            legacy[name] = round(float(value), 2)

    return {"key": key, "legacy": legacy, "events": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--customer", default=DEFAULT_CUSTOMER)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--window-start", default="2024-01-01", dest="window_start")
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

    n_extra = max(0, args.rows - 1)
    sampled = (
        wide.filter(pl.col("customer_key") != args.customer)
        .sample(n=n_extra, seed=17)["customer_key"]
        .to_list()
        if n_extra
        else []
    )
    keys = sorted(set([args.customer, *sampled]))

    stream = (
        pl.read_parquet(stream_path)
        .filter(pl.col("customer_key").is_in(keys))
        .with_columns(pl.col("event_ts").str.to_datetime().alias("_ts"))
        .sort(["customer_key", "_ts"])
    )

    start = datetime.fromisoformat(args.window_start)
    customers = [_customer_rows(key, wide, stream, start) for key in keys]

    out = {
        "window_start": args.window_start,
        "window_days": args.window_days,
        "wide": {"rows": wide.height, "columns": wide.width},
        "types": [{"id": t, "label": label} for t, label in EVENT_TYPES],
        "customers": customers,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(
        f"WROTE {args.out} | {len(customers)} legacy rows | "
        f"{sum(len(c['events']) for c in customers)} events | {wide.width} wide columns"
    )


if __name__ == "__main__":
    main()
