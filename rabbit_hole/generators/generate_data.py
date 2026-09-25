# ruff: noqa: F401
"""Events-first business data generation and customer event-stream materialization.

This is application/demo code showing HOW to generate realistic business data.
It's not a reusable library component—it's specific to this smoke test scenario.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import random
import duckdb
from pathlib import Path
import shutil
import sys
import time

import torch
from datetime import timezone as _tz
from rabbit_hole.generators.business_tables import (
    _bulk_insert,
    _exec_script,
    _flush_contact,
    _flush_email,
    _flush_order_items,
    create_business_tables,
)
from rabbit_hole.generators.event_stream import (
    load_customer_event_stream,
    materialize_customer_event_stream,
)
from rabbit_hole.generators.generate_support import (
    CustomerEventRow,
    ProgressReporter,
    _REFERENCE_NOW,
    _browse_distribution,
    _sample_event_ts,
    _sample_weighted_index,
    _seasonal_wave,
    _weighted_choice,
)
from rabbit_hole.generators.seed_data import seed_business_data


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for data generation scale and output locations."""

    parser = argparse.ArgumentParser(
        description="Build business raw data and customer event stream"
    )
    parser.add_argument(
        "--duckdb-path", type=Path, default=Path("data/duckdb/customer_event_stream.duckdb")
    )
    parser.add_argument(
        "--arrow-path", type=Path, default=Path("data/arrow/customer_event_stream.feather")
    )
    parser.add_argument("--num-customers", type=int, default=120_000)
    parser.add_argument("--num-products", type=int, default=1_200)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--event-count", type=int, default=10_000_000)
    parser.add_argument("--order-ratio", type=float, default=0.08)
    parser.add_argument("--min-orders-per-customer", type=int, default=4)
    parser.add_argument("--activity-sampling-power", type=float, default=1.35)
    parser.add_argument("--lancedb-dir", type=Path, default=Path("scripts/data/lancedb"))
    parser.add_argument(
        "--log-path", type=Path, default=Path("scripts/data/logs/generate_data.log")
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for full synthetic data generation workflow."""

    args = _parse_args()
    args.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(log_path=args.log_path)

    try:
        reporter.log(f"Build log file: {args.log_path}")

        if args.duckdb_path.exists():
            reporter.log(f"Removing existing DuckDB: {args.duckdb_path}")
            args.duckdb_path.unlink()
        if args.lancedb_dir.exists():
            reporter.log(f"Removing existing LanceDB dir: {args.lancedb_dir}")
            shutil.rmtree(args.lancedb_dir)

        conn = duckdb.connect(str(args.duckdb_path))
        try:
            reporter.log("Creating DuckDB schema")
            create_business_tables(conn)
            seed_business_data(
                conn,
                num_customers=args.num_customers,
                num_products=args.num_products,
                years=args.years,
                event_count=args.event_count,
                order_ratio=args.order_ratio,
                min_orders_per_customer=args.min_orders_per_customer,
                activity_sampling_power=args.activity_sampling_power,
                seed=args.seed,
                reporter=reporter,
            )
            event_count = materialize_customer_event_stream(conn, reporter=reporter)
            customer_count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            order_count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            web_count = conn.execute("SELECT COUNT(*) FROM website_browse").fetchone()[0]
            promotion_count = conn.execute("SELECT COUNT(*) FROM promotions").fetchone()[0]
            inventory_count = conn.execute("SELECT COUNT(*) FROM inventory_daily").fetchone()[0]
            first_rows = load_customer_event_stream(conn)[:5]
        finally:
            conn.close()
        # Primary data layer: Arrow IPC (zero-copy, memory-mappable).
        from rabbit_hole import stream as _stream

        reporter.log(f"Writing Arrow IPC stream -> {args.arrow_path}")
        _stream.to_arrow(str(args.duckdb_path), str(args.arrow_path))
    finally:
        reporter.close()

    print("Business event stream built")
    print(f"DuckDB: {args.duckdb_path}")
    print(f"Arrow: {args.arrow_path}")
    print(f"Customers: {customer_count}")
    print(f"Orders: {order_count}")
    print(f"Website browse events: {web_count}")
    print(f"Promotions: {promotion_count}")
    print(f"Inventory snapshots: {inventory_count}")
    print(f"Customer events: {event_count}")
    print("First customer events:")
    for row in first_rows:
        print(row)


if __name__ == "__main__":
    main()
