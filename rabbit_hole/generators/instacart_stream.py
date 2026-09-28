"""Ingest the public Instacart Market Basket tables into the canonical stream.

A *fixture* adapter (not a generator): reads the six source CSVs, joins the
wide tables (products -> aisles/departments), and emits ``customer_events`` in
the exact canonical schema, so the whole looking_glass/plugins pipeline can run
on real public data. Observational by construction — no arms / propensity /
known effect; rabbit_hole's synthetic stream remains the known-truth fixture.

All heavy lifting runs in **DuckDB** (out-of-core): a 37M-row polars build was
OOM-killed at 24GB on a 31GB box; DuckDB spills to disk under a fixed
memory_limit instead.

Timeline reconstruction (Instacart ships no absolute dates):
  - inter-order gap = ``days_since_prior_order`` (RIGHT-CENSORED at 30d;
    reconstructed spans are lower bounds — receipted as ``dsp_censored_frac``)
  - per-user last order = ``REF_END - sha256(user_id) % OFFSET_WINDOW`` days
    (staggered ends so the cohort does not burst on one day)
  - hour-of-day = ``order_hour_of_day``; ``order_dow`` is kept as an attribute
    but NOT used for time: only ~17% of rows satisfy
    ``dow == (prev_dow + gap) % 7`` (receipted as ``dow_field_match_frac``).

Usage (repo root):
    python3 -m rabbit_hole.generators.instacart_stream \
        --src rabbit_hole/data/instacart/src --out rabbit_hole/data/instacart
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Fixed data-window end — same convention as generators/generate_support.py
# (_REFERENCE_NOW), so as-of dates like 2025-11-01 / 2025-12-01 sit inside the
# history with real events on both sides.
REF_END = "2026-01-01"
# Days before REF_END over which per-user last orders are staggered (deterministic).
# Must exceed the label horizon + monthly as-of lead: with 90d every user is
# still active at 2025-10-03, so a 30d purchase label has no negatives before
# 2025-11-03 (measured y_mean=1.0 at as_of=2025-11-01). 180d gives ~half the
# cohort an inactivity tail that predates the first standard monthly as-of.
OFFSET_WINDOW = 180
# Spill cap: the full materialize sorts 37M rows with JSON attrs — bounded RAM,
# temp_directory holds the external-sort spill.
MEMORY_LIMIT = "12GB"

CANONICAL_COLS = (
    "event_id",
    "customer_key",
    "event_ts",
    "brand",
    "event_type",
    "event_attributes",
    "entity_type",
    "entity_id",
    "source_table",
    "value",
)

SOURCE_FILES = (
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _offsets(user_ids: list[int]) -> dict[int, int]:
    """Deterministic per-user end offset: stable across runs and machines."""
    return {
        int(u): int(hashlib.sha256(f"instacart:{u}".encode()).hexdigest()[:8], 16) % OFFSET_WINDOW
        for u in user_ids
    }


def build(src: Path, out_dir: Path, arrow: bool = False) -> dict:
    """Build the canonical stream from Instacart CSVs. Returns the receipt."""
    import duckdb
    import polars as pl

    t0 = time.perf_counter()
    src, out_dir = Path(src), Path(out_dir)
    missing = [f for f in SOURCE_FILES if not (src / f).exists()]
    if missing:
        raise SystemExit(
            f"missing Instacart source files in {src}: {missing} "
            "(download kaggle.com/datasets/psparks/instacart-market-basket-analysis "
            "and unzip there)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    f = {name: str(src / name).replace("'", "''") for name in SOURCE_FILES}

    db_path = out_dir / "customer_event_stream.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
        con.execute(f"SET temp_directory='{out_dir / '_tmp'}'")
        # pinned insertion order forces full materialization of 37M wide rows
        # (OOM at the 8GiB cap); spillable external sort instead.
        con.execute("SET preserve_insertion_order=false")

        # per-user deterministic end offsets (small; computed in python)
        users = [
            u
            for (u,) in con.execute(
                f"SELECT DISTINCT user_id FROM read_csv_auto('{f['orders.csv']}')"
            ).fetchall()
        ]
        off = _offsets(users)
        con.register(
            "offsets_df",
            pl.DataFrame({"user_id": users, "end_offset": [off[u] for u in users]}),
        )

        # ---- order timeline (windowed sums, out-of-core) -------------------
        con.execute(f"""
            CREATE OR REPLACE TABLE ord AS
            SELECT o.order_id, o.user_id, o.eval_set, o.order_number, o.order_dow,
                   o.order_hour_of_day, o.days_since_prior_order,
                   sum(coalesce(CAST(o.days_since_prior_order AS BIGINT), 0)) OVER (
                       PARTITION BY o.user_id ORDER BY o.order_number
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS cum_days
            FROM read_csv_auto('{f["orders.csv"]}') o
        """)
        con.execute("""
            CREATE OR REPLACE TABLE ord AS
            SELECT *, max(cum_days) OVER (PARTITION BY user_id) AS cum_last,
                   CAST(days_since_prior_order AS BIGINT) AS gap_prev
            FROM ord
        """)
        con.execute("""
            CREATE OR REPLACE TABLE ord AS
            SELECT *, (end_offset + cum_last - cum_days) AS day_number
            FROM ord JOIN offsets_df USING (user_id)
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE ord AS
            SELECT *,
                   CAST(DATE '{REF_END}' - CAST(day_number AS INTEGER) AS TIMESTAMP)
                       + (INTERVAL '1 hour' * CAST(order_hour_of_day AS INT))
                       AS ts_naive,
                   CAST(strftime(DATE '{REF_END}' - CAST(day_number AS INTEGER),
                                 '%w') AS INT) AS ts_dow
            FROM ord
        """)

        # ---- wide tables ---------------------------------------------------
        con.execute(f"""
            CREATE OR REPLACE TABLE prod AS
            SELECT p.product_id, p.product_name, a.aisle, d.department
            FROM read_csv_auto('{f["products.csv"]}') p
            LEFT JOIN read_csv_auto('{f["aisles.csv"]}') a USING (aisle_id)
            LEFT JOIN read_csv_auto('{f["departments.csv"]}') d USING (department_id)
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE items AS
            SELECT i.order_id, i.product_id, i.add_to_cart_order, i.reordered
            FROM (
                SELECT * FROM read_csv_auto('{f["order_products__prior.csv"]}')
                UNION ALL BY NAME
                SELECT * FROM read_csv_auto('{f["order_products__train.csv"]}')
            ) i
        """)
        orphans = con.execute(
            "SELECT count(*) FROM items i LEFT JOIN ord o USING (order_id) WHERE o.order_id IS NULL"
        ).fetchone()[0]
        if orphans:
            raise SystemExit(f"{orphans} order_products rows have no parent order")
        con.execute("""
            CREATE OR REPLACE TABLE items AS
            SELECT i.*, o.user_id, o.ts_naive AS ts, p.product_name, p.aisle,
                   p.department
            FROM items i JOIN ord o USING (order_id)
            LEFT JOIN prod p USING (product_id)
        """)
        con.execute("""
            CREATE OR REPLACE TABLE ord AS
            SELECT ord.*, it.n_items
            FROM ord LEFT JOIN (
                SELECT order_id, count(*) AS n_items FROM items GROUP BY order_id
            ) it USING (order_id)
        """)

        # ---- canonical events (UNION ALL payload; ids on NARROW keys only) --
        con.execute("""
            CREATE OR REPLACE TABLE ev AS
            WITH ev AS (
                SELECT 'customer_signup' AS event_type,
                       CAST(user_id AS VARCHAR) AS customer_key,
                       strftime(first_ts, '%Y-%m-%dT%H:%M:%S+00:00') AS event_ts,
                       '' AS brand,
                       to_json({'user_id': CAST(user_id AS VARCHAR),
                                'n_orders': n_orders,
                                'note': 'first order treated as signup'})
                           AS event_attributes,
                       'customer' AS entity_type,
                       CAST(user_id AS VARCHAR) AS entity_id,
                       'instacart_orders' AS source_table,
                       0.0 AS value,
                       0 AS _rank, 0 AS _oid, 0 AS _pos
                FROM (
                    SELECT user_id, min(ts_naive) AS first_ts, count(*) AS n_orders
                    FROM ord GROUP BY user_id
                )
                UNION ALL
                SELECT 'order_placed', CAST(user_id AS VARCHAR),
                       strftime(ts_naive, '%Y-%m-%dT%H:%M:%S+00:00'), '',
                       to_json({'order_id': order_id, 'eval_set': eval_set,
                                'order_number': order_number,
                                'order_dow': order_dow,
                                'order_hour_of_day': order_hour_of_day,
                                'days_since_prior_order': days_since_prior_order,
                                'n_items': n_items,
                                'days_before_ref_end': day_number}),
                       'order', CAST(order_id AS VARCHAR), 'instacart_orders',
                       CAST(n_items AS DOUBLE),
                       1, order_id, 0
                FROM ord
                UNION ALL
                SELECT 'add_to_cart', CAST(user_id AS VARCHAR),
                       strftime(ts, '%Y-%m-%dT%H:%M:%S+00:00'),
                       coalesce(department, ''),
                       to_json({'order_id': order_id, 'product_id': product_id,
                                'product_name': product_name, 'aisle': aisle,
                                'department': department,
                                'add_to_cart_order': add_to_cart_order,
                                'reordered': reordered}),
                       'product', CAST(product_id AS VARCHAR),
                       'instacart_order_products', 1.0,
                       2, order_id, add_to_cart_order
                FROM items
            )
            SELECT customer_key, event_ts, brand, event_type, event_attributes,
                   entity_type, entity_id, source_table, value,
                   _rank, _oid, _pos
            FROM ev
        """)
        # row_number over NARROW keys only (payload is ~11GiB and cannot be
        # pinned through a window sort); ids join back afterwards.
        con.execute("""
            CREATE OR REPLACE TABLE ids AS
            SELECT customer_key, event_ts, _rank, _oid, _pos,
                   CAST(row_number() OVER (
                       ORDER BY customer_key, event_ts, _rank, _oid, _pos
                   ) AS BIGINT) AS event_id
            FROM (SELECT customer_key, event_ts, _rank, _oid, _pos FROM ev)
        """)
        con.execute("""
            CREATE OR REPLACE TABLE customer_events AS
            SELECT i.event_id, e.customer_key, e.event_ts, e.brand, e.event_type,
                   e.event_attributes, e.entity_type, e.entity_id,
                   e.source_table, e.value
            FROM ev e JOIN ids i
              USING (customer_key, event_ts, _rank, _oid, _pos)
        """)
        # 1:1 keys: a fan-out would duplicate events (same key twice)
        n_ev = con.execute("SELECT count(*) FROM ev").fetchone()[0]
        n_ids = con.execute("SELECT count(*) FROM ids").fetchone()[0]
        n = con.execute("SELECT count(*) FROM customer_events").fetchone()[0]
        if not (n == n_ids == n_ev):
            raise SystemExit(f"event key collision: ev={n_ev} ids={n_ids} joined={n}")
        con.execute("DROP TABLE ev")
        con.execute("DROP TABLE ids")
        con.execute(
            "CREATE INDEX idx_customer_events_customer_ts "
            "ON customer_events(customer_key, event_ts)"
        )
        # plugins/base.py labels from an `orders` business table (customer_id,
        # order_ts, gross_margin). Instacart has NO monetary data -> gross_margin
        # is NULL by construction: binary targets (purchase propensity) work,
        # margin/CLV targets are unsupported on this fixture (receipt notes it).
        con.execute("""
            CREATE TABLE orders AS
            SELECT CAST(user_id AS VARCHAR) AS customer_id,
                   strftime(ts_naive, '%Y-%m-%dT%H:%M:%S+00:00') AS order_ts,
                   CAST(NULL AS DOUBLE) AS gross_margin,
                   order_id, n_items
            FROM ord
        """)
        con.execute("CREATE INDEX idx_orders_customer_ts ON orders(customer_id, order_ts)")
        n_orders_tbl = con.execute("SELECT count(*) FROM orders").fetchone()[0]

        # ---- receipt -------------------------------------------------------
        hist = dict(
            con.execute("SELECT event_type, count(*) FROM customer_events GROUP BY 1").fetchall()
        )
        rng = con.execute("SELECT min(event_ts), max(event_ts) FROM customer_events").fetchone()
        n_customers = con.execute(
            "SELECT count(DISTINCT customer_key) FROM customer_events"
        ).fetchone()[0]
        dow_match = con.execute(
            """
            SELECT avg(CASE WHEN gap_prev IS NOT NULL AND ts_dow = order_dow
                        THEN 1.0 ELSE 0.0 END)
            FROM ord WHERE gap_prev IS NOT NULL
            """
        ).fetchone()[0]
        censored = con.execute(
            "SELECT avg(CASE WHEN gap_prev >= 30 THEN 1.0 ELSE 0.0 END) "
            "FROM ord WHERE gap_prev IS NOT NULL"
        ).fetchone()[0]
        max_span = con.execute("SELECT max(cum_last) FROM ord").fetchone()[0]
    finally:
        con.close()

    receipt = {
        "source": "kaggle.com/datasets/psparks/instacart-market-basket-analysis",
        "config": {
            "ref_end": f"{REF_END}T00:00:00+00:00",
            "offset_window_days": OFFSET_WINDOW,
            "memory_limit": MEMORY_LIMIT,
            "timeline": "days_since_prior_order (right-censored at 30) + order_hour_of_day",
        },
        "input_sha256": {name: _sha256(src / name) for name in SOURCE_FILES},
        "n_customers": int(n_customers),
        "n_events": int(n),
        "n_orders_table": int(n_orders_tbl),
        "monetary_data": "none — orders.gross_margin is NULL (binary targets only)",
        "event_hist": {k: int(v) for k, v in hist.items()},
        "event_ts_range": list(rng),
        "dow_field_match_frac": round(float(dow_match or 0.0), 6),
        "dsp_censored_frac": round(float(censored or 0.0), 6),
        "max_reconstructed_span_days": int(max_span),
        "arrow_written": bool(arrow),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }

    if arrow:
        from rabbit_hole.stream import read_frame, write_frame

        write_frame(str(out_dir / "customer_event_stream.feather"), read_frame(str(db_path)))
    (out_dir / "instacart_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="directory with the six CSVs")
    p.add_argument("--out", type=Path, default=Path("rabbit_hole/data/instacart"))
    p.add_argument("--arrow", action="store_true", help="also write the feather copy")
    args = p.parse_args()
    r = build(args.src, args.out, arrow=args.arrow)
    print(
        f"instacart stream: {r['n_events']:,} events / {r['n_customers']:,} customers "
        f"-> {args.out}/customer_event_stream.duckdb ({r['wall_seconds']}s)",
        flush=True,
    )
    print(f"receipt: {args.out}/instacart_receipt.json", flush=True)


if __name__ == "__main__":
    main()
