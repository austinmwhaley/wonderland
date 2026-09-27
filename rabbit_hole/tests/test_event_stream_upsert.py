"""Partitioned upserts + per-customer indexes on the canonical stream.

Contract: materializing twice must REPLACE each event_type partition in place
(upsert semantics) — never duplicate rows, never wipe the table first — and the
per-customer indexes that serve sequence/anchor/label lookups must exist.
"""

from __future__ import annotations

import duckdb

from rabbit_hole.generators import generate_data as gen


def test_partition_upserts_and_customer_indexes(tmp_path):
    db = tmp_path / "stream.duckdb"
    conn = duckdb.connect(str(db))
    try:
        gen.create_business_tables(conn)
        gen.seed_business_data(
            conn,
            num_customers=40,
            num_products=12,
            years=1,
            event_count=400,
            order_ratio=0.20,
            min_orders_per_customer=1,
            seed=7,
        )
        n1 = gen.materialize_customer_event_stream(conn)
        by1 = dict(
            conn.execute("SELECT event_type, count(*) FROM customer_events GROUP BY 1").fetchall()
        )
        assert n1 > 0 and "order_placed" in by1

        # rerun = per-partition replace: identical counts, zero duplication
        n2 = gen.materialize_customer_event_stream(conn)
        by2 = dict(
            conn.execute("SELECT event_type, count(*) FROM customer_events GROUP BY 1").fetchall()
        )
        assert n2 == n1 and by2 == by1

        # a stale row inside a partition is replaced (upsert), not appended to
        conn.execute(
            "INSERT INTO customer_events (customer_key, event_ts, event_type, entity_type, "
            "entity_id, source_table, value, event_attributes, brand) VALUES "
            "('zz', '2099-01-01T00:00:00+00:00', 'order_placed', 'order', 'BOGUS', "
            "'orders', 0.0, '{}', '')"
        )
        assert conn.execute("SELECT count(*) FROM customer_events").fetchone()[0] == n1 + 1
        gen.materialize_customer_event_stream(conn)
        assert conn.execute("SELECT count(*) FROM customer_events").fetchone()[0] == n1
        assert (
            conn.execute("SELECT count(*) FROM customer_events WHERE entity_id='BOGUS'").fetchone()[
                0
            ]
            == 0
        )

        # per-customer traversal indexes (sequences, anchors/labels, email)
        idx = {r[0] for r in conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
        assert "idx_customer_events_customer_ts" in idx
        assert "idx_orders_customer_ts" in idx
        assert "idx_contact_sends_customer" in idx
    finally:
        conn.close()
