"""Canonical event-stream schema (rabbit_hole Layer A): the five universal
fields, and folding of every non-canonical column into event_attributes."""

from __future__ import annotations

import json

import duckdb


def test_canonicalize_folds_brand_keeps_source_table():
    from rabbit_hole.schema import CANONICAL_FIELDS, canonicalize_row

    row = {
        "customer_key": "u1",
        "event_ts": "2020-01-01T00:00:00+00:00",
        "event_type": "view",
        "source_table": "ecommerce_2019",
        "brand": "xiaomi",  # non-canonical -> folded
        "entity_id": "p1",
        "value": 9.9,
        "category_id": "c1",
    }
    c = canonicalize_row(row)
    assert set(c) == set(CANONICAL_FIELDS)  # exactly the 5 canonical fields
    assert c["source_table"] == "ecommerce_2019"
    assert "brand" not in c and "entity_id" not in c
    attrs = json.loads(c["event_attributes"])
    assert attrs["brand"] == "xiaomi" and attrs["entity_id"] == "p1"
    assert attrs["value"] == 9.9 and attrs["category_id"] == "c1"


def test_canonical_view_folds_wide_columns():
    from rabbit_hole.schema import CANONICAL_FIELDS, canonical_view_sql

    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t(customer_key VARCHAR, event_ts VARCHAR, event_type VARCHAR,"
        " brand VARCHAR, event_attributes VARCHAR, value DOUBLE, source_table VARCHAR)"
    )
    con.execute(
        "INSERT INTO t VALUES ('u1','2020','view','xiaomi','{\"category_id\":\"c1\"}',9.9,'ecom')"
    )
    con.execute(canonical_view_sql("t", "v", [r[0] for r in con.execute("DESCRIBE t").fetchall()]))
    names = [d[0] for d in con.execute("SELECT * FROM v").description]
    assert names == list(CANONICAL_FIELDS)
    row = con.execute("SELECT * FROM v").fetchone()
    assert row[3] == "ecom"  # source_table canonical
    attrs = json.loads(row[4])
    assert attrs["brand"] == "xiaomi" and attrs["value"] == 9.9 and attrs["category_id"] == "c1"


def test_write_frame_canonicalizes(tmp_path):
    import polars as pl

    from rabbit_hole.stream import write_frame
    from rabbit_hole.schema import CANONICAL_FIELDS

    df = pl.DataFrame(
        {
            "customer_key": ["u1", "u1"],
            "event_ts": ["2020-01-01T00:00:00+00:00", "2020-01-02T00:00:00+00:00"],
            "event_type": ["view", "cart"],
            "source_table": ["s", "s"],
            "brand": ["acme", None],
            "value": [1.0, 2.0],
            "event_attributes": ["{}", "{}"],
        }
    )
    db = tmp_path / "s.duckdb"
    write_frame(str(db), df)
    con = duckdb.connect(str(db), read_only=True)
    cols = [r[0] for r in con.execute("DESCRIBE customer_events").fetchall()]
    assert cols == list(CANONICAL_FIELDS)  # only the 5 canonical columns
    attrs = json.loads(
        con.execute("SELECT event_attributes FROM customer_events LIMIT 1").fetchone()[0]
    )
    assert attrs["brand"] == "acme" and attrs["value"] == 1.0
    con.close()
