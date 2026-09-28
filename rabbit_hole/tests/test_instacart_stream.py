"""Fixture test: Instacart CSVs -> canonical stream (no network; tiny CSVs)."""

from __future__ import annotations

import duckdb
import polars as pl

from rabbit_hole.generators.instacart_stream import CANONICAL_COLS, build


def _write_src(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # 4 users: u1 dense (gaps < 30), u2 one censored gap (dsp = 30),
    # u3 sparse with an itemless final order (eval_set=test path),
    # u4 two orders only (>=3 events incl. signup is enforced upstream in CFM,
    # not here). dow fields deliberately inconsistent with gaps (real data is).
    orders = pl.DataFrame(
        {
            "order_id": [101, 102, 103, 201, 202, 301, 302, 401, 402],
            "user_id": [1, 1, 1, 2, 2, 3, 3, 4, 4],
            "eval_set": [
                "prior",
                "prior",
                "prior",
                "prior",
                "prior",
                "prior",
                "test",
                "prior",
                "prior",
            ],
            "order_number": [1, 2, 3, 1, 2, 1, 2, 1, 2],
            "order_dow": [1, 1, 1, 3, 0, 5, 5, 6, 6],  # not gap-consistent
            "order_hour_of_day": ["08", "10", "12", "07", "07", "09", "09", "23", "00"],
            "days_since_prior_order": [None, 5.0, 7.0, None, 30.0, None, 12.0, None, 3.0],
        }
    )
    orders.write_csv(src / "orders.csv")
    items = pl.DataFrame(
        {
            "order_id": [101, 101, 102, 103, 103, 201, 202, 301, 401, 402],
            "product_id": [1, 2, 3, 1, 4, 5, 6, 1, 2, 3],
            "add_to_cart_order": [1, 2, 1, 1, 2, 1, 1, 1, 1, 1],
            "reordered": [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        }
    )
    items.filter(pl.col("order_id") != 302).write_csv(src / "order_products__prior.csv")
    items.filter(pl.col("order_id") == 302).clear().write_csv(src / "order_products__train.csv")
    pl.DataFrame(
        {
            "product_id": [1, 2, 3, 4, 5, 6],
            "product_name": [f"P{i}" for i in range(1, 7)],
            "aisle_id": [1, 1, 2, 2, 1, 2],
            "department_id": [10, 10, 20, 20, 10, 20],
        }
    ).write_csv(src / "products.csv")
    pl.DataFrame({"aisle_id": [1, 2], "aisle": ["a1", "a2"]}).write_csv(src / "aisles.csv")
    pl.DataFrame({"department_id": [10, 20], "department": ["d1", "d2"]}).write_csv(
        src / "departments.csv"
    )
    return src


def test_build_canonical_stream(tmp_path):
    src = _write_src(tmp_path)
    out = tmp_path / "out"
    r1 = build(src, out)

    db = out / "customer_event_stream.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    try:
        cols = [c[1] for c in con.execute("PRAGMA table_info(customer_events)").fetchall()]
        assert cols == list(CANONICAL_COLS)
        n = con.execute("SELECT count(*) FROM customer_events").fetchone()[0]
        assert n == r1["n_events"] == sum(r1["event_hist"].values())
        assert set(r1["event_hist"]) == {"customer_signup", "order_placed", "add_to_cart"}
        # chronological per customer (event_id order)
        viol = con.execute("""
            SELECT count(*) FROM (
              SELECT event_ts, lag(event_ts) OVER (PARTITION BY customer_key
                                                   ORDER BY event_id) p
              FROM customer_events)
            WHERE p IS NOT NULL AND p > event_ts
        """).fetchone()[0]
        assert viol == 0
        # signup precedes (or ties) its customer's first order
        first_order = con.execute("""
            SELECT count(*) FROM (
              SELECT customer_key, min(event_ts) FILTER (WHERE event_type='order_placed') o,
                     min(event_ts) FILTER (WHERE event_type='customer_signup') s
              FROM customer_events GROUP BY customer_key)
            WHERE s IS NOT NULL AND o IS NOT NULL AND s > o
        """).fetchone()[0]
        assert first_order == 0
        # wide join landed: item attributes + department as brand
        row = con.execute("""
            SELECT brand, entity_type, entity_id, event_attributes
            FROM customer_events WHERE event_type='add_to_cart' LIMIT 1
        """).fetchone()
        assert row[0] in ("d1", "d2") and row[1] == "product" and '"product_name"' in row[3]
        # plugin contract table: orders with NULL margin (no money in source)
        n_orders, n_margin = con.execute(
            "SELECT count(*), count(gross_margin) FROM orders"
        ).fetchone()
        assert n_orders == r1["n_orders_table"] == r1["event_hist"]["order_placed"]
        assert n_margin == 0
        # window bound: nothing after REF_END (2026-01-01) / before 2024-10-04
        mx = con.execute("SELECT max(event_ts) FROM customer_events").fetchone()[0]
        assert mx < "2026-01-02"
    finally:
        con.close()

    assert r1["n_customers"] == 4
    assert 0.0 <= r1["dow_field_match_frac"] <= 1.0
    assert r1["dsp_censored_frac"] > 0  # u2 has the censored 30d gap
    assert all(len(h) == 64 for h in r1["input_sha256"].values())


def test_build_is_deterministic(tmp_path):
    src = _write_src(tmp_path)
    r1 = build(src, tmp_path / "out1")
    r2 = build(src, tmp_path / "out2")
    assert r1["input_sha256"] == r2["input_sha256"]
    q = (
        "SELECT event_id, customer_key, event_ts, event_type, event_attributes "
        "FROM customer_events ORDER BY event_id"
    )
    with (
        duckdb.connect(
            str(tmp_path / "out1" / "customer_event_stream.duckdb"), read_only=True
        ) as c1,
        duckdb.connect(
            str(tmp_path / "out2" / "customer_event_stream.duckdb"), read_only=True
        ) as c2,
    ):
        assert c1.execute(q).fetchall() == c2.execute(q).fetchall()
