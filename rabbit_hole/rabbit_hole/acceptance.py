"""rabbit_hole acceptance check.

Confirms rabbit_hole does its current job: **transform wide customer tables into
one unified customer event stream that looks and behaves like real customer
activity**, so everything downstream can be tested.

Pipeline under test: ``generate_data.py`` builds the wide tables, then
``materialize_customer_event_stream`` unifies them into ``customer_events``.

    python -m rabbit_hole.acceptance
"""
from __future__ import annotations

import importlib.util
import os
import duckdb
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from .schema import CANONICAL_FIELDS, parse_attributes
from .stream import read_events, write_events

# Event vocabulary the pipeline must cover.
REQUIRED_EVENT_GROUPS = {
    "browse": ("page_view", "product_view", "search"),
    "commerce": ("add_to_cart", "order_placed", "order_returned"),
    "account": ("customer_signup",),
}
CUSTOMER_ATTRS = ("loyalty_tier", "income_band", "acquisition_channel")


def _load_generator():
    path = Path(__file__).parent / "generators" / "generate_data.py"
    spec = importlib.util.spec_from_file_location("rh_generate_data", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rh_generate_data"] = mod   # dataclass needs it registered
    spec.loader.exec_module(mod)
    return mod


_STREAM = None


def build_stream_db(seed=17, customers=300, products=150, events=8000):
    """Run the unified pipeline into a temp DuckDB; return its path."""
    gen = _load_generator()
    tmp = Path(tempfile.mkdtemp(prefix="rh_accept_"))
    db = tmp / "events.duckdb"
    conn = duckdb.connect(str(db))
    try:
        gen.create_business_tables(conn)
        gen.seed_business_data(conn, num_customers=customers,
                               num_products=products, years=1,
                               event_count=events, order_ratio=0.20,
                               min_orders_per_customer=2, seed=seed)
        gen.materialize_customer_event_stream(conn)
    finally:
        conn.close()
    return str(db)


def _iso(s):
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return None


def check(seed=17):
    global _STREAM
    db = build_stream_db(seed=seed)
    rows = read_events(db, table="customer_events")

    def r(name, target, achieved, ok):
        return {"check": name, "target": target, "achieved": achieved, "ok": ok}

    checks = []
    n = len(rows)
    nc = len({x["customer_key"] for x in rows})
    checks.append(r("produced events", ">0", n, n > 0))
    checks.append(r("distinct customers", ">0", nc, nc > 0))

    schema_ok = all(all(f in x for f in CANONICAL_FIELDS) and
                    x["customer_key"] not in (None, "") and
                    x["event_ts"] not in (None, "") and
                    x["event_type"] not in (None, "") for x in rows)
    checks.append(r("canonical schema", "all rows", schema_ok, schema_ok))

    per, ts_ok = {}, True
    for x in rows:
        t = _iso(x["event_ts"])
        if t is None:
            ts_ok = False
            break
        per.setdefault(x["customer_key"], []).append(t)
    checks.append(r("timestamps parse (ISO)", "all", ts_ok, ts_ok))
    chrono_ok = all(v == sorted(v) for v in per.values())
    checks.append(r("chronological per customer", "all", chrono_ok, chrono_ok))

    types = {x["event_type"] for x in rows}
    for group, members in REQUIRED_EVENT_GROUPS.items():
        hit = [m for m in members if m in types]
        checks.append(r(f"event group: {group}", ">=1", len(hit), len(hit) >= 1))

    # funnel causality: an order is preceded by browsing activity by that customer
    by = {}
    for x in rows:
        by.setdefault(x["customer_key"], []).append(x)
    for v in by.values():
        v.sort(key=lambda x: (x["event_ts"], x["event_type"]))
    purchases = with_cart = with_view = 0
    for seq in by.values():
        names = [x["event_type"] for x in seq]
        for i, et in enumerate(names):
            if et == "order_placed":
                purchases += 1
                if "add_to_cart" in names[:i]:
                    with_cart += 1
                if "product_view" in names[:i]:
                    with_view += 1
    cart_frac = with_cart / purchases if purchases else 0.0
    view_frac = with_view / purchases if purchases else 0.0
    checks.append(r("orders preceded by add_to_cart", ">=0.2",
                    round(cart_frac, 3), cart_frac >= 0.2))
    checks.append(r("orders preceded by product_view", ">=0.2",
                    round(view_frac, 3), view_frac >= 0.2))

    # customer attribute diversity on signup events
    attrs = {a: set() for a in CUSTOMER_ATTRS}
    for x in rows:
        if x["event_type"] == "customer_signup":
            p = parse_attributes(x["event_attributes"])
            for a in CUSTOMER_ATTRS:
                if p.get(a) is not None:
                    attrs[a].add(p[a])
    for a, vals in attrs.items():
        checks.append(r(f"signup attr diversity: {a}", ">=2", len(vals),
                        len(vals) >= 2))

    # payload richness
    payload_ok, products_seen = True, set()
    for x in rows:
        try:
            p = parse_attributes(x["event_attributes"])
        except Exception:
            payload_ok = False
            break
        if p.get("entity_type") == "product" and p.get("entity_id"):
            products_seen.add(p["entity_id"])
        elif p.get("product_id"):
            products_seen.add(p["product_id"])
    checks.append(r("payloads parse (JSON)", "all", payload_ok, payload_ok))
    checks.append(r("distinct products referenced", ">=10", len(products_seen),
                    len(products_seen) >= 10))

    # determinism
    outdir = Path(db).parent
    for eng, name in (("duckdb", "stream.duckdb"), ("parquet", "stream.parquet")):
        pth = outdir / name
        write_events(str(pth), rows)
        back = read_events(str(pth))
        fields_ok = all(all(f in r for f in CANONICAL_FIELDS) for r in back[:200])
        same = len(back) == n and fields_ok and             [r["event_ts"] for r in back] == [r["event_ts"] for r in rows]
        checks.append(r(f"{eng} round-trip (5-field)", "equal+5 fields",
                        f"{len(back)} rows", same))

    import hashlib
    def digest(rs):
        h = hashlib.sha256()
        for x in sorted((str(x["customer_key"]), str(x["event_ts"]),
                         str(x["event_type"])) for x in rs):
            h.update(("|".join(x)).encode())
        return h.hexdigest()
    db2 = build_stream_db(seed=seed)
    rows2 = read_events(db2, table="customer_events")
    same = (len(rows2) == n and digest(rows2) == digest(rows))
    checks.append(r("deterministic (same seed)", "equal",
                    f"{n} vs {len(rows2)}", same))
    return rows, checks


def check_no_duplication():
    """looking_glass must POINT at rabbit_hole, not duplicate it."""
    lg = Path(__file__).resolve().parents[2] / "looking_glass" / "scripts"
    out = []
    for name in ("duckdb", "logs"):
        p = lg / "data" / name
        ok = p.is_symlink() and "rabbit_hole" in os.readlink(p)
        out.append({"check": f"no-dup: looking_glass data/{name}",
                    "target": "symlink", "achieved": p.is_symlink(), "ok": ok})
    p = lg / "generate_data.py"
    ok = p.is_symlink() and "rabbit_hole" in os.readlink(p)
    out.append({"check": "no-dup: looking_glass generate_data.py",
                "target": "symlink", "achieved": p.is_symlink(), "ok": ok})
    # retired generators and their datasets must be gone (one unified thing)
    for gone in (lg / "generate_full.py", lg / "gen_toy.py",
                 lg / "data" / "full", lg / "data" / "toy"):
        ok = not gone.exists()
        out.append({"check": f"retired: {gone.name} absent",
                    "target": "absent", "achieved": gone.exists(), "ok": ok})
    return out


def run():
    rows, checks = check()
    checks += check_no_duplication()
    print("== RABBIT_HOLE ACCEPTANCE SCORECARD ==")
    print(f"{'check':46s} {'target':>10s} {'achieved':>12s}  status")
    for c in checks:
        print(f"{c['check']:46s} {str(c['target']):>10s} "
              f"{str(c['achieved']):>12s}  {'PASS' if c['ok'] else 'FAIL'}")
    n_pass = sum(1 for c in checks if c["ok"])
    print(f"completion: {n_pass}/{len(checks)} ({100*n_pass/len(checks):.0f}%)")
    return checks


if __name__ == "__main__":
    sys.exit(0 if all(c["ok"] for c in run()) else 1)
