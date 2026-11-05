#!/usr/bin/env python3
"""Serve the lab-E transition signature as a first-class block (B-002, DEC-603).

The lab speedometer's hierarchical head (DEC-600/601) reads the exact per-customer
transition counts that the stream carries. Before this builder those counts were
recomputed inside the measurement process from the fixture, so the E1.a/E1.b receipts
were not readouts of the frozen products object (BACKWARD B-002).

This script materialises the same counts *inside* the products artifact:

    signature_blocks(customer_key, as_of_epoch, version, classes, dim, counts, last_type)

`counts` is the row-major flattened K x K per-customer transition matrix over the
pre-as_of history, and `last_type` is the index of that customer's last pre-as_of
type. A `signature_receipt.json` records the fixture hash, the class vocabulary, the
block hash and the products identity. `scripts/lab_measure.py --head hier` refuses to
measure without both, so a receipt can only be produced by reading the served block.

The builder is CPU-only and deterministic. It does not retrain or mutate any other
table; re-running it replaces the block and receipt atomically.

    python scripts/build_lab_signature.py \
        --fixture rabbit_hole/data/lab/next_event_x4k \
        --products /tmp/opencode/lab/next_event_x4k_pos \
        --as-of 2024-09-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from looking_glass.stream_adapter import to_ts_expr  # noqa: E402
from measure_input_law import products_identity  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _classes(stream: pl.DataFrame) -> list[str]:
    """The event-type vocabulary of the stream, sorted (the prediction classes).

    The vocabulary is a property of the stream, not of the future window; using the
    whole stream keeps the block build independent of the held-out rows.
    """
    return sorted(str(t) for t in stream["event_type"].unique().to_list())


def _block(stream: pl.DataFrame, as_of: datetime, classes: list[str]):
    """(keys, counts[K*K], last_type[K]) from the pre-as_of history, sorted like the head."""
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    hist = (
        stream.filter(to_ts_expr(pl.col("event_ts")) <= as_of)
        .filter(pl.col("event_type").is_in(classes))
        .sort(["customer_key", "event_ts", "event_type"])
    )
    keys: list[str] = []
    counts: list[np.ndarray] = []
    last: list[int] = []
    for g in hist.partition_by("customer_key", maintain_order=True):
        k = str(g["customer_key"][0])
        types = np.array([idx[str(t)] for t in g["event_type"]], dtype=np.int64)
        C = np.zeros((K, K), dtype=np.float64)
        for a, b in zip(types[:-1], types[1:]):
            C[a, b] += 1.0
        keys.append(k)
        counts.append(C.reshape(-1))
        last.append(int(types[-1]))
    if not keys:
        return [], np.zeros((0, K * K), dtype=np.float64), np.zeros(0, dtype=np.int64)
    return keys, np.vstack(counts), np.asarray(last, dtype=np.int64)


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True, help="lab use-case dir with the stream")
    ap.add_argument("--products", required=True, help="products dir to serve the block in")
    ap.add_argument("--as-of", required=True, help="training boundary (YYYY-MM-DD)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    import duckdb

    t0 = time.perf_counter()
    fixture = Path(a.fixture)
    products = Path(a.products)
    stream_path = fixture / "customer_event_stream.parquet"
    db = products / "cfm_products.duckdb"
    receipt_path = products / "signature_receipt.json"
    if not stream_path.exists():
        raise SystemExit(f"no stream at {stream_path}")
    if not db.exists():
        raise SystemExit(f"no products db at {db}")
    if receipt_path.exists() and not a.overwrite:
        raise SystemExit(f"{receipt_path} exists; pass --overwrite to rebuild the block")

    as_of = datetime.strptime(a.as_of, "%Y-%m-%d")
    stream = pl.read_parquet(stream_path)
    classes = _classes(stream)
    keys, counts, last = _block(stream, as_of, classes)

    con = duckdb.connect(str(db))
    try:
        version = con.execute("select distinct version from state_embeddings").fetchall()
        if len(version) != 1:
            raise SystemExit(f"expected one served version in {db}, got {version}")
        tag = str(version[0][0])
        known = {
            str(r[0]) for r in con.execute("select customer_key from state_embeddings").fetchall()
        }
        missing = sorted(set(keys) - known)
        if missing:
            raise SystemExit(
                f"{len(missing)} signature customers are not in the products state_embeddings"
            )
        con.execute("drop table if exists signature_blocks")
        con.execute(
            "create table signature_blocks ("
            "customer_key varchar, as_of_epoch double, version varchar, "
            "classes varchar[], dim integer, counts double[], last_type integer)"
        )
        if keys:
            con.executemany(
                "insert into signature_blocks values (?,?,?,?,?,?,?)",
                [
                    [k, as_of.timestamp(), tag, classes, len(classes) ** 2, c.tolist(), int(t)]
                    for k, c, t in zip(keys, counts, last, strict=True)
                ],
            )
        # A deterministic hash of the served block, in customer-key order.
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        h = hashlib.sha256()
        for i in order:
            h.update(keys[i].encode())
            h.update(np.asarray(counts[i], dtype=np.float64).tobytes())
            h.update(int(last[i]).to_bytes(4, "little", signed=True))
        block_sha = h.hexdigest()
        served = int(con.execute("select count(*) from signature_blocks").fetchone()[0])
    finally:
        con.close()

    body = {
        "kind": "lab_signature_block",
        "fixture": str(fixture),
        "fixture_sha256": _sha256(stream_path),
        "as_of": a.as_of,
        "version": tag,
        "classes": classes,
        "dim": len(classes) ** 2,
        "n_customers": len(keys),
        "n_stream_customers": int(stream["customer_key"].n_unique()),
        "n_with_history": served,
        "block_sha256": block_sha,
        "products": products_identity(str(products)),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    receipt_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(f"served signature block -> {db}")
    print(f"  classes={classes} dim={body['dim']} rows={served} sha={block_sha[:16]}")
    print(f"  receipt -> {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
