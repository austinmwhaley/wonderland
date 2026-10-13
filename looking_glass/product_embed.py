"""Model-free PRODUCT embeddings from basket co-occurrence (Layer A, §20.2).

Products that co-occur in the same basket should sit near each other — like word2vec
for items. This is computed from the raw stream (order → set of products), with NO
labels and NO customer targets, so it is a legitimate, compact, reusable feature:

  * fixes the SF φ gap (product affinity in successor features) at a bounded dim,
  * is the "attach product embeddings to the data, then train on everything" step.

Output: an `.npz` with `ids` (entity_id strings) and `vec` (float32 [n, d]).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cooccurrence_embed(
    stream: str,
    as_of: str,
    dim: int = 16,
    min_count: int = 3,
    order_key="order_id",
    max_products: int = 5000,
):
    import duckdb

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(stream)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else "customer_events"
    # PRODUCT rows (entity_type='product') grouped by their basket key. This is
    # stream-agnostic: it does not assume a conversion-event name, only that product
    # rows carry an order/session key and an entity_id.
    tc = f"epoch(CAST('{as_of}' AS TIMESTAMPTZ))"
    rows = con.execute(
        f"SELECT json_extract_string(event_attributes,'$.{order_key}') oid,"
        f" json_extract_string(event_attributes,'$.entity_id') eid"
        f" FROM {src} WHERE json_extract_string(event_attributes,'$.entity_type') = 'product'"
        f" AND json_extract_string(event_attributes,'$.entity_id') IS NOT NULL"
        f" AND json_extract_string(event_attributes,'$.{order_key}') IS NOT NULL"
        f" AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {tc}"
    ).fetchall()
    con.close()
    # order -> set(products)
    baskets: dict = {}
    for oid, eid in rows:
        baskets.setdefault(oid, set()).add(str(eid))
    baskets = [s for s in baskets.values() if len(s) >= 2]
    # cap to the most FREQUENT products so the co-occurrence matrix stays bounded
    from collections import Counter

    freq = Counter(e for s in baskets for e in s)
    top = {e for e, _ in freq.most_common(max_products)}
    baskets = [s & top for s in baskets]
    baskets = [s for s in baskets if len(s) >= 2]
    ids = sorted({e for s in baskets for e in s})
    if not ids or not baskets:
        return None, None
    idx = {e: i for i, e in enumerate(ids)}
    n = len(ids)
    C = np.zeros((n, n), dtype=np.float32)
    for s in baskets:
        v = np.array([idx[e] for e in s])
        C[np.ix_(v, v)] += 1.0
    np.fill_diagonal(C, 0.0)
    # PMI-style reweighting + truncated SVD
    tot = C.sum()
    P = C / max(tot, 1.0)
    row = P.sum(1, keepdims=True)
    pmi = np.log((P + 1e-9) / (row @ row.T + 1e-9))
    pmi = np.nan_to_num(pmi)
    # keep products with enough co-occurrence mass
    keep = C.sum(1) >= min_count
    if keep.sum() < dim + 1:
        keep = np.ones(n, dtype=bool)
    M = pmi[keep]
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    emb = U[:, :dim] * np.sqrt(S[:dim])
    ids_keep = [ids[i] for i in np.flatnonzero(keep)]
    return ids_keep, emb.astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--order-key", default="order_id", dest="order_key")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ids, emb = cooccurrence_embed(a.stream, a.as_of, dim=a.dim, order_key=a.order_key)
    if ids is None:
        print("no co-occurrence structure found")
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, ids=np.array(ids, dtype=object), vec=emb)
    print(f"wrote {len(ids)} product embeddings (dim {emb.shape[1]}) -> {a.out}")


if __name__ == "__main__":
    main()
