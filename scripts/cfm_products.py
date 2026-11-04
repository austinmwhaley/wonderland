"""Shared loader for the cfm_v2k B-row producers (DEC-585).

One definition for the served-version resolution + the registry -> live-config replay that
`build_products` itself uses: the products know which encoder served them, and THAT registry
is the identity (DEC-448). The producers import this rather than re-typing the resolution,
which is how a mis-stamp happens (DEC-515: a receipt stamped against a config that did not
produce the bytes).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np


def _db_path(products: str) -> Path:
    """The products db: `--products` is the run directory (default) or the db file itself."""
    p = Path(products)
    return p if p.suffix else p / "cfm_products.duckdb"


def resolve_served_tag(products: str):
    """(tag, registry_path): the tag the products' state_embeddings ACTUALLY carry.

    Exactly one served version must have a registry next to the products; refuse otherwise
    (fail safe — a receipt paired with the wrong config is worse than no receipt).
    """
    db = _db_path(products)
    con = duckdb.connect(str(db), read_only=True)
    try:
        served = [
            r[0] for r in con.execute("select distinct version from state_embeddings").fetchall()
        ]
    finally:
        con.close()
    if len(served) != 1:
        raise SystemExit(f"expected exactly one served version in {db}, got {served}")
    tag = served[0]
    reg = db.parent / f"registry_{tag.replace('.', '_')}.json"
    if not reg.exists():
        raise SystemExit(f"no registry for the served tag {tag!r} next to {db}")
    return tag, reg


def load_products(
    tag: str, products_dir: Path | str, *, cpu: bool = False, require_gpu: bool = False
):
    """(model, cfg, meta): the frozen encoder + its registry-replayed config.

    The same load `_rebuild_products` uses (load_frozen_encoder + _replay_run_cfg), so the
    producers encode the SAME inputs the build did. `cpu=True` moves the trunk to the CPU
    (pre-registered for the runtime benchmark); `require_gpu=True` refuses (and records via
    the caller) when cuda:0 is absent.
    """
    import torch

    from looking_glass.cfm_config import CFMConfig
    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.customer_foundation_model import _replay_run_cfg

    products_dir = Path(products_dir)
    meta = json.loads((products_dir / f"registry_{tag.replace('.', '_')}.json").read_text())
    model, rcfg = load_frozen_encoder(tag, products_dir)
    _replay_run_cfg(meta, rcfg, products_dir, CFMConfig())
    if cpu:
        model = model.cpu()
    elif require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("no cuda device — refusing (the build device is required)")
    return model, rcfg, meta


def stream_sequences(cfg, min_events: int = 1):
    """(df, keys, split, seqs): the canonical stream read + the build's sequence builder,
    over EVERY customer (the producers measure the whole population; the A/B split comes
    from the same seed the build used)."""
    from looking_glass.cfm_data import _customer_keys, _read_stream, assign_split, build_sequences

    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    seqs = list(build_sequences(df, keys, cfg, split, with_anchors=False, min_events=min_events))
    return df, keys, split, seqs


def stored_tables(products: str) -> dict:
    """The stored products the B rows read: faded state, stored counters, served E, the
    A/B populations and the day-1 close receipt row. One duckdb pass, columnar."""
    con = duckdb.connect(str(_db_path(products)), read_only=True)
    try:
        cust = {
            k: (state, ctr, last)
            for k, state, ctr, last in con.execute(
                "select customer_key, state, embedding, last_event_ts from customer_state"
            ).fetchall()
        }
        emb = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in con.execute(
                "select customer_key, embedding from state_embeddings"
            ).fetchall()
        }
        split = dict(con.execute("select customer_key, split from encoder_samples").fetchall())
        close = con.execute(
            "select as_of_epoch, n_absorbed, n_idle, n_embeddings, device, wall_seconds"
            " from state_job_receipts"
        ).fetchall()
    finally:
        con.close()
    return {
        "customer_state": cust,
        "embeddings": emb,
        "split": split,
        "close_receipts": close,
    }


def close_at(products: str) -> float:
    """The day-1 close time the stored states were faded to (the state_job_receipts
    as_of_epoch of the encoder row) — one definition for every producer that must replay
    the fade. The close order for serve_unfaded builds: materialize E from the PRE-fade h
    first, then fade — so the stored E is a function of (pre-fade h, stored counter) and
    the stored state is a function of (pre-fade h, close_at, last_event_ts)."""
    rows = [r for r in stored_tables(products)["close_receipts"] if r[4] == "encoder"]
    if not rows:
        raise SystemExit(
            f"no state_job_receipts encoder row in {products} — cannot replay the close"
        )
    return float(rows[0][0])
