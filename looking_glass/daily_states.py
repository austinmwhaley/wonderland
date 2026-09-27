"""Daily state job (Layer B) — the ONLY writer of state/embedding tables.

Runs once per calendar day D:
  1. cut the stream at D (window SQL-push; only events since the earliest
     live state are read),
  2. advance every customer's state: absorb the day's new events (fade to the
     first one), fade the remainder (no events),
  3. MATERIALIZE the inference embeddings: donor(h) for every customer into
     ``state_embeddings`` at as_of = D.

Plugin inference (Layer C) is then a pure READ of ``state_embeddings`` — it
never mutates state or embeddings. Runs on GPU when available (the encoder's
forward is device-aware); receipts land in ``state_job_receipts``.

  python -m looking_glass.daily_states --as-of 2025-11-02
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

CFM_DIR = Path(__file__).resolve().parents[0] / "artifacts" / "cfm"


def _products_tag(products: Path) -> str:
    import duckdb

    con = duckdb.connect(str(products), read_only=True)
    try:
        vers = [r[0] for r in con.execute("SELECT DISTINCT version FROM customer_state").fetchall()]
    finally:
        con.close()
    if len(vers) != 1:
        raise ValueError(f"products db has {len(vers)} state versions ({vers}); expected exactly 1")
    return vers[0]


def assert_forward_only(products: Path, day: float) -> None:
    """The daily chain only moves forward: refuse to relabel states into the past.

    States already carry an ``as_of``; running a job for an earlier day would
    rewind labels while their content is newer (time travel = silent lie).
    """
    import duckdb

    from datetime import datetime, timezone

    con = duckdb.connect(str(products), read_only=True)
    try:
        hi = con.execute("SELECT max(as_of_epoch) FROM customer_state").fetchone()[0]
    finally:
        con.close()
    if hi is not None and float(hi) > day + 1.0:  # 1s tolerance
        at = datetime.fromtimestamp(float(hi), tz=timezone.utc).isoformat()
        raise ValueError(
            f"state store is already at {at}; daily jobs run forward only "
            f"(requested as_of is in the past — states would be relabeled backwards)"
        )


def update_day(as_of: str, products=None, cfm_dir=None, device: str | None = None) -> dict:
    """Advance states to ``as_of`` and materialize that day's embeddings."""
    import duckdb
    import torch

    from looking_glass.cfm_config import _to_epoch
    from looking_glass.cfm_data import read_stream_window
    from looking_glass.cfm_state import StateStore, advance_to_date, load_frozen_encoder

    t0 = time.perf_counter()
    products = Path(products or (CFM_DIR / "cfm_products.duckdb"))
    cfm_dir = Path(cfm_dir or CFM_DIR)
    tag = _products_tag(products)
    day = _to_epoch(as_of)
    assert_forward_only(products, day)  # fail fast, before any work
    model, cfg = load_frozen_encoder(tag, cfm_dir)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    cfg.as_of = as_of

    # read only the window since the earliest live state (SQL-pushed)
    con = duckdb.connect(str(products), read_only=True)
    try:
        lo = con.execute("SELECT min(as_of_epoch) FROM customer_state").fetchone()[0]
    finally:
        con.close()
    df = read_stream_window(cfg, None if lo is None else float(lo), day)

    store = StateStore(products, cfg, model)
    try:
        n_abs, n_idle = advance_to_date(store, cfg, df, day)
        n_emb = store.materialize_state_embeddings(model, day)
        wall = round(time.perf_counter() - t0, 2)
        now = datetime.now(timezone.utc)
        store.con.execute(
            "INSERT INTO state_job_receipts VALUES (?,?,?,?,?,?,?,?)",
            [float(day), tag, int(n_abs), int(n_idle), int(n_emb), dev, wall, now],
        )
    finally:
        store.close()

    receipt = {
        "as_of": as_of,
        "encoder_version": tag,
        "device": dev,
        "n_absorbed": int(n_abs),
        "n_idle": int(n_idle),
        "n_embeddings": int(n_emb),
        "wall_seconds": wall,
    }
    print(
        f"[daily-state] {as_of}  tag={tag} dev={dev}  absorbed={n_abs} idle={n_idle} "
        f"embeddings={n_emb}  {wall}s",
        flush=True,
    )
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Layer-B daily state + inference-embedding job")
    ap.add_argument("--as-of", required=True, help="ISO date (the day being closed)")
    ap.add_argument("--products", default=None)
    ap.add_argument("--cfm-dir", default=None)
    ap.add_argument("--device", default=None, help="cpu|cuda (default: cuda when available)")
    a = ap.parse_args(argv)
    update_day(a.as_of, products=a.products, cfm_dir=a.cfm_dir, device=a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
