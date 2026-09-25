"""CFM state operations (fade/absorb) and the DuckDB state store."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from looking_glass.cfm_config import CFMConfig, LN2, _to_epoch
from looking_glass.cfm_data import build_sequences
from looking_glass.cfm_model import CFM


# ---------------------------------------------------------------------------
# state operations (fade / absorb)
# ---------------------------------------------------------------------------
def fade(h: torch.Tensor, dt_seconds: float, half_life_days: float) -> torch.Tensor:
    """Decay the state when time passes with NO events."""
    if dt_seconds <= 0:
        return h
    decay = math.exp(-LN2 * dt_seconds / max(half_life_days * 86400.0, 1.0))
    return h * decay


def absorb(model: CFM, seq, h0: torch.Tensor | None = None, as_of_epoch: float | None = None):
    """Push new events through the recurrence from an existing state.

    Fades h0 from as_of to the first new event, then runs only the new events.
    Returns (new_h, embedding).
    """
    with torch.no_grad():
        h = h0
        if h is not None and as_of_epoch is not None and len(seq["event_ts"]):
            h = fade(
                h,
                _to_epoch(seq["event_ts"][0]) - as_of_epoch,
                getattr(model, "half_life_days", 30.0),
            )
        y, h = model(seq, h0=h)
        return h, model.embed(h)


# ---------------------------------------------------------------------------
# state store (DuckDB): customer_state + anchor_embeddings
# ---------------------------------------------------------------------------
class StateStore:
    def __init__(self, path, cfg: CFMConfig, model: CFM):
        import duckdb

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.cfg = cfg
        self.model = model
        self.con = duckdb.connect(self.path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS customer_state ("
            "customer_key TEXT, as_of_epoch DOUBLE, version TEXT, dim INT, "
            "state FLOAT[], embedding FLOAT[], last_event_ts TEXT)"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS anchor_embeddings ("
            "customer_key TEXT, anchor_epoch DOUBLE, version TEXT, dim INT, embedding FLOAT[])"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS encoder_samples ("
            "customer_key TEXT, split TEXT, version TEXT)"
        )

    def write_splits(self, keys, split):
        """Persist the sample-A/B assignment (strict disjoint) as a reusable table."""
        import polars as pl

        self.con.execute(
            "CREATE OR REPLACE TABLE encoder_samples (customer_key TEXT, split TEXT, version TEXT)"
        )
        df = pl.DataFrame(
            {
                "customer_key": list(keys),
                "split": [split[k] for k in keys],
                "version": [self.cfg.tag] * len(keys),
            }
        )
        self.con.register("_splits", df)
        try:
            self.con.execute(
                "INSERT INTO encoder_samples SELECT customer_key, split, version FROM _splits"
            )
        finally:
            self.con.unregister("_splits")

    def get_state(self, key):
        row = self.con.execute(
            "SELECT as_of_epoch, state FROM customer_state WHERE customer_key=? "
            "ORDER BY as_of_epoch DESC LIMIT 1",
            [key],
        ).fetchone()
        if not row or row[1] is None:
            return None, None
        return torch.tensor(row[1], dtype=torch.float32), float(row[0])

    def upsert(self, key, h, emb, as_of_epoch, last_event_ts):
        self.con.execute("DELETE FROM customer_state WHERE customer_key=?", [key])
        self.con.execute(
            "INSERT INTO customer_state VALUES (?,?,?,?,?,?,?)",
            [key, as_of_epoch, self.cfg.tag, len(h), h.tolist(), emb.tolist(), last_event_ts],
        )

    def advance(self, seqs, incremental=True):
        """fade to the first new event, absorb the events, persist."""
        for seq in seqs:
            h0, as_of = self.get_state(seq["customer"]) if incremental else (None, None)
            h, emb = absorb(self.model, seq, h0=h0, as_of_epoch=as_of if h0 is not None else None)
            self.upsert(
                seq["customer"], h, emb, _to_epoch(seq["event_ts"][-1]), seq["event_ts"][-1]
            )

    def add_training(self, key, anchor_epoch, emb):
        self.con.execute(
            "INSERT INTO anchor_embeddings VALUES (?,?,?,?,?)",
            [key, anchor_epoch, self.cfg.tag, len(emb), emb.tolist()],
        )

    def fade_idle(self, now_epoch):
        """Lazily advance as_of of idle states (decay only; no events)."""
        rows = self.con.execute(
            "SELECT customer_key, as_of_epoch, state FROM customer_state"
        ).fetchall()
        for key, as_of, state in rows:
            h = fade(
                torch.tensor(state, dtype=torch.float32),
                now_epoch - as_of,
                self.cfg.state_half_life_days,
            )
            emb = self.model.embed(h)
            self.upsert(key, h, emb, now_epoch, None)

    def count(self):
        s = self.con.execute("SELECT count(*) FROM customer_state").fetchone()[0]
        t = self.con.execute("SELECT count(*) FROM anchor_embeddings").fetchone()[0]
        return s, t

    def close(self):
        self.con.close()


def build_products(cfg, model, vocab, df, keys, split):
    # Rebuild fresh each release so embedding dims never mix across runs.
    pdb = Path(cfg.out_dir) / "cfm_products.duckdb"
    if pdb.exists():
        pdb.unlink()
    store = StateStore(pdb, cfg, model)
    store.write_splits(keys, split)
    # inference states for every customer (full history, as_of = last event)
    store.advance(build_sequences(df, keys, cfg, split, with_anchors=False), incremental=False)
    # training embeddings for B at anchors
    for seq in build_sequences(df, keys, cfg, split, with_anchors=True):
        if seq["group"] == "B" and seq["anchor_epoch"] is not None:
            with torch.no_grad():
                y, h = model(seq)
                store.add_training(seq["customer"], seq["anchor_epoch"], model.donor_seq(y, h, seq))
    s, t = store.count()
    store.close()
    return s, t
