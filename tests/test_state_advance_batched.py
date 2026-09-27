"""Batched state advance == the per-customer fade->absorb path.

The daily job processes thousands of customers per run; advance() now runs
same-length buckets through ONE padded forward and bulk-upserts. These tests
pin numerical equivalence against the reference path (absorb()/fade() per
sequence) on a real (random-init) CFM, for both the initial build
(incremental=False) and the live incremental case (pre-faded stored state).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from looking_glass.cfm_config import CFMConfig, _to_epoch
from looking_glass.cfm_state import StateStore, absorb
from looking_glass.cfm_model import CFM, EventVocab

ET = np.array(["view", "view", "order", "view", "order", "view", "view", "order"], dtype=object)
BR = np.array(["b1", "b1", None, "b2", "b1", "b1", "b2", "b1"], dtype=object)
ENT = np.array(["customer"] * 8, dtype=object)
EID = np.array(["c1"] * 8, dtype=object)
VAL = np.zeros(8, dtype=np.float32)


def _seq(key, day_iso, n=5, end_offset=3600.0):
    """A short sequence of n events on one day (iso timestamps, +1h steps)."""
    ets = [f"{day_iso}T{10 + i:02d}:00:00+00:00" for i in range(n)]
    ts = [_to_epoch(x) for x in ets]
    return {
        "customer": key,
        "group": "B",
        "anchor_epoch": None,
        "event_type": ET[:n],
        "brand": BR[:n],
        "entity_type": ENT[:n],
        "entity_id": EID[:n],
        "value": VAL[:n],
        "event_ts": np.array(ets, dtype=object),
        "ts": ts,
        "co": [[0.0, 0.0]] * n,
    }


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    seqs = [_seq("k1", "2025-01-01"), _seq("k2", "2025-01-02", n=7)]
    return CFM(EventVocab.build(seqs), dim=32, n_experts=1)


def _ref_absorb(model, seq, h0, as_of, hl):
    """Reference path: exactly what absorb() returns."""
    h, emb = absorb(model, seq, h0=h0, as_of_epoch=as_of)
    return h.detach().cpu().numpy(), emb.detach().cpu().numpy()


def test_advance_initial_build_matches_reference(tiny_model, tmp_path):
    cfg = CFMConfig()
    seqs = [
        _seq("k1", "2025-01-01", n=5),
        _seq("k1", "2025-01-01", n=5),
        _seq("k2", "2025-01-02", n=7),
        _seq("k3", "2025-01-03", n=2),
        _seq("k4", "2025-01-03", n=7),
        _seq("k5", "2025-01-04", n=5),
    ]
    # distinct customers
    seen = {}
    for s in seqs:
        seen[s["customer"]] = s
    seqs = list(seen.values())
    store = StateStore(tmp_path / "s.duckdb", cfg, tiny_model)
    store.advance(seqs, incremental=False)

    for s in seqs:
        h_ref, e_ref = _ref_absorb(tiny_model, s, None, None, cfg.state_half_life_days)
        row = store.con.execute(
            "SELECT as_of_epoch, state, embedding, last_event_ts FROM customer_state "
            "WHERE customer_key=?",
            [s["customer"]],
        ).fetchone()
        assert row is not None, s["customer"]
        assert row[0] == _to_epoch(s["event_ts"][-1])
        assert row[3] == str(s["event_ts"][-1])
        np.testing.assert_allclose(np.asarray(row[1]), h_ref, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(np.asarray(row[2]), e_ref, atol=1e-5, rtol=1e-4)
    store.close()


def test_advance_incremental_matches_reference(tiny_model, tmp_path):
    cfg = CFMConfig()
    first = [_seq("k1", "2025-01-01", n=5), _seq("k2", "2025-01-02", n=7)]
    store = StateStore(tmp_path / "s2.duckdb", cfg, tiny_model)
    store.advance(first, incremental=False)

    # new events days later for k1 only; k2 idle
    second = [_seq("k1", "2025-01-06", n=5)]
    # reference: absorb() takes the RAW stored state + as_of and pre-fades once
    h0, as_of = store.get_state("k1")
    assert h0 is not None
    h_ref, e_ref = _ref_absorb(tiny_model, second[0], h0, as_of, cfg.state_half_life_days)

    store.advance(second, incremental=True)
    row = store.con.execute(
        "SELECT as_of_epoch, state, embedding FROM customer_state WHERE customer_key='k1'"
    ).fetchone()
    assert row[0] == _to_epoch(second[0]["event_ts"][-1])
    np.testing.assert_allclose(np.asarray(row[1]), h_ref, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(row[2]), e_ref, atol=1e-5, rtol=1e-4)
    # idle customer untouched
    row2 = store.con.execute(
        "SELECT as_of_epoch FROM customer_state WHERE customer_key='k2'"
    ).fetchone()
    assert row2[0] == _to_epoch(first[1]["event_ts"][-1])
    store.close()
