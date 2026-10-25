"""Canonical discovery: raw stream in, schema + sequences out, then fit/transform."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looking_glass import stream_adapter as SA  # noqa: E402

AS_OF = datetime.datetime(2024, 3, 1).timestamp()


def _stream(n_customers=12, n_events=18, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = datetime.datetime(2024, 1, 1) + datetime.timedelta(days=int(rng.integers(0, 20)))
        segment = c % 3  # constant per customer -> identity field
        for e in range(n_events):
            ts = base + datetime.timedelta(hours=int(rng.integers(0, 400)))
            is_action = e % 4 == 0
            attrs = {
                "category": ["apparel", "home", "outdoor"][int(rng.integers(0, 3))],
                "price": float(rng.integers(5, 200)),
                "channel": int(rng.integers(0, 4)),
                "segment": segment,
                "signup_ts": "2023-12-01T00:00:00",
            }
            if is_action:
                attrs["target_key"] = "channel"
            rows.append(
                {
                    "customer_key": f"c{c:03d}",
                    "event_ts": ts.isoformat(),
                    "event_type": "email_send" if is_action else "view",
                    "event_attributes": __import__("json").dumps(attrs),
                }
            )
    return pl.DataFrame(rows)


def test_discover_builds_a_usable_schema():
    found = SA.discover(_stream(), AS_OF)
    assert found["keys"] and len(found["seqs"]) == len(found["keys"])
    names = [g["field"] for g in found["catspec"]]
    assert "category" in names and "channel" in names
    # numeric field is numeric, ISO attribute is a time delta
    num_names = [f for f, _k in found["num"]]
    assert "price" in num_names and "signup_ts" in num_names
    # behavioural identity: `segment` is constant per customer
    seg_idx = names.index("segment")
    assert seg_idx in found["vs"]["id_idx"]
    # exogenous action event discovered from the declared/structural action signal
    assert found["vs"]["exo_idx"], "email_send should be exogenous"
    assert found["taus"].shape[0] >= 2
    assert len(found["periods"]) >= 1


def test_discover_sequences_match_serving_transform():
    found = SA.discover(_stream(), AS_OF)
    key = found["keys"][0]
    keys2, seqs2, _ = SA.transform(_stream(), found["schema"], AS_OF)
    assert keys2 == found["keys"]
    for j in range(7):
        a = np.nan_to_num(found["seqs"][key][j])
        b = np.nan_to_num(seqs2[key][j])
        assert a.shape == b.shape
        assert np.allclose(a, b, atol=1e-6)


def test_discover_rejects_empty_and_missing_as_of():
    with pytest.raises(ValueError):
        SA.discover(_stream().head(0), AS_OF)
    with pytest.raises(ValueError):
        SA.discover(_stream(), 0.0)


def test_fit_stream_produces_a_transforming_artifact():
    """End-to-end: raw stream -> fit -> transform, no prepared bundle or wide table."""
    from looking_glass.foundation import FrozenFoundation

    frame = _stream(n_customers=60, n_events=16)
    art = FrozenFoundation.fit_stream(frame, AS_OF, seed=0, epochs=1, d=16, layers=1)
    assert art.meta.schema is not None
    keys, state, rcpt = art.transform_events(frame)
    assert rcpt["entities"] == len(keys)
    assert state["shape"].shape[1] == 16
    assert np.isfinite(state["shape"]).all()
    assert state["full"].shape[1] == 16 + art.describe()["blk_dim"]
