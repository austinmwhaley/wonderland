"""Supervised comparator generality (DEC-614 / B-022).

The model-free RFM comparator must DERIVE the stream's monetary field instead of
naming "value" (Universal/P1: no hardcoded field names in the pipeline). These
hermetic tests pin that an unseen stream whose dominant numeric field has another
name is consumed, not crashed on, and that the features depend on the data, not
the column name. They also pin the numeric-free fallback (zeroed monetary
aggregates, not a crash).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from looking_glass.supervised_gate import _raw_features


def _cfg(**over):
    base = dict(as_of="2025-06-01", numeric_fields=None)
    base.update(over)
    return SimpleNamespace(**base)


def _frame(monetary_col, events):
    """A tiny non-ecommerce stream: customer_key, event_ts, event_type, + one numeric."""
    return pl.DataFrame(
        {
            "customer_key": [r[0] for r in events],
            "event_ts": [r[1] for r in events],
            "event_type": [r[2] for r in events],
            monetary_col: [r[3] for r in events],
        }
    )


_EVENTS = [
    ("a", "2025-01-01T00:00:00", "telemetry", 1.0),
    ("a", "2025-02-01T00:00:00", "alarm", 2.0),
    ("a", "2025-03-01T00:00:00", "telemetry", 3.0),
    ("b", "2025-01-02T00:00:00", "telemetry", 10.0),
    ("b", "2025-02-02T00:00:00", "telemetry", 20.0),
    ("b", "2025-03-02T00:00:00", "alarm", 30.0),
]


def test_derives_monetary_field_that_is_not_value():
    # Pre-DEC-614 this raised a polars ColumnNotFoundError on the literal "value".
    d = _frame("reading", _EVENTS)
    assert "value" not in d.columns
    X, pos = _raw_features(d, ["a", "b"], _cfg())
    assert set(pos) == {"a", "b"}
    assert X.shape[0] == 2
    assert np.isfinite(X).all()


def test_comparator_is_name_agnostic():
    # The same data under a different column name must give the same features:
    # the comparator reads the derived dominant numeric field, not a literal name.
    X_read, _ = _raw_features(_frame("reading", _EVENTS), ["a", "b"], _cfg())
    X_val, _ = _raw_features(_frame("value", _EVENTS), ["a", "b"], _cfg())
    assert np.allclose(X_read, X_val)


def test_numeric_free_stream_zeroes_monetary_not_crash():
    # No numeric column at all: monetary aggregates must be zeroed, not crash.
    d = pl.DataFrame(
        {
            "customer_key": ["a", "a", "b"],
            "event_ts": [
                "2025-01-01T00:00:00",
                "2025-02-01T00:00:00",
                "2025-01-02T00:00:00",
            ],
            "event_type": ["t", "t", "t"],
        }
    )
    X, pos = _raw_features(d, ["a", "b"], _cfg())
    assert set(pos) == {"a", "b"}
    assert X.shape[0] == 2
    assert np.isfinite(X).all()
