"""Frozen-schema serving: new events must encode exactly as training did."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looking_glass.stream_adapter import Schema, transform  # noqa: E402

AS_OF = datetime.datetime(2024, 2, 1).timestamp()


def _frame():
    return pl.DataFrame(
        {
            "customer_key": ["a", "a", "b"],
            "event_ts": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "event_type": ["view", "purchase", "view"],
            "event_attributes": ['{"x": 1}', '{"x": 2}', '{"x": 1}'],
        }
    )


def _schema(**over):
    base = dict(
        et_v={"view": 1, "purchase": 2},
        catspec=[{"field": "x", "mode": "vocab", "map": {"1": 1, "2": 2}, "size": 3}],
        num=[],
        rawspec=[("x", "int")],
        periods=[24.0],
        as_of_ts=AS_OF,
    )
    base.update(over)
    return Schema(**base)


def test_schema_roundtrip():
    s = _schema()
    assert Schema.from_dict(s.to_dict()).to_dict() == s.to_dict()


def test_transform_preserves_customer_order_and_events():
    keys, seqs, rcpt = transform(_frame(), _schema())
    assert keys == ["a", "b"]
    assert rcpt["rows"] == 3
    assert rcpt["entities"] == 2
    # frozen vocabulary is applied, not rediscovered
    assert seqs["a"][0].tolist() == [1, 2]
    assert seqs["b"][0].tolist() == [1]


def test_unseen_event_type_maps_to_unknown():
    df = _frame().with_columns(
        pl.when(pl.col("event_type") == "purchase")
        .then(pl.lit("refund"))
        .otherwise(pl.col("event_type"))
        .alias("event_type")
    )
    keys, seqs, _ = transform(df, _schema())
    # "refund" was never in training -> index 0 (unknown), never a valid event id
    assert seqs["a"][0].tolist() == [1, 0]


def test_unseen_category_maps_to_zero():
    df = _frame().with_columns(pl.lit('{"x": 99}').alias("event_attributes"))
    _, seqs, _ = transform(df, _schema())
    assert seqs["a"][1][:, 0].tolist() == [0, 0]


def test_missing_field_does_not_crash():
    df = _frame().with_columns(pl.lit("{}").alias("event_attributes"))
    _, seqs, _ = transform(df, _schema())
    assert seqs["a"][1][:, 0].tolist() == [0, 0]


def test_requires_canonical_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        transform(pl.DataFrame({"customer_key": ["a"]}), _schema())


def test_requires_positive_as_of():
    with pytest.raises(ValueError, match="positive as_of"):
        transform(_frame(), _schema(as_of_ts=0.0))


def test_future_events_are_excluded():
    df = pl.concat(
        [
            _frame(),
            pl.DataFrame(
                {
                    "customer_key": ["a"],
                    "event_ts": ["2024-06-01"],
                    "event_type": ["purchase"],
                    "event_attributes": ['{"x": 2}'],
                }
            ),
        ]
    )
    _, seqs, rcpt = transform(df, _schema())
    assert rcpt["rows"] == 3  # the June event is after AS_OF and must not appear
    assert seqs["a"][0].tolist() == [1, 2]


def test_empty_stream_returns_empty():
    df = _frame().head(0)
    keys, seqs, rcpt = transform(df, _schema())
    assert keys == [] and seqs == {} and rcpt["rows"] == 0
