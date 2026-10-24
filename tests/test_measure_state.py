"""Lawful measure-state tests: exact additivity, bounded vocabulary, finiteness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looking_glass.measure_state import MeasureBuilder  # noqa: E402


def _toy():
    """Two small customers plus a vocabulary wide enough to exercise bucketing."""
    vs = {"gnum": 1, "et": 3, "gcat": [4, 64]}

    def entity(et, cat, raw):
        et = np.asarray(et, dtype=np.int64)
        cat = np.asarray(cat, dtype=np.int64)
        return (
            et,
            cat,
            np.zeros((len(et), 1), np.float32),
            np.zeros((len(et), 1), np.float32),
            np.zeros(len(et), np.float32),
            np.zeros(len(et), np.float64),
            np.asarray(raw, dtype=np.float32).reshape(-1, 1),
        )

    seqs = {
        "a": entity([0, 1, 1], [[1, 5], [2, 6], [1, 7]], [[0.0], [2.0], [3.0]]),
        "b": entity([1, 2], [[1, 5], [1, 8]], [[1.0], [0.0]]),
    }
    merged = tuple(np.concatenate([seqs["a"][j], seqs["b"][j]], axis=0) for j in range(7))
    seqs["__m__"] = merged
    return seqs, vs


def test_measure_dimension_and_finiteness():
    seqs, vs = _toy()
    b = MeasureBuilder(seqs, ["a", "b"], vs)
    m = b.transform(seqs, ["a", "b"])
    assert m.shape == (2, b.dim)
    assert np.isfinite(m).all()


def test_measure_additivity_is_exact():
    """Raw additive blocks must satisfy measure(a + b) == measure(a) + measure(b)."""
    seqs, vs = _toy()
    b = MeasureBuilder(seqs, ["a", "b"], vs)
    a = b.transform(seqs, ["a"])[0]
    c = b.transform(seqs, ["b"])[0]
    m = b.transform(seqs, ["__m__"])[0]
    # Event-type counts and the total count are exact extensive quantities.
    assert np.allclose(m[: vs["et"]], a[: vs["et"]] + c[: vs["et"]])
    assert np.allclose(m[vs["et"]], a[vs["et"]] + c[vs["et"]])
    # Numeric count/sum blocks are additive too (raw accumulation before normalization).
    n_cat = sum(len(k) + 3 for k in b.keep)
    off = vs["et"] + 1 + n_cat
    assert np.allclose(m[off], a[off] + c[off])  # nonzero count
    assert np.allclose(m[off + 1], a[off + 1] + c[off + 1])  # raw sum


def test_vocabulary_is_bounded():
    """A high-cardinality field must not keep every observed value."""
    seqs, vs = _toy()
    b = MeasureBuilder(seqs, ["a", "b"], vs)
    # second field has 64 possible values; the budget must cap it well below that
    assert len(b.keep[1]) < 64


def test_empty_history_is_rejected():
    seqs, vs = _toy()
    empty = tuple(
        np.zeros((0,) + seqs["a"][j].shape[1:], dtype=seqs["a"][j].dtype) for j in range(7)
    )
    seqs["z"] = empty
    b = MeasureBuilder(seqs, ["a"], vs)
    with pytest.raises(ValueError):
        b.transform(seqs, ["z"])
