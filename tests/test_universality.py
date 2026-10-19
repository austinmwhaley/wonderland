"""Tests for the universality gate's pure logic (looking_glass/universality.py).

No encoder/data needed: exercises the oracle-gap normalization and the family
builders on tiny frames. The end-to-end gate is validated on real checkpoints by
the loop; here we lock the metric's arithmetic so it cannot drift.
"""

from __future__ import annotations

import polars as pl
import pytest

from looking_glass.universality import (
    _future_family,
    _history_features,
    _normalized_gap,
    synthetic_machinery_check,
)


def test_normalized_gap_regression():
    # donor = oracle score [E,hist]; e_alone = score(E). gap = what history adds.
    gap, norm = _normalized_gap(0.60, 0.55, "reg")
    assert gap == pytest.approx(0.05)
    assert norm == pytest.approx(0.05 / 0.60)
    # sufficient: E already matches the oracle -> zero gap
    assert _normalized_gap(0.60, 0.60, "reg") == (0.0, 0.0)
    # E cannot beat the oracle for a superset feature model: non-negative, clipped
    assert _normalized_gap(0.60, 0.70, "reg") == (pytest.approx(-0.10), 0.0)


def test_normalized_gap_classification():
    # clf range is above-chance spread (score - 0.5), so gaps are dimensionless
    gap, norm = _normalized_gap(0.90, 0.70, "clf")
    assert gap == pytest.approx(0.20)
    assert norm == pytest.approx(0.20 / 0.40)
    # degenerately low oracle -> no usable range -> no claim of gap
    assert _normalized_gap(0.5000000001, 0.5, "clf") == (pytest.approx(1e-10), 0.0)


def test_normalized_gap_is_bounded_unit_interval():
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, norm = _normalized_gap(1.0, e, "reg")
        assert 0.0 <= norm <= 1.0


def test_history_features_columns():
    past = pl.DataFrame(
        {
            "freq": [1, 2],
            "monetary": [3.0, 4.0],
            "recent30": [0, 1],
            "recent90": [1, 2],
            "past_vmean": [1.5, 2.5],
            "past_vlast": [1.0, 2.0],
        }
    )
    H = _history_features(past)
    assert "recency" in H and H["recency"] is None  # filled by caller
    assert set(H) == {
        "recency",
        "freq",
        "monetary",
        "recent30",
        "recent90",
        "past_vmean",
        "past_vlast",
    }


def test_future_family_is_self_supervised_and_typed():
    j = pl.DataFrame(
        {
            "orders": [0.0, 2.0],
            "ltv": [0.0, 5.0],
            "next_v": [0.0, 9.0],
            "next_delay_days": [30.0, 1.5],
            "repeat_last_type": [0.0, 1.0],
        }
    )
    fam = _future_family(j)
    # every member is a future observable, labelled reg/clf
    assert all(kind in ("reg", "clf") for _, kind in fam.values())
    assert "LTV (future log-margin)" in fam and fam["LTV (future log-margin)"][1] == "reg"
    assert fam["next-event delay (days)"][1] == "reg"
    assert fam["repeat last event type"][1] == "clf"
    assert fam["repeat last event type"][0].tolist() == [0.0, 1.0]
    p = fam["propensity (order in horizon)"][0]
    c = fam["churn (no order in horizon)"][0]
    assert p.tolist() == [0.0, 1.0]
    assert c.tolist() == [1.0, 0.0]
    assert (p + c == 1.0).all()


def test_synthetic_machinery_separates_lossy_and_saturates():
    """Ground-truth check: known sufficient statistic. The gate must (a) score a
    sufficient representation ~1, (b) penalize a representation that dropped the
    magnitude (extensive) axis, and (c) NOT reward extra capacity (saturation)."""
    res = synthetic_machinery_check(n=2000, seeds=(0,))
    suff = res["sufficient [S, logmag]"]["norm_gap"]
    lossy = res["lossy [direction]"]["norm_gap"]
    over = res["over [S, logmag, noise]"]["norm_gap"]
    assert suff < 0.15, f"sufficient rep should leave near-zero gap, got {suff}"
    assert lossy > suff + 0.3, f"dropping magnitude must raise the gap ({lossy} vs {suff})"
    assert over <= suff + 0.1, f"extra capacity must not lower U ({over} vs {suff})"
