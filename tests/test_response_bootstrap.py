"""RL evaluator guards: impossible bootstrap intervals and control independence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from measure_response import ci_or_fail  # noqa: E402


def test_valid_interval_is_reported():
    d = np.linspace(-1.0, 3.0, 400)
    out = ci_or_fail(d)
    assert out["lo"] < out["mean"] < out["hi"]


def test_zero_width_interval_with_nonzero_mean_is_rejected():
    """A degenerate resample must never be published as evidence."""
    with pytest.raises(RuntimeError, match="degenerate bootstrap"):
        ci_or_fail(np.full(50, 2.0))


def test_zero_width_interval_at_zero_is_allowed():
    out = ci_or_fail(np.zeros(50))
    assert out == {"mean": 0.0, "lo": 0.0, "hi": 0.0}


def test_non_finite_interval_is_rejected():
    with pytest.raises(RuntimeError, match="non-finite"):
        ci_or_fail(np.array([np.nan, 0.0, 1.0]))
