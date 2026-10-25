"""Frozen foundation artifact tests: metadata, fail-safe, state-view contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looking_glass.foundation import ArtifactMeta, FrozenFoundation  # noqa: E402


def test_meta_roundtrip():
    m = ArtifactMeta(data="/tmp/x", seed=7, axes=["measure"], as_of_ts=123.0)
    m2 = ArtifactMeta.from_dict(m.to_dict())
    assert m2.data == m.data
    assert m2.seed == 7
    assert m2.axes == ["measure"]
    assert m2.as_of_ts == 123.0
    assert m2.version == m.version


def test_state_fails_safe_without_as_of():
    """A missing as_of must reject, never emit a sanitized NaN state."""
    art = FrozenFoundation(None, {}, ArtifactMeta(as_of_ts=0.0))
    with pytest.raises(ValueError, match="as_of"):
        art.state({}, ["a"])


def test_meta_defaults_are_explicit():
    m = ArtifactMeta()
    assert m.version
    assert m.layout == {}
    assert m.measure is None
