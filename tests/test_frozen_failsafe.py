"""Robustness: a non-finite frozen artifact must fail loudly, never be sanitized."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class _Tiny:
    """Minimal stand-in so `frozen.load` can run without the real encoder."""

    def eval(self):
        return self


def _write_cache(tmp_path, e, blk):
    np.savez(
        str(tmp_path / "cache.npz"),
        E=e,
        BLK=blk,
        Xtab=np.zeros((len(e), 1)),
        hold=np.array(["a"], dtype=object),
        seg=np.array(["s"], dtype=object),
        best=np.array([1.0]),
    )
    torch.save(
        {"model": _Tiny(), "vs": {}, "data": "x", "seed": 0, "axes": []}, str(tmp_path / "cache.pt")
    )


def test_finite_cache_loads(tmp_path, monkeypatch):
    import frozen

    monkeypatch.setattr(frozen.AE, "DEV", "cpu")
    _write_cache(tmp_path, np.zeros((1, 2), np.float32), np.zeros((1, 3), np.float32))
    out = frozen.load(tmp_path / "cache")
    assert out[0].shape == (1, 2)


def test_non_finite_state_is_rejected(tmp_path, monkeypatch):
    import frozen

    monkeypatch.setattr(frozen.AE, "DEV", "cpu")
    bad = np.array([[np.nan, 0.0]], np.float32)
    _write_cache(tmp_path, bad, np.zeros((1, 3), np.float32))
    with pytest.raises(FloatingPointError, match="non-finite"):
        frozen.load(tmp_path / "cache")
