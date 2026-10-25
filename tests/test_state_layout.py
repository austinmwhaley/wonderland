"""The `full` state view has one permanent, documented composition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looking_glass import state_layout as SL  # noqa: E402


class _SSM:
    e = 2


class _Model:
    d = 8
    ssm = [_SSM()]
    n_read = 4
    head_hist = None
    use_suffix = False


def _vs():
    return {
        "et": 3,
        "gcat": [4, 5],
        "gnum": 1,
        "unc_dim": 3,
        "horizons": [1.0, 2.0],
        "fut_dim": 2,
        "past_dim": 5,
        "id_dims": [4],
        "rel_idx": [0],
    }


def test_full_order_is_fixed_and_documented():
    assert SL.FULL_ORDER == ("shape", "learned_state", "forecast", "uncertainty", "heads")


def test_compose_full_equals_declared_dimensions():
    model, vs = _Model(), _vs()
    spec = SL.describe(vs, model)
    shape = np.ones((2, model.d), np.float32)
    blk = np.arange(2 * spec["blk_dim"], dtype=np.float32).reshape(2, spec["blk_dim"])
    full = SL.compose_full(shape, blk, vs, model)
    assert full.shape == (2, spec["full_dim"])
    assert np.allclose(full[:, : model.d], shape)


def test_every_blk_column_is_used_exactly_once():
    model, vs = _Model(), _vs()
    spec = SL.describe(vs, model)
    used = []
    for view in spec["views"].values():
        used.extend(view["blocks"])
    assert sorted(used) == sorted(b["name"] for b in spec["blocks"])
    assert len(used) == len(set(used))
