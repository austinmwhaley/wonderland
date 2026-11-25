#!/usr/bin/env python3
"""Unit tests for `scripts/build_port_artifact.py` — pure logic on synthetic input.

The B3.2/B3.4 receipts certify the DEC-604 delivery port (bank + measure PCA-128; the
bank width is a property of its builder — 179 before DEC-673, 200 after)
PCA-128). These tests pin the properties the reproducibility fix depends on, without reading
the real artifacts: the width is bank_dim + PCA_DIM, every PCA row stays attached to its own customer_key,
a missing source raises instead of fabricating a port, and two calls are byte-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_port_artifact as P  # noqa: E402

BANK_DIM = 179  # the fixture's bank width; the real bank moves with its builder
N = 300
PARTS = {"mass": 96, "shape": 64}  # 160 measure dims: PCA-128 is fittable


def _synthetic(*, reverse_measure: bool = False):
    rng = np.random.default_rng(0)
    keys = [f"c{i:03d}" for i in range(N)]
    bank = {k: rng.normal(size=BANK_DIM) for k in keys}
    measure = {}
    for k in keys:
        full = rng.normal(size=sum(PARTS.values()))
        measure[k] = {"mass": full[: PARTS["mass"]], "shape": full[PARTS["mass"] :]}
    if reverse_measure:  # insertion order must not decide which PCA row a customer gets
        measure = dict(reversed(list(measure.items())))
    split = {k: ("A" if i % 4 else "B") for i, k in enumerate(keys)}
    return keys, bank, measure, split


def _expected_pca(keys, measure, split):
    """The composition oracle: same frozen construction, keys in bank order."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    full = np.vstack([np.concatenate([measure[k][p] for p in PARTS]) for k in keys])
    fit = np.array([split[k] == "A" for k in keys])
    scaler = StandardScaler().fit(full[fit])
    pca = PCA(n_components=P.PCA_DIM, random_state=P.RANDOM_STATE).fit(scaler.transform(full[fit]))
    return pca.transform(scaler.transform(full))


def test_composition_width_follows_the_bank():
    keys, bank, measure, split = _synthetic()
    port = P.compose_port(bank, measure, split, part_order=list(PARTS))
    assert P.PCA_DIM == 128
    assert list(port) == keys
    assert {v.shape for v in port.values()} == {(BANK_DIM + P.PCA_DIM,)}
    assert P.validate_port(port, bank_keys=keys) == N


def test_rows_align_by_customer_key_not_dict_order():
    keys, bank, measure, split = _synthetic(reverse_measure=True)
    port = P.compose_port(bank, measure, split, part_order=list(PARTS))
    expected = _expected_pca(keys, measure, split)
    for i, k in enumerate(keys):
        assert np.array_equal(port[k][:BANK_DIM], bank[k]), f"bank half misaligned for {k}"
        assert np.array_equal(port[k][BANK_DIM:], expected[i]), f"PCA half misaligned for {k}"


def test_output_is_deterministic_across_two_calls():
    keys, bank, measure, split = _synthetic()
    first = P.compose_port(bank, measure, split, part_order=list(PARTS))
    second = P.compose_port(bank, measure, split, part_order=list(PARTS))
    assert list(first) == list(second)
    for k in keys:
        assert np.array_equal(first[k], second[k]), f"row {k} differs between builds"


def test_wrong_width_is_refused():
    keys, bank, measure, split = _synthetic()
    port = P.compose_port(bank, measure, split, part_order=list(PARTS))
    narrow = {k: v[:256] for k, v in port.items()}
    # the builder always names the width; a narrower port must be refused against it
    with pytest.raises(ValueError, match=str(BANK_DIM + P.PCA_DIM)):
        P.validate_port(narrow, dim=BANK_DIM + P.PCA_DIM)
    # and a ragged port is refused even without a named width
    with pytest.raises(ValueError, match="ragged"):
        P.validate_port({**port, "ragged": port[keys[0]][:10]})
    with pytest.raises(ValueError, match="align"):
        P.validate_port(port, bank_keys=keys[:-1] + ["absent"])


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        P._require_products(tmp_path / "absent", "bank")
    with pytest.raises(FileNotFoundError):
        P.resolve_port(tmp_path / "no_bank", tmp_path / "no_cert")
