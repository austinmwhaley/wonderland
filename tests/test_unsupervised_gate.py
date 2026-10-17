"""CPU-only unit tests for the unsupervised gate's pure helpers (no model/GPU)."""

from __future__ import annotations

import numpy as np

from looking_glass.unsupervised_gate import (
    _ari,
    _choose_k,
    _enrichment,
    _eta_squared,
    _top_decile_mask,
)


def _two_blobs(seed=0, per_blob=60):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [30.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(per_blob, 2)) for c in centers])
    truth = np.repeat([0, 1], per_blob)
    return X, truth


def test_ari_permutation_invariant_and_perfect_on_separated_blobs():
    X, truth = _two_blobs()
    k, _score, labels = _choose_k(X, max_k=4, seed=0)
    assert k == 2
    assert _ari(labels, truth) > 0.99
    assert _ari(truth, 1 - truth) == 1.0


def test_ari_handles_disagreement():
    assert _ari([0, 0, 1, 1], [0, 1, 0, 1]) < 0.5


def test_enrichment_finds_planted_seed_cluster():
    rng = np.random.default_rng(1)
    seeds = rng.normal(scale=0.5, size=(40, 2))
    others = rng.normal(scale=0.5, size=(360, 2)) + np.array([40.0, 0.0])
    X = np.vstack([seeds, others])
    seed_mask = np.r_[np.ones(len(seeds), dtype=bool), np.zeros(len(others), dtype=bool)]
    enrichment = _enrichment(X, seed_mask, k=5)
    assert enrichment > 1.5


def test_enrichment_is_near_random_for_shuffled_mask():
    X, _truth = _two_blobs(per_blob=200)
    rng = np.random.default_rng(0)
    seed_mask = np.zeros(len(X), dtype=bool)
    seed_mask[rng.choice(len(X), size=40, replace=False)] = True
    enrichment = _enrichment(X, seed_mask, k=5)
    assert 0.2 < enrichment < 5.0


def test_top_decile_mask_selects_largest_values():
    y = np.arange(100, dtype=float)
    mask = _top_decile_mask(y)
    assert mask.sum() == 10
    assert y[mask].min() == 90.0


def test_eta_squared_separates_value_differentiated_segments():
    labels = np.repeat([0, 1], 50)
    y = np.r_[np.zeros(50), np.ones(50) * 10.0]
    assert _eta_squared(labels, y) > 0.99
