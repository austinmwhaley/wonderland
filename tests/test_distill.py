"""Tests for embedding minimality certification (looking_glass/distill.py)."""

from __future__ import annotations

import numpy as np

from looking_glass.distill import EmbeddingDistiller


def _synthetic(n=1500, D=40, r=3, M=4, signal=1.0, noise=0.5, y_noise=0.3, seed=0):
    """E (n x D) whose predictive signal for Y lives in a rank-r subspace, plus noise.

    Y is generated from the same latent Z that drives the rank-r component of E, so a
    projection of E that recovers span(Z) is sufficient, and a random one is not."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, r))
    B = rng.normal(size=(D, r)) / np.sqrt(r)  # rank-r signal directions in E
    E = signal * (Z @ B.T) + noise * rng.normal(size=(n, D))
    A = rng.normal(size=(r, M))
    Y = Z @ A + y_noise * rng.normal(size=(n, M))
    return E, Y, r


def test_picks_rank_and_retains_sufficiency():
    E, Y, r = _synthetic()
    dist = EmbeddingDistiller(seed=0).fit(E, Y)

    assert abs(dist.k - r) <= 1, f"picked k={dist.k}, expected ~{r}"
    assert dist.sufficiency(E, Y) >= 0.95

    reduced = dist.compress(E)
    assert reduced.shape == (E.shape[0], dist.k)
    assert reduced.dtype == np.float32


def test_random_projection_is_much_worse():
    E, Y, r = _synthetic(seed=1)
    dist = EmbeddingDistiller(seed=0).fit(E, Y)

    rng = np.random.default_rng(123)
    W_rand = np.linalg.qr(rng.normal(size=(E.shape[1], r)))[0]

    fitted = dist.sufficiency(E, Y)
    random = dist.sufficiency(E, Y, k=r, W=W_rand)

    assert random < fitted - 0.2
    assert random < 0.8


def test_deterministic():
    E, Y, _ = _synthetic(seed=2)
    a = EmbeddingDistiller(seed=7).fit(E, Y)
    b = EmbeddingDistiller(seed=7).fit(E, Y)
    assert a.k == b.k
    assert np.allclose(a.compress(E), b.compress(E))
    assert a.sufficiency(E, Y) == b.sufficiency(E, Y)


def test_save_load_roundtrip(tmp_path):
    E, Y, _ = _synthetic(seed=3)
    dist = EmbeddingDistiller(seed=0).fit(E, Y)
    p = tmp_path / "distill.pkl"
    dist.save(p)
    loaded = EmbeddingDistiller.load(p)

    assert loaded.k == dist.k
    assert np.allclose(loaded.compress(E), dist.compress(E))
    assert loaded.sufficiency(E, Y) == dist.sufficiency(E, Y)
