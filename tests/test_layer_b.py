"""Tests for the realizable Layer-B embedding (looking_glass/layer_b.py)."""

from __future__ import annotations

import numpy as np

from looking_glass.layer_b import LayerBEmbedder


def test_layerb_fit_compress_shape():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(200, 30)).astype(np.float32)
    emb = LayerBEmbedder(n_components=8).fit(m)
    c = emb.compress(m)
    assert c.shape == (200, 8)
    assert c.dtype == np.float32


def test_layerb_compress_is_deterministic_and_persists(tmp_path):
    rng = np.random.default_rng(1)
    m = rng.normal(size=(150, 12)).astype(np.float32)
    emb = LayerBEmbedder(n_components=5).fit(m)
    c1 = emb.compress(m)
    p = tmp_path / "emb.pkl"
    emb.save(p)
    emb2 = LayerBEmbedder.load(p)
    c2 = emb2.compress(m)
    assert np.allclose(c1, c2)


def test_layerb_respects_rank_cap():
    m = np.zeros((10, 4), dtype=np.float32)  # rank 0
    emb = LayerBEmbedder(n_components=8).fit(m)
    assert emb.pca.n_components_ <= 4
