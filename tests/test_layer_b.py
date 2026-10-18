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
    # DEC-215: reduce ONLY when asked; when no reduction is needed, keep the
    # measure AXIS-ALIGNED (pca=None) so tree/k-means consumers are not rotated.
    m = np.arange(40, dtype=np.float32).reshape(10, 4)
    reduced = LayerBEmbedder(n_components=2).fit(m)
    assert reduced.pca is not None and reduced.pca.n_components_ == 2
    full = LayerBEmbedder(n_components=8).fit(m)
    assert full.pca is None


def test_generic_measure_exposes_action_response(tmp_path):
    """DEC-175: with declared action keys, the generic measure gains an action-conditioned
    RESPONSE block (empirical dose-response per customer) — decision-sufficiency by construction."""
    import json
    from datetime import datetime, timedelta, timezone

    import polars as pl

    from looking_glass.cfm_config import CFMConfig
    from looking_glass.layer_b import generic_measure

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for c, x0 in (("c0", 0), ("c1", 2)):  # optimal freq per customer
        for w in range(30):
            f = (w + int(c[1])) % 4
            rows.append((c, (base + timedelta(days=w * 7.0)).isoformat(), "send", 0.0, f))
            if f == x0:
                rows.append((c, (base + timedelta(days=w * 7.0 + 1)).isoformat(), "buy", 5.0, None))
    df = pl.DataFrame(
        {
            "customer_key": [r[0] for r in rows],
            "event_ts": [r[1] for r in rows],
            "event_type": [r[2] for r in rows],
            "event_attributes": [
                json.dumps(
                    ({"freq": int(r[4])} if r[4] is not None else {"value": r[3]}),
                    separators=(",", ":"),
                )
                for r in rows
            ],
        }
    )
    p = tmp_path / "s.parquet"
    df.write_parquet(p)
    cfg = CFMConfig(db=str(p), as_of="2024-12-31")
    cfg.effect_actions = ("freq",)
    X, names = generic_measure(cfg, ["c0", "c1"])
    assert any(n.startswith("resp_freq_") for n in names)
    fi = [names.index(f"resp_freq_{i}") for i in range(4)]
    R = X[:, fi]
    assert np.argmax(R[0]) == 0 and np.argmax(R[1]) == 2  # recovers the optimum
    # off (no action keys) -> no response block
    cfg2 = CFMConfig(db=str(p), as_of="2024-12-31")
    X2, names2 = generic_measure(cfg2, ["c0", "c1"])
    assert not any(n.startswith("resp_") for n in names2)


def test_generic_measure_excludes_declared_action_fields(tmp_path):
    """DEC-182: declared action fields must be EXCLUDED from the measure, so `E` never carries
    the logged action (which leaked via `discount_pct=0.05*arm` → E->arm acc 0.999)."""
    import json
    from datetime import datetime, timedelta, timezone

    import polars as pl

    from looking_glass.cfm_config import CFMConfig
    from looking_glass.layer_b import generic_measure

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for c in ("c0", "c1"):
        for w in range(10):
            arm = w % 3
            rows.append(
                (
                    c,
                    (base + timedelta(days=w * 7.0)).isoformat(),
                    "send",
                    json.dumps({"arm": arm, "discount_pct": 0.05 * arm, "freq": 1}),
                )
            )
            rows.append(
                (
                    c,
                    (base + timedelta(days=w * 7.0 + 1)).isoformat(),
                    "buy",
                    json.dumps({"value": 5.0}),
                )
            )
    df = pl.DataFrame(
        {
            "customer_key": [r[0] for r in rows],
            "event_ts": [r[1] for r in rows],
            "event_type": [r[2] for r in rows],
            "event_attributes": [r[3] for r in rows],
        }
    )
    p = tmp_path / "s.parquet"
    df.write_parquet(p)
    cfg = CFMConfig(db=str(p), as_of="2025-01-01")
    cfg.effect_actions = ("freq",)
    cfg.action_fields = ("arm", "discount_pct")
    _, names = generic_measure(cfg, ["c0", "c1"])
    assert not any("discount" in n for n in names)
    assert not any(n.endswith("_arm_mean") or "_arm_" in n for n in names)
    # off -> the action fields ARE (leakily) present
    cfg2 = CFMConfig(db=str(p), as_of="2025-01-01")
    _, names2 = generic_measure(cfg2, ["c0", "c1"])
    assert any("discount" in n for n in names2)
