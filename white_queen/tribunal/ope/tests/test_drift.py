"""Drift diagnostics: non-stationary pools must be flagged, not silently
pooled. CPU only."""
import numpy as np


def _diet(obs, seed=0):
    N = len(obs)
    rng = np.random.default_rng(seed)
    return {"obs": obs.astype(np.float32), "rew": rng.normal(size=N).astype(np.float32),
            "episode": np.repeat(np.arange(max(N // 20, 1)), 20)[:N],
            "act": np.zeros(N, dtype=int), "mu_take": np.full(N, 0.5, np.float32),
            "nA": 2, "N": N}


def test_drift_detected():
    from white_queen.tribunal.ope.drift import drift_report
    rng = np.random.default_rng(0)
    N = 1000
    obs = np.concatenate([rng.normal(0, 1, (N // 2, 4)),
                          rng.normal(2.5, 1, (N // 2, 4))])
    r = drift_report(_diet(obs))
    assert r["drift_flag"] is True
    assert r["obs_shift_std"] > 1.0
    assert "non-stationary" in r["note"]


def test_stationary_not_flagged():
    from white_queen.tribunal.ope.drift import drift_report
    rng = np.random.default_rng(1)
    obs = rng.normal(0, 1, (1200, 4))
    r = drift_report(_diet(obs))
    assert r["drift_flag"] is False
    assert r["obs_shift_std"] < 1.0
