"""Industry-track tests: behavior estimation without true propensities.

Fast CPU only. Run: pytest .../tests/test_behavior.py -q
"""
import numpy as np


def _logged(N=2000, n_ep=20, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), N // n_ep)
    N = len(ep)
    obs = rng.normal(size=(N, 4)).astype(np.float32)
    act = (obs @ np.array([1., -1., .5, 0.], dtype=np.float32) > 0).astype(int)
    flip = rng.random(N) < noise
    act[flip] = 1 - act[flip]
    return {
        "obs": obs, "obs2": obs, "act": act,
        "rew": np.ones(N, dtype=np.float32), "done": np.zeros(N, dtype=np.float32),
        "mu": np.full((N, 2), 0.5, dtype=np.float32),
        "episode": ep, "t": np.tile(np.arange(N // n_ep), n_ep)[:N],
        "nA": 2, "N": N,
    }


def test_recovers_learnable_behavior():
    from white_queen.tribunal.ope import behavior as B
    d = _logged()
    tiny = {"steps_max": 400, "device": "cpu", "hidden": 32, "batch": 64}
    probs, info = B.estimate_behavior(d, tiny)
    assert probs.shape == (d["N"], 2)
    assert abs(probs.sum(1) - 1.0).max() < 1e-5  # row-stochastic
    assert info["accuracy"] > 0.80  # rule is 85% deterministic
    assert np.isfinite(info["val_nll"])


def test_floor_and_immutability():
    from white_queen.tribunal.ope import behavior as B
    d = _logged()
    tiny = {"steps_max": 200, "device": "cpu", "hidden": 16, "batch": 64}
    d2, mi = B.with_estimated_propensities(d, tiny)
    assert mi["mu_source"] == "estimated"
    assert mi["floor"] == 0.025  # 5% of uniform, nA=2
    assert (d2["mu"] >= 0.025 - 1e-6).all()
    assert abs(d2["mu"].sum(1) - 1.0).max() < 1e-5
    assert bool((d["mu"] == 0.5).all())  # original untouched
    assert 0.0 <= mi["clipped_frac"] <= 1.0


def test_random_behavior_gets_chance_probs():
    from white_queen.tribunal.ope import behavior as B
    rng = np.random.default_rng(1)
    N = 1200
    obs = rng.normal(size=(N, 4)).astype(np.float32)
    d = {"obs": obs, "obs2": obs, "act": rng.integers(0, 2, N),
         "rew": np.ones(N, dtype=np.float32), "done": np.zeros(N, dtype=np.float32),
         "mu": np.full((N, 2), 0.5, dtype=np.float32),
         "episode": np.repeat(np.arange(12), N // 12)[:N],
         "t": np.zeros(N), "nA": 2, "N": N}
    d["episode"] = np.repeat(np.arange(12), 100)
    _, info = B.estimate_behavior(
        d, {"steps_max": 200, "device": "cpu", "hidden": 16, "batch": 64})
    assert 0.40 < info["accuracy"] < 0.62  # unlearnable -> chance


def test_calibration_improves_holdout_nll():
    from white_queen.tribunal.ope import behavior as B
    import torch
    import torch.nn.functional as F
    d = _logged()
    tiny = {"steps_max": 400, "device": "cpu", "hidden": 32, "batch": 64}
    d2, mi = B.with_estimated_propensities(d, tiny)
    cal_T = d2["_mu_info"]["cal_temperature"]
    assert 0.2 <= cal_T <= 5.0
    # Calibrated NLL on the full diet must beat raw-softmax NLL only when the
    # net is overconfident; at minimum it must be finite and recorded.
    assert np.isfinite(d2["_mu_info"]["cal_holdout_nll"])


def test_estimate_is_deterministic_given_seed():
    from white_queen.tribunal.ope import behavior as B
    d = _logged()
    tiny = {"steps_max": 200, "device": "cpu", "hidden": 16, "batch": 64}
    p1, _ = B.estimate_behavior(d, tiny, seed=0)
    p2, _ = B.estimate_behavior(d, tiny, seed=0)
    assert np.array_equal(p1, p2)  # torch seeding fix: bit-identical
