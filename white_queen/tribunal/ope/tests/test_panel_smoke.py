"""Smoke test for the autotuned panel path without torch.

Uses a uniform-random mock candidate and a tiny synthetic diet to exercise:
select_temperature, clip-cap resolution, IS/WIS math, blend weighting, and
the receipts (bootstrap CI, behavior stats). Heavy learners (FQE/MIS/etc.)
are NOT invoked here — they need torch + the lab. See test_autotune for
config resolution and test_gate_receipts for the gate.
"""
import numpy as np

from white_queen.tribunal.ope import estimators as E


class UniformCandidate:
    """Mock: uniform over nA, temperature-independent (honest baseline)."""

    def __init__(self, nA=2):
        self.nA = nA

    def act(self, state, eval=True):
        return 0

    def action_probs(self, obs, temperature=1.0):
        o = np.asarray(obs)
        n = len(o) if o.ndim > 1 else 1
        return np.full((n, self.nA), 1.0 / self.nA)


def _diet(n_ep=12, T=8, seed=1):
    rng = np.random.default_rng(seed)
    N = n_ep * T
    obs = rng.normal(size=(N, 4)).astype(np.float32)
    return {
        "obs": obs,
        "obs2": np.roll(obs, -1, axis=0),
        "act": rng.integers(0, 2, N),
        "rew": np.ones(N, dtype=np.float32),
        "done": np.zeros(N, dtype=np.float32),
        "mu": np.full((N, 2), 0.5, dtype=np.float32),
        "episode": np.repeat(np.arange(n_ep), T),
        "t": np.tile(np.arange(T), n_ep),
        "nA": 2,
        "N": N,
    }


def test_episodes_roundtrip():
    d = _diet()
    eps = E.episodes(d)
    assert len(eps) == 12
    assert all(len(e["act"]) == 8 for e in eps)


def test_select_temperature_autotunes():
    d = _diet()
    cand = UniformCandidate()
    # Uniform cand == uniform behavior -> ratios ~1 -> ESS high at any temp.
    sel = E.select_temperature(d, cand, 0.99, meta=None)
    assert sel["temperature"] in E._resolve_meta(d, 0.99, None)["temps"]
    assert 0.0 < sel["ess_frac"] <= 1.0
    # Explicit override wins.
    sel2 = E.select_temperature(d, cand, 0.99, meta={"temps": (1.0,)})
    assert sel2["temperature"] == 1.0


def test_resolve_meta_backfills_legacy_dict():
    d = _diet()
    m = E._resolve_meta(d, 0.99, {"bootstrap_B": 77})
    assert m["bootstrap_B"] == 77
    assert "temps" in m and "clip_quantile" in m


def test_plausible_bounds_admits_early_termination():
    # Regression: CartPole-style all-positive rewards (+1 every step) must NOT
    # collapse the plausibility interval to a single point, which clamped every
    # value estimate to the same number (the 99.3 soft-FQE degeneracy).
    d = _diet()  # rew = +1 everywhere, episodes length 8
    lo, hi = E.plausible_bounds(d, 0.99)
    assert lo == 0.0
    assert hi > 7.0
    assert hi > lo
    # All-negative rewards: symmetric, and again non-degenerate.
    d2 = dict(d, rew=-np.ones(d["N"], dtype=np.float32))
    lo2, hi2 = E.plausible_bounds(d2, 0.99)
    assert hi2 == 0.0 and lo2 < 0.0 and lo2 < hi2
    # Mixed signs keep the full [-G, +G] range.
    rw = np.where(np.arange(d["N"]) % 2, 1.0, -1.0).astype(np.float32)
    lo3, hi3 = E.plausible_bounds(dict(d, rew=rw), 0.99)
    assert lo3 < 0.0 < hi3
