"""Receipts + gate + protocol tests (numpy only, no torch)."""

import numpy as np

from white_queen.tribunal.ope import gate as G
from white_queen.tribunal.ope.protocols import check_candidate
from white_queen.tribunal.ope.receipts import (
    behavior_stats,
    bootstrap_ci,
    ess,
    ess_frac,
)


def _diet(n_ep=30, T=20, seed=0):
    rng = np.random.default_rng(seed)
    N = n_ep * T
    return {
        "obs": rng.normal(size=(N, 4)).astype(np.float32),
        "rew": np.ones(N, dtype=np.float32),
        "episode": np.repeat(np.arange(n_ep), T),
    }


def test_ess_flat_is_n():
    assert abs(ess(np.ones(10)) - 10.0) < 1e-9
    assert abs(ess_frac(np.ones(10)) - 1.0) < 1e-9
    # Degenerate: one episode carries everything -> ESS ~ 1.
    w = np.array([100.0] + [0.01] * 9)
    assert ess(w) < 2.0


def test_bootstrap_ci_autotunes_and_covers():
    rng = np.random.default_rng(0)
    vals = rng.normal(5.0, 1.0, 200)
    lo, hi = bootstrap_ci(vals)  # auto B/alpha from n=200
    assert lo < 5.0 < hi
    lo2, hi2 = bootstrap_ci(vals, B=50, alpha=0.2)
    assert (hi2 - lo2) < (hi - lo)  # narrower alpha -> narrower interval


def test_behavior_stats_discounted_units():
    d = _diet(n_ep=4, T=3)
    # rewards all 1, gamma 1.0 -> return 3 per episode.
    s = behavior_stats(d, gamma=1.0)
    assert abs(s["mean"] - 3.0) < 1e-9
    assert s["n_episodes"] == 4


def test_gate_vetoes_low_ess_and_deploys_healthy():
    good = {
        "blended": 80.0,
        "dr": 80.0,
        "dr_vals": list(np.full(50, 80.0)),
        "is_vals": list(np.full(50, 80.0)),
        "wis": 75.0,
        "is": 80.0,
        "wdr": 80.0,
        "magic": 80.0,
        "magic_w": [],
        "fqe_dm": 79.0,
        "efqe": {"mean": 79.0, "disagreement": 0.5},
        "lstdq": {"dm": 79.0, "cond": 10.0},
        "fve_dm": 79.0,
        "mb": {"mb": 79.0, "se": 1.0, "sims": 100},
        "gdice_mis": 79.0,
        "anchor": 60.0,
        "slope_pick": "dr",
        "slope_val": 80.0,
        "below_anchor": [],
        "mis": 79.0,
        "mis_info": {"mis_ess_frac": 0.5},
        "lambda_dr": 0.8,
        "ess_frac": 0.40,
        "temperature": 1.0,
        "rho_cap": 5.0,
        "truth": 80.0,
    }
    rows = G.adjudicate(
        {"a": good}, behavior_mean=60.0, behavior_std=5.0, gate_cfg=None, meta=None, n_episodes=50
    )
    assert rows["a"]["deploy"] is True
    assert rows["a"]["vetoes"] == []
    assert "gate" in rows["a"] and "bootstrap" in rows["a"]

    bad = dict(good, ess_frac=0.001)
    rows2 = G.adjudicate(
        {"a": bad}, behavior_mean=60.0, behavior_std=5.0, gate_cfg=None, meta=None, n_episodes=50
    )
    assert rows2["a"]["deploy"] is False
    assert any("ESS" in v for v in rows2["a"]["vetoes"])


def test_check_candidate_rejects_broken():
    class NoProbs:
        def act(self, s, eval=True):
            return 0

    try:
        check_candidate(NoProbs())
    except TypeError as e:
        assert "action_probs" in str(e)
    else:
        raise AssertionError("should have raised")

    class Good:
        def act(self, s, eval=True):
            return 0

        def action_probs(self, obs, temperature=1.0):
            import numpy as _np

            o = _np.asarray(obs)
            return _np.full((len(o), 2), 0.5)

    check_candidate(Good())  # no raise
