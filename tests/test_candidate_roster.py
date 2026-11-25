"""Tests for the candidate roster (looking_glass/candidate_roster.py) — CPU, synthetic."""

from __future__ import annotations

import numpy as np

from looking_glass.candidate_roster import ScoreCandidate, build_roster


def test_score_candidate_act_and_probs():
    mean = np.zeros(4)
    std = np.ones(4)

    def f(E):  # argmax always at action 2
        S = np.zeros((len(E), 3))
        S[:, 2] = 1.0
        return S

    c = ScoreCandidate(mean, std, 3, f)
    obs = np.column_stack([np.arange(3), np.zeros((3, 4))])
    assert c.act(obs[0]) == 2
    p = c.action_probs(obs)
    assert np.allclose(p.sum(1), 1.0)
    assert (p.argmax(1) == 2).all()
    ps = c.action_probs(obs, temperature=0.5)
    assert np.allclose(ps.sum(1), 1.0)


def test_build_roster_recovers_best_action():
    """When action 1 is globally best, the value/iql/cql/dt candidates should pick it."""
    rng = np.random.default_rng(0)
    n, d = 500, 8
    E = rng.normal(size=(n, d))
    A = rng.integers(0, 3, n)
    y = (A == 1).astype(float)  # action 1 always best
    roster, mean, std = build_roster(E, A, y)
    # the offline-RL families the roster promises; a silently dropped family is a missing
    # candidate, not a skip
    assert {"value", "iql", "cql", "dt"} <= set(roster), sorted(roster)
    obs = np.column_stack([np.arange(n), E])
    for name in ("value", "iql", "cql", "dt"):
        cand = roster[name]
        probs = cand.action_probs(obs)
        assert probs.shape == (n, 3)
        assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1.0)
        # every candidate yields a real action for every row (no empty/sentinel answer)
        acts = probs.argmax(1).astype(int)
        assert acts.shape == (n,)
        assert set(np.unique(acts)) <= {0, 1, 2}
        assert 0 <= cand.act(obs[0]) < 3
        frac = float((acts == 1).mean())
        assert frac > 0.6, (name, frac)
