"""Sensitivity tests: exactness vs brute force, monotonicity, calibration.
CPU only, instant."""

import itertools

import numpy as np


def test_worst_case_exact_vs_brute_force():
    from white_queen.tribunal.ope.sensitivity import worst_case_mean

    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(2, 9))
        v = rng.normal(50, 20, n)
        w = rng.uniform(0.1, 5.0, n)
        g = float(rng.uniform(1.0, 4.0))
        got = worst_case_mean(v, w, g)
        best = float("inf")
        for bits in itertools.product((0, 1), repeat=n):
            ww = np.where(np.array(bits), w * g, w / g)
            m = float((ww * v).sum() / ww.sum())
            best = min(best, m)
        assert abs(got - best) < 1e-9, (trial, got, best)


def test_frontier_properties():
    from white_queen.tribunal.ope.sensitivity import breakdown_frontier, gamma_star

    rng = np.random.default_rng(1)
    v = rng.normal(80, 15, 300)
    w = rng.uniform(0.5, 2.0, 300)
    assert abs(worst_case_mean(v, w, 1.0) - np.average(v, weights=w)) < 1e-9
    f = breakdown_frontier(v, w, 60.0)
    ms = [m for _, m in f]
    assert all(b >= a - 1e-9 for a, b in zip(ms, ms[1:])) is False  # decreasing
    assert all(b <= a + 1e-9 for a, b in zip(ms, ms[1:]))
    assert min(v) - 1e-9 <= ms[-1] <= v.mean() + 1e-9
    gs, _ = gamma_star(v, w, 60.0)
    assert gs in (None, 1.0) or gs > 1.0
    gs2, _ = gamma_star(v, w, float(np.min(v)) - 1.0)
    assert gs2 == float("inf")  # below every value: never flips at any Gamma


def worst_case_mean(v, w, g):
    from white_queen.tribunal.ope.sensitivity import worst_case_mean as f

    return f(v, w, g)
