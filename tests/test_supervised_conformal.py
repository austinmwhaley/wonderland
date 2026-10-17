from __future__ import annotations

import numpy as np

from looking_glass.supervised_gate import _conformal_split, _stable_bucket


def _groups_with_bucket(bucket, n):
    out = []
    i = 0
    while len(out) < n:
        key = f"customer_{i}"
        if _stable_bucket(key) == bucket:
            out.append(key)
        i += 1
    return out


def test_conformal_split_coverage_nominal():
    rng = np.random.default_rng(0)
    n = 40000
    x = rng.normal(size=n)
    pred = np.sin(x) + 0.5 * x
    y = pred + rng.logistic(scale=0.4, size=n)
    groups = [f"customer_{i}" for i in range(n)]
    coverage, width = _conformal_split(pred, y, groups, alpha=0.1)
    assert abs(coverage - 0.90) <= 0.05, coverage
    assert width > 0


def test_conformal_split_width_shrinks_with_n():
    def width(n):
        pred = np.zeros(n + 1)
        y = np.concatenate([(np.arange(n) + 0.5) / n, [0.0]])
        groups = _groups_with_bucket(0, n) + _groups_with_bucket(1, 1)
        return _conformal_split(pred, y, groups, alpha=0.1)[1]

    widths = [width(n) for n in (20, 100, 2000)]
    assert widths[0] > widths[1] > widths[2], widths
