"""CPU-only tests for the plugin-gate bootstrap CI and verdict logic (no model/GPU)."""

from __future__ import annotations

import numpy as np

from looking_glass.supervised_gate import _bootstrap_skill
from looking_glass.unsupervised_gate import _verdict


def test_bootstrap_ci_passes_clear_signal():
    rng = np.random.default_rng(0)
    n = 400
    y = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
    pred = np.where(y == 1, rng.uniform(0.7, 0.95, n), rng.uniform(0.05, 0.3, n))
    skill, lo, hi, verdict = _bootstrap_skill(pred, y, "clf", "auc", 0.5, n_boot=500, seed=0)
    assert skill > 0.9
    assert lo > 0.5
    assert lo <= skill <= hi
    assert verdict == "PASS"


def test_bootstrap_ci_fails_on_pure_noise():
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, n)
    pred = rng.normal(size=n)
    _skill, lo, _hi, verdict = _bootstrap_skill(pred, y, "clf", "auc", 0.5, n_boot=500, seed=0)
    assert lo <= 0.5
    assert verdict == "FAIL"


def test_bootstrap_ci_regression_signal_and_noise():
    rng = np.random.default_rng(2)
    n = 500
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    skill, lo, _hi, verdict = _bootstrap_skill(2.0 * x, y, "reg", "r2", 0.0, n_boot=500, seed=0)
    assert skill > 0.9 and lo > 0.0 and verdict == "PASS"

    y0 = rng.normal(size=n)
    pred0 = rng.normal(size=n)
    _skill0, lo0, _hi0, verdict0 = _bootstrap_skill(pred0, y0, "reg", "r2", 0.0, n_boot=500, seed=0)
    assert lo0 <= 0.0 and verdict0 == "FAIL"


def test_bootstrap_ci_nominal_coverage():
    rng = np.random.default_rng(3)
    n, reps, p = 300, 150, 0.75
    covered = 0
    for _ in range(reps):
        y = rng.integers(0, 2, n)
        correct = rng.random(n) < p
        pred = np.where(correct, y, 1 - y)
        _skill, lo, hi, _verdict = _bootstrap_skill(pred, y, "clf", "acc", 0.5, n_boot=200, seed=0)
        covered += int(lo <= p <= hi)
    coverage = covered / reps
    assert 0.85 <= coverage <= 1.0, coverage


def _metrics(**overrides):
    metrics = {"silhouette": 0.2, "anova_p": 0.01, "lookalike_enrichment": 1.5}
    metrics.update(overrides)
    return metrics


def test_unsupervised_verdict_pass_and_fail_conditions():
    assert _verdict(_metrics()) == "PASS"
    assert _verdict(_metrics(silhouette=None)) == "FAIL"
    assert _verdict(_metrics(silhouette=0.0)) == "FAIL"
    assert _verdict(_metrics(anova_p=None)) == "FAIL"
    assert _verdict(_metrics(anova_p=0.05)) == "FAIL"
    assert _verdict(_metrics(lookalike_enrichment=None)) == "FAIL"
    assert _verdict(_metrics(lookalike_enrichment=1.0)) == "FAIL"
