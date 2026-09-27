"""Supervised CLV plugins — thin wrapper over the head template.

The four CLV heads and their gate contract are unchanged; the machinery lives
in `plugins/head_template.py` and the target contract in `plugins/targets.py`.
Names are re-exported here so existing imports keep working. New supervised
plugins are added as Targets (not as code).
"""

from __future__ import annotations

from .head_template import (
    HEADS,
    HeadTemplate,
    _metrics,
    _ridge_alpha,
    _split,
    head_baseline,
    head_baseline_binary,
    head_logistic,
    head_point,
    head_quantile,
    head_two_part,
    run_target,
)
from .targets import clv_target

__all__ = [
    "HEADS",
    "HeadTemplate",
    "_metrics",
    "_ridge_alpha",
    "_split",
    "head_baseline",
    "head_baseline_binary",
    "head_logistic",
    "head_point",
    "head_quantile",
    "head_two_part",
    "run",
    "run_target",
]


def run(window_days: int, seed: int = 0):
    """Legacy entry point (CLV target) used by plugins/run.py and the gate."""
    return run_target(clv_target(window_days), seed=seed)
