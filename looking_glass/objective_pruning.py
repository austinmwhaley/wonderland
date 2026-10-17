"""Objective-set minimization via leave-one-out ablation (Layer-1 minimality / C6).

Given held-out combined-metric values for the FULL objective set and for each
leave-one-out ablation (``without[obj]`` = the same metric when ``obj`` is
removed), return the smallest subset of objectives whose removal does **not**
reduce the metric beyond the measured noise floor. This operationalizes our
doctrine: *prefer deletion over accretion*; an objective that carries no unique
held-out signal is "dominated" and is pruned.

The tolerance is data-derived, never a magic number:

    tol = k_se * SE(base)          SE = std(base, ddof=1) / sqrt(n_seeds)

``k_se`` is a documented multiplier on the standard error (default 1.0 = one SE).

Deterministic, numpy-only. Real driving is a JSON ablation receipt
(``load_ablation``) produced later by GPU leave-one-out runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

Metric = object  # array-like of held-out metric values across seeds


def _stats(values: Metric) -> tuple[float, float]:
    """Return (mean, standard-error) of a metric sample across seeds."""
    a = np.asarray(values, dtype=float)
    n = a.size
    if n == 0:
        return float("nan"), 0.0
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, se


def _seed_labels(seeds: object) -> object:
    """Normalize seed identifiers to something a receipt/JSON can carry."""
    if isinstance(seeds, (str, bytes)) or np.isscalar(seeds):
        return seeds
    try:
        return [int(s) if isinstance(s, (int, np.integer)) else s for s in seeds]
    except TypeError:
        return seeds


def prune_objectives(
    base_metric: Metric,
    without: dict[str, Metric],
    full_seeds: object,
    *,
    k_se: float = 1.0,
    evaluate: Callable[[tuple[str, ...]], Metric] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    """Split objectives into kept / pruned by leave-one-out ablation.

    Parameters
    ----------
    base_metric:
        Held-out combined metric across seeds for the FULL objective set.
    without:
        ``{objective: metric-values-when-removed}`` (leave-one-out ablations).
    full_seeds:
        Seed identifiers/labels for provenance; recorded in the receipt.
    k_se:
        Noise-floor multiplier on ``SE(base_metric)`` (documented, default 1.0).
    evaluate:
        Optional callback ``evaluate(removed_tuple) -> metric-values`` that runs
        (or looks up) a JOINT ablation of a set of objectives. Greedy uses it to
        verify multi-prune sets; without it greedy falls back to an additive,
        independent-effect approximation. This is the hook for real GPU runs.

    Returns
    -------
    (kept, pruned, receipt)
    """
    base = np.asarray(base_metric, dtype=float)
    base_mean, base_se = _stats(base)
    tol = float(k_se) * base_se

    per: dict[str, dict] = {}
    kept: list[str] = []
    pruned: list[str] = []
    for obj, values in without.items():
        mean, se = _stats(values)
        delta = mean - base_mean  # >=0 means removal does not hurt / helps
        dominated = bool(mean >= base_mean - tol)
        per[obj] = {
            "mean": mean,
            "se": se,
            "delta_vs_base": delta,
            "dominated": dominated,
            "reason": (
                "dominated: removal within the measured noise floor"
                if dominated
                else "load-bearing: removal reduces the metric beyond the noise floor"
            ),
        }
        pruned.append(obj) if dominated else kept.append(obj)

    # Most-dominated first: extraction hurts least (delta closest to 0 / positive).
    order = sorted(without, key=lambda o: per[o]["delta_vs_base"], reverse=True)

    def _joint_mean(objs: tuple[str, ...]) -> float:
        if evaluate is not None:
            return _stats(evaluate(objs))[0]
        # Additive, independent-effect approximation (no joint data available).
        return base_mean + sum(per[o]["delta_vs_base"] for o in objs)

    # Greedy forward step: grow the removed set from the most-dominated outward
    # and keep the largest prefix whose JOINT removal stays within tolerance.
    greedy: list[str] = []
    for k in range(1, len(order) + 1):
        prefix = order[:k]
        if _joint_mean(tuple(prefix)) >= base_mean - tol:
            greedy = prefix
    greedy_reason = (
        "greedy forward drop verified by joint ablation within tolerance"
        if greedy and evaluate is not None
        else "greedy forward drop under additive independent-effect approximation"
        if greedy
        else "no joint prune set stays within tolerance"
    )

    receipt = {
        "k_se": float(k_se),
        "n_seeds": int(base.size),
        "full_seeds": _seed_labels(full_seeds),
        "base": {"mean": base_mean, "se": base_se},
        "noise_floor": tol,
        "per_objective": per,
        "kept": kept,
        "pruned": pruned,
        "greedy_removed": greedy,
        "greedy_reason": greedy_reason,
    }
    return tuple(kept), tuple(pruned), receipt


def load_ablation(receipt_path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Parse a JSON ablation receipt ``{"base": [...], "without": {obj: [...]}}``.

    Returns ``(base, without)`` as numpy arrays — the inputs to
    :func:`prune_objectives`, so the same logic can be driven by real
    leave-one-out runs written to disk.
    """
    data = json.loads(Path(receipt_path).read_text())
    base = np.asarray(data["base"], dtype=float)
    without = {str(obj): np.asarray(vals, dtype=float) for obj, vals in data["without"].items()}
    return base, without
