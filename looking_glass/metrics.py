"""Threshold-free and business-facing evaluation metrics.

These complement the threshold-based metrics in :mod:`looking_glass.supervised`.
Stakeholders generally care about ranking quality (AUC / PR-AUC) and lift, not a
single thresholded F1, so these are provided as small dependency-light helpers.

All functions accept plain Python sequences or numpy arrays of model scores
(higher = more likely positive) and binary 0/1 labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_arrays(scores: object, labels: object) -> tuple[np.ndarray, np.ndarray]:
    score_arr = np.asarray(list(scores) if not isinstance(scores, np.ndarray) else scores, dtype=np.float64)
    label_arr = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels, dtype=np.float64)
    if score_arr.shape != label_arr.shape:
        raise ValueError(f"scores and labels must align: {score_arr.shape} vs {label_arr.shape}")
    if score_arr.ndim != 1:
        raise ValueError("scores and labels must be 1-D")
    return score_arr, label_arr


def roc_auc(scores: object, labels: object) -> float:
    """Area under the ROC curve via the rank-based Mann-Whitney U statistic.

    Returns 0.5 when only one class is present (AUC is undefined there).
    Ties in score are handled with average ranks.
    """

    score_arr, label_arr = _as_arrays(scores, labels)
    positives = label_arr == 1.0
    n_pos = float(positives.sum())
    n_neg = float((~positives).sum())
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5

    order = np.argsort(score_arr, kind="mergesort")
    ranked = score_arr[order]
    ranks = np.empty(len(score_arr), dtype=np.float64)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1] == ranked[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = float(ranks[positives].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_auc(scores: object, labels: object) -> float:
    """Average precision (area under the precision-recall curve).

    Computed as the sum of precision at each threshold weighted by the increase
    in recall, the standard ``average_precision_score`` definition.
    """

    score_arr, label_arr = _as_arrays(scores, labels)
    n_pos = float((label_arr == 1.0).sum())
    if n_pos == 0.0:
        return 0.0

    order = np.argsort(-score_arr, kind="mergesort")
    sorted_labels = label_arr[order]
    cum_tp = np.cumsum(sorted_labels == 1.0)
    cum_fp = np.cumsum(sorted_labels == 0.0)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1.0)
    recall = cum_tp / n_pos

    prev_recall = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        ap += float(p) * (float(r) - prev_recall)
        prev_recall = float(r)
    return float(ap)


@dataclass(frozen=True)
class LiftBucket:
    """One decile/bucket of a lift table, ordered from highest score to lowest."""

    bucket: int
    count: int
    positives: int
    response_rate: float
    cumulative_response_rate: float
    lift: float
    cumulative_lift: float


def lift_table(scores: object, labels: object, n_bins: int = 10) -> list[LiftBucket]:
    """Build a decile lift table by sorting scores descending and bucketing.

    ``lift`` is a bucket's response rate divided by the overall base rate;
    ``cumulative_lift`` is the response rate across all buckets up to and
    including this one divided by the base rate. A useful targeting model shows
    cumulative_lift well above 1.0 in the top buckets.
    """

    score_arr, label_arr = _as_arrays(scores, labels)
    n = len(score_arr)
    if n == 0:
        return []
    n_bins = max(1, min(int(n_bins), n))

    order = np.argsort(-score_arr, kind="mergesort")
    sorted_labels = label_arr[order]
    base_rate = float(sorted_labels.mean())

    splits = np.array_split(sorted_labels, n_bins)
    buckets: list[LiftBucket] = []
    seen = 0
    seen_pos = 0
    for idx, chunk in enumerate(splits):
        count = int(chunk.size)
        positives = int((chunk == 1.0).sum())
        seen += count
        seen_pos += positives
        response_rate = positives / max(count, 1)
        cumulative_response_rate = seen_pos / max(seen, 1)
        lift = (response_rate / base_rate) if base_rate > 0.0 else 0.0
        cumulative_lift = (cumulative_response_rate / base_rate) if base_rate > 0.0 else 0.0
        buckets.append(
            LiftBucket(
                bucket=idx,
                count=count,
                positives=positives,
                response_rate=response_rate,
                cumulative_response_rate=cumulative_response_rate,
                lift=lift,
                cumulative_lift=cumulative_lift,
            )
        )
    return buckets


def ranking_metrics(scores: object, labels: object, n_bins: int = 10) -> dict[str, float]:
    """Bundle the threshold-free metrics most useful for model comparison."""

    table = lift_table(scores, labels, n_bins=n_bins)
    top_lift = table[0].cumulative_lift if table else 0.0
    return {
        "roc_auc": roc_auc(scores, labels),
        "pr_auc": pr_auc(scores, labels),
        "top_decile_lift": float(top_lift),
    }
