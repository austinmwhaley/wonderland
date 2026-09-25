"""Validation helpers for embedding maps and prediction reports."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from looking_glass.supervised_types import ValidationReport


def validate_embeddings(
    vectors: dict[str, list[float]],
    min_vectors: int = 1,
    min_norm_std: float = 0.0,
    min_nonzero_fraction: float = 0.0,
    max_mean_abs_cosine: float = 1.0,
    cosine_sample_size: int = 512,
    seed: int = 17,
) -> dict[str, float | int | bool]:
    """Check vector-map quality for dimensional or temporal embeddings.

    Validation combines structural checks (count/dimension/finite values) with
    geometric checks (norm spread, non-zero rate, and pairwise cosine collapse).
    """

    count = len(vectors)
    dims = {len(v) for v in vectors.values()}
    vector_dim = next(iter(dims)) if len(dims) == 1 else -1
    finite = True
    mean_norm = 0.0
    norm_std = 0.0
    nonzero_fraction = 0.0
    mean_abs_cosine = 0.0

    if count > 0 and vector_dim > 0:
        matrix = torch.tensor(list(vectors.values()), dtype=torch.float32)
        finite = bool(torch.isfinite(matrix).all().item())
        if not finite:
            matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        norms = torch.linalg.vector_norm(matrix, dim=1)
        mean_norm = float(norms.mean().item())
        norm_std = float(norms.std(unbiased=False).item())
        nonzero_fraction = float((norms > 1e-8).float().mean().item())

        if matrix.size(0) > 1:
            sample_count = min(int(cosine_sample_size), int(matrix.size(0)))
            generator = torch.Generator(device=matrix.device)
            generator.manual_seed(seed)
            sample_idx = torch.randperm(matrix.size(0), generator=generator, device=matrix.device)[
                :sample_count
            ]
            sample = F.normalize(matrix[sample_idx], dim=1, eps=1e-12)
            cosine = sample @ sample.T
            off_diag_mask = ~torch.eye(sample_count, dtype=torch.bool, device=matrix.device)
            off_diag = cosine[off_diag_mask]
            if off_diag.numel() > 0:
                mean_abs_cosine = float(off_diag.abs().mean().item())
    else:
        for vector in vectors.values():
            for value in vector:
                if not math.isfinite(float(value)):
                    finite = False
                    break
            if not finite:
                break

    norm_std_ok = norm_std >= float(min_norm_std)
    nonzero_ok = nonzero_fraction >= float(min_nonzero_fraction)
    cosine_ok = mean_abs_cosine <= float(max_mean_abs_cosine)

    return {
        "ok": (
            count >= min_vectors
            and len(dims) == 1
            and finite
            and norm_std_ok
            and nonzero_ok
            and cosine_ok
        ),
        "vector_count": count,
        "vector_dim": vector_dim,
        "finite": finite,
        "mean_norm": mean_norm,
        "norm_std": norm_std,
        "nonzero_fraction": nonzero_fraction,
        "mean_abs_cosine": mean_abs_cosine,
        "norm_std_ok": norm_std_ok,
        "nonzero_ok": nonzero_ok,
        "cosine_ok": cosine_ok,
    }


def validate_prediction_report(report: ValidationReport) -> dict[str, bool]:
    """Validate that report row counts are non-zero and metric values are finite."""

    has_rows = report.train_rows > 0 and report.validation_rows > 0
    metrics_finite = all(math.isfinite(float(v)) for v in report.metrics.values())
    return {
        "ok": has_rows and metrics_finite,
        "has_rows": has_rows,
        "metrics_finite": metrics_finite,
    }


def validate_classification_success(
    report: ValidationReport,
    min_f1: float = 0.05,
    min_recall: float = 0.05,
    min_precision: float = 0.0,
    min_accuracy: float = 0.0,
) -> dict[str, bool | float]:
    """Evaluate classification report against configurable minimum thresholds."""

    accuracy = float(report.metrics.get("accuracy", 0.0))
    precision = float(report.metrics.get("precision", 0.0))
    f1 = float(report.metrics.get("f1", 0.0))
    recall = float(report.metrics.get("recall", 0.0))
    accuracy_ok = accuracy >= min_accuracy
    precision_ok = precision >= min_precision
    f1_ok = f1 >= min_f1
    recall_ok = recall >= min_recall
    return {
        "ok": accuracy_ok and precision_ok and f1_ok and recall_ok,
        "accuracy_ok": accuracy_ok,
        "precision_ok": precision_ok,
        "f1_ok": f1_ok,
        "recall_ok": recall_ok,
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "recall": recall,
    }


def validate_regression_success(
    report: ValidationReport,
    min_r2: float = 0.0,
    max_rmse: float | None = None,
) -> dict[str, bool | float]:
    """Evaluate regression report against configurable R^2 and optional RMSE bounds."""

    r2 = float(report.metrics.get("r2", float("-inf")))
    rmse = float(report.metrics.get("rmse", float("inf")))
    r2_ok = r2 >= min_r2
    rmse_ok = True if max_rmse is None else rmse <= max_rmse
    return {
        "ok": r2_ok and rmse_ok,
        "r2_ok": r2_ok,
        "rmse_ok": rmse_ok,
        "r2": r2,
        "rmse": rmse,
    }
