"""Gradient-boosted-tree baseline on aggregate features.

The central claim of this project is that a learned sequence backbone beats
hand-engineered aggregates. That claim is only meaningful against a strong,
conventional baseline. This module provides exactly that: an XGBoost model
trained on the generic aggregate features from :mod:`looking_glass.outcomes`,
exposing the same ``fit_predict`` shape as :class:`SupervisedModel` so the two
can be compared head-to-head on identical splits and metrics.

XGBoost is an optional dependency (the ``baseline`` extra). When it is missing
the harness still runs via a dependency-light linear/logistic fallback, so the
comparison is always available — just with a weaker baseline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from .metrics import ranking_metrics

try:
    import xgboost as xgb  # type: ignore

    _HAS_XGB = True
except ImportError:  # pragma: no cover - env dependent
    xgb = None  # type: ignore
    _HAS_XGB = False


def baseline_implementation_name() -> str:
    return "xgboost" if _HAS_XGB else "linear_fallback"


@dataclass(frozen=True)
class BaselineResult:
    """Predictions and metrics from a baseline, comparable to PredictionResults."""

    task: str
    implementation: str
    predictions: dict[str, float]
    metrics: dict[str, float]


def _split(n: int, seed: int, validation_fraction: float) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    val_count = max(1, int(n * validation_fraction))
    return sorted(idx[val_count:]), sorted(idx[:val_count])


def _feature_matrix(rows: list[dict], feature_fields: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(r.get(f, 0.0) or 0.0) for f in feature_fields] for r in rows],
        dtype=np.float64,
    )


def _threshold_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in (i / 100.0 for i in range(10, 91, 2)):
        preds = (probs >= t).astype(np.float64)
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    preds = (probs >= best_t).astype(np.float64)
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": float((preds == labels).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": best_t,
    }


def _regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(pred - target)))
    mse = float(np.mean((pred - target) ** 2))
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2)) or 1e-6
    return {"mae": mae, "mse": mse, "rmse": math.sqrt(mse), "r2": 1.0 - ss_res / ss_tot}


def _xgb_train(x_tr, y_tr, x_all, seed, objective):
    # Native booster API so XGBoost works without the scikit-learn wrapper.
    dtrain = xgb.DMatrix(x_tr, label=y_tr)
    dall = xgb.DMatrix(x_all)
    params = {
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.9,
        "objective": objective,
        "seed": seed,
        "nthread": 0,
    }
    if objective == "binary:logistic":
        params["eval_metric"] = "logloss"
    booster = xgb.train(params, dtrain, num_boost_round=200)
    return booster.predict(dall)


def _fit_classifier(x_tr, y_tr, x_all, seed):
    if _HAS_XGB:
        return _xgb_train(x_tr, y_tr, x_all, seed, "binary:logistic")
    return _logistic_fallback(x_tr, y_tr, x_all)


def _fit_regressor(x_tr, y_tr, x_all, seed):
    if _HAS_XGB:
        return _xgb_train(x_tr, y_tr, x_all, seed, "reg:squarederror")
    return _linear_fallback(x_tr, y_tr, x_all)


def _standardize(x_tr, x_all):
    mean = x_tr.mean(axis=0, keepdims=True)
    std = x_tr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x_tr - mean) / std, (x_all - mean) / std


def _logistic_fallback(x_tr, y_tr, x_all, iters: int = 500, lr: float = 0.1):
    x_tr_s, x_all_s = _standardize(x_tr, x_all)
    x_tr_b = np.hstack([x_tr_s, np.ones((x_tr_s.shape[0], 1))])
    x_all_b = np.hstack([x_all_s, np.ones((x_all_s.shape[0], 1))])
    w = np.zeros(x_tr_b.shape[1])
    for _ in range(iters):
        grad = x_tr_b.T @ (1.0 / (1.0 + np.exp(-x_tr_b @ w)) - y_tr) / len(y_tr)
        w -= lr * grad
    return 1.0 / (1.0 + np.exp(-x_all_b @ w))


def _linear_fallback(x_tr, y_tr, x_all):
    x_tr_s, x_all_s = _standardize(x_tr, x_all)
    x_tr_b = np.hstack([x_tr_s, np.ones((x_tr_s.shape[0], 1))])
    x_all_b = np.hstack([x_all_s, np.ones((x_all_s.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(x_tr_b, y_tr, rcond=None)
    return x_all_b @ coef


class GBTBaseline:
    """Gradient-boosted-tree (or fallback) baseline mirroring SupervisedModel."""

    def __init__(
        self,
        task: str,
        id_field: str,
        target_field: str,
        feature_fields: list[str],
        seed: int = 17,
        validation_fraction: float = 0.2,
    ) -> None:
        task = task.lower().strip()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.id_field = id_field
        self.target_field = target_field
        self.feature_fields = list(feature_fields)
        self.seed = seed
        self.validation_fraction = validation_fraction

    def fit_predict(self, rows: list[dict]) -> BaselineResult:
        if len(rows) < 2:
            raise ValueError("Need at least 2 rows")
        x = _feature_matrix(rows, self.feature_fields)
        y = np.asarray([float(r.get(self.target_field, 0.0) or 0.0) for r in rows], dtype=np.float64)
        train_idx, val_idx = _split(len(rows), self.seed, self.validation_fraction)

        if self.task == "classification":
            pred_all = _fit_classifier(x[train_idx], y[train_idx], x, self.seed)
            val_probs, val_labels = pred_all[val_idx], y[val_idx]
            metrics = _threshold_metrics(val_probs, val_labels)
            metrics.update(ranking_metrics(val_probs.tolist(), val_labels.tolist()))
        else:
            pred_all = _fit_regressor(x[train_idx], y[train_idx], x, self.seed)
            metrics = _regression_metrics(pred_all[val_idx], y[val_idx])

        predictions: dict[str, list[float]] = {}
        for row, value in zip(rows, pred_all):
            predictions.setdefault(str(row[self.id_field]), []).append(float(value))
        return BaselineResult(
            task=self.task,
            implementation=baseline_implementation_name(),
            predictions={k: float(sum(v) / len(v)) for k, v in predictions.items()},
            metrics={k: float(v) for k, v in metrics.items()},
        )


def compare(model_metrics: dict[str, float], baseline_metrics: dict[str, float], keys: list[str]) -> dict[str, dict]:
    """Build a side-by-side comparison table for the headline metrics."""

    table: dict[str, dict] = {}
    for key in keys:
        m = model_metrics.get(key)
        b = baseline_metrics.get(key)
        delta = (float(m) - float(b)) if (m is not None and b is not None) else None
        table[key] = {"model": m, "baseline": b, "delta_model_minus_baseline": delta}
    return table
