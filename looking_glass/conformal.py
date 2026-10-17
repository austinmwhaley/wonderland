"""Split conformal prediction for calibrated, finite-sample uncertainty.

Wraps the point outputs of a foundation-model head with marginal coverage
guarantees under exchangeability — no distributional assumptions. A held-out
CALIBRATION split yields conformity scores; the finite-sample-corrected
``(1 - alpha)`` quantile of those scores defines the prediction region.

Dependency-light (numpy only) and deterministic.
"""

from __future__ import annotations

import math
import pickle
from os import PathLike

import numpy as np

__all__ = ["ConformalRegressor", "ConformalClassifier"]


def _conformal_index(n: int, alpha: float) -> int:
    """Finite-sample-corrected order-statistic index ``ceil((n+1)(1-alpha))``."""
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return max(k, 1)


def _quantile_1d(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float).ravel())
    n = scores.size
    if n == 0:
        raise ValueError("cannot calibrate conformal quantile on an empty calibration set")
    k = _conformal_index(n, alpha)
    if k > n:
        return math.inf
    return float(scores[k - 1])


def _quantile_columns(scores: np.ndarray, alpha: float) -> np.ndarray:
    scores = np.sort(np.asarray(scores, dtype=float), axis=0)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("cannot calibrate conformal quantile on an empty calibration set")
    k = _conformal_index(n, alpha)
    if k > n:
        return np.full(scores.shape[1], math.inf)
    return scores[k - 1].astype(float)


class _Persistable:
    def save(self, path: str | PathLike) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | PathLike):
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"file {path!r} holds {type(obj).__name__}, not {cls.__name__}")
        return obj


class ConformalRegressor(_Persistable):
    """Split-conformal intervals over point predictions.

    ``fit(pred, y, alpha)`` on a held-out CALIBRATION split learns the score
    quantile ``q``; ``interval(pred)`` returns ``pred - q, pred + q``.
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components
        self.q_: float | np.ndarray | None = None
        self.alpha_: float | None = None
        self.n_calib_: int | None = None

    def fit(self, pred, y, alpha: float) -> "ConformalRegressor":
        pred = np.asarray(pred, dtype=float)
        y = np.asarray(y, dtype=float)
        if pred.shape != y.shape:
            raise ValueError(f"pred shape {pred.shape} != y shape {y.shape}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        scores = np.abs(y - pred)
        if scores.ndim <= 1:
            if self.n_components not in (None, 1):
                raise ValueError(f"n_components={self.n_components} but predictions are 1-D")
            self.q_ = _quantile_1d(scores, alpha)
        else:
            if self.n_components is not None and scores.shape[1] != self.n_components:
                raise ValueError(
                    f"n_components={self.n_components} != prediction width {scores.shape[1]}"
                )
            self.n_components = scores.shape[1]
            self.q_ = _quantile_columns(scores, alpha)
        self.alpha_ = float(alpha)
        self.n_calib_ = int(scores.shape[0])
        return self

    def _check_fitted(self) -> None:
        if self.q_ is None:
            raise RuntimeError("ConformalRegressor is not fitted")

    def interval(self, pred):
        self._check_fitted()
        pred = np.asarray(pred, dtype=float)
        return pred - self.q_, pred + self.q_

    def coverage(self, pred, y) -> float:
        lo, hi = self.interval(pred)
        y = np.asarray(y, dtype=float)
        return float(np.mean((y >= lo) & (y <= hi)))


class ConformalClassifier(_Persistable):
    """Split-conformal prediction SETS with marginal coverage ``1 - alpha``.

    ``fit(proba, y, alpha)`` uses the score ``1 - p_true``; ``predict_set(proba)``
    returns, per row, the set of class indices whose score clears the calibrated
    threshold — i.e. classes with ``proba >= 1 - q``.
    """

    def __init__(self) -> None:
        self.q_: float | None = None
        self.alpha_: float | None = None
        self.n_calib_: int | None = None
        self.n_classes_: int | None = None

    def fit(self, proba, y, alpha: float) -> "ConformalClassifier":
        proba = np.asarray(proba, dtype=float)
        y = np.asarray(y)
        if proba.ndim != 2:
            raise ValueError(f"proba must be 2-D (n, n_classes), got shape {proba.shape}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        n, k = proba.shape
        if y.shape[0] != n:
            raise ValueError(f"y length {y.shape[0]} != proba rows {n}")
        y_idx = y.astype(int)
        if y_idx.min() < 0 or y_idx.max() >= k:
            raise ValueError("y must be column indices in [0, n_classes)")
        scores = 1.0 - proba[np.arange(n), y_idx]
        self.q_ = _quantile_1d(scores, alpha)
        self.alpha_ = float(alpha)
        self.n_calib_ = int(n)
        self.n_classes_ = int(k)
        return self

    def _check_fitted(self) -> None:
        if self.q_ is None:
            raise RuntimeError("ConformalClassifier is not fitted")

    def predict_set(self, proba) -> list[set[int]]:
        self._check_fitted()
        proba = np.asarray(proba, dtype=float)
        if proba.ndim != 2:
            raise ValueError(f"proba must be 2-D (n, n_classes), got shape {proba.shape}")
        scores = 1.0 - proba
        keep = scores <= self.q_
        return [set(np.flatnonzero(row).tolist()) for row in keep]

    def coverage(self, sets, y) -> float:
        y = np.asarray(y).astype(int)
        if len(sets) != y.shape[0]:
            raise ValueError(f"len(sets)={len(sets)} != len(y)={y.shape[0]}")
        hits = [int(label) in s for label, s in zip(y, sets)]
        return float(np.mean(hits))
