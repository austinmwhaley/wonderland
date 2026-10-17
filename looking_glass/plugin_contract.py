"""One generic Layer-C plugin contract for training and inference.

Every plugin consumes the frozen embedding ``E`` and exposes the same surface:
``fit(E, targets)`` learns from labelled data, ``infer(E)`` applies the learned
artifact, and ``save``/``load`` persist it with pickle. The three families are
thin adapters over scikit-learn estimators:

* ``SupervisedPlugin``   -- regressor or classifier on ``E``.
* ``UnsupervisedPlugin`` -- KMeans (with an optional scaler) on ``E``.
* ``DecisionPlugin``     -- value model ``V(E, a)``; inference is ``argmax_a``.

``fit_and_freeze`` fits, persists, and returns a frozen copy. A frozen plugin
refuses further ``fit`` calls, so applying it to new customers/times can never
retrain it. Dependency-light: numpy + scikit-learn only (no torch).
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor

__all__ = [
    "DecisionPlugin",
    "Plugin",
    "PluginRegistry",
    "SupervisedPlugin",
    "UnsupervisedPlugin",
    "fit_and_freeze",
]

KINDS = ("supervised", "unsupervised", "decision")


def _as_matrix(E) -> np.ndarray:
    X = np.asarray(E, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"E must be 2-D, got shape {X.shape}")
    return X


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (values - values.max(axis=1, keepdims=True)) / max(float(temperature), 1e-12)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Plugin(ABC):
    """Base contract shared by every Layer-C plugin family."""

    version = "1.0.0"

    def __init__(
        self,
        name: str,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.name = str(name)
        self.kind = kind
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.frozen = False

    @abstractmethod
    def fit(self, E, targets) -> "Plugin":
        """Learn from embedding ``E`` and family-specific ``targets``."""

    @abstractmethod
    def infer(self, E):
        """Apply the fitted plugin to embedding ``E``."""

    def freeze(self) -> "Plugin":
        """Make the fitted plugin immutable: subsequent ``fit`` calls raise."""
        self.frozen = True
        self.metadata["frozen"] = True
        return self

    def _check_mutable(self) -> None:
        if getattr(self, "frozen", False):
            raise RuntimeError(f"plugin {self.name!r} is frozen; inference must not retrain")

    def save(self, path: str | PathLike) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | PathLike):
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"file {path!r} holds {type(obj).__name__}, not {cls.__name__}")
        return obj


class SupervisedPlugin(Plugin):
    """Wrap a scikit-learn regressor or classifier over the embedding ``E``."""

    def __init__(
        self,
        estimator,
        target: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        label = target or type(estimator).__name__
        super().__init__(name or f"supervised_{label}", "supervised", metadata)
        self.estimator = estimator
        self.target = target
        self.estimator_ = None
        self.is_classifier_ = bool(is_classifier(estimator))
        self.is_regressor_ = bool(is_regressor(estimator))
        if not (self.is_classifier_ or self.is_regressor_):
            raise TypeError(
                f"{type(estimator).__name__} is neither a scikit-learn classifier nor a regressor"
            )

    def _check_fitted(self) -> None:
        if getattr(self, "estimator_", None) is None:
            raise RuntimeError(f"plugin {self.name!r} is not fitted")

    def fit(self, E, targets) -> "SupervisedPlugin":
        self._check_mutable()
        X = _as_matrix(E)
        if self.is_classifier_:
            y = np.asarray(targets)
        else:
            y = np.asarray(targets, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"E has {X.shape[0]} rows but targets has {y.shape[0]}")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.target = self.target or "target"
        self.metadata.update(
            {
                "target": self.target,
                "estimator": type(self.estimator).__name__,
                "n_features": int(X.shape[1]),
            }
        )
        return self

    def proba(self, E) -> np.ndarray | None:
        """Class probabilities for a classifier; ``None`` for a regressor."""
        self._check_fitted()
        if not self.is_classifier_:
            return None
        X = _as_matrix(E)
        if hasattr(self.estimator_, "predict_proba"):
            return np.asarray(self.estimator_.predict_proba(X))
        if hasattr(self.estimator_, "decision_function"):
            d = np.asarray(self.estimator_.decision_function(X), dtype=float)
            if d.ndim == 1:
                d = np.column_stack([-d, d])
            return _softmax(d)
        raise AttributeError(f"{type(self.estimator_).__name__} has no probability API")

    def predict_proba(self, E) -> np.ndarray | None:
        return self.proba(E)

    def infer(self, E, return_proba: bool = False):
        self._check_fitted()
        pred = self.estimator_.predict(_as_matrix(E))
        if return_proba:
            return pred, self.proba(E)
        return pred


class UnsupervisedPlugin(Plugin):
    """Wrap KMeans (plus an optional scaler) over the embedding ``E``."""

    def __init__(
        self,
        n_clusters: int = 2,
        scaler=None,
        kmeans: KMeans | None = None,
        name: str | None = None,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name or f"unsupervised_kmeans_{int(n_clusters)}", "unsupervised", metadata)
        self.n_clusters = int(n_clusters)
        self.scaler = scaler
        self.kmeans = kmeans
        self.seed = int(seed)
        self.scaler_ = None
        self.kmeans_ = None

    def _check_fitted(self) -> None:
        if getattr(self, "kmeans_", None) is None:
            raise RuntimeError(f"plugin {self.name!r} is not fitted")

    def _scaled(self, E) -> np.ndarray:
        X = _as_matrix(E)
        return self.scaler_.transform(X) if self.scaler_ is not None else X

    def fit(self, E, targets=None) -> "UnsupervisedPlugin":
        self._check_mutable()
        X = _as_matrix(E)
        self.scaler_ = clone(self.scaler) if self.scaler is not None else None
        if self.scaler_ is not None:
            X = self.scaler_.fit_transform(X)
        self.kmeans_ = (
            clone(self.kmeans)
            if self.kmeans is not None
            else KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed)
        )
        self.kmeans_.fit(X)
        self.metadata.update(
            {
                "n_clusters": int(self.kmeans_.n_clusters),
                "scaler": type(self.scaler).__name__ if self.scaler is not None else None,
            }
        )
        return self

    def infer(self, E) -> np.ndarray:
        self._check_fitted()
        return self.kmeans_.predict(self._scaled(E))

    def transform(self, E) -> np.ndarray:
        """Distances from each row of ``E`` to every cluster centre."""
        self._check_fitted()
        return self.kmeans_.transform(self._scaled(E))


class DecisionPlugin(Plugin):
    """Wrap a value model ``V(E, a)`` and act by ``argmax_a V(E, a)``."""

    def __init__(
        self,
        actions=None,
        estimator=None,
        name: str | None = None,
        seed: int = 0,
        temperature: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name or "decision_value", "decision", metadata)
        self.actions = None if actions is None else np.asarray(list(actions))
        self.estimator = estimator
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.actions_ = None
        self.estimator_ = None

    def _check_fitted(self) -> None:
        if getattr(self, "estimator_", None) is None:
            raise RuntimeError(f"plugin {self.name!r} is not fitted")

    @staticmethod
    def _design(E: np.ndarray, A) -> np.ndarray:
        a = np.asarray(A, dtype=float).reshape(-1, 1)
        if E.shape[0] != a.shape[0]:
            raise ValueError(f"E has {E.shape[0]} rows but actions has {a.shape[0]}")
        return np.column_stack([E, a])

    def fit(self, E, targets) -> "DecisionPlugin":
        self._check_mutable()
        try:
            A, y = targets
        except (TypeError, ValueError) as exc:
            raise ValueError("decision targets must be a pair (A, y)") from exc
        X = _as_matrix(E)
        a = np.asarray(A).ravel()
        yy = np.asarray(y, dtype=float).ravel()
        if not (X.shape[0] == a.shape[0] == yy.shape[0]):
            raise ValueError("E, A and y must have the same number of rows")
        if self.actions is None:
            self.actions_ = np.unique(a)
        else:
            self.actions_ = np.asarray(self.actions)
            unknown = np.setdiff1d(a, self.actions_)
            if unknown.size:
                raise ValueError(f"targets contain undeclared actions {unknown.tolist()}")
        try:
            self.actions_.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError("actions must be numeric") from exc
        est = (
            clone(self.estimator)
            if self.estimator is not None
            else HistGradientBoostingRegressor(random_state=self.seed)
        )
        est.fit(self._design(X, a), yy)
        self.estimator_ = est
        self.metadata.update(
            {
                "actions": self.actions_.tolist(),
                "value_model": type(est).__name__,
                "n_features": int(X.shape[1]),
            }
        )
        return self

    def values(self, E) -> np.ndarray:
        """Value of every declared action for every row of ``E``."""
        self._check_fitted()
        X = _as_matrix(E)
        columns = [
            self.estimator_.predict(self._design(X, np.full(X.shape[0], float(a))))
            for a in self.actions_
        ]
        return np.column_stack(columns)

    def infer(self, E) -> np.ndarray:
        self._check_fitted()
        return self.actions_[np.argmax(self.values(E), axis=1)]

    def action_probs(self, E) -> np.ndarray:
        """Softmax over action values (a deterministic ranking surrogate)."""
        return _softmax(self.values(E), self.temperature)


class PluginRegistry:
    """A name-keyed collection with one pickle file per plugin on disk."""

    def __init__(self, plugins=None) -> None:
        self._plugins: dict[str, Plugin] = {}
        for plugin in plugins or ():
            self.register(plugin)

    def register(self, plugin: Plugin) -> Plugin:
        if not isinstance(plugin, Plugin):
            raise TypeError(f"expected a Plugin, got {type(plugin).__name__}")
        existing = self._plugins.get(plugin.name)
        if existing is not None and existing is not plugin:
            raise ValueError(f"plugin name {plugin.name!r} is already registered")
        self._plugins[plugin.name] = plugin
        return plugin

    def get(self, name: str) -> Plugin:
        try:
            return self._plugins[name]
        except KeyError:
            raise KeyError(f"no plugin named {name!r}; available: {self.list()}") from None

    def list(self) -> list[str]:
        return list(self._plugins)

    def names(self) -> list[str]:
        return self.list()

    def plugins(self) -> list[Plugin]:
        return list(self._plugins.values())

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)

    def save_all(self, directory: str | PathLike) -> list[Path]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, plugin in self._plugins.items():
            path = d / f"{name}.pkl"
            plugin.save(path)
            paths.append(path)
        return paths

    def load_all(self, directory: str | PathLike) -> "PluginRegistry":
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"{directory!r} is not a directory")
        for path in sorted(d.glob("*.pkl")):
            self.register(Plugin.load(path))
        return self

    @classmethod
    def from_dir(cls, directory: str | PathLike) -> "PluginRegistry":
        return cls().load_all(directory)


def fit_and_freeze(plugin: Plugin, E, targets, out_dir: str | PathLike) -> Plugin:
    """Fit ``plugin``, persist it under ``out_dir``, and return a frozen copy."""
    if not isinstance(plugin, Plugin):
        raise TypeError(f"expected a Plugin, got {type(plugin).__name__}")
    plugin.fit(E, targets)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{plugin.name}.pkl"
    plugin.freeze()
    plugin.save(path)
    return type(plugin).load(path)
