"""C8: minimality certification / distillation of the foundation embedding ``E``.

Find the *smallest* linear projection of a frozen embedding ``E`` that retains its
predictive sufficiency about held-out targets ``Y`` — so ``E`` can be shrunk (faster
downstream, smaller artifact) without losing capability.

Method
------
1. Standardize ``E`` and ``Y``.
2. Fit a PLS regression of ``Y`` on ``E``: the supervised directions where ``E``'s
   predictive signal lives (assumed low-rank).
3. Score candidate projections ``k = 1..K`` with a K-fold cross-validated ridge
   *probe* R^2 on the first ``k`` PLS scores (no leakage: PLS is refit per fold).
4. Choose the smallest ``k`` whose CV R^2 is within **one standard error** of the
   full-``E`` CV R^2 (Breiman's 1-SE rule). The tolerance is thus derived from the
   held-out fold variance — no magic number.

Dependency-light: numpy + scikit-learn only (no torch). Deterministic given ``seed``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingDistiller:
    """Minimality certification for a frozen embedding ``E``.

    ``fit(E, Y)`` learns PLS directions of ``E`` for targets ``Y`` and selects the
    smallest ``k`` that retains (within one standard error) the full-``E`` predictive
    sufficiency. ``compress`` applies the certified k-dim projection; ``sufficiency``
    reports the retained signal ratio in ``[0, 1]``.
    """

    def __init__(
        self,
        seed: int = 0,
        max_k: int | None = None,
        n_folds: int = 5,
        alpha: float = 1.0,
    ):
        self.seed = int(seed)
        self.max_k = None if max_k is None else int(max_k)
        self.n_folds = int(n_folds)
        self.alpha = float(alpha)
        self.k: int | None = None
        self.W_: np.ndarray | None = None
        self.scaler_ = None
        self.y_scaler_ = None
        self.pls_ = None
        self.max_k_: int | None = None
        self.r2_full_: float | None = None
        self.se_full_: float | None = None
        self.tolerance_: float | None = None

    @staticmethod
    def _as2d(X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        return X

    def _cv_r2(self, X: np.ndarray, Y: np.ndarray, k: int | None) -> tuple[float, float]:
        """K-fold CV ridge-probe R^2 on ``X`` (already reduced) predicting ``Y``.

        When ``k`` is not None, a fresh PLS with ``k`` components is fit inside each
        training fold and used to project train/test — so the projection never sees
        the held-out fold. Returns ``(mean, standard_error)`` across folds."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        scores = []
        for tr, te in kf.split(X):
            if k is None:
                Xtr, Xte = X[tr], X[te]
            else:
                pls = PLSRegression(n_components=k, scale=False).fit(X[tr], Y[tr])
                Xtr, Xte = pls.transform(X[tr]), pls.transform(X[te])
            probe = Ridge(alpha=self.alpha).fit(Xtr, Y[tr])
            scores.append(r2_score(Y[te], probe.predict(Xte), multioutput="variance_weighted"))
        scores = np.asarray(scores, dtype=np.float64)
        se = float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
        return float(scores.mean()), se

    def fit(self, E, Y) -> "EmbeddingDistiller":
        """Learn the certified minimal projection of ``E`` for held-out targets ``Y``."""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import StandardScaler

        E = self._as2d(E)
        Y = self._as2d(Y)
        n, D = E.shape
        if n < 2:
            raise ValueError("need at least 2 samples to fit EmbeddingDistiller")

        self.scaler_ = StandardScaler().fit(E)
        Es = self.scaler_.transform(E)
        self.y_scaler_ = StandardScaler().fit(Y)
        Ys = self.y_scaler_.transform(Y)

        # PLS components are deflation-ordered, so the first k of a K-component fit
        # equal the k-component fit. Cap K by the smallest training-fold size.
        max_test = (n + self.n_folds - 1) // self.n_folds
        kmax = min(D, n - max_test, n - 1)
        if self.max_k is not None:
            kmax = min(kmax, self.max_k)
        kmax = max(1, int(kmax))
        self.max_k_ = kmax
        self.pls_ = PLSRegression(n_components=kmax, scale=False).fit(Es, Ys)

        r2_full, se_full = self._cv_r2(Es, Ys, k=None)
        self.r2_full_ = float(r2_full)
        self.se_full_ = float(se_full)
        self.tolerance_ = float(se_full)

        best_k = kmax
        for k in range(1, kmax + 1):
            r2_k, _ = self._cv_r2(Es, Ys, k=k)
            if r2_k >= r2_full - se_full:
                best_k = k
                break

        self.k = int(best_k)
        self.W_ = np.asarray(self.pls_.x_rotations_[:, : self.k], dtype=np.float64)
        return self

    def compress(self, E) -> np.ndarray:
        """Apply the certified k-dim projection: ``E`` (n x D) -> (n x k)."""
        if self.pls_ is None or self.W_ is None:
            raise RuntimeError("EmbeddingDistiller not fit")
        Es = self.scaler_.transform(self._as2d(E))
        return (Es @ self.W_).astype(np.float32)

    def sufficiency(
        self,
        E,
        Y,
        k: int | None = None,
        W: np.ndarray | None = None,
    ) -> float:
        """Retained predictive sufficiency of a projection of ``E``, in ``[0, 1]``.

        The metric is the cross-validated ridge-probe R^2 of the projected features
        divided by the full-``E`` R^2 (scale-free). ``W`` (D x k) defaults to the
        fitted certified directions; pass an alternative (e.g. random) projection to
        compare. ``k`` defaults to the certified ``self.k``."""
        if self.scaler_ is None:
            raise RuntimeError("EmbeddingDistiller not fit")
        E = self._as2d(E)
        Y = self._as2d(Y)
        Es = self.scaler_.transform(E)
        Ys = Y
        if self.y_scaler_ is not None and Y.shape[1] == self.y_scaler_.n_features_in_:
            Ys = self.y_scaler_.transform(Y)

        kk = self.k if k is None else int(k)
        if kk >= Es.shape[1]:
            return 1.0
        if W is None:
            W = self.W_[:, :kk]
        W = np.asarray(W, dtype=np.float64)
        projected = Es @ W

        r2_full, _ = self._cv_r2(Es, Ys, k=None)
        if r2_full <= 1e-12:
            return 0.0
        r2_k, _ = self._cv_r2(projected, Ys, k=None)
        return float(np.clip(r2_k / r2_full, 0.0, 1.0))

    def save(self, path) -> None:
        import pickle

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "seed": self.seed,
                    "max_k": self.max_k,
                    "n_folds": self.n_folds,
                    "alpha": self.alpha,
                    "k": self.k,
                    "W_": self.W_,
                    "scaler_": self.scaler_,
                    "y_scaler_": self.y_scaler_,
                    "pls_": self.pls_,
                    "max_k_": self.max_k_,
                    "r2_full_": self.r2_full_,
                    "se_full_": self.se_full_,
                    "tolerance_": self.tolerance_,
                },
                f,
            )

    @classmethod
    def load(cls, path) -> "EmbeddingDistiller":
        import pickle

        with open(path, "rb") as f:
            b = pickle.load(f)
        d = cls(
            seed=b.get("seed", 0),
            max_k=b.get("max_k"),
            n_folds=b.get("n_folds", 5),
            alpha=b.get("alpha", 1.0),
        )
        d.k = b["k"]
        d.W_ = b["W_"]
        d.scaler_ = b["scaler_"]
        d.y_scaler_ = b["y_scaler_"]
        d.pls_ = b.get("pls_")
        d.max_k_ = b.get("max_k_")
        d.r2_full_ = b.get("r2_full_")
        d.se_full_ = b.get("se_full_")
        d.tolerance_ = b.get("tolerance_")
        return d
