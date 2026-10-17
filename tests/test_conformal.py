from __future__ import annotations

import numpy as np

from looking_glass.conformal import ConformalClassifier, ConformalRegressor


def _regression_split(rng, n):
    x = rng.normal(size=n)
    signal = np.sin(x) + 0.5 * x
    noise = rng.logistic(scale=0.4, size=n)
    y = signal + noise
    pred = signal
    return pred, y


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _classification_split(rng, n, k=6):
    y = rng.integers(0, k, size=n)
    logits = rng.normal(scale=1.0, size=(n, k)) + 2.0 * np.eye(k)[y]
    return _softmax(logits), y


def test_regression_coverage_matches_nominal():
    rng = np.random.default_rng(0)
    n_calib, n_test = 4000, 40000
    for alpha in (0.1, 0.05):
        pred_c, y_c = _regression_split(np.random.default_rng(11), n_calib)
        pred_t, y_t = _regression_split(rng, n_test)
        cr = ConformalRegressor().fit(pred_c, y_c, alpha)
        cov = cr.coverage(pred_t, y_t)
        assert abs(cov - (1.0 - alpha)) <= 0.03, (alpha, cov)


def test_regression_width_shrinks_with_calibration_size():
    alpha = 0.1
    sizes = [20, 100, 2000]

    def width(n):
        grid = (np.arange(n) + 0.5) / n
        cr = ConformalRegressor().fit(np.zeros(n), grid, alpha)
        lo, hi = cr.interval(np.zeros(1))
        return float(hi[0] - lo[0])

    widths = [width(n) for n in sizes]
    assert widths[0] > widths[1] > widths[2], widths


def test_classification_coverage_matches_nominal():
    rng = np.random.default_rng(3)
    n_calib, n_test = 4000, 40000
    for alpha in (0.1, 0.05):
        proba_c, y_c = _classification_split(np.random.default_rng(12), n_calib)
        proba_t, y_t = _classification_split(rng, n_test)
        cc = ConformalClassifier().fit(proba_c, y_c, alpha)
        sets = cc.predict_set(proba_t)
        cov = cc.coverage(sets, y_t)
        assert abs(cov - (1.0 - alpha)) <= 0.03, (alpha, cov)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    pred_c, y_c = _regression_split(np.random.default_rng(13), 500)
    pred_t, y_t = _regression_split(rng, 500)
    cr = ConformalRegressor().fit(pred_c, y_c, 0.1)
    path = tmp_path / "reg.pkl"
    cr.save(path)
    cr2 = ConformalRegressor.load(path)
    lo1, hi1 = cr.interval(pred_t)
    lo2, hi2 = cr2.interval(pred_t)
    assert np.allclose(lo1, lo2) and np.allclose(hi1, hi2)
    assert cr2.coverage(pred_t, y_t) == cr.coverage(pred_t, y_t)

    proba_c, yc = _classification_split(np.random.default_rng(14), 500)
    proba_t, yt = _classification_split(rng, 500)
    cc = ConformalClassifier().fit(proba_c, yc, 0.1)
    path = tmp_path / "clf.pkl"
    cc.save(path)
    cc2 = ConformalClassifier.load(path)
    assert cc2.predict_set(proba_t) == cc.predict_set(proba_t)
    assert cc2.coverage(cc.predict_set(proba_t), yt) == cc.coverage(cc.predict_set(proba_t), yt)
