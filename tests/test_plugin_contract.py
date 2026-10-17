"""CPU synthetic tests for the generic Layer-C plugin contract."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import adjusted_rand_score, roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler

from looking_glass.plugin_contract import (
    DecisionPlugin,
    PluginRegistry,
    SupervisedPlugin,
    UnsupervisedPlugin,
    fit_and_freeze,
)


def _regression_data(seed=0, n=600, d=6):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, d))
    y = 2.0 * E[:, 0] - 1.5 * E[:, 1] + 0.5 * E[:, 2]
    y = y + rng.normal(scale=0.05, size=n)
    return E[:450], y[:450], E[450:], y[450:]


def _classification_data(seed=1, n=600, d=6):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, d))
    score = 1.5 * E[:, 0] + E[:, 1] - 0.5 * E[:, 2]
    y = (score > 0).astype(int)
    return E[:450], y[:450], E[450:], y[450:]


def test_supervised_regression_beats_baseline_and_roundtrips(tmp_path):
    Etr, ytr, Ete, yte = _regression_data()
    plugin = SupervisedPlugin(
        HistGradientBoostingRegressor(max_iter=200, random_state=0),
        target="value",
        name="regressor",
    )
    plugin.fit(Etr, ytr)
    pred = plugin.infer(Ete)
    assert r2_score(yte, pred) > 0.8
    assert r2_score(yte, np.full_like(yte, ytr.mean())) < 0.05

    path = tmp_path / "regressor.pkl"
    plugin.save(path)
    loaded = SupervisedPlugin.load(path)
    assert np.array_equal(loaded.infer(Ete), pred)
    assert loaded.target == "value"


def test_supervised_classifier_beats_baseline_and_roundtrips(tmp_path):
    Etr, ytr, Ete, yte = _classification_data()
    plugin = SupervisedPlugin(
        HistGradientBoostingClassifier(max_iter=200, random_state=0),
        target="purchase",
        name="classifier",
    )
    plugin.fit(Etr, ytr)
    pred = plugin.infer(Ete)
    proba = plugin.proba(Ete)
    assert proba.shape == (len(Ete), 2)
    assert roc_auc_score(yte, proba[:, 1]) > 0.9
    assert roc_auc_score(yte, np.full(len(yte), ytr.mean())) == pytest.approx(0.5)
    pred2, proba2 = plugin.infer(Ete, return_proba=True)
    assert np.array_equal(pred2, pred)
    assert np.array_equal(proba2, proba)

    path = tmp_path / "classifier.pkl"
    plugin.save(path)
    loaded = SupervisedPlugin.load(path)
    assert np.array_equal(loaded.infer(Ete), pred)
    assert np.array_equal(loaded.predict_proba(Ete), proba)


def test_unsupervised_separates_blobs_and_roundtrips(tmp_path):
    E, truth = make_blobs(
        n_samples=300,
        centers=[[-5.0, -5.0], [5.0, 5.0]],
        cluster_std=0.5,
        random_state=0,
    )
    plugin = UnsupervisedPlugin(
        n_clusters=2,
        scaler=StandardScaler(),
        name="segments",
        seed=0,
    )
    plugin.fit(E)
    labels = plugin.infer(E)
    distances = plugin.transform(E)
    assert adjusted_rand_score(truth, labels) > 0.99
    assert distances.shape == (len(E), 2)
    assert np.all(distances >= 0.0)

    path = tmp_path / "segments.pkl"
    plugin.save(path)
    loaded = UnsupervisedPlugin.load(path)
    assert np.array_equal(loaded.infer(E), labels)
    assert np.array_equal(loaded.transform(E), distances)


def test_decision_picks_best_action_and_roundtrips(tmp_path):
    rng = np.random.default_rng(2)
    n = 900
    E = rng.normal(size=(n, 4))
    A = rng.integers(0, 3, size=n)
    y = 3.0 * (A == 1) + 0.8 * E[:, 0] - 0.5 * E[:, 1]
    y = y + rng.normal(scale=0.05, size=n)
    plugin = DecisionPlugin(actions=[0, 1, 2], name="nba", seed=0)
    plugin.fit(E, (A, y))

    Ete = rng.normal(size=(200, 4))
    actions = plugin.infer(Ete)
    values = plugin.values(Ete)
    probs = plugin.action_probs(Ete)
    assert np.mean(actions == 1) > 0.95
    assert np.array_equal(plugin.actions_[np.argmax(values, axis=1)], actions)
    assert probs.shape == (len(Ete), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)

    path = tmp_path / "nba.pkl"
    plugin.save(path)
    loaded = DecisionPlugin.load(path)
    assert np.array_equal(loaded.infer(Ete), actions)
    assert np.array_equal(loaded.action_probs(Ete), probs)


def test_registry_save_all_load_all_roundtrip(tmp_path):
    Etr, ytr, Ete, _ = _regression_data()
    classifier = SupervisedPlugin(
        HistGradientBoostingClassifier(max_iter=50, random_state=0),
        name="clf",
    )
    classifier.fit(Etr, (ytr > ytr.mean()).astype(int))
    cluster = UnsupervisedPlugin(n_clusters=2, name="clu", seed=0)
    cluster.fit(Etr)
    decision = DecisionPlugin(actions=[0, 1], name="dec", seed=0)
    decision.fit(Etr, ((Etr[:, 0] > 0).astype(int), ytr))

    registry = PluginRegistry()
    for plugin in (classifier, cluster, decision):
        registry.register(plugin)
    assert registry.list() == ["clf", "clu", "dec"]

    paths = registry.save_all(tmp_path)
    assert {p.name for p in paths} == {"clf.pkl", "clu.pkl", "dec.pkl"}

    restored = PluginRegistry().load_all(tmp_path)
    assert restored.list() == registry.list()
    for name in registry.list():
        assert np.array_equal(restored.get(name).infer(Ete), registry.get(name).infer(Ete))

    direct = PluginRegistry.from_dir(tmp_path)
    assert direct.list() == registry.list()


def test_fit_and_freeze_returns_frozen_plugin(tmp_path):
    Etr, ytr, Ete, _ = _regression_data()
    plugin = SupervisedPlugin(
        HistGradientBoostingRegressor(max_iter=100, random_state=0),
        name="frozen_regressor",
    )
    frozen = fit_and_freeze(plugin, Etr, ytr, tmp_path)
    assert frozen is not plugin
    assert frozen.frozen
    assert (tmp_path / "frozen_regressor.pkl").exists()

    before = frozen.infer(Ete)
    with pytest.raises(RuntimeError):
        frozen.fit(Etr, ytr)
    after = frozen.infer(Ete)
    assert np.array_equal(before, after)

    loaded = SupervisedPlugin.load(tmp_path / "frozen_regressor.pkl")
    assert loaded.frozen
    assert np.array_equal(loaded.infer(Ete), before)

    with pytest.raises(RuntimeError):
        fit_and_freeze(frozen, Etr, ytr, tmp_path)
