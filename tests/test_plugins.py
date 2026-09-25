"""Data-free unit tests for the plugins layer.

Contracts covered: PluginSpec tag/version identity, the gate report contract,
tagged artifact writes, silhouette-based k selection, supervised metric/alpha
helpers, group-split integrity, the four supervised heads on a synthetic
Dataset, end-to-end supervised + segmentation runs on synthetic data, and the
white_queen plugin's bucketing + reward-policy math.

No duckdb, no repo data files, and no looking_glass/rabbit_hole artifacts are
read; artifacts are written only into tmp_path.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

import plugins.base as base
import plugins.segmentation as segmentation
import plugins.supervised as supervised
from plugins.base import Dataset, PluginSpec
from plugins.white_queen_plugin import (
    N_BUCKETS,
    _RewardPolicy,
    _bucket,
    _fit_reward_policy,
)

SEED = 0


def test_plugin_spec_tag_encodes_name_version_revision():
    spec = PluginSpec(
        name="customer_segmentation_365d",
        kind="unsupervised",
        target="gross_margin",
        window_days=365,
    )
    assert spec.version == "v1.0.0"
    assert spec.revision == 1
    assert spec.tag == "customer_segmentation_365d_v1.0.0r1"
    spec.version = "v2.1.0"
    spec.revision = 7
    assert spec.tag == "customer_segmentation_365d_v2.1.0r7"
    payload = asdict(spec)
    assert payload["window_days"] == 365
    assert payload["kind"] == "unsupervised"
    clone = PluginSpec(**payload)
    assert clone.tag == spec.tag


def test_dataset_defaults():
    spec = PluginSpec(name="clv_365d", kind="supervised", target="gross_margin", window_days=365)
    ds = Dataset(
        spec=spec,
        keys=np.array([]),
        anchor_epoch=np.array([]),
        X=np.zeros((0, 2), np.float32),
        y=np.array([]),
        x_base=np.array([]),
    )
    assert ds.price_mat is None
    assert ds.data_end == 0.0
    assert ds.meta == {}
    assert ds.spec is spec


def test_gate_reports_pass_fail_and_completion(capsys):
    rows = [
        {"check": "alpha", "achieved": 1, "ok": True},
        {"check": "beta", "achieved": 0.5, "ok": False},
    ]
    assert base.gate(rows, "GATE") is False
    out = capsys.readouterr().out
    assert "GATE" in out
    assert "PASS" in out
    assert "FAIL" in out
    assert "completion: 1/2 (50%)" in out
    assert base.gate([{"check": "alpha", "achieved": 1, "ok": True}], "ALL") is True


def test_save_artifact_writes_tagged_json(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "OUT", tmp_path)
    spec = PluginSpec(
        name="clv_supervised_30d",
        kind="supervised",
        target="gross_margin",
        window_days=30,
        version="v1.2.0",
        revision=3,
    )
    path = base.save_artifact(spec, {"verdict": True, "score": np.float32(0.5)})
    assert path == tmp_path / "clv_supervised_30d_v1.2.0r3.json"
    blob = json.loads(path.read_text())
    assert blob["tag"] == spec.tag
    assert blob["spec"]["window_days"] == 30
    assert blob["spec"]["revision"] == 3
    assert blob["payload"] == {"verdict": True, "score": 0.5}


def _three_blobs(seed=SEED, per_blob=30, scale=1.0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [40.0, 0.0], [0.0, 40.0]])
    return np.vstack([c + rng.normal(scale=scale, size=(per_blob, 2)) for c in centers])


def test_choose_k_recovers_blob_count_deterministically():
    X = _three_blobs()
    k, score = segmentation._choose_k(X, kmax=5, seed=0)
    assert k == 3
    assert 0.5 < score <= 1.0
    assert segmentation._choose_k(X, kmax=5, seed=0) == (k, score)
    k_small, _ = segmentation._choose_k(X[:10], kmax=100, seed=0)
    assert 2 <= k_small <= 9


def test_metrics_rank_agreement_mae_and_decile_capture():
    y = np.arange(1.0, 101.0)
    metrics = supervised._metrics(y.copy(), y)
    assert set(metrics) == {"spearman", "mae", "top_decile_capture"}
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["top_decile_capture"] == pytest.approx(95.5 / 50.5)
    assert supervised._metrics(-y, y)["spearman"] == pytest.approx(-1.0)


def test_ridge_alpha_scaled_to_feature_gram():
    assert supervised._ridge_alpha(np.zeros((4, 3))) == 1e-6
    X = np.ones((4, 3))
    alpha = supervised._ridge_alpha(X)
    assert alpha == pytest.approx(1e-3 * 4.0)
    assert supervised._ridge_alpha(10.0 * X) == pytest.approx(100.0 * alpha)


def test_split_is_group_disjoint_and_deterministic():
    keys = np.repeat(np.arange(20), 4)
    tr, te = supervised._split(keys, seed=0)
    assert len(tr) + len(te) == len(keys)
    assert set(keys[tr]).isdisjoint(set(keys[te]))
    assert set(keys[tr]) | set(keys[te]) == set(keys)
    tr2, te2 = supervised._split(keys, seed=0, test=0.3)
    assert np.array_equal(tr, tr2)
    assert np.array_equal(te, te2)


@pytest.fixture
def synthetic_dataset():
    rng = np.random.default_rng(2)
    n, dim = 80, 6
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = 1.0 + 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n)
    x_base = rng.normal(loc=1.0, scale=2.0, size=n)
    return Dataset(
        spec=PluginSpec(
            name="clv_supervised_30d",
            kind="supervised",
            target="gross_margin",
            window_days=30,
        ),
        keys=np.arange(n),
        anchor_epoch=np.arange(n, dtype=np.float64) * 86400.0,
        X=X,
        y=y,
        x_base=x_base,
        meta={},
    )


def test_heads_share_split_and_respect_contracts(synthetic_dataset):
    ds = synthetic_dataset
    point = supervised.head_point(ds, 0)
    baseline = supervised.head_baseline(ds, 0)
    two_part = supervised.head_two_part(ds, 0)
    quantile = supervised.head_quantile(ds, 0)
    n_te = len(point["idx"])
    for head in (point, baseline, two_part, quantile):
        assert head["pred"].shape == (n_te,)
        assert len(head["idx"]) == n_te
        assert set(head["idx"]) <= set(range(len(ds.y)))
        assert set(head["metrics"]) == {"spearman", "mae", "top_decile_capture"}
        assert -1.0 <= head["metrics"]["spearman"] <= 1.0
        assert head["metrics"]["mae"] >= 0.0
    assert np.array_equal(point["idx"], baseline["idx"])
    assert np.array_equal(point["idx"], two_part["idx"])
    assert np.array_equal(point["idx"], quantile["idx"])
    np.testing.assert_array_equal(baseline["pred"], ds.x_base[baseline["idx"]])
    assert (two_part["pred"] >= 0.0).all()
    assert point["metrics"]["spearman"] > 0.5
    assert quantile["monotone"] is True
    assert 0.0 <= quantile["coverage_80"] <= 1.0
    assert (quantile["lo"] <= quantile["pred"] + 1e-6).all()
    assert (quantile["pred"] <= quantile["hi"] + 1e-6).all()


def test_supervised_run_passes_gate_and_writes_artifact(monkeypatch, tmp_path, synthetic_dataset):
    monkeypatch.setattr(base, "load_dataset", lambda window_days: synthetic_dataset)
    monkeypatch.setattr(base, "OUT", tmp_path)
    ok, payload = supervised.run(30, seed=0)
    assert ok is True
    assert payload["verdict"] is True
    names = [h["name"] for h in payload["heads"]]
    assert names == ["point", "two_part", "quantile", "trailing_baseline"]
    path = tmp_path / "clv_supervised_30d_v1.0.0r1.json"
    assert path.exists()
    blob = json.loads(path.read_text())
    assert blob["tag"] == "clv_supervised_30d_v1.0.0r1"
    assert blob["payload"]["verdict"] is True


@pytest.fixture
def segmentation_dataset():
    rng = np.random.default_rng(3)
    n_blobs, per_blob = 12, 25
    centers = np.array([[i * 60.0, (i % 4) * 60.0] for i in range(n_blobs)])
    X = np.vstack([c + rng.normal(scale=1.0, size=(per_blob, 2)) for c in centers])
    blob_id = np.repeat(np.arange(n_blobs), per_blob)
    values = np.array([10.0, 17.0, 23.0, 31.0, 41.0, 47.0, 53.0, 61.0, 67.0, 71.0, 79.0, 83.0])
    y = values[blob_id] + rng.normal(scale=0.01, size=n_blobs * per_blob)
    return Dataset(
        spec=PluginSpec(
            name="customer_segmentation_30d",
            kind="unsupervised",
            target="gross_margin",
            window_days=30,
        ),
        keys=np.arange(n_blobs * per_blob),
        anchor_epoch=np.arange(n_blobs * per_blob, dtype=np.float64) * 86400.0,
        X=X.astype(np.float32),
        y=y,
        x_base=np.zeros(n_blobs * per_blob),
        meta={},
    )


def test_segmentation_run_separates_synthetic_segments(monkeypatch, tmp_path, segmentation_dataset):
    monkeypatch.setattr(segmentation, "load_dataset", lambda window_days: segmentation_dataset)
    monkeypatch.setattr(base, "OUT", tmp_path)
    ok, info = segmentation.run(30, seed=0)
    assert ok is True
    assert 2 <= info["k"] <= 10
    assert 0.02 < info["eta2"] <= 1.0
    assert len(info["means"]) == info["k"]
    assert all(np.isfinite(m) for m in info["means"])
    path = tmp_path / "customer_segmentation_30d_v1.0.0r1.json"
    assert path.exists()
    blob = json.loads(path.read_text())
    assert blob["tag"] == "customer_segmentation_30d_v1.0.0r1"
    assert blob["payload"]["verdict"] is True


def test_bucket_quantile_buckets_and_degenerate_fallback():
    assert N_BUCKETS == 4
    counts = np.array([0, 0, 0, 1, 1, 2, 3, 5, 8, 13])
    b, nB = _bucket(counts, n=4)
    assert b.dtype == np.int64
    assert nB == int(b.max()) + 1
    assert len(set(b.tolist())) >= 2
    assert (b[:-1] <= b[1:]).all()
    degenerate, nB_degenerate = _bucket(np.full(10, 3), n=N_BUCKETS)
    np.testing.assert_array_equal(degenerate, np.full(10, 3))
    assert nB_degenerate == 4


def test_reward_policy_probs_act_and_temperature():
    W = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    policy = _RewardPolicy(W, 4)
    x = np.array([1.0, 2.0])
    probs = policy.action_probs(x)
    assert probs.shape == (1, 4)
    assert probs.dtype == np.float32
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert policy.act(x) == 1
    warm = policy.action_probs(x, temperature=10.0)
    assert warm.sum() == pytest.approx(1.0, abs=1e-6)
    assert warm.max() < probs.max()
    batch = policy.action_probs(np.array([[1.0, 2.0], [0.0, 0.0]]))
    assert batch.shape == (2, 4)
    np.testing.assert_allclose(batch.sum(axis=1), 1.0, atol=1e-6)
    assert policy.act(np.zeros(2)) == 0
    flat = _RewardPolicy(np.zeros((4, 2)), 4)
    np.testing.assert_allclose(flat.action_probs(x)[0], 0.25, atol=1e-6)


def test_fit_reward_policy_zeroes_under_sampled_arms():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 4))
    act = np.array([0] * 20 + [1] * 20 + [2] * 5 + [3] * 5)
    y = rng.normal(size=50)
    policy = _fit_reward_policy(X, act, y, nA=4, seed=0)
    assert policy.W.shape == (4, 4)
    np.testing.assert_array_equal(policy.W[2], np.zeros(4))
    np.testing.assert_array_equal(policy.W[3], np.zeros(4))
    assert not np.allclose(policy.W[0], 0.0)
    probs = policy.action_probs(X[0])
    assert probs.shape == (1, 4)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)
