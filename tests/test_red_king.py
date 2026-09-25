"""Data-free unit tests for red_king (counterfactual world model).

Covers the pure / small-config paths: the HTE feature builder, the supervised
effect-model and per-customer HTE receipts (with a synthetic ``build()``), the
RSSM module (step/obs shapes, log-variance clamp, seeded determinism, train
receipt, rollout uncertainty), and the world-model transition builder + group
split. Everything is synthetic, seeded, and CPU-light: no duckdb, no repo
artifacts, no ``red_king/data``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

import red_king.effect_model as effect_model
import red_king.hte_model as hte_model
import red_king.rssm as rssm
import red_king.world_model as world_model

SEED = 0
N_ARM = 4


def _synthetic_build(seed=SEED):
    """Synthetic (X, A, Y, W, keys, optmap) shaped like effect_model.build()."""
    rng = np.random.default_rng(seed)
    n, dim, n_customers = 60, 6, 12
    X = rng.normal(size=(n, dim)).astype(np.float32)
    A = rng.integers(0, N_ARM, n).astype(np.int64)
    Y = rng.normal(loc=2.0, size=n).astype(np.float32)
    W = rng.uniform(1.0, 3.0, n).astype(np.float32)
    keys = [f"C{i % n_customers:02d}" for i in range(n)]
    for c in range(4):
        for j in range(N_ARM):
            keys[c * N_ARM + j] = f"C{c:02d}"
            A[c * N_ARM + j] = j
    optmap = {f"C{i:02d}": i % N_ARM for i in range(n_customers)}
    return X, A, Y, W, keys, optmap


@pytest.fixture
def synthetic_build(monkeypatch):
    data = _synthetic_build()
    monkeypatch.setattr(effect_model, "build", lambda: data)
    monkeypatch.setattr(hte_model, "build", lambda: data)
    return data


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_hte_feature_matrix_contract():
    S = np.arange(12, dtype=np.float32).reshape(4, 3)
    A = np.array([0, 2, 1, 3], np.int64)
    F = hte_model._features(S, A, N_ARM)
    assert F.shape == (4, 3 + 3 * N_ARM + N_ARM)
    assert F.dtype == np.float32
    np.testing.assert_allclose(F[:, :3], S)
    oh = F[:, -N_ARM:]
    assert np.array_equal(oh, np.eye(N_ARM, dtype=np.float32)[A])
    for i in range(4):
        for a in range(N_ARM):
            block = F[i, 3 + a * 3 : 3 + (a + 1) * 3]
            expected = S[i] if a == A[i] else np.zeros(3, np.float32)
            np.testing.assert_array_equal(block, expected)


def test_effect_model_run_receipt_contract(synthetic_build, cpu_only):
    X, A, _, _, keys, _ = synthetic_build
    torch.manual_seed(SEED)
    res = effect_model.run(seed=SEED, K=2, steps=3)
    assert set(res) == {
        "windows",
        "customers",
        "nA",
        "per_customer_rank_acc",
        "majority_baseline",
        "beats_majority",
    }
    assert res["windows"] == len(X)
    assert res["customers"] == len(set(keys))
    assert res["nA"] == int(A.max()) + 1
    assert 0.0 <= res["majority_baseline"] <= 1.0
    assert 0.0 <= res["per_customer_rank_acc"] <= 1.0
    assert res["beats_majority"] is (res["per_customer_rank_acc"] > res["majority_baseline"])


def test_effect_model_run_deterministic_under_seed(synthetic_build, cpu_only):
    torch.manual_seed(SEED)
    first = effect_model.run(seed=SEED, K=2, steps=3)
    torch.manual_seed(SEED)
    second = effect_model.run(seed=SEED, K=2, steps=3)
    assert first == second


def test_hte_run_receipt_contract(synthetic_build):
    _, A, _, _, keys, _ = synthetic_build
    nA = int(A.max()) + 1
    arms_of = {}
    for k, a in zip(keys, A):
        arms_of.setdefault(k, set()).add(int(a))
    expected_switch = sum(1 for s in arms_of.values() if len(s) >= nA)
    res = hte_model.run(seed=SEED)
    assert set(res) == {
        "windows",
        "customers",
        "nA",
        "per_customer_rank_acc",
        "majority_baseline",
        "beats_majority",
        "switchback_customers",
        "switch_per_customer_acc",
        "switch_majority",
        "switch_beats_majority",
    }
    assert res["nA"] == nA
    assert res["customers"] == len(arms_of)
    assert res["switchback_customers"] == expected_switch
    assert 0.0 <= res["per_customer_rank_acc"] <= 1.0
    assert isinstance(res["beats_majority"], bool)
    assert isinstance(res["switch_beats_majority"], bool)
    if res["switchback_customers"]:
        assert 0.0 <= res["switch_per_customer_acc"] <= 1.0
        assert 0.0 <= res["switch_majority"] <= 1.0
    else:
        assert res["switch_per_customer_acc"] is None
        assert res["switch_majority"] is None


def test_hte_run_deterministic_under_seed(synthetic_build):
    first = hte_model.run(seed=SEED)
    second = hte_model.run(seed=SEED)
    assert first == second


def test_rssm_step_and_obs_shapes():
    torch.manual_seed(SEED)
    model, zdim, hdim = rssm.make_rssm(nA=N_ARM, dim=6, zdim=5, hdim=7)
    assert (zdim, hdim) == (5, 7)
    B = 3
    h = torch.zeros(B, hdim)
    z = torch.zeros(B, zdim)
    a = torch.arange(B) % N_ARM
    dt = torch.ones(B, 1)
    s = torch.randn(B, 6)
    h2, (pmu, plv), (qmu, qlv) = model.step(h, z, a, dt, s)
    assert h2.shape == (B, hdim)
    assert pmu.shape == plv.shape == qmu.shape == qlv.shape == (B, zdim)
    rec, rew, cont = model.obs(h2, pmu)
    assert rec.shape == (B, 6)
    assert rew.shape == (B, 1)
    assert cont.shape == (B,)
    assert torch.isfinite(h2).all()
    assert torch.isfinite(rec).all()
    assert torch.isfinite(rew).all()
    assert torch.isfinite(cont).all()


def test_rssm_split_clamps_logvariance():
    model, zdim, _ = rssm.make_rssm(nA=N_ARM, dim=6, zdim=5, hdim=7)
    mu_in = [100.0] * zdim
    lv_in = [100.0, -100.0, 0.0, 50.0, -50.0]
    o = torch.tensor([mu_in + lv_in])
    mu, lv = model.split(o)
    assert torch.equal(mu, torch.tensor([mu_in]))
    assert torch.equal(lv, torch.tensor([[4.0, -6.0, 0.0, 4.0, -6.0]]))


def test_rssm_make_deterministic_under_seed():
    torch.manual_seed(1)
    m1, _, _ = rssm.make_rssm(nA=N_ARM, dim=6, zdim=5, hdim=7)
    torch.manual_seed(1)
    m2, _, _ = rssm.make_rssm(nA=N_ARM, dim=6, zdim=5, hdim=7)
    s1, s2 = m1.state_dict(), m2.state_dict()
    assert s1.keys() == s2.keys()
    for k in s1:
        assert torch.equal(s1[k], s2[k])
    torch.manual_seed(2)
    m3, _, _ = rssm.make_rssm(nA=N_ARM, dim=6, zdim=5, hdim=7)
    s3 = m3.state_dict()
    assert any(not torch.equal(s1[k], s3[k]) for k in s1)


def _synthetic_sequences(seed=SEED, n=6, T=4, dim=5):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, T, dim)).astype(np.float32)
    A = rng.integers(0, N_ARM, (n, T)).astype(np.int64)
    R = rng.normal(size=(n, T)).astype(np.float32)
    DT = np.full((n, T), 7.0, np.float32)
    D = np.zeros((n, T), np.float32)
    D[:, -1] = 1.0
    M = np.ones((n, T), np.float32)
    Pp = np.full((n, T), 0.5, np.float32)
    return S, A, R, DT, D, M, N_ARM, Pp


def test_rssm_train_receipt_and_checkpoint(monkeypatch, tmp_path, cpu_only):
    S, A, R, DT, D, M, nA, Pp = _synthetic_sequences()
    n, T, dim = S.shape
    monkeypatch.setattr(rssm, "build_sequences", lambda: (S, A, R, DT, D, M, nA, Pp))
    ckpt = tmp_path / "rssm.pt"
    monkeypatch.setattr(rssm, "OUT", ckpt)
    res = rssm.train(seed=SEED, K=1, zdim=4, hdim=8, steps=2)
    assert set(res) == {
        "sequences",
        "T",
        "dim",
        "reward_r2",
        "next_state_cos",
        "out",
    }
    assert res["sequences"] == n
    assert res["T"] == T
    assert res["dim"] == dim
    assert np.isfinite(res["reward_r2"])
    assert -1.0 <= res["next_state_cos"] <= 1.0
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert blob["nA"] == nA
    assert blob["dim"] == dim
    assert blob["hdim"] == 8
    assert blob["zdim"] == 4
    assert len(blob["ens"]) == 1


def _write_checkpoint(path, members=1, seed=11):
    torch.manual_seed(seed)
    states = []
    for _ in range(members):
        model, _, _ = rssm.make_rssm(nA=N_ARM, dim=5, zdim=4, hdim=8)
        states.append(model.state_dict())
    torch.save(
        {"ens": states, "hdim": 8, "zdim": 4, "nA": N_ARM, "dim": 5, "rstd": 2.0},
        path,
    )
    return path


def test_rssm_rollout_shapes_and_single_member_zero_spread(tmp_path):
    ckpt = _write_checkpoint(tmp_path / "ens1.pt", members=1)
    states = np.random.default_rng(SEED).normal(size=(3, 5)).astype(np.float32)
    V, SD = rssm.rollout_arm_values(states, horizon=3, gamma=0.9, path=ckpt)
    assert V.shape == SD.shape == (3, N_ARM)
    assert np.isfinite(V).all()
    assert np.isfinite(SD).all()
    assert (SD >= 0).all()
    assert np.all(SD == 0.0)
    assert np.unique(np.round(V[0], 6)).size > 1


def test_rssm_rollout_ensemble_uncertainty_and_determinism(tmp_path):
    ckpt = _write_checkpoint(tmp_path / "ens3.pt", members=3)
    states = np.random.default_rng(SEED).normal(size=(3, 5)).astype(np.float32)
    V1, SD1 = rssm.rollout_arm_values(states, horizon=3, gamma=0.9, path=ckpt, batch=2)
    V2, SD2 = rssm.rollout_arm_values(states, horizon=3, gamma=0.9, path=ckpt, batch=2)
    assert V1.shape == SD1.shape == (3, N_ARM)
    assert np.array_equal(V1, V2)
    assert np.array_equal(SD1, SD2)
    assert (SD1 >= 0).all()
    assert (SD1 > 0).any()
    Vh1, _ = rssm.rollout_arm_values(states, horizon=1, gamma=0.9, path=ckpt, batch=2)
    assert not np.allclose(Vh1, V1)


@pytest.mark.xfail(
    reason="rollout_arm_values imagines from a zero prior and discards the "
    "posterior branch, so V comes out identical for every input state",
    raises=AssertionError,
    strict=False,
)
def test_rssm_rollout_values_depend_on_state(tmp_path):
    ckpt = _write_checkpoint(tmp_path / "ens2.pt", members=2)
    states = np.stack([np.zeros(5, np.float32), np.ones(5, np.float32) * 10.0])
    V, _ = rssm.rollout_arm_values(states, horizon=3, gamma=0.9, path=ckpt)
    assert not np.allclose(V[0], V[1])


def test_world_model_split_is_group_disjoint_and_deterministic():
    groups = np.repeat(np.arange(10), 3)
    tr, te = world_model._split(groups, seed=0, test=0.3)
    assert len(tr) + len(te) == len(groups)
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    assert set(groups[tr]) | set(groups[te]) == set(groups)
    tr2, te2 = world_model._split(groups, seed=0, test=0.3)
    assert np.array_equal(tr, tr2)
    assert np.array_equal(te, te2)
    tr3, te3 = world_model._split(groups, seed=1, test=0.3)
    assert not (np.array_equal(tr, tr3) and np.array_equal(te, te3))


@pytest.fixture
def wm_transitions(monkeypatch):
    keys = []
    epochs = []
    embs = []
    for i, k in enumerate(["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"]):
        keys.append(k)
        epochs.append(float((i % 3) * 100))
        embs.append([float(i), float(i)])
    keys += ["E", "E"]
    epochs += [50.0, 50.0]
    embs += [[12.0, 12.0], [13.0, 13.0]]
    anchors = pl.DataFrame({"customer_key": keys, "anchor_epoch": epochs, "embedding": embs})
    send_times = [
        ("A", 0.0),
        ("A", 100.0),
        ("A", 150.0),
        ("A", 200.0),
        ("B", 10.0),
        ("B", 20.0),
        ("B", 110.0),
        ("B", 120.0),
        ("B", 130.0),
        ("C", 10.0),
        ("C", 20.0),
        ("C", 30.0),
        ("C", 40.0),
        ("C", 110.0),
        ("C", 120.0),
        ("C", 130.0),
        ("C", 140.0),
        ("C", 150.0),
        ("D", 10.0),
        ("D", 20.0),
        ("D", 30.0),
        ("D", 40.0),
        ("D", 50.0),
        ("D", 60.0),
        ("D", 110.0),
        ("D", 120.0),
        ("D", 130.0),
        ("D", 140.0),
        ("D", 150.0),
        ("D", 160.0),
        ("D", 170.0),
    ]
    sends = pl.DataFrame(
        {
            "customer_key": [k for k, _ in send_times],
            "t": [t for _, t in send_times],
        }
    )
    orders = pl.DataFrame(
        {
            "customer_key": ["A", "A", "A", "B"],
            "t": [0.0, 100.0, 150.0, 50.0],
            "gm": [99.0, 5.0, 7.0, 2.5],
        }
    )
    monkeypatch.setattr(world_model, "_load_anchors", lambda: anchors)
    monkeypatch.setattr(world_model, "_load_facts", lambda: (sends, orders))
    return anchors


def test_world_model_build_transitions_windows_and_buckets(wm_transitions):
    S, A, R, S2, D, nAb, C = world_model.build_transitions(nA=4)
    assert S.shape == (8, 2)
    assert S2.shape == (8, 2)
    assert A.shape == R.shape == D.shape == C.shape == (8,)
    assert S.dtype == np.float32 and S2.dtype == np.float32
    np.testing.assert_array_equal(C, np.array(["A", "A", "B", "B", "C", "C", "D", "D"]))
    assert "E" not in set(C.tolist())
    np.testing.assert_array_equal(
        S, np.array([[0, 0], [1, 1], [3, 3], [4, 4], [6, 6], [7, 7], [9, 9], [10, 10]], np.float32)
    )
    np.testing.assert_array_equal(
        S2,
        np.array([[1, 1], [2, 2], [4, 4], [5, 5], [7, 7], [8, 8], [10, 10], [11, 11]], np.float32),
    )
    np.testing.assert_allclose(R, np.array([5.0, 7.0, 2.5, 0, 0, 0, 0, 0], np.float32))
    np.testing.assert_array_equal(D, np.zeros(8, np.float32))
    counts = np.array([1, 2, 2, 3, 4, 5, 6, 7], dtype=float)
    edges = np.unique(np.quantile(counts, np.linspace(0, 1, 5)[1:-1]))
    expected = np.digitize(counts, edges).astype(np.int64)
    np.testing.assert_array_equal(A, expected)
    assert nAb == int(expected.max()) + 1
    assert 1 <= nAb <= 5


def test_world_model_empty_transition_set_is_rejected(monkeypatch):
    anchors = pl.DataFrame(
        {
            "customer_key": ["A", "A"],
            "anchor_epoch": [100.0, 100.0],
            "embedding": [[1.0, 1.0], [2.0, 2.0]],
        }
    )
    sends = pl.DataFrame({"customer_key": ["A"], "t": [50.0]})
    orders = pl.DataFrame({"customer_key": ["A"], "t": [50.0], "gm": [1.0]})
    monkeypatch.setattr(world_model, "_load_anchors", lambda: anchors)
    monkeypatch.setattr(world_model, "_load_facts", lambda: (sends, orders))
    with pytest.raises(IndexError):
        world_model.build_transitions(nA=2)
