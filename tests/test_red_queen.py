"""Data-free unit tests for red_queen (NBA engine + controllers).

Contracts covered:
  * engine: fail-safe gate (act only where the lower bound > 0), reject-not-clamp
    under the per-customer cap, budget allocation in value order, derived default
    budget, cadence tiers from value quantiles, plan/receipt schema.
  * nba: greedy budget never exceeded, cheapest-arm fallback, unserved = -1.
  * controller: affordable positive-value allocation, HOLD on non-positive value,
    zero budget, seeded determinism, plan schema.
  * scheduler: target clipped by per-epoch cap, within-epoch spread, softmax arm
    mix, max_rows bound.
  * decision_log: cadence boundaries, action-set log built from synthetic frames,
    plus an xfail pinning the arm-slot counting bug.
  * controller_certified: bucket discretisation, certification-gated HOLD
    (no deploy / no witness), and budget-capped DEPLOY.

All inputs are synthetic in-memory frames/npz under tmp_path; duckdb is faked
where the source imports it; no repo artifacts or data files are read.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import red_queen.controller as controller
import red_queen.controller_certified as controller_certified
import red_queen.decision_log as decision_log
import red_queen.engine as engine
import red_queen.nba as nba
import red_queen.scheduler as scheduler

SEED = 0
WEEK = 7 * 86400.0


@pytest.fixture
def make_engine(monkeypatch, tmp_path):
    def _make(eff, who, cap=None):
        n = len(who)
        keys = [f"C{i}" for i in range(n)]
        E = np.random.default_rng(SEED).normal(size=(n, 4)).astype(np.float32)
        monkeypatch.setattr(engine, "_embeddings", lambda: (keys, E))
        monkeypatch.setattr(engine, "_validated_arm_effects", lambda: np.asarray(eff, dtype=float))
        monkeypatch.setattr(engine, "_donor_value", lambda k: np.asarray(who, dtype=float))
        monkeypatch.setattr(engine, "OUT", tmp_path / "engine")
        if cap is not None:
            monkeypatch.setattr(engine, "PER_CUSTOMER_CAP", cap)
        return keys

    return _make


def _read_plan(tmp_path):
    return json.loads((tmp_path / "engine" / "nba_plan.json").read_text())


def test_engine_failsafe_acts_only_on_positive_value(make_engine, tmp_path):
    keys = make_engine([1.0, 2.0, 0.5, 0.1], [5.0, 4.0, 3.0, -2.0, -1.0, 0.0])
    rec = engine.run()
    assert set(rec) == {
        "customers",
        "acted",
        "no_action_failsafe",
        "no_action_budget",
        "sends_used_wk",
        "budget_wk",
        "arm_mix",
        "cadence_mix",
        "expected_weekly_incremental_gp",
    }
    best = int(np.argmax([1.0, 2.0, 0.5, 0.1]))
    assert rec["customers"] == len(keys)
    assert rec["acted"] == 3
    assert rec["no_action_failsafe"] == 3
    assert rec["no_action_budget"] == 0
    assert rec["sends_used_wk"] == pytest.approx(3 * engine.CADENCE[best])
    assert rec["sends_used_wk"] <= rec["budget_wk"]
    assert rec["arm_mix"] == [0, 3, 0, 0]
    assert set(rec["cadence_mix"]) == {"weekly", "biweekly", "monthly"}
    assert sum(rec["cadence_mix"].values()) == rec["acted"]
    plan = _read_plan(tmp_path)
    assert set(plan) == {
        "customer_key",
        "arm",
        "cadence",
        "weekly_sends",
        "expected_incremental_gp",
        "value_sd",
    }
    assert plan["customer_key"] == keys
    assert plan["arm"][:3] == [best] * 3
    assert plan["arm"][3:] == [-1, -1, -1]
    assert plan["weekly_sends"][3:] == [0.0, 0.0, 0.0]
    assert plan["expected_incremental_gp"][3:] == [0.0, 0.0, 0.0]
    assert plan["value_sd"] == [0.0] * len(keys)


def test_engine_reject_not_clamp_when_arm_over_cap(make_engine, tmp_path):
    assert (engine.CADENCE > 1.0).any()
    keys = make_engine([0.1, 0.2, 0.5, 2.0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.5], cap=1.0)
    rec = engine.run()
    assert rec["acted"] == 0
    assert rec["no_action_budget"] == 0
    assert rec["sends_used_wk"] == 0.0
    assert rec["arm_mix"] == []
    plan = _read_plan(tmp_path)
    assert plan["arm"] == [-1] * len(keys)
    assert plan["weekly_sends"] == [0.0] * len(keys)


def test_engine_budget_allocated_in_value_order(make_engine, tmp_path):
    make_engine([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0, 5.0, 9.0])
    rec = engine.run(weekly_budget=0.5)
    assert rec["budget_wk"] == pytest.approx(0.5)
    assert rec["sends_used_wk"] == pytest.approx(0.4)
    assert rec["sends_used_wk"] <= rec["budget_wk"]
    assert rec["acted"] == 2
    assert rec["no_action_budget"] == 4
    plan = _read_plan(tmp_path)
    assert plan["arm"][5] == 0 and plan["arm"][4] == 0
    assert plan["arm"][:4] == [-1, -1, -1, -1]
    assert rec["expected_weekly_incremental_gp"] == pytest.approx(9.0 + 5.0)


def test_engine_default_budget_is_derived_from_cadence(make_engine, tmp_path):
    keys = make_engine([1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
    rec = engine.run()
    assert rec["budget_wk"] == pytest.approx(len(keys) * engine.CADENCE.mean())
    assert rec["acted"] == len(keys)
    assert rec["sends_used_wk"] <= rec["budget_wk"]


def test_engine_cadence_tiers_follow_value_quantiles(make_engine, tmp_path):
    who = [float(v) for v in range(1, 10)]
    keys = make_engine([1.0, 0.0, 0.0, 0.0], who)
    rec = engine.run()
    plan = _read_plan(tmp_path)
    vbest = np.maximum(np.asarray(who), 0.0)
    q0, q1 = np.quantile(vbest, [0.33, 0.66])
    expected = ["weekly" if v >= q1 else "biweekly" if v >= q0 else "monthly" for v in vbest]
    assert plan["cadence"] == expected
    for tier in ("weekly", "biweekly", "monthly"):
        assert rec["cadence_mix"][tier] == expected.count(tier)
    assert sum(rec["cadence_mix"].values()) == len(keys)


class _ArmValue:
    def __init__(self, bias, scale=0.0):
        self.bias = float(bias)
        self.scale = float(scale)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.scale * X[:, 0] + self.bias


@pytest.fixture
def make_nba(monkeypatch, tmp_path):
    class _TmpRoot:
        def __init__(self, base):
            self._base = base

        def resolve(self):
            return self

        @property
        def parents(self):
            return [self]

        def __truediv__(self, other):
            return self._base / other

    def _make(biases, scales=None, n=6):
        scales = scales or [0.0] * len(biases)
        keys = [f"C{i}" for i in range(n)]
        E = np.random.default_rng(SEED).normal(size=(n, 4)).astype(np.float32)
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler().fit(E)
        models = {a: _ArmValue(biases[a], scales[a]) for a in range(len(biases))}
        monkeypatch.setattr(nba, "_value_model", lambda seed=0: (sc, models, len(biases)))
        monkeypatch.setattr(nba, "_latest_embeddings", lambda: (keys, E))
        monkeypatch.setattr(nba, "Path", lambda *a, **k: _TmpRoot(tmp_path))
        return keys

    return _make


def test_nba_budget_never_exceeded_with_cheapest_arm_fallback(make_nba, tmp_path):
    keys = make_nba([0.2, 0.3, 0.4, 5.0], n=6)
    rec = nba.run(weekly_send_budget=0.7)
    assert set(rec) == {
        "customers",
        "budget_sends_wk",
        "sends_used_wk",
        "arm_mix",
        "unserved",
        "expected_weekly_gp",
    }
    assert rec["customers"] == len(keys)
    assert rec["budget_sends_wk"] == pytest.approx(0.7)
    assert rec["sends_used_wk"] <= rec["budget_sends_wk"]
    assert rec["sends_used_wk"] == pytest.approx(0.6)
    assert rec["arm_mix"] == [3, 0, 0, 0]
    assert rec["unserved"] == 3
    assert sum(rec["arm_mix"]) + rec["unserved"] == len(keys)
    plan = json.loads((tmp_path / "artifacts" / "nba_plan.json").read_text())
    assert plan["customer_key"] == keys
    assert plan["arm"].count(-1) == 3
    for arm, sends, gp in zip(plan["arm"], plan["weekly_sends"], plan["expected_gp"]):
        if arm == -1:
            assert sends == 0.0 and gp == 0.0
        else:
            assert 0 <= arm < 4
            assert sends == engine.CADENCE[arm]


def test_nba_default_budget_serves_within_behaviour_capacity(make_nba, tmp_path):
    keys = make_nba([0.2, 0.3, 0.4, 5.0], n=6)
    rec = nba.run()
    assert rec["budget_sends_wk"] == pytest.approx(len(keys) * nba.CADENCE.mean())
    assert rec["sends_used_wk"] <= rec["budget_sends_wk"]
    assert rec["sends_used_wk"] == pytest.approx(6.0)
    assert rec["arm_mix"] == [0, 0, 0, 3]
    assert rec["unserved"] == 3


def test_nba_run_deterministic(make_nba, tmp_path):
    make_nba([0.2, 0.3, 0.4, 5.0], scales=[0.3, 0.3, 0.3, 0.3], n=6)
    first = nba.run(weekly_send_budget=1.0)
    first_plan = (tmp_path / "artifacts" / "nba_plan.json").read_text()
    second = nba.run(weekly_send_budget=1.0)
    second_plan = (tmp_path / "artifacts" / "nba_plan.json").read_text()
    assert first == second
    assert first_plan == second_plan


def _write_controller_log(tmp_path, reward, counts, cadence, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(reward)
    S = rng.normal(size=(n, 4)).astype(np.float32)
    action = np.zeros((n, 5), np.float32)
    action[:, 0] = counts
    path = tmp_path / "decision_log.npz"
    np.savez(
        path,
        state=S,
        next_state=S,
        action=action,
        reward=np.asarray(reward, np.float32),
        dt_days=np.full(n, 7.0, np.float32),
        cadence=np.asarray(cadence, np.int64),
        traj=np.repeat(np.arange(n // 4), 4).astype(np.int64),
        done=np.zeros(n, np.float32),
    )
    return path


@pytest.fixture
def make_controller(monkeypatch, tmp_path):
    def _make(reward, counts=None, cadence=None):
        n = len(reward)
        counts = np.zeros(n, np.float32) if counts is None else np.asarray(counts, np.float32)
        cadence = np.zeros(n, np.int64) if cadence is None else np.asarray(cadence, np.int64)
        path = _write_controller_log(tmp_path, reward, counts, cadence)
        monkeypatch.setattr(controller, "LOG", path)
        monkeypatch.setattr(controller, "OUT", tmp_path / "nba_schedule.json")
        return path

    return _make


def test_controller_allocates_affordable_positive_value(make_controller, tmp_path):
    n = 24
    counts = np.tile(np.arange(6, dtype=np.float32), 4)
    reward = 0.3 + 0.4 * counts
    make_controller(reward, counts=counts, cadence=np.arange(n) % 3)
    rec = controller.run(global_budget=5.0)
    assert set(rec) == {"epochs", "acted", "used", "by_cadence"}
    assert rec["used"] <= 5.0
    assert rec["used"] == pytest.approx(4.8)
    assert rec["acted"] >= 1
    blob = json.loads((tmp_path / "nba_schedule.json").read_text())
    assert sum(rec["by_cadence"].values()) == sum(p["actions"] for p in blob["plan"])
    assert blob["budget"] == pytest.approx(5.0)
    assert blob["used"] == pytest.approx(rec["used"])
    actions = [p["actions"] for p in blob["plan"]]
    assert max(actions) <= controller.PER_EPOCH_CAP
    assert min(actions) >= 0
    assert sum(actions) > 0


def test_controller_holds_when_value_not_positive(make_controller, tmp_path):
    n = 24
    make_controller(np.full(n, -1.0), cadence=np.arange(n) % 3)
    rec = controller.run(global_budget=5.0)
    assert rec["acted"] == 0
    assert rec["used"] == 0.0
    assert sum(rec["by_cadence"].values()) == 0
    blob = json.loads((tmp_path / "nba_schedule.json").read_text())
    assert all(p["actions"] == 0 for p in blob["plan"])
    assert all(p["expected_incremental_gp"] == 0.0 for p in blob["plan"])


def test_controller_zero_budget_takes_no_touches(make_controller, tmp_path):
    n = 24
    counts = np.tile(np.arange(6, dtype=np.float32), 4)
    make_controller(0.3 + 0.4 * counts, counts=counts, cadence=np.arange(n) % 3)
    rec = controller.run(global_budget=0.0)
    assert rec["used"] == 0.0
    assert rec["acted"] == 0
    assert rec["epochs"] == 1


def test_controller_run_deterministic_under_seed(make_controller):
    n = 24
    counts = np.tile(np.arange(6, dtype=np.float32), 4)
    make_controller(0.3 + 0.4 * counts, counts=counts, cadence=np.arange(n) % 3)
    first = controller.run(global_budget=5.0, seed=0)
    second = controller.run(global_budget=5.0, seed=0)
    assert first == second


def test_controller_plan_schema_and_cadence_labels(make_controller, tmp_path):
    n = 24
    counts = np.tile(np.arange(6, dtype=np.float32), 4)
    cadence = np.arange(n) % 3
    make_controller(0.3 + 0.4 * counts, counts=counts, cadence=cadence)
    controller.run(global_budget=5.0, seed=0)
    blob = json.loads((tmp_path / "nba_schedule.json").read_text())
    label = {0: "daily", 1: "weekly", 2: "monthly"}
    for entry in blob["plan"]:
        assert entry["cadence"] == label[int(cadence[entry["epoch"]])]
        assert entry["actions"] >= 0
        assert entry["actions"] <= controller.PER_EPOCH_CAP
        assert 0.0 < entry["discount"] <= 1.0
        assert entry["discount"] == pytest.approx(controller.GAMMA**7.0, abs=1e-3)
        assert entry["expected_incremental_gp"] >= 0.0


@pytest.fixture
def make_scheduler(monkeypatch, tmp_path):
    def _make(eff, n_customers=20, rows_per=3, n_sends=3):
        customers = np.array([f"CUST_{i:03d}" for i in range(n_customers) for _ in range(rows_per)])
        action = np.zeros((len(customers), 5), np.float32)
        action[:, 0] = n_sends
        path = tmp_path / "decision_log_weekly.npz"
        np.savez(
            path,
            customer=customers,
            epoch_ts=np.full(len(customers), 1.7e9, np.float64),
            action=action,
        )
        monkeypatch.setattr(engine, "_validated_arm_effects", lambda: np.asarray(eff, dtype=float))
        monkeypatch.setattr(scheduler, "LOG", path)
        monkeypatch.setattr(scheduler, "OUT", tmp_path / "touch_schedule.json")
        return path

    return _make


def test_scheduler_softmax_mix_spread_and_composition(make_scheduler, tmp_path):
    eff = np.array([1.0, 3.0, 2.0, 0.5])
    make_scheduler(eff)
    rec = scheduler.run(max_rows=1000, per_epoch_cap=8)
    mix = np.exp(eff - eff.max())
    mix = mix / mix.sum()
    assert rec["customers"] == 20
    assert rec["best_arm"] == int(eff.argmax())
    assert rec["touches"] == 60
    expected_arms = [int(np.searchsorted(np.cumsum(mix), (j + 0.5) / 3)) for j in range(3)]
    expected_by_arm = np.bincount(expected_arms, minlength=4) * rec["customers"]
    assert rec["by_arm"] == expected_by_arm.tolist()
    blob = json.loads((tmp_path / "touch_schedule.json").read_text())
    assert blob["target_per_week"] == 3
    assert blob["best_arm"] == int(eff.argmax())
    assert blob["mix"] == pytest.approx(np.round(mix, 3).tolist())
    assert pytest.approx(sum(blob["mix"]), abs=1e-3) == 1.0
    base = datetime.fromtimestamp(1.7e9, tz=timezone.utc)
    by_customer = {}
    for row in blob["rows"]:
        by_customer.setdefault(row["customer_key"], []).append(row)
    assert len(by_customer) == 20
    for rows in by_customer.values():
        assert [r["arm"] for r in rows] == expected_arms
        stamps = [datetime.fromisoformat(r["ts"]) for r in rows]
        for ts in stamps:
            delta = (ts - base).total_seconds()
            assert 0.0 < delta < WEEK
        assert stamps == sorted(stamps)


def test_scheduler_target_capped_by_per_epoch_cap(make_scheduler):
    make_scheduler([1.0, 3.0, 2.0, 0.5], n_sends=5)
    rec = scheduler.run(max_rows=1000, per_epoch_cap=2)
    assert rec["touches"] == 20 * 2


def test_scheduler_max_rows_bounds_rows_and_customers(make_scheduler, tmp_path):
    make_scheduler([1.0, 3.0, 2.0, 0.5])
    rec = scheduler.run(max_rows=2, per_epoch_cap=8)
    assert rec["customers"] == 1
    assert rec["touches"] == 3
    assert 2 <= rec["touches"] <= 2 + 3 - 1
    rec2 = scheduler.run(max_rows=10, per_epoch_cap=8)
    assert rec2["touches"] >= 10
    assert rec2["touches"] <= 10 + 3 - 1
    assert rec2["customers"] == 4


def test_scheduler_touches_stay_within_epoch_window(make_scheduler, tmp_path):
    make_scheduler([1.0, 3.0, 2.0, 0.5], n_customers=4)
    scheduler.run(max_rows=1000, per_epoch_cap=4)
    blob = json.loads((tmp_path / "touch_schedule.json").read_text())
    assert blob["target_per_week"] == 3
    base = datetime.fromtimestamp(1.7e9, tz=timezone.utc)
    latest = base + timedelta(seconds=WEEK)
    for row in blob["rows"]:
        ts = datetime.fromisoformat(row["ts"])
        assert base < ts < latest


def _cadence(dt_days: float) -> int:
    return 0 if dt_days <= 1.0 else (1 if dt_days <= 7.0 else 2)


def test_decision_log_cadence_boundaries():
    assert decision_log._cadence(0.0) == 0
    assert decision_log._cadence(1.0) == 0
    assert decision_log._cadence(1.0001) == 1
    assert decision_log._cadence(7.0) == 1
    assert decision_log._cadence(7.0001) == 2
    assert decision_log._cadence(90.0) == 2
    values = [0.0, 1.0, 1.0001, 7.0, 7.0001, 90.0]
    labels = [decision_log._cadence(v) for v in values]
    assert labels == sorted(labels)
    assert labels == [_cadence(v) for v in values]


def _fake_duckdb(anchors, splits, sends, inc):
    class _Cursor:
        def __init__(self, sql):
            self.sql = sql

        def pl(self):
            if "encoder_samples" in self.sql:
                return splits
            if "JOIN orders" in self.sql:
                return inc
            if "anchor_embeddings" in self.sql:
                return anchors
            if "email_sends" in self.sql:
                return sends
            raise AssertionError(f"unexpected sql: {self.sql}")

    class _Con:
        def execute(self, sql):
            return _Cursor(sql)

        def close(self):
            return None

    mod = types.ModuleType("duckdb")
    mod.connect = lambda *a, **k: _Con()
    return mod


@pytest.fixture
def decision_log_env(monkeypatch, tmp_path):
    anchors = pl.DataFrame(
        {
            "customer_key": ["A", "A", "A", "B", "B", "C", "C", "D"],
            "anchor_epoch": [0.0, 100000.0, 200000.0, 0.0, 50000.0, 50000.0, 50000.0, 0.0],
            "embedding": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [9.0, 9.0, 9.0],
                [8.0, 8.0, 8.0],
                [5.0, 5.0, 5.0],
            ],
        }
    ).sort("customer_key", "anchor_epoch")
    splits = pl.DataFrame({"customer_key": ["A", "B", "C", "D"], "split": ["B", "B", "B", "A"]})
    sends = pl.DataFrame(
        {"k": ["A", "A", "A", "A"], "t": [1000.0, 2000.0, 90000.0, 150000.0], "arm": [0, 1, 7, 2]}
    )
    inc = pl.DataFrame({"k": ["A", "A"], "t": [5000.0, 120000.0], "gm": [10.0, 5.5]})
    monkeypatch.setitem(sys.modules, "duckdb", _fake_duckdb(anchors, splits, sends, inc))
    out = tmp_path / "decision_log.npz"
    monkeypatch.setattr(decision_log, "OUT", out)
    return out


def test_decision_log_build_action_set_contract(decision_log_env):
    rep = decision_log.build()
    assert rep["steps"] == 3
    assert rep["trajectories"] == 3
    assert rep["state_dim"] == 3
    assert rep["action_dim"] == decision_log.ACTION_DIM == 5
    assert rep["cadence_mix"] == {"daily": 1, "weekly": 2, "monthly": 0}
    assert rep["mean_actions_per_epoch"] == 1.33
    assert rep["max_actions_per_epoch"] == 3
    assert rep["reward_mean"] == 5.17
    assert rep["out"] == str(decision_log_env)
    assert decision_log_env.exists()
    z = np.load(decision_log_env)
    assert z["state"].shape == (3, 3)
    assert z["next_state"].shape == (3, 3)
    assert z["action"].shape == (3, 5)
    np.testing.assert_array_equal(
        z["state"], np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]], np.float32)
    )
    np.testing.assert_array_equal(
        z["next_state"], np.array([[4, 5, 6], [7, 8, 9], [1, 1, 1]], np.float32)
    )
    np.testing.assert_allclose(z["reward"], np.array([10.0, 5.5, 0.0], np.float32))
    np.testing.assert_array_equal(z["cadence"], np.array([1, 1, 0], np.int64))
    np.testing.assert_array_equal(z["done"], np.array([0.0, 1.0, 1.0], np.float32))
    np.testing.assert_array_equal(z["traj"], np.array([0, 0, 1], np.int64))
    np.testing.assert_allclose(
        z["dt_days"],
        np.array([100000.0 / 86400.0, 100000.0 / 86400.0, 50000.0 / 86400.0], np.float32),
    )
    np.testing.assert_array_equal(z["action"][:, 0], np.array([3.0, 1.0, 0.0], np.float32))
    for i in range(3):
        assert z["action"][i, 1:].sum() <= z["action"][i, 0]


@pytest.mark.xfail(
    reason="decision_log.build packs window timestamps into `win` "
    "(`[a for (a, b) in sm ...]` takes the time, not the arm), so the per-arm "
    "slots action[:,1:] are never incremented and stay 0",
    raises=AssertionError,
    strict=False,
)
def test_decision_log_arm_slots_count_sends_by_arm(decision_log_env):
    decision_log.build()
    z = np.load(decision_log_env)
    np.testing.assert_array_equal(z["action"][0], np.array([3.0, 1.0, 1.0, 0.0, 0.0], np.float32))
    np.testing.assert_array_equal(z["action"][1], np.array([1.0, 0.0, 0.0, 1.0, 0.0], np.float32))


def _discretize_reference(n, nA=4):
    edges = np.unique(np.quantile(n, np.linspace(0, 1, nA + 1)[1:-1]))
    b = np.digitize(n, edges)
    rep = {i: float(n[b == i].mean()) if (b == i).any() else 0.0 for i in range(int(b.max()) + 1)}
    return b.astype(np.int64), rep, int(b.max()) + 1


def test_controller_certified_discretize_buckets_and_representatives():
    n = np.array([0, 0, 2, 2, 4, 4, 6, 6], dtype=np.float64)
    b, rep, nA = controller_certified._discretize(n, nA=4)
    b_ref, rep_ref, nA_ref = _discretize_reference(n, nA=4)
    np.testing.assert_array_equal(b, b_ref)
    assert nA == nA_ref == 4
    assert rep == rep_ref
    assert rep == {0: 0.0, 1: 2.0, 2: 4.0, 3: 6.0}
    assert set(rep) == set(range(nA))


def test_controller_certified_discretize_degenerate_collapses_to_one_bucket():
    n = np.full(8, 3.0)
    b, rep, nA = controller_certified._discretize(n, nA=4)
    assert nA == 2
    assert set(b.tolist()) == {1}
    assert rep[1] == 3.0
    assert rep[0] == 0.0


@pytest.fixture
def certified_env(monkeypatch, tmp_path):
    import white_queen.tribunal.ope.data as wq_data
    import white_queen.tribunal.ope.pipeline as pl_mod

    class _Policy:
        bucket = 3

        def act(self, state):
            return int(self.bucket)

    state = {"report": {}, "bucket": 3}

    def fake_train(*args, **kwargs):
        return {"iql": _Policy()}

    def fake_eval(*args, **kwargs):
        return dict(state["report"])

    monkeypatch.setattr(pl_mod, "train_candidates", fake_train)
    monkeypatch.setattr(pl_mod, "evaluate_pool", fake_eval)
    monkeypatch.setattr(wq_data, "to_canonical", lambda *args, **kwargs: {})
    n = 24
    counts = np.tile(np.arange(6, dtype=np.float32), 4)
    path = tmp_path / "decision_log.npz"
    S = np.random.default_rng(SEED).normal(size=(n, 4)).astype(np.float32)
    action = np.zeros((n, 5), np.float32)
    action[:, 0] = counts
    np.savez(
        path,
        state=S,
        next_state=S,
        action=action,
        reward=(0.3 + 0.4 * counts).astype(np.float32),
        dt_days=np.full(n, 7.0, np.float32),
        cadence=np.arange(n) % 3,
        done=np.zeros(n, np.float32),
        traj=np.repeat(np.arange(6), 4).astype(np.int64),
    )
    monkeypatch.setattr(controller_certified, "LOG", path)
    monkeypatch.setattr(controller_certified, "OUT", tmp_path / "certified.json")

    def _set(deployed, witnesses, bucket=3):
        state["bucket"] = bucket
        state["report"] = {
            "deployed": list(deployed),
            "decisions": {name: {"witnesses": witnesses} for name in deployed},
            "behavior_mean": 1.0,
            "bar": 0.5,
            "witnesses": witnesses,
            "rationale": "synthetic",
        }

    return _set


def test_controller_certified_holds_without_deployed_candidate(certified_env, tmp_path):
    certified_env([], 0)
    rec = controller_certified.run(seed=0, global_budget=4.0)
    assert rec["deployed"] == []
    assert rec["epochs"] == 0
    assert rec["used"] == 0.0
    assert rec["by_cadence"] == {}
    blob = json.loads((tmp_path / "certified.json").read_text())
    assert blob["plan"] == []
    assert blob["used"] == 0.0


def test_controller_certified_holds_certificate_without_witness(certified_env, tmp_path):
    certified_env(["iql"], witnesses=0)
    rec = controller_certified.run(seed=0, global_budget=4.0)
    assert rec["deployed"] == []
    assert rec["epochs"] == 0
    assert rec["used"] == 0.0
    assert rec["by_cadence"] == {}
    blob = json.loads((tmp_path / "certified.json").read_text())
    assert blob["plan"] == []


def test_controller_certified_deploy_respects_budget_and_cap(certified_env, tmp_path):
    certified_env(["iql"], witnesses=2, bucket=3)
    rec = controller_certified.run(seed=0, global_budget=4.0)
    assert rec["deployed"] == ["iql"]
    assert rec["epochs"] >= 1
    assert rec["used"] <= 4.0
    blob = json.loads((tmp_path / "certified.json").read_text())
    actions = [p["actions"] for p in blob["plan"]]
    assert sum(actions) == rec["used"]
    assert max(actions) <= controller_certified.PER_EPOCH_CAP
    assert min(actions) >= 0
    assert set(rec["by_cadence"]) <= {"daily", "weekly", "monthly"}
    assert sum(rec["by_cadence"].values()) == sum(actions)
    for entry in blob["plan"]:
        assert entry["action_bucket"] == 3


def test_controller_certified_run_deterministic(certified_env):
    certified_env(["iql"], witnesses=2, bucket=3)
    first = controller_certified.run(seed=0, global_budget=4.0)
    second = controller_certified.run(seed=0, global_budget=4.0)
    assert first == second
