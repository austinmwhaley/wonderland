"""Data-free unit tests for caterpillar (read-only explain).

Contracts covered: the render formatter (error + full record), nearest-neighbour
ranking in frozen-donor space (cosine similarity, self excluded, k bounded,
plan-membership filter), provenance fields, the CLI entry point, and an xfail
pinning the plan-schema mismatch between engine.run and explain().
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import caterpillar.explain as explain_mod

SEED = 0


@pytest.fixture
def explain_env(monkeypatch, tmp_path):
    rng = np.random.default_rng(SEED)
    keys = [f"C{i}" for i in range(6)]
    E = rng.normal(size=(6, 4)).astype(np.float32)
    plan = {
        "customer_key": list(keys),
        "arm": [0, 1, 2, -1, 1, 0],
        "weekly_sends": [0.2, 0.6, 1.2, 0.0, 0.6, 0.2],
        "expected_gp": [1.0, 2.0, 3.0, 0.0, 2.5, 0.5],
    }
    plan_path = tmp_path / "nba_plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(explain_mod, "_load", lambda: ("cfm_v1", keys, E))
    monkeypatch.setattr(explain_mod, "NBA_PLAN", plan_path)

    def _write(new_plan):
        plan_path.write_text(json.dumps(new_plan))
        return plan_path

    return SimpleNamespace(keys=keys, E=E, plan=plan, write=_write)


def _cosine_neighbors(E, keys, query, k):
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ki = keys.index(query)
    sims = En @ En[ki]
    idx = np.argsort(-sims)[1 : k + 1]
    return [(keys[int(j)], float(sims[int(j)])) for j in idx]


def test_render_error_path():
    out = explain_mod._render({"error": "customer not in plan", "customer": "X"})
    assert isinstance(out, str)
    assert "caterpillar" in out
    assert "customer not in plan" in out
    assert "X" in out


def test_render_full_record_formatting():
    rec = {
        "customer": "C1",
        "donor_version": "v7",
        "recommended_arm": 2,
        "weekly_sends": 1.2,
        "expected_weekly_incremental_gp": 12.34,
        "similar_customers": [
            {"customer": "C2", "similarity": 0.912, "recommended_arm": 1, "expected_gp": 4.5},
            {"customer": "C3", "similarity": 0.75, "recommended_arm": -1, "expected_gp": 0.0},
        ],
        "provenance": {
            "nba_plan": "/x/y.json",
            "embeddings": "anchor_embeddings",
            "cfm_version": "v7",
        },
    }
    out = explain_mod._render(rec)
    lines = out.splitlines()
    assert lines[0] == "Customer C1  (donor v7)"
    assert "arm 2 (1.2 sends/week)" in out
    assert "Expected incremental GP/week : 12.3" in out
    assert "    - C2 (sim 0.912) -> arm 1, E[GP] 4.5" in out
    assert "    - C3 (sim 0.75) -> arm -1, E[GP] 0.0" in out
    assert "Provenance" in out and "cfm_version" in out


def test_explain_neighbours_match_cosine_ranking(explain_env):
    rec = explain_mod.explain("C1", k=3)
    assert rec["customer"] == "C1"
    assert rec["donor_version"] == "cfm_v1"
    assert rec["recommended_arm"] == 1
    assert rec["weekly_sends"] == 0.6
    assert rec["expected_weekly_incremental_gp"] == 2.0
    expected = _cosine_neighbors(explain_env.E, explain_env.keys, "C1", 3)
    got = [(s["customer"], s["similarity"]) for s in rec["similar_customers"]]
    assert got == [(name, round(sim, 3)) for name, sim in expected]
    sims = [s["similarity"] for s in rec["similar_customers"]]
    assert sims == sorted(sims, reverse=True)
    assert "C1" not in [s["customer"] for s in rec["similar_customers"]]
    plan = explain_env.plan
    for entry in rec["similar_customers"]:
        i = plan["customer_key"].index(entry["customer"])
        assert entry["recommended_arm"] == plan["arm"][i]
        assert entry["expected_gp"] == plan["expected_gp"][i]


def test_explain_unknown_customer_returns_error(explain_env):
    rec = explain_mod.explain("NOPE")
    assert rec == {"error": "customer not in plan", "customer": "NOPE"}
    assert "caterpillar" in explain_mod._render(rec)


def test_explain_k_bounded_by_available_neighbours(explain_env):
    rec = explain_mod.explain("C1", k=50)
    assert len(rec["similar_customers"]) == len(explain_env.keys) - 1
    rec_small = explain_mod.explain("C1", k=2)
    assert len(rec_small["similar_customers"]) == 2


def test_explain_skips_neighbours_missing_from_plan(explain_env):
    subset = {k: explain_env.plan[k][:5] for k in explain_env.plan}
    explain_env.write(subset)
    rec = explain_mod.explain("C1", k=5)
    names = [s["customer"] for s in rec["similar_customers"]]
    assert "C5" not in names
    assert set(names) <= set(subset["customer_key"])
    assert len(names) == 4


def test_explain_provenance_and_main(explain_env, capsys):
    rec = explain_mod.explain("C1", k=2)
    prov = rec["provenance"]
    assert prov["nba_plan"] == str(explain_env.write({**explain_env.plan}))
    assert prov["embeddings"] == "anchor_embeddings"
    assert prov["cfm_version"] == "cfm_v1"
    assert explain_mod.main(["--customer", "C1", "--k", "2"]) == 0
    out = capsys.readouterr().out
    assert "Customer C1" in out
    assert "Provenance" in out


@pytest.mark.xfail(
    reason="caterpillar.explain reads plan['expected_gp'] but engine.run "
    "writes 'expected_incremental_gp' into the same nba_plan.json",
    raises=KeyError,
    strict=False,
)
def test_explain_rejects_engine_shaped_plan(explain_env):
    engine_plan = {
        "customer_key": list(explain_env.keys),
        "arm": [0, 1, 2, -1, 1, 0],
        "cadence": ["monthly"] * 6,
        "weekly_sends": [0.0] * 6,
        "expected_incremental_gp": [1.0] * 6,
        "value_sd": [0.0] * 6,
    }
    explain_env.write(engine_plan)
    rec = explain_mod.explain("C1", k=2)
    assert isinstance(rec, dict)
