"""The objectives scorecard: tier structure, honest UNKNOWN, and rollups."""

from __future__ import annotations

import json

from looking_glass.objectives import (
    FAIL,
    OBJECTIVES,
    PARTIAL,
    PASS,
    UNKNOWN,
    load,
    record,
    scorecard,
    to_markdown,
)


def _by_id(card):
    return {o["id"]: o for o in card["objectives"]}


def test_registry_tiers_are_separate_and_complete():
    ids = [o.id for o in OBJECTIVES]
    assert len(ids) == len(set(ids))
    counts = {}
    for o in OBJECTIVES:
        counts[o.tier] = counts.get(o.tier, 0) + 1
    assert counts == {0: 1, 1: 6, 2: 5, 3: 4, 4: 4}


def test_missing_artifacts_are_unknown_never_green():
    card = scorecard({})
    for o in card["objectives"]:
        assert o["status"] == UNKNOWN, o["id"]
    assert card["by_tier"]["0"]["counts"][UNKNOWN] == 1


def _all_pass():
    return {
        "encoder": {"schema_universal": True, "self_supervised": True, "frozen_donor": True},
        "card": {"solved": True, "u": 0.88},
        "minimality": {"measure_dim": 128, "joint_certified": True},
        "use_cases": {"deploy": 3, "hold": 2, "nulls_hold": 5, "nulls_total": 5},
        "calibration": {"coverage": 0.9, "target": 0.9},
        "accessibility": {"passed": 1, "total": 1},
        "geometry": {"passed": 1, "total": 1},
        "transfer": {"passed": 1, "total": 1},
        "supervised": {"passed": 4, "total": 4},
        "unsupervised": {"passed": 1, "total": 1},
        "ope": {"passed": 10, "total": 10},
        "known_effect": {"passed": 1, "total": 1},
        "robustness": {"passed": 1, "total": 1},
        "efficiency": {"passed": 1, "total": 1},
        "reproducibility": {"passed": 1, "total": 1},
        "honest_gates": {"passed": 1, "total": 1},
    }


def test_all_pass_rolls_up_to_north_star():
    card = scorecard(_all_pass())
    tiers = card["by_tier"]
    assert tiers["0"]["counts"][PASS] == 1
    assert tiers["1"]["counts"][PASS] == 6
    assert tiers["2"]["counts"][PASS] == 5
    assert tiers["3"]["counts"][PASS] == 4
    assert tiers["4"]["counts"][PASS] == 4
    assert _by_id(card)["t0.north_star"]["status"] == PASS


def test_measure_only_minimality_makes_north_star_partial():
    a = _all_pass()
    a["minimality"] = {"measure_dim": 128, "joint_certified": False}
    card = scorecard(a)
    assert _by_id(card)["t2.minimality"]["status"] == PARTIAL
    assert _by_id(card)["t0.north_star"]["status"] == PARTIAL


def test_null_control_incomplete_marks_decision_partial():
    a = _all_pass()
    a["use_cases"] = {"deploy": 3, "hold": 2, "nulls_hold": 4, "nulls_total": 5}
    card = scorecard(a)
    assert _by_id(card)["t2.decision_sufficiency"]["status"] == PARTIAL


def test_no_deploy_fails_decision():
    a = _all_pass()
    a["use_cases"] = {"deploy": 0, "hold": 5, "nulls_hold": 5, "nulls_total": 5}
    assert _by_id(scorecard(a))["t2.decision_sufficiency"]["status"] == FAIL


def test_explicit_status_override_is_honored():
    a = _all_pass()
    a["ope"] = record(FAIL, current="8/10", evidence="recall")
    assert _by_id(scorecard(a))["t3.ope"]["status"] == FAIL


def test_markdown_and_json_roundtrip(tmp_path):
    md = to_markdown(scorecard(_all_pass()))
    assert "Objectives scorecard" in md and "Roll-up" in md
    p = tmp_path / "artifacts.json"
    p.write_text(json.dumps(_all_pass()))
    card = scorecard(load(p))
    out = tmp_path / "card.json"
    out.write_text(json.dumps(card))
    assert json.loads(out.read_text())["by_tier"]["1"]["n"] == 6
