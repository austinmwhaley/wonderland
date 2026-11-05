"""The B scorecard must be unable to lie (DEC-562).

The rule under test is the same as the judge card's: a row is PASS only on a CURRENT receipt that
meets its gate. Missing and legacy receipts are UNKNOWN and name the producer; a legacy number
never becomes a green row. The B section must not change the judge summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import b_spec as B  # noqa: E402
import judge_spec as J  # noqa: E402
import report_card as RC  # noqa: E402

HASH = "abc123"


def _meta(h: str = HASH) -> dict:
    return {
        "generated_at": "2026-11-02T00:00:00+00:00",
        "git_commit": "deadbeef",
        "behavior_hash": h,
    }


def _stage(producer: str, working: bool = True, h: str = HASH) -> dict:
    return {
        "working": working,
        "evidence": "test block",
        "_stamp": dict(_meta(h), producer=producer),
    }


def _write(tmp_path: Path, name: str, payload) -> None:
    (tmp_path / name).write_text(json.dumps(payload))


def test_missing_b_rows_are_unknown_and_name_the_producer(tmp_path):
    md, info = RC.render(tmp_path, HASH)
    b = info["summary"]["b"]
    assert b["pass"] == 0
    assert b["unknown"] >= 15
    assert b["missing"] >= 12
    assert b["sections_green"] == 0
    assert info["b_verdicts"]["B2.1"] == J.UNKNOWN
    assert info["b_verdicts"]["B4.1"] == J.UNKNOWN
    assert "B — foundational scorecard" in md
    assert "measure_sufficiency_law.py" in md


def test_unstamped_b_receipt_is_unknown_not_pass(tmp_path):
    """DEC-585: the 8 legacy/missing rows are re-stamped kinds now. A receipt with the
    passing NUMBERS but no _meta stamp (i.e. measured by a dead producer) is UNKNOWN,
    and names the producer that must (re)run it."""
    body = _construction_body()
    body.pop("_meta")  # the point of the test: numbers without a fresh stamp
    _write(tmp_path, "encoder_lawfulness.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B1.2"] == J.UNKNOWN
    md, _ = RC.render(tmp_path, HASH)
    assert "measure_construction_law.py" in md


# ---------------------------------------------------------------------------
# DEC-585 kinds: cfm_v2k re-stamp gates (frozen bars in b_spec; verdicts in the card)
# ---------------------------------------------------------------------------
def _construction_body(h: str = HASH, field_perm: float = 1e-09) -> dict:
    return {
        "_meta": _meta(h),
        "complete": True,
        "n_sequences": 20,
        "device": "cpu",
        "checks": {
            "field_permutation": {"max_rel_err": field_perm, "working": True},
            "scan_associativity": {"worst_rel_err": 1.1e-15, "working": True},
            "mass_additivity": {"worst_rel_err": 5e-16, "working": True},
            "determinism": {"bit_identical": True},
            "schema_variant": {"working": True},
            "frozen_identity": {"present": True},
        },
    }


def test_construction_law_passes_when_all_six_checks_meet_the_bars(tmp_path):
    _write(tmp_path, "encoder_lawfulness.json", _construction_body())
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B1.2"] == J.PASS
    assert "field-perm" in md


def test_construction_law_fails_when_the_field_perm_exceeds_the_bar(tmp_path):
    _write(tmp_path, "encoder_lawfulness.json", _construction_body(field_perm=1e-03))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B1.2"] == J.FAIL
    assert "C1 field-perm" in md


def test_decision_dose_passes_with_controls_below_the_brain(tmp_path):
    _write(
        tmp_path,
        "encoder_decision_law.json",
        {
            "_meta": _meta(),
            "decision_dose": {
                "complete": True,
                "n": 1485,
                "n_levels": 4,
                "levels": [0, 1, 2, 3],
                "share_E": 0.55,
                "share_table": 0.40,
                "ci_lo": 0.05,
                "ci_hi": 0.25,
                "share_random": 0.25,
                "share_destroyed": 0.26,
                "thin_levels": [],
                "oracle_ceiling": 0.6404,
            },
        },
    )
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B4.1"] == J.PASS
    assert "oracle ceiling 0.6404" in md


def test_decision_dose_fails_when_E_loses_to_the_table(tmp_path):
    body = {
        "_meta": _meta(),
        "decision_dose": {
            "complete": True,
            "n": 1485,
            "share_E": 0.30,
            "share_table": 0.40,
            "ci_lo": -0.10,
            "ci_hi": 0.05,
            "share_random": 0.25,
            "share_destroyed": 0.26,
        },
    }
    _write(tmp_path, "encoder_decision_law.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B4.1"] == J.FAIL


def test_decision_dose_fails_when_a_control_reaches_the_brain(tmp_path):
    body = {
        "_meta": _meta(),
        "decision_dose": {
            "complete": True,
            "n": 1485,
            "share_E": 0.40,
            "share_table": 0.30,
            "ci_lo": 0.05,
            "ci_hi": 0.25,
            "share_random": 0.40,
            "share_destroyed": 0.26,
        },
    }
    _write(tmp_path, "encoder_decision_law.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B4.1"] == J.FAIL


def test_decision_ope_passes_when_the_diff_and_ci_exclude_zero(tmp_path):
    _write(
        tmp_path,
        "encoder_decision_law.json",
        {
            "_meta": _meta(),
            "decision_ope": {
                "complete": True,
                "arm": "offer",
                "E_ips": 500.0,
                "T_ips": 487.5,
                "random": 500.0,
                "diff": 12.5,
                "lo": 3.2,
                "hi": 21.4,
                "accE": 0.31,
                "accT": 0.30,
                "n_train": 1038,
                "n_test": 445,
                "n_pop": 1483,
                "n_excluded": 17,
                "level_frequencies": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
            },
        },
    )
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B4.2"] == J.PASS
    assert "12.5" in md


def test_decision_ope_fails_when_the_ci_straddles_zero(tmp_path):
    body = {
        "_meta": _meta(),
        "decision_ope": {
            "complete": True,
            "E_ips": 480.0,
            "T_ips": 492.5,
            "random": 500.0,
            "diff": -12.5,
            "lo": -21.4,
            "hi": -3.2,
            "accE": 0.30,
            "accT": 0.31,
            "n_test": 445,
        },
    }
    _write(tmp_path, "encoder_decision_law.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B4.2"] == J.FAIL


def _decision_invariance_body(ok: bool = True, s2=0.53, s3=0.41, h: str = HASH) -> dict:
    return {
        "_meta": _meta(h),
        "decision_invariance": {
            "complete": True,
            "check1_cutoff_band": {
                "ok": ok,
                "why": "band ok",
                "max_delta_S": 3.6e-07,
                "max_delta_E": 9.2e-05,
                "band_S": 7.1e-07,
                "band_E": 2.6e-04,
                "teeth_ok": True,
            },
            "check2_marker_invariance": {
                "complete": True,
                "share_E_prime": s2,
                "share_E_stored": 0.55,
                "ok": s2 >= 0.55 if s2 is not None else None,
            },
            "check3_level_teeth": {
                "complete": True,
                "share_permuted": s3,
                "share_table": 0.40,
                "abs_delta_inf": 0.07,
                "ok": True,
            },
        },
    }


def test_decision_invariance_passes_on_the_band_cert_with_finite_teeth(tmp_path):
    _write(tmp_path, "encoder_decision_law.json", _decision_invariance_body())
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B4.3"] == J.PASS
    assert "band cert ok" in md


def test_decision_invariance_fails_when_the_band_cert_is_not_ok(tmp_path):
    _write(tmp_path, "encoder_decision_law.json", _decision_invariance_body(ok=False))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B4.3"] == J.FAIL


def test_decision_invariance_is_unknown_when_the_teeth_values_are_missing(tmp_path):
    _write(tmp_path, "encoder_decision_law.json", _decision_invariance_body(s2=None, s3=None))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B4.3"] == J.UNKNOWN


def _runtime_body(n=5000, rss=31000.0, cap=32.0, complete=True, h: str = HASH) -> dict:
    return {
        "_meta": _meta(h),
        "complete": complete,
        "n": n,
        "n_expected": 5000,
        "cold_seconds": 40.0,
        "warm_seconds": 620.0,
        "rows_per_s": 7.4,
        "close_seconds": 2.1,
        "dims_state": 256,
        "dims_served": 256,
        "peak_rss_mb": rss,
        "slice_cap_gb": cap,
        "incremental": {
            "as_of_epoch": 1735689600.0,
            "n_absorbed": 0,
            "n_idle": 5000,
            "n_embeddings": 5000,
            "device": "encoder",
            "wall_seconds": 0.66,
        },
    }


def test_runtime_law_passes_within_the_slice(tmp_path):
    _write(tmp_path, "runtime.json", _runtime_body())
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B5.2"] == J.PASS
    assert "rows/s" in md


def test_runtime_law_fails_when_n_does_not_match_the_registry(tmp_path):
    _write(tmp_path, "runtime.json", _runtime_body(n=4000))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B5.2"] == J.FAIL


def test_runtime_law_fails_when_the_peak_rss_exceeds_the_slice(tmp_path):
    _write(tmp_path, "runtime.json", _runtime_body(rss=40000.0))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B5.2"] == J.FAIL


def test_runtime_law_is_ci_no_op_when_the_slice_cap_is_absent(tmp_path):
    _write(tmp_path, "runtime.json", _runtime_body(rss=40000.0, cap=None))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B5.2"] == J.PASS


def _donor_body(
    E=0.52, raw=0.40, fair=0.40, ci_lo=0.02, random=0.05, destroyed=0.06, h: str = HASH
) -> dict:
    t = {
        "metric": "r2",
        "degenerate": False,
        "E": E,
        "raw": raw,
        "fair": fair,
        "table": fair,
        "table_E": E + 0.05,
        "random": random,
        "destroyed": destroyed,
        "ci_lo": ci_lo,
        "ci_hi": ci_lo + 0.1,
    }
    return {
        "_meta": _meta(h),
        "complete": True,
        "targets": {
            "y_value_90": dict(t),
            "y_orders_90": dict(t),
            "y_churn_90": {"metric": "auc", "degenerate": True, "reason": "target has <2 classes"},
        },
        "n_A": 3515,
        "n_B": 1485,
        "raw_equals_fair": True,
    }


def test_donor_battery_passes_with_controls_below_the_brain(tmp_path):
    _write(tmp_path, "encoder_donor.json", _donor_body())
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B6.2"] == J.PASS
    assert "excluded (degenerate)" in md


def test_donor_battery_fails_when_E_loses_to_raw(tmp_path):
    _write(tmp_path, "encoder_donor.json", _donor_body(E=0.30, raw=0.40))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B6.2"] == J.FAIL


def test_donor_battery_fails_when_the_add_ci_straddles_zero(tmp_path):
    _write(tmp_path, "encoder_donor.json", _donor_body(ci_lo=-0.01))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B6.2"] == J.FAIL


def _b_props_body(ece=0.05, present=True, h: str = HASH) -> dict:
    return {
        "_meta": _meta(h),
        "complete": True,
        "calibration": {
            "complete": True,
            "ece_next_event": ece,
            "acc": 0.32,
            "n_next": 1400,
            "n_pop": 1485,
            "n_oov": 0,
            "ood": {"ece": 0.07, "acc": 0.30, "n": 85, "horizon_days": 30},
            "device": "cuda:0",
            "why": "multiclass ECE of the shipped head_next",
        },
        "identity": {
            "complete": True,
            "present": present,
            "tag": "0.11.0r490759-test",
            "registry_sha256": "abcdef0123456789",
            "data_signature": "3543c711bc98",
            "as_of": "2025-01-01",
            "n_customers": 5000,
            "config_sha256": "9876543210fedcba",
            "checkpoint_sha256": "1234567890abcdef",
            "why": "registry resolved",
        },
    }


def test_calibration_law_passes_under_the_ece_bar(tmp_path):
    _write(tmp_path, "encoder_b_props.json", _b_props_body(ece=0.05))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B7.1"] == J.PASS
    assert "next-type ECE 0.0500" in md


def test_calibration_law_fails_above_the_ece_bar(tmp_path):
    _write(tmp_path, "encoder_b_props.json", _b_props_body(ece=0.15))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B7.1"] == J.FAIL


def test_calibration_law_fails_when_the_ece_was_not_measured(tmp_path):
    _write(tmp_path, "encoder_b_props.json", _b_props_body(ece=None))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B7.1"] == J.FAIL


def test_identity_law_passes_when_the_registry_identity_is_present(tmp_path):
    _write(tmp_path, "encoder_b_props.json", _b_props_body())
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B8.1"] == J.PASS
    assert "registry abcdef0123456789" in md


def test_identity_law_fails_when_the_checkpoint_is_missing(tmp_path):
    _write(tmp_path, "encoder_b_props.json", _b_props_body(present=False))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B8.1"] == J.FAIL


def test_fresh_stage_blocks_can_pass_b_stage_rows(tmp_path):
    blocks = {
        key: _stage(producer)
        for key, producer in (
            ("S3", "measure_dynamics_law.py"),
            ("S4", "measure_views_law.py"),
            ("S5", "measure_battery_law.py"),
            ("S6", "measure_certification_law.py"),
        )
    }
    _write(tmp_path, "judge_stages.json", {"_meta": _meta(), **blocks})
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B1.1"] == J.PASS
    assert info["b_verdicts"]["B3.1"] == J.PASS
    assert info["b_verdicts"]["B6.1"] == J.PASS
    assert info["b_verdicts"]["B8.2"] == J.PASS


def test_stale_stage_block_is_unknown_not_pass(tmp_path):
    _write(
        tmp_path,
        "judge_stages.json",
        {"_meta": _meta(), "S3": _stage("measure_dynamics_law.py", h="stale")},
    )
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B1.1"] == J.UNKNOWN


def test_b_section_does_not_change_the_judge_summary(tmp_path):
    _, info = RC.render(tmp_path, HASH)
    # no judge leg receipt exists, so every judge row is UNKNOWN; hygiene rows may PASS
    assert all(info["verdicts"][r.ident] == J.UNKNOWN for r in J.LEG_ROWS)
    # the B rollup is separate, so the judge gate keeps its own meaning
    assert info["summary"]["b"]["pass"] == 0


def test_stage_decomposition_receipt_can_pass_and_carries_the_numbers(tmp_path):
    body = {
        "_meta": _meta(),
        "complete": True,
        "stages": {
            "M0_exact_counter": {"persona_geometry": 0.189336, "future_geometry": 0.172705},
            "M1_fold_only": {"persona_geometry": 0.065718, "future_geometry": 0.093234},
            "M4_served": {"persona_geometry": 0.065962, "future_geometry": 0.090950},
        },
        "attribution": {"persona_geometry": "M0_exact_counter->M1_fold_only"},
        "validation": {"m3_m4_distance_alignment": 0.998063},
    }
    _write(tmp_path, "encoder_stage_decomposition.json", body)
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.2"] == J.PASS
    assert "M0 0.1893" in md
    assert "M0_exact_counter->M1_fold_only" in md


def test_an_incomplete_stage_decomposition_is_not_pass(tmp_path):
    body = {"_meta": _meta(), "complete": False, "stages": {}, "attribution": {}}
    _write(tmp_path, "encoder_stage_decomposition.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.2"] == J.FAIL


def test_a_stale_stage_decomposition_is_unknown(tmp_path):
    body = {
        "_meta": _meta("oldhash"),
        "complete": True,
        "stages": {},
        "attribution": {},
        "validation": {},
    }
    _write(tmp_path, "encoder_stage_decomposition.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.2"] == J.UNKNOWN


def test_linear_access_receipt_can_pass_and_carries_the_delta(tmp_path):
    body = {
        "_meta": _meta(),
        "complete": True,
        "linear_access": {
            "complete": True,
            "arms": {
                "served": {"mean_r2": 0.4677, "n_fields": 11},
                "path_a": {"mean_r2": 0.4126, "n_fields": 11},
            },
        },
    }
    _write(tmp_path, "encoder_block_serving.json", body)
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.3"] == J.PASS
    assert "0.4677" in md and "0.4126" in md


def test_equal_dimension_receipt_can_pass_and_carries_the_blocks(tmp_path):
    body = {
        "_meta": _meta(),
        "complete": True,
        "equal_dimension": {
            "complete": True,
            "arms": {
                "equal_39": {"persona_geometry": 0.1721, "future_geometry": 0.1440},
                "equal_128": {"persona_geometry": 0.1361, "future_geometry": 0.1174},
                "counter_tile_39": {"persona_geometry": 0.1893, "future_geometry": 0.1727},
                "shape_pca_39": {"persona_geometry": 0.0814, "future_geometry": 0.0573},
                "path_a": {"persona_geometry": 0.3867, "future_geometry": 0.1810},
            },
        },
    }
    _write(tmp_path, "encoder_block_serving.json", body)
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.5"] == J.PASS
    assert "equal_39 0.1721" in md


def _served_object_body(persona_by_arm: dict, h: str = HASH, complete: bool = True) -> dict:
    arms = {
        n: {"persona_geometry": p, "future_geometry": None if p is None else p - 0.01}
        for n, p in persona_by_arm.items()
    }
    return {
        "_meta": _meta(h),
        "complete": complete,
        "ledger_arms": [
            n for n in persona_by_arm if n != "path_a" and "counter" not in n and "shape" not in n
        ],
        "reference_arms": ["exact_counter", "shape_served", "path_a"],
        "arms": arms,
        "linear_access": {
            "complete": True,
            "arms": {n: {"mean_r2": 0.1, "n_fields": 11} for n in persona_by_arm},
        },
    }


def test_served_object_receipt_passes_and_computes_the_branch(tmp_path):
    body = _served_object_body(
        {
            "ledger_full": 0.0573,
            "part_category_share": 0.0210,
            "part_numeric_stats": 0.3120,
            "ledger_pca_39": 0.1800,
            "ledger_pca_128": 0.2400,
            "exact_counter": 0.1893,
            "shape_served": 0.0660,
            "path_a": 0.3867,
        }
    )
    _write(tmp_path, "encoder_served_object.json", body)
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.6"] == J.PASS
    # the branch is computed from the arms, not trusted from the receipt: best ledger arm is
    # part_numeric_stats at 0.312 >= 0.30 -> exposure
    assert "branch exposure" in md
    assert "part_numeric_stats" in md


def test_served_object_branch_partial_when_best_arm_is_between_the_floors(tmp_path):
    body = _served_object_body(
        {
            "ledger_full": 0.1000,
            "part_category_share": 0.0500,
            "ledger_pca_39": 0.1600,
            "ledger_pca_128": 0.2200,
            "exact_counter": 0.1893,
            "shape_served": 0.0660,
            "path_a": 0.3867,
        }
    )
    _write(tmp_path, "encoder_served_object.json", body)
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.6"] == J.PASS
    assert "branch partial" in md
    assert "ledger_pca_128" in md


def test_served_object_branch_insufficient_when_no_ledger_arm_reaches_0_15(tmp_path):
    body = _served_object_body(
        {
            "ledger_full": 0.0573,
            "part_numeric_stats": 0.1200,
            "ledger_pca_128": 0.0900,
            "exact_counter": 0.1893,
            "shape_served": 0.0660,
            "path_a": 0.3867,
        }
    )
    _write(tmp_path, "encoder_served_object.json", body)
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.6"] == J.PASS
    assert "branch insufficient" in md


def test_an_incomplete_served_object_is_not_pass(tmp_path):
    body = _served_object_body({"ledger_full": 0.2, "path_a": 0.3867}, complete=False)
    _write(tmp_path, "encoder_served_object.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.6"] == J.FAIL


def test_a_stale_served_object_is_unknown(tmp_path):
    body = _served_object_body(
        {"ledger_full": 0.4, "exact_counter": 0.1893, "shape_served": 0.0660, "path_a": 0.3867},
        h="oldhash",
    )
    _write(tmp_path, "encoder_served_object.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.6"] == J.UNKNOWN


def test_served_object_without_any_finite_ledger_arm_is_unknown(tmp_path):
    body = _served_object_body(
        {"ledger_full": None, "part_numeric_stats": None, "path_a": 0.3867},
    )
    body["arms"] = {
        n: {"persona_geometry": p, "future_geometry": None} for n, p in body["arms"].items()
    }
    _write(tmp_path, "encoder_served_object.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B3.6"] == J.UNKNOWN


def test_the_plan_is_the_top_line_and_is_derived_from_the_b_rows(tmp_path, monkeypatch):
    """DEC-579: the plan tree replaced the sprint's capability-axis table as the top line.

    One axis definition only — the tree — so the card cannot say "done" on one view while the
    other is red. With no receipts every measured atom is UNKNOWN and the root is never PASS.
    The backward input is isolated (no review recorded -> 0 open requirements); the live
    AUDIT.md is the current project record, not a fixture.
    """
    monkeypatch.setattr(RC, "AUDIT_MD", tmp_path / "AUDIT.md")
    md, info = RC.render(tmp_path, HASH)
    assert "The plan — the tree this card measures" in md
    assert "B capability — top line" not in md
    assert info["plan"]["root"] == J.UNKNOWN
    assert info["plan"]["green"] == 0
    assert "cap_verdicts" not in info  # the retired duplicate view is gone, not aliased
    # every B row is an atomic problem exactly once: the plan cannot silently drop a row
    atoms = [n.ident for n in RC.P.NODES if n.kind == "atomic" and n.evidence == "card"]
    assert {r.ident for r in B.ROWS} <= set(atoms)


# ---------------------------------------------------------------------------
# B2 sufficiency rows (DEC-570): the bar lives in b_spec.SUFFICIENCY_BAR, never the receipt
# ---------------------------------------------------------------------------
def _suff_body(
    u_served: float | None = 0.9757,
    u_learned: float | None = 1.0769,
    probes: dict | None = None,
    complete: bool = True,
    h: str = HASH,
) -> dict:
    if probes is None:
        probes = {
            "next-type": {
                "included": True,
                "reason": "included",
                "loss": {
                    "served_object": 1.479336,
                    "learned_only": 1.463038,
                    "full": 1.475424,
                    "floor": 1.31443,
                },
                "u": {"served_object": u_served, "learned_only": u_learned},
            },
            "dt": {"included": False, "reason": "headroom -0.121934 <= 0 — probe excluded"},
        }
    return {
        "_meta": _meta(h),
        "complete": complete,
        "probes": probes,
        "u": {"served_object": u_served, "learned_only": u_learned},
        "u_all_arms": {"served_object": u_served, "learned_only": u_learned},
        "worst_probe": {"served_object": "next-type", "learned_only": "next-type"},
        "excluded_probes": {
            "dt": "headroom -0.121934 <= 0 — probe excluded, reported (degenerate)"
        },
    }


def test_sufficiency_u_passes_when_served_clears_the_frozen_bar(tmp_path):
    _write(tmp_path, "encoder_sufficiency_law.json", _suff_body(u_served=0.9757))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.1"] == J.PASS
    assert "served U 0.9757" in md
    assert f"(bar {B.SUFFICIENCY_BAR:g}, DEC-570)" in md


def test_sufficiency_u_fails_when_served_is_below_the_bar(tmp_path):
    _write(tmp_path, "encoder_sufficiency_law.json", _suff_body(u_served=0.31))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.1"] == J.FAIL


def test_sufficiency_u_fails_on_an_incomplete_receipt(tmp_path):
    _write(tmp_path, "encoder_sufficiency_law.json", _suff_body(complete=False))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.1"] == J.FAIL


def test_sufficiency_u_is_unknown_when_not_finite_or_stale(tmp_path):
    _write(tmp_path, "encoder_sufficiency_law.json", _suff_body(u_served=None))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.1"] == J.UNKNOWN
    _write(tmp_path, "encoder_sufficiency_law.json", _suff_body(h="oldhash"))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.1"] == J.UNKNOWN


def test_sufficiency_gap_passes_when_every_included_probe_clears_the_bar(tmp_path):
    body = _suff_body()
    body["probes"]["numeric-next"] = {
        "included": True,
        "reason": "included",
        "loss": {"served_object": 0.9, "full": 0.8, "floor": 0.5},
        "u": {"served_object": 0.6, "learned_only": 0.7},
    }
    _write(tmp_path, "encoder_sufficiency_law.json", body)
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.2"] == J.PASS


def test_sufficiency_gap_fails_when_an_included_probe_dips_below_the_bar(tmp_path):
    body = _suff_body()
    body["probes"]["numeric-next"] = {
        "included": True,
        "reason": "included",
        "loss": {"served_object": 1.0, "full": 0.8, "floor": 0.5},
        "u": {"served_object": 0.4, "learned_only": 0.5},
    }
    _write(tmp_path, "encoder_sufficiency_law.json", body)
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.2"] == J.FAIL
    assert "numeric-next" in md


def test_sufficiency_gap_fails_when_no_probe_has_headroom(tmp_path):
    body = _suff_body(probes={"dt": {"included": False, "reason": "headroom <= 0"}})
    _write(tmp_path, "encoder_sufficiency_law.json", body)
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.2"] == J.FAIL
    assert "unfilled" in md


def test_sufficiency_attribution_passes_when_learned_only_is_finite(tmp_path):
    _write(tmp_path, "encoder_sufficiency_law.json", _suff_body(u_learned=1.0769))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.3"] == J.PASS
    assert "learned-only U 1.0769" in md


def test_sufficiency_attribution_unknown_when_learned_only_missing(tmp_path):
    _write(tmp_path, "encoder_sufficiency_law.json", _suff_body(u_learned=None))
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B2.3"] == J.UNKNOWN
