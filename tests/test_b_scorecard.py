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


def test_legacy_b_receipt_is_unknown_not_pass(tmp_path):
    _write(tmp_path, "encoder_lawfulness.json", {"scan_merge_delta": 1e-13})
    _, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B1.2"] == J.UNKNOWN
    md, _ = RC.render(tmp_path, HASH)
    assert "legacy receipt" in md


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


def test_capability_top_line_is_derived_from_the_b_rows(tmp_path):
    # no receipts at all: every axis UNKNOWN, and the table is present with the frozen axes
    md, info = RC.render(tmp_path, HASH)
    assert "B capability — top line" in md
    assert info["cap_verdicts"]["U · sufficiency"] == J.UNKNOWN
    assert info["cap_verdicts"]["shape-only geometry"] == J.UNKNOWN
    assert info["cap_all_green"] is False
    assert "measure_sufficiency_law.py" in md
    # the top line must list every frozen axis from b_spec
    for axis, _ in B.CAPABILITY_AXES:
        assert axis in info["cap_verdicts"]


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
