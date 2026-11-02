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
    assert b["missing"] >= 15
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
