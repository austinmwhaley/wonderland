"""The judge card must be unable to lie (DEC-390, certification-integrity rules 3-5).

These tests are the falsifiable form of "the verdict is computed, not narrated": every route by
which a row could become PASS without real evidence is exercised and must come back UNKNOWN or
FAIL. A renderer that can be talked into a green row is worse than no renderer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_card as JC  # noqa: E402
import judge_spec as J  # noqa: E402

HASH = "abc123"


def _meta(h=HASH):
    return {
        "generated_at": "2026-10-28T00:00:00+00:00",
        "git_commit": "deadbeef",
        "behavior_hash": h,
    }


def _task(name="churn90", brain=0.78, pathA=0.7, lo=0.02, hi=0.14, n=6000, **kw):
    t = {
        "name": name,
        "brain": brain,
        "pathA": pathA,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": n,
        "random": 0.5,
        "destroyed": 0.51,
    }
    t.update(kw)
    return t


def _receipt(column_tasks, h=HASH, **extra):
    body = {"_meta": _meta(h), "columns": {c: {"tasks": list(t)} for c, t in column_tasks.items()}}
    body.update(extra)
    return body


def _write(tmp_path: Path, name: str, payload) -> Path:
    (tmp_path / name).write_text(json.dumps(payload))
    return tmp_path


def _good():
    tasks = [_task("churn90"), _task("value90", 0.61, 0.52, 0.03, 0.15)]
    return _receipt({"brain": tasks, "brain_ledger": tasks})


def test_honest_receipt_passes(tmp_path):
    _write(tmp_path, "judge_leg_supervised.json", _good())
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.PASS
    assert info["verdicts"]["JUDGE-supervised-brain_ledger"] == J.PASS


def test_missing_receipt_is_unknown_not_pass(tmp_path):
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN
    assert info["verdicts"]["JUDGE-rl-brain"] == J.UNKNOWN
    md, _ = JC.render(tmp_path, HASH)
    assert "judge_leg_supervised.json" in md  # names the exact missing receipt


def test_stale_behavior_hash_is_unknown(tmp_path):
    _write(tmp_path, "judge_leg_supervised.json", _good())
    _, info = JC.render(tmp_path, "different_hash")
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_receipt_without_meta_is_unknown(tmp_path):
    r = _good()
    r["_meta"].pop("behavior_hash")
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_receipt_may_not_define_its_own_criterion(tmp_path):
    r = _good()
    r["criterion"] = "PASS if it feels right"
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_no_pass_with_a_self_comparator(tmp_path):
    r = _good()
    r["columns"]["brain"]["tasks"][0]["comparator"] = "n/a"
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_no_pass_when_ci_includes_zero(tmp_path):
    r = _good()
    r["columns"]["brain"]["tasks"][0]["ci_lo"] = -0.01
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.FAIL


def test_no_pass_when_beats_a_weak_number_only(tmp_path):
    """brain > pathA numerically but the paired CI does not exclude zero -> FAIL, not PASS."""
    r = _good()
    for t in r["columns"]["brain"]["tasks"]:
        t["ci_lo"] = -0.001
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.FAIL


def test_no_pass_without_controls(tmp_path):
    r = _good()
    del r["columns"]["brain"]["tasks"][0]["destroyed"]
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_no_pass_when_a_control_beats_the_brain(tmp_path):
    r = _good()
    r["columns"]["brain"]["tasks"][0]["random"] = 0.99
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.FAIL


def test_degenerate_targets_are_excluded_not_forced(tmp_path):
    r = _good()
    for t in r["columns"]["brain"]["tasks"]:
        t["degenerate"] = True
        t["degenerate_reason"] = "target has no variance"
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.EXCLUDED
    assert info["summary"]["excluded"] >= 1


def test_non_finite_measurement_is_unknown(tmp_path):
    r = _good()
    r["columns"]["brain"]["tasks"][0]["brain"] = float("nan")
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_missing_n_is_unknown(tmp_path):
    r = _good()
    del r["columns"]["brain"]["tasks"][0]["n"]
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = JC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_stages_unknown_without_receipt(tmp_path):
    _, info = JC.render(tmp_path, HASH)
    assert all(info["verdicts"][s.ident] == J.UNKNOWN for s in J.STAGE_ROWS)


def test_stages_pass_only_when_reported_working(tmp_path):
    _write(
        tmp_path,
        "judge_stages.json",
        dict(
            {
                "_meta": _meta(),
                **{s.ident: {"working": True, "evidence": "x"} for s in J.STAGE_ROWS},
            }
        ),
    )
    _, info = JC.render(tmp_path, HASH)
    assert all(info["verdicts"][s.ident] == J.PASS for s in J.STAGE_ROWS)


def test_assessment_never_claims_done_without_green_rows(tmp_path):
    _, info = JC.render(tmp_path, HASH)
    assert "NOT DONE" in info["assessment"]
    assert "SOLVED" not in info["assessment"]
    assert "universal" not in info["assessment"].lower().replace("universality", "")


def test_assessment_claims_done_only_when_everything_is_green(tmp_path):
    for leg in J.LEGS:
        _write(tmp_path, f"judge_leg_{leg}.json", _good())
    _write(
        tmp_path,
        "judge_stages.json",
        dict(
            {
                "_meta": _meta(),
                **{s.ident: {"working": True, "evidence": "x"} for s in J.STAGE_ROWS},
            }
        ),
    )
    _write(tmp_path, "env_receipt.json", {"torch": {"matmul_smoke": {"ok": True}}})
    _, info = JC.render(tmp_path, HASH)
    assert "ENCODER DONE" in info["assessment"]


def test_behavior_hash_is_stable_and_content_sensitive(tmp_path):
    h1 = JC.behavior_hash()
    assert h1 == JC.behavior_hash()  # deterministic
