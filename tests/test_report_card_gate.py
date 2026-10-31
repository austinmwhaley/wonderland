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

import report_card as RC  # noqa: E402
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


def _stage_blocks(h=HASH, producer="measure_input_law.py"):
    """Stage blocks as a producer actually writes them: each carries its OWN stamp.

    `judge_stages.json` is a merge of four producers, so the file-level `_meta` belongs to
    whoever wrote last — a block without a stamp of its own cannot claim currency (rules 6, 7).
    """
    stamp = dict(_meta(h), command=f"{producer} --products x", producer=producer)
    return {
        s.ident: {"working": True, "evidence": "x", "_stamp": dict(stamp)} for s in J.STAGE_ROWS
    }


def test_honest_receipt_passes(tmp_path):
    _write(tmp_path, "judge_leg_supervised.json", _good())
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.PASS
    assert info["verdicts"]["JUDGE-supervised-brain_ledger"] == J.PASS


def test_missing_receipt_is_unknown_not_pass(tmp_path):
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN
    assert info["verdicts"]["JUDGE-rl-brain"] == J.UNKNOWN
    md, _ = RC.render(tmp_path, HASH)
    assert "judge_leg_supervised.json" in md  # names the exact missing receipt


def test_stale_behavior_hash_is_unknown(tmp_path):
    _write(tmp_path, "judge_leg_supervised.json", _good())
    _, info = RC.render(tmp_path, "different_hash")
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_receipt_without_meta_is_unknown(tmp_path):
    r = _good()
    r["_meta"].pop("behavior_hash")
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_receipt_may_not_define_its_own_criterion(tmp_path):
    r = _good()
    r["criterion"] = "PASS if it feels right"
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_no_pass_with_a_self_comparator(tmp_path):
    r = _good()
    r["columns"]["brain"]["tasks"][0]["comparator"] = "n/a"
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_no_pass_when_ci_includes_zero(tmp_path):
    r = _good()
    r["columns"]["brain"]["tasks"][0]["ci_lo"] = -0.01
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.FAIL


def test_no_pass_when_beats_a_weak_number_only(tmp_path):
    """brain > pathA numerically but the paired CI does not exclude zero -> FAIL, not PASS."""
    r = _good()
    for t in r["columns"]["brain"]["tasks"]:
        t["ci_lo"] = -0.001
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.FAIL


def test_no_pass_without_controls(tmp_path):
    r = _good()
    del r["columns"]["brain"]["tasks"][0]["destroyed"]
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_no_pass_when_a_control_beats_the_brain(tmp_path):
    r = _good()
    r["columns"]["brain"]["tasks"][0]["random"] = 0.99
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.FAIL


def test_degenerate_targets_are_excluded_not_forced(tmp_path):
    r = _good()
    for t in r["columns"]["brain"]["tasks"]:
        t["degenerate"] = True
        t["degenerate_reason"] = "target has no variance"
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.EXCLUDED
    assert info["summary"]["excluded"] >= 1


def test_non_finite_measurement_is_unknown(tmp_path):
    r = _good()
    r["columns"]["brain"]["tasks"][0]["brain"] = float("nan")
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_missing_n_is_unknown(tmp_path):
    r = _good()
    del r["columns"]["brain"]["tasks"][0]["n"]
    _write(tmp_path, "judge_leg_supervised.json", r)
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.UNKNOWN


def test_stages_unknown_without_receipt(tmp_path):
    _, info = RC.render(tmp_path, HASH)
    assert all(info["verdicts"][s.ident] == J.UNKNOWN for s in J.STAGE_ROWS)


def test_stages_pass_only_when_reported_working(tmp_path):
    _write(
        tmp_path,
        "judge_stages.json",
        dict({"_meta": _meta(), **_stage_blocks()}),
    )
    _, info = RC.render(tmp_path, HASH)
    assert all(info["verdicts"][s.ident] == J.PASS for s in J.STAGE_ROWS)


def test_a_stage_measured_against_other_behavior_cannot_be_green(tmp_path):
    """A merged file must not let one producer vouch for another's evidence (rules 6, 7).

    The real defect this pins: `judge_stages.json` carried one `_meta` for five blocks written
    by four scripts, so re-running S4's producer stamped S1/S2/S3/S5 — measured days earlier
    against `/tmp` inputs that no longer exist — as current.
    """
    blocks = _stage_blocks(h="stalehash")
    _write(tmp_path, "judge_stages.json", dict({"_meta": _meta(), **blocks}))
    _, info = RC.render(tmp_path, HASH)
    assert all(info["verdicts"][s.ident] == J.UNKNOWN for s in J.STAGE_ROWS), info["verdicts"]
    # and the reason names who measured it and against what, so the fix is a command not a guess
    md, _info = RC.render(tmp_path, HASH)
    assert "stalehash" in md and "measure_input_law.py" in md


def test_a_stage_block_with_no_stamp_of_its_own_cannot_be_green(tmp_path):
    """A block that cannot say who measured it, or against which build, is not evidence."""
    untracked = {s.ident: {"working": True, "evidence": "x"} for s in J.STAGE_ROWS}
    _write(tmp_path, "judge_stages.json", dict({"_meta": _meta(), **untracked}))
    _, info = RC.render(tmp_path, HASH)
    assert all(info["verdicts"][s.ident] == J.UNKNOWN for s in J.STAGE_ROWS)


def test_a_block_that_denies_its_own_checks_cannot_be_green(tmp_path):
    """Self-consistency (rule 7): `working: true` may not outrank the checks recorded beside it.

    The frozen criterion is still the block's `working` flag — this refuses only a block whose own
    receipts contradict it, which is a producer defect the reader must not launder into a PASS.
    """
    blocks = _stage_blocks()
    first = J.STAGE_ROWS[0].ident
    blocks[first] = dict(
        blocks[first],
        working=True,
        checks={"the_law": {"working": False, "evidence": "F(g∘f) != F(g)∘F(f)"}},
    )
    _write(tmp_path, "judge_stages.json", dict({"_meta": _meta(), **blocks}))
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"][first] == J.UNKNOWN
    md, _ = RC.render(tmp_path, HASH)
    assert "self-contradictory" in md and "the_law" in md


def test_a_red_stage_names_the_check_that_failed(tmp_path):
    """Diagnosability (principle 8): a red row must name its failing check, not its first check.

    Real defect: S4/S5 rendered `[outlet_schema] OK: ...` — the summary began with a passing
    check, so a red stage visually read as a green one.
    """
    blocks = _stage_blocks()
    ident = J.STAGE_ROWS[0].ident
    blocks[ident] = dict(
        blocks[ident],
        working=False,
        evidence="[good] OK: looks fine | [bad] FAILED: the law did not hold",
        checks={
            "good": {"working": True, "evidence": "looks fine"},
            "bad": {"working": False, "evidence": "the law did not hold"},
        },
    )
    _write(tmp_path, "judge_stages.json", dict({"_meta": _meta(), **blocks}))
    md, info = RC.render(tmp_path, HASH)
    assert info["verdicts"][ident] == J.FAIL
    assert "FAILED bad: the law did not hold" in md
    assert "[good] OK" not in md


def test_assessment_never_claims_done_without_green_rows(tmp_path):
    _, info = RC.render(tmp_path, HASH)
    assert "NOT DONE" in info["assessment"]
    assert "SOLVED" not in info["assessment"]
    assert "universal" not in info["assessment"].lower().replace("universality", "")


def test_assessment_claims_done_only_when_everything_is_green(tmp_path):
    for leg in J.LEGS:
        _write(tmp_path, f"judge_leg_{leg}.json", _good())
    _write(
        tmp_path,
        "judge_stages.json",
        dict({"_meta": _meta(), **_stage_blocks()}),
    )
    _write(tmp_path, "env_receipt.json", {"torch": {"matmul_smoke": {"ok": True}}})
    _, info = RC.render(tmp_path, HASH)
    assert "ENCODER DONE" in info["assessment"]


def test_behavior_hash_is_stable_and_content_sensitive(tmp_path):
    h1 = RC.behavior_hash()
    assert h1 == RC.behavior_hash()  # deterministic


def test_the_card_can_report_a_clean_tree(tmp_path):
    """Certification rule 6 asks for a card regenerated at a CLEAN recorded commit.

    The card is a committed generated artifact, so before this fix every render counted its own
    uncommitted bytes and printed `dirty` forever — a flag that can never tell the truth trains
    readers to ignore it. Anything else in the tree must still make it dirty.
    """
    assert RC._worktree_dirty("") is False
    assert RC._worktree_dirty(f"M {RC.CARD_NAME}") is False
    assert (
        RC._worktree_dirty(f" M {RC.CARD_NAME}\n") is False
    )  # real porcelain lines have a lead space
    assert RC._worktree_dirty(f"?? {RC.CARD_NAME}\n M looking_glass/cfm_state.py") is True
    assert RC._worktree_dirty("?? docs/report_cards/experiments/judge_leg_rl.json") is True


def _ident(leg, column):
    """Row identity as the SPEC defines it, so a spec rename cannot silently pass a test."""
    return next(r.ident for r in J.LEG_ROWS if r.leg == leg and r.column == column)


# ------------------------------------------------------------------ H4 vacuity (DEC-433)


def test_a_pass_may_not_rest_on_a_control_that_could_never_have_won(tmp_path):
    """H4 is satisfied only by a control that COULD have caught the row.

    Plain language: a negative control is a deliberately-broken copy of the input. If the model
    predicts well from that too, the harness is fooling itself. A control at R2 = -5000 was never
    in the running, so "we beat it" proves the head is not insane — not that the embedding carries
    signal. Before this, such a row could go green on a gate that could not have gone red.
    """
    absurd = [
        _task(
            "churn90", brain=0.80, pathA=0.60, lo=0.05, hi=0.20, random=-5000.0, destroyed=-9000.0
        )
    ]
    _write(
        tmp_path, "judge_leg_supervised.json", _receipt({"brain": absurd, "brain_ledger": absurd})
    )
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"][_ident(J.LEG_SUPERVISED, "brain")] == J.UNKNOWN
    md, _ = RC.render(tmp_path, HASH)
    assert "H4 is vacuous" in md


def test_one_readable_control_is_enough_for_a_pass(tmp_path):
    """The rule refuses VACUITY, not difficulty: a destroyed control at 0.4 is a real test.

    `random` is a gaussian draw and is distribution-mismatched by construction on wide designs
    (it legitimately lands far below zero); `destroyed` permutes each column and keeps its
    marginals, so it is the informative one. Requiring *one* readable control keeps the gate
    honest without demanding the impossible from the gaussian arm.
    """
    mixed = [
        _task("churn90", brain=0.80, pathA=0.60, lo=0.05, hi=0.20, random=-5000.0, destroyed=0.40)
    ]
    _write(tmp_path, "judge_leg_supervised.json", _receipt({"brain": mixed, "brain_ledger": mixed}))
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"][_ident(J.LEG_SUPERVISED, "brain")] == J.PASS


def test_a_real_failure_outranks_a_vacuous_control(tmp_path):
    """Ordering matters: a row that loses to PATH A must read FAIL, not hide behind UNKNOWN.

    Without this precedence, adding the vacuity rule would have relabelled the supervised leg's
    honest FAIL (5 traits not beating the incumbent) as an unmeasured gap — the one way a
    tightening could accidentally make the card look less committed than it is.
    """
    losing = [
        _task(
            "churn90", brain=0.40, pathA=0.60, lo=-0.05, hi=0.02, random=-5000.0, destroyed=-9000.0
        )
    ]
    _write(
        tmp_path, "judge_leg_supervised.json", _receipt({"brain": losing, "brain_ledger": losing})
    )
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"][_ident(J.LEG_SUPERVISED, "brain")] == J.FAIL


def test_an_empty_porcelain_means_clean_not_dirty():
    """`git()` returns "?" when a command printed nothing; empty status IS the clean tree.

    Observed on the real repo: with the card committed and nothing else modified, the card
    rendered `dirty` — because the sentinel for "no output" was counted as a changed file. That is
    the DEC-430 defect in its worse direction: not a flag that is always red, but one whose honest
    value (clean) is unreachable, so a reader learns to ignore the single signal rule 6 has.
    """
    assert RC._worktree_dirty("") is False
    assert RC._worktree_dirty("?") is False
    # the card's own bytes never count: a render is allowed to rewrite them
    assert RC._worktree_dirty(" M REPORT_CARD.md") is False
    # a source edit does, and that is the whole point of the check
    assert RC._worktree_dirty(" M scripts/report_card.py") is True
    assert RC._worktree_dirty("?? notes.md") is True


def _all_green(tmp_path):
    for leg in J.LEGS:
        _write(tmp_path, f"judge_leg_{leg}.json", _good())
    _write(tmp_path, "judge_stages.json", dict({"_meta": _meta(), **_stage_blocks()}))
    _write(tmp_path, "env_receipt.json", {"torch": {"matmul_smoke": {"ok": True}}})


def test_an_open_breaking_backward_requirement_blocks_done(tmp_path, monkeypatch):
    """DEC-497: BACKWARD findings are card requirements, not prose beside the card."""
    _all_green(tmp_path)
    req = tmp_path / "AUDIT.md"
    req.write_text(
        "<!-- BACKWARD_REQUIREMENTS_JSON\n"
        + json.dumps(
            {
                "requirements": [
                    {
                        "id": "B-001",
                        "severity": "BREAKING",
                        "principle": "P4/E16 (SPEED)",
                        "status": "open",
                        "finding": "the comparator leaks a future column",
                        "forward_subgoal": "rebuild the comparator without the future column",
                        "audit_ref": "A-001",
                    }
                ]
            }
        )
        + "\n-->\n"
    )
    monkeypatch.setattr(RC, "AUDIT_MD", req)
    md, info = RC.render(tmp_path, HASH)
    assert "ENCODER DONE" not in info["assessment"]
    assert "backward open: 1" in info["assessment"]
    assert "B-001" in md
    assert "P4/E16" in md and "A-001" in md
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.PASS  # rows stay what they are


def test_a_closed_backward_requirement_does_not_block_done(tmp_path, monkeypatch):
    _all_green(tmp_path)
    req = tmp_path / "AUDIT.md"
    req.write_text(
        "<!-- BACKWARD_REQUIREMENTS_JSON\n"
        + json.dumps(
            {
                "requirements": [
                    {
                        "id": "B-001",
                        "severity": "BREAKING",
                        "status": "closed",
                        "finding": "fixed",
                        "forward_subgoal": "done",
                    }
                ]
            }
        )
        + "\n-->\n"
    )
    monkeypatch.setattr(RC, "AUDIT_MD", req)
    _, info = RC.render(tmp_path, HASH)
    assert "ENCODER DONE" in info["assessment"]
    assert info["summary"]["backward_open"] == 0


def test_an_unreadable_backward_file_is_breaking_not_empty(tmp_path, monkeypatch):
    _all_green(tmp_path)
    req = tmp_path / "AUDIT.md"
    req.write_text("<!-- BACKWARD_REQUIREMENTS_JSON\n{not json\n-->\n")
    monkeypatch.setattr(RC, "AUDIT_MD", req)
    _, info = RC.render(tmp_path, HASH)
    assert "ENCODER DONE" not in info["assessment"]
    assert "backward open: 1" in info["assessment"]


def test_delivery_is_non_inferiority_not_superiority(tmp_path):
    """DEC-506: delivery = never materially worse, progress = a strict win."""
    brain = [_task("x0", brain=0.90, pathA=0.92, lo=-0.02, hi=0.01, n=724, sst=1.0)]
    delivery = [_task("x0", brain=0.915, pathA=0.92, lo=-0.004, hi=0.02, n=724, sst=1.0)]
    _write(
        tmp_path,
        "judge_leg_supervised.json",
        _receipt({"brain": brain, "brain_ledger": delivery}),
    )
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain"] == J.FAIL  # strict win: CI includes 0
    assert info["verdicts"]["JUDGE-supervised-brain_ledger"] == J.PASS  # within margin

    outside = [_task("x0", brain=0.90, pathA=0.92, lo=-0.02, hi=0.01, n=724, sst=1.0)]
    _write(
        tmp_path,
        "judge_leg_supervised.json",
        _receipt({"brain": brain, "brain_ledger": outside}),
    )
    _, info = RC.render(tmp_path, HASH)
    assert info["verdicts"]["JUDGE-supervised-brain_ledger"] == J.FAIL  # outside the margin
