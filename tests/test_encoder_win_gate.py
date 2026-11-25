"""The B9.1 outcome gate (`encoder_win`) must be unable to lie.

The gate is the one row that certifies the encoder's OUTCOME (it beats the 1-gram baseline on
the customer event stream), so every way a receipt can fail to be evidence — a stale behavior
stamp, a missing paired CI, a population mismatch, a claim that is simply false — must render
as UNKNOWN/FAIL, never PASS. The branch previously had zero tests.

The producer's provenance is tested here too: the certified receipt was measured with the
multinomial logistic head while its protocol prose named the ridge head, and the CLI default
was the ridge. The default and the prose are now functions of the measured head.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_spec as J  # noqa: E402
import measure_encoder_win as M  # noqa: E402
import report_card as RC  # noqa: E402

HASH = "abc123"


def _meta(h: str = HASH) -> dict:
    return {
        "generated_at": "2026-11-25T00:00:00+00:00",
        "git_commit": "deadbeef",
        "behavior_hash": h,
    }


def _write(tmp_path: Path, name: str, payload) -> None:
    (tmp_path / name).write_text(json.dumps(payload))


def _win_body(
    *,
    beats: bool = True,
    ci_lo: float | None = -0.290842,
    ci_hi: float | None = -0.211559,
    complete: bool = True,
    block_complete: bool = True,
    h: str = HASH,
) -> dict:
    """A synthetic B9.1 receipt in the shape `measure_encoder_win.py` writes."""
    return {
        "_meta": _meta(h),
        "kind": "encoder_win",
        "complete": complete,
        "encoder_win": {
            "complete": block_complete,
            "beats_unigram": beats,
            "paired_ci_enc_minus_base": [ci_lo, ci_hi],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "nll_enc": 1.173099,
            "nll_base": 1.423192,
            "n_A": 3500,
            "n_B_rows": 1500,
            "head": "logistic",
        },
        "population": {"demo_customers": 5000, "products_n_customers": 5000, "match": True},
        "products": {"path": "p", "tag": "0.11.1r490760-test", "registry_sha256": "abc"},
    }


def test_encoder_win_passes_when_it_beats_the_1gram_with_a_ci_below_zero(tmp_path):
    _write(tmp_path, "encoder_win.json", _win_body())
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B9.1"] == J.PASS
    assert "beats_unigram=True" in md
    assert "paired CI [-0.2908, -0.2116] < 0" in md


def test_encoder_win_fails_when_it_does_not_beat_the_1gram(tmp_path):
    _write(tmp_path, "encoder_win.json", _win_body(beats=False, ci_lo=0.0016, ci_hi=0.0554))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B9.1"] == J.FAIL
    assert "does NOT beat the 1-gram baseline" in md


def test_encoder_win_fails_when_the_paired_ci_still_includes_zero(tmp_path):
    _write(tmp_path, "encoder_win.json", _win_body(ci_lo=-0.02, ci_hi=0.0))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B9.1"] == J.FAIL
    assert "includes 0" in md


def test_encoder_win_is_unknown_when_the_paired_ci_is_missing(tmp_path):
    _write(tmp_path, "encoder_win.json", _win_body(ci_lo=None, ci_hi=None))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B9.1"] == J.UNKNOWN
    assert "paired CI values missing" in md


def test_encoder_win_is_unknown_when_the_behavior_hash_is_stale(tmp_path):
    _write(tmp_path, "encoder_win.json", _win_body(h="stalehash"))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B9.1"] == J.UNKNOWN
    assert "stale" in md
    assert "measure_encoder_win.py" in md


def test_encoder_win_fails_when_the_population_guard_marks_it_incomplete(tmp_path):
    """A population mismatch is written `complete=false` (DEC-650): it is not a comparison."""
    _write(tmp_path, "encoder_win.json", _win_body(complete=False, beats=False))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B9.1"] == J.FAIL
    assert "encoder-win receipt incomplete" in md


def test_encoder_win_fails_when_the_encoder_win_block_is_incomplete(tmp_path):
    _write(tmp_path, "encoder_win.json", _win_body(block_complete=False))
    md, info = RC.render(tmp_path, HASH)
    assert info["b_verdicts"]["B9.1"] == J.FAIL
    assert "encoder-win block incomplete" in md


# ---------------------------------------------------------------------------
# producer provenance: the default and the protocol prose are the measured head's
# ---------------------------------------------------------------------------
def test_the_default_head_reproduces_the_certified_receipt():
    ap = M._parser()
    assert ap.parse_args([]).head == "logistic"
    assert ap.parse_args(["--head", "ridge"]).head == "ridge"  # ridge stays selectable


def test_the_protocol_text_names_the_measured_head_and_calibration():
    logistic = M._protocol_text("logistic", "none", 0)
    assert "DEC-635" in logistic
    assert "multinomial logistic" in logistic
    assert "ridge" not in logistic
    assert "no post-hoc temperature" in logistic
    assert "last-" not in logistic  # no recent-k block was served

    ridge = M._protocol_text("ridge", "vector", 3)
    assert "ridge" in ridge
    assert "vector scaling" in ridge
    assert "last-3 event-type indicators" in ridge
    assert logistic != ridge
