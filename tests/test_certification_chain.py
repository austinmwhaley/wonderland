"""S6's verdict: did the chain RUN, as distinct from did it measure everything green.

The distinction is the whole point of a certification card. A producer that reports a failing
check is doing its job — `measure_views_law` documents `0 if all working else 1`. A gate that
demanded exit 0 would push every producer to report green in order to look healthy, which is the
incentive the card exists to remove. Conversely, a producer that died without writing its receipt
must not pass just because the exit code looked familiar.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import measure_certification_law as MC  # noqa: E402

STAGES = set(MC.STAGES)
LEGS = {"supervised", "unsupervised", "rl", "simulator"}


def _run(exits, stages=None, legs=None):
    return MC.chain_verdict(
        exits,
        STAGES if stages is None else stages,
        STAGES,
        LEGS if legs is None else legs,
        LEGS,
        n_producers=len(MC.CHAIN),
    )


def test_a_failing_check_is_not_a_broken_chain():
    """Exactly the case that made S6 red on its first real run: S4 fails its gate and exits 1.

    S4's failure is a MEASUREMENT (the response outlet does not beat its attribution gate). If S6
    went red for it, the card would be saying "we could not regenerate" when the truth is "we
    regenerated, and here is a red row".
    """
    ok, detail = _run({"S1+S2": 0, "S3": 0, "S4": 1, "S5": 1, "legs": 0})
    assert ok is True
    assert detail["reported_failing"] == ["S4", "S5"]
    assert detail["fatal"] == {}


def test_a_producer_that_could_not_run_is_a_broken_chain():
    """Exit >= 2 means "could not run at all" and must fail the chain."""
    ok, detail = _run({"S1+S2": 0, "S3": 2, "S4": 0, "S5": 0, "legs": 0})
    assert ok is False
    assert detail["fatal"] == {"S3": 2}


def test_a_producer_that_did_not_refresh_its_receipt_is_a_broken_chain():
    """The artifact is the evidence. A zero exit with a stale receipt proves nothing ran."""
    ok, detail = _run(
        {"S1+S2": 0, "S3": 0, "S4": 0, "S5": 0, "legs": 0},
        stages=STAGES - {"S3"},
    )
    assert ok is False
    assert "S3" in detail["missing"]


def test_a_missing_leg_receipt_is_a_broken_chain():
    """The top line is four legs: a run that regenerated three has not regenerated the card."""
    ok, detail = _run(
        {"S1+S2": 0, "S3": 0, "S4": 0, "S5": 0, "legs": 0},
        legs=LEGS - {"simulator"},
    )
    assert ok is False
    assert "simulator" in detail["missing"]


def test_everything_green_is_green():
    ok, detail = _run({"S1+S2": 0, "S3": 0, "S4": 0, "S5": 0, "legs": 0})
    assert ok is True
    assert detail == {"fatal": {}, "missing": [], "reported_failing": []}
