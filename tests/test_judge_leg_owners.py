"""Rule 6 — one owner per receipt: the certified judge legs have exactly ONE writer.

`docs/report_cards/experiments/judge_leg_{supervised,unsupervised,rl,simulator}.json` are what
the JUDGE card's gate-of-record reads. Two producers once wrote those same four paths: the
bank-v2 judge (`measure_bc_bank.py`, DEC-598) and the older flat protocol
(`measure_judge_law.py`), so whichever ran last silently decided what the card certified — a
clean-tree re-stamp through the stale producer replaced bank-v2 numbers with v1 values and
flipped the plan root SOLVED 39/39 -> 35/39 (DEC-612, AUDIT B-021).

The rule enforced here: `measure_bc_bank.py` owns the certified names, and
`measure_judge_law.py` publishes its flat-era numbers under `judge_leg_v1flat_<leg>.json` and
cannot name a certified receipt at all. Static by construction — this reads source text and
never imports or executes either producer, so the guard does not depend on artifacts existing.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

CERTIFIED_LEGS = ("supervised", "unsupervised", "rl", "simulator")
V1_PRODUCER = "measure_judge_law.py"
GATE_PRODUCER = "measure_bc_bank.py"
DIAGNOSTIC_PREFIX = "judge_leg_v1flat_"


def _source(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def _compact(name: str) -> str:
    """Source with whitespace removed, so a literal survives reformatting (incl. line breaks)."""
    return "".join(_source(name).split())


def test_the_v1_flat_producer_cannot_name_a_certified_receipt():
    src = _source(V1_PRODUCER)
    for leg in CERTIFIED_LEGS:
        forbidden = f"judge_leg_{leg}.json"
        assert forbidden not in src, f"{V1_PRODUCER} names the certified receipt {forbidden!r}"
    # The old bug was the f-string `f"judge_leg_{leg}.json"`; any `judge_leg_` NOT followed by
    # the diagnostic prefix is a path this producer could write, and must not be able to.
    stray = re.findall(r"judge_leg_(?!v1flat_)", src)
    assert not stray, (
        f"{V1_PRODUCER} mentions `judge_leg_` outside the {DIAGNOSTIC_PREFIX!r} prefix "
        f"({len(stray)} occurrence(s)); the gate-of-record names belong to {GATE_PRODUCER}"
    )


def test_the_v1_flat_producer_writes_the_diagnostic_name():
    code = _compact(V1_PRODUCER)
    assert f'RECEIPT_PREFIX="{DIAGNOSTIC_PREFIX}"' in code
    assert 'out_dir/f"{RECEIPT_PREFIX}{leg}.json"' in code


def test_the_bank_v2_producer_owns_the_certified_names():
    code = _compact(GATE_PRODUCER)
    # One f-string path writes all four legs, so the template is the ownership claim.
    assert 'out_dir/f"judge_leg_{leg}.json"' in code
    assert DIAGNOSTIC_PREFIX not in code
