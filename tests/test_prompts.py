"""The operating prompts must drive the plan->card loop, not drift back to row-chasing.

DEC-580 (operator directive): PROMPT_FORWARD and PROMPT_BACKWARD work directly with
`PLAN.md` -> `REPORT_CARD.md` at two scales. These assertions are a drift ratchet: the prompts
may be edited, but they may not lose the MACRO contract (a named plan node, the route, the
node-movement read) or fall back to the retired micro-only instruction.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plan_spec as P  # noqa: E402


def _read(name: str) -> str:
    return (ROOT / name).read_text()


def test_forward_prompt_declares_the_macro_contract():
    t = _read("PROMPT_FORWARD.md")
    # the three-line declaration, before anything is chosen
    for token in ("NODE:", "ROUTE:", "MOVE:"):
        assert token in t, f"PROMPT_FORWARD lost its {token} line"
    # the read at both scales
    assert "MOVEMENT:" in t
    # the quality/reporting distinction is part of node selection
    assert "Green means" in t and "reporting" in t
    # the map is generated and checked, not hand-edited
    assert "PLAN.md" in t and "scripts/plan.py --check" in t
    assert "scripts/plan_spec.py" in t


def test_forward_prompt_requires_the_macro_reflection():
    """DEC-587 (operator directive): every forward run first answers 'why isn't the project
    successful yet?', names the layer the gap lives in, and re-answers the outcome."""
    t = _read("PROMPT_FORWARD.md")
    for token in ("WHY-NOT:", "EVIDENCE:", "GAP:", "LAYER:", "THIS RUN:"):
        assert token in t, f"PROMPT_FORWARD lost its {token} line"
    assert "WHY-NOT-CHANGED:" in t
    # the reflection precedes node selection and names the four layers
    assert t.index("WHY-NOT:") < t.index("NODE:")
    for layer in ("capability", "objective", "instrument", "identity"):
        assert layer in t


def test_forward_prompt_documents_the_plan_first_card_shape():
    t = _read("PROMPT_FORWARD.md")
    assert "The plan — the tree this card measures" in t
    assert t.index("The plan — the tree this card measures") < t.index("The judge — four legs")


def test_backward_prompt_attacks_the_tree_before_the_rows():
    t = _read("PROMPT_BACKWARD.md")
    assert "The tree (MACRO)" in t
    assert "Macro verdict" in t
    assert "plan_spec.statuses" in t  # recompute, do not trust the render
    assert "Green means" in t  # reporting-vs-quality misuse is a finding
    assert "NODE:<node>" in t  # the scoped mode
    # both prompts must name the two modes
    assert "FULL" in t and "NODE" in t


def test_prompts_reference_real_plan_nodes_and_route_steps():
    fwd, bwd = _read("PROMPT_FORWARD.md"), _read("PROMPT_BACKWARD.md")
    for token in ("P1", "P2", "P5", "E1", "R1", "G"):
        assert token in fwd or token in bwd, f"prompts reference no {token}"
    # the worked example's node is the DEC-frozen trunk atom (E1 cut to E1.a–c, DEC-584)
    assert P.BY_ID["E1.a"].evidence == "dec"
    assert P.CURRENT_ROUTE in {s.ident for s in P.ROUTE}


def test_prompts_do_not_regress_to_the_retired_micro_only_wording():
    fwd = _read("PROMPT_FORWARD.md")
    bwd = _read("PROMPT_BACKWARD.md")
    # the retired capability-axis top line and the old flat selection rule
    assert "B capability — top line" not in fwd
    assert "highest-leverage unmet subgoal" not in fwd
    # BACKWARD's old single-level claim ("attack two things separately")
    assert "Attack two things separately" not in bwd
