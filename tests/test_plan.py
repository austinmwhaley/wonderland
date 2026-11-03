"""PLAN.md -> REPORT_CARD.md (DEC-579): the tree is one definition, and it cannot lie.

The plan is the goal-side decomposition; the card is its measurement surface. These tests are
the falsifiable form of "the card cannot grow rows the plan does not contain, and the plan cannot
claim a state the card did not measure":

* every frozen B/judge row is a leaf exactly once (no orphan row, no duplicate leaf);
* the tree is acyclic, has one root, and every parent exists;
* the plan carries no gate/threshold of its own (gates live in the frozen specs);
* interior verdicts are DERIVED (worst-child), and the root cannot read PASS while a DEC-frozen
  or v1 leaf is unmeasured — T0 is not SOLVED;
* PLAN.md is current with the spec, so editing one without the other fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import b_spec as B  # noqa: E402
import judge_spec as J  # noqa: E402
import plan as PL  # noqa: E402
import plan_spec as P  # noqa: E402


def _all_leaves() -> list[str]:
    return [leaf for n in P.NODES for leaf in n.leaves]


# ------------------------------------------------------------------ structure integrity


def test_every_frozen_row_is_a_leaf_exactly_once():
    """The plan is the card's decomposition: no row may sit outside it, none may be counted twice."""
    leaves = _all_leaves()
    assert len(leaves) == len(set(leaves)), "a card row appears under two plan nodes"
    frozen = {r.ident for r in B.ROWS} | {r.ident for r in J.LEG_ROWS}
    assert set(leaves) == frozen, (
        f"plan/card mismatch: missing from plan {sorted(frozen - set(leaves))}, "
        f"unknown to the card {sorted(set(leaves) - frozen)}"
    )


def test_the_tree_is_acyclic_with_one_root_and_real_parents():
    roots = [n for n in P.NODES if n.parent is None]
    assert len(roots) == 1 and roots[0].ident == "G"
    assert len({n.ident for n in P.NODES}) == len(P.NODES), "duplicate node id"
    for n in P.NODES:
        if n.parent is not None:
            assert n.parent in P.BY_ID, f"{n.ident} has no parent"
    for n in P.NODES:
        assert P.level(n.ident) >= 0  # raises on a cycle


def test_every_node_is_reachable_from_the_goal():
    seen = set()

    def walk(i):
        seen.add(i)
        for c in P.children(i):
            walk(c.ident)

    walk("G")
    assert seen == set(P.BY_ID)


def test_the_plan_defines_no_gate_of_its_own():
    """Gates live in the frozen specs. A node with a threshold/criterion would be a new gate."""
    for n in P.NODES:
        for forbidden in ("gate", "thresh", "threshold", "criterion"):
            assert not hasattr(n, forbidden), f"{n.ident} carries its own {forbidden}"
    # every measured leaf resolves to a frozen row — asserted above; and every v1/dec leaf says
    # where its evidence comes from instead of pretending to be a card row
    for n in P.NODES:
        if n.evidence == "dec":
            assert n.dec, f"{n.ident} is DEC-frozen but names no DEC"
            assert not n.leaves
        if n.evidence == "v1":
            assert not n.leaves


def test_interior_nodes_are_derived_from_their_children():
    for n in P.NODES:
        kids = P.children(n.ident)
        if not kids:
            continue
        if n.evidence in ("dec", "v1"):
            continue
        for kid in kids:
            assert kid.ident in P.BY_ID


# ------------------------------------------------------------------ derived verdicts


def _green_leaves() -> dict[str, str]:
    return {r.ident: J.PASS for r in B.ROWS} | {r.ident: J.PASS for r in J.LEG_ROWS}


def test_all_green_rows_close_the_card_scope_but_not_the_plan_root():
    """T0 is not SOLVED: with every card leaf green, the root still cannot read PASS while the
    DEC-frozen port leaf and the v1 leaves are unmeasured."""
    st = P.statuses(_green_leaves())
    assert st["B2"] == J.PASS and st["P4"] == J.PASS
    assert st["E1"] == P.FRONTIER
    assert st["V1.1"] == P.NOT_ASSESSED
    assert st["G"] == J.UNKNOWN


def test_the_root_is_pass_only_when_every_scope_is_closed():
    closed = {n.ident for n in P.NODES if n.evidence in ("dec", "v1")}
    st = P.statuses(_green_leaves(), closed=closed)
    assert st["G"] == J.PASS


def test_one_red_leaf_fails_every_ancestor():
    rows = _green_leaves()
    rows["B2.1"] = J.FAIL
    st = P.statuses(rows)
    assert st["B2"] == J.FAIL and st["P2"] == J.FAIL and st["G"] == J.FAIL


def test_a_missing_row_is_unknown_never_pass():
    rows = _green_leaves()
    del rows["B3.2"]
    st = P.statuses(rows)
    assert st["B3"] == J.UNKNOWN and st["P3"] == J.UNKNOWN


# ------------------------------------------------------------------ the renders agree


def test_plan_md_is_current_with_the_spec():
    """`PLAN.md` is generated: editing the spec without regenerating the doc fails here."""
    p = ROOT / "PLAN.md"
    assert p.exists(), "PLAN.md missing — run `python3 scripts/plan.py`"
    assert p.read_text() == PL.render_plan(), "PLAN.md is stale — run `python3 scripts/plan.py`"


def test_plan_md_contains_every_node_and_no_measurements():
    text = PL.render_plan()
    for n in P.NODES:
        assert f"{n.ident} — {n.title}" in text, f"{n.ident} missing from PLAN.md"
    # the plan is structure only: no verdict words that would go stale with receipts
    for word in ("✅ PASS", "❌ FAIL", "❔ UNKNOWN", "served U "):
        assert word not in text, f"PLAN.md carries a measurement ({word})"


def test_the_card_renders_the_same_tree_with_verdicts(tmp_path):
    import report_card as RC

    md, info = RC.render(tmp_path, "abc123")
    assert "The plan — the tree this card measures" in md
    plan = info["plan"]
    assert plan["root"] == J.UNKNOWN  # no receipts, nothing measured -> never green
    assert plan["card_leaves"] == len(_all_leaves())
    assert plan["v1"] == 5 and plan["dec"] == 1
    for n in P.NODES:
        assert n.ident in plan["nodes"]
        assert f"**{n.ident}**" in md
