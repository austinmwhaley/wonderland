"""PLAN.md -> REPORT_CARD.md (DEC-579; DEC-584 atomic reframe): the tree is one definition,
and it cannot lie.

The plan is the goal-side decomposition, cut to atomic problems; the card is its measurement
surface. These tests are the falsifiable form of "the card cannot grow rows the plan does not
contain, and the plan cannot claim a state the card did not measure":

* every frozen B/judge row is an atomic problem exactly once (no orphan row, no duplicate atom);
* the tree is acyclic, has one root, and every parent exists;
* the plan carries no gate/threshold of its own (gates live in the frozen specs);
* interior verdicts are DERIVED (worst-child), and the root cannot read PASS while a DEC-frozen
  or v1 atom is unmeasured — T0 is not SOLVED;
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


def _card_atoms() -> list[str]:
    """The atomic problems the card measures: one per frozen B/judge row."""
    return [n.ident for n in P.NODES if n.kind == "atomic" and n.evidence == "card"]


# ------------------------------------------------------------------ structure integrity


def test_every_frozen_row_is_an_atomic_problem_exactly_once():
    """The plan is the card's decomposition: no row may sit outside it, none may be counted twice."""
    atoms = _card_atoms()
    assert len(atoms) == len(set(atoms)), "a card row appears under two plan nodes"
    frozen = {r.ident for r in B.ROWS} | {r.ident for r in J.LEG_ROWS}
    assert set(atoms) == frozen, (
        f"plan/card mismatch: missing from plan {sorted(frozen - set(atoms))}, "
        f"unknown to the card {sorted(set(atoms) - frozen)}"
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
    # every measured atom resolves to a frozen row — asserted above; and every v1/dec atom says
    # where its evidence comes from instead of pretending to be a card row
    for n in P.NODES:
        if n.evidence == "dec":
            assert n.dec, f"{n.ident} is DEC-frozen but names no DEC"
    # an interior node is judged ONLY by its children: its evidence kind stays "card" so the
    # derived verdict can never be short-circuited to a fixed badge
    for n in P.NODES:
        if n.kind != "atomic":
            assert n.evidence == "card", f"interior {n.ident} carries its own evidence kind"


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
    assert plan["card"] == 31 and plan["dec"] == 3 and plan["v1"] == 5
    assert plan["atomic_total"] == plan["card"] + plan["dec"] + plan["v1"]
    for n in P.NODES:
        assert n.ident in plan["nodes"]
        # card atoms render in their B/judge section, not in the plan table; every other node
        # carries its own bolded row there
        if n.kind == "atomic" and n.evidence == "card":
            continue
        assert f"**{n.ident}**" in md, f"{n.ident} missing from the card's plan table"


# ------------------------------------------------------------------ what a green proves


def test_every_b_row_declares_what_a_green_proves():
    """DEC-579: quality vs reporting is derived from the frozen gate's kind, never asserted in
    prose. The canonical cases are the ones that misled a reader before: B3.2's gate is
    `attribution recorded`, B2.1's is a frozen U bar."""
    kinds = {r.ident: r.gate_kind for r in B.ROWS}
    assert set(kinds.values()) <= {"quality", "reporting", "unmeasured"}
    assert kinds["B3.2"] == "reporting"
    assert kinds["B3.4"] == "reporting"
    assert kinds["B3.5"] == "reporting"
    assert kinds["B3.6"] == "reporting"
    assert kinds["B2.1"] == "quality"
    assert kinds["B2.2"] == "quality"
    assert kinds["B2.3"] == "reporting"
    assert kinds["B3.3"] == "quality"
    assert kinds["B4.1"] == "unmeasured"
    # the label changes no verdict: the frozen gate is untouched
    assert B.ROWS[0].gate


def test_plan_md_labels_what_a_green_proves():
    text = PL.render_plan()
    assert "| Atomic problem | Property | Frozen gate | Green means |" in text
    assert "`B3.2`" in text and "reporting" in text
    assert "quality" in text
    # the DEC-frozen trunk atoms are named, with their bars, in the plan itself
    assert "`E1.a`" in text and "`E1.b`" in text and "`E1.c`" in text
    assert "The census" in text and "atomic problems" in text


# ------------------------------------------------------------------ the route


def test_the_route_is_ordered_references_real_nodes_and_names_evidence():
    assert [s.ident for s in P.ROUTE] == [f"R{i}" for i in range(1, len(P.ROUTE) + 1)]
    assert P.CURRENT_ROUTE in {s.ident for s in P.ROUTE}
    for s in P.ROUTE:
        for node in s.advances.split(","):
            assert node.strip() in P.BY_ID, f"{s.ident} advances unknown node {node}"
        assert s.evidence, f"{s.ident} names no evidence"


def test_every_atom_maps_to_its_owner_and_problem():
    """Each atomic problem hangs under its interior owner, which in turn serves a named problem."""
    for n in P.NODES:
        if n.kind != "atomic":
            continue
        o, pr = P.owner(n.ident), P.problem_of(n.ident)
        assert o and pr, f"{n.ident} has no owner/problem"
        assert P.BY_ID[o].parent == pr


# ------------------------------------------------------------------ the frontier readings


def test_the_card_reports_the_frontier_from_receipts_without_gating_it(tmp_path):
    """E1 has no card row, so the card shows its measured values as REPORTED readings.

    They cannot flip a row (the lab receipts never do), but a blob that is invisible is worse
    than one that is measured: the reader must see g and eff_rank beside the frozen bars.
    """
    import json as _json

    import report_card as RC

    (tmp_path / "lab").mkdir()
    (tmp_path / "lab" / "next_event_iter2.json").write_text(
        _json.dumps(
            {
                "_meta": {"behavior_hash": "abc123"},
                "positive": {"g": 1.15, "eff_rank": 1.47, "beats_base": False},
                "complete": True,
            }
        )
    )
    md, _info = RC.render(tmp_path, "abc123")
    assert "Frontier readings" in md
    assert "g 1.15" in md and "eff_rank 1.47" in md and "bar ≤ 0.5" in md
    assert "beats 1-gram: no" in md
    assert "measured under" not in md  # measured at the current hash: not historical


def test_a_stale_frontier_reading_says_which_behavior_it_came_from(tmp_path):
    import json as _json

    import report_card as RC

    (tmp_path / "lab").mkdir()
    (tmp_path / "lab" / "next_event_iter1.json").write_text(
        _json.dumps(
            {
                "_meta": {"behavior_hash": "deadbeefcafe"},
                "positive": {"g": 1.147, "eff_rank": 1.153, "beats_base": False},
                "complete": True,
            }
        )
    )
    md, _info = RC.render(tmp_path, "abc123")
    assert "measured under `deadbeef`" in md
