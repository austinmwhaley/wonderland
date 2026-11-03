#!/usr/bin/env python3
"""Render PLAN.md — the project goal cut backwards into solvable problems (DEC-579; DEC-584:
the tree is cut to atomic problem nodes, at whatever depth that takes).

    python3 scripts/plan.py [--out PLAN.md] [--check]

`PLAN.md` -> `REPORT_CARD.md`: the tree is defined ONCE in `scripts/plan_spec.py`; this file
renders the readable plan, and `scripts/report_card.py` renders the same tree with the frozen
rows' verdicts attached. Editing the plan means editing the spec, never the markdown.

This document carries NO measurements and NO verdicts, on purpose: measurement lives only in
`REPORT_CARD.md` (STATUS.md's rule), so the plan cannot drift out of date when receipts change.
It is a pure function of the spec — which makes `--check` meaningful in CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import b_spec as B  # noqa: E402
import judge_spec as J  # noqa: E402
import plan_spec as P  # noqa: E402

#: The frozen rows' render data, kept in the frozen specs (single source of truth). The plan
#: names the atoms; it does not restate their gates except in the table below, which reads
#: straight from these specs.
ATOM_NAME = {r.ident: r.name for r in B.ROWS}
ATOM_GATE = {r.ident: r.gate for r in B.ROWS}
ATOM_PRODUCER = {r.ident: r.producer for r in B.ROWS}
ATOM_KIND = {r.ident: r.gate_kind for r in B.ROWS}
for _r in J.LEG_ROWS:
    ATOM_NAME[_r.ident] = _r.name
    ATOM_GATE[_r.ident] = _r.gate
    ATOM_PRODUCER[_r.ident] = _r.source
    ATOM_KIND[_r.ident] = "quality"  # a paired CI vs PATH A with controls is a level, not a report


def _solved_when(n: P.Node) -> str:
    """The solved-condition, derived from the node's place in the tree — never a new threshold."""
    if P.children(n.ident):
        return "every child below is solved (worst-child rule)"
    if n.evidence == "dec":
        return (
            f"its pre-registered run meets the frozen bars and the decision is recorded — {n.dec}"
        )
    if n.evidence == "v1":
        return "its v1 gate passes (not measured on this card; AGENTS.md definition of done)"
    return (
        "its frozen gate passes on a current receipt (the gate lives in the frozen spec — see "
        "the atomic-problems table and the card's row)"
    )


def _node_block(n: P.Node) -> list[str]:
    depth = P.level(n.ident)
    hashes = "#" * (3 + depth)
    lines = [f"{hashes} {n.ident} — {n.title}", "", f"*{n.question}*", "", n.why, ""]
    if n.evidence == "dec":
        lines.append(f"**Bars frozen in:** {n.dec}")
    if n.evidence == "v1":
        lines.append("**Evidence:** the v1 stack's own gate — this card does not measure it.")
    lines += [f"**Solved when:** {_solved_when(n)}", ""]
    return lines


def _tree(ident: str) -> list[str]:
    n = P.BY_ID[ident]
    out = _node_block(n)
    for c in P.children(ident):
        out.extend(_tree(c.ident))
    return out


def render_plan() -> str:
    census = P.census()
    lines = [
        "# PLAN.md — the goal, cut backwards into solvable problems",
        "",
        "*Generated from `scripts/plan_spec.py` by `scripts/plan.py` — do not edit by hand.*",
        "",
        "> **Read order.** `AGENTS.md` (doctrine) wins over this plan, and over every other file.",
        "> `REPORT_CARD.md` is the only measurement surface: it renders this same tree with the",
        "> frozen rows' current verdicts. `STATUS.md` is live state; `DECISIONS.md` is the log;",
        "> `AUDIT.md` holds the BACKWARD requirements.",
        "",
        "## 0. The goal — and why it is the goal",
        "",
        P.GOAL,
        "",
        "Why this and not something easier:",
        "",
        P.GOAL_WHY,
        "",
        "## 1. How to read this plan — the backward cut",
        "",
        "The tree starts at the goal and asks, at every node, *what must be true for this to be",
        "true?* Each answer is a smaller, independently measurable problem, broken down until",
        "the atoms are conditions a receipt — or a single recorded decision — can settle. How",
        "many levels that takes is what it takes: there is no forced depth, and the tree no",
        "longer speaks of sections or leaves. Solving the atoms bottom-up is solving the goal",
        "— but only if every atom is solved, which is why the tree names the ones the card does",
        "not measure yet instead of pretending they do not exist.",
        "",
        "- **Every node is a necessary condition of its parent.** Nothing here is an",
        "  implementation preference; each node answers *why the parent cannot hold without it*.",
        "- **The plan defines no gate.** Every measured atomic problem is a frozen row whose gate",
        "  lives in `scripts/b_spec.py` or `scripts/judge_spec.py`, frozen before its result. An",
        "  interior verdict is the worst of its children — the card's existing rule, not a new one.",
        "- **Progress is the tree.** How many problems and atomic problems are solved, read off",
        "  the hierarchy at every level, IS the progress measure: there is no separate progress",
        "  block, and no milestone ladder beside the tree.",
        "- **Three kinds of atomic problem, deliberately distinguished.** `card` atoms are",
        "  computed from receipts; `dec` atoms have bars frozen in a DEC and close only through",
        "  a pre-registered run; `v1` atoms belong to the stack's own gates and are not assessed",
        "  on this card. A green card alone therefore proves T0 — never SOLVED.",
        "- **No relabelling.** A problem moves from open to solved by evidence, never by editing",
        "  this file. Changing what an atom measures is a new protocol version, not a plan edit.",
        "",
        "## 2. The tree",
        "",
        f"The census: **{census['atomic_total']} atomic problems** ({census['card']} measured · "
        f"{census['dec']} DEC-frozen · {census['v1']} v1) under **{census['interior']} interior "
        f"problems** — progress is the solved count, read off the tree at every level.",
        "",
        *_tree("G"),
        "## 3. The atomic problems — what evidence closes each one",
        "",
        "### The measured atomic problems (frozen rows)",
        "",
        "| Atomic problem | Property | Frozen gate | Green means | Producer / source |",
        "|---|---|---|---|---|",
    ]
    for n in P.NODES:
        if n.kind == "atomic" and n.evidence == "card":
            lines.append(
                f"| `{n.ident}` | {ATOM_NAME.get(n.ident, n.title)} | "
                f"{ATOM_GATE.get(n.ident, '?')} | {ATOM_KIND.get(n.ident, '?')} | "
                f"{ATOM_PRODUCER.get(n.ident, '?')} |"
            )
    lines += [
        "",
        "### The DEC-frozen atomic problems (no card row; each closes only by its recorded evidence)",
        "",
        "| Atomic problem | What must be true | Bars frozen in |",
        "|---|---|---|",
    ]
    for n in P.NODES:
        if n.kind == "atomic" and n.evidence == "dec":
            lines.append(f"| `{n.ident}` | {n.question} | {n.dec} |")
    lines += [
        "",
        "### The v1 atomic problems this card does not measure",
        "",
        "| Atomic problem | Requirement | Evidence |",
        "|---|---|---|",
    ]
    for n in P.NODES:
        if n.evidence == "v1":
            lines.append(f"| `{n.ident}` | {n.title} | the v1 stack's own gate |")
    lines += [
        "",
        "Two v1 requirements are already owned elsewhere in the tree, so they are references,",
        "not duplicate atoms:",
        "",
    ]
    lines += [f"- **{a}** -> {b}" for a, b in P.V1_REFERENCES]
    lines += [
        "",
        '## 4. What "all solved" means — exactly',
        "",
        f"- Every **measured atom** ({census['card']}) passes its frozen gate on a current",
        "  receipt: that is `T0`, the encoder battery.",
        f"- Every **DEC-frozen atom** ({census['dec']}) closes only by its pre-registered run",
        "  meeting the recorded bars and the decision being logged — never by a lab receipt and",
        "  never by narration.",
        f"- Every **v1 atom** ({census['v1']}) passes the stack's own gate.",
        f"- Every **interior problem** ({census['interior']}) is solved when all the atoms under",
        "  it are solved: the worst-child rule, nothing else.",
        "- **No open BREAKING/MAJOR BACKWARD requirement** (AUDIT.md).",
        "",
        "All of the above = **project solved**. Anything less is a named, located gap: the tree",
        "says which problem is open and the card says which atom is missing.",
        "",
        "This is the doctrine's T0/SOLVED distinction made structural: a fully green encoder",
        "card is `ENCODER DONE`, not `SOLVED`, because the v1 scope and the DEC-frozen trunk",
        "atoms are separate atoms. The card renders `SOLVED` only when its generated root status",
        "is PASS — which cannot happen while an atom in this tree is unmeasured.",
        "",
        "## 5. The route — the order the evidence is earned",
        "",
        "The tree says what must be true; the route says in what order it is earned. It is a",
        "pointer list, not a second scorecard: no step carries a verdict, and the card never",
        "computes one. Moving the pointer is a reviewed spec edit like any other.",
        "",
        "| Step | What is earned | Advances | Evidence that closes it |",
        "|---|---|---|---|",
    ]
    for s in P.ROUTE:
        mark = " ← now" if s.ident == P.CURRENT_ROUTE else ""
        lines.append(f"| **{s.ident}{mark}** | {s.title} | {s.advances} | {s.evidence} |")
    lines += [
        "",
        "The order is forced by dependency and by cost: the dev rig is minutes and earns the",
        "right to the real-data port; the port closes E1.c; only then can the remaining B atoms",
        "be read as evidence about a working trunk; only then is a B+C re-score diagnostic rather",
        "than a measurement of the blob.",
        "",
        "## 6. The current frontier",
        "",
        P.FRONTIER_NOTE,
        "",
        "**Why the card can read green while the encoder is not mastered — read this before",
        "trusting a green row.** The plan labels every measured atom with what its green PROVES",
        "(`scripts/b_spec.py: gate_kind`, derived from the frozen gate's kind, never asserted):",
        "",
    ]
    reporting = [
        n.ident
        for n in P.NODES
        if n.kind == "atomic" and n.evidence == "card" and ATOM_KIND.get(n.ident) == "reporting"
    ]
    lines += [
        "- **reporting** — green certifies that the measurement happened and was attributed: "
        + ", ".join(f"`{x}`" for x in reporting)
        + ".",
        "- **quality** — green certifies a level against a frozen bar or comparator (the remaining"
        " atoms).",
        "",
        "The quality levels for the reporting-gated axes are the DEC-frozen bars (U >= 0.5 in",
        "DEC-570; eff_rank >= 30 and persona >= 0.15 in DEC-572; on the dev rig g <= 0.5 and",
        "beating the 1-gram in DEC-576/577), and the measured values are in the receipts, not in",
        "this plan.",
        "",
        "That is exactly why the trunk is a problem (E1) cut to three DEC-frozen atoms (E1.a–c)",
        "and not a footnote: a blob can satisfy every reporting gate while failing the quality",
        "bars, and the plan must show the difference instead of letting a green card imply more",
        "than it proved.",
        "",
        "## 7. The method contract — how work is allowed to move the tree",
        "",
        "1. **One axis per iteration**, pre-registered in `DECISIONS.md` before the run",
        "   (gate, bars, a-priori branches). No post-hoc bar edits in any branch.",
        "2. **Dev rig first, real stream second.** Known-truth lab fixtures (`next_event` ->",
        "   `intent` -> rate heterogeneity -> entities -> trend -> action peak) at N=500 make a",
        "   full A->B->C loop minutes long; a candidate earns the right to a real-data run.",
        "3. **Lab receipts are a development instrument.** They never flip a B-row verdict;",
        "   certification stays on the real stream. The port fires only when both `next_event`",
        "   and `intent` are mastered with clean controls (DEC-576).",
        "4. **Every behaviour move folds its re-stamp**: frozen artifacts are never retrained to",
        "   re-label them; their receipts are regenerated by re-running the recorded producers.",
        "5. **The frontier is chosen by the tree, not by convenience**: the open node closest to",
        "   the goal whose blocker is measured — cheapest experiment first.",
        "",
        "## 8. Provenance and the one-source rule",
        "",
        "```text",
        "scripts/plan_spec.py ──> scripts/plan.py ──> PLAN.md            (structure only)",
        "                    └──> scripts/report_card.py ──> REPORT_CARD.md (structure + verdicts)",
        "```",
        "",
        "The card imports the spec; the spec imports the frozen row definitions (`b_spec`,",
        "`judge_spec`); neither imports the other's render. A test asserts every frozen row is an",
        "atomic problem exactly once, the tree is acyclic, and `PLAN.md` is current with the spec.",
        "",
        "This file was reinstated by operator request on 2026-11-03 (DEC-579), and reframed the",
        "same day (DEC-584) at the operator's directive: the tree is a recursive decomposition",
        'to ATOMIC problems at whatever depth that takes — "section" and "leaf" retired as',
        "vocabulary, progress read off the solved count at every level. DEC-513 once retired",
        'PLAN.md because "the card is the plan"; the difference now is direction and',
        "single-source: the plan is the goal-side decomposition, the card is its measurement",
        "surface, and both render from one spec — so the card cannot grow rows the plan does not",
        "contain, and the plan cannot claim a state the card did not measure.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "PLAN.md"))
    ap.add_argument("--check", action="store_true", help="fail if the file is stale")
    a = ap.parse_args()
    text = render_plan()
    p = Path(a.out)
    if a.check:
        if not p.exists() or p.read_text() != text:
            print(f"PLAN.md is stale at {p} — run `python3 scripts/plan.py`")
            return 1
        print(f"PLAN.md is current ({len(text.splitlines())} lines)")
        return 0
    p.write_text(text)
    print(f"plan -> {p}")
    print(f"nodes: {len(P.NODES)} · census: {P.census()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
