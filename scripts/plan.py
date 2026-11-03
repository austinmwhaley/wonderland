#!/usr/bin/env python3
"""Render PLAN.md — the project goal cut backwards into solvable problems (DEC-579).

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

LEAF_NAME = {r.ident: r.name for r in B.ROWS}
LEAF_GATE = {r.ident: r.gate for r in B.ROWS}
LEAF_PRODUCER = {r.ident: r.producer for r in B.ROWS}
for _r in J.LEG_ROWS:
    LEAF_NAME[_r.ident] = _r.name
    LEAF_GATE[_r.ident] = _r.gate
    LEAF_PRODUCER[_r.ident] = _r.source


def _solved_when(n: P.Node) -> str:
    """The solved-condition, derived from the node's children — never a new threshold."""
    kids = P.children(n.ident)
    if n.evidence == "dec":
        return (
            f"its pre-registered run meets the frozen bars and the decision is recorded — {n.dec}"
        )
    if n.evidence == "v1":
        return "its v1 gate passes (not measured on this card; AGENTS.md definition of done)"
    if kids:
        return "every child below is solved (worst-child rule)"
    if n.leaves:
        return (
            f"every leaf below passes its frozen gate ({len(n.leaves)} row"
            f"{'s' if len(n.leaves) > 1 else ''})"
        )
    return "its own recorded evidence (see the node)"


def _node_block(n: P.Node) -> list[str]:
    depth = P.level(n.ident)
    hashes = "#" * (3 + depth)
    lines = [f"{hashes} {n.ident} — {n.title}", "", f"*{n.question}*", "", n.why, ""]
    if n.leaves:
        lines.append("**Leaves:** " + ", ".join(f"`{x}`" for x in n.leaves))
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
    census = P.leaf_census()
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
        "true?* Each answer is a smaller, independently measurable problem, until the leaves are",
        "conditions a receipt can settle. Solving the leaves bottom-up is solving the goal — but",
        "only if every leaf is solved, which is why the tree names the ones the card does not",
        "measure yet instead of pretending they do not exist.",
        "",
        "- **Every node is a necessary condition of its parent.** Nothing here is an",
        "  implementation preference; each node answers *why the parent cannot hold without it*.",
        "- **The plan defines no gate.** Every measured leaf is a frozen row whose gate lives in",
        "  `scripts/b_spec.py` or `scripts/judge_spec.py`, frozen before its result. An interior",
        "  verdict is the worst of its children — the card's existing rule, not a new one.",
        "- **Three leaf kinds, deliberately distinguished.** `card` leaves are computed from",
        "  receipts; `dec` leaves have bars frozen in a DEC and close only through a",
        "  pre-registered run; `v1` leaves belong to the stack's own gates and are not assessed",
        "  on this card. A green card alone therefore proves T0 — never SOLVED.",
        "- **No relabelling.** A node moves from open to solved by evidence, never by editing this",
        "  file. Changing what a leaf measures is a new protocol version, not a plan edit.",
        "",
        "## 2. The tree",
        "",
        f"The census: **{census['card_leaves']} card leaves**, **{census['dec_leaves']} DEC-frozen",
        f"leaf**, **{census['v1_leaves']} v1 leaves** — {census['total']} conditions in all.",
        "",
        *_tree("G"),
        "## 3. The measured leaves — what evidence closes each one",
        "",
        "| Leaf | Property | Frozen gate | Producer / source |",
        "|---|---|---|---|",
    ]
    for n in P.NODES:
        for leaf in n.leaves:
            lines.append(
                f"| `{leaf}` | {LEAF_NAME.get(leaf, '?')} | {LEAF_GATE.get(leaf, '?')} | "
                f"{LEAF_PRODUCER.get(leaf, '?')} |"
            )
    lines += [
        "",
        "### The leaves this card does not measure (v1)",
        "",
        "| Leaf | Requirement | Evidence |",
        "|---|---|---|",
    ]
    for n in P.NODES:
        if n.evidence == "v1":
            lines.append(f"| `{n.ident}` | {n.title} | the v1 stack's own gate |")
    lines += [
        "",
        "Two v1 requirements are already owned elsewhere in the tree, so they are references,",
        "not duplicate nodes:",
        "",
    ]
    lines += [f"- **{a}** -> {b}" for a, b in P.V1_REFERENCES]
    lines += [
        "",
        '## 4. What "all solved" means — exactly',
        "",
        f"- Every **card leaf** ({census['card_leaves']}) passes its frozen gate on a current",
        "  receipt: that is `T0`, the encoder battery.",
        f"- The **DEC-frozen leaf** ({census['dec_leaves']}) closes only by its pre-registered run",
        "  meeting the recorded bars and the decision being logged — never by a lab receipt and",
        "  never by narration.",
        f"- Every **v1 leaf** ({census['v1_leaves']}) passes the stack's own gate.",
        "- **No open BREAKING/MAJOR BACKWARD requirement** (AUDIT.md).",
        "",
        "All of the above = **project solved**. Anything less is a named, located gap: the tree",
        "says which node is open and the card says which leaf is missing.",
        "",
        "This is the doctrine's T0/SOLVED distinction made structural: a fully green encoder",
        "card is `ENCODER DONE`, not `SOLVED`, because the v1 scope and the DEC-frozen port run",
        "are separate leaves. The card renders `SOLVED` only when its generated root status is",
        "PASS — which cannot happen while a leaf in this tree is unmeasured.",
        "",
        "## 5. The current frontier",
        "",
        P.FRONTIER_NOTE,
        "",
        "**Why the card can read green while the encoder is not mastered — read this before",
        "trusting a green row.** Several B gates certify *measurement and attribution*, not a",
        "quality level: B3.2's gate is `attribution recorded`, B3.4/B3.5/B3.6's is `complete`,",
        "B1.1/B3.1/B6.1/B8.2 are `working=True`. Those rows prove the pipeline was measured and",
        "the loss was localized; they do not prove the embedding is good. The quality levels",
        "for those axes are the DEC-frozen bars (U >= 0.5 in DEC-570; eff_rank >= 30 and",
        "persona >= 0.15 in DEC-572; on the dev rig g <= 0.5 and beating the 1-gram in",
        "DEC-576/577), and the measured values are in the receipts, not in this plan.",
        "",
        "That is exactly why the trunk is a node (E1) and not a footnote: a blob can satisfy",
        "every reporting gate while failing the quality bars, and the plan must show the",
        "difference instead of letting a green card imply more than it proved.",
        "",
        "## 6. The method contract — how work is allowed to move the tree",
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
        "## 7. Provenance and the one-source rule",
        "",
        "```text",
        "scripts/plan_spec.py ──> scripts/plan.py ──> PLAN.md            (structure only)",
        "                    └──> scripts/report_card.py ──> REPORT_CARD.md (structure + verdicts)",
        "```",
        "",
        "The card imports the spec; the spec imports the frozen row definitions (`b_spec`,",
        "`judge_spec`); neither imports the other's render. A test asserts every frozen row is a",
        "leaf exactly once, the tree is acyclic, and `PLAN.md` is current with the spec.",
        "",
        "This file was reinstated by operator request on 2026-11-03 (DEC-579). DEC-513 once",
        'retired PLAN.md because "the card is the plan"; the difference now is direction and',
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
    print(f"nodes: {len(P.NODES)} · leaves: {P.leaf_census()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
