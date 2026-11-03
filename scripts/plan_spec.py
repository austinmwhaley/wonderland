"""The long-term plan as a BACKWARD decomposition (operator request, 2026-11-03; DEC-579).

`PLAN.md` -> `REPORT_CARD.md`, never the other way round: this file is the single definition of
the tree. `scripts/plan.py` renders `PLAN.md` from it; `scripts/report_card.py` renders the card's
plan section from it and attaches the frozen rows' verdicts. Neither renderer may invent a node.

The shape of the argument:

    goal  <- "to achieve this, these must all be true" <- smaller problems <- frozen card rows

Every node is a NECESSARY CONDITION of its parent. Interior nodes carry no gate of their own: an
interior verdict is the worst of its children (the card's existing rule), and every measured leaf
is a row whose gate already lives in `b_spec.py` / `judge_spec.py` (frozen before its result).
The plan therefore cannot be gamed: it adds no threshold, no criterion, and no frontier item may
be relabelled closed here.

Three kinds of leaf, deliberately distinguished because "all green" must not mean one thing:

* ``card``  — a frozen B or judge row. The card computes its verdict from receipts.
* ``dec``   — a condition whose bars are frozen in a DEC and measured by a pre-registered run
  (the trunk's capacity, DEC-572/575/576/577). No card row exists yet; it closes only by that
  run's receipt and the recorded decision, never by narration.
* ``v1``    — the stack's own gates (AGENTS.md "definition of done" v1). This card does not
  measure them; they are named so the tree spans the whole goal and "solved" cannot be claimed
  on the encoder alone.

The doctrine this implements: `T0 is not SOLVED` (certification integrity rule 2) — a green
encoder battery (`T0`) is necessary, never sufficient, for the project goal.
"""

from __future__ import annotations

from dataclasses import dataclass

import b_spec as B
import judge_spec as J

PASS, FAIL, UNKNOWN = J.PASS, J.FAIL, J.UNKNOWN
FRONTIER = "FRONTIER"  # a dec-kind node: frozen criteria, no card row yet
NOT_ASSESSED = "NOT ASSESSED"  # a v1 leaf: outside this card's evidence

#: The goal, stated once. `why` is the backward argument for why THIS is the goal.
GOAL = (
    "One frozen, self-supervised, schema-agnostic encoder per event stream whose embedding is a "
    "universal donor: a bounded per-customer, per-time vector that loses nothing that matters "
    "about the future — for prediction AND for decisions — readable by simple consumers, "
    "transferable beyond the tasks we know, honest under shift, and certified at fixed capacity "
    "against the full-history oracle, so that any present or future consumer is covered by "
    "construction rather than by enumeration."
)

GOAL_WHY = (
    "A per-use-case wide table (PATH A) must be rebuilt for every new task and stream and "
    "silently encodes the tasks we already know; it cannot be a foundation. A frozen per-stream "
    "embedding amortizes one self-supervised fit across present and future consumers. But 'it "
    "works on our tasks' is not universality: the only defensible claim is worst-case "
    "sufficiency against the full history (U), with accessibility, transfer and robustness "
    "certified separately, because the consumer set is open-ended and known use cases may "
    "veto — never shape — the encoder."
)


@dataclass(frozen=True)
class Node:
    """One node of the tree. Identity + the backward argument; never a gate."""

    ident: str
    parent: str | None
    kind: str  # goal | problem | section | leg | enabler | v1
    title: str
    question: str
    why: str
    principle: str
    leaves: tuple[str, ...] = ()
    evidence: str = "card"  # card | dec | v1
    dec: str = ""  # for evidence=dec: where the frozen bars live
    frontier: str = ""  # a pointer, never a verdict


#: One line per B section: why that section is NECESSARY for its parent problem.
_WHY_SECTION = {
    "B1": "A state update that breaks compositionality or equivariance is not a homomorphism "
    "over events; no sufficiency claim survives it.",
    "B2": "Sufficiency is the donor contract itself: without U the embedding is an unmeasured "
    "summary, and every consumer inherits whatever it dropped.",
    "B3": "Information that a simple head cannot read does not reach a consumer; present is not "
    "usable.",
    "B4": "Prediction sufficiency does not imply decision sufficiency — the action family is "
    "part of the goal, not a downstream detail.",
    "B5": "Without minimality, 'sufficient' is trivially satisfied by shipping the history; "
    "minimality is what makes the object bounded and affordable.",
    "B6": "Transfer beyond our known legs is the only evidence against designing for the tasks "
    "we already have.",
    "B7": "Uncertainty and missingness are where a learned state lies most easily — unlogged "
    "events must be carried, never faked.",
    "B8": "An artifact that cannot be rebuilt at a recorded commit is a transcript, not a "
    "certificate.",
}

#: The B section is necessary for which problem.
_SECTION_PARENT = {
    "B1": "P1",
    "B8": "P1",
    "B2": "P2",
    "B4": "P2",
    "B5": "P2",
    "B3": "P3",
    "B6": "P3",
    "B7": "P3",
}

#: One line per judge leg: why that consumer family is a necessary test of the consequence.
_WHY_LEG = {
    J.LEG_SUPERVISED: "Forecast targets are the most direct consumer of a predictive embedding; "
    "if the frozen state cannot be read for them, the donor claim is empty.",
    J.LEG_UNSUPERVISED: "Geometry is the structure-revealing half of the contract: distance must "
    "mean shared futures, not shared summary.",
    J.LEG_RL: "This is the decision use case: policy value from a frozen pre-action state on "
    "randomized data, with controls.",
    J.LEG_SIMULATOR: "Known-effect recovery: the state must carry action-conditional response, "
    "not correlation, or the decision layer has nothing to stand on.",
}


def _problem(
    ident: str, title: str, question: str, why: str, principle: str, parent: str = "G"
) -> Node:
    return Node(ident, parent, "problem", title, question, why, principle)


#: Level 1 — the five necessary conditions of the goal, in dependency order.
PROBLEMS = (
    _problem(
        "P1",
        "The donor exists",
        "Is there a per-stream artifact built from the right algebra, and is it an identity?",
        "Nothing downstream is meaningful about an object that is not lawful or not "
        "reproducible: a state update that breaks compositionality cannot be a homomorphism "
        "over events, and an artifact that cannot be rebuilt cannot support a claim. This is "
        "the floor under every other node.",
        "Lawful · Robust · Reproducible",
    ),
    _problem(
        "P2",
        "The donor suffices",
        "Does the embedding lose nothing that matters about the future — for prediction and "
        "decision — and is it the smallest object that does?",
        "A donor that is lawful but lossy is a summary, not a foundation: any consumer would "
        "inherit the loss, and no amount of consumer tuning could recover it. Sufficiency is "
        "judged worst-case against the full-history oracle, not against our tasks; decision "
        "sufficiency extends the claim to the action family; minimality removes the degenerate "
        "escape of shipping the history. Necessary for P4: a consumer cannot beat PATH A from "
        "an embedding that dropped the signal.",
        "Universal · Honest · Efficient",
    ),
    _problem(
        "P3",
        "The donor is usable",
        "Can simple consumers read it, does it transfer beyond our known tasks, and does it "
        "stay honest under shift?",
        "Sufficiency proves information is present; it does not prove it is recoverable. A "
        "sufficient-but-entangled representation fails the consumer contract, and a donor that "
        "only helps the tasks we designed for is not universal. Robustness closes the honesty "
        "gap. Necessary for P4 because the judge's heads are simple by design: if the simple "
        "head cannot read it, the consequence cannot appear.",
        "Accessible · Transferable · Robust",
    ),
    _problem(
        "P4",
        "The consequence is delivered",
        "Does PATH B (frozen encoder -> simple head) beat PATH A (wide table -> GBM incumbent) "
        "for the four consumer families, with controls, brain alone and brain+ledger?",
        "This is the project's definition of progress. It is downstream of P1-P3 by "
        "construction: known use cases may veto, never shape, the encoder. A green P4 with red "
        "P2/P3 would mean the consumers were tuned, not the foundation.",
        "Honest · Diagnosable",
    ),
    _problem(
        "P5",
        "The stack is v1-complete",
        "Does the full system deliver a certified decision: identifiable stream -> donor -> "
        "plugins -> red_king -> red_queen, with caterpillar explaining from artifacts?",
        "The foundation exists to serve decisions. v1 completeness is a separate scope from the "
        "encoder, and this card does not assess it; the leaves are named so the tree spans the "
        "whole goal and so that 'all solved' cannot be claimed on the encoder alone.",
        "The whole charter",
    ),
)

#: P2's live enabler: the state-update algebra. Its bars are frozen in DECs, not in a card row —
#: the card cannot compute it, so it renders FRONTIER and no green is possible from this file.
TRUNK = Node(
    "E1",
    "P2",
    "enabler",
    "Trunk capacity (the state-update algebra)",
    "Can the state carry the stream's structure — not just its mass?",
    "A sufficient statistic must have somewhere to live. When the trunk's effective rank "
    "collapses, the embedding is a blob and every downstream row measures the blob, not the "
    "stream. This is the measured blocker (O3 attempt 4 rejected; the dev rig reproduces it in "
    "minutes), and it is the one condition whose bars are frozen outside the card.",
    "Lawful (FR v2.0 pillar 1: rank lives in the state update) · Efficient",
    evidence="dec",
    dec="DEC-572/573 (real data: eff_rank>=30, persona>=0.15, R2>=0.45, stable) · "
    "DEC-575/576/577 (dev rig: g<=0.5 and beats the 1-gram, controls clean)",
    frontier="live: O3 REJECTED; next_event CLOSED on the toy (DEC-581/582 outcomes: 4 algebra "
    "values + capacity @256 all negative, 1-gram floor binding at N=500); remainder = intent "
    "under DEC-583 (on hold)",
)

#: v1 leaves (AGENTS.md definition of done). V1.2 (universal donor) = P1-P3; V1.7 (versioned,
#: receipts) = B8 — referenced, never duplicated.
V1 = (
    Node(
        "V1.1",
        "P5",
        "v1",
        "Causally identifiable stream",
        "Does rabbit_hole emit randomized arms with a known effect, recorded as such?",
        "Without a known-effect stream no counterfactual claim can be validated; every decision "
        "result would be untestable.",
        "Honest · Robust",
        evidence="v1",
    ),
    Node(
        "V1.2",
        "P5",
        "v1",
        "Each plugin passes its independent gate",
        "Do the supervised, unsupervised and white_queen consumers each pass their own gate?",
        "Each consumer family must be certified on its own terms; a single composite score "
        "would hide a failing family.",
        "Honest · Diagnosable",
        evidence="v1",
    ),
    Node(
        "V1.3",
        "P5",
        "v1",
        "red_king recovers known effects",
        "Does red_king recover the known counterfactual effect and improve the certified "
        "decision with vs without it?",
        "The decision layer must add causal value over logging, and must never ship worse than "
        "logging.",
        "Honest · Robust",
        evidence="v1",
    ),
    Node(
        "V1.4",
        "P5",
        "v1",
        "red_queen composes actions",
        "Does red_queen compose next-best actions across cadences under constraints?",
        "The system's output is an action, not a score; composition is where the cadences and "
        "constraints meet.",
        "Efficient · Adaptive",
        evidence="v1",
    ),
    Node(
        "V1.5",
        "P5",
        "v1",
        "caterpillar explains from artifacts",
        "Does caterpillar answer 'why' from artifacts only?",
        "A decision that cannot be explained from its artifacts is not operable, and an "
        "explanation that reads the model live cannot be audited.",
        "Diagnosable · Honest",
        evidence="v1",
    ),
)

#: Stated references so the tree does not duplicate nodes: these v1 requirements are already
#: owned elsewhere in the tree.
V1_REFERENCES = (
    ("V1.2 (AGENTS: universal donor usable by every plugin)", "P1-P3 + B6"),
    ("V1.7 (AGENTS: every artifact versioned + reproducible, receipts)", "B8"),
)


def _section_node(sec) -> Node:
    leaves = tuple(r.ident for r in B.ROWS if r.section == sec.ident)
    return Node(
        sec.ident,
        _SECTION_PARENT[sec.ident],
        "section",
        f"{sec.name}",
        sec.question,
        _WHY_SECTION[sec.ident],
        "see the row gates",
        leaves=leaves,
    )


def _leg_node(leg: str) -> Node:
    return Node(
        f"JUDGE-{leg}",
        "P4",
        "leg",
        f"{leg} leg",
        J.LEG_MEANING[leg],
        _WHY_LEG[leg],
        "see the row gates",
        leaves=tuple(r.ident for r in J.LEG_ROWS if r.leg == leg),
    )


#: Pre-order: the order in which the backward argument reads (existence -> sufficiency ->
#: usability -> consequence -> stack).
NODES: tuple[Node, ...] = (
    Node(
        "G",
        None,
        "goal",
        "A certified universal donor, delivered",
        "What is the whole problem?",
        GOAL_WHY,
        "the whole charter",
    ),
    PROBLEMS[0],
    _section_node(B.SECTIONS[0]),  # B1
    _section_node(B.SECTIONS[7]),  # B8
    PROBLEMS[1],
    _section_node(B.SECTIONS[1]),  # B2
    TRUNK,
    _section_node(B.SECTIONS[3]),  # B4
    _section_node(B.SECTIONS[4]),  # B5
    PROBLEMS[2],
    _section_node(B.SECTIONS[2]),  # B3
    _section_node(B.SECTIONS[5]),  # B6
    _section_node(B.SECTIONS[6]),  # B7
    PROBLEMS[3],
    *(_leg_node(leg) for leg in J.LEGS),
    PROBLEMS[4],
    *V1,
)

BY_ID = {n.ident: n for n in NODES}

#: Rendered anywhere the plan needs to say what is being worked on now. A pointer, not a status.
FRONTIER_NOTE = (
    "The live frontier is P2's enabler E1 (the trunk's state-update algebra). O3 attempt 4 was "
    "rejected on its frozen bars and the dev rig reproduced the blob in minutes; all three "
    "single-axis trunk-algebra candidates are now negative on the toy (DEC-577 outcome: "
    "candidate 1; DEC-578 outcome: candidate 2; DEC-581 outcome: candidate 3), and the "
    "capacity axis is negative too (DEC-582 outcome: baseline algebra at dim=256, g 1.149121 "
    "≈ iteration 1's 1.14727, no usable rank gained) — the 1-gram floor is binding on "
    "next_event at N=500, a finite-sample estimation gap, not a trunk deficiency. The next "
    "dev-rig structure is `intent` (DEC-583, on hold) and then ONE pre-registered port run "
    "under DEC-572's bars; no lab receipt flips a B-row verdict. The PRIMARY progress track "
    "until then is R4: the 11 remaining B sub-problems (6 legacy re-stamps at the current "
    "behavior + 5 new O-series producers), certification-first per the operator's 2026-11-03 "
    "directive."
)


@dataclass(frozen=True)
class RouteStep:
    """One step of the ROUTE — the ordered walk through the tree, distinct from the tree itself.

    The tree says what must be true; the route says in what order the evidence is earned. A step
    names the node it advances and the evidence that closes it. Steps are pointers, not verdicts:
    the card never computes a step's status, and moving the pointer is a spec edit reviewed like
    any other.
    """

    ident: str
    title: str
    advances: str  # node id(s), comma-separated
    evidence: str


#: The route from today to solved. Ordered by dependency, cheapest-first inside the dev rig.
ROUTE: tuple[RouteStep, ...] = (
    RouteStep(
        "R1",
        "Dev rig: master next_event (N=500, known truth)",
        "E1",
        "CLOSED negative at the toy (DEC-581/582 outcomes: 4 algebra values + capacity @256, "
        "g 1.147–1.182 vs bar 0.5, 1-gram floor binding at N=500); remainder = intent under "
        "DEC-583 (on hold); never a B-row verdict",
    ),
    RouteStep(
        "R2",
        "Dev rig: master intent (curriculum step 2)",
        "E1",
        "lab receipts + the intent iteration's pre-registration; both structures clean",
    ),
    RouteStep(
        "R3",
        "Port: ONE real-data run of the winning trunk under DEC-572's bars",
        "E1",
        "a pre-registered port receipt: eff_rank>=30, persona>=0.15, R2>=0.45, stable + B1 canary",
    ),
    RouteStep(
        "R4",
        "Real stream: fill the remaining B rows with their producers",
        "P1, P2, P3",
        "B1.2, B4.1-B4.3, B5.1-B5.2, B6.2-B6.3, B7.1-B7.2, B8.1 at the current behavior",
    ),
    RouteStep(
        "R5",
        "Real stream: re-score the four judge legs (brain and brain+ledger)",
        "P4",
        "judge_leg_*.json at the current behavior, both columns, controls readable",
    ),
    RouteStep(
        "R6",
        "Stack: build and certify v1 (identifiable stream, plugins, red_king, red_queen, "
        "caterpillar)",
        "P5",
        "each v1 gate on the stack's own card rows / receipts",
    ),
    RouteStep(
        "R7",
        "BACKWARD: adversarial review with no BREAKING findings",
        "G",
        "AUDIT.md requirements closed",
    ),
)

#: Which route step the work is on now (a pointer; keep it honest when the frontier moves).
CURRENT_ROUTE = "R4"


def owner(leaf: str) -> str:
    """The section/leg node a frozen row hangs under ('' if the row is not in the tree)."""
    for n in NODES:
        if leaf in n.leaves:
            return n.ident
    return ""


def problem_of(leaf: str) -> str:
    """The level-1 problem a frozen row ultimately serves ('' if unknown)."""
    o = owner(leaf)
    return BY_ID[o].parent if o and BY_ID[o].parent else ""


# --------------------------------------------------------------------- structure integrity


def level(ident: str) -> int:
    """Depth from the goal (G = 0). Raises on a cycle or a dangling parent."""
    seen = set()
    n = BY_ID[ident]
    depth = 0
    while n.parent is not None:
        if n.ident in seen:
            raise ValueError(f"cycle in plan tree at {n.ident}")
        seen.add(n.ident)
        n = BY_ID[n.parent]
        depth += 1
    return depth


def children(ident: str) -> tuple[Node, ...]:
    return tuple(n for n in NODES if n.parent == ident)


def leaves_under(ident: str) -> tuple[str, ...]:
    """Every measured card leaf under a node, pre-order, de-duplicated."""
    out: list[str] = []
    for n in NODES:
        if n.ident == ident:
            out.extend(n.leaves)
    for c in children(ident):
        out.extend(leaves_under(c.ident))
    return tuple(dict.fromkeys(out))


# --------------------------------------------------------------------- the derived verdict


def _worst(values) -> str:
    vals = list(values)
    if any(v == FAIL for v in vals):
        return FAIL
    if any(v != PASS for v in vals):
        return UNKNOWN
    return PASS


def statuses(row_verdicts: dict[str, str], closed: set[str] | None = None) -> dict[str, str]:
    """Verdict per node, DERIVED: leaves from their frozen rows, interiors from children.

    ``closed`` names dec/v1 nodes that have been closed by their own recorded evidence (a
    pre-registered port receipt for E1; a v1 gate for the stack). The card passes the empty set,
    so a green card root is unreachable until those scopes are actually measured — that is the
    point (certification integrity rule 2: T0 is not SOLVED).
    """
    closed = set(closed or ())
    out: dict[str, str] = {}

    def resolve(ident: str) -> str:
        if ident in out:
            return out[ident]
        n = BY_ID[ident]
        if n.evidence == "dec":
            out[ident] = PASS if ident in closed else FRONTIER
            return out[ident]
        if n.evidence == "v1":
            out[ident] = PASS if ident in closed else NOT_ASSESSED
            return out[ident]
        kids = children(ident)
        if kids:
            out[ident] = _worst(resolve(k.ident) for k in kids)
            return out[ident]
        if n.leaves:
            # a section/leg node: the worst of its frozen rows. A row absent from the render is
            # UNKNOWN, never PASS. Leaf verdicts are recorded too, so a caller counting green
            # leaves reads them from the same derived map (never from a raw table beside it).
            for leaf in n.leaves:
                out[leaf] = row_verdicts.get(leaf, UNKNOWN)
            out[ident] = _worst(out[leaf] for leaf in n.leaves)
            return out[ident]
        # malformed node: neither children nor rows — cannot claim anything
        out[ident] = UNKNOWN
        return out[ident]

    for n in NODES:
        resolve(n.ident)
    return out


def leaf_census() -> dict:
    """Counts that make the 'all solved' arithmetic explicit (no receipt needed)."""
    card = [n for n in NODES if n.kind in ("section", "leg") for _ in n.leaves]
    return {
        "card_leaves": len(card),
        "dec_leaves": sum(1 for n in NODES if n.evidence == "dec"),
        "v1_leaves": sum(1 for n in NODES if n.evidence == "v1"),
        "total": len(card) + sum(1 for n in NODES if n.evidence in ("dec", "v1")),
    }
