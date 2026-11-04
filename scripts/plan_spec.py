"""The long-term plan as a BACKWARD decomposition (operator request, 2026-11-03; DEC-579;
reframed to atomic problem nodes by DEC-584, 2026-11-03).

`PLAN.md` -> `REPORT_CARD.md`, never the other way round: this file is the single definition of
the tree. `scripts/plan.py` renders `PLAN.md` from it; `scripts/report_card.py` renders the card's
plan section from it and attaches the frozen rows' verdicts. Neither renderer may invent a node.

The shape of the argument:

    goal  <- "to achieve this, these must all be true" <- smaller problems, recursively, to atoms

Every node is a NECESSARY CONDITION of its parent. Interior problems carry no gate of their
own: an interior verdict is the worst of its children (the card's existing rule). The tree is
cut to ATOMIC problems — each one closable by a single measurement or a single recorded
decision — at whatever depth that takes: there is no forced layer count, and the tree no
longer speaks of "sections" or "leaves". Progress is read off the tree itself: how many
atomic problems are solved, at every level.

Three kinds of atomic problem, deliberately distinguished because "all green" must not mean
one thing:

* ``card``  — a frozen B or judge row. The card computes its verdict from receipts.
* ``dec``   — a condition whose bars are frozen in a DEC and closed only by a pre-registered
  run's receipt and the recorded decision (the trunk: next_event toy, intent toy, port).
  No card row exists yet; it can never read green from this file.
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
FRONTIER = "FRONTIER"  # a dec-kind atom: frozen criteria, no card row yet
NOT_ASSESSED = "NOT ASSESSED"  # a v1 atom: outside this card's evidence

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
    """One node of the tree. Identity + the backward argument; never a gate.

    Interior (goal/problem) nodes are judged only by their children. Atomic nodes are judged by
    their evidence kind: ``card`` from the frozen row's receipt verdict, ``dec`` from a
    pre-registered run's recorded close, ``v1`` from the stack's own gate.
    """

    ident: str
    parent: str | None
    kind: str  # goal | problem | atomic
    title: str
    question: str
    why: str
    principle: str
    evidence: str = "card"  # card | dec | v1 (used by atoms; interiors derive from children)
    dec: str = ""  # for evidence=dec: where the frozen bars live
    frontier: str = ""  # a pointer, never a verdict


#: One line per B problem-group: why that group of rows is NECESSARY for its parent problem.
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

#: The B problem-group is necessary for which problem.
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

#: One line per frozen B row: why THAT atom is necessary for its problem-group.
_WHY_ROW_B = {
    "B1.1": "The merge / equivariance / mass laws are the homomorphism floor: a state update "
    "that does not compose exactly over event sequences is not a lawful merge, and no "
    "sufficiency claim survives it.",
    "B1.2": "The construction laws (C1–C6) are the v6 algebraic-law pillar in testable form — "
    "scan associativity, field permutation, determinism, schema variants, the frozen hash: a "
    "construction that breaks one is a transcript, not a certificate.",
    "B2.1": "U is the donor contract itself: without a worst-case bound on the oracle gap at "
    "fixed capacity, the embedding is an unmeasured summary and every consumer inherits "
    "whatever it dropped.",
    "B2.2": "The per-probe excess loss vs the full-history arm is the content of the U bound: "
    "without it, U is a number with no referent, and 'sufficient' is a word, not a "
    "measurement.",
    "B2.3": "The sufficiency claim must belong to the learned shape, not the served statistics: "
    "without the learned-only attribution, a green U could be the ledger's, not the encoder's "
    "(the circular-evaluation guard).",
    "B3.1": "An outlet that cannot read its own bucket means the information is present but "
    "unusable: per-bucket accessibility parity, with negative controls, is the floor of "
    "recoverability.",
    "B3.2": "A loss must be attributable to a stage: the M0->M1->served decomposition is what "
    "turns 'the embedding lost it' into 'the fold lost it' — a diagnosis, not a verdict.",
    "B3.3": "A simple linear head that cannot beat PATH A at readout means no downstream "
    "consumer can: linear access is the floor of the Accessibility principle.",
    "B3.4": "Serving allocation must be measured against the exact counter at equal variance: "
    "without it, a geometry gain could be an allocation artifact, and the shape block's share "
    "would be unattributed.",
    "B3.5": "Dimension-matched shape + counter tiles separate a real serving gain from a raw "
    "dimension-count gain: without the equal-dimension control, 'serving helps' is confounded.",
    "B3.6": "The ledger's named parts must be exposed under the frozen consumers: the X-ray "
    "localizes where the fine structure lives (the exposure branch) so serving is designed, "
    "not tuned.",
    "B4.1": "Prediction sufficiency does not imply decision sufficiency: the action-conditional "
    "response shape must be readable from the state itself, or the action family is not "
    "covered by construction.",
    "B4.2": "Known-effect recovery with a paired CI excluding 0 is the causal grounding of the "
    "decision claim: without it, 'the state carries the effect' is correlation wearing a coat.",
    "B4.3": "A representation that moves with the assignment encodes the intervention, not the "
    "state: action-invariance is the test that the decision-grade state is separable from the "
    "action.",
    "B5.1": "Without minimality, 'sufficient' is trivially satisfied by shipping the history: "
    "k_min (the reconstruction certificate) is what makes the object bounded, fixed and "
    "affordable.",
    "B5.2": "A capability that is correct but unacceptably slow or state-unbounded is not done: "
    "the runtime / constant-state receipt is the cost side of the minimality bar.",
    "B6.1": "Content pressures that do not beat their naive comparators leave the breadth of "
    "the embedding unproven: the objective battery is the anti-overfit clause made measurable.",
    "B6.2": "Does-not-lose / adds / destroyed-vs-random controls against raw and the fair "
    "table: without them, 'adds unique signal' is a point estimate, not a certification.",
    "B6.3": "Signal that appears only on the eight known legs means the donor was designed for "
    "our tasks: a broad held-out transfer family is the only evidence against that.",
    "B7.1": "Calibrated uncertainty under shift is where a learned state lies most easily: ECE "
    "/ OOD retention on the reference stream is the honesty half of the robustness claim.",
    "B7.2": "Unlogged events must be carried as uncertainty, never faked: missingness handling "
    "is what makes 'coverage over functions of the observed stream' true instead of "
    "aspirational.",
    "B8.1": "An artifact whose hash, config and data revision are not recorded is a transcript, "
    "not a certificate: identity is the floor on which reproducibility stands.",
    "B8.2": "If the card cannot be regenerated by one command at a recorded commit, the "
    "certification cannot be re-derived: S6 is the reproducibility claim itself.",
}

#: One line per judge row: the column's role in the leg's argument.
_WHY_ROW_J = {
    "JUDGE-supervised-brain": "The PROGRESS column of the most direct consumer: without a "
    "strict win over PATH A (paired CI excluding 0, controls below brain) on every "
    "non-degenerate task, the donor claim is empty here.",
    "JUDGE-supervised-brain_ledger": "The DELIVERY column: brain + ledger must be non-inferior "
    "within the frozen margin — the shipping claim; ledger points are never a brain win, so "
    "this row measures what the customer actually gets.",
    "JUDGE-unsupervised-brain": "The PROGRESS column of the structure-revealing half: distance "
    "must mean shared futures, not shared summary — a strict win over PATH A with controls, "
    "or the geometry claim is empty.",
    "JUDGE-unsupervised-brain_ledger": "The DELIVERY column: brain + ledger must keep the "
    "geometry non-inferior within the frozen margin — what the customer actually gets when "
    "the ledger is attached.",
    "JUDGE-rl-brain": "The PROGRESS column of the decision use case: policy value from a "
    "frozen pre-action state on randomized data must beat table and random with controls, or "
    "the decision layer has nothing to stand on.",
    "JUDGE-rl-brain_ledger": "The DELIVERY column: brain + ledger must keep the decision value "
    "non-inferior within the frozen margin.",
    "JUDGE-simulator-brain": "The PROGRESS column of known-effect recovery: the state must "
    "carry action-conditional response, not correlation — a strict win over PATH A with the "
    "destroyed control below, or the counterfactual claim is empty.",
    "JUDGE-simulator-brain_ledger": "The DELIVERY column: brain + ledger must keep the "
    "known-effect recovery non-inferior within the frozen margin.",
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
        "encoder, and this card does not assess it; the atoms are named so the tree spans the "
        "whole goal and so that 'all solved' cannot be claimed on the encoder alone.",
        "The whole charter",
    ),
)

#: P2's live enabler: the state-update algebra, cut to three DEC-frozen atoms. The card cannot
#: compute any of them, so they render FRONTIER and no green is possible from this file.
TRUNK = Node(
    "E1",
    "P2",
    "problem",
    "Trunk capacity (the state-update algebra)",
    "Can the state carry the stream's structure — not just its mass?",
    "A sufficient statistic must have somewhere to live. When the trunk's effective rank "
    "collapses, the embedding is a blob and every downstream row measures the blob, not the "
    "stream. This is the measured blocker (O3 attempt 4 rejected; the dev rig reproduces it in "
    "minutes), and it is the one problem whose bars are frozen outside the card — which is why "
    "it is cut into atoms, each with its own DEC-frozen close.",
    "Lawful (FR v2.0 pillar 1: rank lives in the state update) · Efficient",
    frontier="live: O3 REJECTED; next_event exhausted negative at the toy (DEC-581/582); "
    "intent pre-registered (DEC-586, awaiting operator approval); port pending E1.a+E1.b",
)

E1_ATOMS = (
    Node(
        "E1.a",
        "E1",
        "atomic",
        "next_event mastered on the toy (curriculum step 1)",
        "Does the frozen trunk beat the 1-gram on the known-truth next_event structure at the "
        "dev-rig scale, with clean controls?",
        "The dev rig is the minutes-scale instrument that localizes the blob; next_event is "
        "curriculum step 1, and DEC-576's port protocol needs this structure mastered before "
        "any real-data claim.",
        "Honest · Diagnosable",
        evidence="dec",
        dec="DEC-576/577 (dev rig: g<=0.5 and beats the 1-gram, controls clean) — exhausted "
        "negative at N=500 across all four single-axis trunk-algebra values plus the dim=256 "
        "capacity read (DEC-581/582 outcomes: the 1-gram floor is binding, a finite-sample "
        "estimation gap, not a trunk deficiency)",
    ),
    Node(
        "E1.b",
        "E1",
        "atomic",
        "intent mastered on the toy (curriculum step 2)",
        "Does the frozen trunk beat the 1-gram on the per-customer intent structure, at a "
        "scale where the sample can determine it, with clean controls?",
        "The next_event outcome located the shortfall in the sample, not the trunk: intent at a "
        "determinable scale is the structure the trunk must next master (DEC-583; the snr "
        "difficulty dial is rejected as an axis).",
        "Honest · Diagnosable",
        evidence="dec",
        dec="DEC-586 (curriculum step 2: the intent structure, one axis, at the frozen "
        "difficulty; fixture intent=5.0/br=0.3/N=500/seed=17, frozen DEC-576 bars in corrected "
        "units, branches (i)-(v), 4-file behavior move + full cfm_v2k re-stamp)",
    ),
    Node(
        "E1.c",
        "E1",
        "atomic",
        "port: ONE real-data run under DEC-572's bars",
        "Does the winning trunk master the real stream — eff_rank>=30, persona>=0.15, "
        "R2>=0.45, stable, plus the B1 canary — in ONE pre-registered run?",
        "Toy mastery earns the right to exactly one real-data run; the port receipt is what "
        "makes B's green a statement about a working trunk, not about a blob (DEC-576's port "
        "protocol).",
        "Honest · Efficient",
        evidence="dec",
        dec="DEC-572/573 (real data: eff_rank>=30, persona>=0.15, R2>=0.45, stable)",
    ),
)

#: v1 atoms (AGENTS.md definition of done). V1.2 (universal donor) = P1-P3; V1.7 (versioned,
#: receipts) = B8 — referenced, never duplicated.
V1 = (
    Node(
        "V1.1",
        "P5",
        "atomic",
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
        "atomic",
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
        "atomic",
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
        "atomic",
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
        "atomic",
        "caterpillar explains from artifacts",
        "Does caterpillar answer 'why' from artifacts only?",
        "A decision that cannot be explained from its artifacts is not operable, and an "
        "explanation that reads the model live cannot be audited.",
        "Diagnosable · Honest",
        evidence="v1",
    ),
)

#: Stated references so the tree does not duplicate atoms: these v1 requirements are already
#: owned elsewhere in the tree.
V1_REFERENCES = (
    ("V1.2 (AGENTS: universal donor usable by every plugin)", "P1-P3 + B6"),
    ("V1.7 (AGENTS: every artifact versioned + reproducible, receipts)", "B8"),
)


def _section_node(sec) -> Node:
    return Node(
        sec.ident,
        _SECTION_PARENT[sec.ident],
        "problem",
        f"{sec.name}",
        sec.question,
        _WHY_SECTION[sec.ident],
        "see the row gates",
    )


def _leg_node(leg: str) -> Node:
    return Node(
        f"JUDGE-{leg}",
        "P4",
        "problem",
        f"{leg} leg",
        J.LEG_MEANING[leg],
        _WHY_LEG[leg],
        "see the row gates",
    )


def _b_atom(row) -> Node:
    """A frozen B row, as a first-class atomic problem (its gate stays in b_spec)."""
    return Node(
        row.ident,
        row.section,
        "atomic",
        row.name,
        row.metric,
        _WHY_ROW_B[row.ident],
        "the row's frozen gate",
        evidence="card",
    )


def _j_atom(row) -> Node:
    """A judge leg x column row, as a first-class atomic problem (its gate stays in judge_spec)."""
    return Node(
        row.ident,
        f"JUDGE-{row.leg}",
        "atomic",
        row.name,
        row.gate,
        _WHY_ROW_J[row.ident],
        "the row's frozen gate",
        evidence="card",
    )


def _with_atoms(sec) -> tuple[Node, ...]:
    return (_section_node(sec), *(_b_atom(r) for r in B.ROWS if r.section == sec.ident))


def _leg_with_atoms(leg: str) -> tuple[Node, ...]:
    return (_leg_node(leg), *(_j_atom(r) for r in J.LEG_ROWS if r.leg == leg))


#: Pre-order: the order in which the backward argument reads (existence -> sufficiency ->
#: usability -> consequence -> stack), with every atom under its parent.
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
    *_with_atoms(B.SECTIONS[0]),  # B1
    *_with_atoms(B.SECTIONS[7]),  # B8
    PROBLEMS[1],
    *_with_atoms(B.SECTIONS[1]),  # B2
    TRUNK,
    *E1_ATOMS,
    *_with_atoms(B.SECTIONS[3]),  # B4
    *_with_atoms(B.SECTIONS[4]),  # B5
    PROBLEMS[2],
    *_with_atoms(B.SECTIONS[2]),  # B3
    *_with_atoms(B.SECTIONS[5]),  # B6
    *_with_atoms(B.SECTIONS[6]),  # B7
    PROBLEMS[3],
    *(item for leg in J.LEGS for item in _leg_with_atoms(leg)),
    PROBLEMS[4],
    *V1,
)

BY_ID = {n.ident: n for n in NODES}

#: Rendered anywhere the plan needs to say what is being worked on now. A pointer, not a status.
FRONTIER_NOTE = (
    "The B+B+C success definition is met under DEC-598: B is certified (23/23 rows, 8/8 "
    "sections) and the four judge legs are 8/8 on the served block bank (supervised ridge "
    "alpha=10 + delivery bank + compact ledger; block-aware geometry; RL and simulator "
    "canonical). The v1 flat judge was 4/8 and is recorded beside it. The full plan root is "
    "still OPEN at E1/v1: E1.a (next_event) is exhausted negative; E1.b (intent) is a clean "
    "negative (g 2.665; mass counter 2.633; signature 2.600); E1.c waits; the v1 stack "
    "(5 atoms) is unassessed. Those are outside the B+B+C definition; the next stack work is "
    "R6 (the v1 plugin gates)."
)

#: The macro review rendered in PLAN.md and REPORT_CARD.md. A reading of the tree and the
#: receipts, never a new gate; numbers are current at behavior 7baf744f.
MACRO_REVIEW = (
    "At behavior `7baf744f`, B is certified (23/23 rows, 8/8 sections) and B+C is 8/8 "
    "leg-columns under the bank protocol (DEC-598): a served block bank (32 certified shape "
    "PCs + level + response + RFM + action + exact counter + schema profile + numeric/mass "
    "blocks) with a compact judge ledger; the supervised consumer is a standardized ridge "
    "alpha=10, the geometry consumer fits non-negative block weights on split A. The v1 flat "
    "judge (4/8) and the GBM delivery head are recorded beside it. The full plan root remains "
    "OPEN because E1 (the trunk) is FRONTIER and the v1 stack is unassessed — outside the "
    "B+B+C definition. The next work under that definition is the plugin layer (R6); the "
    "trunk remains the open research frontier."
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
        "E1.a",
        "CLOSED negative at the toy (DEC-581/582 outcomes: 4 algebra values + capacity @256, "
        "g 1.147–1.182 vs bar 0.5, 1-gram floor binding at N=500); the intent remainder was "
        "executed under DEC-586; never a B-row verdict",
    ),
    RouteStep(
        "R2",
        "Dev rig: master intent (curriculum step 2)",
        "E1.b",
        "EXECUTED negative (DEC-586): intent fixture g 2.665; mass counter 2.633; served "
        "signature 2.600; all worse than the 1-gram. The atom stays FRONTIER; the next "
        "candidate is the state algebra (mass/shape or signature state).",
    ),
    RouteStep(
        "R3",
        "Port: ONE real-data run of the winning trunk under DEC-572's bars",
        "E1.c",
        "a pre-registered port receipt: eff_rank>=30, persona>=0.15, R2>=0.45, stable + B1 canary",
    ),
    RouteStep(
        "R4",
        "Real stream: certify the B foundational battery",
        "P1, P2, P3",
        "DONE: B certified 23/23 rows, 8/8 sections at behavior 7baf744f (DEC-594/595/596); "
        "the B gate is no longer the constraint.",
    ),
    RouteStep(
        "R5",
        "Real stream: re-score the four judge legs (brain and brain+ledger)",
        "P4",
        "DONE under the bank protocol (DEC-598): 8/8 leg-columns on the served block bank; "
        "the v1 flat judge (4/8) and the GBM delivery head are recorded beside it.",
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
CURRENT_ROUTE = "R6"


def owner(ident: str) -> str:
    """The interior problem an atomic problem hangs under ('' if ident is not an atom)."""
    n = BY_ID.get(ident)
    if n is None or n.kind != "atomic":
        return ""
    return n.parent or ""


def problem_of(ident: str) -> str:
    """The level-1 problem an atomic problem ultimately serves ('' if unknown)."""
    o = owner(ident)
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


def atomics_under(ident: str) -> tuple[str, ...]:
    """Every atomic problem under a node, pre-order, de-duplicated (the node itself if atomic)."""
    out: list[str] = []
    n = BY_ID[ident]
    if n.kind == "atomic":
        out.append(n.ident)
    for c in children(ident):
        out.extend(atomics_under(c.ident))
    return tuple(dict.fromkeys(out))


# --------------------------------------------------------------------- the derived verdict


def _worst(values) -> str:
    """The worst-child rule, refined for the atomic reframe (DEC-584).

    FAIL > UNKNOWN (a missing row) > NOT_ASSESSED-as-UNKNOWN (an unmeasured scope keeps its
    ancestors open, never green) > FRONTIER-only (a scope that is entirely DEC-frozen renders
    FRONTIER, preserving "no card row yet") > PASS. A mix of measured PASS and open FRONTIER
    children is UNKNOWN: the scope is partly measured and partly still to be earned.
    """
    vals = list(values)
    if any(v == FAIL for v in vals):
        return FAIL
    if any(v in (UNKNOWN, NOT_ASSESSED) for v in vals):
        return UNKNOWN
    if vals and all(v == FRONTIER for v in vals):
        return FRONTIER
    if any(v == FRONTIER for v in vals):
        return UNKNOWN
    return PASS


def statuses(row_verdicts: dict[str, str], closed: set[str] | None = None) -> dict[str, str]:
    """Verdict per node, DERIVED: atoms from their frozen rows (or their recorded closes),
    interiors from their children.

    ``closed`` names dec/v1 atoms that have been closed by their own recorded evidence (a
    pre-registered port receipt for E1.c; a v1 gate for the stack). The card passes the empty
    set, so a green card root is unreachable until those scopes are actually measured — that
    is the point (certification integrity rule 2: T0 is not SOLVED).
    """
    closed = set(closed or ())
    out: dict[str, str] = {}

    def resolve(ident: str) -> str:
        if ident in out:
            return out[ident]
        n = BY_ID[ident]
        kids = children(ident)
        if kids:
            out[ident] = _worst(resolve(k.ident) for k in kids)
            return out[ident]
        if n.evidence == "dec":
            out[ident] = PASS if ident in closed else FRONTIER
            return out[ident]
        if n.evidence == "v1":
            out[ident] = PASS if ident in closed else NOT_ASSESSED
            return out[ident]
        if n.kind == "atomic":
            # a card atom: its frozen row's verdict, recorded so a caller counting green
            # atoms reads them from the same derived map (never from a raw table beside it).
            out[ident] = row_verdicts.get(ident, UNKNOWN)
            return out[ident]
        # malformed atom: no children, no evidence kind — cannot claim anything
        out[ident] = UNKNOWN
        return out[ident]

    for n in NODES:
        resolve(n.ident)
    return out


def census() -> dict:
    """Counts that make the 'all solved' arithmetic explicit (no receipt needed)."""
    atoms = [n for n in NODES if n.kind == "atomic"]
    return {
        "card": sum(1 for n in atoms if n.evidence == "card"),
        "dec": sum(1 for n in atoms if n.evidence == "dec"),
        "v1": sum(1 for n in atoms if n.evidence == "v1"),
        "atomic_total": len(atoms),
        "interior": sum(1 for n in NODES if n.kind in ("goal", "problem")),
    }
