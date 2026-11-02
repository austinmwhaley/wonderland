"""Frozen B-foundational scorecard (DEC-562).

The judge card measures B+C consequences: does a simple consumer beat PATH A? That is not the
same claim as "B is foundational". This spec measures B itself, in the order the project's own
definition uses:

    B1 Lawful   - invariants by construction (algebra, schema, determinism, frozen artifact)
    B2 Sufficient - does E lose nothing about the future vs the full history, worst-case?
    B3 Accessible - can simple heads read what is there, at every stage?
    B4 Decisive  - does E support action/counterfactual decisions, not just prediction?
    B5 Minimal & affordable - is it the smallest sufficient object, and fast enough?
    B6 Transferable - does it add signal on held-out consumers, with controls?
    B7 Robust   - does it carry uncertainty under shift and missingness?
    B8 Reproducible - is the artifact an identity (hash, N, one command, commit)?

A row is PASS only when its receipt is current at the card's behavior hash AND its gate is met.
A missing receipt is UNKNOWN and names the producer that must be run. A legacy receipt (one
written before the behavior stamp existed) is UNKNOWN: its number may be useful for diagnosis,
but it is not evidence about the current encoder. No B row becomes PASS by narration.

The B rows are deliberately separate from the judge rows. B+C is the consequence; B is the goal.
DEC-562 keeps one card: this section goes first, the judge follows.
"""

from __future__ import annotations

from dataclasses import dataclass


#: O0 ledger X-ray branches (DEC-569, pre-registered before the run; the card computes which
#: fired from the receipt's arm numbers — a receipt may not define its own criterion).
#: `best_ledger_arm` = argmax persona_geometry over the ledger arms (full, parts, block-PCAs).
O0_BRANCH_READABLE = 0.30  # >= : the additive family is readable under the frozen consumer; the
#  gap is B's serving contract (exposure), an L1 fix inside B.
O0_BRANCH_PARTIAL = 0.15  # in [0.15, 0.30): partial; the non-additive residual matters,
#  O3/O4 retrain territory.
#  < 0.15: the additive family is insufficient under the frozen consumer; inspect PATH A's
#  derived features (ratios/entropy/windows) as the O2 `U` task family.
O0_BRANCH_LABELS = (
    ("exposure", O0_BRANCH_READABLE, ">= .30: readable; the fix is inside B's serving contract"),
    ("partial", O0_BRANCH_PARTIAL, ".15-.30: partial; O3/O4 retrain territory"),
    (
        "insufficient",
        0.0,
        "< .15: additive family insufficient; PATH A's derived features are the task family",
    ),
)


def o0_branch(persona: float) -> str:
    """Which DEC-569 branch a ledger-arm persona score falls into (one definition)."""
    for label, floor, _why in O0_BRANCH_LABELS:
        if persona >= floor:
            return label
    return O0_BRANCH_LABELS[-1][0]


@dataclass(frozen=True)
class Section:
    ident: str
    name: str
    question: str


@dataclass(frozen=True)
class BRow:
    section: str
    ident: str
    name: str
    metric: str
    gate: str
    kind: str  # stage | missing | legacy | bool | le | ge | gt | present
    source: tuple[str, str] | None = None
    thresh: float = 0.0
    producer: str = ""


#: The B capability top line (O8 sprint; DEC-568/D2): one axis per capability, each derived from
#: the B rows that own it. The card renders the axis verdict as the WORST of its rows (FAIL if
#: any row FAILs, PASS only when every row PASSes, else UNKNOWN), so the top line can never say
#: "done" while a detail row is red, and can never be improved by narration. B+C stays a reported
#: consequence section, never a top-line axis.
CAPABILITY_AXES = (
    ("U · sufficiency", ("B2.1",)),
    ("shape-only geometry", ("B3.2",)),
    ("decision-shape", ("B4.1",)),
    ("minimality rank", ("B5.1",)),
    ("transfer", ("B6.1", "B6.3")),
    ("robust", ("B7.1", "B7.2")),
    ("reproducible", ("B8.1", "B8.2")),
)


SECTIONS = (
    Section("B1", "Lawful", "Is the state built from the right algebra?"),
    Section("B2", "Sufficient", "Does B lose nothing about the future?"),
    Section("B3", "Accessible", "Can simple heads read what is there, at every stage?"),
    Section("B4", "Decisive", "Does B support action/counterfactual decisions?"),
    Section("B5", "Minimal & affordable", "Is it the smallest sufficient object, and fast enough?"),
    Section("B6", "Transferable", "Does it add signal on held-out consumers?"),
    Section("B7", "Robust", "Does it stay honest under shift and missingness?"),
    Section("B8", "Reproducible", "Is the artifact an identity?"),
)


ROWS = (
    # ---- B1: lawful -------------------------------------------------------------
    BRow(
        "B1",
        "B1.1",
        "Lawful dynamics (S3)",
        "merge/equivariance/mass-additivity exact on data; tau receipted; stable from step 0",
        "working=True",
        "stage",
        ("judge_stages.json", "S3"),
        producer="measure_dynamics_law.py",
    ),
    BRow(
        "B1",
        "B1.2",
        "Construction laws (C1-C6)",
        "scan / field-permutation / mass / determinism / schema variant / frozen hash",
        "all construction laws met",
        "legacy",
        ("encoder_lawfulness.json", "scan_merge_delta"),
        producer="the legacy encoder-construction producer",
    ),
    # ---- B2: sufficient ---------------------------------------------------------
    BRow(
        "B2",
        "B2.1",
        "Worst-case sufficiency U",
        "U = 1 - worst-case oracle gap vs full history, fixed capacity, broad published family",
        "U >= the declared bar (frozen before the run)",
        "missing",
        producer="measure_sufficiency_law.py",
    ),
    BRow(
        "B2",
        "B2.2",
        "Full-history oracle gap",
        "excess loss of E vs the full-history oracle at fixed capacity",
        "gap <= the declared bar (frozen before the run)",
        "missing",
        producer="measure_sufficiency_law.py",
    ),
    BRow(
        "B2",
        "B2.3",
        "Learned-only attribution",
        "sufficiency of the learned shape vs the served statistics, reported separately",
        "attribution recorded",
        "missing",
        producer="measure_sufficiency_law.py",
    ),
    # ---- B3: accessible ---------------------------------------------------------
    BRow(
        "B3",
        "B3.1",
        "Outlet accessibility (S4)",
        "per-bucket accessibility parity + per-bucket negative controls",
        "working=True",
        "stage",
        ("judge_stages.json", "S4"),
        producer="measure_views_law.py",
    ),
    BRow(
        "B3",
        "B3.2",
        "Stage decomposition M0->M1->served",
        "the fine-structure metric at each stage, so a loss is attributed to a stage",
        "attribution recorded",
        "stage_decomposition",
        ("encoder_stage_decomposition.json", "complete"),
        producer="measure_stage_decomposition.py",
    ),
    BRow(
        "B3",
        "B3.3",
        "Linear accessibility (current)",
        "mean ridge R2 on A->B over the lab's persona+future fields; served vs PATH A",
        "served >= PATH A - 0.01",
        "linear_access",
        ("encoder_block_serving.json", "linear_access.complete"),
        producer="measure_block_serving.py",
    ),
    BRow(
        "B3",
        "B3.4",
        "Block-aware serving geometry",
        "exact counter block beside the learned shape, equal total variance; persona/future/ARI",
        "complete",
        "block_serving",
        ("encoder_block_serving.json", "complete"),
        producer="measure_block_serving.py",
    ),
    BRow(
        "B3",
        "B3.5",
        "Equal-dimension serving geometry",
        "shape PCA + counter tile at 128/128 and 39/39; persona/future/ARI at full size",
        "complete",
        "equal_dimension",
        ("encoder_block_serving.json", "equal_dimension.complete"),
        producer="measure_block_serving.py",
    ),
    BRow(
        "B3",
        "B3.6",
        "Served-object X-ray (ledger exposure)",
        "the ledger alone, its named parts, and ledger block-PCA under the frozen B3.2/B3.4 "
        "consumers, against {exact counter, served shape, PATH A}; the branch is computed here "
        "from the arm numbers (DEC-569)",
        "complete + branch decided",
        "served_object",
        ("encoder_served_object.json", "complete"),
        producer="measure_served_object.py",
    ),
    # ---- B4: decisive -----------------------------------------------------------
    BRow(
        "B4",
        "B4.1",
        "B-only dose-response / argmax",
        "action-conditional response shape readable from the state (not through a judge leg)",
        "gate met",
        "missing",
        producer="measure_decision_law.py",
    ),
    BRow(
        "B4",
        "B4.2",
        "OPE / IPS known-effect recovery (legacy)",
        "E recovers the known effect with a paired CI excluding 0",
        "CI excludes 0 on the declared arms",
        "legacy",
        ("encoder_causal_ci.json", "0"),
        producer="the legacy causal producer",
    ),
    BRow(
        "B4",
        "B4.3",
        "Action-invariance",
        "representation invariant to assignment, still predictive of the outcome",
        "gate met",
        "missing",
        producer="measure_decision_law.py",
    ),
    # ---- B5: minimal & affordable ----------------------------------------------
    BRow(
        "B5",
        "B5.1",
        "Minimality k_min",
        "served rank = the frozen reconstruction certificate",
        "k_min",
        "missing",
        producer="measure_minimality_law.py",
    ),
    BRow(
        "B5",
        "B5.2",
        "Cost / runtime / constant-state (legacy)",
        "runtime, memory and incremental-update receipt",
        "present and within budget",
        "legacy",
        ("runtime.json", ""),
        producer="the legacy runtime producer",
    ),
    # ---- B6: transferable -------------------------------------------------------
    BRow(
        "B6",
        "B6.1",
        "Objective battery (S5)",
        "every content pressure beats its naive comparator (or is excluded and reported)",
        "working=True",
        "stage",
        ("judge_stages.json", "S5"),
        producer="measure_battery_law.py",
    ),
    BRow(
        "B6",
        "B6.2",
        "Donor does-not-lose / adds / controls (legacy)",
        "E vs raw, fair table, table+seq; random/destroyed negative controls",
        "all donor gates met",
        "legacy",
        ("encoder_donor.json", ""),
        producer="the legacy donor producer",
    ),
    BRow(
        "B6",
        "B6.3",
        "Broad held-out transfer family",
        "adds signal on a family broader than the eight judge legs",
        "family published and passed",
        "missing",
        producer="measure_transfer_law.py",
    ),
    # ---- B7: robust -------------------------------------------------------------
    BRow(
        "B7",
        "B7.1",
        "ECE / OOD retention (legacy)",
        "calibration and OOD retention on the old reference",
        "ECE <= 0.1 and OOD retained",
        "legacy",
        ("encoder_b_props.json", "ece_next_event"),
        producer="the legacy calibration producer",
    ),
    BRow(
        "B7",
        "B7.2",
        "Missingness / unlogged confounders",
        "uncertainty carried for unobserved events, never faked",
        "coverage reported",
        "missing",
        producer="measure_robustness_law.py",
    ),
    # ---- B8: reproducible -------------------------------------------------------
    BRow(
        "B8",
        "B8.1",
        "Artifact identity (legacy)",
        "artifact hash + config + data revision recorded",
        "present",
        "legacy",
        ("encoder_b_props.json", "artifact_hash"),
        producer="the legacy artifact producer",
    ),
    BRow(
        "B8",
        "B8.2",
        "One-command reproducibility (S6)",
        "the card is regenerated in one invocation at a recorded commit",
        "working=True",
        "stage",
        ("judge_stages.json", "S6"),
        producer="measure_certification_law.py",
    ),
)
