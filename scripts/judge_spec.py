"""Frozen gate spec for the JUDGE card (DEC-390) — the court is built before the cases.

The project's definition of progress is one question, asked four ways:

    PATH B (event stream -> frozen encoder -> simple head)  >  PATH A (wide table -> GBM incumbent)

and it is answered twice per leg, because the ledger is a companion and not a win (DEC-383):

    brain            = encoder alone                      -> PROGRESS
    brain + ledger   = encoder + mechanical ledger views   -> DELIVERY

Ledger points never count as progress. A row may PASS only with a finite comparator, a paired CI
that excludes zero, controls that sit below the brain, and a fresh receipt. Anything else renders
UNKNOWN -- never PASS by absence, never PASS by narration.

Gate definitions live HERE, in code, and are frozen before any result is observed
(certification-integrity rule 1). Receipts report measurements; they may not define their own
criterion: a receipt introducing a new `rule`/`criterion` string is treated as UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- the judge (top line)

LEG_SUPERVISED = "supervised"
LEG_UNSUPERVISED = "unsupervised"
LEG_RL = "rl"
LEG_SIMULATOR = "simulator"

LEGS = (LEG_SUPERVISED, LEG_UNSUPERVISED, LEG_RL, LEG_SIMULATOR)

#: the two numbers reported per leg (DEC-383's two-number card)
COLUMNS = ("brain", "brain_ledger")

COLUMN_LABEL = {
    "brain": "brain alone (PROGRESS)",
    "brain_ledger": "brain + ledger (DELIVERY, non-inferiority)",
}

LEG_MEANING = {
    LEG_SUPERVISED: "forecast targets beat the incumbent GBM on the same splits",
    LEG_UNSUPERVISED: "distance equals shared futures; structure beats table/random/destroyed",
    LEG_RL: "policy value beats table and random on randomized/known-effect data",
    LEG_SIMULATOR: "recovers the built-in counterfactual effects (red_king role)",
}


@dataclass(frozen=True)
class LegRow:
    """One card row = one leg x one column. Source receipt: judge_leg_<leg>.json."""

    leg: str
    column: str

    @property
    def ident(self) -> str:
        return f"JUDGE-{self.leg}-{self.column}"

    @property
    def name(self) -> str:
        return f"{self.leg} :: {COLUMN_LABEL[self.column]}"

    @property
    def source(self) -> str:
        return f"judge_leg_{self.leg}.json"

    @property
    def gate(self) -> str:
        return "every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain"


LEG_ROWS = tuple(LegRow(leg, col) for leg in LEGS for col in COLUMNS)


# ---------------------------------------------------------------- the stage ladder (S1-S6)


@dataclass(frozen=True)
class StageRow:
    ident: str
    name: str
    gate: str
    source: str = "judge_stages.json"


STAGE_ROWS = (
    StageRow(
        "S1",
        "Schema resolution (universal input)",
        "renamed-schema + new-field pass, zero code edits",
    ),
    StageRow(
        "S2",
        "Vocabulary & numeric encoding",
        "unseen-value path exercised; transform receipts; permutation invariance <= tol",
    ),
    StageRow(
        "S3",
        "Lawful dynamics (the engine)",
        "merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0",
    ),
    StageRow(
        "S4",
        "Buckets (the outlets)",
        "per-bucket accessibility parity + per-bucket negative controls",
    ),
    StageRow(
        "S5",
        "The objective battery",
        "every content pressure beats its naive comparator (or is excluded and reported)",
    ),
    StageRow(
        "S6",
        "Certification (the card)",
        "one command regenerates the card at a recorded commit + hashes",
    ),
)


# ---------------------------------------------------------------- certification hygiene


@dataclass(frozen=True)
class HygieneRow:
    ident: str
    name: str
    gate: str


HYGIENE_ROWS = (
    HygieneRow("H1", "Environment receipted", "env_receipt.json present, with a real device smoke"),
    HygieneRow(
        "H2",
        "Behavior hash recorded",
        "every source receipt carries a behavior_hash equal to the current one",
    ),
    HygieneRow("H3", "N recorded", "no PASS whose sample count is missing"),
    HygieneRow(
        "H4",
        "Controls present",
        "no PASS without random + destroyed controls at or below the brain",
    ),
    HygieneRow(
        "H5",
        "Comparator finite and fair",
        "no PASS with comparator n/a, nan, self, or a saturated baseline",
    ),
)


# ---------------------------------------------------------------- verdicts & the frozen rules

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"
EXCLUDED = "EXCLUDED"

#: values that must never appear as a real comparator or baseline (certification-integrity rule 5)
INVALID_COMPARATORS = {"", "n/a", "na", "nan", "none", "null", "self", "itself", "-", "tbd"}

#: a receipt that invents its own criterion is not evidence (rule 3)
CRITERION_KEYS = {"criterion", "rule", "gate", "gate_text", "threshold_rule"}

#: minimum fields every task in a leg column must carry to be evaluable at all
REQUIRED_TASK_FIELDS = (
    "name",
    "brain",
    "pathA",
    "ci_lo",
    "ci_hi",
    "n",
)

#: fields a receipt may not add without pre-registration
UNREGISTERED_NOTE = "receipt defines its own criterion"

#: A negative control whose held-out R2 falls below this is VACUOUS: it is not evidence that the
#: real arm has signal, only that the control was absurd. R2 = 1 - SSE/SST, so -1.0 is the point
#: where the control's squared error exceeds the target's own variance by more than that variance
#: again — predictions wilder than "predict the mean" by a full factor of the target's spread.
#: A gate satisfied by such a control ("control <= brain") is satisfied trivially, which is the
#: false green this constant exists to refuse (DEC-433). It is an evaluation-contract constant,
#: not a tuned threshold: nothing is ever adjusted to make a row pass.
VACUOUS_CONTROL_R2 = -1.0

#: Delivery is a NON-INFERIORITY claim ("never materially worse than the table"), not a
#: superiority claim. The margin is the evaluation-contract materiality floor: 0.01 R^2
#: (or 0.01 in the task's own units when no SST is recorded). Progress stays a strict win.
#: Frozen here, before the run; changing it is a gate-touching operator decision.
DELIVERY_MARGIN_R2 = 0.01
DELIVERY_MARGIN_RAW = 0.01

#: What "behavior change" means, mechanically (DEC-390 freshness law: any behavior change to the
#: encoder or serving path marks dependent receipts stale). Derived from the package tree rather
#: than a curated file list, so a new module cannot be accidentally excluded from the law.
BEHAVIOR_TREES = ("looking_glass", "rabbit_hole")
BEHAVIOR_SUFFIXES = (".py",)
