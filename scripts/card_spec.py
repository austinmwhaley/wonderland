"""Frozen gate spec for the Layer-B (encoder) report card.

Gates live HERE, in code -- never in a receipt (certification-integrity rule).
The renderer (`scripts/report_card.py`) resolves each row's source receipt and
evaluates the gate. A missing source => NOT MEASURED (never silently dropped).

Sections: construction (invariants) -> donor (does not lose / adds / controls)
          -> axes (8 content axes) -> objectives (the self-supervised battery).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    section: str
    ident: str
    name: str
    metric: str
    comparator: str
    gate: str
    source: tuple | None  # (receipt filename, dotted path)
    kind: str  # le | eq | present | gt | ge | ci_pos | ci_excl
    thresh: float = 0.0


AXES = "axes"
CONSTRUCTION = "construction"
DONOR = "donor"
OBJECTIVES = "objectives"

ROWS = [
    # ---- construction: invariants by construction -------------------------------
    Row(CONSTRUCTION, "C1", "Field-permutation equivariance",
        "max |E(x) - E(perm(x))|", "exact (0)", "<= 1e-6",
        ("encoder_lawfulness.json", "field_perm_delta"), "le", 1e-6),
    Row(CONSTRUCTION, "C2", "Scan (monoid) exactness",
        "max |scan(g o f) - scan(g) o scan(f)|", "exact (0)", "<= 1e-12",
        ("encoder_lawfulness.json", "scan_merge_delta"), "le", 1e-12),
    Row(CONSTRUCTION, "C3", "Mass additivity (extensive algebra)",
        "|mass(a+b) - mass(a) - mass(b)|", "exact (0)", "== 0",
        ("encoder_b_props.json", "mass_additivity_error"), "le", 1e-9),
    Row(CONSTRUCTION, "C4", "Determinism (same seed => identical)",
        "val trajectory bit-identical across repeats", "itself", "identical",
        ("encoder_construction.json", "determinism.identical"), "bool"),
    Row(CONSTRUCTION, "C5", "Schema-agnosticism (variant, zero code change)",
        "E trains on renamed schema + new field", "raw v3", "pass",
        ("encoder_construction.json", "schema_variant.pass"), "bool"),
    Row(CONSTRUCTION, "C6", "Frozen + reproducible artifact",
        "artifact hash recorded", "present", "hash present",
        ("encoder_b_props.json", "artifact_hash"), "present"),

    # ---- donor: does not lose / adds / controls ---------------------------------
    Row(DONOR, "D1", "E vs raw table (6k)",
        "value/orders/churn R2-AUC margin", "raw table", "E >= raw on all",
        ("encoder_6k_auto_3seed.json", "0"), "donor_raw"),
    Row(DONOR, "D2", "E vs fair table (6k)",
        "value/orders/churn margin", "fair table", "E >= fair on all",
        None, "unmeasured"),
    Row(DONOR, "D3", "E adds over table+sequence",
        "R2/auc gain of table+seq+E vs table+seq", "table+seq", "delta > 0 (CI)",
        ("encoder_geometry_base_1seed.json", "0"), "donor_add"),
    Row(DONOR, "D4", "Negative control: random E",
        "same battery with random E", "learned E", "random < learned (all 3)",
        ("encoder_controls.json", ""), "ctrl_random"),
    Row(DONOR, "D5", "Negative control: destroyed E",
        "same battery with shuffled E dims", "learned E", "destroyed < learned (all 3)",
        ("encoder_controls.json", ""), "ctrl_destroyed"),

    # ---- axes: 8 content axes ---------------------------------------------------
    Row(AXES, "A1", "Shape", "retrieval / churn-nn / eff-rank", "table (n/a)",
        "retrieval > 0 and churn-nn > 0.6", ("demo_run.json", "runs.0.retrieval"), "gt", 0.0),
    Row(AXES, "A2", "Mass", "past-magnitude reconstruction corr2; additivity", "oracle",
        "corr2 >= 0.4 and additivity exact", ("encoder_controls.json", "retention.past_r2"), "ge", 0.4),
    Row(AXES, "A3", "Identity", "kNN channel + category purity", "table",
        "both >= table", ("demo_run.json", "runs.0"), "identity2"),
    Row(AXES, "A4", "Rate", "churn AUC", "table",
        "E >= table", ("demo_run.json", "runs.0"), "rate2"),
    Row(AXES, "A5", "Uncertainty", "ECE; OOD retention", "coverage target",
        "ECE <= 0.1 and OOD retained", ("encoder_b_props.json", "ece_next_event"), "le", 0.1),
    Row(AXES, "A6", "Causal", "IPS known-effect recovery", "random; table",
        "E recovers effect (CI excludes 0)", ("encoder_causal_ci.json", "0"), "causal"),
    Row(AXES, "A7", "Order/Path", "event-shuffle drop", "table (invariant)",
        "E drops, table does not", ("demo_run.json", "runs.0"), "shuffle2"),
    Row(AXES, "A8", "Relational/Context", "cross-entity purity", "chance",
        "> chance", ("encoder_order_rel_probe.json", ""), "relational"),

    # ---- objectives: the self-supervised battery --------------------------------
    Row(OBJECTIVES, "O1", "next", "linear probe next-event skill", "naive (marginal)", "beats naive", ("encoder_objectives.json", "objectives.next"), "obj"),
    Row(OBJECTIVES, "O2", "dt", "gap/rate probe skill", "naive (prev gap)", "beats naive", ("encoder_objectives.json", "objectives.dt"), "obj"),
    Row(OBJECTIVES, "O3", "jepa", "latent future prediction", "naive (hist mean)", "beats naive", ("encoder_objectives.json", "objectives.jepa"), "obj"),
    Row(OBJECTIVES, "O4", "mask", "masked-event recovery", "naive (marginal)", "beats naive", ("encoder_objectives.json", "objectives.mask"), "obj"),
    Row(OBJECTIVES, "O5", "value", "value probe skill", "naive (hist mean)", "beats naive", ("encoder_objectives.json", "objectives.value"), "obj"),
    Row(OBJECTIVES, "O6", "entity", "entity/identity probe", "naive (marginal)", "beats naive", ("encoder_objectives.json", "objectives.entity"), "obj"),
    Row(OBJECTIVES, "O7", "order", "temporal-order probe", "naive (random)", "beats naive", ("encoder_objectives.json", "objectives.order"), "obj"),
    Row(OBJECTIVES, "O8", "agg", "aggregate/window probe", "naive (hist mean)", "beats naive", ("encoder_objectives.json", "objectives.agg"), "obj"),
    Row(OBJECTIVES, "O9", "query", "query-conditioned probe", "naive (hist mean)", "beats naive", ("encoder_objectives.json", "objectives.query"), "obj"),
    Row(OBJECTIVES, "O10", "sf", "successor-feature probe", "naive (prev value)", "beats naive", ("encoder_objectives.json", "objectives.sf"), "obj"),
]
