# AUDIT — principle alignment and adversarial findings

> **Written by `PROMPT_BACKWARD.md`. Read by `PROMPT_FORWARD.md`.**
> This is the detailed, human-readable audit of **how the work aligns with the principles**
> (AGENTS.md 1–16 + the top-level charter). It is not a second report card: the machine-readable
> gate is the `BACKWARD_REQUIREMENTS_JSON` block at the end of **this file**, which
> `REPORT_CARD.md` renders. BACKWARD writes this one `.md` and nothing else. Every finding here
> carries the same `B-<id>` as its JSON requirement, so the two cannot drift.
> **This file carries no measurements.** The single measurement surface — every metric, verdict,
> comparator, CI and N — is `REPORT_CARD.md`; this file only records requirements and findings.
>
> **Status:** no all-green card has been reviewed by BACKWARD yet. The scorecard below is
> unfilled on purpose; a review that has not run may not claim alignment. The first
> `PROMPT_BACKWARD.md` run populates it, and `PROMPT_FORWARD.md` closes findings with receipts.

---

## 1. How to use this file

**BACKWARD (one file only).**
1. Verify the finding with a concrete check, command, or receipt — never a vibe.
2. Append the finding under **Open findings** here, and append the matching object to the
   `BACKWARD_REQUIREMENTS_JSON` block at the end of this same file:
   `id` (B-<n>), `severity`, `principle`, `status: open`, `finding`, `forward_subgoal`,
   `gate_touched`, `evidence`, `audit_ref: A-<n>`, `date`.
3. Mark the principle's scorecard row with the finding ID.
4. Write **no other file**. `docs/BACKWARD_CARD.md` and `docs/backward_requirements.json` are retired.

**FORWARD (close with evidence, same file).**
1. Run the pre-registered protocol that resolves the finding.
2. Set the requirement's `status` to `closed` in the JSON block (or, for a `MAJOR` only, ask
   the operator to set `accepted` with a reason).
3. Append a **Closure** entry under Closed findings with the receipt, date, and behavior hash.
   Never edit the original finding text.
4. `BREAKING` is never accepted, only closed by evidence. An open `BREAKING`/`MAJOR` blocks the
   completion condition in `REPORT_CARD.md`.

---

## 2. Principle scorecard

Status vocabulary: **aligned** (audited, evidence cited) · **gap** (real weakness, requirement
open) · **violated** (principle broken, requirement open) · **not yet audited**.

| # | Principle | What the audit must prove | Status | Finding(s) |
|---|---|---|---|---|
| P1 | **Universal** — one architecture, any event stream; per-stream model; schema-agnostic | renamed-schema + new-field pass, zero code edits; no domain literals; unseen stream refuses, never cross-applies | not yet audited | — |
| P2 | **Adaptive & dynamic** — everything resolves or learns from input | no fixed dims/layers/epochs/thresholds; config derived at fit; runtime toggles; `principle_audit.py` clean | not yet audited | — |
| P3 | **Lawful** — compositionality, equivariance, correct algebra per quantity | merge/equivariance/mass-additivity receipts; per-sample contract; no batch-coupled serving statistics; additive mass scored in raw units | not yet audited | — |
| P4 | **Efficient** — linear time, constant state, incremental, minimal capacity | no per-row Python loops in hot paths; scans/gathers/einsum; O(1) serve; derived budgets; caps enforced | not yet audited | — |
| P5 | **Robust** — fail safe, reject not clamp, calibrated uncertainty | determinism; divergence = hard failure; no silent clamps/NaNs; widened uncertainty on OOD; caps/co-tenancy | not yet audited | — |
| P6 | **Honest** — grounded, falsifiable, label-free upstream | pre-registration before the run; frozen acceptance; paired CIs; controls; no training on the graded metric; no post-hoc gate edits | not yet audited | — |
| P7 | **Accessible** — recoverable by simple consumers | linear shipping head; kNN unsupervised; MLP diagnostic only; decodability/parity receipts; no bucket dilution hidden | not yet audited | — |
| P8 | **Diagnosable** — attribute a shortfall to an axis and propose the fix | every red row names its axis; protocol names one axis; rejected alternatives recorded; coupling disclosed | not yet audited | — |

### AGENTS.md engineering principles (1–16)

| # | Principle | Audit question | Status | Finding(s) |
|---|---|---|---|---|
| E1 | No magic numbers | Is every threshold/budget/dim/rate derived, learned, or a documented fallback? | not yet audited | — |
| E2 | Train to convergence, not counts | Is the stop criterion held-out/grounded, not a fixed step count? | not yet audited | — |
| E3 | Make it learn; do not teach it | Are rules replaced by learned/calibrated behavior where labels exist? | not yet audited | — |
| E4 | Adapt to the input by construction | Does an unseen schema/dim/horizon/action space work with no code edits? | not yet audited | — |
| E5 | Modularity and contracts | One responsibility per module; no cross-layer imports; explicit interfaces | not yet audited | — |
| E6 | Single source of truth | Any duplicated definition/constant/artifact? Copies are bugs | not yet audited | — |
| E7 | Fail safe; reject, do not clamp | Any silent default/clamp/NaN rescue? | not yet audited | — |
| E8 | Uncertainty first | Every estimate has a calibrated interval/coverage; point estimates do not drive gates | not yet audited | — |
| E9 | Measure everything; gate on ground truth | Is every capability independently validated with known truth? | not yet audited | — |
| E10 | Reproducibility and versioning | Deterministic, seeded, hash/version/commit recorded; one command regenerates | not yet audited | — |
| E11 | Observability | Receipts say what was derived/measured/assumed/fallback-fired | not yet audited | — |
| E12 | Implement intent, not the example | Is the mechanism adaptive rather than a copied literal? | not yet audited | — |
| E13 | Prefer deletion over accretion | Dead code/superseded paths/one-off special cases removed? | not yet audited | — |
| E14 | Use the best tool | Mature library over hand-rolled where it fits; reimplementation justified | not yet audited | — |
| E15 | Cost-aware capability | Capability per unit cost; caching/bulk/columnar; no needless capacity | not yet audited | — |
| E16 | Speed is first-class | Vectorized by default; no per-row loops in hot paths; runtime receipt; scales predictably | not yet audited | — |

---

## 3. Open findings

*None recorded. The first `PROMPT_BACKWARD.md` review populates this table with verified findings
and matching `B-<id>` requirements.*

| ID | Severity | Principle | Finding | Evidence | Requirement |
|---|---|---|---|---|---|

---

## 4. Closed findings

*None recorded.*

| ID | Closed by | Receipt / DEC | Date | Behavior hash |
|---|---|---|---|---|

---

## 5. Mechanical baseline (disclosure, not a verdict)

| Check | Command | Baseline | Note |
|---|---|---|---|
| Principle audit (hot-path patterns, cadence/capacity literals, new criterion strings) | `python3 scripts/principle_audit.py looking_glass scripts --strict` | **0 findings** (2026-10-31) | Narrow mechanical check; not a substitute for the code audit |
| Lint | `ruff check .` | known pre-existing failures in archived/older scripts and tests | disclose; do not call the repo lint-clean until fixed |
| Tests | `pytest -q -m "not slow"` | known pre-existing failures (ledger recency expectation, gpu-budget direction, tolerance drift) | disclose; a green card does not make the suite green |
| Report card | `python3 scripts/measure_certification_law.py ...` | S6 6/6 at the recorded commit | the only S6-certifying path |

A BACKWARD finding must still be backed by a **specific** file:line or command; this table is
context, not evidence of alignment.

---

## 6. What a principle finding looks like

```text
ID: B-007
Severity: MAJOR
Principle: E16 / SPEED
Finding: `looking_glass/<module>.py:NNN` iterates rows in Python inside the hot path;
         measured 4.2 s per 1k rows and it is O(N) Python overhead.
Evidence: `python3 scripts/bench_<x>.py` -> 4.2 s/1k; profile shows the loop.
Forward subgoal: replace with a vectorized scan/gather; prove identical output and a
         measured speedup; keep the CPU fallback.
Gate touched: no
Audit ref: A-007
```

A finding is not a principle violation unless the audit can point to the principle, the
mechanism, and the consequence. Severity is about the **claim/goal**, not about style.

---

## 7. The completion condition (mirrors `REPORT_CARD.md`)

The card is not all green while any **BREAKING** or **MAJOR** finding here is open. A
`MAJOR` may be closed by a forwarding protocol or explicitly `accepted` by the operator;
a `BREAKING` may only be closed by evidence. `MINOR`/`INFO` findings are disclosed and may
remain open without blocking completion.

---

## 8. Machine-readable requirements (the only gate BACKWARD writes)

> **BACKWARD:** append each finding here with the same `B-<id>` as the Open findings table.
> **FORWARD:** close it here with evidence; do not delete the finding.

<!-- BACKWARD_REQUIREMENTS_JSON
{
  "requirements": []
}
-->
