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
> **Status:** FULL BACKWARD review 2026-11-04 at commit `560a6c4`, behavior
> `df73c2aa1427b471`, artifact tag `0.11.1r490760-b64d256e3lr4e-05o4c3b5899`, bank registry
> `919a932f1b657064`, card `196c043bcf9d171b`.
> **The completion claim is NOT 100%: 5 BREAKING + 3 MAJOR blocking findings remain (B-004/B-010 closed; B-011 minor).** The card's
> `SOLVED` root is a plan-status render; the completion condition below is blocked by the
> open findings. This review is adversarial and read-only.

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
| P1 | **Universal** — one architecture, any event stream; per-stream model; schema-agnostic | renamed-schema + new-field pass, zero code edits; no domain literals; unseen stream refuses, never cross-applies | **gap** | B-001 |
| P2 | **Adaptive & dynamic** — everything resolves or learns from input | no fixed dims/layers/epochs/thresholds; config derived at fit; runtime toggles; `principle_audit.py` clean | not yet audited | — |
| P3 | **Lawful** — compositionality, equivariance, correct algebra per quantity | merge/equivariance/mass-additivity receipts; per-sample contract; no batch-coupled serving statistics; additive mass scored in raw units | not yet audited | — |
| P4 | **Efficient** — linear time, constant state, incremental, minimal capacity | no per-row Python loops in hot paths; scans/gathers/einsum; O(1) serve; derived budgets; caps enforced | not yet audited | — |
| P5 | **Robust** — fail safe, reject not clamp, calibrated uncertainty | determinism; divergence = hard failure; no silent clamps/NaNs; widened uncertainty on OOD; caps/co-tenancy | not yet audited | — |
| P6 | **Honest** — grounded, falsifiable, label-free upstream | pre-registration before the run; frozen acceptance; paired CIs; controls; no training on the graded metric; no post-hoc gate edits | **gap** | B-002–B-005, B-008–B-010 |
| P7 | **Accessible** — recoverable by simple consumers | linear shipping head; kNN unsupervised; MLP diagnostic only; decodability/parity receipts; no bucket dilution hidden | **gap** | B-008 |
| P8 | **Diagnosable** — attribute a shortfall to an axis and propose the fix | every red row names its axis; protocol names one axis; rejected alternatives recorded; coupling disclosed | **gap** | B-002, B-003, B-007 |

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
| E9 | Measure everything; gate on ground truth | Is every capability independently validated with known truth? | **gap** | B-005, B-006 |
| E10 | Reproducibility and versioning | Deterministic, seeded, hash/version/commit recorded; one command regenerates | **gap** | B-004, B-009, B-010 |
| E11 | Observability | Receipts say what was derived/measured/assumed/fallback-fired | not yet audited | — |
| E12 | Implement intent, not the example | Is the mechanism adaptive rather than a copied literal? | not yet audited | — |
| E13 | Prefer deletion over accretion | Dead code/superseded paths/one-off special cases removed? | not yet audited | — |
| E14 | Use the best tool | Mature library over hand-rolled where it fits; reimplementation justified | not yet audited | — |
| E15 | Cost-aware capability | Capability per unit cost; caching/bulk/columnar; no needless capacity | not yet audited | — |
| E16 | Speed is first-class | Vectorized by default; no per-row loops in hot paths; runtime receipt; scales predictably | not yet audited | — |

---

## 3. Open findings

| ID | Severity | Principle | Finding | Evidence | Requirement |
|---|---|---|---|---|---|
| B-001 | **BREAKING** | P1 / P6 | The certified artifact is the synthetic demo, not a real production stream. `cfm_v2k_vf2` and the bank both resolve `db=/…/demo_v2/customer_event_stream.parquet`; `reference_truth.meta.generator=layer_a_v2_prototype` (seed 7). The card's "real stream" language is unsupported. | registry `db`; `reference_truth.json` meta | M-001: run the B/C/E1/v1 gates on the real production stream, or scope every claim explicitly to the demo |
| B-002 | **BREAKING** | P6 / P8 | E1.a/E1.b do not read the frozen served encoder. `scripts/lab_measure.py::_measure_pair_hier` builds the prediction from the fixture's per-customer transition counts (`hist`), never from `state_embeddings`; the product is used only for the row intersection and `eff_rank`. The E1.a/E1.b `mastered` is a hand-built pair-counter head, not a trunk readout. | `lab_measure.py:246-340`; `positive.products` is null in the E1 receipts | M-002: serve the signature as a first-class lab-E block and evaluate the frozen E; or reclassify E1.a/E1.b as a signature-head gate |
| B-003 | **BREAKING** | P6 / P8 | E1.c is a ledger-backed port, and the receipt is unstamped. The port object is `bank 179 + measure PCA-128`; the bank's own effective rank is **10.327** (base **4.12**), while the port's **30.907** comes from the measure PCA. `lab/port_dec572.json` has no `_meta`/products. The B3.6 rule says a ledger-backed number is not a brain/trunk capacity win. | `lab/port_dec572.json`; bank covariance; `B3.6` | M-003: port the trunk's own state without the measure ledger and meet the bars, or record E1.c as a delivery/ledger port |
| B-005 | **BREAKING** | P6 / E9 | V1.3 red_king is synthetic. `v1_red_king.json` uses `red_king.scorecard.run` (`_bandit`, known synthetic rewards); the real `red_king/validate_known.py` still queries the retired `email_arm`/`email_sends` schema and was not run. The bank judge RL/simulator rows are the judge's evidence, not red_king's own gate. | `red_king/scorecard.py`; `red_king/validate_known.py`; `v1_red_king.json` | M-005: run red_king's known-effect gate on the bank/demo randomized stream and record the real receipt |
| B-006 | **MAJOR** | P6 / E9 | V1.4 red_queen reads a stored old certificate. `red_queen/certification/channel_certification.json` has `behavior` 2.69 vs `bar` 5.64 with `deploy: true`, provenance `estimated`; it was not regenerated on the bank/demo. | the certificate file; `v1_red_queen.json` | M-006: regenerate the channel/discount composition certification on the current bank/demo |
| B-007 | **MAJOR** | P6 / P8 | V1.5's plan is synthetic. `v1_nba_plan.json` is built by `measure_v1_stack._v15_plan` from `reference_truth.offer_opt` and the response `outcome_mean`, not from a red_queen decision plan. The explanation is artifact-only, but the artifact is not the stack's decision. | `v1_caterpillar.json`; `scripts/measure_v1_stack.py::_v15_plan` | M-007: run caterpillar on a real red_queen NBA plan |
| B-008 | **BREAKING** | P6 / P7 | The B+C `brain` column is a ledger hybrid. `cfm_v2k_vf2_bank/state_embeddings` includes the measure parts `numeric_stats` (60), `event_mass` (7) and `return_ratio` (6), plus the schema profile; the B3.6 rule says the ledger is a companion, not a brain win. The bank-v2 `brain` 8/8 is partly ledger-backed. | `build_bank_e.py`; bank `bank_receipt.json`; `B3.6` | M-008: separate the donor block bank from the measure ledger and re-score the brain column; or pre-register the bank as the donor and update B3.6 |
| B-009 | **MAJOR** | P6 / E10 | The B gate is a mixed-artifact certification. B2/B3.6 read `cfm_v2k_vf2`; B3.2-B3.5 read the port object; B5.1/B6.2/B6.3/B4.2 read the bank; B4.1/B4.3 read the base; B1.2/B5.2/B7/B8 read the bank. "B CERTIFIED" is not one artifact. | the B receipts' `products` fields; card B scope note | M-009: certify B on one served artifact |
| B-011 | MINOR | P6 | The earlier `AUDIT.md` status claimed "reviewed, no findings" while the v1 receipts were unstamped and the E1.b receipt stale. The review was a declaration, not an audit. | this file's previous status; v1 JSONs | M-011: keep the review verdict tied to the receipt stamps |

## 4. Closed findings

| ID | Closed by | Receipt / DEC | Date | Behavior hash |
|---|---|---|---|---|
| B-004 | v1 gate receipts re-written through `stamp_merge_write` with `_meta` and bank products identity | `v1_*.json` | 2026-11-04 | `df73c2aa` |
| B-010 | E1.b intent/intent_long re-run at the current behavior; g −0.721 / 0.181, mastered true | `lab/intent_dec601.json`, `lab/intent_long_dec601.json` | 2026-11-04 | `df73c2aa` |

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
  "requirements": [
    {"id":"B-001","severity":"BREAKING","principle":"P1/P6","status":"open","finding":"The certified artifact is the synthetic demo, not a real production stream.","forward_subgoal":"Run the B/C/E1/v1 gates on the real production stream, or scope every claim to the demo.","gate_touched":"B/B+C/E1/v1","evidence":"cfm_v2k_vf2 registry db=demo_v2/customer_event_stream.parquet; reference_truth.meta.generator=layer_a_v2_prototype","audit_ref":"A-001","date":"2026-11-04"},
    {"id":"B-002","severity":"BREAKING","principle":"P6/P8","status":"open","finding":"E1.a/E1.b read the fixture pair counts, not the frozen served encoder.","forward_subgoal":"Serve the signature as a lab-E block and evaluate the frozen E, or reclassify E1.a/E1.b.","gate_touched":"E1.a/E1.b","evidence":"lab_measure.py::_measure_pair_hier uses hist; positive.products null","audit_ref":"A-002","date":"2026-11-04"},
    {"id":"B-003","severity":"BREAKING","principle":"P6/P8","status":"open","finding":"E1.c is a ledger-backed port; receipt unstamped.","forward_subgoal":"Port the trunk's own state without the measure ledger, or record E1.c as a delivery port.","gate_touched":"E1.c","evidence":"port_dec572.json no _meta/products; bank eff_rank 10.327 vs port 30.907","audit_ref":"A-003","date":"2026-11-04"},
    {"id":"B-004","severity":"BREAKING","principle":"P6/E10","status":"closed","finding":"The five v1 receipts are unstamped and lack products identity.","forward_subgoal":"Stamp/re-run the v1 gates at the recorded behavior/commit/products.","gate_touched":"V1.1-V1.5","evidence":"v1_*.json now carry _meta and the bank products identity","audit_ref":"A-004","date":"2026-11-04"},
    {"id":"B-005","severity":"BREAKING","principle":"P6/E9","status":"open","finding":"V1.3 red_king is synthetic; the real known-effect path is not run.","forward_subgoal":"Run red_king's known-effect gate on the bank/demo randomized stream.","gate_touched":"V1.3","evidence":"red_king/scorecard.py _bandit; validate_known.py old email_arm/email_sends","audit_ref":"A-005","date":"2026-11-04"},
    {"id":"B-006","severity":"MAJOR","principle":"P6/E9","status":"open","finding":"V1.4 red_queen reads a stored old certificate with behavior 2.69 vs bar 5.64 and deploy true.","forward_subgoal":"Regenerate the channel/discount composition certification on the current bank/demo.","gate_touched":"V1.4","evidence":"red_queen/certification/channel_certification.json","audit_ref":"A-006","date":"2026-11-04"},
    {"id":"B-007","severity":"MAJOR","principle":"P6/P8","status":"open","finding":"V1.5's plan is synthetic (offer_opt + outcome_mean), not a red_queen decision plan.","forward_subgoal":"Run caterpillar on a real red_queen NBA plan.","gate_touched":"V1.5","evidence":"measure_v1_stack.py::_v15_plan; v1_caterpillar.json","audit_ref":"A-007","date":"2026-11-04"},
    {"id":"B-008","severity":"BREAKING","principle":"P6/P7","status":"open","finding":"The B+C brain column includes measure-ledger parts (numeric_stats/event_mass/return_ratio).","forward_subgoal":"Separate the donor block bank from the measure ledger and re-score; or pre-register the bank as the donor and update B3.6.","gate_touched":"B+C","evidence":"build_bank_e.py; bank_receipt.json; B3.6","audit_ref":"A-008","date":"2026-11-04"},
    {"id":"B-009","severity":"MAJOR","principle":"P6/E10","status":"open","finding":"The B gate is a mixed-artifact certification (base, port, bank).","forward_subgoal":"Certify B on one served artifact.","gate_touched":"B","evidence":"B receipts' products fields","audit_ref":"A-009","date":"2026-11-04"},
    {"id":"B-010","severity":"MAJOR","principle":"P6/E10","status":"closed","finding":"E1.b positive receipts are stale (behavior c6f56a2e vs df73c2aa).","forward_subgoal":"Re-run intent/intent_long hier at the current behavior.","gate_touched":"E1.b","evidence":"lab/intent_dec601.json, lab/intent_long_dec601.json at df73c2aa","audit_ref":"A-010","date":"2026-11-04"},
    {"id":"B-011","severity":"MINOR","principle":"P6","status":"open","finding":"The earlier audit status claimed no findings while the v1 receipts were unstamped and E1.b stale.","forward_subgoal":"Tie the review verdict to receipt stamps.","gate_touched":"AUDIT","evidence":"previous AUDIT status; v1 JSONs","audit_ref":"A-011","date":"2026-11-04"}
  ]
}
-->
