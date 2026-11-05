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
> **Status:** FULL BACKWARD re-review 2026-11-04 at commit `e72c795`, behavior
> `df73c2aa1427b471`, artifact tag `0.11.1r490760-b64d256e3lr4e-05o4c3b5899`, bank registry
> `919a932f1b657064`, certification bundle `cfm_v2k_vf2_cert` registry `116087fb206c2d81`.
> **All 11 findings are closed:** 8 by evidence (B-002, B-004, B-005, B-006, B-007, B-009, B-010,
> B-011) and 3 by the explicit scope/reclassification their requirements allow (B-001, B-003,
> B-008). The residual scope is stated with each closure: the certified artifact is the synthetic
> `demo_v2` prototype; E1 is served-object capacity (signature head + delivery/ledger port), not a
> trunk-only capacity win; the B+C brain is the served bank, with the measure-free donor-only
> diagnostic recorded beside it. The completion condition is green only for that scope.

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
| P1 | **Universal** — one architecture, any event stream; per-stream model; schema-agnostic | renamed-schema + new-field pass, zero code edits; no domain literals; unseen stream refuses, never cross-applies | **aligned (scope: demo)** | B-001 closed |
| P2 | **Adaptive & dynamic** — everything resolves or learns from input | no fixed dims/layers/epochs/thresholds; config derived at fit; runtime toggles; `principle_audit.py` clean | not yet audited | — |
| P3 | **Lawful** — compositionality, equivariance, correct algebra per quantity | merge/equivariance/mass-additivity receipts; per-sample contract; no batch-coupled serving statistics; additive mass scored in raw units | not yet audited | — |
| P4 | **Efficient** — linear time, constant state, incremental, minimal capacity | no per-row Python loops in hot paths; scans/gathers/einsum; O(1) serve; derived budgets; caps enforced | not yet audited | — |
| P5 | **Robust** — fail safe, reject not clamp, calibrated uncertainty | determinism; divergence = hard failure; no silent clamps/NaNs; widened uncertainty on OOD; caps/co-tenancy | not yet audited | — |
| P6 | **Honest** — grounded, falsifiable, label-free upstream | pre-registration before the run; frozen acceptance; paired CIs; controls; no training on the graded metric; no post-hoc gate edits | **aligned (scope: demo)** | B-002–B-005, B-008–B-010 closed |
| P7 | **Accessible** — recoverable by simple consumers | linear shipping head; kNN unsupervised; MLP diagnostic only; decodability/parity receipts; no bucket dilution hidden | **aligned (boundary recorded)** | B-008 closed |
| P8 | **Diagnosable** — attribute a shortfall to an axis and propose the fix | every red row names its axis; protocol names one axis; rejected alternatives recorded; coupling disclosed | **aligned** | B-002, B-003, B-007 closed |

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
| E9 | Measure everything; gate on ground truth | Is every capability independently validated with known truth? | **aligned** | B-005, B-006 closed |
| E10 | Reproducibility and versioning | Deterministic, seeded, hash/version/commit recorded; one command regenerates | **aligned** | B-004, B-009, B-010 closed |
| E11 | Observability | Receipts say what was derived/measured/assumed/fallback-fired | not yet audited | — |
| E12 | Implement intent, not the example | Is the mechanism adaptive rather than a copied literal? | not yet audited | — |
| E13 | Prefer deletion over accretion | Dead code/superseded paths/one-off special cases removed? | not yet audited | — |
| E14 | Use the best tool | Mature library over hand-rolled where it fits; reimplementation justified | not yet audited | — |
| E15 | Cost-aware capability | Capability per unit cost; caching/bulk/columnar; no needless capacity | not yet audited | — |
| E16 | Speed is first-class | Vectorized by default; no per-row loops in hot paths; runtime receipt; scales predictably | not yet audited | — |

---

## 3. Open findings

*None recorded. All 11 findings are closed below; the residual scope is stated with each closure.*

## 4. Closed findings

| ID | Closed by | Receipt / DEC | Date | Behavior hash |
|---|---|---|---|---|
| B-001 | **Scope decision (allowed by M-001):** every claim is explicitly the synthetic `demo_v2` prototype; the card/PLAN no longer claim a production stream. The bundle records `scope=synthetic demo_v2 prototype`. | `cfm_v2k_vf2_cert/bundle_receipt.json`; DEC-605 | 2026-11-04 | `df73c2aa` |
| B-002 | **Served signature block:** the lab products now carry `signature_blocks`; `_measure_pair_hier` reads it and refuses without the fixture/block hashes. E1.a/E1.b re-run at `dec603` with identical g, products identity and block hash. | `scripts/build_lab_signature.py`; `lab/next_event_x4k_dec603.json`, `lab/intent_dec603.json`, `lab/intent_long_dec603.json`; DEC-603 | 2026-11-04 | `df73c2aa` |
| B-003 | **Reclassified as a delivery/ledger port (allowed by M-003):** `port_dec604.json` records the bars, the delivery port (30.9/0.267/0.460) and every component. No measure-free object clears all three: shape32 31.9 rank but 0.076 persona/0.400 R2; donor106 11.76 rank; bank179 10.33 rank. | `lab/port_dec604.json`; DEC-604 | 2026-11-04 | `df73c2aa` |
| B-004 | **Stamped v1 receipts:** re-written through `stamp_merge_write` with `_meta` and the bank products identity. | `v1_*.json` | 2026-11-04 | `df73c2aa` |
| B-005 | **Real red_king gate:** per-customer randomized send-day estimator; 73.3% vs 25% chance (majority 25.4%), held-out policy 0.371 vs 0.208 random, CI [+0.155, +0.173]. `v1_red_king.json` reads this receipt. | `red_king_known.json`; `red_king/demo_effect.py`; `scripts/measure_red_king_known.py` | 2026-11-04 | `df73c2aa` |
| B-006 | **Certificate regenerated on demo:** email/sms/push offer policies DEPLOY (5.23/5.39/4.99 vs 3.05/3.10/2.87 behavior, CI > 0); discount/frequency HOLD/not identified. | `red_queen/certification/channel_certification.json`; `scripts/certify_demo_actions.py` | 2026-11-04 | `df73c2aa` |
| B-007 | **Real plan:** `red_queen/nba_demo.py` builds a 5000-customer plan from the randomized panel + certificate; caterpillar explains `v20000000` from that plan and the bank anchors, arm/GP match. | `red_queen/artifacts/nba_demo_plan.json`; `v1_caterpillar.json` | 2026-11-04 | `df73c2aa` |
| B-008 | **Boundary recorded, not hidden (allowed by M-008):** the B+C brain is the served bank 179; the measure-free donor-only 106 rerun is in `bc_donor_diagnostic.json` and fails unsupervised persona/future and supervised h0/return_proneness. The bundle maps both components and B3.6 carries the boundary. Closed as a scope correction (served-bank result, not encoder-alone). | `bc_donor_diagnostic.json`; `cfm_v2k_vf2_cert/bundle_receipt.json`; DEC-605 | 2026-11-04 | `df73c2aa` |
| B-009 | **One artifact:** `cfm_v2k_vf2_cert` contains base 272 / bank 179 / donor-only 106 / port 307 / measure view and a manifest mapping every B row to its component; bundle registry `116087fb206c2d81`. | `cfm_v2k_vf2_cert/bundle_receipt.json`; DEC-605 | 2026-11-04 | `df73c2aa` |
| B-010 | **E1.b re-run** at the current behavior; g −0.721 / 0.181, mastered true. | `lab/intent_dec601.json`, `lab/intent_long_dec601.json` | 2026-11-04 | `df73c2aa` |
| B-011 | **Review verdict tied to stamps:** this re-review records the behavior, commit, artifact, bank and bundle hashes; the earlier declaration is superseded. | this file header; `REPORT_CARD.md` | 2026-11-04 | `df73c2aa` |

---|---|---|---|---|
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
    {
      "id": "B-001",
      "severity": "BREAKING",
      "principle": "P1/P6",
      "status": "closed",
      "closure": "scope",
      "finding": "The certified artifact is the synthetic demo, not a real production stream.",
      "forward_subgoal": "Scope every claim to the demo prototype (M-001 option 2).",
      "gate_touched": "B/B+C/E1/v1",
      "evidence": "cfm_v2k_vf2_cert bundle scope=synthetic demo_v2 prototype; card/PLAN text updated; DEC-605",
      "audit_ref": "A-001",
      "date": "2026-11-04"
    },
    {
      "id": "B-002",
      "severity": "BREAKING",
      "principle": "P6/P8",
      "status": "closed",
      "closure": "evidence",
      "finding": "E1.a/E1.b read the fixture pair counts, not the frozen served encoder.",
      "forward_subgoal": "Serve the signature as a lab-E block and evaluate the frozen E.",
      "gate_touched": "E1.a/E1.b",
      "evidence": "signature_blocks served by build_lab_signature.py; lab_measure reads it; dec603 receipts carry products + block hash; DEC-603",
      "audit_ref": "A-002",
      "date": "2026-11-04"
    },
    {
      "id": "B-003",
      "severity": "BREAKING",
      "principle": "P6/P8",
      "status": "closed",
      "closure": "scope",
      "finding": "E1.c is a ledger-backed port; receipt unstamped.",
      "forward_subgoal": "Record E1.c as a delivery/ledger port (M-003 option 2).",
      "gate_touched": "E1.c",
      "evidence": "lab/port_dec604.json: delivery port 30.9/0.267/0.460; trunk shape32 31.9 rank but 0.076 persona/0.400 R2; donor106 11.76; bank179 10.33; plan E1 text updated; DEC-604",
      "audit_ref": "A-003",
      "date": "2026-11-04"
    },
    {
      "id": "B-004",
      "severity": "BREAKING",
      "principle": "P6/E10",
      "status": "closed",
      "closure": "evidence",
      "finding": "The five v1 receipts are unstamped and lack products identity.",
      "forward_subgoal": "Stamp/re-run the v1 gates at the recorded behavior/commit/products.",
      "gate_touched": "V1.1-V1.5",
      "evidence": "v1_*.json carry _meta and the bank products identity",
      "audit_ref": "A-004",
      "date": "2026-11-04"
    },
    {
      "id": "B-005",
      "severity": "BREAKING",
      "principle": "P6/E9",
      "status": "closed",
      "closure": "evidence",
      "finding": "V1.3 red_king is synthetic; the real known-effect path is not run.",
      "forward_subgoal": "Run red_king's known-effect gate on the demo randomized stream.",
      "gate_touched": "V1.3",
      "evidence": "red_king_known.json: 73.3% vs 25% chance; policy 0.371 vs 0.208 random, CI [+0.155,+0.173]",
      "audit_ref": "A-005",
      "date": "2026-11-04"
    },
    {
      "id": "B-006",
      "severity": "MAJOR",
      "principle": "P6/E9",
      "status": "closed",
      "closure": "evidence",
      "finding": "V1.4 red_queen reads a stored old certificate with behavior 2.69 vs bar 5.64 and deploy true.",
      "forward_subgoal": "Regenerate the channel/discount composition certification on the current demo.",
      "gate_touched": "V1.4",
      "evidence": "channel_certification.json regenerated: email/sms/push offer DEPLOY (CI > 0), discount/frequency HOLD/not identified",
      "audit_ref": "A-006",
      "date": "2026-11-04"
    },
    {
      "id": "B-007",
      "severity": "MAJOR",
      "principle": "P6/P8",
      "status": "closed",
      "closure": "evidence",
      "finding": "V1.5's plan is synthetic (offer_opt + outcome_mean), not a red_queen decision plan.",
      "forward_subgoal": "Run caterpillar on a real red_queen NBA plan.",
      "gate_touched": "V1.5",
      "evidence": "red_queen/artifacts/nba_demo_plan.json built by red_queen/nba_demo.py; v1_caterpillar arm/GP match",
      "audit_ref": "A-007",
      "date": "2026-11-04"
    },
    {
      "id": "B-008",
      "severity": "BREAKING",
      "principle": "P6/P7",
      "status": "closed",
      "closure": "scope",
      "finding": "The B+C brain column includes measure-ledger parts.",
      "forward_subgoal": "Separate the donor block bank from the measure ledger; record the served-bank boundary (M-008 option 2).",
      "gate_touched": "B+C",
      "evidence": "bc_donor_diagnostic.json: donor-only fails unsupervised persona/future + supervised h0/return_proneness; bundle receipt names state_embeddings (bank 179) vs donor_embeddings (106); B3.6 text updated; DEC-605",
      "audit_ref": "A-008",
      "date": "2026-11-04"
    },
    {
      "id": "B-009",
      "severity": "MAJOR",
      "principle": "P6/E10",
      "status": "closed",
      "closure": "evidence",
      "finding": "The B gate is a mixed-artifact certification (base, port, bank).",
      "forward_subgoal": "Certify B on one served artifact.",
      "gate_touched": "B",
      "evidence": "cfm_v2k_vf2_cert/bundle_receipt.json: base 272 + bank 179 + donor-only 106 + port 307 + measure view; every B row mapped to its component",
      "audit_ref": "A-009",
      "date": "2026-11-04"
    },
    {
      "id": "B-010",
      "severity": "MAJOR",
      "principle": "P6/E10",
      "status": "closed",
      "closure": "evidence",
      "finding": "E1.b positive receipts are stale (behavior c6f56a2e vs df73c2aa).",
      "forward_subgoal": "Re-run intent/intent_long at the current behavior.",
      "gate_touched": "E1.b",
      "evidence": "lab/intent_dec601.json, lab/intent_long_dec601.json at df73c2aa",
      "audit_ref": "A-010",
      "date": "2026-11-04"
    },
    {
      "id": "B-011",
      "severity": "MINOR",
      "principle": "P6",
      "status": "closed",
      "closure": "evidence",
      "finding": "The earlier audit status claimed no findings while the v1 receipts were unstamped and E1.b stale.",
      "forward_subgoal": "Tie the review verdict to receipt stamps.",
      "gate_touched": "AUDIT",
      "evidence": "this review records behavior/commit/artifact/bank/bundle hashes; the earlier status is superseded",
      "audit_ref": "A-011",
      "date": "2026-11-04"
    }
  ]
}
-->
