# AUDIT — principle alignment and adversarial findings

> **Written by `PROMPT_BACKWARD.md`. Read by `PROMPT_FORWARD.md.`**
> This is the detailed, human-readable audit of **how the work aligns with the principles**
> (AGENTS.md 1–16 + the top-level charter). It is not a second report card: the machine-readable
> gate is the `BACKWARD_REQUIREMENTS_JSON` block at the end of **this file**, which
> `REPORT_CARD.md` renders. BACKWARD writes this one `.md` and nothing else. Every finding here
> carries the same `B-<id>` as its JSON requirement, so the two cannot drift.
> **This file carries no measurements.** The single measurement surface — every metric, verdict,
> comparator, CI and N — is `REPORT_CARD.md`; this file only records requirements and findings.
>
> **Status:** FULL BACKWARD review 2026-11-05 at commit `38740c8` (`foundation-rebuild`),
> behavior `df73c2aa1427b471`, artifact tag `0.11.1r490760-b64d256e3lr4e-05o4c3b5899`, bank
> registry `919a932f1b657064`, certification bundle `cfm_v2k_vf2_cert` registry
> `116087fb206c2d81`, card sha256 `3d025fbe3269fb2a` (committed at `38740c8`; the header-only
> re-render at the same tree is `46eedddfeaa74f2d`). Mode: **FULL** — the first FULL review at
> the SOLVED tree (the previous FULL, `dcad1bd`, ran at the pre-SOLVED tree `560a6c4`).
> **Closing FULL 2026-11-05 at the closure tree `eb9381b`** (§6.9): NO BREAKING, all MAJORs
> closed by evidence; the tick-tock stop condition (§6.8) is met at that tree.
> **Rescan 2026-11-05 at `b512068`** (operator-requested full adversarial pass; §6.10):
> NO BREAKING; one new MINOR finding (B-019 — the R7 route text in `plan_spec.py` lags
> this record), closed 2026-11-05 by DEC-610 (`2976485`).
> **Verdict: NO BREAKING FINDINGS.** Every one of the card's 39 atoms was re-derived from its
> receipt and frozen gate (§6.2–§6.4); no measurement is falsified. The findings were
> record-and-consistency findings: **3 MAJOR** — B-012 (the decision log is missing the entries
> the card/plan/status cite as load-bearing), B-013 (the unsupervised leg receipt contradicts
> itself and all four leg receipts carry flat-era `arms`), B-014 (two of the three AGENTS.md
> commit gates are red at the SOLVED commit) — **and 1 MINOR** (B-015, the bundle manifest is
> unstamped). **All four are closed 2026-11-05 by evidence** (closures: §4; code at
> `08e37e4`/`470bb5e`/`4cf477b`/`672c164`; closing FULL at the closure tree: §6.9). B-016
> (defects in this file itself) is closed by this rewrite. B-017/B-018 are disclosed INFO
> (non-blocking; the operator ruling on B-017 remains queued).

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
| P1 | **Universal** — one architecture, any event stream; per-stream model; schema-agnostic | renamed-schema + new-field pass, zero code edits; no domain literals; unseen stream refuses, never cross-applies | **aligned (scope: demo)** | B-001 closed; B-022 closed 2026-11-06 (DEC-614: derived monetary field, no-op on reference); B-023 open (MINOR); B-024/B-025 open (INFO, named frozen-contract scope) |
| P2 | **Adaptive & dynamic** — everything resolves or learns from input | no fixed dims/layers/epochs/thresholds; config derived at fit; runtime toggles; `principle_audit.py` clean | not yet audited | — |
| P3 | **Lawful** — compositionality, equivariance, correct algebra per quantity | merge/equivariance/mass-additivity receipts; per-sample contract; no batch-coupled serving statistics; additive mass scored in raw units | not yet audited | — |
| P4 | **Efficient** — linear time, constant state, incremental, minimal capacity | no per-row Python loops in hot paths; scans/gathers/einsum; O(1) serve; derived budgets; caps enforced | not yet audited | — |
| P5 | **Robust** — fail safe, reject not clamp, calibrated uncertainty | determinism; divergence = hard failure; no silent clamps/NaNs; widened uncertainty on OOD; caps/co-tenancy | not yet audited | — |
| P6 | **Honest** — grounded, falsifiable, label-free upstream | pre-registration before the run; frozen acceptance; paired CIs; controls; no training on the graded metric; no post-hoc gate edits | **aligned** | B-002–B-005, B-008–B-010 closed; B-012, B-013, B-014 closed 2026-11-05 by evidence (`470bb5e`/`4cf477b`/`08e37e4`); B-016 closed by this review; B-017 open (INFO); B-019 closed 2026-11-05 (DEC-610, `2976485`) |
| P7 | **Accessible** — recoverable by simple consumers | linear shipping head; kNN unsupervised; MLP diagnostic only; decodability/parity receipts; no bucket dilution hidden | **aligned (boundary recorded)** | B-008 closed |
| P8 | **Diagnosable** — attribute a shortfall to an axis and propose the fix | every red row names its axis; protocol names one axis; rejected alternatives recorded; coupling disclosed | **gap** | B-002, B-003, B-007 closed; B-017 open (INFO) |

### AGENTS.md engineering principles (1–16)

| # | Principle | Audit question | Status | Finding(s) |
|---|---|---|---|---|
| E1 | No magic numbers | Is every threshold/budget/dim/rate derived, learned, or a documented fallback? | not yet audited | — |
| E2 | Train to convergence, not counts | Is the stop criterion held-out/grounded, not a fixed step count? | not yet audited | — |
| E3 | Make it learn; do not teach it | Are rules replaced by learned/calibrated behavior where labels exist? | not yet audited | — |
| E4 | Adapt to the input by construction | Does an unseen schema/dim/horizon/action space work with no code edits? | **audited (partial)** | B-022 closed 2026-11-06 (DEC-614): the supervised RFM comparator now derives its monetary field, so a non-`value` stream is consumed, not crashed on; B-024/B-025 name the frozen-contract scopes that remain bounded by design |
| E5 | Modularity and contracts | One responsibility per module; no cross-layer imports; explicit interfaces | **gap** | B-023 open (MINOR): `onboard._derive_reward_event` accepts a `value_col` param it ignores — the signature does not name the canonical-frame contract it actually relies on |
| E6 | Single source of truth | Any duplicated definition/constant/artifact? Copies are bugs | not yet audited | — |
| E7 | Fail safe; reject, do not clamp | Any silent default/clamp/NaN rescue? | not yet audited | — |
| E8 | Uncertainty first | Every estimate has a calibrated interval/coverage; point estimates do not drive gates | not yet audited | — |
| E9 | Measure everything; gate on ground truth | Is every capability independently validated with known truth? | **aligned** | B-005, B-006 closed |
| E10 | Reproducibility and versioning | Deterministic, seeded, hash/version/commit recorded; one command regenerates | **aligned** | B-004, B-009, B-010 closed; B-013, B-014, B-015 closed 2026-11-05 (`4cf477b`/`08e37e4`/`672c164`); B-018 open (INFO) |
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
| B-017 | **INFO** | P6/P8 | The supervised leg's evidence discloses two open, non-gating questions: (i) negative controls are vacuous (R² < −1, a wide-design conditioning artifact) on 5/7 traits — the frozen `VACUOUS_CONTROL_R2` doctrine (`judge_spec.py:179–183`) says such controls prove nothing, and the gate does not count them; (ii) the ledger-only diagnostic beats the table on 4/7 traits (best R² 0.8659 on x0 vs the table's 0.8791), which falsifies SPINE §5's "the ledger is what the table already is" corollary and is recorded as "needs an operator ruling". | `judge_leg_supervised.json` `checks.supervised.evidence` (both WARNINGs); leg verdicts derive from `columns` via the frozen `_task_passes` (`report_card.py:829`), not from `checks` | None required (disclosure); the SPINE §5 ruling is queued to the operator. If the ruling revises the corollary, B3.6's branch text and the delivery framing update in a forward run. |
| B-018 | **INFO** | E10 | `plugin_acceptance.json` (the V1.2 reference receipt, n_rows 9) has no `_meta`; the gate receipt `v1_plugins.json` that V1.2's atom actually reads is stamped (behavior `df73c2aa`, `passed: true`, `n_rows: 9`). | key lists of the two receipts; `measure_v1_stack.py` stamps the gate receipt, not the reference | None required (disclosure); stamp the reference in a future producer pass if V1.2 is re-run. |
| B-023 | **MINOR** | E5/E12 | `onboard._derive_reward_event(df, value_col)` (`onboard.py:585`) accepts `value_col` but aggregates the literal `pl.col("value")` (`:590`); it is correct only because its input is pre-canonicalized to a `value` column (`:710`/`:757`) — the parameter is vestigial and the signature is dishonest (P12: implement intent, not the example). | `looking_glass/onboard.py:585-595` (param ignored; canonical frame always has `value`) | One-axis forward subgoal: make the signature honest (use `value_col`, or drop it and name the canonical-frame contract) with a no-op proof on the canonical frame; re-stamp. |
| B-024 | **INFO (named scope)** | P1 | The supervised LEG's target family is a frozen ecommerce evaluation contract: `_labels` reads `$.value`/`$.touchpoint`/`$.category_code` and filters `event_type == "purchase"/"return"` (`supervised_gate.py:37-55`); `TARGETS` (`:68-78`) names the 8 targets. A non-ecommerce reference stream is a new operator-level evaluation contract, not a pipeline fix — the comparator generality (B-022) is the pipeline half, already closed. | `looking_glass/supervised_gate.py:37-55,68-78` | None (disclosure): the leg's generality is bounded to the declared target contract; a new reference stream is a separate DEC. |
| B-025 | **INFO (named scope)** | P1 | The decision/RL LEG's reward contract is declared on the randomized demo stream: `_arms_rewards` hardcodes `event_type LIKE '%send%'` (the arm) and `event_type='purchase'` + `$.value` (the incremental-margin reward) (`decision_certify.py:48-59`). A different reward field or arm vocabulary is a new causality contract (AGENTS: evaluation targets + causality are non-negotiable fixed things), not a pipeline defect. | `looking_glass/decision_certify.py:48-59` | None (disclosure): the decision leg's generality is bounded to the declared arm/reward contract. |

---

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
| B-011 | **Review verdict tied to stamps:** the 75df2b0 re-review recorded the behavior, commit, artifact, bank and bundle hashes; the earlier declaration is superseded. | previous AUDIT header; `REPORT_CARD.md` | 2026-11-04 | `df73c2aa` |
| B-016 | **AUDIT document defect, closed by this rewrite (permitted for defects in BACKWARD's own file):** the previous §4 carried an orphaned `---|---|---|---|---|` separator with B-004/B-010 rows duplicated outside the closed table, and §5 disclosed fast-tier failures (ledger recency, gpu-budget direction, tolerance drift) that no longer exist while omitting the current red gates (now B-014). The repaired table and the refreshed baseline in this file are the evidence. | this file §4/§5 (rewritten 2026-11-05) | 2026-11-05 | `df73c2aa` |
| B-012 | **Decision-log record repair (evidence):** post-hoc-labeled entries DEC-598 (bank v2 judge protocol), DEC-599 (E1.c port PASS + branch-(iii) long fixtures + B re-stamp), DEC-600 (hierarchical Dirichlet signature head), DEC-601 (E1.a/E1.b green) inserted between DEC-596 and DEC-603, reconstructed only from the pre-registration surfaces (the `measure_bc_bank.py` docstring, the `lab_measure.py` protocol, the commit bodies of `e9dfd6b`/`e7918a4`/`580b62c`/`0bc0d1f`/`7a3ae15`, and STATUS) plus the receipts. DEC-597/602 sealed as reserved/unallocated (no trace in history; nothing fabricated). `tests/test_decision_log_citations.py` now fails the suite if any in-log-range DEC citation (elided forms expanded) has no entry or seal, and keeps the disciplined region (≥ DEC-590) monotone. Log diff purely additive: 155 insertions, 0 deletions. | DEC-607; `470bb5e`; test green (773 passed) | 2026-11-05 | `df73c2aa` |
| B-014 | **Repo commit gates green (evidence):** `ruff format` applied to the 9 `scripts/measure_*.py` files (argparse-default wrapping only; none in the behavior trees); `measure_port_demo.py` + `measure_red_king_known.py` added to `gpu_watchdog.DEFAULT_PATTERNS`; the three card-mechanics tests that read the live AUDIT.md as a fixture now isolate it (`monkeypatch RC.AUDIT_MD`), while the dedicated backward-gate tests keep pinning the open-requirement behavior. At the closure commit: `ruff check .` / `ruff format --check .` / `pytest -m "not slow"` all RC=0; `principle_audit --all --strict` 0-new; behavior hash `df73c2aa` unchanged. | DEC-606; `08e37e4` | 2026-11-05 | `df73c2aa` |
| B-013 | **Leg receipts self-consistent on the bank (evidence):** `measure_bc_bank.py` writes, for every leg it scores, a current `checks` rollup from its own `_evidence`, bank-v2 `arms` (one definition per leg, `BANK_ARMS`), and `path_a` from its own frozen-comparator metadata; `stamp_merge_write` gained `drop=` so the producer sheds the flat-era blocks it does not own (`diagnostics`, `incumbent`). All four legs re-run: `columns` byte-identical to the committed receipts (determinism proof — no verdict moves), `checks.<leg>.working == working` on all four (the unsupervised self-contradiction is gone), `arms` name the served block bank, no flat-era block survives, receipts re-stamped at the new commit. | DEC-608; `4cf477b` | 2026-11-05 | `df73c2aa` |
| B-015 | **Bundle manifest stamped (evidence):** `build_b_cert_bundle.py` writes `bundle_receipt.json` through the shared `stamp_merge_write` — the manifest now carries `_meta` (behavior `df73c2aa`, commit, producer, command) like every row receipt it maps. Builder re-run idempotently: bundle registry sha unchanged (`116087fb206c2d81` — the registry hashes `registry_{tag}.json`, not the receipt), 23-row map and scope string intact, card B scope line content-identical. | DEC-609; `672c164` | 2026-11-05 | `df73c2aa` |
| B-019 | **Route text states the durable outcome (evidence):** R7's evidence sentence in `plan_spec.py` now reads "every FULL BACKWARD at the SOLVED tree returned NO BREAKING (opening at `38740c8`; closing at `eb9381b`). All BREAKING/MAJOR findings are closed — by evidence, or by the explicit scope/reclassification their requirements allow (B-001, B-003, B-008); open items are disclosed non-blocking in AUDIT.md" — the DEC-605-era count is gone and no count that would go stale at the next finding was written in its place. PLAN.md and the card re-rendered from the spec; behavior hash `df73c2aa` unchanged; commit gates RC=0. | DEC-610; `2976485` | 2026-11-05 | `df73c2aa` |
| B-020 | **Operator-filed claim-scope finding, closed the same day by the DEC-611 campaign (evidence; filed by the operator 2026-11-05, NOT by a BACKWARD run — provenance at §6.11):** the closing BACKWARD reviews falsified the record's plumbing (citations, manifests, self-consistency, route text) but never the scope of the claim against the top-level Universal principle — every B-row and pre-DEC-611 lab fixture is one ecom simulator family on the `demo_v2` prototype; S1 is schema-variant, not structure-variant. The DEC-611 campaign is the empirical answer: three structurally distinct known-truth domains (sensor_fleet regime-ordering + continuous levels; clickstream_sessions bursty timing + rare class; server_logs change-point ordering + diurnal distractor) through the ONE simulator, full three-piece controls, frozen `lab_measure`; claims stay scoped to the `demo_v2` prototype per B-001, and the six receipts are dev instruments per DEC-576. Branch option (operator ruling, separate DEC): grow the tree so multi-domain mastery becomes a gated card row. | six `lab/*_dec611{,_hier}.json` receipts (`complete=true`); DEC-611 outcome addendum | 2026-11-05 | `89cec70b` |
| B-021 | **Stale legs-producer map in the certification CHAIN (fix + re-stamp evidence; forward-filed 2026-11-05 during closeout forensics, NOT by a BACKWARD run — provenance at §6.12):** the S6 one-command chain (`measure_certification_law.py`) still mapped the `legs` step to `measure_judge_law.py`, whose `main()` implements the pre-DEC-598 flat-era supervised/unsupervised legs and writes the SAME four `judge_leg_*.json` files that `measure_bc_bank.py` (bank v2, DEC-598) owns — so every clean-tree closeout re-stamp overwrote the bank-v2 receipts with stale v1 numbers (supervised x0 0.886, CI straddling 0; unsupervised persona_geometry 0.338608; segment_structure ARI ≈ 0) and flipped the plan root SOLVED 39/39 → 35/39. The card's own gate caught it (verdicts recompute from receipts; no narration could override). rl/simulator legs were unaffected (bc_bank delegates them to the judge_law `LEGS`). Fixed by the one-line CHAIN map `legs` → `measure_bc_bank.py` (script-only; behavior stays `89cec70b`; gate/criterion untouched); the clean one-command re-stamp at `0470c79` re-stamped all four legs with bank-v2 values byte-identical to the `9af0d0d` wave receipts and re-rendered the plan root SOLVED 39/39 (B CERTIFIED 23/23, 8/8; S6 OK; backward open 0). | DEC-612; `0470c79` (CHAIN fix, five gates RC=0); regenerated receipts + card in this commit | 2026-11-05 | `89cec70b` |
| B-022 | **Situation-blocker in the model-free comparator (filed by the §4c generality hunt; closed same turn by DEC-614):** `supervised_gate._raw_features` (the RFM/GBM comparator) hardcoded `pl.col("value")` for the monetary aggregate (`supervised_gate.py:239`) — a pipeline read on the raw stream, so any reference stream whose dominant numeric field is not `value` (sensor → `reading`, server → `code`) crashed the supervised comparator while the reference stream hid it (demo_v2's only numeric column is `value`). Fixed DEC-614: the field is now **derived** via `cfm_data._primary_numeric_field(d, cfg)` with a numeric-free zero fallback (`pl.lit(0.0)`), matching the encoder's own field resolution (the DEC-611 `2a49b8d` precedent); `tests/test_supervised_generality.py` pins name-agnostic output + the no-numeric fallback. No-op on the reference stream (derived == `value`), so the re-stamp at behavior `7f20fd0b` moves no verdict-bearing value and the card holds SOLVED 39/39. | DEC-614; `looking_glass/supervised_gate.py:239→derived`; `tests/test_supervised_generality.py`; re-stamp receipts at `7f20fd0b` | 2026-11-06 | `7f20fd0b` |

---

## 5. Mechanical baseline (disclosure, not a verdict)

Measured 2026-11-05 at `38740c8` (`.venv` toolchain, ruff 0.17.3, pytest 9.2.0); re-verified
at the closure tree `eb9381b` — every row RC=0, fast suite 773 passed (closing FULL: §6.9).

| Check | Command | State at `38740c8` | Note |
|---|---|---|---|
| Principle audit | `python3 scripts/principle_audit.py --all --strict` | **RC=0** — 74 blocking / 0 new beyond the 67-entry baseline; 141 notes | whole behaviour surface; baseline may only shrink |
| Lint | `ruff check .` | **RC=0** — clean | was "known pre-existing failures" in the old baseline; now clean |
| Format | `ruff format --check .` | **RC=0** — clean (re-measured after B-014 closure) | red at `38740c8`: 9 `scripts/` files (last touched `63e13da`; argparse-default wrapping only; none in the behavior trees) → closed 2026-11-05 at `08e37e4` (B-014) |
| Fast suite | `python3 -m pytest -q -m "not slow"` | **RC=0** — 771 passed (re-measured after B-014 closure; 773 after the B-012 citation tests) | red at `38740c8`: `test_every_producer_that_can_allocate_is_policed` (`measure_port_demo.py`, `measure_red_king_known.py` absent from `gpu_watchdog.DEFAULT_PATTERNS`) → closed 2026-11-05 at `08e37e4` (B-014); the old baseline's three named failures no longer exist |
| Plan freshness | `python3 scripts/plan.py --check` | PASS — `PLAN.md` current (680 lines) | spec is the single tree definition |
| Card determinism | `python3 scripts/report_card.py` (re-render) | byte-identical except the header commit/timestamp | committed card rendered at `346d62b`; content delta = 0 |
| Environment receipt | `docs/report_cards/experiments/env_receipt.json` | dated 2026-10-29 at `67a0bba` (dirty); H1's frozen gate is presence + device smoke (`cupy smoke ok`, `torch matmul ok`), so the row is green under its own gate | context only; the current suite state of record is the table above, not the env receipt |

A BACKWARD finding must still be backed by a **specific** file:line or command; this table is
context, not evidence of alignment.

---

## 6. FULL review 2026-11-05 (mode: FULL at the SOLVED tree)

### 6.1 Verdict

**NO BREAKING FINDINGS.** The SOLVED claim's measurements survive every attack in §6.4: all 39
atoms re-derive to PASS from stamped, behavior-current receipts under frozen gates; the macro
derivation is mechanical (`plan_spec.statuses`) and was recomputed independently; the route
pointer is honest; no `reporting` green is promoted to quality evidence. What fails is the
certification **record**, not the certification: the decision log has a hole in exactly the ids
the completion claim cites (B-012), one leg receipt contradicts itself (B-013), and the repo's
own commit gates were red across the SOLVED span (B-014). None of these falsifies a number; all
three are MAJOR and block the completion condition while open (`backward_open=3` → root FAILED,
B+C NOT DONE — the card gates honestly, as designed).

### 6.2 Macro verdict (the tree)

Census: 39 atomic problems = 23 B rows + 8 leg columns + 3 dec (E1.a/b/c) + 5 v1; 19 interior
nodes. `plan.py --check` PASS; card re-render deterministic. Recomputed from the receipts at
behavior `df73c2aa`:

| Node | Claimed (card) | Recomputed | Honest? | Finding |
|---|---|---|---|---|
| G (root) | SOLVED 39/39 | 31 card atoms PASS + E1.a/b/c closed + V1.1–V1.5 closed; `backward_open=0` at the recorded tree → SOLVED. With B-012/013/014 open, `plan_spec.statuses` forces G=FAIL until they close | yes, at the recorded tree; gates now | B-012/013/014 |
| P1 The donor exists (4) | PASS | B1.1 S3 working; B1.2 C1–C6 exact; B8.1 identity consistent; B8.2 S6 one-command | yes | — |
| P2 The donor suffices (11) | PASS | B2.1–B2.3 (U 0.9802; learned-only carries); E1.a g −0.580876 CI [−0.0240, −0.000662] excl. 0, neg control null, adversarial control NO_SIGNAL-as-expected; E1.b g −0.721/+0.181 ≤ bar 0.5; E1.c port 30.88/0.2672/0.4600 vs bars 30/0.15/0.45, trunk_only=false recorded; B4.1–B4.3; B5.1–B5.2 | yes | B-013 (record only) |
| P3 The donor is usable (11) | PASS | B3.1–B3.6; B6.1–B6.3 (13/15 family; table+E CIs > 0); B7.1–B7.2 (calibrated ECE 0.0302; OOD n=0 recorded under the frozen "OOD recorded" gate) | yes | — |
| P4 The consequence is delivered (8) | PASS | all 8 leg columns: every task ci_lo > 0 and brain > pathA under `_task_passes` (`report_card.py:829`); delivery columns clear the SST-scaled margin; Ns recorded per task | yes, for the verdict | B-013 (receipt consistency), B-017 (disclosed evidence questions) |
| P5 The stack is v1-complete (5) | PASS | V1.1 stream sha recorded; V1.2 plugins 9/9 passed; V1.3 0.7334 vs 0.25 chance + policy CI [+0.1551, +0.1728]; V1.4 cert sha256 pinned + matching; V1.5 plan present + arm/GP match | yes | B-018 (reference receipt) |

**Route honesty.** `CURRENT_ROUTE` R7 names the true frontier (root SOLVED; production-stream
scope explicitly not claimed). R1–R6 are all DONE with their receipts; the sequencing was
respected — the E1.c port ran after both dev-rig structures were mastered (dec603 re-stamps),
and B+C bank v2 preceded the v1 stack. No step skipped.

**Gate-kind / "Green means" misuse.** Checked every `reporting` row (B1.1, B2.3, B3.1, B3.2,
B3.4, B3.5, B3.6, B4.3, B5.2, B6.1, B7.2, B8.1, B8.2 and the stages): none is cited as quality
evidence in the card prose, the DEC outcomes, STATUS or the plan; the card's own caveat line
("A green atom means its frozen gate is met … not a quality level") is present. B2.3's
learned-only attribution (U 1.0764 ≥ served 0.9802, "the ledger does not add") is exactly the
attribution the rule asks for. No misuse found.

**Tree sufficiency.** No necessary condition of the goal lacks a node, beyond the recorded scope
decisions: production-stream certification is out of scope by B-001/DEC-605; the OPE broad
acceptance contract is separated from the stream gate by B-003's scope. Both are operator-level
and already ruled. No new proposed node.

### 6.3 Findings (detail)

**B-012 (MAJOR, P6) — the decision log is missing the entries the completion claim cites.**
- *Claim attacked:* the pre-registration record of the bank-v2 / E1 re-stamp protocols.
- *Why it may be false:* certification integrity rule 1 says gate definitions are "recorded in
  `DECISIONS.md` and the receipt **before** running". The protocol substance exists —
  `measure_bc_bank.py`'s docstring, the `protocol` fields of the leg receipts, the commit
  messages of `e9dfd6b`/`e7918a4`/`0bc0d1f`/`7a3ae15`, the STATUS entries — but the designated
  record has no DEC-598/599/600/601 entries, and PLAN/STATUS/card cite those ids as load-bearing
  (`PLAN.md:617` "bank protocol (DEC-598)"; `plan_spec.py:338` "DEC-586/600/601 bars";
  `STATUS.md:301` "(DEC-598, 2026-11-04)"). DEC-597/602 have no trace anywhere in the git
  history (`git log --all -S` empty for both): they are unallocated ids, not lost entries.
- *Reproduction:* `git log --all -S "DEC-598" --oneline -- DECISIONS.md` (empty);
  `awk '/DEC-596/,/DEC-603/' DECISIONS.md | head -1` (the log jumps straight on).
- *Gate touched:* no.
- *Forward subgoal:* §6.7, item 1.
- *What would change my mind:* an entry for each of DEC-598/599/600/601 in `DECISIONS.md`,
  explicitly labeled as a post-hoc record repair, citing the commit + receipt evidence, with
  597/602 sealed as unallocated — plus a mechanical test that fails if a cited in-log-range
  DEC id has no entry.

**B-013 (MAJOR, P6/E10) — the unsupervised leg receipt contradicts itself; all four legs carry
flat-era `arms`.**
- *Claim attacked:* receipt self-consistency (integrity rule 7) and the freshness law for the
  P4 evidence.
- *Why it may be false:* `judge_leg_unsupervised.json` says `working: true` and
  `checks.unsupervised.working: false` in the same file; its `arms` still describe the flat-era
  consumer ("state_embeddings donor; pairwise-distance and k-means geometry") while `protocol`
  says "bank v2 (DEC-598): per-block distances …". Five keys (`arms`, `checks`, `diagnostics`,
  `path_a`, `excluded`) are byte-identical to the receipt at `379f189` (flat era, `working:
  false`). Mechanism: `measure_bc_bank.py` scores new `columns` and writes `working`/`evidence`
  but never `checks`/`arms`; `stamp_merge_write` (`measure_input_law.py:487`) merges into the
  existing file, so the flat-era blocks survive every bank-era re-stamp. The card verdict is
  unaffected (it derives from `columns` via `_task_passes`), and the renderer's
  self-contradiction guard (`_block_problem`) covers stage blocks only — so nothing in the
  current pipeline would have caught this. That is the real risk: a future reader (or a future
  renderer that trusts `checks`) would conclude the leg "did not work".
- *Reproduction:* diff the receipt's keys against `git show 379f189:docs/report_cards/experiments/judge_leg_unsupervised.json` — `arms`/`checks`/`diagnostics`/`path_a`/`excluded` identical; top `working` differs (false→true).
- *Gate touched:* no.
- *Forward subgoal:* §6.7, item 2.
- *What would change my mind:* all four leg receipts carry bank-v2 `arms` and a `checks` rollup
  consistent with their own `working`, with `columns` byte-identical to the committed receipts
  (determinism proof), re-stamped at the new commit.

**B-014 (MAJOR, P6/E10) — the repo's own commit gates were red across the SOLVED span.**
- *Claim attacked:* the "quality gates before every commit" contract (AGENTS.md) and the audit's
  duty to disclose the true mechanical state.
- *Why it may be false:* at `38740c8` (and at every commit from `63e13da` onward for the format
  gate, from `75df2b0` onward for the suite gate): `ruff format --check .` → 9 files under
  `scripts/` would be reformatted (long absolute-path argparse defaults introduced by the
  workspace consolidation at `63e13da`); `pytest -m "not slow"` → 1 failure,
  `test_every_producer_that_can_allocate_is_policed` — the two evidence scripts added at
  `75df2b0` (`measure_port_demo.py`, `measure_red_king_known.py`) are not in
  `gpu_watchdog.DEFAULT_PATTERNS`, so the memory guard cannot see two producers that can allocate
  large frames. The previous AUDIT §5 disclosed neither: it listed three failures that no
  longer exist and called the format gate "known pre-existing" without measuring it.
- *Reproduction:* `.venv/bin/ruff format --check .` (RC=1, 9 files); `.venv/bin/python -m pytest -q -m "not slow"` (RC=1, 1 failed); `grep -c "measure_port_demo" scripts/gpu_watchdog.py` → 0.
- *Gate touched:* no — the fix is to make the repo's standing gates green, not to change any
  measurement gate. (Behavior safety: all changes are under `scripts/`; `BEHAVIOR_TREES` is
  `looking_glass`/`rabbit_hole`, so `df73c2aa` is untouched.)
- *Forward subgoal:* §6.7, item 3.
- *What would change my mind:* all three gates RC=0 at the closure commit, `principle_audit`
  still 0-new, and a re-rendered card with unchanged content.

**B-015 (MINOR, E10) — the bundle manifest is unstamped.**
- *Claim attacked:* provenance of the single-artifact claim (B-009's closure).
- *Why it may be false:* `bundle_receipt.json` (registry `116087fb206c2d81`, 23 B rows → 6
  components, scope + brain-boundary recorded) has no `_meta`: a reader cannot tie the manifest
  to the behavior/commit that produced it. Every receipt it maps is stamped; the manifest is
  the one unstamped link.
- *Reproduction:* `python3 -c "import json;print('_meta' in json.load(open('looking_glass/artifacts/cfm_v2k_vf2_cert/bundle_receipt.json')))"` → False.
- *Gate touched:* no.
- *Forward subgoal:* §6.7, item 4.
- *What would change my mind:* the receipt carries `_meta` and the registry sha is unchanged
  (registry = `registry_{tag}.json` = bank registry + component manifest; the receipt is not
  part of that hash).

**B-016 (MINOR, P6) — closed by this rewrite.** See §4: the previous §4 fragment (orphaned
separator + duplicated B-004/B-010 rows) and the stale §5 disclosure are repaired in this file.

**B-017 / B-018 (INFO).** See §3: disclosed, non-blocking; no forward subgoal required.

### 6.4 Per-row attack

Every green atom attacked against its receipt + frozen gate; stages and hygiene included.
"AGREE" = re-derivation matches the card's verdict and the evidence is stamped at
`df73c2aa` (git_commit ≤ `38740c8`).

| Row | Attack performed | Result |
|---|---|---|
| B1.1 | S3 block working + per-producer `_stamp` behavior `df73c2aa` (`judge_stages.json`) | AGREE |
| B1.2 | C1–C6 invariants re-read: field-perm 1.714e-07, scan 1.105e-15, mass 4.819e-16, determinism bit-identical, schema-variant ok, identity present; N=20 recorded | AGREE |
| B2.1 | served U 0.9802 ≥ bar 0.5 (DEC-570, defined); worst included probe next-type; 4/5 probes excluded under the frozen headroom≤0 rule, each with its reason in the receipt | AGREE |
| B2.2 | every included probe clears the bar | AGREE |
| B2.3 | learned-only U 1.0764 ≥ served 0.9802 → "the ledger does not add" attribution present (rule 4) | AGREE |
| B3.1 | S4 working + stamp | AGREE |
| B3.2 | M0→M1→served fine-structure recorded; largest drop attributed to a named stage pair (reporting gate) | AGREE |
| B3.3 | served 0.46 vs PATH A 0.4126, delta +0.0474 ≥ margin −0.01, n_fields 11 | AGREE |
| B3.4 | block-aware geometry complete; counter-weight sensitivity flat, recorded (reporting gate) | AGREE |
| B3.5 | equal-dim 128/128 + 39/39 complete (reporting gate) | AGREE |
| B3.6 | branch-exposure arms recorded (best ledger arm `part_numeric_stats` 0.3196); reporting gate; served-bank boundary stated | AGREE (DEC-598 citation → B-012, record only) |
| B4.1 | share_E 0.2983 > table 0.2438, CI [+0.0249, +0.0855]; random 0.2586 / destroyed 0.2384 below E; n=1485; oracle ceiling 0.6404 recorded | AGREE |
| B4.2 | IPS E 0.1701 vs table 0.1517, diff +0.0183, CI [+0.012, +0.0246] > 0; n_test 74746 / n_train 177017; arm frequencies 0.2482–0.252 (near-uniform randomized design); protocol v2 = DEC-595 (defined) | AGREE |
| B4.3 | P1==P2 bit-identical; P1 vs stored within bands; teeth=True; level-permutation null ok recorded | AGREE |
| B5.1 | served shape rank 32 == k_min 32 (frozen reconstruction certificate); explicit block 147; joint floor 48 (unconstrained) recorded; n_train/n_test 451/439; noise 0.0477 | AGREE |
| B5.2 | runtime receipt complete (cold 11.8 s / warm 84.7 s, 53.1 rows/s, peak RSS 3360 MB, incremental absorbed 0) | AGREE |
| B6.1 | S5 working + stamp | AGREE |
| B6.2 | 3 targets: table+E CI lo +0.0037/+0.0052/+0.0008 all > 0; E ≥ raw/fair on all; controls random/destroyed below E; hstack v1-arm failures recorded (CI lo −0.042/−0.049) not hidden | AGREE |
| B6.3 | 13/15 wins ≥ 2/3 bar; family mean E 0.4331 vs table 0.3857 (diff +0.0475 > 0); random 0.0098 / destroyed 0.0043 below E; family published in the receipt | AGREE |
| B7.1 | calibrated ECE 0.0302 ≤ 0.1; temperature T=0.5394 fit on split A only (n_calib_A 3515), applied to B (DEC-593, defined); OOD as_of+30d n=0 recorded under the pre-registered "OOD recorded" gate — n=0 is compliant, not a hidden exclusion | AGREE |
| B7.2 | next-time 0.8 coverage 0.800 (n 1485, MAE 1.15d); occurrence AUC 0.6339 / Brier 0.2794; missingness ladder 10%→0.094, 25%→0.233 recorded | AGREE |
| B8.1 | tag/registry/config/data/checkpoint shas mutually consistent (registry 919a932f…, config 4f368093…, data 3543c711…, checkpoint 33f9b803…); as_of + n recorded | AGREE |
| B8.2 | S6 one-invocation chain: 6 producers with timings; deterministic re-render verified (byte-identical except header) | AGREE |
| E1.a | positive: g −0.580876, paired CI [−0.0240, −0.000662] excludes 0, beats_base/mastered true; negative control CI [−4.7e-05, +3.3e-05] straddles 0 (no signal, as expected); adversarial control: beats_base via the echo yet classified NO_SIGNAL, leak-teeth-on-A-side recorded; signature block + fixture hashes pinned; products registry recorded | AGREE |
| E1.b | intent g −0.721 (365d) / +0.181 (730d) ≤ bar 0.5, mastered both; dec603 receipts stamped at df73c2aa; bars cite DEC-586/600/601 — 600/601 are the B-012 record hole (measurement unaffected) | AGREE (record: B-012) |
| E1.c | port 30.88/0.2672/0.4600 vs DEC-572/573 bars 30/0.15/0.45; meets_all_bars true; trunk_only=false; classification "delivery/ledger port; NOT a trunk-only capacity win"; no measure-free object clears all three bars (shape32/donor106/bank179 recorded) | AGREE |
| P4 supervised · brain | 7 tasks: ci_lo 6.3e-05 … 2373.9 all > 0 and brain > pathA; controls negative and readable except the 5 vacuous traits (disclosed, B-017) | AGREE (receipt consistency: B-013) |
| P4 supervised · brain_ledger | same 7 tasks, ci_lo all > 0; delivery margin cleared | AGREE |
| P4 unsupervised · brain | persona 0.3976 vs 0.3867 CI [+0.0084, +0.0131]; future 0.1927 vs 0.1810 CI [+0.0109, +0.0151]; segment ARI 1.0 vs 0.2670 CI [+0.704, +0.758] against the generator's declared latent (ground truth, not circular); random/destroyed ≈ 0 below | AGREE (receipt self-contradiction: B-013) |
| P4 unsupervised · brain_ledger | ledger adds on future geometry (0.1980, CI [+0.0163, +0.0204]) — recorded, not booked as a brain win | AGREE |
| P4 rl · brain / brain_ledger | uplift CI [+0.0010, +0.0026] / [+0.00098, +0.0025] > 0; policy value CI > 0; controls readable, below brain | AGREE (flat-era arms text: B-013) |
| P4 simulator · brain / brain_ledger | dose CI [+0.00012, +0.0005] > 0 (n=5000 periods); argmax CI [+0.058, +0.119] > 0; controls below | AGREE |
| Stages S1–S6 | each block's own `_stamp` at df73c2aa from its own producer (input/dynamics/views/battery/certification law); working=true; sub-check evidence OK | AGREE |
| H1 | env receipt present + device smoke ok (frozen gate = presence + smoke; receipt dated 2026-10-29 is context, disclosed in §5) | AGREE |
| H2–H5 | enforced per row by the renderer (no PASS without behavior hash / N / controls / finite fair comparator); spot-checked on the rows above | AGREE |
| Frontier readings | dev-rig iter1–4, cap256, intent_dec586* all negative on the g bar (1.147–2.665 > 0.5) and O3 attempt 4 REJECTED — reported as frontier evidence, not gated, not promoted | AGREE |

**Not attacked because:** no hidden rows exist — the card lists every atom; EXCLUDED rows are
the four B2.1 probes (disclosed per-receipt under the frozen rule); the `bc_donor_diagnostic.json`
and `plugin_acceptance.json` reference receipts are not gate evidence (disclosed, B-018).

### 6.5 Goal-level findings

1. **Scope residual (recorded, ruled):** everything is certified on the synthetic `demo_v2`
   prototype; the trunk-only effective rank is 4.11 (base) / 10.33 (bank); E1.c clears its bars
   only with the measure PCA-128; the B+C brain is the served bank and the measure-free
   donor-only rerun fails the legs (B-001/B-003/B-008, closed as scope). A production-stream
   claim would be new scope, not a fix.
2. **Record maturity is the remaining gap.** The measurements are honest; the certification
   record around them (decision log completeness, receipt self-consistency, repo gate state)
   is not yet at the bar this project has applied to itself. B-012/013/014 are exactly that gap.
3. **Operator ruling queued (B-017):** the ledger-only diagnostic falsifies the SPINE §5
   corollary on this stream; baseline-vs-corollary needs a ruling. Non-blocking today.
4. **Effect-size honesty:** the unsupervised wins are small in absolute terms (persona +0.011,
   future +0.012/+0.017) — the CIs are clean and the card reports the numbers without quality
   inflation; this is a strength of the record, noted so the next reader does not mistake the
   margin for a claim.

### 6.6 Process audit

- **Pre-registration.** Gates are frozen in code (`judge_spec.py`, `b_spec.py`,
  `plan_spec.py`) and were not edited after results in the SOLVED span (verified by the
  receipts' `protocol` fields matching the frozen logic). The DEC-598/599/600/601 protocols
  were pre-registered in the producer docstrings, commit titles and STATUS before/with their
  result commits — but the designated record (`DECISIONS.md`) is missing them (B-012).
- **Honest FAIL reporting.** Negative dev-rig results, the REJECTED O3 attempt, the failing
  hstack v1 arms, the donor-only leg failures and the vacuous negative controls are all
  recorded, not relabeled.
- **Freshness.** All 23 B receipts, the 4 leg receipts (columns), 3 lab receipts, the port
  receipt and 5 v1 receipts are stamped at behavior `df73c2aa` with commits ≤ `38740c8`.
  Stale-without-stamp: the flat-era `checks`/`arms` residue inside the bank-era leg receipts
  (B-013); `bundle_receipt.json` (B-015); `env_receipt.json` (context only, §5).
- **Commits.** The SOLVED span was committed through red commit gates (B-014). This review's own
  commit is made with B-014 open and disclosed as such — a docs-only commit; the gate fix is
  forward subgoal 3.
- **Append-only log.** No existing entry was edited in the span; four entries are missing
  (B-012). DEC-574 is documented as reserved (`DECISIONS.md:9090`).
- **Branch.** `foundation-rebuild`, clean at review start, pushed; card rendered at the clean
  parent (`346d62b`), content-identical re-render verified at `38740c8`.

### 6.7 Handoff to FORWARD

1. **B-012 — record axis (no measurement axis).** `NODE: P2/P4 record`. `MOVE:` append
   post-hoc-labeled entries DEC-598 (bank v2 protocol + outcome, from `e9dfd6b`/`e7918a4` +
   leg receipts + STATUS), DEC-599 (E1.c port PASS + B re-stamp, from `0bc0d1f`/`580b62c` +
   `lab/port_dec604.json`), DEC-600/601 (E1.a/E1.b green under the hierarchical Dirichlet
   signature head, from `7a3ae15` + `lab/*_dec601.json`), each labeled "recorded post-hoc
   2026-11-05 as a record repair (B-012); pre-registered in <docstring/commit/STATUS> before
   the run"; seal DEC-597/602 as unallocated (no trace in history). Add
   `tests/test_decision_log_citations.py`: every `DEC-NNN` cited in `PLAN.md`,
   `REPORT_CARD.md`, `STATUS.md`, `AUDIT.md`, `DECISIONS.md`, `scripts/plan_spec.py`,
   `scripts/report_card.py` with NNN ≥ the log's oldest entry and not documented as reserved
   must have an entry (elided citations `DEC-a/b` expanded). Frozen acceptance: test green,
   log monotone, zero existing entries edited. Cost: docs + one test.
2. **B-013 — measurement-consistency axis.** `NODE: P4`. `MOVE:` in `measure_bc_bank.py`,
   write a current `checks` rollup (from its own `_evidence`) and bank-v2 `arms` for every leg
   it scores; on merge, drop the stale keys it redefines so the flat-era residue cannot
   survive. Re-run the producer for all four legs. Frozen acceptance: `columns` byte-identical
   to the committed receipts (determinism proof), `checks.<leg>.working == working`, `arms`
   describe the bank-v2 consumer, receipts re-stamped. Cost: one producer edit + four re-runs
   (S6 recorded the leg chain at 44.2 s).
3. **B-014 — repo-hygiene axis.** `NODE: — (process)`. `MOVE:` `ruff format` the 9 `scripts/`
   files; add `measure_port_demo.py` + `measure_red_king_known.py` to
   `gpu_watchdog.DEFAULT_PATTERNS`. Frozen acceptance: `ruff check .` + `ruff format --check .`
   + `pytest -m "not slow"` all RC=0; `principle_audit --all --strict` still 0-new; behavior
   hash unchanged (`scripts/` is not in `BEHAVIOR_TREES`); card content unchanged on re-render.
   Cost: minutes.
4. **B-015 — provenance axis.** `NODE: B (single-artifact record)`. `MOVE:`
   `build_b_cert_bundle.py` stamps `_meta` (behavior/commit/producer/command/generated_at) on
   `bundle_receipt.json`; re-run the builder idempotently. Frozen acceptance: receipt carries
   `_meta`; bundle registry sha unchanged (`116087fb206c2d81`); card B scope line unchanged.
   Cost: one producer edit + one re-run.

After 1–4: re-render the card, then run the closing FULL review (B-016's precedent: the review
is tied to stamps). B-017/B-018 carry no forward subgoal (INFO, disclosed).

### 6.8 Stop condition

The tick-tock stops when, at a closure tree: (1) DEC-598/599/600/601 are recorded and the
citation test is green with 597/602 sealed; (2) all four leg receipts are self-consistent
(current `checks`/`arms`, identical `columns`); (3) `ruff check`, `ruff format --check` and the
fast pytest tier are all RC=0 at that commit; (4) `bundle_receipt.json` is stamped with the
registry sha unchanged; (5) the re-rendered card derives root SOLVED with `backward_open=0`
and a FULL review at that tree returns NO BREAKING with all MAJORs closed. Until then the card
gates FAILED — correctly.

### 6.9 Closing FULL review at the closure tree `eb9381b` (2026-11-05)

Mode: **FULL** — the completion review §6.8 names: the card's root is `SOLVED` and every
MAJOR from the opening FULL is closed. Tied to the closure tree's stamps (B-016's precedent:
a review is current when its stamps are).

**Per-finding verification (every check run at `eb9381b`):**

1. **B-012 (record).** `tests/test_decision_log_citations.py` green. `DECISIONS.md` carries
   336 entries (oldest DEC-275): DEC-598/599/600/601 are present, post-hoc-labeled, between
   DEC-596 and DEC-603; DEC-597/602 sealed reserved/unallocated (`DECISIONS.md:11306`,
   `:11418`); entry ids monotone from DEC-590 on (590–596, 598–601, 603–609). Every
   in-log-range `DEC-NNN` cited by the seven steering documents, elided citations expanded to
   all members, resolves to an entry or a seal.
2. **B-013 (measurement consistency).** All four `judge_leg_*.json` re-stamped by
   `measure_bc_bank.py` at behavior `df73c2aa`: `columns` byte-identical to the `38740c8`
   receipts (determinism — no verdict moved), `checks.<leg>.working == working` on all four
   (the unsupervised self-contradiction is gone), `arms` name the served block bank, and no
   flat-era `diagnostics`/`incumbent` block survives.
3. **B-014 (repo gates).** `ruff check .` RC=0; `ruff format --check .` RC=0 (487 files);
   `pytest -m "not slow"` RC=0 — 773 passed, 14 deselected;
   `principle_audit --all --strict` RC=0 — 74 blocking / 0 new beyond the 67-entry baseline;
   `plan.py --check` PASS; behavior hash `df73c2aa1427b471` unchanged.
4. **B-015 (provenance).** `cfm_v2k_vf2_cert/bundle_receipt.json` carries `_meta` (producer
   `build_b_cert_bundle.py`, behavior `df73c2aa`, commit, command); the bundle registry sha
   is still `116087fb206c2d81` (the registry hashes `registry_{tag}.json`, not the receipt);
   the 23-row B map and the scope string are intact.
5. **Macro (goal level).** No receipt outside the four leg files changed across the whole
   closure round (`38740c8` → `eb9381b`); the card at the clean closure tree derives
   root **SOLVED 39/39** · 0 FAIL · B **CERTIFIED 8/8** · B+C **DONE (ENCODER DONE v1)** ·
   V1.1–V1.5 green · `backward_open=0` (card sha256 `c2ca1fc60ccbfae5`), and the
   `FRONTIER_NOTE` claim ("no open BACKWARD requirement", `plan_spec.py:530`) is true again.

**Stop condition (§6.8, items 1–5): met.** Item 5 is this review.

**Verdict: NO BREAKING FINDINGS, all MAJORs closed.** The opening FULL's
record-and-consistency defects (B-012, B-013, B-014 + B-015) are closed by evidence (§4;
DEC-606/607/608/609); nothing new was found at goal level at the closure tree. The recorded
residuals stand unchanged: B-001 — the certified artifact is the synthetic `demo_v2`
prototype; B-003 — E1.c is a delivery/ledger port (trunk-only rank 4.11 base / 10.33 bank);
B-008 — the B+C brain is the served bank, and the measure-free donor-only diagnostic does not
clear all legs; B-017 (open INFO) — the flat-era vacuous negative controls + the ledger-only
diagnostic falsifying SPINE §5's corollary, operator ruling queued (recomputing them would be
a new measurement, out of scope for this record repair); B-018 (open INFO) — the
`plugin_acceptance.json` reference receipt is unstamped; the gate receipt `v1_plugins.json`
is stamped. Per §8 the two INFOs do not block completion.

**Completion.** The tick-tock stop condition (AGENTS.md operating mode) is met: the card's
plan root is `SOLVED` and a FULL BACKWARD at that tree returns NO BREAKING with all MAJORs
closed. The only remaining git action is the push, pending explicit operator permission.

---

### 6.10 Rescan FULL review at `b512068` (2026-11-05, operator-requested)

Operator-requested full adversarial rescan at the tree *after* the closing review
(`b512068` = closing review + the test-only citation-elision fix). Independent
re-derivation: every check re-executed, no number taken from §6.9 on trust.

| Attack | Result |
|---|---|
| Mechanical gates at `b512068` | `ruff check .` / `ruff format --check .` RC=0 (487 files); `pytest -m "not slow"` 773 passed / 14 deselected RC=0; `principle_audit --all --strict` 74 blocking / 0 new vs the 67-entry baseline; `plan.py --check` PASS (680 lines) |
| Behavior hash | recomputed `df73c2aa1427b471` = card claim; the behavior trees are unchanged by the spec/script-only commits since `38740c8` |
| Card determinism | fresh render vs the committed card: content delta = the header line only |
| Macro re-derivation | plan root PASS; 31 card atoms green + 8 `plan_spec.CLOSED` record closures (3 dec + 5 v1) = 39/39; `backward_open=0`; B 23/23, 8/8 sections |
| JSON audit state | 18 requirements: 16 closed (closure kinds: evidence ×13, scope ×3 — no operator-accepted), B-017/B-018 open INFO; zero open BREAKING/MAJOR |
| Registries | bundle `116087fb206c2d81` and bank `919a932f1b657064` recomputed, both match the claims |
| Receipt drift 38740c8→b512068 | exactly the 4 leg receipts (the DEC-608 re-stamp); no other receipt touched |
| Leg columns (8) | re-derived task-by-task under the frozen `_task_verdict`: progress legs `ci_lo > 0 and brain > pathA`; delivery legs the SST-scaled margin rules; controls random/destroyed ≤ brain — all PASS |
| E1 citations | every number in the dec-citation text re-checked against the lab receipts: E1.a g −0.580876, `paired_ci_enc_minus_base` [−0.0240014, −0.0006619], bar_g 0.5, A-side control gate complete; E1.b g −0.721396 / 0.181074 (long, beats_base + mastered true both); E1.c port eff_rank 30.881072 / persona 0.267182 / served_mean_r2 0.459995 vs bars {30, 0.15, 0.45}, `meets_all_bars: true`; trunk-only base272 4.105023 / bank179 10.330055 recorded, both `meets_all_bars: false` |
| V1.1–V1.5 | all five stamped receipts `passed: true` at `df73c2aa` (V1.1 = `v1_identifiable_stream.json`: one-factor randomized weekly offer, `offer_frequencies` {0.2502, 0.2499, 0.2499, 0.25}, n_customers 5000, n_email_sends 375456) |
| B receipts + stages | all 23 B rows at current behavior (the 13 reporting-gate rows are disclosed on the card; none promoted in prose); S1–S6 block stamps current, `working: true` |
| "universal" usage | in the card only goal line G and S1 (a receipted row); no unearned universality claim in card prose |
| Decision log | 336 entries (oldest 275, newest 609); seals at the reserved 597/602; the disciplined tail ≥ 590 is monotone [590–596, 598–601, 603–609]; citation test green; R1–R6 route texts accurate against the receipts |
| §6.9 spot check | card sha256 at `eb9381b` recomputed = `c2ca1fc60ccbfae5` (the first mismatch was a scan-side bug — printing the file's byte-hex instead of the digest; `sha256sum` and `hashlib` agree) |

**Finding B-019 (MINOR, P6/E10).** R7's route evidence text — `plan_spec.py:631`,
rendered at `PLAN.md:619` and in the card — still reads the DEC-605-era snapshot
"closed all 11 findings (8 by evidence, 3 by the scope/reclassification…)"; the record
at this tree is 16 of 18 closed (13 evidence + B-001/B-003/B-008 scope) with the two
open items disclosed INFO. A card contradicting the audit in prose is a finding under
PROMPT_BACKWARD §7. Non-blocking: route steps are pointers, not verdicts, and no gate
reads this text. Forward subgoal: one-line durable-outcome repair — state the outcome,
name the three scope closures, do not hardcode a count that would go stale at the next
finding (rejected: re-counting "16 of 18" into the text; editing this file to match the
plan — the record is the newer, correct one).

**Verdict: NO BREAKING.** One new MINOR (B-019) filed in §3/§9; everything else
re-verified clean at `b512068`. **Closure:** B-019 closed 2026-11-05 by DEC-610 /
commit `2976485` (route text now states the durable outcome; §4).

### 6.11 Operator-filed requirement B-020 (2026-11-05)

B-020 (MAJOR, P1 Universal / Honest claim-scope) was **not filed by a BACKWARD run**.
The operator reviewed the closing reviews (§6.9/§6.10) and named the gap their
adversarial coverage should have surfaced: the closing reviews falsified the record's
*plumbing* (citations, manifests, self-consistency, route text) but never the *scope of
the claim* — every B-row and every pre-DEC-611 lab fixture is one ecom simulator family
on the `demo_v2` prototype, and S1 is schema-variant, not structure-variant. Filed in
§3/§9 the same day with the DEC-611 campaign as its forward subgoal; closed the same day
under the closure criterion stated at filing (six lab receipts `complete=true` + the
DEC-611 outcome addendum) — B-012…B-019 precedent (filed and closed by evidence in the
same working day). The branch option (grow the tree so multi-domain mastery becomes a
gated card row) is a separate DEC requiring operator ruling; DEC-576 keeps these
receipts as development instruments until then.

### 6.12 Forward-filed requirement B-021 (2026-11-05, DEC-611 closeout forensics)

B-021 (MINOR, P6 single-source-of-truth / P10 reproducibility / Honest) was **not filed
by a BACKWARD run**: the forward loop found it while verifying the DEC-611 closeout —
the clean-tree S6 chain re-stamp had flipped the plan root SOLVED 39/39 → 35/39 because
the chain's `legs` step still ran the pre-DEC-598 producer (`measure_judge_law.py`
`main`) against receipt files owned by `measure_bc_bank.py` since bank v2. The card's own
gate caught it (the verdict recomputed from the overwritten receipts; the incident is
undetectable by any producer's exit code, which is why the reader — not the producer —
is the check of record). Filed in §4/§9 and closed the same day under B-012…B-019
precedent (fix + re-stamp evidence; DEC-612, one-line CHAIN map, script-only so behavior
stays `89cec70b`; gate/criterion untouched).

---

## 7. What a principle finding looks like

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

## 8. The completion condition (mirrors `REPORT_CARD.md`)

The card is not all green while any **BREAKING** or **MAJOR** finding here is open. A
`MAJOR` may be closed by a forwarding protocol or explicitly `accepted` by the operator;
a `BREAKING` may only be closed by evidence. `MINOR`/`INFO` findings are disclosed and may
remain open without blocking completion.

---

## 9. Machine-readable requirements (the only gate BACKWARD writes)

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
      "evidence": "the 75df2b0 re-review recorded behavior/commit/artifact/bank/bundle hashes; the earlier status is superseded",
      "audit_ref": "A-011",
      "date": "2026-11-04"
    },
    {
      "id": "B-012",
      "severity": "MAJOR",
      "principle": "P6",
      "status": "closed",
      "closure": "evidence",
      "finding": "DEC-598/599/600/601 are cited as load-bearing by PLAN/STATUS/card/plan_spec/report_card and the commit titles of e9dfd6b/0bc0d1f/7a3ae15, but have no entry in DECISIONS.md (log jumps DEC-596 -> DEC-603); DEC-597/602 are unallocated (no trace in git history).",
      "forward_subgoal": "Append post-hoc-labeled DEC-598/599/600/601 entries reconstructed from commit messages + receipts + STATUS (record repair, B-012), seal 597/602 as unallocated, add tests/test_decision_log_citations.py: every in-log-range DEC-NNN cited in PLAN/REPORT_CARD/STATUS/AUDIT/DECISIONS/plan_spec/report_card (elided DEC-a/b expanded, documented reservations allowed) resolves to an entry; log monotone; zero existing entries edited.",
      "gate_touched": "none (record only)",
      "evidence": "DEC-607; post-hoc-labeled entries inserted between DEC-596 and DEC-603 (commit 470bb5e, 155 insertions / 0 deletions); DEC-597/602 sealed as reserved/unallocated; tests/test_decision_log_citations.py green (773 passed); original finding: git log --all -S 'DEC-598' -- DECISIONS.md empty; PLAN.md:617; plan_spec.py:338; STATUS.md:301",
      "audit_ref": "A-012",
      "date": "2026-11-05",
      "closed": "2026-11-05"
    },
    {
      "id": "B-013",
      "severity": "MAJOR",
      "principle": "P6/E10",
      "status": "closed",
      "closure": "evidence",
      "finding": "judge_leg_unsupervised.json is self-contradictory (working:true vs checks.unsupervised.working:false) with arms/diagnostics/path_a/excluded/checks byte-identical to the flat-era receipt at 379f189; rl/simulator legs carry flat-era arms; cause is stamp_merge_write merging keys measure_bc_bank.py never rewrites.",
      "forward_subgoal": "measure_bc_bank.py writes a current checks rollup + bank-v2 arms for every leg it scores and drops the stale keys it redefines on merge; re-run all four legs; columns must be byte-identical to the committed receipts; receipts re-stamped.",
      "gate_touched": "none (verdict derives from columns under frozen _task_passes)",
      "evidence": "DEC-608; commit 4cf477b: measure_bc_bank.py writes current checks rollup + BANK_ARMS + path_a, stamp_merge_write drop=('diagnostics','incumbent'); all four legs re-run - columns byte-identical to the committed receipts (determinism proof, no verdict moves), checks.<leg>.working == working on all four, arms name the served block bank, no flat-era block survives, receipts re-stamped; original finding: key diff vs git show 379f189:...",
      "audit_ref": "A-013",
      "date": "2026-11-05",
      "closed": "2026-11-05"
    },
    {
      "id": "B-014",
      "severity": "MAJOR",
      "principle": "P6/E10",
      "status": "closed",
      "closure": "evidence",
      "finding": "At the SOLVED commit 38740c8 the AGENTS.md commit gates are red: ruff format --check fails on 9 scripts/ files (since 63e13da) and the fast pytest tier fails test_memory_guard (measure_port_demo.py + measure_red_king_known.py, added 75df2b0, absent from gpu_watchdog.DEFAULT_PATTERNS).",
      "forward_subgoal": "ruff format the 9 scripts/ files; add both producers to DEFAULT_PATTERNS; ruff check + ruff format --check + pytest -m 'not slow' all RC=0; principle_audit still 0-new; behavior hash df73c2aa unchanged.",
      "gate_touched": "none (repo commit gates, not measurement gates)",
      "evidence": "DEC-606; commit 08e37e4: ruff format on the 9 scripts/ files + both producers in gpu_watchdog.DEFAULT_PATTERNS + card-mechanics tests isolated from the live AUDIT.md; at the closure commit ruff check / ruff format --check / pytest -m 'not slow' all RC=0 (771 passed), principle_audit 0-new, behavior df73c2aa unchanged; original finding: ruff format --check RC=1 (9 files), pytest 1 failed, gpu_watchdog.py:34",
      "audit_ref": "A-014",
      "date": "2026-11-05",
      "closed": "2026-11-05"
    },
    {
      "id": "B-015",
      "severity": "MINOR",
      "principle": "E10",
      "status": "closed",
      "closure": "evidence",
      "finding": "cfm_v2k_vf2_cert/bundle_receipt.json (the B-009 single-artifact manifest, registry 116087fb206c2d81) has no _meta stamp while all 23 row receipts it maps do.",
      "forward_subgoal": "build_b_cert_bundle.py stamps _meta on the bundle receipt; re-run the builder; bundle registry sha must remain 116087fb206c2d81 and the card B scope line unchanged.",
      "gate_touched": "none",
      "evidence": "DEC-609; commit 672c164: build_b_cert_bundle.py writes the receipt through shared stamp_merge_write; builder re-run idempotently - _meta present (behavior df73c2aa, commit, producer, command), registry sha 116087fb206c2d81 unchanged, 23-row map + scope intact, card B scope line content-identical; original finding: bundle_receipt.json keys lack _meta; build_b_cert_bundle.py:266",
      "audit_ref": "A-015",
      "date": "2026-11-05",
      "closed": "2026-11-05"
    },
    {
      "id": "B-016",
      "severity": "MINOR",
      "principle": "P6",
      "status": "closed",
      "closure": "evidence",
      "finding": "The previous AUDIT §4 carried an orphaned table separator with B-004/B-010 rows duplicated outside the closed table, and §5 disclosed fast-tier failures that no longer exist while omitting the current red gates.",
      "forward_subgoal": "Repair the file (defects in BACKWARD's own file may be fixed by the rewrite).",
      "gate_touched": "none",
      "evidence": "this rewrite: §4 table repaired, §5 baseline re-measured at 38740c8 (2026-11-05)",
      "audit_ref": "A-016",
      "date": "2026-11-05"
    },
    {
      "id": "B-017",
      "severity": "INFO",
      "principle": "P6/P8",
      "status": "open",
      "finding": "Supervised leg evidence discloses: 5/7 traits have vacuous negative controls (R2 < -1, wide-design conditioning artifact; frozen VACUOUS_CONTROL_R2 doctrine) and the ledger-only diagnostic (beats table on 4/7 traits, best R2 0.8659 on x0 vs table 0.8791) falsifies the SPINE §5 corollary and 'needs an operator ruling'.",
      "forward_subgoal": "none required (disclosure); SPINE §5 baseline-vs-corollary ruling queued to the operator.",
      "gate_touched": "none (verdict derives from columns; checks are evidence text)",
      "evidence": "judge_leg_supervised.json checks.supervised.evidence; judge_spec.py:179-183",
      "audit_ref": "A-017",
      "date": "2026-11-05"
    },
    {
      "id": "B-018",
      "severity": "INFO",
      "principle": "E10",
      "status": "open",
      "finding": "plugin_acceptance.json (V1.2 reference receipt, n_rows 9) has no _meta; the gate receipt v1_plugins.json it points at is stamped (behavior df73c2aa, passed=true).",
      "forward_subgoal": "none required (disclosure); stamp the reference on a future V1.2 producer pass.",
      "gate_touched": "none",
      "evidence": "key lists of plugin_acceptance.json vs v1_plugins.json",
      "audit_ref": "A-018",
      "date": "2026-11-05"
    },
    {
      "id": "B-019",
      "severity": "MINOR",
      "principle": "P6/E10",
      "status": "closed",
      "closure": "evidence",
      "finding": "R7 route step evidence text (plan_spec.py:631, rendered at PLAN.md:619 and on the card) still reads the DEC-605-era count 'closed all 11 findings (8 by evidence, 3 by the scope/reclassification...)'; this file's record at the rescan tree is 16 of 18 closed (13 by evidence, 3 by scope, B-017/B-018 open INFO) - a card-audit prose contradiction.",
      "forward_subgoal": "One-line durable-outcome repair of the R7 evidence text without a hardcoded finding count (state the outcome; name the three scope closures B-001/B-003/B-008); re-render PLAN/card.",
      "gate_touched": "none (route prose; steps are pointers, not verdicts)",
      "evidence": "DEC-610; commit 2976485: R7 route evidence text rewritten to the durable outcome (no hardcoded count); PLAN.md and card re-rendered from the spec, behavior df73c2aa unchanged, commit gates RC=0; original finding: plan_spec.py:631 / PLAN.md:619 carried the DEC-605-era count",
      "audit_ref": "A-019",
      "date": "2026-11-05",
      "closed": "2026-11-05"
    },
    {
      "id": "B-020",
      "severity": "MAJOR",
      "principle": "P1 (Universal) / Honest (claim scope)",
      "status": "closed",
      "closure": "evidence",
      "finding": "The closing BACKWARD reviews (FULL at 38740c8...b512068) falsified the record's plumbing (citations, manifests, self-consistency, route text) but never the scope of the claim against the top-level Universal principle: every B-row and every pre-DEC-611 lab fixture is one ecom simulator family (TYPES/P_FIXED, view/cart/purchase) on the demo_v2 prototype. S1 proves the pipeline survives schema renaming; no empirical row tested that the embedding is useful on structurally different event domains. 'SOLVED 39/39 on the synthetic demo_v2 prototype' is route-honest, yet the charter's Universal principle outran its evidence base.",
      "forward_subgoal": "DEC-611 campaign (pre-registered 2b37adf): three structurally distinct known-truth domains through the ONE simulator, full three-piece controls, frozen lab_measure protocol. Closure: six lab receipts complete=true + outcome addendum recorded in DEC-611 (strengths/weaknesses + axis attribution + defect disclosure). Branch (separate DEC if exercised, operator ruling): grow the tree so multi-domain mastery becomes a gated card row - DEC-576 keeps these receipts dev instruments until then.",
      "gate_touched": "none (claims stay scoped to the demo_v2 prototype per B-001; dev-rig receipts are not B-row evidence per DEC-576; the card's verdict is unchanged)",
      "evidence": "six campaign receipts complete=true stamped 89cec70b (lab/sensor_fleet_dec611{,_hier}.json, lab/clickstream_sessions_dec611{,_hier}.json, lab/server_logs_dec611{,_hier}.json); DEC-611 outcome addendum (per-domain scorecard, axis attribution table, defect disclosure, re-stamp statement); this commit",
      "audit_ref": "A-020",
      "date": "2026-11-05",
      "closed": "2026-11-05",
      "provenance": "Operator-filed 2026-11-05 (critique of the closing reviews' adversarial coverage: 'PROMPT_BACKWARD should have been more adversarial and suggested testing on multiple types of streams'). NOT filed by a BACKWARD run; closed the same day by the DEC-611 campaign under its own stated closure criterion."
    },
    {
      "id": "B-021",
      "severity": "MINOR",
      "principle": "P6 (single source of truth) / P10 (reproducibility) / Honest",
      "status": "closed",
      "closure": "fix + re-stamp evidence",
      "finding": "The certification CHAIN in measure_certification_law.py still mapped the legs step to measure_judge_law.py, whose main() implements the pre-DEC-598 flat-era supervised/unsupervised legs and writes the same four judge_leg_*.json files that measure_bc_bank.py (bank v2, DEC-598) owns. Every clean-tree S6 re-stamp therefore overwrote the bank-v2 receipts with stale v1 numbers (supervised x0 0.886 with CI straddling 0; unsupervised persona_geometry 0.338608; segment_structure ARI ~0) and flipped the plan root from SOLVED 39/39 to 35/39. rl/simulator legs were unaffected (bc_bank delegates them to the judge_law LEGs).",
      "forward_subgoal": "Fix the CHAIN legs mapping to measure_bc_bank.py (one producer stamps all four files), script-only so behavior stays 89cec70b; re-stamp via the clean one-command chain; the card re-renders SOLVED 39/39.",
      "gate_touched": "none (gate/criterion frozen; the fix restores the receipts' producer to the one the DEC-598 protocol and the judge_spec rows expect; no value or bar changed)",
      "evidence": "DEC-612; 0470c79 (CHAIN fix, five commit gates RC=0); clean one-command re-stamp at 0470c79: all four legs stamped by measure_bc_bank.py, bank-v2 values byte-identical to the 9af0d0d wave receipts (x0 0.921226, CI [+0.000063, +0.000151]; persona_geometry 0.397574; segment ARI 1.0; rl/simulator unchanged), plan root SOLVED 39/39, B CERTIFIED 23/23, 8/8, S6 OK at the clean tree, backward open 0 (this commit carries the regenerated receipts + card)",
      "audit_ref": "A-021",
      "date": "2026-11-05",
      "closed": "2026-11-05",
      "provenance": "Forward-filed 2026-11-05 during DEC-611 closeout forensics (the clean-tree re-render flipped the plan root; receipt _meta forensics localized the overwrite to the chain's legs step). NOT filed by a BACKWARD run; closed the same day by fix + re-stamp evidence."
    },
    {
      "id": "B-022",
      "severity": "MAJOR",
      "principle": "P1 (Universal) / E4 (adapt to input by construction)",
      "status": "closed",
      "closure": "fix + no-op proof + re-stamp evidence (DEC-614)",
      "finding": "supervised_gate._raw_features (the model-free RFM/GBM comparator) hardcoded pl.col('value') for the monetary aggregate (supervised_gate.py:239) - a pipeline read on the raw stream, so any reference stream whose dominant numeric field is not 'value' (sensor -> reading, server -> code) crashed the supervised comparator; the demo_v2 reference stream hid it (its only numeric column is 'value').",
      "forward_subgoal": "Derive the monetary field via cfm_data._primary_numeric_field(d, cfg) with a numeric-free zero fallback, matching the encoder's own field resolution (the DEC-611 2a49b8d precedent); pin generality with tests/test_supervised_generality.py; no-op on the reference stream; re-stamp.",
      "gate_touched": "none (the comparator's field resolution is pipeline config, not the frozen target/fold/baseline/gate contract; reference-stream values are byte-identical)",
      "evidence": "DEC-614; looking_glass/supervised_gate.py:239 -> derived; tests/test_supervised_generality.py (3 hermetic tests: non-'value' monetary field consumed, name-agnostic output, numeric-free zero fallback); demo_v2 has a single numeric column 'value' so _primary_numeric_field resolves 'value' -> no-op; re-stamp at behavior 7f20fd0b moves no verdict-bearing value, card holds SOLVED 39/39",
      "audit_ref": "A-022",
      "date": "2026-11-06",
      "closed": "2026-11-06",
      "provenance": "Filed by the PROMPT_BACKWARD 4c generality & improvement hunt (operator directive 2026-11-06: 'make the project more adaptable, more situations'); closed the same turn by DEC-614."
    },
    {
      "id": "B-023",
      "severity": "MINOR",
      "principle": "E5 (modularity/contracts) / E12 (implement intent, not the example)",
      "status": "open",
      "finding": "onboard._derive_reward_event(df, value_col) (onboard.py:585) accepts value_col but aggregates the literal pl.col('value') (:590); correct only because its input is pre-canonicalized to a 'value' column (:710/:757). The parameter is vestigial and the signature is dishonest.",
      "forward_subgoal": "One-axis subgoal: make the signature honest - use value_col, or drop it and name the canonical-frame contract it actually relies on - with a no-op proof on the canonical frame; re-stamp.",
      "gate_touched": false,
      "evidence": "looking_glass/onboard.py:585-595",
      "audit_ref": "A-023",
      "date": "2026-11-06"
    },
    {
      "id": "B-024",
      "severity": "INFO",
      "principle": "P1 (Universal) - named scope",
      "status": "open",
      "finding": "The supervised LEG's target family is a frozen ecommerce evaluation contract: _labels reads $.value/$.touchpoint/$.category_code and filters event_type == 'purchase'/'return' (supervised_gate.py:37-55); TARGETS (:68-78) names the 8 targets. A non-ecommerce reference stream is a new operator-level evaluation contract, not a pipeline fix (the comparator half is B-022, closed).",
      "forward_subgoal": "None required (disclosure): the leg's generality is bounded to the declared target contract; a new reference stream is a separate DEC.",
      "gate_touched": false,
      "evidence": "looking_glass/supervised_gate.py:37-55,68-78",
      "audit_ref": "A-024",
      "date": "2026-11-06"
    },
    {
      "id": "B-025",
      "severity": "INFO",
      "principle": "P1 (Universal) - named scope",
      "status": "open",
      "finding": "The decision/RL LEG's reward contract is declared on the randomized demo stream: _arms_rewards hardcodes event_type LIKE '%send%' (arm) and event_type='purchase' + $.value (incremental-margin reward) (decision_certify.py:48-59). A different reward field or arm vocabulary is a new causality contract (AGENTS: evaluation targets + causality are non-negotiable fixed things), not a pipeline defect.",
      "forward_subgoal": "None required (disclosure): the decision leg's generality is bounded to the declared arm/reward contract.",
      "gate_touched": false,
      "evidence": "looking_glass/decision_certify.py:48-59",
      "audit_ref": "A-025",
      "date": "2026-11-06"
    }
  ]
}
-->
