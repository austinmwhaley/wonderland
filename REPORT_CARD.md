# REPORT_CARD — PATH B > PATH A

*Generated `2026-11-02T16:32:03+00:00` · commit `6aeb94f` (`foundation-rebuild` · clean) · encoder behavior `d0ac576c3c7aa069`*

**Assessment:** NOT DONE. Legs green: 2/8 · stages green: 5/6 · backward open: 0. This is a statement about receipts, not about effort; nothing here is a claim of universality (DEC-410: that word appears only when the coverage rows are green). B foundational: 0/8 sections green · 3 rows PASS · 1 FAIL · 16 UNKNOWN/MISSING (B-only; B+C is the judge).

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## B — foundational scorecard (B-only, DEC-562)

The judge below is the **consequence** (B+C). This section is **B itself**: a row is PASS only on a current receipt that meets its gate; a missing or legacy receipt is UNKNOWN and names the producer that must be run. Legacy numbers are not evidence.

### B1 — Lawful · ❔ UNKNOWN

*Is the state built from the right algebra?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B1.1 | Lawful dynamics (S3) | merge/equivariance/mass-additivity exact on data; tau receipted; stable from step 0 | ✅ PASS | judge_stages.json:S3 working=True |
| B1.2 | Construction laws (C1-C6) | scan / field-permutation / mass / determinism / schema variant / frozen hash | ❔ UNKNOWN | legacy receipt encoder_lawfulness.json (no behavior stamp); re-run the legacy encoder-construction producer |

### B2 — Sufficient · ❔ UNKNOWN

*Does B lose nothing about the future?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B2.1 | Worst-case sufficiency U | U = 1 - worst-case oracle gap vs full history, fixed capacity, broad published family | ❔ UNKNOWN | missing producer: measure_sufficiency_law.py |
| B2.2 | Full-history oracle gap | excess loss of E vs the full-history oracle at fixed capacity | ❔ UNKNOWN | missing producer: measure_sufficiency_law.py |
| B2.3 | Learned-only attribution | sufficiency of the learned shape vs the served statistics, reported separately | ❔ UNKNOWN | missing producer: measure_sufficiency_law.py |

### B3 — Accessible · ❔ UNKNOWN

*Can simple heads read what is there, at every stage?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B3.1 | Outlet accessibility (S4) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | judge_stages.json:S4 working=True |
| B3.2 | Stage decomposition M0->M1->served | the fine-structure metric at each stage, so a loss is attributed to a stage | ❔ UNKNOWN | missing producer: measure_stage_decomposition.py |
| B3.3 | Linear accessibility (legacy) | ridge probe E vs table on the old 6k reference | ❔ UNKNOWN | legacy receipt encoder_accessibility.json (no behavior stamp); re-run the legacy accessibility producer |

### B4 — Decisive · ❔ UNKNOWN

*Does B support action/counterfactual decisions?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B4.1 | B-only dose-response / argmax | action-conditional response shape readable from the state (not through a judge leg) | ❔ UNKNOWN | missing producer: measure_decision_law.py |
| B4.2 | OPE / IPS known-effect recovery (legacy) | E recovers the known effect with a paired CI excluding 0 | ❔ UNKNOWN | legacy receipt encoder_causal_ci.json (no behavior stamp); re-run the legacy causal producer |
| B4.3 | Action-invariance | representation invariant to assignment, still predictive of the outcome | ❔ UNKNOWN | missing producer: measure_decision_law.py |

### B5 — Minimal & affordable · ❔ UNKNOWN

*Is it the smallest sufficient object, and fast enough?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B5.1 | Minimality k_min | served rank = the frozen reconstruction certificate | ❔ UNKNOWN | missing producer: measure_minimality_law.py |
| B5.2 | Cost / runtime / constant-state (legacy) | runtime, memory and incremental-update receipt | ❔ UNKNOWN | legacy receipt runtime.json (no behavior stamp); re-run the legacy runtime producer |

### B6 — Transferable · ❔ UNKNOWN

*Does it add signal on held-out consumers?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B6.1 | Objective battery (S5) | every content pressure beats its naive comparator (or is excluded and reported) | ✅ PASS | judge_stages.json:S5 working=True |
| B6.2 | Donor does-not-lose / adds / controls (legacy) | E vs raw, fair table, table+seq; random/destroyed negative controls | ❔ UNKNOWN | legacy receipt encoder_donor.json (no behavior stamp); re-run the legacy donor producer |
| B6.3 | Broad held-out transfer family | adds signal on a family broader than the eight judge legs | ❔ UNKNOWN | missing producer: measure_transfer_law.py |

### B7 — Robust · ❔ UNKNOWN

*Does it stay honest under shift and missingness?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B7.1 | ECE / OOD retention (legacy) | calibration and OOD retention on the old reference | ❔ UNKNOWN | legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy calibration producer |
| B7.2 | Missingness / unlogged confounders | uncertainty carried for unobserved events, never faked | ❔ UNKNOWN | missing producer: measure_robustness_law.py |

### B8 — Reproducible · ❌ FAIL

*Is the artifact an identity?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B8.1 | Artifact identity (legacy) | artifact hash + config + data revision recorded | ❔ UNKNOWN | legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy artifact producer |
| B8.2 | One-command reproducibility (S6) | the card is regenerated in one invocation at a recorded commit | ❌ FAIL | FAILED commit_recorded_clean: commit 99da0fbd2a3d on foundation-rebuild, behavior d0ac576c3c7aa069; tree carries 1 non-generated change(s):  (+1 more) |

## The judge — four legs × two numbers

Progress (`brain alone`) is a strict win. Delivery (`brain + ledger`) is a **non-inferiority** claim vs PATH A (margin 0.01 R2); a delivery tie is not a brain win.

| Leg | Column | Verdict | Evidence |
|---|---|---|---|
| supervised | brain alone (PROGRESS) | ❌ FAIL | return_proneness: does not beat PATH A with CI excluding 0; failing: return_proneness |
| supervised | brain + ledger (DELIVERY, non-inferiority) | ❌ FAIL | price_sensitivity: random control >= brain (-0.061398 vs -0.079803); failing: price_sensitivity, return_proneness |
| unsupervised | brain alone (PROGRESS) | ❌ FAIL | persona_geometry: does not beat PATH A with CI excluding 0; failing: persona_geometry, future_geometry, segment_structure |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | ❌ FAIL | persona_geometry: outside the delivery non-inferiority margin (0.01 R2); failing: persona_geometry, future_geometry, segment_structure |
| rl | brain alone (PROGRESS) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| rl | brain + ledger (DELIVERY, non-inferiority) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| simulator | brain alone (PROGRESS) | ❌ FAIL | argmax_optimum: does not beat PATH A with CI excluding 0; failing: argmax_optimum |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ❌ FAIL | argmax_optimum: outside the delivery non-inferiority margin (0.01 R2); failing: argmax_optimum |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ✅ PASS | [renamed_end_to_end] OK: renamed client_id/occurred_at/event/attributes/total -> onboard resolved {'customer_key': 'client_id', 'event_ts':   ·measure_input_law.py on 986da0a |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ✅ PASS | [unseen_value_path] OK: frozen schema (7 event types, vocab sizes [1, 10, 4, 3, 7, 3, 1, 4, 4, 396, 1, 1, 1, 4, 4]) encoded a batch containi  ·measure_input_law.py on 986da0a |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ✅ PASS | [scan_associativity] OK: merge law F(g∘f)=F(g)∘F(f) on 6 real sequences x 5 split points (float64, model's own decay/drive, continuous_time=  ·measure_dynamics_law.py on 420957f |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | [outlet_schema] OK: 7 views / 35000 rows; declared-not-served [] advertised | [decodability] OK: linear-head decoders beat random+destroyed   ·measure_views_law.py on 99da0fb |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ✅ PASS | [ledger_reconstruction] OK: ledger recomputed from the raw stream by DuckDB (not a table copy); LINEAR head on the learned-shape donor (embe  ·measure_battery_law.py on 99da0fb |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ❌ FAIL | FAILED commit_recorded_clean: commit 99da0fbd2a3d on foundation-rebuild, behavior d0ac576c3c7aa069; tree carries 1 non-generated change(s):  (+1 more)  ·measure_certification_law.py on 99da0fb |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash d0ac576c3c7aa069 |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json :: brain — return_proneness: does not beat PATH A with CI excluding 0; failing: return_proneness`
- `judge_leg_supervised.json :: brain_ledger — price_sensitivity: random control >= brain (-0.061398 vs -0.079803); failing: price_sensitivity, return_proneness`
- `judge_leg_unsupervised.json :: brain — persona_geometry: does not beat PATH A with CI excluding 0; failing: persona_geometry, future_geometry, segment_structure`
- `judge_leg_unsupervised.json :: brain_ledger — persona_geometry: outside the delivery non-inferiority margin (0.01 R2); failing: persona_geometry, future_geometry, segment_structure`
- `judge_leg_simulator.json :: brain — argmax_optimum: does not beat PATH A with CI excluding 0; failing: argmax_optimum`
- `judge_leg_simulator.json :: brain_ledger — argmax_optimum: outside the delivery non-inferiority margin (0.01 R2); failing: argmax_optimum`
- `judge_stages.json — FAILED commit_recorded_clean: commit 99da0fbd2a3d on foundation-rebuild, behavior d0ac576c3c7aa069; tree carries 1 non-generated change(s):  (+1 more)  ·measure_certification_law.py on 99da0fb`
- `B B1.2 Construction laws (C1-C6) — legacy receipt encoder_lawfulness.json (no behavior stamp); re-run the legacy encoder-construction producer`
- `B B2.1 Worst-case sufficiency U — missing producer: measure_sufficiency_law.py`
- `B B2.2 Full-history oracle gap — missing producer: measure_sufficiency_law.py`
- `B B2.3 Learned-only attribution — missing producer: measure_sufficiency_law.py`
- `B B3.2 Stage decomposition M0->M1->served — missing producer: measure_stage_decomposition.py`
- `B B3.3 Linear accessibility (legacy) — legacy receipt encoder_accessibility.json (no behavior stamp); re-run the legacy accessibility producer`
- `B B4.1 B-only dose-response / argmax — missing producer: measure_decision_law.py`
- `B B4.2 OPE / IPS known-effect recovery (legacy) — legacy receipt encoder_causal_ci.json (no behavior stamp); re-run the legacy causal producer`
- `B B4.3 Action-invariance — missing producer: measure_decision_law.py`
- `B B5.1 Minimality k_min — missing producer: measure_minimality_law.py`
- `B B5.2 Cost / runtime / constant-state (legacy) — legacy receipt runtime.json (no behavior stamp); re-run the legacy runtime producer`
- `B B6.2 Donor does-not-lose / adds / controls (legacy) — legacy receipt encoder_donor.json (no behavior stamp); re-run the legacy donor producer`
- `B B6.3 Broad held-out transfer family — missing producer: measure_transfer_law.py`
- `B B7.1 ECE / OOD retention (legacy) — legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy calibration producer`
- `B B7.2 Missingness / unlogged confounders — missing producer: measure_robustness_law.py`
- `B B8.1 Artifact identity (legacy) — legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy artifact producer`
- `B B8.2 One-command reproducibility (S6) — FAILED commit_recorded_clean: commit 99da0fbd2a3d on foundation-rebuild, behavior d0ac576c3c7aa069; tree carries 1 non-generated change(s):  (+1 more)`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*This is the only card. B-only rows come from `scripts/b_spec.py`; B+C rows from `scripts/judge_spec.py`; both read the receipts under `docs/report_cards/experiments/`. There is no separate audit card, and no row may be parked elsewhere (DEC-512).*
