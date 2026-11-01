# REPORT_CARD — PATH B > PATH A

*Generated `2026-11-01T19:34:30+00:00` · commit `6129656` (`foundation-rebuild` · dirty) · encoder behavior `a4fdd768fbf009a0`*

**Assessment:** NOT DONE. Legs green: 2/8 · stages green: 4/6 · backward open: 0. This is a statement about receipts, not about effort; nothing here is a claim of universality (DEC-410: that word appears only when the coverage rows are green).

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## The judge — four legs × two numbers

Progress (`brain alone`) is a strict win. Delivery (`brain + ledger`) is a **non-inferiority** claim vs PATH A (margin 0.01 R2); a delivery tie is not a brain win.

| Leg | Column | Verdict | Evidence |
|---|---|---|---|
| supervised | brain alone (PROGRESS) | ❌ FAIL | x0: does not beat PATH A with CI excluding 0; failing: x0, h0, return_proneness |
| supervised | brain + ledger (DELIVERY, non-inferiority) | ❌ FAIL | h0: outside the delivery non-inferiority margin (0.01 R2); failing: h0, price_sensitivity, return_proneness |
| unsupervised | brain alone (PROGRESS) | ❌ FAIL | persona_geometry: does not beat PATH A with CI excluding 0; failing: persona_geometry, future_geometry, segment_structure |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | ❌ FAIL | persona_geometry: outside the delivery non-inferiority margin (0.01 R2); failing: persona_geometry, future_geometry, segment_structure |
| rl | brain alone (PROGRESS) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| rl | brain + ledger (DELIVERY, non-inferiority) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| simulator | brain alone (PROGRESS) | ❌ FAIL | dose_response_curve: does not beat PATH A with CI excluding 0; failing: dose_response_curve, argmax_optimum |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ❌ FAIL | argmax_optimum: outside the delivery non-inferiority margin (0.01 R2); failing: argmax_optimum |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ✅ PASS | [renamed_end_to_end] OK: renamed client_id/occurred_at/event/attributes/total -> onboard resolved {'customer_key': 'client_id', 'event_ts':   ·measure_input_law.py on 6129656 |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ✅ PASS | [unseen_value_path] OK: frozen schema (7 event types, vocab sizes [1, 10, 4, 3, 7, 3, 1, 4, 4, 396, 1, 1, 1, 4, 4]) encoded a batch containi  ·measure_input_law.py on 6129656 |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ❌ FAIL | FAILED clock_periodicity: week-shift (exact cycle multiple): worst state/E drift 6.185e-02 (gate 1e-4); gaps-x10 control (tokens fixed): (+1 more)  ·measure_dynamics_law.py on 6129656 |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | [outlet_schema] OK: 7 views / 35000 rows; declared-not-served [] advertised | [decodability] OK: linear-head decoders beat random+destroyed   ·measure_views_law.py on 6129656 |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ❌ FAIL | FAILED distance_regression: lab-truth latent distance (x0/h0/d0/price_sens/return_prone/touch_pref, z-scored) vs distance over 20000 custo  ·measure_battery_law.py on 6129656 |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ✅ PASS | [full_chain_in_one_invocation] OK: 6 producer(s) run by this one command (S1+S2 189.5s, S3 13.5s, S4 72.7s, S5 2.2s, legs 85.0s, prov 2.2s);  ·measure_certification_law.py on 6129656 |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash a4fdd768fbf009a0 |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json :: brain — x0: does not beat PATH A with CI excluding 0; failing: x0, h0, return_proneness`
- `judge_leg_supervised.json :: brain_ledger — h0: outside the delivery non-inferiority margin (0.01 R2); failing: h0, price_sensitivity, return_proneness`
- `judge_leg_unsupervised.json :: brain — persona_geometry: does not beat PATH A with CI excluding 0; failing: persona_geometry, future_geometry, segment_structure`
- `judge_leg_unsupervised.json :: brain_ledger — persona_geometry: outside the delivery non-inferiority margin (0.01 R2); failing: persona_geometry, future_geometry, segment_structure`
- `judge_leg_simulator.json :: brain — dose_response_curve: does not beat PATH A with CI excluding 0; failing: dose_response_curve, argmax_optimum`
- `judge_leg_simulator.json :: brain_ledger — argmax_optimum: outside the delivery non-inferiority margin (0.01 R2); failing: argmax_optimum`
- `judge_stages.json — FAILED clock_periodicity: week-shift (exact cycle multiple): worst state/E drift 6.185e-02 (gate 1e-4); gaps-x10 control (tokens fixed): (+1 more)  ·measure_dynamics_law.py on 6129656`
- `judge_stages.json — FAILED distance_regression: lab-truth latent distance (x0/h0/d0/price_sens/return_prone/touch_pref, z-scored) vs distance over 20000 custo  ·measure_battery_law.py on 6129656`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*This is the only card. It is a pure function of the frozen gate spec (`scripts/card_spec.py`) and the receipts under `docs/report_cards/experiments/`; there is no separate audit card, and no row may be parked elsewhere (DEC-512).*
