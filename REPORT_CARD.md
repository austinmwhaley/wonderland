# REPORT_CARD — PATH B > PATH A

*Generated `2026-11-01T16:44:19+00:00` · commit `38170df` (`foundation-rebuild` · dirty) · encoder behavior `a4fdd768fbf009a0`*

**Assessment:** NOT DONE. Legs green: 0/8 · stages green: 3/6 · backward open: 0. This is a statement about receipts, not about effort; nothing here is a claim of universality (DEC-410: that word appears only when the coverage rows are green).

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
| rl | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (94857be6815a96f3 != a4fdd768fbf009a0) |
| rl | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (94857be6815a96f3 != a4fdd768fbf009a0) |
| simulator | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (94857be6815a96f3 != a4fdd768fbf009a0) |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (94857be6815a96f3 != a4fdd768fbf009a0) |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ✅ PASS | [renamed_end_to_end] OK: renamed client_id/occurred_at/event/attributes/total -> onboard resolved {'customer_key': 'client_id', 'event_ts':   ·measure_input_law.py on 38170df |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ✅ PASS | [unseen_value_path] OK: frozen schema (7 event types, vocab sizes [1, 10, 4, 3, 7, 3, 1, 4, 4, 396, 1, 1, 1, 4, 4]) encoded a batch containi  ·measure_input_law.py on 38170df |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ❌ FAIL | FAILED clock_periodicity: week-shift (exact cycle multiple): worst state/E drift 2.814e-02 (gate 1e-4); gaps-x10 control (tokens fixed): (+1 more)  ·measure_dynamics_law.py on 38170df |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | [outlet_schema] OK: 7 views / 35000 rows; declared-not-served [] advertised | [decodability] OK: linear-head decoders beat random+destroyed   ·measure_views_law.py on 38170df |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ❌ FAIL | FAILED action_conditioned_outcome: insufficient identified customers (A=0, B=0; >= 10 arm-days/side required) (+1 more)  ·measure_battery_law.py on 38170df |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ❌ FAIL | FAILED full_chain_in_one_invocation: 6 producer(s) run by this one command (S1+S2 166.9s, S3 12.8s, S4 82.0s, S5 2.3s, legs 66.4s, prov 2.3s); fres  ·measure_certification_law.py on 38170df |

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
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (94857be6815a96f3 != a4fdd768fbf009a0)`
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (94857be6815a96f3 != a4fdd768fbf009a0)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (94857be6815a96f3 != a4fdd768fbf009a0)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (94857be6815a96f3 != a4fdd768fbf009a0)`
- `judge_stages.json — FAILED clock_periodicity: week-shift (exact cycle multiple): worst state/E drift 2.814e-02 (gate 1e-4); gaps-x10 control (tokens fixed): (+1 more)  ·measure_dynamics_law.py on 38170df`
- `judge_stages.json — FAILED action_conditioned_outcome: insufficient identified customers (A=0, B=0; >= 10 arm-days/side required) (+1 more)  ·measure_battery_law.py on 38170df`
- `judge_stages.json — FAILED full_chain_in_one_invocation: 6 producer(s) run by this one command (S1+S2 166.9s, S3 12.8s, S4 82.0s, S5 2.3s, legs 66.4s, prov 2.3s); fres  ·measure_certification_law.py on 38170df`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*This is the only card. It is a pure function of the frozen gate spec (`scripts/card_spec.py`) and the receipts under `docs/report_cards/experiments/`; there is no separate audit card, and no row may be parked elsewhere (DEC-512).*
