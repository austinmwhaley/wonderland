# JUDGE CARD — PATH B > PATH A

*Generated `2026-10-30T14:03:12+00:00` · commit `843ba2c` (`foundation-rebuild` · dirty) · encoder behavior `fc533e2c09846f7e`*

**Assessment:** NOT DONE. Legs green: 1/8 · stages green: 5/6. This is a statement about receipts, not about effort; nothing here is a claim of universality (DEC-410: that word appears only when the coverage rows are green).

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## The judge — four legs × two numbers

| Leg | Column | Verdict | Evidence |
|---|---|---|---|
| supervised | brain alone (PROGRESS) | ❌ FAIL | price_sensitivity: destroyed control >= brain (-0.021761 vs -0.024886) |
| supervised | brain + ledger (DELIVERY) | ❌ FAIL | does not beat PATH A with CI excluding 0: x0, return_proneness, churn_day, regime_switches, final_regime_days |
| unsupervised | brain alone (PROGRESS) | ❌ FAIL | persona_geometry: random control >= brain (0.00093 vs 9.1e-05) |
| unsupervised | brain + ledger (DELIVERY) | ❌ FAIL | persona_geometry: destroyed control >= brain (0.000413 vs 5.7e-05) |
| rl | brain alone (PROGRESS) | ❌ FAIL | known_effect_uplift: destroyed control >= brain (0.000771 vs -0.004207) |
| rl | brain + ledger (DELIVERY) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| simulator | brain alone (PROGRESS) | ❌ FAIL | does not beat PATH A with CI excluding 0: dose_response_curve, argmax_optimum |
| simulator | brain + ledger (DELIVERY) | ❌ FAIL | does not beat PATH A with CI excluding 0: dose_response_curve, argmax_optimum |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ✅ PASS | [renamed_end_to_end] OK: renamed client_id/occurred_at/event/attributes/total -> onboard resolved {'customer_key': 'client_id', 'event_ts':   ·measure_input_law.py on 843ba2c |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ✅ PASS | [unseen_value_path] OK: frozen schema (11 event types, vocab sizes [4, 10, 16, 6, 4, 7, 4, 24, 4, 4, 0, 77, 3, 4, 4, 5, 7]) encoded a batch   ·measure_input_law.py on 843ba2c |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ✅ PASS | [scan_associativity] OK: merge law F(g∘f)=F(g)∘F(f) on 6 real sequences x 5 split points (float64, model's own decay/drive, continuous_time=  ·measure_dynamics_law.py on 843ba2c |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | [outlet_schema] OK: 7 views / 17500 rows; declared-not-served [] advertised | [decodability] OK: linear-head decoders beat random+destroyed   ·measure_views_law.py on 843ba2c |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ❌ FAIL | FAILED action_conditioned_outcome: known-effect uplift from the one-factor-daily randomized offer arm (P(buy|offer=opt) - P(buy|off>=2), >= 10 ar (+2 more)  ·measure_battery_law.py on 843ba2c |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ✅ PASS | [full_chain_in_one_invocation] OK: 6 producer(s) run by this one command (S1+S2 475.8s, S3 490.4s, S4 53.5s, S5 2.0s, legs 45.1s, prov 2.5s)  ·measure_certification_law.py on 843ba2c |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash fc533e2c09846f7e |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json :: brain — price_sensitivity: destroyed control >= brain (-0.021761 vs -0.024886)`
- `judge_leg_supervised.json :: brain_ledger — does not beat PATH A with CI excluding 0: x0, return_proneness, churn_day, regime_switches, final_regime_days`
- `judge_leg_unsupervised.json :: brain — persona_geometry: random control >= brain (0.00093 vs 9.1e-05)`
- `judge_leg_unsupervised.json :: brain_ledger — persona_geometry: destroyed control >= brain (0.000413 vs 5.7e-05)`
- `judge_leg_rl.json :: brain — known_effect_uplift: destroyed control >= brain (0.000771 vs -0.004207)`
- `judge_leg_simulator.json :: brain — does not beat PATH A with CI excluding 0: dose_response_curve, argmax_optimum`
- `judge_leg_simulator.json :: brain_ledger — does not beat PATH A with CI excluding 0: dose_response_curve, argmax_optimum`
- `judge_stages.json — FAILED action_conditioned_outcome: known-effect uplift from the one-factor-daily randomized offer arm (P(buy|offer=opt) - P(buy|off>=2), >= 10 ar (+2 more)  ·measure_battery_law.py on 843ba2c`

---

*The inherited status-quo card (`REPORT_CARD.md`, generated by `scripts/report_card.py`) measures the pipeline as it was. This card measures the rebuild; the two are not interchangeable and neither overrides the other (DEC-415).*
