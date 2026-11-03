# REPORT_CARD — B foundational, then PATH B > PATH A

*Generated `2026-11-03T02:16:46+00:00` · commit `8a55837` (`foundation-rebuild` · dirty) · encoder behavior `d2ad6f6058e9607d`*

**Assessment:**

- ❌ **B foundational:** NOT CERTIFIED — the current gate. 1/8 sections green · 8 rows PASS · 0 FAIL · 15 UNKNOWN/MISSING.
- ❌ **B+C consequence:** NOT DONE — 2/8 legs green · stages 5/6 · backward open: 0. Reported for diagnosis; downstream of B.
- ⚪ **Project v1:** not assessed on this card (full stack is a separate scope).

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## B capability — top line (O8 sprint; DEC-568/D2)

The sprint's unit of progress is one capability delta on these axes. Each axis is the worst verdict of the B rows that own it, computed from the receipts below — B+C remains the reported consequence section, never a top-line axis.

| Capability | B rows | Verdict | Headline |
|---|---|---|---|
| U · sufficiency | B2.1 | ❔ UNKNOWN | missing producer: measure_sufficiency_law.py |
| shape-only geometry | B3.2 | ✅ PASS | complete · persona M0 0.1893 → M1 0.06572 → served 0.06596 · future M0 0.1727 → M1 0.09323 → served 0.09095 · largest … |
| decision-shape | B4.1 | ❔ UNKNOWN | missing producer: measure_decision_law.py |
| minimality rank | B5.1 | ❔ UNKNOWN | missing producer: measure_minimality_law.py |
| transfer | B6.1, B6.3 | ❔ UNKNOWN | judge_stages.json:S5 working=True |
| robust | B7.1, B7.2 | ❔ UNKNOWN | legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy calibration producer |
| reproducible | B8.1, B8.2 | ❔ UNKNOWN | legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy artifact producer |

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

### B3 — Accessible · ✅ PASS

*Can simple heads read what is there, at every stage?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B3.1 | Outlet accessibility (S4) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | judge_stages.json:S4 working=True |
| B3.2 | Stage decomposition M0->M1->served | the fine-structure metric at each stage, so a loss is attributed to a stage | ✅ PASS | complete · persona M0 0.1893 → M1 0.06572 → served 0.06596 · future M0 0.1727 → M1 0.09323 → served 0.09095 · largest drop M0_exact_counter->M1_fold_only · M3/M4 0.9981 |
| B3.3 | Linear accessibility (current) | mean ridge R2 on A->B over the lab's persona+future fields; served vs PATH A | ✅ PASS | served 0.4677 vs PATH A 0.4126 (delta +0.0551, margin -0.01; n_fields 11) |
| B3.4 | Block-aware serving geometry | exact counter block beside the learned shape, equal total variance; persona/future/ARI | ✅ PASS | complete · persona served 0.06596 / counter 0.1893 / block 0.09592 / PATH A 0.3867 · future served 0.09095 / counter 0.1727 / block 0.1169 / PATH A 0.181 · counter-weight sensitivity flat (block scale erased by per-dim z-score; dimension count is the lever) |
| B3.5 | Equal-dimension serving geometry | shape PCA + counter tile at 128/128 and 39/39; persona/future/ARI at full size | ✅ PASS | complete · persona equal_39 0.1721 / equal_128 0.1361 / counter 0.1893 / shape_pca_39 0.08139 / PATH A 0.3867 · future equal_39 0.144 / equal_128 0.1174 / counter 0.1727 / PATH A 0.181 |
| B3.6 | Served-object X-ray (ledger exposure) | the ledger alone, its named parts, and ledger block-PCA under the frozen B3.2/B3.4 consumers, against {exact counter, served shape, PATH A}; the branch is computed here from the arm numbers (DEC-569) | ✅ PASS | complete · branch exposure (DEC-569: best ledger arm persona 0.3196 from part_numeric_stats; full 0.01722 / pca_39 0.2012 / pca_128 0.1036) · references persona counter 0.1893 / shape 0.06596 / PATH A 0.3867 · linear access ledger_full 0.3551 / counter 0.3937 / shape 0.4677 / PATH A 0.4126 |

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

### B8 — Reproducible · ❔ UNKNOWN

*Is the artifact an identity?*

| # | Property | Metric | Verdict | Evidence / missing producer |
|---|---|---|---|---|
| B8.1 | Artifact identity (legacy) | artifact hash + config + data revision recorded | ❔ UNKNOWN | legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy artifact producer |
| B8.2 | One-command reproducibility (S6) | the card is regenerated in one invocation at a recorded commit | ❔ UNKNOWN | stale: measured by measure_certification_law.py against behavior d0ac576c3c7a (current d2ad6f6058e9) — re-run that producer |

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
| simulator | brain alone (PROGRESS) | ❌ FAIL | argmax_optimum: destroyed control >= brain (0.253872 vs 0.245791); failing: argmax_optimum |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ❌ FAIL | argmax_optimum: outside the delivery non-inferiority margin (0.01 R2); failing: argmax_optimum |

## Measured values — every task on the card

The leg verdict above is computed from these rows; a task can PASS while its leg fails on another task. Every number below is read from the receipts, never narrated.

| Leg | Column | Task | Brain | PATH A | Delta | CI | Random | Destroyed | N | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised | brain alone (PROGRESS) | x0 | 0.9463 | 0.8791 | +0.06718 | [0.000118, 0.000215] | -1554 | -1529 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | h0 | 0.9741 | 0.9708 | +0.003331 | [0.000501, 0.001914] | -15.53 | -15.18 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | price_sensitivity | 0.02063 | -0.06363 | +0.08426 | [0.002668, 0.005767] | -7.036 | -7.378 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | return_proneness | 0.4118 | 0.4879 | -0.07612 | [-0.000375, -0.000185] | -42.99 | -38.56 | 1485 | ❌ FAIL |
| supervised | brain alone (PROGRESS) | churn_day | 0.3793 | -0.01375 | +0.393 | [6018, 8024] | -3588 | -3056 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | regime_switches | 0.1501 | -0.008355 | +0.1585 | [1.626, 2.954] | -330.8 | -318.6 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | final_regime_days | 0.003228 | -0.07448 | +0.07771 | [197.8, 454.1] | -0.01486 | -0.009254 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | x0 | 0.9368 | 0.8791 | +0.05768 | [0.000104, 0.000173] | -1147 | -1239 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | h0 | 0.9719 | 0.9708 | +0.00111 | [-0.000407, 0.001382] | -3.212 | -2.962 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | price_sensitivity | -0.0798 | -0.06363 | -0.01617 | [-0.002806, 0.001198] | -0.0614 | -0.08624 | 1485 | ❌ FAIL |
| supervised | brain + ledger (DELIVERY, non-inferiority) | return_proneness | 0.4689 | 0.4879 | -0.01903 | [-0.000146, 4e-06] | -0.1026 | -0.1078 | 1485 | ❌ FAIL |
| supervised | brain + ledger (DELIVERY, non-inferiority) | churn_day | 0.2578 | -0.01375 | +0.2715 | [3617, 6088] | -2266 | -2273 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | regime_switches | 0.0166 | -0.008355 | +0.02496 | [-0.4869, 1.2] | -196.2 | -183.9 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | final_regime_days | -0.07093 | -0.07448 | +0.003557 | [-159.2, 188.2] | -0.1289 | -0.09549 | 1485 | ✅ PASS |
| unsupervised | brain alone (PROGRESS) | persona_geometry | 0.06596 | 0.3867 | -0.3207 | [-0.3275, -0.3069] | 0.000598 | 0.00091 | 1.102e+06 | ❌ FAIL |
| unsupervised | brain alone (PROGRESS) | future_geometry | 0.09095 | 0.181 | -0.09006 | [-0.09544, -0.08207] | 3.4e-05 | 0.000506 | 1.102e+06 | ❌ FAIL |
| unsupervised | brain alone (PROGRESS) | segment_structure | -0.001097 | 0.2685 | -0.2696 | [-0.2964, -0.243] | 0.000108 | -0.001076 | 1485 | ❌ FAIL |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | persona_geometry | 0.05732 | 0.3867 | -0.3293 | [-0.34, -0.3253] | 0.000501 | 0.000492 | 1.102e+06 | ❌ FAIL |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | future_geometry | 0.08352 | 0.181 | -0.09749 | [-0.1024, -0.09381] | 1e-05 | 0.000304 | 1.102e+06 | ❌ FAIL |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | segment_structure | -0.000616 | 0.2685 | -0.2691 | [-0.2948, -0.2428] | 0.000561 | 0.000573 | 1485 | ❌ FAIL |
| rl | brain alone (PROGRESS) | known_effect_uplift | 0.2421 | 0.1794 | +0.06266 | [0.000937, 0.00255] | 0 | 0 | 1149 | ✅ PASS |
| rl | brain alone (PROGRESS) | policy_value | 0.2486 | 0.2382 | +0.01034 | [0.004621, 0.01619] | 0.1788 | 0.1846 | 1149 | ✅ PASS |
| rl | brain + ledger (DELIVERY, non-inferiority) | known_effect_uplift | 0.1766 | 0.1794 | -0.002855 | [-0.001061, 0.000996] | -0.09492 | -0.1009 | 1149 | ✅ PASS |
| rl | brain + ledger (DELIVERY, non-inferiority) | policy_value | 0.24 | 0.2382 | +0.001801 | [-0.004613, 0.008883] | 0.1668 | 0.1677 | 1149 | ✅ PASS |
| simulator | brain alone (PROGRESS) | dose_response_curve | 0.3164 | 0.3039 | +0.01246 | [7.5e-05, 0.000482] | 0 | 0 | 5000 | ✅ PASS |
| simulator | brain alone (PROGRESS) | argmax_optimum | 0.2458 | 0.2418 | +0.00404 | [-0.02559, 0.03165] | 0.2444 | 0.2539 | 1485 | ❌ FAIL |
| simulator | brain + ledger (DELIVERY, non-inferiority) | dose_response_curve | 0.3027 | 0.3039 | -0.001214 | [-0.000236, 0.000184] | -0.04729 | -0.05368 | 5000 | ✅ PASS |
| simulator | brain + ledger (DELIVERY, non-inferiority) | argmax_optimum | 0.2593 | 0.2418 | +0.01751 | [-0.0128, 0.04714] | 0.2559 | 0.2431 | 1485 | ❌ FAIL |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ✅ PASS | [renamed_end_to_end] OK: renamed client_id/occurred_at/event/attributes/total -> onboard resolved {'customer_key': 'client_id', 'event_ts':   ·measure_input_law.py on 8a55837 |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ✅ PASS | [unseen_value_path] OK: frozen schema (7 event types, vocab sizes [1, 10, 4, 3, 7, 3, 1, 4, 4, 396, 1, 1, 1, 4, 4]) encoded a batch containi  ·measure_input_law.py on 8a55837 |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ✅ PASS | [scan_associativity] OK: merge law F(g∘f)=F(g)∘F(f) on 6 real sequences x 5 split points (float64, model's own decay/drive, continuous_time=  ·measure_dynamics_law.py on 8a55837 |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | [outlet_schema] OK: 7 views / 35000 rows; declared-not-served [] advertised | [decodability] OK: linear-head decoders beat random+destroyed   ·measure_views_law.py on 8a55837 |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ✅ PASS | [ledger_reconstruction] OK: ledger recomputed from the raw stream by DuckDB (not a table copy); LINEAR head on the learned-shape donor (embe  ·measure_battery_law.py on 8a55837 |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ❔ UNKNOWN | stale: measured by measure_certification_law.py against behavior d0ac576c3c7a (current d2ad6f6058e9) — re-run that producer |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash d2ad6f6058e9607d |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json :: brain — return_proneness: does not beat PATH A with CI excluding 0; failing: return_proneness`
- `judge_leg_supervised.json :: brain_ledger — price_sensitivity: random control >= brain (-0.061398 vs -0.079803); failing: price_sensitivity, return_proneness`
- `judge_leg_unsupervised.json :: brain — persona_geometry: does not beat PATH A with CI excluding 0; failing: persona_geometry, future_geometry, segment_structure`
- `judge_leg_unsupervised.json :: brain_ledger — persona_geometry: outside the delivery non-inferiority margin (0.01 R2); failing: persona_geometry, future_geometry, segment_structure`
- `judge_leg_simulator.json :: brain — argmax_optimum: destroyed control >= brain (0.253872 vs 0.245791); failing: argmax_optimum`
- `judge_leg_simulator.json :: brain_ledger — argmax_optimum: outside the delivery non-inferiority margin (0.01 R2); failing: argmax_optimum`
- `judge_stages.json — stale: measured by measure_certification_law.py against behavior d0ac576c3c7a (current d2ad6f6058e9) — re-run that producer`
- `B B1.2 Construction laws (C1-C6) — legacy receipt encoder_lawfulness.json (no behavior stamp); re-run the legacy encoder-construction producer`
- `B B2.1 Worst-case sufficiency U — missing producer: measure_sufficiency_law.py`
- `B B2.2 Full-history oracle gap — missing producer: measure_sufficiency_law.py`
- `B B2.3 Learned-only attribution — missing producer: measure_sufficiency_law.py`
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
- `B B8.2 One-command reproducibility (S6) — stale: measured by measure_certification_law.py against behavior d0ac576c3c7a (current d2ad6f6058e9) — re-run that producer`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*This is the only card. B-only rows come from `scripts/b_spec.py`; B+C rows from `scripts/judge_spec.py`; both read the receipts under `docs/report_cards/experiments/`. There is no separate audit card, and no row may be parked elsewhere (DEC-512).*
