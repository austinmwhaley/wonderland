# REPORT_CARD — B foundational, then PATH B > PATH A

*Generated `2026-11-03T18:54:57+00:00` · commit `b5566ca` (`foundation-rebuild` · clean) · encoder behavior `d5df3a2c1211c402`*

**Assessment:**

- ❌ **Plan root:** FAILED — 14/31 measured leaves green · 6 FAIL · 1 DEC-frozen · 5 v1 unassessed · reds: P4 (6).
- ❌ **B foundational:** NOT CERTIFIED — the current gate. 2/8 sections green · 12 rows PASS · 0 FAIL · 11 UNKNOWN/MISSING.
- ❌ **B+C consequence:** NOT DONE — 2/8 legs green · stages 6/6 · backward open: 0. Reported for diagnosis; downstream of B.
- ⚪ **Project v1:** not assessed on this card (full stack is a separate scope).

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## The plan — the tree this card measures (PLAN.md)

The goal, cut backwards into the problems that must all be solved. This section is generated from `scripts/plan_spec.py` — the single definition of the tree; `PLAN.md` is its readable render. The plan defines no gate: every measured leaf is a frozen row below, and a node's verdict is the worst of its children.

**Plan root:** ❌ FAILED — 14/31 measured leaves green · 6 FAIL · 1 DEC-frozen leaf (no card row) · 5 v1 leaves not assessed on this card.

| Level | Node | What must be true | Leaves | Verdict |
|---|---|---|---|---|
| 0 | **G** A certified universal donor, delivered | What is the whole problem? | 31 leaves | ❌ FAIL |
| 1 | **P1** The donor exists | Is there a per-stream artifact built from the right algebra, and is it an identity? | 4 leaves | ❔ UNKNOWN |
| 2 | **B1** Lawful | Is the state built from the right algebra? | B1.1, B1.2 | ❔ UNKNOWN |
| 2 | **B8** Reproducible | Is the artifact an identity? | B8.1, B8.2 | ❔ UNKNOWN |
| 1 | **P2** The donor suffices | Does the embedding lose nothing that matters about the future — for prediction and decision — … | 8 leaves | ❔ UNKNOWN |
| 2 | **B2** Sufficient | Does B lose nothing about the future? | B2.1, B2.2, B2.3 | ✅ PASS |
| 2 | **E1** Trunk capacity (the state-update algebra) | Can the state carry the stream's structure — not just its mass? | — | ⚪ FRONTIER |
| 2 | **B4** Decisive | Does B support action/counterfactual decisions? | B4.1, B4.2, B4.3 | ❔ UNKNOWN |
| 2 | **B5** Minimal & affordable | Is it the smallest sufficient object, and fast enough? | B5.1, B5.2 | ❔ UNKNOWN |
| 1 | **P3** The donor is usable | Can simple consumers read it, does it transfer beyond our known tasks, and does it stay honest … | 11 leaves | ❔ UNKNOWN |
| 2 | **B3** Accessible | Can simple heads read what is there, at every stage? | B3.1, B3.2, B3.3, B3.4, B3.5, B3.6 | ✅ PASS |
| 2 | **B6** Transferable | Does it add signal on held-out consumers? | B6.1, B6.2, B6.3 | ❔ UNKNOWN |
| 2 | **B7** Robust | Does it stay honest under shift and missingness? | B7.1, B7.2 | ❔ UNKNOWN |
| 1 | **P4** The consequence is delivered | Does PATH B (frozen encoder -> simple head) beat PATH A (wide table -> GBM incumbent) for the … | 8 leaves | ❌ FAIL |
| 2 | **JUDGE-supervised** supervised leg | forecast targets beat the incumbent GBM on the same splits | 2 rows | ❌ FAIL |
| 2 | **JUDGE-unsupervised** unsupervised leg | distance equals shared futures; structure beats table/random/destroyed | 2 rows | ❌ FAIL |
| 2 | **JUDGE-rl** rl leg | policy value beats table and random on randomized/known-effect data | 2 rows | ✅ PASS |
| 2 | **JUDGE-simulator** simulator leg | recovers the built-in counterfactual effects (red_king role) | 2 rows | ❌ FAIL |
| 1 | **P5** The stack is v1-complete | Does the full system deliver a certified decision: identifiable stream -> donor -> plugins -> … | — | ❔ UNKNOWN |
| 2 | **V1.1** Causally identifiable stream | Does rabbit_hole emit randomized arms with a known effect, recorded as such? | — | ⚪ NOT ASSESSED |
| 2 | **V1.2** Each plugin passes its independent gate | Do the supervised, unsupervised and white_queen consumers each pass their own gate? | — | ⚪ NOT ASSESSED |
| 2 | **V1.3** red_king recovers known effects | Does red_king recover the known counterfactual effect and improve the certified decision with … | — | ⚪ NOT ASSESSED |
| 2 | **V1.4** red_queen composes actions | Does red_queen compose next-best actions across cadences under constraints? | — | ⚪ NOT ASSESSED |
| 2 | **V1.5** caterpillar explains from artifacts | Does caterpillar answer 'why' from artifacts only? | — | ⚪ NOT ASSESSED |

*Frontier:* The live frontier is P2's enabler E1 (the trunk's state-update algebra). O3 attempt 4 was rejected on its frozen bars and the dev rig reproduced the blob in minutes; DEC-575 candidate 1 as shipped was negative on the toy (DEC-577 outcome), and candidate 2 as shipped — the looped / weight-tied nonlinear block on the state (DEC-578) — is negative on the toy too (DEC-578 outcome: no g improvement, CI still straddles the 1-gram, controls clean). The next move is candidate 3 — the looped block on the DRIVE side — declared DEC-581 and pre-registered under its own DEC, then the curriculum (intent), then ONE pre-registered port run under DEC-572's bars. No lab receipt flips a B-row verdict.

**Frontier readings (reported, not gated — the evidence behind E1's FRONTIER badge):**

- **dev rig · next_event iter1** — g 1.147 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.153 · beats 1-gram: no — `lab/next_event_iter1.json` · measured under `6a8e2663`
- **dev rig · next_event iter2** — g 1.154 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.473 · beats 1-gram: no — `lab/next_event_iter2.json` · measured under `4053d2f9`
- **dev rig · next_event iter3 (DEC-578, candidate 2)** — g 1.182 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.248 · beats 1-gram: no — `lab/next_event_iter3.json` · measured under `905b4f88`
- **real stream · O3 attempt 4 (REJECTED)** — M4 persona 0.04646 (bar ≥ 0.15, DEC-572) · served coarse R² 0.4675 (bar ≥ 0.45) — `o3axis_i/` · measured under `6a8e2663`

A green leaf means its frozen gate is met. For these leaves the gate certifies *measurement and attribution*, not a quality level (derived in `scripts/b_spec.py: gate_kind`, not asserted here): `B1.1`, `B2.3`, `B3.1`, `B3.2`, `B3.4`, `B3.5`, `B3.6`, `B6.1`, `B8.2`. Their quality levels are DEC-frozen and live in the receipts (`PLAN.md` §6). Read the plan before reading a green row as 'good'.

## B — foundational scorecard (B-only, DEC-562)

The judge below is the **consequence** (B+C). This section is **B itself**: a row is PASS only on a current receipt that meets its gate; a missing or legacy receipt is UNKNOWN and names the producer that must be run. Legacy numbers are not evidence.

### B1 — Lawful · ❔ UNKNOWN

*Is the state built from the right algebra?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B1.1 | Lawful dynamics (S3) | merge/equivariance/mass-additivity exact on data; tau receipted; stable from step 0 | reporting | ✅ PASS | judge_stages.json:S3 working=True |
| B1.2 | Construction laws (C1-C6) | scan / field-permutation / mass / determinism / schema variant / frozen hash | unmeasured | ❔ UNKNOWN | legacy receipt encoder_lawfulness.json (no behavior stamp); re-run the legacy encoder-construction producer |

### B2 — Sufficient · ✅ PASS

*Does B lose nothing about the future?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B2.1 | Worst-case sufficiency U | U = 1 - worst-case oracle gap vs full history, fixed capacity, broad published family | quality | ✅ PASS | served U 0.9757 (bar 0.5, DEC-570) · worst probe next-type · 1/5 probes included · excluded: attr-next, dt, horizon-count-value, numeric-next |
| B2.2 | Full-history oracle gap | excess loss of E vs the full-history arm at fixed capacity, per included probe | quality | ✅ PASS | every included probe clears the bar · next-type 0.9757 |
| B2.3 | Learned-only attribution | sufficiency of the learned shape vs the served statistics, reported separately | reporting | ✅ PASS | learned-only U 1.0769 · worst probe next-type · learned shape carries (the ledger does not add) |

### B3 — Accessible · ✅ PASS

*Can simple heads read what is there, at every stage?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B3.1 | Outlet accessibility (S4) | per-bucket accessibility parity + per-bucket negative controls | reporting | ✅ PASS | judge_stages.json:S4 working=True |
| B3.2 | Stage decomposition M0->M1->served | the fine-structure metric at each stage, so a loss is attributed to a stage | reporting | ✅ PASS | complete · persona M0 0.1893 → M1 0.06572 → served 0.06596 · future M0 0.1727 → M1 0.09323 → served 0.09095 · largest drop M0_exact_counter->M1_fold_only · M3/M4 0.9981 |
| B3.3 | Linear accessibility (current) | mean ridge R2 on A->B over the lab's persona+future fields; served vs PATH A | quality | ✅ PASS | served 0.4677 vs PATH A 0.4126 (delta +0.0551, margin -0.01; n_fields 11) |
| B3.4 | Block-aware serving geometry | exact counter block beside the learned shape, equal total variance; persona/future/ARI | reporting | ✅ PASS | complete · persona served 0.06596 / counter 0.1893 / block 0.09592 / PATH A 0.3867 · future served 0.09095 / counter 0.1727 / block 0.1169 / PATH A 0.181 · counter-weight sensitivity flat (block scale erased by per-dim z-score; dimension count is the lever) |
| B3.5 | Equal-dimension serving geometry | shape PCA + counter tile at 128/128 and 39/39; persona/future/ARI at full size | reporting | ✅ PASS | complete · persona equal_39 0.1721 / equal_128 0.1361 / counter 0.1893 / shape_pca_39 0.08139 / PATH A 0.3867 · future equal_39 0.144 / equal_128 0.1174 / counter 0.1727 / PATH A 0.181 |
| B3.6 | Served-object X-ray (ledger exposure) | the ledger alone, its named parts, and ledger block-PCA under the frozen B3.2/B3.4 consumers, against {exact counter, served shape, PATH A}; the branch is computed here from the arm numbers (DEC-569) | reporting | ✅ PASS | complete · branch exposure (DEC-569: best ledger arm persona 0.3196 from part_numeric_stats; full 0.01722 / pca_39 0.2012 / pca_128 0.1036) · references persona counter 0.1893 / shape 0.06596 / PATH A 0.3867 · linear access ledger_full 0.3551 / counter 0.3937 / shape 0.4677 / PATH A 0.4126 |

### B4 — Decisive · ❔ UNKNOWN

*Does B support action/counterfactual decisions?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B4.1 | B-only dose-response / argmax | action-conditional response shape readable from the state (not through a judge leg) | unmeasured | ❔ UNKNOWN | missing producer: measure_decision_law.py |
| B4.2 | OPE / IPS known-effect recovery (legacy) | E recovers the known effect with a paired CI excluding 0 | unmeasured | ❔ UNKNOWN | legacy receipt encoder_causal_ci.json (no behavior stamp); re-run the legacy causal producer |
| B4.3 | Action-invariance | representation invariant to assignment, still predictive of the outcome | unmeasured | ❔ UNKNOWN | missing producer: measure_decision_law.py |

### B5 — Minimal & affordable · ❔ UNKNOWN

*Is it the smallest sufficient object, and fast enough?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B5.1 | Minimality k_min | served rank = the frozen reconstruction certificate | unmeasured | ❔ UNKNOWN | missing producer: measure_minimality_law.py |
| B5.2 | Cost / runtime / constant-state (legacy) | runtime, memory and incremental-update receipt | unmeasured | ❔ UNKNOWN | legacy receipt runtime.json (no behavior stamp); re-run the legacy runtime producer |

### B6 — Transferable · ❔ UNKNOWN

*Does it add signal on held-out consumers?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B6.1 | Objective battery (S5) | every content pressure beats its naive comparator (or is excluded and reported) | reporting | ✅ PASS | judge_stages.json:S5 working=True |
| B6.2 | Donor does-not-lose / adds / controls (legacy) | E vs raw, fair table, table+seq; random/destroyed negative controls | unmeasured | ❔ UNKNOWN | legacy receipt encoder_donor.json (no behavior stamp); re-run the legacy donor producer |
| B6.3 | Broad held-out transfer family | adds signal on a family broader than the eight judge legs | unmeasured | ❔ UNKNOWN | missing producer: measure_transfer_law.py |

### B7 — Robust · ❔ UNKNOWN

*Does it stay honest under shift and missingness?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B7.1 | ECE / OOD retention (legacy) | calibration and OOD retention on the old reference | unmeasured | ❔ UNKNOWN | legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy calibration producer |
| B7.2 | Missingness / unlogged confounders | uncertainty carried for unobserved events, never faked | unmeasured | ❔ UNKNOWN | missing producer: measure_robustness_law.py |

### B8 — Reproducible · ❔ UNKNOWN

*Is the artifact an identity?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B8.1 | Artifact identity (legacy) | artifact hash + config + data revision recorded | unmeasured | ❔ UNKNOWN | legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy artifact producer |
| B8.2 | One-command reproducibility (S6) | the card is regenerated in one invocation at a recorded commit | reporting | ✅ PASS | judge_stages.json:S6 working=True |

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
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ✅ PASS | [renamed_end_to_end] OK: renamed client_id/occurred_at/event/attributes/total -> onboard resolved {'customer_key': 'client_id', 'event_ts':   ·measure_input_law.py on cb7c0d9 |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ✅ PASS | [unseen_value_path] OK: frozen schema (7 event types, vocab sizes [1, 10, 4, 3, 7, 3, 1, 4, 4, 396, 1, 1, 1, 4, 4]) encoded a batch containi  ·measure_input_law.py on cb7c0d9 |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ✅ PASS | [scan_associativity] OK: merge law F(g∘f)=F(g)∘F(f) on 6 real sequences x 5 split points (float64, model's own decay/drive, continuous_time=  ·measure_dynamics_law.py on cb7c0d9 |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | [outlet_schema] OK: 7 views / 35000 rows; declared-not-served [] advertised | [decodability] OK: linear-head decoders beat random+destroyed   ·measure_views_law.py on cb7c0d9 |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ✅ PASS | [ledger_reconstruction] OK: ledger recomputed from the raw stream by DuckDB (not a table copy); LINEAR head on the learned-shape donor (embe  ·measure_battery_law.py on cb7c0d9 |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ✅ PASS | [full_chain_in_one_invocation] OK: 6 producer(s) run by this one command (S1+S2 318.3s, S3 20.5s, S4 567.2s, S5 6.7s, legs 222.5s, prov 2.2s  ·measure_certification_law.py on cb7c0d9 |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash d5df3a2c1211c402 |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json :: brain [P4·JUDGE-supervised] — return_proneness: does not beat PATH A with CI excluding 0; failing: return_proneness`
- `judge_leg_supervised.json :: brain_ledger [P4·JUDGE-supervised] — price_sensitivity: random control >= brain (-0.061398 vs -0.079803); failing: price_sensitivity, return_proneness`
- `judge_leg_unsupervised.json :: brain [P4·JUDGE-unsupervised] — persona_geometry: does not beat PATH A with CI excluding 0; failing: persona_geometry, future_geometry, segment_structure`
- `judge_leg_unsupervised.json :: brain_ledger [P4·JUDGE-unsupervised] — persona_geometry: outside the delivery non-inferiority margin (0.01 R2); failing: persona_geometry, future_geometry, segment_structure`
- `judge_leg_simulator.json :: brain [P4·JUDGE-simulator] — argmax_optimum: destroyed control >= brain (0.253872 vs 0.245791); failing: argmax_optimum`
- `judge_leg_simulator.json :: brain_ledger [P4·JUDGE-simulator] — argmax_optimum: outside the delivery non-inferiority margin (0.01 R2); failing: argmax_optimum`
- `B B1.2 [P1·B1] Construction laws (C1-C6) — legacy receipt encoder_lawfulness.json (no behavior stamp); re-run the legacy encoder-construction producer`
- `B B4.1 [P2·B4] B-only dose-response / argmax — missing producer: measure_decision_law.py`
- `B B4.2 [P2·B4] OPE / IPS known-effect recovery (legacy) — legacy receipt encoder_causal_ci.json (no behavior stamp); re-run the legacy causal producer`
- `B B4.3 [P2·B4] Action-invariance — missing producer: measure_decision_law.py`
- `B B5.1 [P2·B5] Minimality k_min — missing producer: measure_minimality_law.py`
- `B B5.2 [P2·B5] Cost / runtime / constant-state (legacy) — legacy receipt runtime.json (no behavior stamp); re-run the legacy runtime producer`
- `B B6.2 [P3·B6] Donor does-not-lose / adds / controls (legacy) — legacy receipt encoder_donor.json (no behavior stamp); re-run the legacy donor producer`
- `B B6.3 [P3·B6] Broad held-out transfer family — missing producer: measure_transfer_law.py`
- `B B7.1 [P3·B7] ECE / OOD retention (legacy) — legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy calibration producer`
- `B B7.2 [P3·B7] Missingness / unlogged confounders — missing producer: measure_robustness_law.py`
- `B B8.1 [P1·B8] Artifact identity (legacy) — legacy receipt encoder_b_props.json (no behavior stamp); re-run the legacy artifact producer`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*This is the only card. B-only rows come from `scripts/b_spec.py`; B+C rows from `scripts/judge_spec.py`; both hang under the tree in `scripts/plan_spec.py` (rendered readably in `PLAN.md`); all read the receipts under `docs/report_cards/experiments/`. There is no separate audit card, and no row may be parked elsewhere (DEC-512).*
