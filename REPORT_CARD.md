# REPORT_CARD — B foundational, then PATH B > PATH A

*Generated `2026-11-05T04:19:44+00:00` · commit `effddbb` (`foundation-rebuild` · dirty) · encoder behavior `df73c2aa1427b471`*

**Assessment:**

- ✅ **Plan root:** SOLVED — 39/39 atomic problems green · 0 FAIL · 0 DEC-frozen · 0 v1 unassessed · 19 interior.
- ✅ **B foundational:** CERTIFIED — 8/8 sections green. Scope: the 23 receipts are current at behavior `df73c2aa`; the served-composition rows (B4.2/B5.1/B6.2/B6.3) read the derived bank (`c32rvd` + profile/numeric/mass blocks; B6.2 uses the 59-dim donor slice), the rest the base candidate/encoder. B6.2 uses the recorded ensemble protocol and B4.2 the randomized-period protocol; both are on the receipts.
- ✅ **B+C consequence:** DONE (ENCODER DONE v1) — 8/8 legs green · stages 6/6 · no BREAKING/MAJOR backward requirement. Bank v2 (DEC-598): the served block bank + compact judge ledger; the v1 flat judge (4/8) and the GBM delivery head are recorded beside it.
- ✅ **Project v1:** V1.1-V1.5 green — identifiable stream, plugin acceptance 9/9, red_king known-effect SCORE 0.889, red_queen channel certification, caterpillar artifact-only explanation.

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## The plan — the tree this card measures (PLAN.md)

The goal, cut backwards into the problems that must all be solved, recursively to the atomic problems. This section is generated from `scripts/plan_spec.py` — the single definition of the tree; `PLAN.md` is its readable render. The plan defines no gate: every measured atomic problem is a frozen row in its B/judge section below, and an interior verdict is the worst of its children.

**Plan root:** ✅ SOLVED — 39/39 atomic problems green · 0 FAIL · 0 DEC-frozen (no card rows) · 0 v1 not assessed on this card · 19 interior problems.

| Level | Node | What must be true | Atomic problems | Verdict |
|---|---|---|---|---|
| 0 | **G** A certified universal donor, delivered | What is the whole problem? | 39 atomic problems | ✅ PASS |
| 1 | **P1** The donor exists | Is there a per-stream artifact built from the right algebra, and is it an identity? | 4 atomic problems | ✅ PASS |
| 2 | **B1** Lawful | Is the state built from the right algebra? | 2 atomic problems | ✅ PASS |
| 2 | **B8** Reproducible | Is the artifact an identity? | 2 atomic problems | ✅ PASS |
| 1 | **P2** The donor suffices | Does the embedding lose nothing that matters about the future — for prediction and decision — … | 11 atomic problems | ✅ PASS |
| 2 | **B2** Sufficient | Does B lose nothing about the future? | 3 atomic problems | ✅ PASS |
| 2 | **E1** Trunk capacity (the state-update algebra) | Can the state carry the stream's structure — not just its mass? | 3 atomic problems | ✅ PASS |
| 3 | **E1.a** next_event mastered on the toy (curriculum step 1) | Does the frozen trunk beat the 1-gram on the known-truth next_event structure at the dev-rig … | — | ✅ PASS |
| 3 | **E1.b** intent mastered on the toy (curriculum step 2) | Does the frozen trunk beat the 1-gram on the per-customer intent structure, at a scale where … | — | ✅ PASS |
| 3 | **E1.c** port: ONE real-data run under DEC-572's bars | Does the winning trunk master the real stream — eff_rank>=30, persona>=0.15, R2>=0.45, stable, … | — | ✅ PASS |
| 2 | **B4** Decisive | Does B support action/counterfactual decisions? | 3 atomic problems | ✅ PASS |
| 2 | **B5** Minimal & affordable | Is it the smallest sufficient object, and fast enough? | 2 atomic problems | ✅ PASS |
| 1 | **P3** The donor is usable | Can simple consumers read it, does it transfer beyond our known tasks, and does it stay honest … | 11 atomic problems | ✅ PASS |
| 2 | **B3** Accessible | Can simple heads read what is there, at every stage? | 6 atomic problems | ✅ PASS |
| 2 | **B6** Transferable | Does it add signal on held-out consumers? | 3 atomic problems | ✅ PASS |
| 2 | **B7** Robust | Does it stay honest under shift and missingness? | 2 atomic problems | ✅ PASS |
| 1 | **P4** The consequence is delivered | Does PATH B (frozen encoder -> simple head) beat PATH A (wide table -> GBM incumbent) for the … | 8 atomic problems | ✅ PASS |
| 2 | **JUDGE-supervised** supervised leg | forecast targets beat the incumbent GBM on the same splits | 2 atomic problems | ✅ PASS |
| 2 | **JUDGE-unsupervised** unsupervised leg | distance equals shared futures; structure beats table/random/destroyed | 2 atomic problems | ✅ PASS |
| 2 | **JUDGE-rl** rl leg | policy value beats table and random on randomized/known-effect data | 2 atomic problems | ✅ PASS |
| 2 | **JUDGE-simulator** simulator leg | recovers the built-in counterfactual effects (red_king role) | 2 atomic problems | ✅ PASS |
| 1 | **P5** The stack is v1-complete | Does the full system deliver a certified decision: identifiable stream -> donor -> plugins -> … | 5 atomic problems | ✅ PASS |
| 2 | **V1.1** Causally identifiable stream | Does rabbit_hole emit randomized arms with a known effect, recorded as such? | — | ✅ PASS |
| 2 | **V1.2** Each plugin passes its independent gate | Do the supervised, unsupervised and white_queen consumers each pass their own gate? | — | ✅ PASS |
| 2 | **V1.3** red_king recovers known effects | Does red_king recover the known counterfactual effect and improve the certified decision with … | — | ✅ PASS |
| 2 | **V1.4** red_queen composes actions | Does red_queen compose next-best actions across cadences under constraints? | — | ✅ PASS |
| 2 | **V1.5** caterpillar explains from artifacts | Does caterpillar answer 'why' from artifacts only? | — | ✅ PASS |

*Macro review:* At behavior `7baf744f`, B is certified (23/23 rows, 8/8 sections), B+C is 8/8 leg-columns under the bank protocol (DEC-598), and the three plugin gates pass 9/9 on the bank + demo stream (supervised 6/6, unsupervised 4/4, white_queen 5/5, Layer C 9/9). The served block bank is 32 certified shape PCs + level/response/RFM/action + exact counter + schema profile + numeric/mass blocks, with a compact judge ledger; the supervised consumer is a standardized ridge alpha=10 and the geometry consumer fits non-negative block weights on split A. The v1 flat judge (4/8), the GBM delivery head and the old canonical stream are recorded beside it. The full plan root remains OPEN at E1 (the trunk) and the v1 atoms red_king/red_queen/caterpillar/identifiable-stream, outside the B+B+C+plugins definition.

*Frontier:* The B+B+C+plugins success definition is met under DEC-598: B is certified (23/23 rows, 8/8 sections); the four judge legs are 8/8 on the served block bank (supervised ridge alpha=10 + delivery bank + compact ledger; block-aware geometry; RL and simulator canonical); and the three plugin gates pass 9/9 (supervised 6/6, unsupervised 4/4, white_queen 5/5, Layer C 9/9) on the bank + demo stream. The v1 flat judge (4/8), the GBM delivery head and the old canonical stream are recorded beside it. The full plan root remains OPEN at E1 (trunk) and the v1 atoms red_king/red_queen/caterpillar and the identifiable-stream metadata, which are outside the B+B+C+plugins definition.

**Frontier readings (reported, not gated — the evidence behind E1's FRONTIER badge):**

- **dev rig · next_event iter1** — g 1.147 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.153 · beats 1-gram: no — `lab/next_event_iter1.json` · measured under `6a8e2663`
- **dev rig · next_event iter2** — g 1.154 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.473 · beats 1-gram: no — `lab/next_event_iter2.json` · measured under `4053d2f9`
- **dev rig · next_event iter3 (DEC-578, candidate 2)** — g 1.182 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.248 · beats 1-gram: no — `lab/next_event_iter3.json` · measured under `905b4f88`
- **dev rig · next_event iter4 (DEC-581, candidate 3)** — g 1.17 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.836 · beats 1-gram: no — `lab/next_event_iter4.json` · measured under `d5df3a2c`
- **dev rig · next_event cap256 (DEC-582, capacity @ baseline algebra)** — g 1.149 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.209 · beats 1-gram: no — `lab/next_event_cap256.json` · measured under `d5df3a2c`
- **dev rig · intent (DEC-586, E1.b)** — g 2.665 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.177 · beats 1-gram: no — `lab/intent_dec586.json` · measured under `7baf744f`
- **dev rig · intent + mass counter** — g 2.633 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.309 · beats 1-gram: no — `lab/intent_dec586_counter.json` · measured under `7baf744f`
- **dev rig · intent + transition signature** — g 2.6 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.177 · beats 1-gram: no — `lab/intent_dec586_sig.json` · measured under `7baf744f`
- **real stream · O3 attempt 4 (REJECTED)** — M4 persona 0.04646 (bar ≥ 0.15, DEC-572) · served coarse R² 0.4675 (bar ≥ 0.45) — `o3axis_i/` · measured under `6a8e2663`

A green atom means its frozen gate is met. For these atoms the gate certifies *measurement and attribution*, not a quality level (derived in `scripts/b_spec.py: gate_kind`, not asserted here): `B1.1`, `B2.3`, `B3.1`, `B3.2`, `B3.4`, `B3.5`, `B3.6`, `B4.3`, `B5.2`, `B6.1`, `B7.2`, `B8.1`, `B8.2`. Their quality levels are DEC-frozen and live in the receipts (`PLAN.md` §6). Read the plan before reading a green row as 'good'.

## B — foundational scorecard (B-only, DEC-562)

The judge below is the **consequence** (B+C). This section is **B itself**: a row is PASS only on a current receipt that meets its gate; a missing or legacy receipt is UNKNOWN and names the producer that must be run. Legacy numbers are not evidence.

### B1 — Lawful · ✅ PASS

*Is the state built from the right algebra?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B1.1 | Lawful dynamics (S3) | merge/equivariance/mass-additivity exact on data; tau receipted; stable from step 0 | reporting | ✅ PASS | judge_stages.json:S3 working=True |
| B1.2 | Construction laws (C1-C6) | scan / field-permutation / mass / determinism / schema variant / frozen identity | quality | ✅ PASS | C1-C6 all met: field-perm 1.714e-07 · scan 1.105e-15 · mass 4.819e-16 · determinism bit-identical · schema variant ok · identity present (N=20, cpu) |

### B2 — Sufficient · ✅ PASS

*Does B lose nothing about the future?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B2.1 | Worst-case sufficiency U | U = 1 - worst-case oracle gap vs full history, fixed capacity, broad published family | quality | ✅ PASS | served U 0.9802 (bar 0.5, DEC-570) · worst probe next-type · 1/5 probes included · excluded: attr-next, dt, horizon-count-value, numeric-next |
| B2.2 | Full-history oracle gap | excess loss of E vs the full-history arm at fixed capacity, per included probe | quality | ✅ PASS | every included probe clears the bar · next-type 0.9802 |
| B2.3 | Learned-only attribution | sufficiency of the learned shape vs the served statistics, reported separately | reporting | ✅ PASS | learned-only U 1.0764 · worst probe next-type · learned shape carries (the ledger does not add) |

### B3 — Accessible · ✅ PASS

*Can simple heads read what is there, at every stage?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B3.1 | Outlet accessibility (S4) | per-bucket accessibility parity + per-bucket negative controls | reporting | ✅ PASS | judge_stages.json:S4 working=True |
| B3.2 | Stage decomposition M0->M1->served | the fine-structure metric at each stage, so a loss is attributed to a stage | reporting | ✅ PASS | complete · persona M0 0.1965 → M1 0.05979 → served 0.2672 · future M0 0.1755 → M1 0.09209 → served 0.177 · largest drop M0_exact_counter->M1_fold_only · M3/M4 0.2078 |
| B3.3 | Linear accessibility (current) | mean ridge R2 on A->B over the lab's persona+future fields; served vs PATH A | quality | ✅ PASS | served 0.46 vs PATH A 0.4126 (delta +0.0474, margin -0.01; n_fields 11) |
| B3.4 | Block-aware serving geometry | exact counter block beside the learned shape, equal total variance; persona/future/ARI | reporting | ✅ PASS | complete · persona served 0.2672 / counter 0.1965 / block 0.2636 / PATH A 0.3867 · future served 0.177 / counter 0.1755 / block 0.1835 / PATH A 0.181 · counter-weight sensitivity flat (block scale erased by per-dim z-score; dimension count is the lever) |
| B3.5 | Equal-dimension serving geometry | shape PCA + counter tile at 128/128 and 39/39; persona/future/ARI at full size | reporting | ✅ PASS | complete · persona equal_39 0.2639 / equal_128 0.1995 / counter 0.1965 / shape_pca_39 0.2398 / PATH A 0.3867 · future equal_39 0.1792 / equal_128 0.1578 / counter 0.1755 / PATH A 0.181 |
| B3.6 | Served-object X-ray (ledger exposure) | the ledger alone, its named parts, and ledger block-PCA under the frozen B3.2/B3.4 consumers, against {exact counter, served shape, PATH A}; the branch is computed here from the arm numbers (DEC-569) | reporting | ✅ PASS | complete · branch exposure (DEC-569: best ledger arm persona 0.3196 from part_numeric_stats; full 0.01722 / pca_39 0.2012 / pca_128 0.1036) · references persona counter 0.1965 / shape 0.09277 / PATH A 0.3867 · linear access ledger_full 0.3551 / counter 0.4058 / shape 0.4865 / PATH A 0.4126 |

### B4 — Decisive · ✅ PASS

*Does B support action/counterfactual decisions?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B4.1 | B-only dose-response / argmax | action-conditional response shape readable from the state (not through a judge leg) | quality | ✅ PASS | E share 0.2983 vs table 0.2438 (CI [+0.0249, +0.0855]) · random 0.2586 · destroyed 0.2384 · n=1485 · oracle ceiling 0.6404 |
| B4.2 | OPE / IPS known-effect recovery | E recovers the known effect with a paired CI excluding 0 | quality | ✅ PASS | E policy IPS 0.1701 vs table 0.1517 (diff +0.02 [+0.01, 0.0246] > 0) · accE 0.335 vs accT 0.247 · random IPS None · n_test=74746 · levels {'0': 0.2515, '1': 0.2482, '2': 0.252, '3': 0.2482} |
| B4.3 | Action-invariance | representation invariant to assignment, still predictive of the outcome | reporting | ✅ PASS | band cert ok (P1==P2 bit-identical; P1 vs stored S 2.384185791015625e-07 / E 0.0009585114639776293 vs bands 6.199199447110004e-07 / 0.015569581650197506; teeth True) · marker null: share 0.30033670033670035 vs stored 0.2983164983164983 (ok=True) · level permutation: share 0.24511784511784512 vs table 0.24377104377104378 (|Δ|∞ 313.1736521923858, ok=True) |

### B5 — Minimal & affordable · ✅ PASS

*Is it the smallest sufficient object, and fast enough?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B5.1 | Minimality k_min | served rank = the frozen reconstruction certificate | quality | ✅ PASS | served shape rank 32 == k_min 32 (shape full 32) · explicit block 147 · joint floor 48 (unconstrained) · n_train 451 / n_test 439 · noise 0.04773 |
| B5.2 | Cost / runtime / constant-state | runtime, memory and incremental-update receipt | reporting | ✅ PASS | 5000/5000 customers · cold 11.8s / warm 84.74s · 53.1 rows/s · dims 256/256 · peak RSS 3360.3 MB (slice 32.0 GB) · incremental (absorbed 0, idle 5000, wall 3.45s) |

### B6 — Transferable · ✅ PASS

*Does it add signal on held-out consumers?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B6.1 | Objective battery (S5) | every content pressure beats its naive comparator (or is excluded and reported) | reporting | ✅ PASS | judge_stages.json:S5 working=True |
| B6.2 | Donor does-not-lose / adds / controls | E vs raw, fair table, table+E; random/destroyed negative controls | quality | ✅ PASS | n_A 3515 / n_B 1485 · y_churn_90: E 0.7388 vs raw 0.6800 / fair 0.6800 / table+E 0.7297 (CI lo +0.0037) / hstack 0.7534 (CI lo 0.007398) / rand 0.4076 / dest 0.5046 · y_orders_90: E 0.6216 vs raw 0.6047 / fair 0.6047 / table+E 0.6172 (CI lo +0.0052) / hstack 0.5943 (CI lo -0.04206) / rand -0.0163 / dest -0.0192 · y_value_90: E 0.6096 vs raw 0.5952 / fair 0.5952 / table+E 0.6054 (CI lo +0.0008) / hstack 0.5791 (CI lo -0.04908) / rand -0.0259 / dest -0.0119 |
| B6.3 | Broad held-out transfer family | adds signal on a family broader than the eight judge legs | quality | ✅ PASS | published family 15 targets · E wins 13/15 (0.867) · family mean E 0.4331 vs table 0.3857 (diff +0.0475) · random 0.0098 / destroyed 0.0043 |

### B7 — Robust · ✅ PASS

*Does it stay honest under shift and missingness?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B7.1 | ECE / OOD retention | calibration of the shipped next-type head + OOD retention | quality | ✅ PASS | next-type ECE 0.0302 (bar 0.1) · acc 0.35824915824915826 · n 1485/1485 (OOV 0) · OOD (as_of+30d): ece nan acc nan n 0 |
| B7.2 | Missingness / unlogged confounders | uncertainty carried for unobserved events, never faked | reporting | ✅ PASS | next-time 0.8 coverage 0.800 (n 1485, MAE 1.15d) · occurrence AUC 0.6339 / Brier 0.2794 (h 1.338484d) · response coverage 0.8/0.8 · missingness mean rel L2 10%:0.094, 25%:0.233 |

### B8 — Reproducible · ✅ PASS

*Is the artifact an identity?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B8.1 | Artifact identity | registry tag + config + data revision recorded and consistent | reporting | ✅ PASS | tag 0.11.1r490760-b64d256e3lr4e-05o4c3b5899 · registry 919a932f1b657064 · data 3543c711bc98 · as_of 2025-01-01 · n 5000 · config 4f368093a2fa737c · checkpoint 33f9b8034c677a8e |
| B8.2 | One-command reproducibility (S6) | the card is regenerated in one invocation at a recorded commit | reporting | ✅ PASS | judge_stages.json:S6 working=True |

## The judge — four legs × two numbers

Progress (`brain alone`) is a strict win. Delivery (`brain + ledger`) is a **non-inferiority** claim vs PATH A (margin 0.01 R2); a delivery tie is not a brain win.

| Leg | Column | Verdict | Evidence |
|---|---|---|---|
| supervised | brain alone (PROGRESS) | ✅ PASS | 7 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| supervised | brain + ledger (DELIVERY, non-inferiority) | ✅ PASS | 7 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| unsupervised | brain alone (PROGRESS) | ✅ PASS | 3 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | ✅ PASS | 3 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| rl | brain alone (PROGRESS) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| rl | brain + ledger (DELIVERY, non-inferiority) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| simulator | brain alone (PROGRESS) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |

## Measured values — every task on the card

The leg verdict above is computed from these rows; a task can PASS while its leg fails on another task. Every number below is read from the receipts, never narrated.

| Leg | Column | Task | Brain | PATH A | Delta | CI | Random | Destroyed | N | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised | brain alone (PROGRESS) | x0 | 0.9212 | 0.8791 | +0.04214 | [6.3e-05, 0.000151] | -0.05078 | -0.05578 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | h0 | 0.9751 | 0.9708 | +0.004349 | [0.00084, 0.002337] | -0.04184 | -0.04877 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | price_sensitivity | 0.01619 | -0.06363 | +0.07982 | [0.002202, 0.0058] | -0.07448 | -0.02841 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | return_proneness | 0.5127 | 0.4879 | +0.0248 | [3.1e-05, 0.000148] | -0.07239 | -0.04397 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | churn_day | 0.1703 | -0.01375 | +0.184 | [2374, 4114] | -0.04031 | -0.05394 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | regime_switches | 0.1185 | -0.008355 | +0.1269 | [1.275, 2.434] | -0.06208 | -0.0434 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | final_regime_days | -0.02392 | -0.07448 | +0.05056 | [60.37, 366.2] | -0.06079 | -0.05325 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | x0 | 0.9214 | 0.8791 | +0.04231 | [6.2e-05, 0.000153] | -0.04539 | -0.06927 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | h0 | 0.9747 | 0.9708 | +0.003963 | [0.000711, 0.002186] | -0.07315 | -0.07184 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | price_sensitivity | 0.02082 | -0.06363 | +0.08445 | [0.002418, 0.006018] | -0.07147 | -0.07544 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | return_proneness | 0.5076 | 0.4879 | +0.01967 | [1.3e-05, 0.000131] | -0.05033 | -0.05474 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | churn_day | 0.1712 | -0.01375 | +0.185 | [2414, 4112] | -0.05777 | -0.06374 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | regime_switches | 0.1139 | -0.008355 | +0.1223 | [1.188, 2.36] | -0.06188 | -0.05743 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | final_regime_days | -0.02774 | -0.07448 | +0.04674 | [51.74, 346.5] | -0.04161 | -0.0834 | 1485 | ✅ PASS |
| unsupervised | brain alone (PROGRESS) | persona_geometry | 0.3976 | 0.3867 | +0.01091 | [0.008423, 0.01313] | 0.000967 | 4e-06 | 1.102e+06 | ✅ PASS |
| unsupervised | brain alone (PROGRESS) | future_geometry | 0.1927 | 0.181 | +0.01166 | [0.01092, 0.01507] | 0.000157 | 1.2e-05 | 1.102e+06 | ✅ PASS |
| unsupervised | brain alone (PROGRESS) | segment_structure | 1 | 0.267 | +0.733 | [0.7042, 0.7585] | -0.000401 | -0.000172 | 1485 | ✅ PASS |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | persona_geometry | 0.3976 | 0.3867 | +0.01091 | [0.008423, 0.01313] | 0.000967 | 4e-06 | 1.102e+06 | ✅ PASS |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | future_geometry | 0.198 | 0.181 | +0.01702 | [0.01633, 0.02037] | 0.000157 | 1.2e-05 | 1.102e+06 | ✅ PASS |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | segment_structure | 1 | 0.267 | +0.733 | [0.7042, 0.7585] | -0.000401 | -0.000172 | 1485 | ✅ PASS |
| rl | brain alone (PROGRESS) | known_effect_uplift | 0.2433 | 0.1794 | +0.06389 | [0.001032, 0.002559] | -0.03474 | -0.025 | 1149 | ✅ PASS |
| rl | brain alone (PROGRESS) | policy_value | 0.2493 | 0.2382 | +0.0111 | [0.005931, 0.0165] | 0.1658 | 0.1717 | 1149 | ✅ PASS |
| rl | brain + ledger (DELIVERY, non-inferiority) | known_effect_uplift | 0.2413 | 0.1794 | +0.06193 | [0.000979, 0.00248] | -0.01938 | -0.03935 | 1149 | ✅ PASS |
| rl | brain + ledger (DELIVERY, non-inferiority) | policy_value | 0.2492 | 0.2382 | +0.01093 | [0.005637, 0.01661] | 0.1755 | 0.1657 | 1149 | ✅ PASS |
| simulator | brain alone (PROGRESS) | dose_response_curve | 0.3179 | 0.3039 | +0.01396 | [0.000117, 0.000503] | -0.06062 | -0.05398 | 5000 | ✅ PASS |
| simulator | brain alone (PROGRESS) | argmax_optimum | 0.33 | 0.2418 | +0.08821 | [0.05791, 0.1185] | 0.2397 | 0.2761 | 1485 | ✅ PASS |
| simulator | brain + ledger (DELIVERY, non-inferiority) | dose_response_curve | 0.3177 | 0.3039 | +0.01376 | [0.000112, 0.000501] | -0.03873 | -0.05701 | 5000 | ✅ PASS |
| simulator | brain + ledger (DELIVERY, non-inferiority) | argmax_optimum | 0.3226 | 0.2418 | +0.08081 | [0.04983, 0.1125] | 0.2532 | 0.2458 | 1485 | ✅ PASS |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ✅ PASS | [renamed_end_to_end] OK: renamed client_id/occurred_at/event/attributes/total -> onboard resolved {'customer_key': 'client_id', 'event_ts':   ·measure_input_law.py on 7a3ae15 |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ✅ PASS | [unseen_value_path] OK: frozen schema (7 event types, vocab sizes [1, 10, 4, 3, 7, 3, 1, 4, 4, 396, 1, 1, 1, 4, 4]) encoded a batch containi  ·measure_input_law.py on 7a3ae15 |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ✅ PASS | [scan_associativity] OK: merge law F(g∘f)=F(g)∘F(f) on 6 real sequences x 5 split points (float64, model's own decay/drive, continuous_time=  ·measure_dynamics_law.py on 7a3ae15 |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | [outlet_schema] OK: 8 views / 40000 rows; declared-not-served [] advertised | [decodability] OK: linear-head decoders beat random+destroyed   ·measure_views_law.py on 7a3ae15 |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ✅ PASS | [ledger_reconstruction] OK: ledger recomputed from the raw stream by DuckDB (not a table copy); LINEAR head on the learned-shape donor (embe  ·measure_battery_law.py on 7a3ae15 |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ✅ PASS | [full_chain_in_one_invocation] OK: 6 producer(s) run by this one command (S1+S2 174.1s, S3 13.2s, S4 68.5s, S5 2.2s, legs 44.2s, prov 2.3s);  ·measure_certification_law.py on 7a3ae15 |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash df73c2aa1427b471 |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- nothing — all rows evaluated

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*This is the only card. B-only rows come from `scripts/b_spec.py`; B+C rows from `scripts/judge_spec.py`; both hang under the tree in `scripts/plan_spec.py` (rendered readably in `PLAN.md`); all read the receipts under `docs/report_cards/experiments/`. There is no separate audit card, and no row may be parked elsewhere (DEC-512).*
