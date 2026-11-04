# REPORT_CARD — B foundational, then PATH B > PATH A

*Generated `2026-11-04T22:07:43+00:00` · commit `13b7003` (`foundation-rebuild` · dirty) · encoder behavior `7baf744fc78ba3b3`*

**Assessment:**

- ❌ **Plan root:** FAILED — 27/39 atomic problems green · 4 FAIL · 3 DEC-frozen · 5 v1 unassessed · 19 interior · reds: P4 (4).
- ✅ **B foundational:** CERTIFIED — 8/8 sections green.
- ❌ **B+C consequence:** NOT DONE — 4/8 legs green · stages 6/6 · backward open: 0. Reported for diagnosis; downstream of B.
- ⚪ **Project v1:** not assessed on this card (full stack is a separate scope).

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## The plan — the tree this card measures (PLAN.md)

The goal, cut backwards into the problems that must all be solved, recursively to the atomic problems. This section is generated from `scripts/plan_spec.py` — the single definition of the tree; `PLAN.md` is its readable render. The plan defines no gate: every measured atomic problem is a frozen row in its B/judge section below, and an interior verdict is the worst of its children.

**Plan root:** ❌ FAILED — 27/39 atomic problems green · 4 FAIL · 3 DEC-frozen (no card rows) · 5 v1 not assessed on this card · 19 interior problems.

| Level | Node | What must be true | Atomic problems | Verdict |
|---|---|---|---|---|
| 0 | **G** A certified universal donor, delivered | What is the whole problem? | 39 atomic problems | ❌ FAIL |
| 1 | **P1** The donor exists | Is there a per-stream artifact built from the right algebra, and is it an identity? | 4 atomic problems | ✅ PASS |
| 2 | **B1** Lawful | Is the state built from the right algebra? | 2 atomic problems | ✅ PASS |
| 2 | **B8** Reproducible | Is the artifact an identity? | 2 atomic problems | ✅ PASS |
| 1 | **P2** The donor suffices | Does the embedding lose nothing that matters about the future — for prediction and decision — … | 11 atomic problems | ❔ UNKNOWN |
| 2 | **B2** Sufficient | Does B lose nothing about the future? | 3 atomic problems | ✅ PASS |
| 2 | **E1** Trunk capacity (the state-update algebra) | Can the state carry the stream's structure — not just its mass? | 3 atomic problems | ⚪ FRONTIER |
| 3 | **E1.a** next_event mastered on the toy (curriculum step 1) | Does the frozen trunk beat the 1-gram on the known-truth next_event structure at the dev-rig … | — | ⚪ FRONTIER |
| 3 | **E1.b** intent mastered on the toy (curriculum step 2) | Does the frozen trunk beat the 1-gram on the per-customer intent structure, at a scale where … | — | ⚪ FRONTIER |
| 3 | **E1.c** port: ONE real-data run under DEC-572's bars | Does the winning trunk master the real stream — eff_rank>=30, persona>=0.15, R2>=0.45, stable, … | — | ⚪ FRONTIER |
| 2 | **B4** Decisive | Does B support action/counterfactual decisions? | 3 atomic problems | ✅ PASS |
| 2 | **B5** Minimal & affordable | Is it the smallest sufficient object, and fast enough? | 2 atomic problems | ✅ PASS |
| 1 | **P3** The donor is usable | Can simple consumers read it, does it transfer beyond our known tasks, and does it stay honest … | 11 atomic problems | ✅ PASS |
| 2 | **B3** Accessible | Can simple heads read what is there, at every stage? | 6 atomic problems | ✅ PASS |
| 2 | **B6** Transferable | Does it add signal on held-out consumers? | 3 atomic problems | ✅ PASS |
| 2 | **B7** Robust | Does it stay honest under shift and missingness? | 2 atomic problems | ✅ PASS |
| 1 | **P4** The consequence is delivered | Does PATH B (frozen encoder -> simple head) beat PATH A (wide table -> GBM incumbent) for the … | 8 atomic problems | ❌ FAIL |
| 2 | **JUDGE-supervised** supervised leg | forecast targets beat the incumbent GBM on the same splits | 2 atomic problems | ❌ FAIL |
| 2 | **JUDGE-unsupervised** unsupervised leg | distance equals shared futures; structure beats table/random/destroyed | 2 atomic problems | ❌ FAIL |
| 2 | **JUDGE-rl** rl leg | policy value beats table and random on randomized/known-effect data | 2 atomic problems | ✅ PASS |
| 2 | **JUDGE-simulator** simulator leg | recovers the built-in counterfactual effects (red_king role) | 2 atomic problems | ✅ PASS |
| 1 | **P5** The stack is v1-complete | Does the full system deliver a certified decision: identifiable stream -> donor -> plugins -> … | 5 atomic problems | ❔ UNKNOWN |
| 2 | **V1.1** Causally identifiable stream | Does rabbit_hole emit randomized arms with a known effect, recorded as such? | — | ⚪ NOT ASSESSED |
| 2 | **V1.2** Each plugin passes its independent gate | Do the supervised, unsupervised and white_queen consumers each pass their own gate? | — | ⚪ NOT ASSESSED |
| 2 | **V1.3** red_king recovers known effects | Does red_king recover the known counterfactual effect and improve the certified decision with … | — | ⚪ NOT ASSESSED |
| 2 | **V1.4** red_queen composes actions | Does red_queen compose next-best actions across cadences under constraints? | — | ⚪ NOT ASSESSED |
| 2 | **V1.5** caterpillar explains from artifacts | Does caterpillar answer 'why' from artifacts only? | — | ⚪ NOT ASSESSED |

*Frontier:* The live frontier is P2's enabler E1 (the trunk's state-update algebra), now cut to three DEC-frozen atomic problems. E1.a (next_event mastered on the toy) is exhausted negative at N=500: all three single-axis trunk-algebra candidates are negative on the toy (DEC-577, DEC-578 and DEC-581 outcomes) and the capacity axis is negative too (DEC-582 outcome, g 1.149121 ≈ iteration 1's 1.14727, no usable rank gained) — the 1-gram floor is binding, a finite-sample estimation gap, not a trunk deficiency. E1.b (intent, curriculum step 2) is the next structure (DEC-583; the snr difficulty dial is rejected as an axis); its full one-axis pre-registration is DEC-586 (fixture point, frozen bars in corrected units, branches, the 4-file behavior move + cfm_v2k re-stamp), drafted 2026-11-04 and awaiting operator approval; E1.c (the port under DEC-572's bars) waits on both. No lab receipt flips a B-row verdict. The PRIMARY progress track meanwhile is R4: the 11 remaining B atomic problems, certification-first per the operator's 2026-11-03 directive. DEC-585 pre-registered the 8 rows it covers (B1.2, B4.1-B4.3, B5.2, B6.2, B7.1, B8.1) as 5 cfm_v2k producers with frozen gate kinds and bars in b_spec (chunk 1 = gates + producers + card branches, no measurement; chunk 2 = the sequential measurement, receipts verified before verdicts); the remaining 3 (B5.1, B6.3, B7.2) each get their own DEC.

**Frontier readings (reported, not gated — the evidence behind E1's FRONTIER badge):**

- **dev rig · next_event iter1** — g 1.147 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.153 · beats 1-gram: no — `lab/next_event_iter1.json` · measured under `6a8e2663`
- **dev rig · next_event iter2** — g 1.154 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.473 · beats 1-gram: no — `lab/next_event_iter2.json` · measured under `4053d2f9`
- **dev rig · next_event iter3 (DEC-578, candidate 2)** — g 1.182 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.248 · beats 1-gram: no — `lab/next_event_iter3.json` · measured under `905b4f88`
- **dev rig · next_event iter4 (DEC-581, candidate 3)** — g 1.17 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.836 · beats 1-gram: no — `lab/next_event_iter4.json` · measured under `d5df3a2c`
- **dev rig · next_event cap256 (DEC-582, capacity @ baseline algebra)** — g 1.149 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.209 · beats 1-gram: no — `lab/next_event_cap256.json` · measured under `d5df3a2c`
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
| B3.2 | Stage decomposition M0->M1->served | the fine-structure metric at each stage, so a loss is attributed to a stage | reporting | ✅ PASS | complete · persona M0 0.1965 → M1 0.05979 → served 0.09277 · future M0 0.1755 → M1 0.09209 → served 0.1108 · largest drop M0_exact_counter->M1_fold_only · M3/M4 0.2043 |
| B3.3 | Linear accessibility (current) | mean ridge R2 on A->B over the lab's persona+future fields; served vs PATH A | quality | ✅ PASS | served 0.4865 vs PATH A 0.4126 (delta +0.0739, margin -0.01; n_fields 11) |
| B3.4 | Block-aware serving geometry | exact counter block beside the learned shape, equal total variance; persona/future/ARI | reporting | ✅ PASS | complete · persona served 0.09277 / counter 0.1965 / block 0.1166 / PATH A 0.3867 · future served 0.1108 / counter 0.1755 / block 0.1299 / PATH A 0.181 · counter-weight sensitivity flat (block scale erased by per-dim z-score; dimension count is the lever) |
| B3.5 | Equal-dimension serving geometry | shape PCA + counter tile at 128/128 and 39/39; persona/future/ARI at full size | reporting | ✅ PASS | complete · persona equal_39 0.2213 / equal_128 0.1715 / counter 0.1965 / shape_pca_39 0.1615 / PATH A 0.3867 · future equal_39 0.1618 / equal_128 0.1377 / counter 0.1755 / PATH A 0.181 |
| B3.6 | Served-object X-ray (ledger exposure) | the ledger alone, its named parts, and ledger block-PCA under the frozen B3.2/B3.4 consumers, against {exact counter, served shape, PATH A}; the branch is computed here from the arm numbers (DEC-569) | reporting | ✅ PASS | complete · branch exposure (DEC-569: best ledger arm persona 0.3196 from part_numeric_stats; full 0.01722 / pca_39 0.2012 / pca_128 0.1036) · references persona counter 0.1965 / shape 0.09277 / PATH A 0.3867 · linear access ledger_full 0.3551 / counter 0.4058 / shape 0.4865 / PATH A 0.4126 |

### B4 — Decisive · ✅ PASS

*Does B support action/counterfactual decisions?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B4.1 | B-only dose-response / argmax | action-conditional response shape readable from the state (not through a judge leg) | quality | ✅ PASS | E share 0.2983 vs table 0.2438 (CI [+0.0249, +0.0855]) · random 0.2586 · destroyed 0.2384 · n=1485 · oracle ceiling 0.6404 |
| B4.2 | OPE / IPS known-effect recovery | E recovers the known effect with a paired CI excluding 0 | quality | ✅ PASS | E policy IPS 0.1598 vs table 0.1517 (diff +0.01 [+0.00, 0.0146] > 0) · accE 0.295 vs accT 0.247 · random IPS None · n_test=74746 · levels {'0': 0.2515, '1': 0.2482, '2': 0.252, '3': 0.2482} |
| B4.3 | Action-invariance | representation invariant to assignment, still predictive of the outcome | reporting | ✅ PASS | band cert ok (P1==P2 bit-identical; P1 vs stored S 2.384185791015625e-07 / E 0.0009585114639776293 vs bands 6.199199447110004e-07 / 0.015569581650197506; teeth True) · marker null: share 0.30033670033670035 vs stored 0.2983164983164983 (ok=True) · level permutation: share 0.24511784511784512 vs table 0.24377104377104378 (|Δ|∞ 313.1736521923858, ok=True) |

### B5 — Minimal & affordable · ✅ PASS

*Is it the smallest sufficient object, and fast enough?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B5.1 | Minimality k_min | served rank = the frozen reconstruction certificate | quality | ✅ PASS | served shape rank 32 == k_min 32 (shape full 32) · explicit block 27 · joint floor 48 (unconstrained) · n_train 451 / n_test 439 · noise 0.04773 |
| B5.2 | Cost / runtime / constant-state | runtime, memory and incremental-update receipt | reporting | ✅ PASS | 5000/5000 customers · cold 19.67s / warm 150.63s · 29.9 rows/s · dims 256/256 · peak RSS 3384.8 MB (slice 32.0 GB) · incremental (absorbed 0, idle 5000, wall 3.45s) |

### B6 — Transferable · ✅ PASS

*Does it add signal on held-out consumers?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B6.1 | Objective battery (S5) | every content pressure beats its naive comparator (or is excluded and reported) | reporting | ✅ PASS | judge_stages.json:S5 working=True |
| B6.2 | Donor does-not-lose / adds / controls | E vs raw, fair table, table+E; random/destroyed negative controls | quality | ✅ PASS | n_A 3515 / n_B 1485 · y_churn_90: E 0.7388 vs raw 0.6800 / fair 0.6800 / table+E 0.7297 (CI lo +0.0037) / hstack 0.7534 (CI lo 0.007398) / rand 0.4076 / dest 0.5046 · y_orders_90: E 0.6216 vs raw 0.6047 / fair 0.6047 / table+E 0.6172 (CI lo +0.0052) / hstack 0.5943 (CI lo -0.04206) / rand -0.0163 / dest -0.0192 · y_value_90: E 0.6096 vs raw 0.5952 / fair 0.5952 / table+E 0.6054 (CI lo +0.0008) / hstack 0.5791 (CI lo -0.04908) / rand -0.0259 / dest -0.0119 |
| B6.3 | Broad held-out transfer family | adds signal on a family broader than the eight judge legs | quality | ✅ PASS | published family 15 targets · E wins 14/15 (0.933) · family mean E 0.4252 vs table 0.3857 (diff +0.0395) · random 0.0174 / destroyed 0.0257 |

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
| B8.1 | Artifact identity | registry tag + config + data revision recorded and consistent | reporting | ✅ PASS | tag 0.11.1r490760-b64d256e3lr4e-05o4c3b5899 · registry 676e52359995983a · data 3543c711bc98 · as_of 2025-01-01 · n 5000 · config 4f368093a2fa737c · checkpoint 33f9b8034c677a8e |
| B8.2 | One-command reproducibility (S6) | the card is regenerated in one invocation at a recorded commit | reporting | ✅ PASS | judge_stages.json:S6 working=True |

## The judge — four legs × two numbers

Progress (`brain alone`) is a strict win. Delivery (`brain + ledger`) is a **non-inferiority** claim vs PATH A (margin 0.01 R2); a delivery tie is not a brain win.

| Leg | Column | Verdict | Evidence |
|---|---|---|---|
| supervised | brain alone (PROGRESS) | ❌ FAIL | return_proneness: does not beat PATH A with CI excluding 0; failing: return_proneness |
| supervised | brain + ledger (DELIVERY, non-inferiority) | ❌ FAIL | price_sensitivity: outside the delivery non-inferiority margin (0.01 R2); failing: price_sensitivity, return_proneness |
| unsupervised | brain alone (PROGRESS) | ❌ FAIL | persona_geometry: does not beat PATH A with CI excluding 0; failing: persona_geometry, future_geometry, segment_structure |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | ❌ FAIL | persona_geometry: outside the delivery non-inferiority margin (0.01 R2); failing: persona_geometry, future_geometry, segment_structure |
| rl | brain alone (PROGRESS) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| rl | brain + ledger (DELIVERY, non-inferiority) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| simulator | brain alone (PROGRESS) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ✅ PASS | 2 task(s) beat PATH A; paired CI excludes 0; controls below brain |

## Measured values — every task on the card

The leg verdict above is computed from these rows; a task can PASS while its leg fails on another task. Every number below is read from the receipts, never narrated.

| Leg | Column | Task | Brain | PATH A | Delta | CI | Random | Destroyed | N | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| supervised | brain alone (PROGRESS) | x0 | 0.9472 | 0.8791 | +0.06815 | [0.00012, 0.000218] | -1454 | -1447 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | h0 | 0.9739 | 0.9708 | +0.003079 | [0.00042, 0.001837] | -9.368 | -8.98 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | price_sensitivity | 0.00777 | -0.06363 | +0.0714 | [0.002151, 0.004996] | -0.009219 | -0.01381 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | return_proneness | 0.5027 | 0.4879 | +0.01481 | [-2.6e-05, 0.000124] | -0.3323 | -0.3592 | 1485 | ❌ FAIL |
| supervised | brain alone (PROGRESS) | churn_day | 0.3884 | -0.01375 | +0.4021 | [6182, 8157] | -3326 | -3410 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | regime_switches | 0.1425 | -0.008355 | +0.1509 | [1.521, 2.889] | -356.5 | -340.7 | 1485 | ✅ PASS |
| supervised | brain alone (PROGRESS) | final_regime_days | 0.001817 | -0.07448 | +0.0763 | [189.8, 447.5] | -0.006253 | -0.006 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | x0 | 0.9395 | 0.8791 | +0.06044 | [0.000105, 0.000191] | -591.2 | -602.8 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | h0 | 0.9713 | 0.9708 | +0.000538 | [-0.000605, 0.001161] | -0.1482 | -0.137 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | price_sensitivity | -0.07738 | -0.06363 | -0.01375 | [-0.002753, 0.001359] | -0.1117 | -0.1169 | 1485 | ❌ FAIL |
| supervised | brain + ledger (DELIVERY, non-inferiority) | return_proneness | 0.4712 | 0.4879 | -0.01671 | [-0.000138, 1.2e-05] | -0.08702 | -0.05232 | 1485 | ❌ FAIL |
| supervised | brain + ledger (DELIVERY, non-inferiority) | churn_day | 0.2331 | -0.01375 | +0.2468 | [2618, 5985] | -4325 | -4376 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | regime_switches | 0.01576 | -0.008355 | +0.02412 | [-0.4998, 1.187] | -198.3 | -201.2 | 1485 | ✅ PASS |
| supervised | brain + ledger (DELIVERY, non-inferiority) | final_regime_days | -0.07218 | -0.07448 | +0.002305 | [-162.7, 186.9] | -0.12 | -0.09144 | 1485 | ✅ PASS |
| unsupervised | brain alone (PROGRESS) | persona_geometry | 0.09277 | 0.3867 | -0.2939 | [-0.3064, -0.2889] | 2e-06 | 0.000921 | 1.102e+06 | ❌ FAIL |
| unsupervised | brain alone (PROGRESS) | future_geometry | 0.1108 | 0.181 | -0.07018 | [-0.07993, -0.06884] | 1.8e-05 | 0.000205 | 1.102e+06 | ❌ FAIL |
| unsupervised | brain alone (PROGRESS) | segment_structure | -0.001128 | 0.2685 | -0.2696 | [-0.2965, -0.2429] | 0.000288 | -0.000551 | 1485 | ❌ FAIL |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | persona_geometry | 0.06983 | 0.3867 | -0.3168 | [-0.3229, -0.3094] | 0.000249 | 0.00021 | 1.102e+06 | ❌ FAIL |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | future_geometry | 0.09223 | 0.181 | -0.08878 | [-0.09156, -0.08321] | 8.6e-05 | 5e-06 | 1.102e+06 | ❌ FAIL |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | segment_structure | -0.000401 | 0.2685 | -0.2689 | [-0.2949, -0.2425] | -0.000336 | -0.000859 | 1485 | ❌ FAIL |
| rl | brain alone (PROGRESS) | known_effect_uplift | 0.2401 | 0.1794 | +0.06071 | [0.000981, 0.002442] | -0.007093 | -0.01283 | 1149 | ✅ PASS |
| rl | brain alone (PROGRESS) | policy_value | 0.2485 | 0.2382 | +0.01024 | [0.005097, 0.01547] | 0.1758 | 0.1762 | 1149 | ✅ PASS |
| rl | brain + ledger (DELIVERY, non-inferiority) | known_effect_uplift | 0.1751 | 0.1794 | -0.004292 | [-0.001116, 0.000945] | -0.06428 | -0.05141 | 1149 | ✅ PASS |
| rl | brain + ledger (DELIVERY, non-inferiority) | policy_value | 0.24 | 0.2382 | +0.001713 | [-0.004757, 0.008866] | 0.1749 | 0.1855 | 1149 | ✅ PASS |
| simulator | brain alone (PROGRESS) | dose_response_curve | 0.3159 | 0.3039 | +0.01197 | [8.2e-05, 0.000452] | -0.1724 | -0.1893 | 5000 | ✅ PASS |
| simulator | brain alone (PROGRESS) | argmax_optimum | 0.2983 | 0.2418 | +0.05657 | [0.02559, 0.08756] | 0.2586 | 0.2384 | 1485 | ✅ PASS |
| simulator | brain + ledger (DELIVERY, non-inferiority) | dose_response_curve | 0.3017 | 0.3039 | -0.002229 | [-0.000259, 0.000165] | -0.04182 | -0.02497 | 5000 | ✅ PASS |
| simulator | brain + ledger (DELIVERY, non-inferiority) | argmax_optimum | 0.301 | 0.2418 | +0.05926 | [0.02694, 0.08956] | 0.2411 | 0.2391 | 1485 | ✅ PASS |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ✅ PASS | [renamed_end_to_end] OK: renamed client_id/occurred_at/event/attributes/total -> onboard resolved {'customer_key': 'client_id', 'event_ts':   ·measure_input_law.py on 13b7003 |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ✅ PASS | [unseen_value_path] OK: frozen schema (7 event types, vocab sizes [1, 10, 4, 3, 7, 3, 1, 4, 4, 396, 1, 1, 1, 4, 4]) encoded a batch containi  ·measure_input_law.py on 13b7003 |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ✅ PASS | [scan_associativity] OK: merge law F(g∘f)=F(g)∘F(f) on 6 real sequences x 5 split points (float64, model's own decay/drive, continuous_time=  ·measure_dynamics_law.py on 13b7003 |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ✅ PASS | [outlet_schema] OK: 7 views / 35000 rows; declared-not-served [] advertised | [decodability] OK: linear-head decoders beat random+destroyed   ·measure_views_law.py on 13b7003 |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ✅ PASS | [ledger_reconstruction] OK: ledger recomputed from the raw stream by DuckDB (not a table copy); LINEAR head on the learned-shape donor (embe  ·measure_battery_law.py on 13b7003 |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ✅ PASS | [full_chain_in_one_invocation] OK: 6 producer(s) run by this one command (S1+S2 156.6s, S3 12.6s, S4 58.0s, S5 2.3s, legs 70.1s, prov 2.2s);  ·measure_certification_law.py on 13b7003 |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash 7baf744fc78ba3b3 |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json :: brain [P4·JUDGE-supervised] — return_proneness: does not beat PATH A with CI excluding 0; failing: return_proneness`
- `judge_leg_supervised.json :: brain_ledger [P4·JUDGE-supervised] — price_sensitivity: outside the delivery non-inferiority margin (0.01 R2); failing: price_sensitivity, return_proneness`
- `judge_leg_unsupervised.json :: brain [P4·JUDGE-unsupervised] — persona_geometry: does not beat PATH A with CI excluding 0; failing: persona_geometry, future_geometry, segment_structure`
- `judge_leg_unsupervised.json :: brain_ledger [P4·JUDGE-unsupervised] — persona_geometry: outside the delivery non-inferiority margin (0.01 R2); failing: persona_geometry, future_geometry, segment_structure`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*This is the only card. B-only rows come from `scripts/b_spec.py`; B+C rows from `scripts/judge_spec.py`; both hang under the tree in `scripts/plan_spec.py` (rendered readably in `PLAN.md`); all read the receipts under `docs/report_cards/experiments/`. There is no separate audit card, and no row may be parked elsewhere (DEC-512).*
