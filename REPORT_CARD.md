# REPORT_CARD — B foundational, then PATH B > PATH A

*Generated `2026-11-05T01:20:40+00:00` · commit `580b62c` (`foundation-rebuild` · dirty) · encoder behavior `f1617f4e2a2990ce`*

**Assessment:**

- ⚪ **Plan root:** OPEN — 19/39 atomic problems green · 0 FAIL · 2 DEC-frozen · 5 v1 unassessed · 19 interior.
- ❌ **B foundational:** NOT CERTIFIED — the current gate. 4/8 sections green · 19 rows PASS · 0 FAIL · 4 UNKNOWN/MISSING.
- ❌ **B+C consequence:** NOT DONE — 0/8 legs green · stages 0/6 · backward open: 0. B is certified; this is the consequence gap (P4), not a missing B row.
- ⚪ **Project v1:** plugin acceptance 9/9 on the bank + demo stream (supervised 6/6, unsupervised 4/4, white_queen 5/5, Layer C 9/9); red_king/red_queen/caterpillar and the identifiable-stream metadata remain unassessed.

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## The plan — the tree this card measures (PLAN.md)

The goal, cut backwards into the problems that must all be solved, recursively to the atomic problems. This section is generated from `scripts/plan_spec.py` — the single definition of the tree; `PLAN.md` is its readable render. The plan defines no gate: every measured atomic problem is a frozen row in its B/judge section below, and an interior verdict is the worst of its children.

**Plan root:** ⚪ OPEN — 19/39 atomic problems green · 0 FAIL · 2 DEC-frozen (no card rows) · 5 v1 not assessed on this card · 19 interior problems.

| Level | Node | What must be true | Atomic problems | Verdict |
|---|---|---|---|---|
| 0 | **G** A certified universal donor, delivered | What is the whole problem? | 39 atomic problems | ❔ UNKNOWN |
| 1 | **P1** The donor exists | Is there a per-stream artifact built from the right algebra, and is it an identity? | 4 atomic problems | ❔ UNKNOWN |
| 2 | **B1** Lawful | Is the state built from the right algebra? | 2 atomic problems | ❔ UNKNOWN |
| 2 | **B8** Reproducible | Is the artifact an identity? | 2 atomic problems | ❔ UNKNOWN |
| 1 | **P2** The donor suffices | Does the embedding lose nothing that matters about the future — for prediction and decision — … | 11 atomic problems | ❔ UNKNOWN |
| 2 | **B2** Sufficient | Does B lose nothing about the future? | 3 atomic problems | ✅ PASS |
| 2 | **E1** Trunk capacity (the state-update algebra) | Can the state carry the stream's structure — not just its mass? | 3 atomic problems | ❔ UNKNOWN |
| 3 | **E1.a** next_event mastered on the toy (curriculum step 1) | Does the frozen trunk beat the 1-gram on the known-truth next_event structure at the dev-rig … | — | ⚪ FRONTIER |
| 3 | **E1.b** intent mastered on the toy (curriculum step 2) | Does the frozen trunk beat the 1-gram on the per-customer intent structure, at a scale where … | — | ⚪ FRONTIER |
| 3 | **E1.c** port: ONE real-data run under DEC-572's bars | Does the winning trunk master the real stream — eff_rank>=30, persona>=0.15, R2>=0.45, stable, … | — | ✅ PASS |
| 2 | **B4** Decisive | Does B support action/counterfactual decisions? | 3 atomic problems | ✅ PASS |
| 2 | **B5** Minimal & affordable | Is it the smallest sufficient object, and fast enough? | 2 atomic problems | ✅ PASS |
| 1 | **P3** The donor is usable | Can simple consumers read it, does it transfer beyond our known tasks, and does it stay honest … | 11 atomic problems | ❔ UNKNOWN |
| 2 | **B3** Accessible | Can simple heads read what is there, at every stage? | 6 atomic problems | ❔ UNKNOWN |
| 2 | **B6** Transferable | Does it add signal on held-out consumers? | 3 atomic problems | ❔ UNKNOWN |
| 2 | **B7** Robust | Does it stay honest under shift and missingness? | 2 atomic problems | ✅ PASS |
| 1 | **P4** The consequence is delivered | Does PATH B (frozen encoder -> simple head) beat PATH A (wide table -> GBM incumbent) for the … | 8 atomic problems | ❔ UNKNOWN |
| 2 | **JUDGE-supervised** supervised leg | forecast targets beat the incumbent GBM on the same splits | 2 atomic problems | ❔ UNKNOWN |
| 2 | **JUDGE-unsupervised** unsupervised leg | distance equals shared futures; structure beats table/random/destroyed | 2 atomic problems | ❔ UNKNOWN |
| 2 | **JUDGE-rl** rl leg | policy value beats table and random on randomized/known-effect data | 2 atomic problems | ❔ UNKNOWN |
| 2 | **JUDGE-simulator** simulator leg | recovers the built-in counterfactual effects (red_king role) | 2 atomic problems | ❔ UNKNOWN |
| 1 | **P5** The stack is v1-complete | Does the full system deliver a certified decision: identifiable stream -> donor -> plugins -> … | 5 atomic problems | ❔ UNKNOWN |
| 2 | **V1.1** Causally identifiable stream | Does rabbit_hole emit randomized arms with a known effect, recorded as such? | — | ⚪ NOT ASSESSED |
| 2 | **V1.2** Each plugin passes its independent gate | Do the supervised, unsupervised and white_queen consumers each pass their own gate? | — | ⚪ NOT ASSESSED |
| 2 | **V1.3** red_king recovers known effects | Does red_king recover the known counterfactual effect and improve the certified decision with … | — | ⚪ NOT ASSESSED |
| 2 | **V1.4** red_queen composes actions | Does red_queen compose next-best actions across cadences under constraints? | — | ⚪ NOT ASSESSED |
| 2 | **V1.5** caterpillar explains from artifacts | Does caterpillar answer 'why' from artifacts only? | — | ⚪ NOT ASSESSED |

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

### B1 — Lawful · ❔ UNKNOWN

*Is the state built from the right algebra?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B1.1 | Lawful dynamics (S3) | merge/equivariance/mass-additivity exact on data; tau receipted; stable from step 0 | reporting | ❔ UNKNOWN | stale: measured by measure_dynamics_law.py against behavior 7baf744fc78b (current f1617f4e2a29) — re-run that producer |
| B1.2 | Construction laws (C1-C6) | scan / field-permutation / mass / determinism / schema variant / frozen identity | quality | ✅ PASS | C1-C6 all met: field-perm 1.714e-07 · scan 1.105e-15 · mass 4.819e-16 · determinism bit-identical · schema variant ok · identity present (N=20, cpu) |

### B2 — Sufficient · ✅ PASS

*Does B lose nothing about the future?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B2.1 | Worst-case sufficiency U | U = 1 - worst-case oracle gap vs full history, fixed capacity, broad published family | quality | ✅ PASS | served U 0.9802 (bar 0.5, DEC-570) · worst probe next-type · 1/5 probes included · excluded: attr-next, dt, horizon-count-value, numeric-next |
| B2.2 | Full-history oracle gap | excess loss of E vs the full-history arm at fixed capacity, per included probe | quality | ✅ PASS | every included probe clears the bar · next-type 0.9802 |
| B2.3 | Learned-only attribution | sufficiency of the learned shape vs the served statistics, reported separately | reporting | ✅ PASS | learned-only U 1.0764 · worst probe next-type · learned shape carries (the ledger does not add) |

### B3 — Accessible · ❔ UNKNOWN

*Can simple heads read what is there, at every stage?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B3.1 | Outlet accessibility (S4) | per-bucket accessibility parity + per-bucket negative controls | reporting | ❔ UNKNOWN | stale: measured by measure_views_law.py against behavior 7baf744fc78b (current f1617f4e2a29) — re-run that producer |
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
| B5.2 | Cost / runtime / constant-state | runtime, memory and incremental-update receipt | reporting | ✅ PASS | 5000/5000 customers · cold 25.93s / warm 101.93s · 44.1 rows/s · dims 256/256 · peak RSS 3386.4 MB (slice 32.0 GB) · incremental (absorbed 0, idle 5000, wall 3.45s) |

### B6 — Transferable · ❔ UNKNOWN

*Does it add signal on held-out consumers?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B6.1 | Objective battery (S5) | every content pressure beats its naive comparator (or is excluded and reported) | reporting | ❔ UNKNOWN | stale: measured by measure_battery_law.py against behavior 7baf744fc78b (current f1617f4e2a29) — re-run that producer |
| B6.2 | Donor does-not-lose / adds / controls | E vs raw, fair table, table+E; random/destroyed negative controls | quality | ✅ PASS | n_A 3515 / n_B 1485 · y_churn_90: E 0.7388 vs raw 0.6800 / fair 0.6800 / table+E 0.7297 (CI lo +0.0037) / hstack 0.7534 (CI lo 0.007398) / rand 0.4076 / dest 0.5046 · y_orders_90: E 0.6216 vs raw 0.6047 / fair 0.6047 / table+E 0.6172 (CI lo +0.0052) / hstack 0.5943 (CI lo -0.04206) / rand -0.0163 / dest -0.0192 · y_value_90: E 0.6096 vs raw 0.5952 / fair 0.5952 / table+E 0.6054 (CI lo +0.0008) / hstack 0.5791 (CI lo -0.04908) / rand -0.0259 / dest -0.0119 |
| B6.3 | Broad held-out transfer family | adds signal on a family broader than the eight judge legs | quality | ✅ PASS | published family 15 targets · E wins 13/15 (0.867) · family mean E 0.4331 vs table 0.3857 (diff +0.0475) · random 0.0098 / destroyed 0.0043 |

### B7 — Robust · ✅ PASS

*Does it stay honest under shift and missingness?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B7.1 | ECE / OOD retention | calibration of the shipped next-type head + OOD retention | quality | ✅ PASS | next-type ECE 0.0302 (bar 0.1) · acc 0.35824915824915826 · n 1485/1485 (OOV 0) · OOD (as_of+30d): ece nan acc nan n 0 |
| B7.2 | Missingness / unlogged confounders | uncertainty carried for unobserved events, never faked | reporting | ✅ PASS | next-time 0.8 coverage 0.800 (n 1485, MAE 1.15d) · occurrence AUC 0.6339 / Brier 0.2794 (h 1.338484d) · response coverage 0.8/0.8 · missingness mean rel L2 10%:0.094, 25%:0.233 |

### B8 — Reproducible · ❔ UNKNOWN

*Is the artifact an identity?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B8.1 | Artifact identity | registry tag + config + data revision recorded and consistent | reporting | ✅ PASS | tag 0.11.1r490760-b64d256e3lr4e-05o4c3b5899 · registry 919a932f1b657064 · data 3543c711bc98 · as_of 2025-01-01 · n 5000 · config 4f368093a2fa737c · checkpoint 33f9b8034c677a8e |
| B8.2 | One-command reproducibility (S6) | the card is regenerated in one invocation at a recorded commit | reporting | ❔ UNKNOWN | stale: measured by measure_certification_law.py against behavior 7baf744fc78b (current f1617f4e2a29) — re-run that producer |

## The judge — four legs × two numbers

Progress (`brain alone`) is a strict win. Delivery (`brain + ledger`) is a **non-inferiority** claim vs PATH A (margin 0.01 R2); a delivery tie is not a brain win.

| Leg | Column | Verdict | Evidence |
|---|---|---|---|
| supervised | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| supervised | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| unsupervised | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| rl | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| rl | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| simulator | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |

## Measured values — every task on the card

The leg verdict above is computed from these rows; a task can PASS while its leg fails on another task. Every number below is read from the receipts, never narrated.

| Leg | Column | Task | Brain | PATH A | Delta | CI | Random | Destroyed | N | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| — | — | no task rows (receipts missing or stale) | — | — | — | — | — | — | — | ❔ UNKNOWN |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ❔ UNKNOWN | stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce) |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash f1617f4e2a2990ce |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_leg_supervised.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_leg_unsupervised.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_leg_unsupervised.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_stages.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_stages.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_stages.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_stages.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_stages.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `judge_stages.json — stale: encoder behavior changed since receipt (7baf744fc78ba3b3 != f1617f4e2a2990ce)`
- `B B1.1 [P1·B1] Lawful dynamics (S3) — stale: measured by measure_dynamics_law.py against behavior 7baf744fc78b (current f1617f4e2a29) — re-run that producer`
- `B B3.1 [P3·B3] Outlet accessibility (S4) — stale: measured by measure_views_law.py against behavior 7baf744fc78b (current f1617f4e2a29) — re-run that producer`
- `B B6.1 [P3·B6] Objective battery (S5) — stale: measured by measure_battery_law.py against behavior 7baf744fc78b (current f1617f4e2a29) — re-run that producer`
- `B B8.2 [P1·B8] One-command reproducibility (S6) — stale: measured by measure_certification_law.py against behavior 7baf744fc78b (current f1617f4e2a29) — re-run that producer`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*This is the only card. B-only rows come from `scripts/b_spec.py`; B+C rows from `scripts/judge_spec.py`; both hang under the tree in `scripts/plan_spec.py` (rendered readably in `PLAN.md`); all read the receipts under `docs/report_cards/experiments/`. There is no separate audit card, and no row may be parked elsewhere (DEC-512).*
