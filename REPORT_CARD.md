# REPORT_CARD — B foundational, then PATH B > PATH A

*Generated `2026-11-04T18:42:23+00:00` · commit `0735fe2` (`foundation-rebuild` · clean) · encoder behavior `c9f26925b418b284`*

**Assessment:**

- ⚪ **Plan root:** OPEN — 0/39 atomic problems green · 0 FAIL · 3 DEC-frozen · 5 v1 unassessed · 19 interior.
- ❌ **B foundational:** NOT CERTIFIED — the current gate. 0/8 sections green · 0 rows PASS · 0 FAIL · 23 UNKNOWN/MISSING.
- ❌ **B+C consequence:** NOT DONE — 0/8 legs green · stages 0/6 · backward open: 0. Reported for diagnosis; downstream of B.
- ⚪ **Project v1:** not assessed on this card (full stack is a separate scope).

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## The plan — the tree this card measures (PLAN.md)

The goal, cut backwards into the problems that must all be solved, recursively to the atomic problems. This section is generated from `scripts/plan_spec.py` — the single definition of the tree; `PLAN.md` is its readable render. The plan defines no gate: every measured atomic problem is a frozen row in its B/judge section below, and an interior verdict is the worst of its children.

**Plan root:** ⚪ OPEN — 0/39 atomic problems green · 0 FAIL · 3 DEC-frozen (no card rows) · 5 v1 not assessed on this card · 19 interior problems.

| Level | Node | What must be true | Atomic problems | Verdict |
|---|---|---|---|---|
| 0 | **G** A certified universal donor, delivered | What is the whole problem? | 39 atomic problems | ❔ UNKNOWN |
| 1 | **P1** The donor exists | Is there a per-stream artifact built from the right algebra, and is it an identity? | 4 atomic problems | ❔ UNKNOWN |
| 2 | **B1** Lawful | Is the state built from the right algebra? | 2 atomic problems | ❔ UNKNOWN |
| 2 | **B8** Reproducible | Is the artifact an identity? | 2 atomic problems | ❔ UNKNOWN |
| 1 | **P2** The donor suffices | Does the embedding lose nothing that matters about the future — for prediction and decision — … | 11 atomic problems | ❔ UNKNOWN |
| 2 | **B2** Sufficient | Does B lose nothing about the future? | 3 atomic problems | ❔ UNKNOWN |
| 2 | **E1** Trunk capacity (the state-update algebra) | Can the state carry the stream's structure — not just its mass? | 3 atomic problems | ⚪ FRONTIER |
| 3 | **E1.a** next_event mastered on the toy (curriculum step 1) | Does the frozen trunk beat the 1-gram on the known-truth next_event structure at the dev-rig … | — | ⚪ FRONTIER |
| 3 | **E1.b** intent mastered on the toy (curriculum step 2) | Does the frozen trunk beat the 1-gram on the per-customer intent structure, at a scale where … | — | ⚪ FRONTIER |
| 3 | **E1.c** port: ONE real-data run under DEC-572's bars | Does the winning trunk master the real stream — eff_rank>=30, persona>=0.15, R2>=0.45, stable, … | — | ⚪ FRONTIER |
| 2 | **B4** Decisive | Does B support action/counterfactual decisions? | 3 atomic problems | ❔ UNKNOWN |
| 2 | **B5** Minimal & affordable | Is it the smallest sufficient object, and fast enough? | 2 atomic problems | ❔ UNKNOWN |
| 1 | **P3** The donor is usable | Can simple consumers read it, does it transfer beyond our known tasks, and does it stay honest … | 11 atomic problems | ❔ UNKNOWN |
| 2 | **B3** Accessible | Can simple heads read what is there, at every stage? | 6 atomic problems | ❔ UNKNOWN |
| 2 | **B6** Transferable | Does it add signal on held-out consumers? | 3 atomic problems | ❔ UNKNOWN |
| 2 | **B7** Robust | Does it stay honest under shift and missingness? | 2 atomic problems | ❔ UNKNOWN |
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

*Frontier:* The live frontier is P2's enabler E1 (the trunk's state-update algebra), now cut to three DEC-frozen atomic problems. E1.a (next_event mastered on the toy) is exhausted negative at N=500: all three single-axis trunk-algebra candidates are negative on the toy (DEC-577, DEC-578 and DEC-581 outcomes) and the capacity axis is negative too (DEC-582 outcome, g 1.149121 ≈ iteration 1's 1.14727, no usable rank gained) — the 1-gram floor is binding, a finite-sample estimation gap, not a trunk deficiency. E1.b (intent, curriculum step 2) is the next structure (DEC-583; the snr difficulty dial is rejected as an axis); its full one-axis pre-registration is DEC-586 (fixture point, frozen bars in corrected units, branches, the 4-file behavior move + cfm_v2k re-stamp), drafted 2026-11-04 and awaiting operator approval; E1.c (the port under DEC-572's bars) waits on both. No lab receipt flips a B-row verdict. The PRIMARY progress track meanwhile is R4: the 11 remaining B atomic problems, certification-first per the operator's 2026-11-03 directive. DEC-585 pre-registered the 8 rows it covers (B1.2, B4.1-B4.3, B5.2, B6.2, B7.1, B8.1) as 5 cfm_v2k producers with frozen gate kinds and bars in b_spec (chunk 1 = gates + producers + card branches, no measurement; chunk 2 = the sequential measurement, receipts verified before verdicts); the remaining 3 (B5.1, B6.3, B7.2) each get their own DEC.

**Frontier readings (reported, not gated — the evidence behind E1's FRONTIER badge):**

- **dev rig · next_event iter1** — g 1.147 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.153 · beats 1-gram: no — `lab/next_event_iter1.json` · measured under `6a8e2663`
- **dev rig · next_event iter2** — g 1.154 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.473 · beats 1-gram: no — `lab/next_event_iter2.json` · measured under `4053d2f9`
- **dev rig · next_event iter3 (DEC-578, candidate 2)** — g 1.182 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.248 · beats 1-gram: no — `lab/next_event_iter3.json` · measured under `905b4f88`
- **dev rig · next_event iter4 (DEC-581, candidate 3)** — g 1.17 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.836 · beats 1-gram: no — `lab/next_event_iter4.json` · measured under `d5df3a2c`
- **dev rig · next_event cap256 (DEC-582, capacity @ baseline algebra)** — g 1.149 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.209 · beats 1-gram: no — `lab/next_event_cap256.json` · measured under `d5df3a2c`
- **real stream · O3 attempt 4 (REJECTED)** — M4 persona 0.04646 (bar ≥ 0.15, DEC-572) · served coarse R² 0.4675 (bar ≥ 0.45) — `o3axis_i/` · measured under `6a8e2663`

A green atom means its frozen gate is met. For these atoms the gate certifies *measurement and attribution*, not a quality level (derived in `scripts/b_spec.py: gate_kind`, not asserted here): `B1.1`, `B2.3`, `B3.1`, `B3.2`, `B3.4`, `B3.5`, `B3.6`, `B4.3`, `B5.2`, `B6.1`, `B8.1`, `B8.2`. Their quality levels are DEC-frozen and live in the receipts (`PLAN.md` §6). Read the plan before reading a green row as 'good'.

## B — foundational scorecard (B-only, DEC-562)

The judge below is the **consequence** (B+C). This section is **B itself**: a row is PASS only on a current receipt that meets its gate; a missing or legacy receipt is UNKNOWN and names the producer that must be run. Legacy numbers are not evidence.

### B1 — Lawful · ❔ UNKNOWN

*Is the state built from the right algebra?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B1.1 | Lawful dynamics (S3) | merge/equivariance/mass-additivity exact on data; tau receipted; stable from step 0 | reporting | ❔ UNKNOWN | stale: measured by measure_dynamics_law.py against behavior d5df3a2c1211 (current c9f26925b418) — re-run that producer |
| B1.2 | Construction laws (C1-C6) | scan / field-permutation / mass / determinism / schema variant / frozen identity | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_construction_law.py |

### B2 — Sufficient · ❔ UNKNOWN

*Does B lose nothing about the future?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B2.1 | Worst-case sufficiency U | U = 1 - worst-case oracle gap vs full history, fixed capacity, broad published family | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| B2.2 | Full-history oracle gap | excess loss of E vs the full-history arm at fixed capacity, per included probe | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| B2.3 | Learned-only attribution | sufficiency of the learned shape vs the served statistics, reported separately | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |

### B3 — Accessible · ❔ UNKNOWN

*Can simple heads read what is there, at every stage?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B3.1 | Outlet accessibility (S4) | per-bucket accessibility parity + per-bucket negative controls | reporting | ❔ UNKNOWN | stale: measured by measure_views_law.py against behavior d5df3a2c1211 (current c9f26925b418) — re-run that producer |
| B3.2 | Stage decomposition M0->M1->served | the fine-structure metric at each stage, so a loss is attributed to a stage | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| B3.3 | Linear accessibility (current) | mean ridge R2 on A->B over the lab's persona+future fields; served vs PATH A | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| B3.4 | Block-aware serving geometry | exact counter block beside the learned shape, equal total variance; persona/future/ARI | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| B3.5 | Equal-dimension serving geometry | shape PCA + counter tile at 128/128 and 39/39; persona/future/ARI at full size | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| B3.6 | Served-object X-ray (ledger exposure) | the ledger alone, its named parts, and ledger block-PCA under the frozen B3.2/B3.4 consumers, against {exact counter, served shape, PATH A}; the branch is computed here from the arm numbers (DEC-569) | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |

### B4 — Decisive · ❔ UNKNOWN

*Does B support action/counterfactual decisions?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B4.1 | B-only dose-response / argmax | action-conditional response shape readable from the state (not through a judge leg) | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_decision_law.py |
| B4.2 | OPE / IPS known-effect recovery | E recovers the known effect with a paired CI excluding 0 | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_decision_law.py |
| B4.3 | Action-invariance | representation invariant to assignment, still predictive of the outcome | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_decision_law.py |

### B5 — Minimal & affordable · ❔ UNKNOWN

*Is it the smallest sufficient object, and fast enough?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B5.1 | Minimality k_min | served rank = the frozen reconstruction certificate | unmeasured | ❔ UNKNOWN | missing producer: measure_minimality_law.py |
| B5.2 | Cost / runtime / constant-state | runtime, memory and incremental-update receipt | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_runtime.py |

### B6 — Transferable · ❔ UNKNOWN

*Does it add signal on held-out consumers?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B6.1 | Objective battery (S5) | every content pressure beats its naive comparator (or is excluded and reported) | reporting | ❔ UNKNOWN | stale: measured by measure_battery_law.py against behavior d5df3a2c1211 (current c9f26925b418) — re-run that producer |
| B6.2 | Donor does-not-lose / adds / controls | E vs raw, fair table, table+E; random/destroyed negative controls | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_donor.py |
| B6.3 | Broad held-out transfer family | adds signal on a family broader than the eight judge legs | unmeasured | ❔ UNKNOWN | missing producer: measure_transfer_law.py |

### B7 — Robust · ❔ UNKNOWN

*Does it stay honest under shift and missingness?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B7.1 | ECE / OOD retention | calibration of the shipped next-type head + OOD retention | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_b_props.py |
| B7.2 | Missingness / unlogged confounders | uncertainty carried for unobserved events, never faked | unmeasured | ❔ UNKNOWN | missing producer: measure_robustness_law.py |

### B8 — Reproducible · ❔ UNKNOWN

*Is the artifact an identity?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B8.1 | Artifact identity | registry tag + config + data revision recorded and consistent | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_b_props.py |
| B8.2 | One-command reproducibility (S6) | the card is regenerated in one invocation at a recorded commit | reporting | ❔ UNKNOWN | stale: measured by measure_certification_law.py against behavior d5df3a2c1211 (current c9f26925b418) — re-run that producer |

## The judge — four legs × two numbers

Progress (`brain alone`) is a strict win. Delivery (`brain + ledger`) is a **non-inferiority** claim vs PATH A (margin 0.01 R2); a delivery tie is not a brain win.

| Leg | Column | Verdict | Evidence |
|---|---|---|---|
| supervised | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| supervised | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| unsupervised | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| rl | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| rl | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| simulator | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |

## Measured values — every task on the card

The leg verdict above is computed from these rows; a task can PASS while its leg fails on another task. Every number below is read from the receipts, never narrated.

| Leg | Column | Task | Brain | PATH A | Delta | CI | Random | Destroyed | N | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| — | — | no task rows (receipts missing or stale) | — | — | — | — | — | — | — | ❔ UNKNOWN |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ❔ UNKNOWN | stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash c9f26925b418b284 |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_leg_supervised.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_leg_unsupervised.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_leg_unsupervised.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_stages.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_stages.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_stages.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_stages.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_stages.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `judge_stages.json — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `B B1.1 [P1·B1] Lawful dynamics (S3) — stale: measured by measure_dynamics_law.py against behavior d5df3a2c1211 (current c9f26925b418) — re-run that producer`
- `B B1.2 [P1·B1] Construction laws (C1-C6) — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_construction_law.py`
- `B B2.1 [P2·B2] Worst-case sufficiency U — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `B B2.2 [P2·B2] Full-history oracle gap — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `B B2.3 [P2·B2] Learned-only attribution — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `B B3.1 [P3·B3] Outlet accessibility (S4) — stale: measured by measure_views_law.py against behavior d5df3a2c1211 (current c9f26925b418) — re-run that producer`
- `B B3.2 [P3·B3] Stage decomposition M0->M1->served — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `B B3.3 [P3·B3] Linear accessibility (current) — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `B B3.4 [P3·B3] Block-aware serving geometry — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `B B3.5 [P3·B3] Equal-dimension serving geometry — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `B B3.6 [P3·B3] Served-object X-ray (ledger exposure) — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284)`
- `B B4.1 [P2·B4] B-only dose-response / argmax — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_decision_law.py`
- `B B4.2 [P2·B4] OPE / IPS known-effect recovery — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_decision_law.py`
- `B B4.3 [P2·B4] Action-invariance — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_decision_law.py`
- `B B5.1 [P2·B5] Minimality k_min — missing producer: measure_minimality_law.py`
- `B B5.2 [P2·B5] Cost / runtime / constant-state — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_runtime.py`
- `B B6.1 [P3·B6] Objective battery (S5) — stale: measured by measure_battery_law.py against behavior d5df3a2c1211 (current c9f26925b418) — re-run that producer`
- `B B6.2 [P3·B6] Donor does-not-lose / adds / controls — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_donor.py`
- `B B6.3 [P3·B6] Broad held-out transfer family — missing producer: measure_transfer_law.py`
- `B B7.1 [P3·B7] ECE / OOD retention — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_b_props.py`
- `B B7.2 [P3·B7] Missingness / unlogged confounders — missing producer: measure_robustness_law.py`
- `B B8.1 [P1·B8] Artifact identity — stale: encoder behavior changed since receipt (d5df3a2c1211c402 != c9f26925b418b284) — re-run measure_b_props.py`
- `B B8.2 [P1·B8] One-command reproducibility (S6) — stale: measured by measure_certification_law.py against behavior d5df3a2c1211 (current c9f26925b418) — re-run that producer`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*This is the only card. B-only rows come from `scripts/b_spec.py`; B+C rows from `scripts/judge_spec.py`; both hang under the tree in `scripts/plan_spec.py` (rendered readably in `PLAN.md`); all read the receipts under `docs/report_cards/experiments/`. There is no separate audit card, and no row may be parked elsewhere (DEC-512).*
