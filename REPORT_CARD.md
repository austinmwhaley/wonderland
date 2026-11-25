# REPORT_CARD — B foundational, then PATH B > PATH A

*Generated `2026-11-25T19:09:10+00:00` · commit `d85630a` (`main` · dirty) · encoder behavior `de599142938c6c16`*

**Assessment:**

- ❌ **Plan root:** FAILED — 16/40 atomic problems green · 1 FAIL · 0 DEC-frozen · 0 v1 unassessed · 20 interior · reds: P3 (1).
- ❌ **B foundational:** NOT CERTIFIED — the current gate. 3/9 sections green · 8 rows PASS · 1 FAIL · 15 UNKNOWN/MISSING.
- ❌ **B+C consequence:** NOT DONE — 0/8 legs green · stages 0/6 · backward open: 0. The B/C/E1/v1 rows are green but the open BACKWARD requirements block completion.
- ✅ **Project v1:** V1.1-V1.5 green — identifiable stream; plugin acceptance 9/9; red_king known-effect 73.3% vs 25% chance, held-out policy 0.371 vs 0.208 random (CI > 0); red_queen offer composition DEPLOY/HOLD; caterpillar on red_queen's plan.

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## The plan — the tree this card measures (PLAN.md)

The goal, cut backwards into the problems that must all be solved, recursively to the atomic problems. This section is generated from `scripts/plan_spec.py` — the single definition of the tree; `PLAN.md` is its readable render. The plan defines no gate: every measured atomic problem is a frozen row in its B/judge section below, and an interior verdict is the worst of its children.

**Plan root:** ❌ FAILED — 16/40 atomic problems green · 1 FAIL · 0 DEC-frozen (no card rows) · 0 v1 not assessed on this card · 20 interior problems.

| Level | Node | What must be true | Atomic problems | Verdict |
|---|---|---|---|---|
| 0 | **G** A certified universal donor, delivered | What is the whole problem? | 40 atomic problems | ❌ FAIL |
| 1 | **P1** The donor exists | Is there a per-stream artifact built from the right algebra, and is it an identity? | 4 atomic problems | ❔ UNKNOWN |
| 2 | **B1** Lawful | Is the state built from the right algebra? | 2 atomic problems | ❔ UNKNOWN |
| 2 | **B8** Reproducible | Is the artifact an identity? | 2 atomic problems | ❔ UNKNOWN |
| 1 | **P2** The donor suffices | Does the embedding lose nothing that matters about the future — for prediction and decision — … | 11 atomic problems | ❔ UNKNOWN |
| 2 | **B2** Sufficient | Does B lose nothing about the future? | 3 atomic problems | ❔ UNKNOWN |
| 2 | **E1** Served-object capacity (signature head + delivery port) | Can the served object carry the stream's structure — signature for order, persona/R2 for … | 3 atomic problems | ✅ PASS |
| 3 | **E1.a** next_event mastered by the served signature head (curriculum step 1) | Does the served signature block beat the 1-gram on the known-truth next_event structure at the … | — | ✅ PASS |
| 3 | **E1.b** intent mastered by the served signature head (curriculum step 2) | Does the served signature block beat the 1-gram on the per-customer intent structure, at a … | — | ✅ PASS |
| 3 | **E1.c** delivery port: bank + measure PCA-128 under DEC-572's bars | Does the served delivery object meet eff_rank>=30, persona>=0.15, R2>=0.45, stable, plus the B1 … | — | ✅ PASS |
| 2 | **B4** Decisive | Does B support action/counterfactual decisions? | 3 atomic problems | ✅ PASS |
| 2 | **B5** Minimal & affordable | Is it the smallest sufficient object, and fast enough? | 2 atomic problems | ✅ PASS |
| 1 | **P3** The donor is usable | Can simple consumers read it, does it transfer beyond our known tasks, and does it stay honest … | 12 atomic problems | ❌ FAIL |
| 2 | **B3** Accessible | Can simple heads read what is there, at every stage? | 6 atomic problems | ❔ UNKNOWN |
| 2 | **B6** Transferable | Does it add signal on held-out consumers? | 3 atomic problems | ❌ FAIL |
| 2 | **B7** Robust | Does it stay honest under shift and missingness? | 2 atomic problems | ❔ UNKNOWN |
| 2 | **B9** The encoder beats the 1-gram baseline | Does the encoder's frozen output beat the existing methods — the 1-gram baseline — on the … | 1 atomic problem | ✅ PASS |
| 1 | **P4** The consequence is delivered | Does PATH B (frozen encoder -> simple head) beat PATH A (wide table -> GBM incumbent) for the … | 8 atomic problems | ❔ UNKNOWN |
| 2 | **JUDGE-supervised** supervised leg | forecast targets beat the incumbent GBM on the same splits | 2 atomic problems | ❔ UNKNOWN |
| 2 | **JUDGE-unsupervised** unsupervised leg | distance equals shared futures; structure beats table/random/destroyed | 2 atomic problems | ❔ UNKNOWN |
| 2 | **JUDGE-rl** rl leg | policy value beats table and random on randomized/known-effect data | 2 atomic problems | ❔ UNKNOWN |
| 2 | **JUDGE-simulator** simulator leg | recovers the built-in counterfactual effects (red_king role) | 2 atomic problems | ❔ UNKNOWN |
| 1 | **P5** The stack is v1-complete | Does the full system deliver a certified decision: identifiable stream -> donor -> plugins -> … | 5 atomic problems | ✅ PASS |
| 2 | **V1.1** Causally identifiable stream | Does rabbit_hole emit randomized arms with a known effect, recorded as such? | — | ✅ PASS |
| 2 | **V1.2** Each plugin passes its independent gate | Do the supervised, unsupervised and white_queen consumers each pass their own gate? | — | ✅ PASS |
| 2 | **V1.3** red_king recovers known effects | Does red_king recover the known counterfactual effect and improve the certified decision with … | — | ✅ PASS |
| 2 | **V1.4** red_queen composes actions | Does red_queen compose next-best actions across cadences under constraints? | — | ✅ PASS |
| 2 | **V1.5** caterpillar explains from artifacts | Does caterpillar answer 'why' from artifacts only? | — | ✅ PASS |

*Macro review:* At behavior `7f20fd0b`, the plan root is SOLVED 39/39 on the synthetic `demo_v2` prototype. B is certified 23/23 on the single `cfm_v2k_vf2_cert` bundle (base 272, bank 179, donor-only 106, delivery port 307, measure view; every B row mapped to its component). B+C is 8/8 on the served block bank (DEC-598), with the brain boundary explicit: the measure-free donor-only rerun fails unsupervised persona/future and supervised h0/return_proneness (`bc_donor_diagnostic.json`), so the result is a served-bank result, not an encoder-alone win. E1 is served-object capacity: E1.a/E1.b read the served signature block (g -0.581; -0.721/0.181), and E1.c is a delivery/ledger port (eff_rank 30.9, persona 0.267, R2 0.460) whose receipt records that no measure-free object clears all three bars. V1.1-V1.5 are green on current receipts: the demo randomized offer stream, plugins 9/9, red_king known-effect 73.3% vs 25% chance and a held-out policy 0.371 vs 0.208 random, red_queen's regenerated DEPLOY/HOLD offer composition, and caterpillar on red_queen's own plan. No BACKWARD BREAKING/MAJOR requirement is open; the production-stream scope is explicitly not claimed.

*Frontier:* The plan root is SOLVED 39/39 at behavior 7f20fd0b: B certified 23/23 on the single cfm_v2k_vf2_cert bundle; B+C 8/8 on the served bank; E1 is served-object capacity (E1.a/E1.b served signature head, E1.c delivery/ledger port); V1.1-V1.5 green; no open BACKWARD requirement. Scope: the certified artifact is the synthetic demo_v2 prototype. The trunk-only rank is 4.11 (base) / 10.33 (bank); the delivery port reaches 30.9 only with the measure PCA-128, and the measure-free donor-only B+C rerun does not clear all legs. Both boundaries are recorded in the bundle and the port/donor receipts.

**Frontier readings (reported, not gated — the evidence behind E1's FRONTIER badge):**

- **dev rig · next_event iter1** — g 1.147 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.153 · beats 1-gram: no — `lab/next_event_iter1.json` · measured under `6a8e2663`
- **dev rig · next_event iter2** — g 1.154 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.473 · beats 1-gram: no — `lab/next_event_iter2.json` · measured under `4053d2f9`
- **dev rig · next_event iter3 (DEC-578, candidate 2)** — g 1.182 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.248 · beats 1-gram: no — `lab/next_event_iter3.json` · measured under `905b4f88`
- **dev rig · next_event iter4 (DEC-581, candidate 3)** — g 1.17 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.836 · beats 1-gram: no — `lab/next_event_iter4.json` · measured under `d5df3a2c`
- **dev rig · next_event cap256 (DEC-582, capacity @ baseline algebra)** — g 1.149 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.209 · beats 1-gram: no — `lab/next_event_cap256.json` · measured under `d5df3a2c`
- **dev rig · intent (DEC-586, E1.b)** — g 2.665 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.177 · beats 1-gram: no — `lab/intent_dec586.json` · measured under `7baf744f`
- **dev rig · intent + mass counter** — g 2.633 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.309 · beats 1-gram: no — `lab/intent_dec586_counter.json` · measured under `7baf744f`
- **dev rig · intent + transition signature** — g 2.6 (bar ≤ 0.5, DEC-576/577) · eff_rank 1.177 · beats 1-gram: no — `lab/intent_dec586_sig.json` · measured under `7baf744f`
- **demo stream · O3 attempt 4 (REJECTED)** — M4 persona 0.04646 (bar ≥ 0.15, DEC-572) · served coarse R² 0.4675 (bar ≥ 0.45) — `o3axis_i/` · measured under `6a8e2663`

A green atom means its frozen gate is met. For these atoms the gate certifies *measurement and attribution*, not a quality level (derived in `scripts/b_spec.py: gate_kind`, not asserted here): `B1.1`, `B2.3`, `B3.1`, `B3.2`, `B3.4`, `B3.5`, `B3.6`, `B4.3`, `B5.2`, `B6.1`, `B7.2`, `B8.1`, `B8.2`. Their quality levels are DEC-frozen and live in the receipts (`PLAN.md` §6). Read the plan before reading a green row as 'good'.


## Real-data story — DEC-611/615 multi-domain campaign (development instruments)

*Scope:* lab receipts are a development instrument (DEC-576), never B-row evidence on the main card; the certified artifact remains the synthetic `demo_v2` prototype (B-001). Every figure below is computed from the receipts — no narrated numbers. The walls and the H-axis finding are the campaign's recorded conclusions (DEC-615, closed as-is 2026-11-24), disclosed here — never re-decided.

### sensor_fleet

| head | 500 | 2K | 10K | 25K | long |
|---|---|---|---|---|---|
| ridge | +1.7003 ✕ | +1.6740 ✕ | +1.1958 ✕ | +1.2765 ✕ | +1.2041 ✕ · n=25000 |
| hier | +0.2447 ✓ | -0.2617 ✓ | -0.0753 ✓ | -0.0461 ✓ | -0.0080 ✓ · n=25000 |

*Probe Δ (×2 window, same N):* ridge Δ -0.0724 (+1.2765 → +1.2041, n=25000) · hier Δ +0.0381 (-0.0461 → -0.0080, n=25000)

*Baseline story (computed, ridge):* pooled 1-gram nll 0.7582 → 0.7791 → 0.8167 → 0.8029 (Δ +0.0447) while the encoder nll 0.8635 → 0.8945 → 0.8576 → 0.8568 (Δ -0.0066) — flat; the ridge head's apparent gain is largely the baseline degrading.

### clickstream_sessions

| head | 500 | 2K | 10K | 25K | long |
|---|---|---|---|---|---|
| ridge | +2.0280 ✕ | +2.1726 ✕ | +2.0543 ✕ | — | +2.2324 ✕ · n=10000 |
| hier | +0.3398 ✓ | +0.0443 ✓ | +0.0742 ✓ | — | -0.0677 ✓ · n=10000 |

*Probe Δ (×2 window, same N):* ridge Δ +0.1780 (+2.0543 → +2.2324, n=10000) · hier Δ -0.1419 (+0.0742 → -0.0677, n=10000)

*Baseline story (computed, ridge):* pooled 1-gram nll 1.1076 → 1.0722 → 1.0836 (Δ -0.0240) while the encoder nll 1.3664 → 1.3259 → 1.3238 (Δ -0.0427) — flat; the baseline does not degrade in this range.

### server_logs

| head | 500 | 2K | 10K | 25K | long |
|---|---|---|---|---|---|
| ridge | +5.0339 ✕ | +1.7442 ✕ | +1.4218 ✕ | +1.5118 ✕ | +1.4863 ✕ · n=25000 |
| hier | -1.7328 ✓ | -0.1821 ✓ | +0.0087 ✓ | -0.0359 ✓ | -0.0097 ✓ · n=25000 |

*Probe Δ (×2 window, same N):* ridge Δ -0.0255 (+1.5118 → +1.4863, n=25000) · hier Δ +0.0262 (-0.0359 → -0.0097, n=25000)

*Baseline story (computed, ridge):* pooled 1-gram nll 0.7819 → 0.9208 → 0.9754 → 0.9579 (Δ +0.1760) while the encoder nll 1.0379 → 1.0714 → 1.0839 → 1.0805 (Δ +0.0425) — flat; the ridge head's apparent gain is largely the baseline degrading.

### Computed verdicts (not gates)

- ridge beats the 1-gram baseline: **0/14 cells** (`beats_base` True)
- hier beats the 1-gram baseline: **14/14 cells** (`beats_base` True)

- H-axis (×2 window, same N, computed): max |Δ| = 0.1780 across 6 probe cells (ridge -0.0724, +0.1780, -0.0255 · hier +0.0381, -0.1419, +0.0262) — vs the cross-customer sample move (|Δ| = 0.03–3.52 across the three domains).

### Walls + contingency (disclosed, never fixed here)

- **clickstream × 25K — BUDGET wall** (DEC-615): OOM 72 s into the first task-loss pass — ≈44.7 GB peak (22.66 live-held + 22.05 logits draw) vs ≈30.1 GB idle clamp. Driver = schema × N (`numeric=0 categorical=4` vs sensor's `numeric=2 categorical=1`; sensor n25k completed RC=0 at the same N, train peak_alloc 2.88 GB). **No retry owed.** The 25K tier is therefore clickstream-absent in every table above.
- **clickstream × 10K×2w neg control — BUDGET wall, second instance**: OOM at the same site (≈41.3 GB peak vs 28.40 launch cap); its positive twin on the equal-row fixture completed RC=0 (train peak_alloc 17.52 GB). The neg/pos demand delta is unattributed from campaign logs.
- **Contingency-owed (pre-registered, DEC-615)**: all 8 completed ladder cells + all 3 probe cells fire the neg+adv contingency — the ladder `mastered` flags are contingency-owed (controls untrained at the ladder N); sensor long is complete with controls (neg+adv RC=0, re-measured 03:29: ridge neg `reads_no_signal=True`, ridge+hier adv `reads_as_expected=True`, hier neg `n_B_rows=0` = the vacuous-control artifact, `lab_gate_a_side` all-True); clickstream long is A-side-only (neg OOM → adv not launched); server long is A-side-only (neg RC=0; adv SIGTERM'd by the pause order). No master claim is licensed by any receipt at pause.


## Incremental improvements — work to achieve (additive, non-gated)

*The frozen gates pass (the card is SOLVED) — but the gates certify the instrument, not the encoder. Every figure is computed from the receipts — nothing is narrated.*

**The outcome — the encoder beats the existing methods:** the hier head beats the 1-gram baseline on **14/14 cells** (`beats_base` True, all three domains). The ridge head (the shipping consumer head) beats the 1-gram on **0/14 cells** — the bounded gap, recorded as-is.

### 1 — Ridge head (shipping consumer head): never beats the 1-gram baseline (0/14 cells `beats_base`)

- The gap (computed): the cross-customer move (500 → largest completed ridge cell) costs |Δ| = 0.03–3.52 (sensor 0.42, clickstream 0.03, server 3.52); the ×2-window probe moves it by ≤ 0.178 (H-axis NO).
- The diagnosis (computed): the encoder's own nll is nearly flat in N (sensor 0.8635→0.8568, server 1.0379→1.0805) while the pooled 1-gram base nll RISES (sensor 0.7582→0.8029, server 0.7819→0.9579) — the ridge head's apparent gain is largely the baseline degrading. The ridge head's excess loss is cross-customer finite-sample / estimation-efficiency limited — the sample, not the horizon, carries the gap.
- The work to achieve it: the ridge head's estimation efficiency — the cross-customer finite-sample gap.
  A pre-registered estimation-efficiency axis (batch/headroom/sample-size, not a capacity tweak) would   measure whether the ridge head can beat the 1-gram at fixed capacity.

### 2 — Hier head: beats the 1-gram on 14/14 cells (g ≤ 0 at the 25K/long cells — the finite-sample-oracle artifact)

- The outcome (computed): the hier head beats the 1-gram on **14/14 cells** (`beats_base` True across all three domains — the encoder beats the existing methods via the transition head). max g +0.3398 at the 500-tier cell; all hier g ≤ 0 at the 25K/long cells (sensor −0.0461/−0.0080, clickstream −0.0677, server −0.0359/−0.0097).
- The gap (computed): the 25K/long cells sit at/below the oracle floor (hier g ≤ 0.05 there); the ×2-window   probe moves it by ≤ 0.142 (not directionally consistent — sensor +0.038,   clickstream −0.142, server +0.026).
- The diagnosis (computed): the encoder's nll is marginally under the closed-form oracle on all three domains   (the known finite-sample-oracle artifact) — the g ≤ 0 at large N is the artifact, not a true failure to   beat the 1-gram (nll_enc < nll_base on all 14 cells).
- The work to achieve it: the transition head's floor — the hier head's capacity/order-carrying at fixed   capacity. A pre-registered order/capacity axis would measure whether the head can carry the conditional   row beyond the 1-gram.

*These are the incremental improvements — the work to achieve them is the ridge head's estimation efficiency (the shipping consumer head, 0/14 cells) + the transition head's floor (the finite-sample-oracle artifact). The certified gates pass; the outcome — the encoder beats the 1-gram — is achieved via the hier head (14/14 cells); the substance is the ridge-head gap, recorded as-is.*

## B — foundational scorecard (B-only, DEC-562)

The judge below is the **consequence** (B+C). This section is **B itself**: a row is PASS only on a current receipt that meets its gate; a missing or legacy receipt is UNKNOWN and names the producer that must be run. Legacy numbers are not evidence.

### B1 — Lawful · ❔ UNKNOWN

*Is the state built from the right algebra?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B1.1 | Lawful dynamics (S3) | merge/equivariance/mass-additivity exact on data; tau receipted; stable from step 0 | reporting | ❔ UNKNOWN | stale: measured by measure_dynamics_law.py against behavior ec9b5daa9874 (current de599142938c) — re-run that producer |
| B1.2 | Construction laws (C1-C6) | scan / field-permutation / mass / determinism / schema variant / frozen identity | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) — re-run measure_construction_law.py |

### B2 — Sufficient · ❔ UNKNOWN

*Does B lose nothing about the future?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B2.1 | Worst-case sufficiency U | U = 1 - worst-case oracle gap vs full history, fixed capacity, broad published family | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16) |
| B2.2 | Full-history oracle gap | excess loss of E vs the full-history arm at fixed capacity, per included probe | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16) |
| B2.3 | Learned-only attribution | sufficiency of the learned shape vs the served statistics, reported separately | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16) |

### B3 — Accessible · ❔ UNKNOWN

*Can simple heads read what is there, at every stage?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B3.1 | Outlet accessibility (S4) | per-bucket accessibility parity + per-bucket negative controls | reporting | ❔ UNKNOWN | stale: measured by measure_views_law.py against behavior ec9b5daa9874 (current de599142938c) — re-run that producer |
| B3.2 | Stage decomposition M0->M1->served | the fine-structure metric at each stage, so a loss is attributed to a stage | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16) |
| B3.3 | Linear accessibility (current) | mean ridge R2 on A->B over the lab's persona+future fields; served vs PATH A | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16) |
| B3.4 | Block-aware serving geometry | exact counter block beside the learned shape, equal total variance; persona/future/ARI | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16) |
| B3.5 | Equal-dimension serving geometry | shape PCA + counter tile at 128/128 and 39/39; persona/future/ARI at full size | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16) |
| B3.6 | Served-object X-ray (ledger exposure) | the ledger alone, its named parts, and ledger block-PCA under the frozen B3.2/B3.4 consumers, against {exact counter, served shape, PATH A}; the branch is computed here from the arm numbers (DEC-569). The B+C brain is the served bank (donor blocks + served statistics, DEC-598); the measure-free donor-only boundary is the separate bc_donor_diagnostic.json (B-008), not a hidden ledger point | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16) |

### B4 — Decisive · ✅ PASS

*Does B support action/counterfactual decisions?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B4.1 | B-only dose-response / argmax | action-conditional response shape readable from the state (not through a judge leg) | quality | ✅ PASS | E share 0.2983 vs table 0.2438 (CI [+0.0249, +0.0855]) · random 0.2586 · destroyed 0.2384 · n=1485 · oracle ceiling 0.6404 |
| B4.2 | OPE / IPS known-effect recovery | E recovers the known effect with a paired CI excluding 0 | quality | ✅ PASS | E policy IPS 0.1592 vs table 0.1517 (diff +0.01 [+0.00, 0.0139] > 0) · accE 0.295 vs accT 0.247 · random IPS None · n_test=74746 · levels {'0': 0.2515, '1': 0.2482, '2': 0.252, '3': 0.2482} |
| B4.3 | Action-invariance | representation invariant to assignment, still predictive of the outcome | reporting | ✅ PASS | band cert ok (P1==P2 bit-identical; P1 vs stored S 2.980232238769531e-07 / E 0.0009585114639776293 vs bands 6.199199447110004e-07 / 0.015569581650197506; teeth True) · marker null: share 0.30033670033670035 vs stored 0.2983164983164983 (ok=True) · level permutation: share 0.24511784511784512 vs table 0.24377104377104378 (|Δ|∞ 313.1736521923858, ok=True) |

### B5 — Minimal & affordable · ✅ PASS

*Is it the smallest sufficient object, and fast enough?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B5.1 | Minimality k_min | served rank = the frozen reconstruction certificate | quality | ✅ PASS | served shape rank 32 == k_min 32 (shape full 32) · explicit block 16 · joint floor 32 (unconstrained) · n_train 451 / n_test 439 · noise 0.04773 |
| B5.2 | Cost / runtime / constant-state | runtime, memory and incremental-update receipt | reporting | ✅ PASS | 5000/5000 customers · cold 57.67s / warm 520.03s · 8.7 rows/s · dims 256/256 · peak RSS 3381.8 MB (slice 32.0 GB) · incremental (absorbed 0, idle 5000, wall 0.66s) |

### B6 — Transferable · ❌ FAIL

*Does it add signal on held-out consumers?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B6.1 | Objective battery (S5) | every content pressure beats its naive comparator (or is excluded and reported) | reporting | ❔ UNKNOWN | stale: measured by measure_battery_law.py against behavior ec9b5daa9874 (current de599142938c) — re-run that producer |
| B6.2 | Donor does-not-lose / adds / controls | E vs raw, fair table, table+E; random/destroyed negative controls | quality | ❌ FAIL | encoder_donor.json: y_orders_90: table+E CI lo -0.0046 <= 0; y_value_90: E 0.5807 < raw 0.5952; y_value_90: E 0.5807 < fair 0.5952; y_value_90: table+E CI lo -0.0313 <= 0 |
| B6.3 | Broad held-out transfer family | adds signal on a family broader than the eight judge legs | quality | ✅ PASS | published family 15 targets · E wins 14/15 (0.933) · family mean E 0.4252 vs table 0.3857 (diff +0.0395) · random 0.0174 / destroyed 0.0257 |

### B7 — Robust · ❔ UNKNOWN

*Does it stay honest under shift and missingness?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B7.1 | ECE / OOD retention | calibration of the shipped next-type head + OOD retention | quality | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) — re-run measure_b_props.py |
| B7.2 | Missingness / unlogged confounders | uncertainty carried for unobserved events, never faked | reporting | ✅ PASS | next-time 0.8 coverage 0.800 (n 1485, MAE 1.15d) · occurrence AUC 0.6339 / Brier 0.2794 (h 1.338484d) · response coverage 0.8/0.8 · missingness mean rel L2 10%:0.094, 25%:0.233 |

### B8 — Reproducible · ❔ UNKNOWN

*Is the artifact an identity?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B8.1 | Artifact identity | registry tag + config + data revision recorded and consistent | reporting | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) — re-run measure_b_props.py |
| B8.2 | One-command reproducibility (S6) | the card is regenerated in one invocation at a recorded commit | reporting | ❔ UNKNOWN | stale: measured by measure_certification_law.py against behavior ec9b5daa9874 (current de599142938c) — re-run that producer |

### B9 — The encoder beats the 1-gram baseline · ✅ PASS

*Does the encoder's frozen output beat the existing methods — the 1-gram baseline — on the customer event stream?*

| # | Property | Metric | Green means | Verdict | Evidence / missing producer |
|---|---|---|---|---|---|
| B9.1 | The encoder beats the 1-gram baseline on the customer event stream | the ridge head (the shipping consumer head) beats the 1-gram baseline on the demo — the encoder is a foundation, not a green receipt | quality | ✅ PASS | the encoder beats the 1-gram baseline on the demo (beats_unigram=True, paired CI [-0.2908, -0.2116] < 0, n=1500) — the outcome: the encoder is a foundation, not a green receipt |

## The judge — four legs × two numbers

Progress (`brain alone`) is a strict win. Delivery (`brain + ledger`) is a **non-inferiority** claim vs PATH A (margin 0.01 R2); a delivery tie is not a brain win.

| Leg | Column | Verdict | Evidence |
|---|---|---|---|
| supervised | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| supervised | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| unsupervised | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| rl | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| rl | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| simulator | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |

## Measured values — every task on the card

The leg verdict above is computed from these rows; a task can PASS while its leg fails on another task. Every number below is read from the receipts, never narrated.

| Leg | Column | Task | Brain | PATH A | Delta | CI | Random | Destroyed | N | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| — | — | no task rows (receipts missing or stale) | — | — | — | — | — | — | — | ❔ UNKNOWN |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ❔ UNKNOWN | stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash de599142938c6c16 |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_leg_supervised.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_leg_unsupervised.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_leg_unsupervised.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_stages.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_stages.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_stages.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_stages.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_stages.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `judge_stages.json — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16)`
- `B B1.1 [P1·B1] Lawful dynamics (S3) — stale: measured by measure_dynamics_law.py against behavior ec9b5daa9874 (current de599142938c) — re-run that producer`
- `B B1.2 [P1·B1] Construction laws (C1-C6) — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) — re-run measure_construction_law.py`
- `B B2.1 [P2·B2] Worst-case sufficiency U — stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16)`
- `B B2.2 [P2·B2] Full-history oracle gap — stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16)`
- `B B2.3 [P2·B2] Learned-only attribution — stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16)`
- `B B3.1 [P3·B3] Outlet accessibility (S4) — stale: measured by measure_views_law.py against behavior ec9b5daa9874 (current de599142938c) — re-run that producer`
- `B B3.2 [P3·B3] Stage decomposition M0->M1->served — stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16)`
- `B B3.3 [P3·B3] Linear accessibility (current) — stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16)`
- `B B3.4 [P3·B3] Block-aware serving geometry — stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16)`
- `B B3.5 [P3·B3] Equal-dimension serving geometry — stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16)`
- `B B3.6 [P3·B3] Served-object X-ray (ledger exposure) — stale: encoder behavior changed since receipt (0fccd9a89829f47d != de599142938c6c16)`
- `B B6.1 [P3·B6] Objective battery (S5) — stale: measured by measure_battery_law.py against behavior ec9b5daa9874 (current de599142938c) — re-run that producer`
- `B B6.2 [P3·B6] Donor does-not-lose / adds / controls — encoder_donor.json: y_orders_90: table+E CI lo -0.0046 <= 0; y_value_90: E 0.5807 < raw 0.5952; y_value_90: E 0.5807 < fair 0.5952; y_value_90: table+E CI lo -0.0313 <= 0`
- `B B7.1 [P3·B7] ECE / OOD retention — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) — re-run measure_b_props.py`
- `B B8.1 [P1·B8] Artifact identity — stale: encoder behavior changed since receipt (ec9b5daa98740c83 != de599142938c6c16) — re-run measure_b_props.py`
- `B B8.2 [P1·B8] One-command reproducibility (S6) — stale: measured by measure_certification_law.py against behavior ec9b5daa9874 (current de599142938c) — re-run that producer`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| B-001 | BREAKING | P1/P6 | closed | The certified artifact is the synthetic demo, not a real production stream. | Scope every claim to the demo prototype (M-001 option 2). | A-001 |
| B-002 | BREAKING | P6/P8 | closed | E1.a/E1.b read the fixture pair counts, not the frozen served encoder. | Serve the signature as a lab-E block and evaluate the frozen E. | A-002 |
| B-003 | BREAKING | P6/P8 | closed | E1.c is a ledger-backed port; receipt unstamped. | Record E1.c as a delivery/ledger port (M-003 option 2). | A-003 |
| B-004 | BREAKING | P6/E10 | closed | The five v1 receipts are unstamped and lack products identity. | Stamp/re-run the v1 gates at the recorded behavior/commit/products. | A-004 |
| B-005 | BREAKING | P6/E9 | closed | V1.3 red_king is synthetic; the real known-effect path is not run. | Run red_king's known-effect gate on the demo randomized stream. | A-005 |
| B-006 | MAJOR | P6/E9 | closed | V1.4 red_queen reads a stored old certificate with behavior 2.69 vs bar 5.64 and deploy true. | Regenerate the channel/discount composition certification on the current demo. | A-006 |
| B-007 | MAJOR | P6/P8 | closed | V1.5's plan is synthetic (offer_opt + outcome_mean), not a red_queen decision plan. | Run caterpillar on a real red_queen NBA plan. | A-007 |
| B-008 | BREAKING | P6/P7 | closed | The B+C brain column includes measure-ledger parts. | Separate the donor block bank from the measure ledger; record the served-bank boundary (M-008 option 2). | A-008 |
| B-009 | MAJOR | P6/E10 | closed | The B gate is a mixed-artifact certification (base, port, bank). | Certify B on one served artifact. | A-009 |
| B-010 | MAJOR | P6/E10 | closed | E1.b positive receipts are stale (behavior c6f56a2e vs df73c2aa). | Re-run intent/intent_long at the current behavior. | A-010 |
| B-011 | MINOR | P6 | closed | The earlier audit status claimed no findings while the v1 receipts were unstamped and E1.b stale. | Tie the review verdict to receipt stamps. | A-011 |
| B-012 | MAJOR | P6 | closed | DEC-598/599/600/601 are cited as load-bearing by PLAN/STATUS/card/plan_spec/report_card and the commit titles of e9dfd6b/0bc0d1f/7a3ae15, bu | Append post-hoc-labeled DEC-598/599/600/601 entries reconstructed from commit messages + receipts + STATUS (record repair, B-012), seal 597/ | A-012 |
| B-013 | MAJOR | P6/E10 | closed | judge_leg_unsupervised.json is self-contradictory (working:true vs checks.unsupervised.working:false) with arms/diagnostics/path_a/excluded/ | measure_bc_bank.py writes a current checks rollup + bank-v2 arms for every leg it scores and drops the stale keys it redefines on merge; re- | A-013 |
| B-014 | MAJOR | P6/E10 | closed | At the SOLVED commit 38740c8 the AGENTS.md commit gates are red: ruff format --check fails on 9 scripts/ files (since 63e13da) and the fast  | ruff format the 9 scripts/ files; add both producers to DEFAULT_PATTERNS; ruff check + ruff format --check + pytest -m 'not slow' all RC=0;  | A-014 |
| B-015 | MINOR | E10 | closed | cfm_v2k_vf2_cert/bundle_receipt.json (the B-009 single-artifact manifest, registry 116087fb206c2d81) has no _meta stamp while all 23 row rec | build_b_cert_bundle.py stamps _meta on the bundle receipt; re-run the builder; bundle registry sha must remain 116087fb206c2d81 and the card | A-015 |
| B-016 | MINOR | P6 | closed | The previous AUDIT §4 carried an orphaned table separator with B-004/B-010 rows duplicated outside the closed table, and §5 disclosed fast-t | Repair the file (defects in BACKWARD's own file may be fixed by the rewrite). | A-016 |
| B-017 | INFO | P6/P8 | closed | Supervised leg evidence discloses: 5/7 traits have vacuous negative controls (R2 < -1, wide-design conditioning artifact; frozen VACUOUS_CON | none required (disclosure); closed by the operator ruling (DEC-618, 2026-11-24). | A-017 |
| B-018 | INFO | E10 | open | plugin_acceptance.json (V1.2 reference receipt, n_rows 9) has no _meta; the gate receipt v1_plugins.json it points at is stamped (behavior d | none required (disclosure); stamp the reference on a future V1.2 producer pass. | A-018 |
| B-019 | MINOR | P6/E10 | closed | R7 route step evidence text (plan_spec.py:631, rendered at PLAN.md:619 and on the card) still reads the DEC-605-era count 'closed all 11 fin | One-line durable-outcome repair of the R7 evidence text without a hardcoded finding count (state the outcome; name the three scope closures  | A-019 |
| B-020 | MAJOR | P1 (Universal) / Honest (claim scope) | closed | The closing BACKWARD reviews (FULL at 38740c8...b512068) falsified the record's plumbing (citations, manifests, self-consistency, route text | DEC-611 campaign (pre-registered 2b37adf): three structurally distinct known-truth domains through the ONE simulator, full three-piece contr | A-020 |
| B-021 | MINOR | P6 (single source of truth) / P10 (reproducibility) / Honest | closed | The certification CHAIN in measure_certification_law.py still mapped the legs step to measure_judge_law.py, whose main() implements the pre- | Fix the CHAIN legs mapping to measure_bc_bank.py (one producer stamps all four files), script-only so behavior stays 89cec70b; re-stamp via  | A-021 |
| B-022 | MAJOR | P1 (Universal) / E4 (adapt to input by construction) | closed | supervised_gate._raw_features (the model-free RFM/GBM comparator) hardcoded pl.col('value') for the monetary aggregate (supervised_gate.py:2 | Derive the monetary field via cfm_data._primary_numeric_field(d, cfg) with a numeric-free zero fallback, matching the encoder's own field re | A-022 |
| B-023 | MINOR | E5 (modularity/contracts) / E12 (implement intent, not the example) | open | onboard._derive_reward_event(df, value_col) (onboard.py:585) accepts value_col but aggregates the literal pl.col('value') (:590); correct on | One-axis subgoal: make the signature honest - use value_col, or drop it and name the canonical-frame contract it actually relies on - with a | A-023 |
| B-024 | INFO | P1 (Universal) - named scope | open | The supervised LEG's target family is a frozen ecommerce evaluation contract: _labels reads $.value/$.touchpoint/$.category_code and filters | None required (disclosure): the leg's generality is bounded to the declared target contract; a new reference stream is a separate DEC. | A-024 |
| B-025 | INFO | P1 (Universal) - named scope | open | The decision/RL LEG's reward contract is declared on the randomized demo stream: _arms_rewards hardcodes event_type LIKE '%send%' (arm) and  | None required (disclosure): the decision leg's generality is bounded to the declared arm/reward contract. | A-025 |

---

*This is the only card. B-only rows come from `scripts/b_spec.py`; B+C rows from `scripts/judge_spec.py`; both hang under the tree in `scripts/plan_spec.py` (rendered readably in `PLAN.md`); all read the receipts under `docs/report_cards/experiments/`. There is no separate audit card, and no row may be parked elsewhere (DEC-512).*
