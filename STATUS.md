# STATUS — wonderland

*Live state only. **Measurement lives ONLY in `REPORT_CARD.md`** — the single card. This file is
state, blockers and next actions; history is `DECISIONS.md`; BACKWARD requirements are `AUDIT.md`
(requirements only, never a measurement card). Rewritten 2026-11-02 (DEC-562/563: B is the first
section and the first gate; B+C is the consequence; project v1 is a later scope; 32 GB max).*

## Macro

**Goal (Layer 0, now):** one frozen, self-supervised, schema-agnostic encoder per event stream
whose embeddings are a **universal donor** — supervised, unsupervised, RL and simulator consumers
all get signal from it through simple heads.

**Definition of progress:** **PATH B (event stream → frozen encoder → simple head) > PATH A
(wide table → GBM)**, with paired CIs and controls. The card reports two numbers per leg:
`brain alone` = **progress**, `brain + ledger` = **delivery** (non-inferiority). Ledger points are
never a brain win.

**Stage:** Layer 0 — the encoder is not yet mastered. B is now measured directly on the card
(B1–B8) before the B+C consequences. **Macro reframe DEC-567 (2026-11-02):** the X-ray is far
enough to locate the gaps (exposure / learning / decision objective). **Scope tightening DEC-568
(2026-11-02):** B-only for now — every DEC-567 option re-anchored to B's own rows; B+C kept on the
card as a reported consequence, not the near-term target. **D2–D5 accepted 2026-11-02**
("ok do what needs to be done"); DEC-569 (O0) and DEC-570 (`U` spec) pre-registered; the sprint
runs O0 first. D1 stays deferred with the B+C judge re-aim. **DEV-RIG pivot DEC-576
(2026-11-03, operator):** the real-data outer loop was dominated by protocol/environment
defects (4 of O3's 5 attempts died before the science); make ANY foundation model on the lab's
known-truth streams (fixtures were pre-built, 0/32 ever run) before THE foundation model on the
real stream. Lab receipts are a development instrument — never B-row evidence on the main card;
certification stays on the real stream.

## The problem in plain terms (DEC-588)

The encoder turns each customer's history into one vector. That vector is effectively a
weighted average, so it holds **what** the customer does — how often they view/search/buy,
their value profile — but not **what follows what** (the order/transition structure).
Measured on cfm_v2k: 98–99% of the history's histogram/quantiles is recoverable from the
vector, but only ~50% of the transition matrix. Decisions and the 90-day value/orders rows
need the missing half. The fix is a small **order memory** in the state: pair statistics
(what followed what) updated as events arrive, plus a training objective that predicts the
next window's transition matrix. The mass/shape split was already tried and works for the
histogram half; it is not the missing piece.

## Current state

- **The one card is `REPORT_CARD.md`.** It contains, in order: **the plan section (the goal cut
  backwards into problems, rendered from `scripts/plan_spec.py`; `PLAN.md` is its readable
  render)**, the B-foundational scorecard (B1 Lawful, B2 Sufficient, B3 Accessible, B4 Decisive,
  B5 Minimal & affordable, B6 Transferable, B7 Robust, B8 Reproducible), the four-leg judge, the
  stage ladder, hygiene, the blocking list and BACKWARD requirements. No other file is a
  measurement surface. `PLAN.md` is structure only (no numbers) — DEC-579.
- **B is not certified, but B3 Accessible is green (DEC-564/565/566/571).** B3.2 measured the fold
  as the wall: exact counter persona/future 0.189/0.173, M0→M1 fold 0.066/0.093, trunk +0.001,
  projection/boundary ~0. B3.3 measured linear access: served 0.4677 vs PATH A 0.4126 (+0.0551),
  so the coarse readout is non-inferior. B3.4/B3.5 tested block-aware serving and the 128/128
  equal-dimension idea: equal_39 reaches 0.172/0.144, equal_128 0.136/0.117, but **no block
  combination exceeds the exact counter alone (0.189/0.173)**, and the shape block alone is
  0.081/0.057. Serving allocation is necessary but not sufficient; the missing signal is the
  **learned shape**. **B3.6 X-rayed the ledger (DEC-571): the EXPOSURE branch fired —
  part_numeric_stats 0.3196 persona ≥ 0.30 while ledger_full dilutes to 0.0172 and block-PCA is
  not the right exposure (0.2012/0.1036); the fine structure lives in the small named parts
  (numeric_stats / event_mass / return_ratio / total_events), so B's serving contract must expose
  the ledger as named blocks. The learned shape remains the weak object (0.0660 persona, below
  the exact counter's 0.1893), and segment structure is absent from the ledger entirely
  (ARI ~0 vs PATH A 0.2685).** B2 sufficiency is green (U 0.9757 served / 1.0769 learned-only,
  DEC-570; 1/5 probes carries headroom, disclosed); DEC-585 closed the 8 remaining B rows on
  cfm_v2k to **17 PASS / 3 FAIL / 3 UNKNOWN** — the 3 reds are B4.1/B4.2 (decisions) and B6.2
  (transfer), the 3 opens are B5.1 (minimality), B6.3 (transfer), B7.2 (robustness/missingness).
- **Macro reframe (DEC-567, pending operator D1–D5).** The product read located the gaps:
  (1) the judge's delivery column ("brain+ledger") is a naive 851-d concat and has never scored
  the 595-d ledger alone — unsupervised geometry dilutes 0.066→0.057 while
  `return_proneness` nearly closes (Δ −0.019, CI straddles 0); (2) the shape state is a
  2.4-effective-dim blob, below its own law target (18.2), with temporal skill (order +0.88)
  but no identity skill (next ≈ unigram); (3) the decision objective was off in the product run
  even though the stream is 100% randomized on the offer arm. Path: O0 ledger X-ray (no
  retrain) → O3/O4 one-axis retrains → O2 `U` → O5 minimality → B+C re-score. Unit of progress
  moves from row-flip to capability delta (O8) if approved.
- **B+C:** RL is green in both columns; supervised (`return_proneness`, `price_sensitivity`),
  unsupervised (geometry trio) and simulator (`argmax_optimum`) are red. The measured structural
  diagnosis — coarse per-customer attributes are present, fine structure is not readable — is on
  the card and in DEC-560/561. B3.2 localizes the fine-structure loss to the fold; B3.3 shows the
  coarse linear readout is non-inferior; B3.4/B3.5 show serving allocation cannot create the
  missing learned shape. The B+C reds are the capability consequence, not a measurement gap.
- **Certification:** S1/S2 are fixed and fresh (DEC-561: producer batch arithmetic, not a random
  co-tenant; 32 GB cap). S3/S4/S5 are fresh. **S6 is PASS** (chain v4: six producers in one
  invocation; re-certified at clean commit `cb7c0d9` under `d5df3a2c` and re-run on every
  behavior move).
- **O3 is answered: REJECT (DEC-573 outcome addendum, landed 2026-11-03).** Attempt 4 ran clean
  to a governed patience stop (574/1650, best val 5.9242, ~3.2 h) and failed the two geometry
  bars: eff_rank 3.665 (bar ≥ 30; cfm_v2k 2.419) and M4_served persona 0.046461 (bar ≥ 0.15;
  cfm_v2k 0.06596); it kept the access bar (mean R² 0.467521 ≥ 0.45) and stability. Frozen branch:
  "1 fails → trunk/objective axis confirmed" — the soft-penalty floor pair bought no rank
  structure (`rank` skill −0.0038 ± 0.0004), refuting H1. No canonical swap: cfm_v2k stays
  canonical; O3's artifact lives at `/tmp/opencode/o3axis_i` with diagnostics under
  `docs/report_cards/experiments/o3axis_i/`.
- **B2 `U` is measured (DEC-570 spec; `measure_sufficiency_law.py` first run on cfm_v2k).**
  Served-object U = 0.9757, learned-only U = 1.0769 — the learned shape alone clears the
  0.5 bar; worst probe next-type. Only 1/5 probes has positive headroom (the four
  regression-style probes: PATH A's wide table beats the population-constant floor, headroom ≤ 0
  — excluded-and-reported per the frozen spec, so the B2 evidence base is 1 probe, disclosed on
  the card). B2.1/B2.2/B2.3 PASS; the U · sufficiency capability axis is green.
- **Dev-rig iteration 1 landed (DEC-576; the minutes loop works).** `next_event` N=500
  (seed 17, snr=1.0): NOT mastered — g 1.147 (bar 0.5), does not beat the 1-gram (CI enc−base
  [−0.016, +0.110]), eff_rank 1.153 at the autotuned dim=32. Controls clean: NEGATIVE reads
  NO_SIGNAL; ADVERSARIAL beats the 1-gram via the echo while its structure claim is NO_SIGNAL —
  the false-PASS trap is caught. The trunk blob reproduces on known-truth data: the bottleneck
  axis is the trunk, confirming the real-stream O3 diagnosis in minutes.
- **Dev-rig iteration 2 landed (DEC-577 outcome addendum; DEC-575 candidate 1 as-shipped):
  NOT MASTERED, branch (iii) taken.** Bilinear `h⊗x` fixed-point ON, everything else
  byte-identical to iteration 1 (launch receipts verified: `bilinear_recurrence: True`, dim=32,
  tag renamed `…o67b1bcb7` → `…o2obe1b6690` by the new `|bl` tag axis): g 1.154482 (iteration 1
  1.14727 — no improvement), does not beat the 1-gram (CI [−0.0160, +0.1132], unchanged),
  eff_rank 1.473 (iteration 1 1.153 — the one axis that moved, +0.32d, recorded as a deviation
  from branch (iii)'s literal antecedent). Controls read as designed: NEG NO_SIGNAL ✓, ADV
  echo-beats-1-gram while structure stays NO_SIGNAL ✓. Candidate 1 as-shipped is negative on the
  toy for the decisive bars; DEC-575's port protocol does not fire; escalation pre-registered as
  DEC-578 (candidate 2: looped / weight-tied nonlinear block, its own DEC, its own
  pre-registration; alternatives — capacity confound dim 32 vs 256, gate tuning, curriculum
  advance — recorded and barred from this verdict).
- **Dev-rig iteration 3 landed (DEC-578 outcome addendum; DEC-575 candidate 2 as-shipped):
   NOT MASTERED, branch (iii) taken.** Looped / weight-tied nonlinear block on the state
   (`F(z) = unitnorm(tanh(W₂ z))` composed T times, T data-derived on a 1e-4 block-change tol
   capped at 4; the state-implicit fixed point solved by iterated affine scans, capped at 8
   passes; β₂ learned), `--set looped_recurrence=true` atop iteration 1's configuration
   (bilinear OFF), everything else identical (launch receipts verified before the read:
   `[set]` line, registry `looped_recurrence: true` at dim=32, tags moved onto the `|lp`
   axis): g **1.181525** (iteration 1 1.14727, iteration 2 1.154482 — no improvement,
   slightly worse), paired CI(enc−base) [−0.0096, +0.1248] still straddles the 1-gram,
   eff_rank **1.248** (iteration 1 1.153, iteration 2 1.473). Controls read as designed: NEG
   NO_SIGNAL ✓, ADV echo-beats-1-gram while structure stays NO_SIGNAL ✓. Candidate 2
   as-shipped is negative on the toy for the decisive bars; the frozen escalation fires:
   DEC-575 candidate 3 (the looped block on the DRIVE side) is declared **DEC-581** and gets
   its full pre-registration in the next forward run. Disclosed in the outcome addendum: the
   convergence receipt hits both pre-registered caps at the final forward (depth 4/4, outer
   8/8, unconverged on the 1e-4 tol — the anticipated "higher stability risk", receipted,
   not UNSTABLE: no non-finite/divergence, lr_collapsed false, portfolio `ok: false` 11/15
   matches the iteration-1/2 toy baseline); and the /tmp launch-script log-naming defect
   (bash `local` expansion — all three per-run stdout logs collapsed into one file,
   summaries survived in `iter3_summary.log`).
- **Dev-rig iteration 4 + the capacity read landed (DEC-581/582 outcome addenda): NOT
   MASTERED on both, branch (iii) on both.** Candidate 3 — the drive-side looped /
   weight-tied block (`--set drive_looped_recurrence=true` atop iteration 1's
   configuration): g **1.169947** (axis baseline iter1 1.14727; ladder 1.14727 →
   1.154482 → 1.181525 → 1.169947 — no improvement), paired CI(enc−base) [−0.0104,
   +0.1198] straddles the 1-gram, eff_rank 1.836 (the ladder high, not converted into an
   NLL gain), controls read as designed (NEG NO_SIGNAL ✓; ADV echo-beats-1-gram while
   structure stays NO_SIGNAL ✓). Cap-hit disclosure: the drive-loop composition ran at
   depth 4/4 with `depth_converged: false` on the final forward — a bounded contraction
   (unit-norm × tanh), NO outer fixed-point loop, structurally distinct from candidate
   2's double cap-hit; not UNSTABLE. Capacity axis (DEC-582: baseline algebra at
   `--set dim=256`): g **1.149121** ≈ iter1's 1.14727 (Δ +0.0019, within noise),
   eff_rank 1.209 (vs 1.153 @32 — 8× capacity adds no usable rank), controls clean. All
   THREE single-axis trunk-algebra candidates are exhausted at dim=32 and capacity is not
   the constraint either: the 1-gram floor is binding on `next_event` at N=500 — the
   per-customer intent structure is real but underdetermined for ANY model at ~29 events
   per customer (the 0.334-nat oracle gap is a finite-sample estimation gap). The
   `next_event` fixture is CLOSED for the dev-rig.
- **Curriculum decision: DEC-583 (2026-11-03).** The next dev-rig structure is `intent`
   (curriculum step 2: the per-customer intent heterogeneity as the DOMINANT structure,
   at a scale where the sample can determine it). The snr knob (1.0 → 0.5) is REJECTED:
   snr scales ALL structure toward the null (the documented difficulty dial), and halving
   it shrinks g's denominator (the oracle-vs-1-gram gap), ill-conditioning the
   speedometer. First implementation check at resume: can the intent fixture be generated
   config-only from the existing simulator, or does it need a new `POSITIVE` entry in
   `rabbit_hole/lab_oracles.py` (a rabbit_hole edit = behavior move = hash move +
   re-stamp)? The full one-axis pre-registration lands at resume under a follow-up DEC.
- **DEC-585 COMPLETE (2026-11-03): all 8 cfm_v2k B rows measured.** Final card: **B 17 PASS /
  3 FAIL / 3 UNKNOWN; 4/8 sections green** (construction, decisions-reporting B4.3, runtime,
  calibration+identity); root FAILED 19/39 (P2 red: B4.1+B4.2; P3 red: B6.2; P4 red: 6 judge
  rows). PASS: B1.2 construction, B4.3 invariance (reporting), B5.2 runtime, B7.1 calibration,
  B8.1 identity. **FAIL: B4.1 dose** (all five arms 0.244–0.254 ≈ uniform chance over 4 levels),
  **B4.2 OPE** (E_ips = T_ips = 152.93 exactly; both ridge policies collapsed to constant level
  1, 446/446 agreement), **B6.2 donor** (y_churn_90 passes every bar, 0.7449 AUC; y_orders_90
  fails its CI; y_value_90 fails E≥raw/fair and CI). One shared diagnosis: the frozen E is
  certified for construction/invariance/runtime/calibration law and churn, but its DECISION
  value on the offer-level family and its regression value on the 90-day value/order targets
  are not yet certified. Commits `20e6b8f`+`8b7ac49`, `401eda2`+`5f4365b`, `5c17758`+`aec6d3f`.
- **DEC-583 implementation check DONE (2026-11-04, read-only).** The `intent` fixture is NOT
  config-only: it requires a 4-file behavior move (`rabbit_hole/lab.py` USE_CASES,
  `rabbit_hole/lab_oracles.py` POSITIVE + `_TRANS`, `looking_glass/lab_gate.py` `_TRANSITION`,
  `rabbit_hole/data_spec.py` log-space stable softmax — the raw intent math underflows to NaN
  at b ≳ 5.8, oracle NaN for H ≥ 2.0). Two findings: (a) within the old math's safe range
  (H ≤ 1.5) the speedometer headroom is 0.05–0.14 nats ≈ B-pair noise → fixture must be
  dominant (H=5.0, headroom 0.256); (b) **the frozen g instrument has a unit bug** —
  `transition_oracle` writes nats but `lab_measure` multiplies by ln2, deflating the oracle
  30% and putting the `g ≤ 0.5` bar BELOW the Bayes floor on every fixture (next_event: bar
  needs 0.8125 vs floor 0.9312; true headroom 0.0483, not DEC-582's 0.334). Correction is a
  new instrument protocol version, applies from intent forward; closed next_event receipts
  stand as recorded.
- **DEC-586 DRAFTED, AWAITING OPERATOR APPROVAL (2026-11-04).** Full one-axis pre-registration:
  fixture `DataSpec("intent", transition=True, intent=5.0, snr=1.0, base_rate=0.3,
  n_customers=500, seed=17)` as-of 2024-09-01; verdict = frozen DEC-576 protocol in corrected
  units (g ≤ 0.5 + beats_base + complete + control reads); branches (i)–(v) frozen; move +
  full cfm_v2k re-stamp (13 producers) with no-op proof. Nothing executed; tree clean at
  `aec6d3f`, behavior `d5df3a2c`.
- **Macro reflection is binding now (DEC-587, operator directive 2026-11-04).** Every forward
  run writes `WHY-NOT / EVIDENCE / GAP / LAYER / THIS RUN` **before** choosing a node and
  re-answers it as `WHY-NOT-CHANGED` at the outcome; `PROMPT_FORWARD.md` §3.0 + the
  `tests/test_prompts.py` ratchet enforce it. The recorded WHY-NOT: the state is a first
  moment (convex combination), the objectives only ask for a mean, and the toy instrument
  could not see shape (0.048 nats headroom vs ~0.1 noise). The shape program is ranked by
  principle alignment — mass/shape algebra split 26/27, centered drive 25 (diagnostic),
  window-distribution objective 25, delta-rule memory 24, KME/signature/HiPPO 23, bounded
  window 22, Titans 20 — with **Phase 0 = the shape battery before any new architecture**
  (histogram/transition/quantile/persona decode from E vs the exact counter, plus a
  mixture-transition toy where pair statistics are necessary).
- **Order-only fixture + before measurement (DEC-588, 2026-11-04).** The `persistence` knob
  makes `P_i = (1−p)e_i + p·π`: the marginal π is identical for every customer, only the
  transition structure differs. `order_shape` (p=0.5, N=500) has 0.2252 nats of order-only
  headroom (vs 0.048 on next_event) and passes the lab gate 6/6. The current encoder on it:
  **g 8.022, nll 0.9632 vs the pooled bigram 0.8524 vs oracle 0.8366, beats_base false,
  eff_rank 1.094** — it is worse than a last-type→next-type table on a fixture whose only
  signal is order. The Layer A diff is **staged** (`/tmp/opencode/order_shape_layerA.patch`,
  139 lines); the tree is restored to behavior `d5df3a2c` so the B receipts stay fresh until
  the change ships with its mandatory re-stamp. The fixture and the
  `order_shape_before.json` diagnostic remain; two scripts-side fixes landed
  (`transition_oracle` now reports the 1-gram marginal H(π); `lab_measure` no longer
  multiplies nats by ln 2). The pair-input attempt (option 1: `x_t = token_t +
  MLP([token_t, token_{t-1}])`) was **negative**: nll 0.9675 vs 0.9632 before; the
  history-transition decode from E moved only 0.354 → 0.383; the last-event decode is
  0.993. E carries the last event and part of the joint, but a **linear** head cannot
  multiply them into `P(next|last)` — the missing piece is the **conditional row made
  linearly readable**. Code reverted (patch `/tmp/opencode/pair_input.patch`); next axis is
  serving the model's own last-tick readout `y_last` (compact, no table) and/or the
  future-transition objective.
- **Shape problem Phase 0 LANDED (DEC-588, 2026-11-04): the strong diagnosis is falsified and
  the problem is narrowed to ORDER.** `shape_battery.json` (cfm_v2k, n_B 1485, paired
  bootstrap): E decodes the **history's** type histogram 0.9943, gap quantiles 0.9766, value
  quantiles 0.9844 — but the **transition matrix only 0.5020** (shuffles ≤ 0.004). On the
  **future** 90-day shape E is within ~0.01 R² of the exact counter (type hist 0.358 vs 0.368;
  transition 0.196 vs 0.204; gap 0.508 vs 0.516; value 0.771 = 0.771). Persona rows are
  contaminated by attribute leakage (segment/tier/region in `event_attributes`; cf. B4.3
  check-2) and are not shape evidence. Read: a mean of one-hot/embedding features *is* a
  histogram, so first-order shape is carried; the missing statistic is the **second moment /
  order**, which is what a mean cannot carry. Geometry is separate: eff_rank 2.4 coexists with
  0.99 histogram decode, so low rank ≠ low information. **Revised priority: the order channel
  (level-2 signature / covariance + Lévy area; delta-rule only if recall is binding) with a
  transition-matrix objective — not the mass/shape split, which is already largely effective
  at the interface.**
- **Route: R4 (certification-first, operator 2026-11-03).** The 11 open B sub-problems
   are the PRIMARY progress track. DEC-585 pre-registered the 8 rows it covers (B1.2,
   B4.1, B4.2, B4.3, B5.2, B6.2, B7.1, B8.1) as 5 cfm_v2k producers with frozen gate
   kinds/bars in `b_spec`; chunk 1 (gates + producers + card branches, no measurement)
   lands first, chunk 2 runs them sequentially (CPU donor/runtime, then GPU
   construction/decision/b_props, one GPU job at a time) and appends the outcome. The
   remaining 3 B rows (B5.1 minimality, B6.3 transfer, B7.2 missingness) each get their
   own DEC after B is green. The dev-rig remainder (R2 intent, R3 port) stays on hold;
   B+C re-scores only after B is green; the operator's pacing target is fast training
   iteration right after B is measured.
- **Plan tree reframe (DEC-584, 2026-11-03).** At the operator's directive the plan is a
   uniform recursive decomposition to ATOMIC problems — no "section"/"leaf" vocabulary, no
   forced layer count, no separate PROGRESS block (progress = the solved-problem count read
   off the tree at every level). All 31 frozen card rows are first-class atomic problem
   nodes; E1 is cut to three dec atoms (E1.a next_event toy — exhausted negative per
   DEC-581/582; E1.b intent toy — DEC-583, on hold; E1.c port under DEC-572's bars). Census:
   **39 atomic problems** (31 measured · 3 DEC-frozen · 5 v1) under **19 interior problems**.
   Verdicts a no-op: root FAILED, 14/39 green (the denominator now includes the dec/v1
   atoms), B 12 PASS / 0 FAIL / 11 UNKNOWN, E1 stays FRONTIER, P5/G stay UNKNOWN under an
   all-green card scope.
- **Operating prompts are two-scale now (DEC-580, operator directive).** `PROMPT_FORWARD` requires
  `NODE` / `ROUTE` / `MOVE` before every run and `MOVEMENT` after it (a run that moves no node is
  a diagnosis, not progress); `PROMPT_BACKWARD` attacks the tree first (necessity, placement,
  gate-kind misuse, route honesty, recomputed statuses, tree sufficiency) and has two modes:
  scoped NODE reviews after a node changes, and the FULL completion review (triggered when the
  card is ENCODER DONE v1). Completion requires the card's plan root `SOLVED` (E1 port + v1),
  not merely a green encoder battery. `AGENTS.md` aligned; `tests/test_prompts.py` is the drift
  ratchet.
- **Products:** the canonical full-size products are at `/home/austin_whaley/wonderland_work/cfm_v2k`
  (fs_full, tag `…b64…o4c3b5899`, trained under behavior `d0ac576c`); the stale `0f881d99`
  artifacts are backed up at `cfm_v2k_stale_0f881d99`. Code behavior is now `d5df3a2c`
  (DEC-573: the `resolve_cfm` batch-pinning fix → `d2ad6f60`; autotune defect class 9 →
  `6a8e2663` — inert on the real stream, disclosed; DEC-577 identity fixes: `bilinear_recurrence`
  joins the tag signature `|bl` axis and the warm-start architecture comparison → `4053d2f9`;
  DEC-578: the looped/weight-tied block (candidate 2) joins the trunk as the `|lp` tag axis +
  warm-start algebra refusal + construction guard against two trunk algebras at once →
  `905b4f88`; DEC-581: the drive-side looped block (candidate 3) joins the trunk as the
  `|dl` tag axis + the drive-loop composition receipt → `d5df3a2c`, default-OFF inert).
  cfm_v2k's frozen artifacts are unchanged; every canonical receipt is re-stamped under
  each behavior move (latest: `d5df3a2c`, S6 chain at clean commit `b5566ca` — all six
  checks OK, deterministic 251-line render — plus the four B-row producers on cfm_v2k
  defaults, RC=0; the card diff vs the `905b4f88` render is header/hash/commit/timing
  lines only: 12 PASS / 0 FAIL / 11 UNKNOWN, plan root 14/31 unchanged — the default-off
  no-op proof; commits `b5566ca` + `a195fef`).

## Closed front (history; see DECISIONS for the record)

The unsupervised-geometry front is closed at the rung on both axes: objective (DEC-552/554) and
construction (DEC-555/556). DINO, SwAV, contrast removal, VICReg and block-serving were all
falsified at the frozen bar. The mass metric's drop localizes at M0→M1 (the `w_counter` fold); at
the rung the exact counter column cleared the bar, but at full size B3.2 measures it at 0.189
persona vs PATH A 0.387 — necessary, not sufficient. The serving contract has now been measured
(B3.4/B3.5): allocation helps but cannot exceed the counter, and the shape block is weaker. The
next question is the learned shape, not another consumer tuning pass.

## Next 3 actions

1. **Order-channel chunk (DEC-588 Phase 1, the active program).** Apply the staged Layer A
   diff (`/tmp/opencode/order_shape_layerA.patch`); then the cheap axis first: serve the
   model's own last-tick readout `y_last` as a compact block (the per-step conditional, no
   table, still a linear consumer) and/or add the future-transition objective to raise the
   joint. Run the order fixture (must beat the 0.8524 bigram; the pair input alone was
   0.9675) and the history-transition decode (must rise above the measured 0.383 on the
   fixture / 0.502 on cfm_v2k). Then the full cfm_v2k re-stamp (S6 chain + 13 B-row
   producers, GPU-serialized, no-op proof for the B rows).
2. **Re-run the shape battery after the change** (same protocol, same split): the target is
   history-transition decode above the measured **0.502** and future transition at or above
   the exact counter, with the first-order rows not regressing. This is the Phase-1
   acceptance; a failure is recorded and reverted, not narrated.
3. **Then the real-stream rows.** Only after the battery and the order fixture move: the
   B3.2-lineage shape probes, B6.2's 90-day value/orders, and B4.1/B4.2 if the order channel
   is what the level-conditional response needs. The 3 remaining B UNKNOWNs (B5.1, B6.3,
   B7.2) keep their own DECs; lab receipts never flip a B row (DEC-576). DEC-586 (intent
   fixture) stays pending operator approval and is not on this path.

## Operating facts

- **Repo:** `/home/austin_whaley/Documents/Repos/wonderland`, branch `foundation-rebuild`, venv `.venv`.
- **Testbed:** Layer A v2 demo at `/home/austin_whaley/wonderland_work/demo_v2/`.
- **Card:** `.venv/bin/python scripts/report_card.py` → `REPORT_CARD.md` (the only measurement surface).
- **Resources:** GB10, 121.63 GiB unified memory, 20 CPU cores, 1.5 TB free disk, CUDA 13.0.
  **Project maximum: 32 GB** (operator 2026-11-02). `wonderland.slice` = `MemoryMax=34359738368`;
  `safe_run` reports 32.0 GB. The slice bounds CPU/host memory only; GPU/unified allocations are
  capped by `looking_glass/gpu_budget.py` and by `derive_batch`'s memory-derived budget.
- **Canonical launcher:** `scripts/safe_run.py` (slice containment + receipt). Raw harness
  placement outside the slice is a policy breach.
- **Test tier:** fast tier green on the committed tree.

## Open operator decisions

**Decided 2026-11-02 (operator: "ok do what needs to be done"):** D2, D3, D4, D5 are all
accepted — the capability sprint is the unit of progress, O0 runs now, O3 law-ablation is
authorized as labeled diagnostics, and the O2 `U` spec is the DEC-570 pre-registration
(defaults as proposed: five self-supervised probe types, fixed standardized-ridge capacity
shared by both arms, gap normalized by headroom, bar U ≥ 0.5 on the served object with
learned-only alongside). Pre-registrations: DEC-569 (O0 arms/consumers/branches), DEC-570
(U spec). D1 stays deferred with B+C.

**Decided 2026-11-03 (operator pivot: "make ANY foundation model before we make THE
foundation model"):** DEV-RIG loop pre-registered as DEC-576 — A↔B→C iterations on
known-truth lab fixtures (`rabbit_hole.data_spec` + `lab_oracles` + `lab_gate` +
`head_battery`), curriculum one structure axis at a time, N=500 dev size, binding
discipline: lab receipts are a development instrument and never flip a B-row verdict on
the main card. O3 attempt 4 then LANDED and was REJECTED under its frozen bars (eff_rank
3.665 < 30; persona 0.046461 < 0.15; R² and stability kept) — no canonical swap, DEC-575
triggered; the trunk candidate is exercised on the dev rig first (iteration 2).

**Decided 2026-11-03 (late; operator: "give me a review of your work" then "I need to
understand the long-term plan"):** (a) work review delivered; (b) iteration 2 (DEC-575
candidate 1 as-shipped, bilinear ON) LANDED NOT MASTERED — branch (iii) taken, candidate 1
negative on the toy for the decisive bars (g 1.154 ≈ iteration 1's 1.147; 1-gram CI
unchanged; eff_rank 1.153 → 1.473 recorded; controls clean) — escalation to DEC-578 /
candidate 2 pre-registered next; (c) **PLAN.md reinstated (DEC-579)** as the goal-side
decomposition, generated from `scripts/plan_spec.py`, with the card rendering the same tree
(structure + verdicts) — direction is PLAN.md → REPORT_CARD.md, the plan defines no gate, and
the root cannot read SOLVED while the DEC-frozen trunk leaf and the v1 leaves are unmeasured.

**Decided 2026-11-03 (operator: certification first):** (a) the 11 open B sub-problems are
the PRIMARY progress track (ROUTE R1 → R4); the dev-rig remainder (`intent` under DEC-583,
port under R3) is ON HOLD; (b) progress is read OFF the plan tree — the solved-problem count
at each level — not from a separate PROGRESS block or milestone ladder; (c) the tree must be
a recursive decomposition to ATOMIC problems at whatever depth it takes: no forced layer
count, no "section"/"leaf" vocabulary (recorded as DEC-584 at the reframe).
