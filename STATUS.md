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
  (ARI ~0 vs PATH A 0.2685).** B2 (sufficiency `U`), B4 (B-only decision), B5 (minimality/cost),
  B7 (robustness breadth) and B8 (artifact identity) remain missing or legacy.
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
  co-tenant; 32 GB cap). S3/S4/S5 are fresh. S6 still fails until chain v4 runs the whole chain in
  one invocation.
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
  artifacts are backed up at `cfm_v2k_stale_0f881d99`. Code behavior is now `4053d2f9`
  (DEC-573: the `resolve_cfm` batch-pinning fix → `d2ad6f60`; autotune defect class 9 →
  `6a8e2663` — inert on the real stream, disclosed; DEC-577 identity fixes: `bilinear_recurrence`
  joins the tag signature `|bl` axis and the warm-start architecture comparison → `4053d2f9`;
  DEC-578: the looped/weight-tied block (candidate 2) joins the trunk as the `|lp` tag axis +
  warm-start algebra refusal + construction guard against two trunk algebras at once →
  `905b4f88`). cfm_v2k's frozen artifacts are unchanged; every canonical receipt is
  re-stamped under `905b4f88` (S6 chain at clean commit `4bb221b` — all six checks OK,
  deterministic 251-line render — plus the four B-row producers on cfm_v2k defaults, RC=0;
  the card diff vs the `4053d2f9` render is header/hash lines only: 12 PASS / 0 FAIL /
  11 UNKNOWN, plan root unchanged — the default-off no-op proof; commits `8346a1c` +
  `5963462`).

## Closed front (history; see DECISIONS for the record)

The unsupervised-geometry front is closed at the rung on both axes: objective (DEC-552/554) and
construction (DEC-555/556). DINO, SwAV, contrast removal, VICReg and block-serving were all
falsified at the frozen bar. The mass metric's drop localizes at M0→M1 (the `w_counter` fold); at
the rung the exact counter column cleared the bar, but at full size B3.2 measures it at 0.189
persona vs PATH A 0.387 — necessary, not sufficient. The serving contract has now been measured
(B3.4/B3.5): allocation helps but cannot exceed the counter, and the shape block is weaker. The
next question is the learned shape, not another consumer tuning pass.

## Next 3 actions

1. **DEC-578 outcome recorded (this chunk, 2026-11-03).** Dev-rig iteration 3 NOT MASTERED,
   branch (iii) taken: candidate 2 as-shipped is negative on the toy (g 1.181525 vs iteration
   1 1.14727 / iteration 2 1.154482 — no improvement; paired CI [−0.0096, +0.1248] still
   straddles the 1-gram; eff_rank 1.248; NEG/ADV controls read as designed). Convergence
   receipt hits both pre-registered caps at the final forward (disclosed, not UNSTABLE); the
   /tmp launch-script log-naming defect is disclosed. The card's frontier readings gain the
   iter3 row; the plan's R1 evidence pointer moves to the DEC-581 run; no behavior move.
2. **DEV-RIG iteration 4 = DEC-575 candidate 3 (pre-register as DEC-581 FIRST).** The looped /
   weight-tied block on the DRIVE side (the block applied to the token drive; the state-side
   algebra stays affine), on `next_event` N=500, one axis, with its own frozen verdict +
   a-priori branches (iteration 3's branch (iii) is the standing escalation; the recorded
   alternatives — dim-256 capacity test on the toy, gate tuning, curriculum advance — stay
   barred from that verdict). The DEC must carry `NODE: P2/E1`, `ROUTE: R1`, `MOVE:` the toy
   reading, and the outcome must record `MOVEMENT:`. Also fix the /tmp launch-script
   log-naming defect (bash `local` expansion).
3. **Then the curriculum resumes: `intent` (iteration 5), then the port.** When BOTH
   `next_event` and `intent` are mastered with clean controls, DEC-576's port protocol fires
   (`ROUTE` R3): ONE pre-registered real-data run of the winning trunk candidate under DEC-575's
   frozen gates (the four O3 bars + the B1 canary, atop cfm_v2k). Queue behind it, unchanged:
   O4 decision law (DEC-574), O5 minimality, B6.3, B7.2, dev-rig iterations 6+; B+C re-scores
   only after B is green on the real stream; project v1 third; BACKWARD at milestones only
   (DEC-580's FULL mode).

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
