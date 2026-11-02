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
runs O0 first. D1 stays deferred with the B+C judge re-aim.

## Current state

- **The one card is `REPORT_CARD.md`.** It contains, in order: the B-foundational scorecard
  (B1 Lawful, B2 Sufficient, B3 Accessible, B4 Decisive, B5 Minimal & affordable, B6 Transferable,
  B7 Robust, B8 Reproducible), the four-leg judge, the stage ladder, hygiene, the blocking list and
  BACKWARD requirements. No other file is a measurement surface.
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
- **Products:** the canonical full-size products are at `/home/austin_whaley/wonderland_work/cfm_v2k`
  (fs_full, tag `…b64…o4c3b5899`, behavior `d0ac576c`); the stale `0f881d99` artifacts are backed
  up at `cfm_v2k_stale_0f881d99`.

## Closed front (history; see DECISIONS for the record)

The unsupervised-geometry front is closed at the rung on both axes: objective (DEC-552/554) and
construction (DEC-555/556). DINO, SwAV, contrast removal, VICReg and block-serving were all
falsified at the frozen bar. The mass metric's drop localizes at M0→M1 (the `w_counter` fold); at
the rung the exact counter column cleared the bar, but at full size B3.2 measures it at 0.189
persona vs PATH A 0.387 — necessary, not sufficient. The serving contract has now been measured
(B3.4/B3.5): allocation helps but cannot exceed the counter, and the shape block is weaker. The
next question is the learned shape, not another consumer tuning pass.

## Next 3 actions

1. **O0 ledger X-ray — DONE (DEC-571):** EXPOSURE branch fired (part_numeric_stats 0.3196
   persona ≥ 0.30; ledger_full dilutes to 0.0172; block-PCA is not the right exposure). B's
   serving contract must expose the ledger as named blocks; DEC-570's U arm is determined by the
   measurement. The learned shape remains the weak object → the retrain axes are still on.
2. **O3 law-ablation retrain (DEC-572 pre-registration, one axis):** ablate the
   isometric-band/rank invariant family on the 2.4-dim blob state, accepted on B rows:
   eff_rank ≥ 30 in the retrain's own receipt AND shape-only geometry (B3.2 M4_served persona,
   frozen consumer) ≥ 0.15 AND coarse linear access (B3.3 served mean R²) ≥ 0.45. GPU allowed
   under the 32 GB ceiling (operator 2026-11-02: use whatever is best for the task). Then O4
   (decision objective on the stream's own 100%-randomized offer arm; acceptance = **B4.1**).
3. **O2 `U` on the winning object (block-aware exposure per DEC-571, spec per DEC-570, bar
   U ≥ 0.5 with learned-only alongside), then O5 minimality (B5.1), remaining B rows, chain v4
   for the re-stamp. B+C re-scores only after B is green; project v1 THIRD; BACKWARD only at
   all-green.**

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
