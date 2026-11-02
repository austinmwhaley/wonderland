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
(B1–B8) before the B+C consequences.

## Current state

- **The one card is `REPORT_CARD.md`.** It contains, in order: the B-foundational scorecard
  (B1 Lawful, B2 Sufficient, B3 Accessible, B4 Decisive, B5 Minimal & affordable, B6 Transferable,
  B7 Robust, B8 Reproducible), the four-leg judge, the stage ladder, hygiene, the blocking list and
  BACKWARD requirements. No other file is a measurement surface.
- **B is not certified, but B3 Accessible is green (DEC-564/565/566).** B3.2 measured the fold
  as the wall: exact counter persona/future 0.189/0.173, M0→M1 fold 0.066/0.093, trunk +0.001,
  projection/boundary ~0. B3.3 measured linear access: served 0.4677 vs PATH A 0.4126 (+0.0551),
  so the coarse readout is non-inferior. B3.4/B3.5 tested block-aware serving and the 128/128
  equal-dimension idea: equal_39 reaches 0.172/0.144, equal_128 0.136/0.117, but **no block
  combination exceeds the exact counter alone (0.189/0.173)**, and the shape block alone is
  0.081/0.057. Serving allocation is necessary but not sufficient; the missing signal is the
  **learned shape**. B2 (sufficiency `U`), B4 (B-only decision), B5 (minimality/cost), B7
  (robustness breadth) and B8 (artifact identity) remain missing or legacy.
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

1. **B FIRST — B4.1/B2.1, the learned shape:** B3 is green as accessibility, and B3.4/B3.5 show
   serving allocation cannot exceed the exact counter. The missing signal is the learned shape, so
   the next increment is the B-only action-conditional probe (B4.1) and the worst-case sufficiency
   `U` (B2.1); if the probe shows the information is absent at B, the objective is the next
   capability change. No more fold/block tuning.
2. **B, continued — the remaining rows:** B1.2/B5.1/B5.2/B6.2/B6.3/B7.1/B7.2/B8.1, one axis per
   run; the one card renders each. B3 is the first green section; make the others current before
   claiming B.
3. **B+C SECOND — after B is certified:** run chain v4 (one invocation, 32 GB cap) for S6/B8 and
   the full re-stamp, then close the remaining B+C reds (supervised, unsupervised, simulator).
   **Project v1 THIRD** after B+C; the trunk/state axis is **DEC-567**; BACKWARD only at all-green.

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

None pending. The queue is pre-registered: B producers → B+C (chain v4 + remaining reds) →
project v1 / DEC-564 trunk/state → BACKWARD at all-green. Standing: push requires explicit permission.
