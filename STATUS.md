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
- **B is not certified.** B1/B3/B6 have current evidence only through the S3/S4/S5 stage blocks.
  B2 (sufficiency `U`), B4 (B-only decision), B5 (minimality/cost), B7 (robustness breadth) and
  B8 (artifact identity) are missing or legacy; each row on the card names the producer that must
  be run. Legacy encoder receipts are explicitly UNKNOWN, never evidence.
- **B+C:** RL is green in both columns; supervised (`return_proneness`, `price_sensitivity`),
  unsupervised (geometry trio) and simulator (`argmax_optimum`) are red. The measured structural
  diagnosis — coarse per-customer attributes are present, fine structure is not readable — is on
  the card and in DEC-560/561. The exact counter and the identified oracle show the information
  exists upstream; the stage decomposition is the missing X-ray.
- **Certification:** S1/S2 are fixed and fresh (DEC-561: producer batch arithmetic, not a random
  co-tenant; 32 GB cap). S3/S4/S5 are fresh. S6 still fails until chain v4 runs the whole chain in
  one invocation.
- **Products:** the canonical full-size products are at `/home/austin_whaley/wonderland_work/cfm_v2k`
  (fs_full, tag `…b64…o4c3b5899`, behavior `d0ac576c`); the stale `0f881d99` artifacts are backed
  up at `cfm_v2k_stale_0f881d99`.

## Closed front (history; see DECISIONS for the record)

The unsupervised-geometry front is closed at the rung on both axes: objective (DEC-552/554) and
construction (DEC-555/556). DINO, SwAV, contrast removal, VICReg and block-serving were all
falsified at the frozen bar. The mass metric's drop localizes at M0→M1 (the `w_counter` fold); the
exact counter column clears the bar but the served composite dilutes it. The full-size question
remains open and is now a B-measurement question (B2/B3), not a consumer-tuning question.

## Next 3 actions

1. **B FIRST — run the missing B producers, one axis per run, pre-registered:** B3.2 stage
   decomposition (M0→M1→served), B4.1 B-only decision, B2.1 worst-case sufficiency `U`, B5.1
   minimality, B7.2 robustness, B6.3 broad transfer, plus the legacy re-runs (B1.2/B3.3/B4.2/
   B5.2/B6.2/B7.1/B8.1). Each writes a source receipt; the one card renders it. B is the gate;
   do not start B+C work while B is uncertified.
2. **B+C SECOND — after B is certified:** run chain v4 (one invocation, 32 GB cap) for S6/B8 and
   the full re-stamp, then close the remaining B+C reds (supervised, unsupervised, simulator).
   B+C is the consequence, not the current claim.
3. **PROJECT v1 THIRD — after B+C:** the full stack (rabbit_hole causal stream, every plugin gate,
   red_king/red_queen/white_queen, caterpillar, live randomized holdout). The trunk/state modeling
   axis is **DEC-564**; BACKWARD runs only when the card is all green.

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
