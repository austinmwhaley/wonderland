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
card as a reported consequence, not the near-term target. Execution is pending operator decisions
D2–D5 (D1 deferred with the B+C judge re-aim).

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

1. **Operator decisions D2–D5 (DEC-567/568, B-only).** The capability sprint is queued on B's own
   rows; execution is gated on: D2 (O8 sprint as the unit of progress + card top line = B
   capability), D3 (O0 ledger X-ray, no retrain), D4 (O3 law-ablation as labeled diagnostics),
   D5 (O2 `U` spec: family, fixed capacity, declared bar). D1 (B+C judge delivery-column re-aim)
   is deferred with B+C.
2. **O0 ledger X-ray (no retrain, ~30 min)** once D3 lands: score the 595-d ledger alone, its
   parts, and ledger block-PCA (fit on A) under the frozen B3.2/B3.4 consumers, against
   {counter 39, shape 256, PATH A}. Pre-registered branches decide whether B's served object
   (ledger included) carries the fine structure — an exposure fix inside B's serving contract —
   or whether the additive family itself is the limit (O3/O4 territory).
3. **First capability retrains, one axis each, accepted on B rows:** O3 (law-ablation: the state
   is a 2.4-dim blob below its own 18.2 target; acceptance = eff_rank ≥ 30 + shape-only geometry
   movement + B2 `U`) then O4 (decision objective on the stream's own 100%-randomized offer arm,
   which the product run left off; acceptance = **B4.1** B-only action-conditional probe). Then
   O2 `U` on the winning object, O5 minimality (B5.1), remaining B rows, chain v4 for the
   re-stamp. **B+C re-scores only after B is green; project v1 THIRD; BACKWARD only at all-green.**

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

Pending (DEC-567/568, stop-and-ask per certification rule 8, **B-only**):
- **D2:** adopt the capability sprint (O8) — unit of progress = a B-row capability delta, card
  top line = B capability (U · shape-only geometry · decision-shape · minimality · transfer ·
  robust · reproducible); B+C stays a reported consequence.
- **D3:** pre-register and run O0 (ledger X-ray, no retrain, ~30 min) as a B2/B3 measurement.
- **D4:** run O3 law-ablation retrains as labeled diagnostics (invariants as hypothesis, not
  non-negotiable), recording the Lawful↔Sufficient trade both sides; acceptance on B rows.
- **D5:** declare the O2 `U` spec — probe family, fixed consumer capacity, declared bar.
- **D1 (deferred with B+C):** re-aim the judge's delivery column at the doctrine's served object.
  Held until B's rows are green.

Standing: push requires explicit permission.
