# STATUS — wonderland

*Live state only. History lives in `DECISIONS.md` (append-only) and `git log`.
Rewritten 2026-11-02 (DINO falsified DEC-552; SwAV falsified DEC-554; composite-construction front). If this file grows past ~150 lines, it is becoming a log again.*

## Macro

**Goal (Layer 0, now):** one frozen, self-supervised, schema-agnostic encoder per event stream
whose embeddings are a **universal donor** — supervised, unsupervised, RL and simulator consumers
all get signal from it through simple heads.

**Definition of progress:** **PATH B (event stream → frozen encoder → simple head) > PATH A
(wide table → GBM)**, with paired CIs and controls. Two numbers per leg: `brain alone` =
**progress**, `brain + ledger` = **delivery** (non-inferiority, 0.01 R² margin). Ledger points
are never a brain win.

**Stage:** Layer 0 — encoder not yet mastered. Layer 1 (certification, `U`) is not started.

**Progress:** **2/8 leg-columns, 6/6 stages, 0 UNKNOWN, 0 open backward requirements.**

## Micro — current state (measured)

Card generated at commit `a32bbee` · behavior hash `0f881d99d9d8203e` (H2). Judge receipts:
800-customer rung (`j_base800`) plus full-size `cfm_v2k` where noted.

| Leg | brain alone | brain + ledger |
|---|---|---|
| supervised | ❌ FAIL — `return_proneness` **full-size only** (0.4145 vs PATH A 0.4879; passes at the 800-rung: 0.4726 vs 0.3826) | ❌ FAIL — `price_sensitivity` delivery column (random control ≥ brain) + `return_proneness` |
| unsupervised | ❌ FAIL — `persona_geometry` 0.14 vs 0.404, `future_geometry` 0.10 vs 0.14, `segment_structure` 0.001 vs 0.309 (rung) | ❌ FAIL — same rows outside the delivery margin |
| **rl** | ✅ **PASS** (DEC-545 control substitution) | ✅ **PASS** |
| simulator | ❌ FAIL — `argmax_optimum` brain +0.0026, CI includes 0 | ❌ FAIL — same row, delivery margin |

Stages S1–S6: **6/6 PASS** (receipts stamped at `a32bbee`). Hygiene H1–H5: **5/5 PASS.**
BACKWARD not yet run — it runs only against an all-green card.

## The one active front: unsupervised geometry (entanglement)

Diagnosis chain (DEC-548…552, all rung measurements):

- **Temporal drift is not the axis** (DEC-548): `proj(y)`-based contrast falsified.
- **Attribution probe** (CPU, validates the judge to the 3rd decimal): the served donor decodes
  persona/future fields **linearly as well as PATH A** (probe R² 0.583 vs 0.587; 0.364 vs 0.307)
  yet pairwise distance fidelity is 0.14 vs 0.404 and segment ARI 0.001 vs 0.309. The information
  is PRESENT; the **geometry is entangled** (principle 7). Trunk and counter alone each decode
  ~0.32; the fold combines to 0.583 — the 256-direction ARRANGEMENT is what scrambles it. The
  boundary is an isometry, so the fix cannot live in the readout.
- **Falsified cheaply, in sequence:** contrast-removal (persona 0.1295), VICReg isotropy (0.1400) —
  both within the rung noise band 0.1295–0.1400. No covariance-shaping objective moves it.
- **DINO self-distillation FALSIFIED (DEC-550/552):** two dose runs on the served composite.
  Run 1 (B=16, 266 governed steps, 8,512 views): persona 0.1520 (~2σ above the noise band
  0.1295–0.1400), bar not met. Run 4 (B=64, `safe_run` slice-conformant, deterministic): the
  governor stopped at step 266 on patience, but the SAVED state is the best-val step-38 state
  (4,864 views, teacher 14% progressed) — the composite val, now including the self-referential
  dino term, drifted monotonically 4.93 → 6.04. Judge: persona **0.1490** (bar ≥ 0.217),
  segment-ARI **−0.0002** (bar ≥ 0.095), supervised guard clean (worst churn_day −0.0298) →
  **falsified** per the pre-registered clause. Dose-response across runs 1 and 4 is FLAT
  (0.1520 vs 0.1490 at 1.75× the view dose): DINO's structure pressure saturates at ~+2σ,
  ~1/6 of the bar's +0.08. Pre-launch audit (DEC-551): the running code was the pre-registered
  mechanism; bundle now reverted (`78f8633`), `dino_*` fields remain inert committed defaults.
- **Decisive attribution fact (run-4 probe, four arms, judge protocol):** persona distance
  fidelity lives in the **EXACT COUNTER arm — 0.2162, i.e. 99.6% of the frozen bar (0.217)**.
  Trunk alone 0.0112; served composite 0.149 (the blend dilutes the mass arm); PATH A 0.404.
  Entanglement persists: served probe-persona R² 0.585 ≈ PATH A 0.587 while ARI ≈ 0.
- **Harness failures closed (runs 2–3, post-mortem `12a1f54`):** both died to the watchdog's
  live-fire test sweep (out-of-slice harness placement + my own pytest). The sweep test now skips
  while a producer is live (verified against run 4); jobs launch through `scripts/safe_run.py`
  into the 48G slice (run-4 receipts: slice cap 48.0 GB, seatbelt clamp 45.06 GB, peak 13.94 GiB).

**SwAV prototypes FALSIFIED (DEC-553 pre-reg, DEC-554 record):** run 5 (B=64, k=9,
safe_run, ~15 min, slice peak 13.93 GiB): the governor-exclusion worked as pre-registered
(receipt fired; swav trained at weight 1.0, scale 8.69; sinkhorn row resid 1.2e-07) but the
judge found persona **0.1320** (bar ≥ 0.217; Δ −0.0049 vs base — zero movement within noise)
and segment-ARI **−0.0034** (bar ≥ 0.095); supervised guard clean; probe-persona R² 0.5825
(collapse guard ≥ 0.40 met — entanglement persists, no collapse). Two distinct structure
pressures on the served composite (DINO agreement, SwAV clustering) plus the earlier
covariance shapers have all failed to move it. **Interpretive correction recorded in DEC-554:**
the baseline's own govern trajectory shows the identical drift + patience stop (best = first
eval) — the dose ceiling is a BASELINE property of the 11-objective portfolio at this rung, not
a self-referential-selection artifact; j_base800 was judged at the same effective dose.
**Decisive attribution (run-5 probe, 4 arms):** the mass arm — the EXACT stored counter
column, no learned weights — now **CLEARS the bar: 0.2174 ≥ 0.217** (0.2162 on run 4; it is a
data-side quantity, stable across runs). The served composite dilutes it to 0.1320: the fold
(`w_counter`) + shared `proj` + boundary is many-to-one and destroys the mass half's pairwise
structure.

**COMPOSITE CONSTRUCTION (DEC-555, pre-registered `512af35`):** stage-1 decomposition probe
(measured, CPU) put **100% of the drop at M0→M1 — the `w_counter` fold alone** (0.2174 →
0.1279); addition of h, shared proj, and the boundary each contribute ~0 (M3 = M4 EXACT —
isometry confirmed on the judge protocol). M4-vs-stored 0.0031 gap resolved by mechanism, not
numerics: with `serve_unfaded` the graded `state_embeddings` are materialized PRE-fade while
`customer_state.state` is rewritten post-fade in place. Stage 2 (pre-fixed lever: native-mass
block serving, E = [z(current 256-d pipeline) | z(exact 39-d mass)], split-A block scales) is
IMPLEMENTED (3 files: config flag, `set_serve_mass_block`/`serve_mass_block` + no-op guard in
`serve_project`, `donor_states` branch + split-A fit persisted as the per-tag sidecar
`serve_massblock_<tag>.npz` (DEC-213 pattern) + `load_frozen_encoder` hook + day-1-close fit in
both serve branches) — smoke passed every pre-registered criterion (exact z-score blocks,
sidecar reload reproduces the served vectors to 0.0) and caught one real bug (stack across
mismatched widths in the finiteness check — fixed). The verdict run = in-repo `products`
rebuild of the run-5 artifacts, no retraining.

**Tree:** all gates green — ruff check/format, principle audit strict rc=0 (ratchet 67),
fast tier rc=0. SwAV bundle REVERTED (DEC-554); kept, pre-registered as a training-regime fix:
the governor-selection exclusion in `cfm_training.py` (inert unless "swav" is in objectives;
the general rule — self-referential terms train, the predictive portfolio selects — now covers
any future EMA-teacher objective). `swav_*` config fields reverted with the bundle. The
DEC-555 stage-2 bundle (config flag + model serve transform + state fit/sidecar/donor_states
branch, 3 files) is UNCOMMITTED until the verdict (same protocol as DINO/SwAV: falsify →
revert).

## Next 3 actions

1. **Stage-2 verdict instruments (IN FLIGHT, CPU):** the in-repo `products` rebuild of the
   run-5 artifacts with `serve_mass_block=true` is DONE (EXIT=0, ~3 min: `[mass-block] scales
   fit on split-A N=562 -> 295-d served E`, sidecar persisted, 800 rows, all finite). Judge
   (`judge_r6.py`, frozen DEC-555 bars vs j_base800) + 5-arm scorecard probe
   (`probe_stage2.py`: served-old / served-new / M0 c / H0 trunk / PATH A — instrument
   acceptance: served-old ~0.1320, M0 ~0.2174, PATH A ~0.4038) are running.
2. **Verdict per the pre-registered decision table:** bar met (persona ≥ 0.217 OR segment-ARI
   ≥ 0.095; no supervised row < −0.03; probe-persona R² ≥ 0.40) → commit the bundle, full-size
   adoption run (re-estimate from measured s/step + co-tenancy; the DEC-554 dose-ceiling fact
   applies to any retrain), S6 chain, card regeneration. Bar missed + guard clean → record
   DEC-556, revert, front → FULL-SIZE re-measurement. Guard breaks → the coupling is real:
   record, retain the fold, front → full-size re-measurement.
3. After the geometry front: re-measure `return_proneness` (full-size 0.4145 vs 0.4879) and
   simulator `argmax_optimum` (+0.0026, CI⊃0) → card 8/8 → run BACKWARD.

## Operating facts

- **Repo:** `/home/austin_whaley/Documents/Repos/wonderland`, branch `foundation-rebuild`, venv `.venv`.
- **Testbed:** Layer A v2 demo (certified reference) at
  `/home/austin_whaley/wonderland_work/demo_v2/` (stream, relational/, truth, manifest).
- **Artifacts:** rung/scratch products under `/tmp/opencode/`; full receipts only via the S6
  chain into `/home/austin_whaley/wonderland_work/cfm_v2k/` (behavior `4ea9684f506ddbf1` there is
  stale vs current code — receipts regenerate, never relabel).
- **Card:** `python3 scripts/report_card.py` → `REPORT_CARD.md`.
- **Resources:** GB10 unified memory; `wonderland.slice` = 48 GB containment ceiling (operator
  directive, 2026-11-01; gpu_budget walks /proc/self/cgroup to it — verified in run 4's receipt);
  per-job seatbelt ≈ 41–43.7 GB today. LLM co-tenant (this session's server) shares the GPU:
  heavy-turn windows measurably slow training (run 3: 21 s/step vs ~5–6 s/step idle).
  **Canonical job launcher is `scripts/safe_run.py`** (slice containment + receipt); raw
  harness placement lands outside the slice and is a policy breach.
- **Test tier:** fast tier green on the committed tree. The watchdog sweep test skips while a
  live producer runs — running pytest alongside training is now safe.

## Open operator decisions

None pending. The queue is pre-registered (composite construction → [if falsified: re-attribute
via the 4-arm scorecard] → return_proneness / argmax_optimum re-measurement). Standing: push
requires explicit permission.
