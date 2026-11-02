# STATUS — wonderland

*Live state only. History lives in `DECISIONS.md` (append-only) and `git log`.
Rewritten 2026-11-02 (full-size re-measurement DEC-558: re-stamp 5/6, S1+S2 co-tenant OOM;
front = chain re-run, then argmax_optimum). If this file grows past ~150 lines, it is becoming a
log again.*

## Macro

**Goal (Layer 0, now):** one frozen, self-supervised, schema-agnostic encoder per event stream
whose embeddings are a **universal donor** — supervised, unsupervised, RL and simulator consumers
all get signal from it through simple heads.

**Definition of progress:** **PATH B (event stream → frozen encoder → simple head) > PATH A
(wide table → GBM)**, with paired CIs and controls. Two numbers per leg: `brain alone` =
**progress**, `brain + ledger` = **delivery** (non-inferiority, 0.01 R² margin). Ledger points
are never a brain win.

**Stage:** Layer 0 — encoder not yet mastered. Layer 1 (certification, `U`) is not started.

**Progress:** **re-stamp 5/6 + argmax protocol v2 adopted (DEC-558/560).** The full-size
re-measurement is DONE (fs_full: dose ceiling 10.45 epochs, batch 64, tag `…b64…o4c3b5899`;
products canonical at `cfm_v2k`; old artifacts at `cfm_v2k_stale_0f881d99`). S3/S4/S5 + 4
judge legs + provenance are fresh at `d0ac576c` and working; S1+S2 died twice on co-tenant
unified-memory spikes (CUDA OOM, then SIGKILL at 135 s) → the S6 chain re-runs in a
no-generation window. Operator adopted **protocol v2 for `argmax_optimum`** (level-conditional
heads; DEC-559/560): measured — brain 0.2458 / delivery 0.2593, controls now informative,
destroyed ≥ brain → **genuine finding: the served donor does not carry per-customer response
shape**; the simulator stays red both columns. Verdicts otherwise unchanged: legs 2/8.

## Micro — current state (measured)

**Card is the NEW full-size re-measurement (DEC-558), receipts stamped at `d0ac576c`.**
The behavior hash moved (`0f881d99` → `d0ac576c`) with the net `looking_glass/` changes since
`a32bbee` (DEC-544…547 memory/batch/rl + kept DINO/SwAV governor-exclusion + replay fix
`e7498b6`), forcing the re-stamp; the full-size retrain (fs_full) + S6 chain v1 re-scored
everything against the FROZEN PATH A comparators. The table below is the current measured
state (new checkpoint, same population split_seed 6529885151769, A=3515/B=1485):

| Leg | brain alone | brain + ledger |
|---|---|---|
| supervised | ❌ FAIL — `return_proneness` 0.4118 vs 0.4879 (gap 0.076; all other 6 rows pass) | ❌ FAIL — `price_sensitivity` random control ≥ delivered (−0.0614 vs −0.0798) + `return_proneness` 0.4689 vs 0.4879 (gap 0.019, CI hi +0.000004) |
| unsupervised | ❌ FAIL — `persona_geometry` 0.066 vs 0.387, `future_geometry` 0.091 vs 0.181, `segment_structure` −0.001 vs 0.268 (full-size) | ❌ FAIL — same rows outside the delivery margin |
| **rl** | ✅ **PASS** — uplift 0.2421 vs 0.1794, policy_value 0.2486 vs 0.2382 | ✅ **PASS** |
| simulator | ❌ FAIL — `argmax_optimum` v2: 0.2458 vs 0.2418, gap +0.004, CI (−0.0256, +0.0317) ⊃ 0, destroyed control ≥ brain (0.2539) — the donor does not carry per-customer response shape (DEC-560, pre-registered finding) | ❌ FAIL — v2 delivery 0.2593 passes the point floor (0.2318) but ci_lo −0.0128 < −0.01; dose_response 0.3164 vs 0.3039 PASSES |

Stages: S3/S4/S5 **PASS** (re-verified on the current pipeline + new products); S1/S2 UNKNOWN
(stale `0f881d99` — chain S1+S2 died on co-tenant unified-memory pressure twice: CUDA OOM at
249 s, then SIGKILL at 135 s, both during active-generation turns; the next chain runs in a
no-generation window, DEC-559 addendum); S6 FAIL (`full_chain_in_one_invocation` — same
cause). Hygiene H1–H5: **5/5 PASS.**
BACKWARD not yet run — it runs only against an all-green card.

## Closed front: unsupervised geometry at the rung (DEC-548…556)

The rung front is CLOSED on both axes (objective DEC-552/554; construction DEC-555/556 — stage-2
block serving FALSIFIED at the frozen bar, 3-file bundle reverted), and DEC-558 now answers the
FULL-SIZE question it left: the geometry rows stay red at full size (persona 0.066 vs 0.387;
future 0.091 vs 0.181; segment −0.001 vs 0.268) — the rung closure persists at scale.
Diagnosis record:

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

**COMPOSITE CONSTRUCTION (DEC-555/556) — rung front CLOSED, stage 2 FALSIFIED:** stage-1
decomposition (measured, CPU) put **100% of the mass-metric drop at M0→M1 — the `w_counter`
fold alone** (0.2174 → 0.1279); h-addition / shared proj / boundary each ~0 (M3 = M4 EXACT,
isometry confirmed; the M4-vs-stored 0.0031 gap was the serve_unfaded pre-fade materialization,
not numerics). Stage 2 (pre-fixed lever: native-mass block serving E = [z(256-d shape) |
z(39-d exact mass)], split-A block scales, sidecar-persisted; in-repo `products` rebuild of
run-5 artifacts, no retraining): **persona 0.1514 (bar ≥ 0.217), segment −0.0026 (bar ≥
0.095) → FALSIFIED per the pre-registered clause; supervised guard CLEAN; collapse guard
passed (probe R² 0.580 ≥ 0.40).** Measured fact: block serving improved EVERY geometry row vs
the fold (+0.0194 persona) and ships the mass exact and un-entangled, but the bar's
combined-vector distance is variance-dominated by the 256-d shape block at the 39:256 ratio
(per-dim z-score equalizes dims, not block norms). Bundle REVERTED; the replay-identity fix
(`e7498b6` — a latent repo bug stage 2 exposed: `products` rebuilds re-stamped the tag from
DEFAULTS) is committed and kept. The unsup-geometry front is exhausted at the rung on both
axes (objective: DEC-552/554; construction: DEC-555/556) — the remaining question is
FULL-SIZE.

**Tree:** all gates green — ruff check/format, principle audit strict rc=0 (ratchet 67),
fast tier rc=0. SwAV bundle REVERTED (DEC-554); kept, pre-registered as a training-regime fix:
the governor-selection exclusion in `cfm_training.py` (inert unless "swav" is in objectives;
the general rule — self-referential terms train, the predictive portfolio selects — now covers
any future EMA-teacher objective). `swav_*` config fields reverted with the bundle. The
DEC-555 stage-2 bundle (3 files) REVERTED after the FALSIFIED verdict (DEC-556: serving-side
only, falsified at the frozen bar — same protocol as DINO/SwAV).

## Next 3 actions

1. **S6 chain v3 (launching, no-generation window):** the SAME command (`measure_certification_law.py
   --products cfm_v2k --demo demo_v2 --train-log cfm_v2k/train.log` via `safe_run.py`,
   `LG_SCAN=eager`), now with protocol-v2 `argmax_optimum` (DEC-560). Acceptance: S1+S2 + S6
   fresh and working (card → legs 2/8 · stages 6/6 · 0 UNKNOWN · 6 FAIL, the substantive reds
   only); argmax values equal the deterministic scratch (0.245791 / 0.259259). If S1+S2 dies on
   co-tenant pressure again: one retry in a no-generation window — never weaken the check,
   never touch the 48 GB slice cap.
2. **Commit the completed re-stamp** (receipts + card + STATUS tick) once the chain exits with
   S6 working=True; verify the five other S6 checks (commit_recorded_clean on the new HEAD,
   render determinism, artifact hashes, N on all rows, comparator provenance).
3. **DEC-561 — pre-register the next one-axis front on the consolidated structural defect
   (DEC-560):** the served composite carries per-customer coarse attributes (probe R² ≈ PATH A)
   but not per-customer fine structure (geometry trio + level-conditional response shape +
   return_proneness gap). First attribution probe, cheapest: does the fine structure exist in
   the pre-fold state (full-size M0→M1 decomposition, rung-attribution generalized) or is it
   lost in the trunk's contractive dynamics? One measured axis per run; the rung closures
   (objective DEC-552/554, construction DEC-555/556) bound what that answer can fix.
   When the card's gate-of-record is all green → run BACKWARD.

## Operating facts

- **Repo:** `/home/austin_whaley/Documents/Repos/wonderland`, branch `foundation-rebuild`, venv `.venv`.
- **Testbed:** Layer A v2 demo (certified reference) at
  `/home/austin_whaley/wonderland_work/demo_v2/` (stream, relational/, truth, manifest).
- **Artifacts:** rung/scratch products under `/tmp/opencode/`; the canonical full-size products
  at `/home/austin_whaley/wonderland_work/cfm_v2k/` are the CURRENT-tree run (fs_full, tag
  `…b64…o4c3b5899`, behavior `d0ac576c`); the stale `0f881d99`-era artifacts are backed up at
  `cfm_v2k_stale_0f881d99/`.
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

None pending. The queue is pre-registered (DEC-558: S6 chain re-run to finish the re-stamp →
advance the remaining reds per the measured-gap ranking → BACKWARD at all-green).
Standing: push requires explicit permission.
