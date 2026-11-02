# STATUS — wonderland

*Live state only. History lives in `DECISIONS.md` (append-only) and `git log`.
Rewritten 2026-11-02 (DINO falsified DEC-552; SwAV falsified DEC-554; stage-2 construction
falsified DEC-556 — front = full-size re-measurement). If this file grows past ~150 lines, it
is becoming a log again.*

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

1. **FULL-SIZE re-measurement (DEC-556 pre-registered — the rung's falsify clause):** train the
   CURRENT tree (no swav, no contrast, no mass-block — all reverted) on the full 5000-customer
   demo_v2 population (`as_of=2025-01-01`, `split_seed=6529885151769`, A=3515/B=1485,
   `anchors=6`, the 4 adopted flags), so the card's full-size brain-side receipts are re-scored
   against the FROZEN PATH A comparators (return_proneness 0.4879; persona 0.404 / future 0.14 /
   segment 0.309) and the stale cfm_v2k values (0.4145 / 0.14 / 0.10 / 0.001) are REPLACED, not
   relabeled. Estimate from measured numbers: **~1.5–2.5 h total** (cfm_v2k training 2310 steps @
   1.72 s/step = 66 min old tree B=16; idle co-tenant ~2 s/step; DEC-554 dose ceiling re-derives
   at the first `[govern]` receipt). One GPU job at a time via `safe_run.py`.
2. **Judge + probe the full-size products** against the card's frozen full-size comparators
   (supervised leg + unsup leg + the M0 mass-only power-control arm, instrument acceptance:
   reproduce the PATH A values to 3 decimals). Record in DEC-557: does the mass arm clear the
   0.217 bar with headroom at full size, and does the served composite (fold) move off the rung
   noise band?
3. After the geometry front: re-measure supervised `return_proneness` (full-size 0.4145 vs
   0.4879) and simulator `argmax_optimum` (+0.0026, CI⊃0) → card 8/8 → run BACKWARD.

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

None pending. The queue is pre-registered (DEC-556 full-size re-measurement → judge/probe vs the
frozen card comparators → re-measure the remaining reds per the card → BACKWARD at all-green).
Standing: push requires explicit permission.
