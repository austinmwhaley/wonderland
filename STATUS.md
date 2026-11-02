# STATUS — wonderland

*Live state only. History lives in `DECISIONS.md` (append-only) and `git log`.
Rewritten 2026-11-01 (DINO dose front). If this file grows past ~150 lines, it is becoming a log again.*

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

Diagnosis chain (DEC-548…550, all rung measurements):

- **Temporal drift is not the axis** (DEC-548): `proj(y)`-based contrast falsified.
- **Attribution probe** (CPU, validates the judge to the 3rd decimal): the served donor decodes
  persona/future fields **linearly as well as PATH A** (probe R² 0.583 vs 0.587; 0.364 vs 0.307)
  yet pairwise distance fidelity is 0.14 vs 0.404 and segment ARI 0.001 vs 0.309. The information
  is PRESENT; the **geometry is entangled** (principle 7). Trunk and counter alone each decode
  ~0.32; the fold combines to 0.583 — the 256-direction ARRANGEMENT is what scrambles it. The
  boundary is an isometry, so the fix cannot live in the readout.
- **Falsified cheaply, in sequence:** contrast-removal (persona 0.1295), VICReg isotropy (0.1400) —
  both within the rung noise band 0.1295–0.1400. No covariance-shaping objective moves it.
- **DINO self-distillation (DEC-550, pre-registered):** label-free EMA-teacher structure pressure
  on the exact served composite `proj(h + W_c·M)`. Run 1 (B=16, 266 governed steps): directional
  (persona 0.1520, ~2σ above noise) but under-dosed vs the bar (0.217); mechanical cause —
  32 views/step and teacher still ~35% initialized.
- **Runs 2 and 3 both died to the watchdog's live-fire test sweep** (post-mortem in DECISIONS,
  2026-11-01): the harness placed them in a vte scope outside `wonderland.slice`, and my own
  pytest (`test_gpu_budget`'s `--once` sweep) SIGTERM'd them — not OOM, not the inductor.
  Fixed: the sweep test now skips while a producer is live (verified against run 4), and jobs
  launch through `scripts/safe_run.py` into the 48G slice.
- **Run 4 (IN FLIGHT, est. 3–4 h + judge):** the pre-registered dose unchanged —
  `LG_SCAN=eager`, `--set batch=64 --set budget_steps=800` (128 views/step; teacher to ~4%
  init; governor convergence stop intact, 800 is a ceiling — principle 2), launched via
  `safe_run.py` (conformance receipt: slice cap 48.0 GB, seatbelt cgroup-clamp 45.06 GB).
  **Frozen bars:** rung segment-ARI ≥ 0.095 OR persona-dist ≥ 0.217; no supervised row regressing
  > 0.03 abs vs `j_base800`; probe-persona R² ≥ 0.40 (collapse guard).

**Uncommitted DINO bundle:** `cfm_model.py` + the DINO hunks in `cfm_training.py` (active code).
The `dino_*` recipe fields are committed with the `budget_steps` mechanism (inert defaults).
Gates (ruff / principle audit / pytest) run on it only after the dose verdict; falsify → revert
the bundle, keep the recorded result. **Fast tier is fully green (rc=0) on the committed tree** —
14 pre-existing reds closed this session (one-serving-readout contract, tuple unpacks, eff_rank,
minimal-frame resolver, watchdog pattern drift; DECISIONS triage entry).

## Next 3 actions

1. **Judge DINO run 4** vs the frozen DEC-550 bars (judge + bar compare pre-staged at
   `/tmp/opencode/judge_r4.py`; products at `/tmp/opencode/r4_dino64/`).
2. **If a bar is met:** attribution probe on the new products (collapse guard R² ≥ 0.40),
   full-size adoption run (re-estimate with DINO's ~3 extra ssm passes + co-tenancy), S6 chain,
   card regeneration; commit the DINO bundle after ruff + `principle_audit --all --strict` + pytest.
3. **If falsified or the run dies again:** record exit + last log lines, revert the bundle,
   next axis = SwAV prototypes (Sinkhorn-balanced clustering, k data-derived — NOT from judged
   segment cardinality, which is teaching).

After the geometry front: re-measure `return_proneness` (full-size) and `argmax_optimum`, then
the card can go 8/8 → run BACKWARD.

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

None pending. The queue is pre-registered (DINO dose → SwAV → return_proneness /
argmax_optimum re-measurement). Standing: push requires explicit permission.
