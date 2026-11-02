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
- **Run 2 (IN FLIGHT, ~60 min):** `--set batch=64 --set budget_steps=800` (128 views/step;
  teacher to ~4% init; governor convergence stop intact — the 800 is a ceiling, principle 2).
  **Frozen bars:** rung segment-ARI ≥ 0.095 OR persona-dist ≥ 0.217; no supervised row regressing
  > 0.03 abs vs `j_base800`; probe-persona R² ≥ 0.40 (collapse guard).

**Uncommitted DINO bundle:** `cfm_model.py` + `cfm_training.py` (active code) + `dino_*` recipe
fields in `cfm_config.py` (inert defaults, committed with the `budget_steps` mechanism field).
Gates (ruff / principle audit / pytest) run on it only after the dose verdict; falsify → revert
the bundle, keep the recorded result.

## Next 3 actions

1. **Judge DINO run 2** vs the frozen DEC-550 bars (judge + bar compare pre-staged at
   `/tmp/opencode/judge_r4.py`).
2. **If a bar is met:** attribution probe on the new products, full-size adoption run
   (~2,310+ steps, re-estimate with DINO's ~3 extra ssm passes), S6 chain, card regeneration;
   commit the DINO bundle after ruff + `principle_audit --all --strict` + pytest.
3. **If falsified:** revert the bundle, record the DEC-550 result, next axis = SwAV prototypes
   (Sinkhorn-balanced clustering, k from stream attribute cardinality).

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
- **Resources:** GB10 unified memory, budget cap ~44.9 GB (DEC-547); compute-bound ~40–46
  samples/s (DEC-546); GPU time free per operator — one job at a time; commits local until the
  operator asks to push.

## Open operator decisions

None pending. The queue is pre-registered (DINO dose → SwAV → return_proneness /
argmax_optimum re-measurement). Standing: push requires explicit permission.
