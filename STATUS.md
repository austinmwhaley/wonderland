# STATUS — wonderland

*Live state only. History lives in `DECISIONS.md` (append-only) and `git log`.
Rewritten 2026-11-02 (DINO falsified, DEC-552; SwAV front). If this file grows past ~150 lines, it is becoming a log again.*

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

**Next axis (pre-registered by DEC-550): SwAV prototypes** — Sinkhorn-balanced assignment to k
learned prototypes on the served composite; k from the measured parallel-analysis signal rank
(~15), NOT from judged segment cardinality (teaching the encoder its exam answer violates the
anti-overfit clause). DEC-552's flag is part of the SwAV protocol: the governor's val sum
EXCLUDES the self-referential objective's own term (the 11-objective portfolio sum selects; the
new term trains only). Frozen rung bars unchanged. If SwAV falsifies too, the 4-arm table
reframes the front: the mass arm already sits at the bar, so the cheapest remaining lever is the
composite CONSTRUCTION (state/interface axis — keep the mass arm's distance structure in the
blend), not further objective pressure.

**Tree:** fast tier rc=0, ruff check/format clean, principle audit strict rc=0 — all gates green
on the committed tree (14 pre-existing test reds + 53 ruff errors + 10 unformatted files closed
this session; audit ratchet shrank 69→67 with no new logical acceptances).

## Next 3 actions

1. **SwAV pre-registration (DEC-553):** derive k from the measured parallel-analysis signal rank
   on the base rung products (`measure_rank_attainability.py --derive`), freeze the protocol
   (Sinkhorn-balanced assignment, EMA teacher head, governor val sum excludes the swav term,
   B=64, 800-step ceiling, frozen rung bars), then implement + smoke.
2. **SwAV dose run** (one axis: the objective; launched via `safe_run.py`), judge against the
   frozen bars, probe for attribution.
3. **If met:** full-size adoption run + S6 chain + card regeneration; commit the SwAV bundle
   after all gates. **If falsified:** record + revert, then the front shifts to composite
   construction (the mass arm already sits at 0.2162 ≈ the bar; keep its distance structure in
   the blend — state/interface axis).

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

None pending. The queue is pre-registered (SwAV dose → [if falsified: composite construction]
→ return_proneness / argmax_optimum re-measurement). Standing: push requires explicit permission.
