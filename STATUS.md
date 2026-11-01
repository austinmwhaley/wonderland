# STATUS — wonderland

*Live state only. History lives in `DECISIONS.md` (append-only) and `git log`.
Rewritten 2026-11-01. If this file grows past ~150 lines, it is becoming a log again.*

## Macro

**Goal (Layer 0, now):** one frozen, self-supervised, schema-agnostic encoder per event stream
whose embeddings are a **universal donor** — supervised, unsupervised, RL and simulator consumers
all get signal from it through simple heads.

**Definition of progress:** **PATH B (event stream → frozen encoder → simple head) > PATH A
(wide table → GBM)**, with paired CIs and controls. Two numbers per leg: `brain alone` =
**progress**, `brain + ledger` = **delivery** (non-inferiority, 0.01 R² margin). Ledger points
are never a brain win.

**Stage:** Layer 0 — encoder not yet mastered. Layer 1 (certification, `U`) is not started.

**Progress:** **2/8 leg-columns, 0/6 stages, 0 open BREAKING/MAJOR backward requirements.**

## Micro — current state (measured)

Encoder `0.11.0r103115-b26d256e3lr3e-04o39ad8593` · behavior hash `94857be6815a96f3`.

| Leg | brain alone | brain + ledger |
|---|---|---|
| supervised | ❌ FAIL (x0, return_proneness, churn_day, regime_switches, final_regime_days) | ❌ FAIL |
| unsupervised | ❌ FAIL (persona_geometry — random control ≥ brain) | ❌ FAIL |
| **rl** | ✅ **PASS** | ✅ **PASS** |
| simulator | ❌ FAIL (dose_response_curve, argmax_optimum) | ❌ FAIL |

| Stage | State |
|---|---|
| S1 schema resolution | working-state proof passes; **stale stamp** (measured against `1b238954570c`) |
| S2 vocab & numeric encoding | working-state proof passes; **stale stamp** |
| **S3 lawful dynamics** | ❌ **FAIL — `clock_periodicity`** (week-shift drift 2.2e-2 vs gate 1e-4) |
| S4 buckets | working-state proof passes; **stale stamp** |
| **S5 objective battery** | ❌ **FAIL — `action_conditioned_outcome`, `distance_regression`, `ledger_reconstruction`** |
| S6 certification | working-state proof passes; **stale stamp** |

Hygiene H1–H5: **5/5 PASS.** Backward requirements: **none recorded** (BACKWARD has not been run —
it only runs against an all-green card).

## The one blocker

**The encoder's representation is collapsed.** Measured on 266 held-out states:

| quantity | value | reference |
|---|---|---|
| effective rank | **6.155** | dim 256; random null ≈ 90.1; target ≈ 30 (`rank_target` 0.117) |
| eff-rank ratio to null | **0.068** | 1.0 = as spread as the null |
| mean pairwise cosine | 0.148 | high similarity between customers |
| off-diagonal redundancy | 0.295 | null 0.049 |

Everything red traces to this. Information is *present* (7/7 views decode; ledger reconstruction
R² 0.80–0.98 on spend/margin/purchase count) but it is not *accessible as a shared space*: a
linear head on ~6 effective directions loses to a GBM on 48 wide-table columns, and the
unsupervised geometry control is indistinguishable from random.

**The geometry governor is saturated, not undertuned.** λ ramped 1.0 → **1,000,000 (its cap)**,
bank rank 0.0228 against a 0.117 target, history populated (10 entries, 8192-row bank). More of
the same soft pressure is not a lever. The project's own FR v2.0 receipt says the mechanism class
is wrong: soft rank penalties were measurably rejected, while rank **by construction** inside the
state update (bilinear `h⊗x`) raised *transferable* cross-sample rank **1.79×**.

## Next 3 actions (in order)

1. **Measure attainable rank** — eigendecomposition of the full-history measure. Is ~30 directions
   actually supported by the stream, or is ~6 the truth? Cheap, read-only, and it converts an
   open-ended optimization into a yes/no. *(If ~6 is attainable, `rank_target` is wrong and the
   geometry gate is defensible; if ~30, the encoder is genuinely lossy.)*
2. **Re-run the four stale stage producers** so S1/S2/S4/S6 carry the current behavior hash:
   `measure_input_law.py`, `measure_views_law.py`, `measure_certification_law.py` (S3 and S5 are
   already fresh). The card cannot count a stage green on a stale stamp.
3. **Operator decision on geometry** — either (A) move rank into the recurrence by construction
   (the mechanism class with evidence), or (B) re-derive `rank_target` from (1) and defend it.
   These are mutually exclusive and gate-touching; they need the operator, not the loop.

**Not the blocker:** receipts (fresh), ledger (8 parts publish, G1–G4 pass), stability (governed,
no divergence), plumbing (products rebuild, config replay, UTC cut all fixed and committed).

## Operating facts

- **Repo:** `/home/austin_whaley/Documents/Repos/wonderland`, branch `foundation-rebuild`, venv `.venv`.
- **Artifacts:** products/registry under `/home/austin_whaley/wonderland_work/cfm_m8_counter/`;
  demo stream + truth under `/home/austin_whaley/wonderland_work/demo/`.
- **Card:** `python3 scripts/report_card.py` → `REPORT_CARD.md` (pure function of the receipts under
  `docs/report_cards/experiments/`).
- **Resources:** one GPU job at a time; 20 GB training container; system floor 8 GB; commits local
  until the operator asks to push.

## Open operator decisions

- **Geometry mechanism** (A) construction vs (B) re-derived target — see action 3.
- **P8 provenance** (`SPINE.md` Appendix B): the reference table contains random columns absent
  from the stream. Fix the generator and re-certify, or drop them from the fair baseline.
- **P2 testbed**: the synthetic stream may not be able to test the learned bet at all. A
  measure-insufficient stream is the decisive experiment and is not built.
