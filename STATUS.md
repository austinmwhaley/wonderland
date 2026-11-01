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

**The geometry gate's target is mis-derived — the encoder is not missing rank.** Measured
read-only on 724 held-out B customers (`rank_attainability.json`, DEC-514):

| quantity | value | meaning |
|---|---|---|
| exhaustive measure, parallel analysis | **15 directions** (of 104) | the stream's own signal rank |
| learned shape, parallel analysis | **15 directions** (of 256) | already at that count |
| learned shape effective rank | 5.414 / null 110.585 | variance concentrated in ~5 of the 15 |
| served `E` | 21 directions (of 360) | |
| **card target** | **30 directions** (0.117 × 256) | **2× what the stream carries** |

So the earlier "6 of 30 directions" reading was comparing the state to a number the stream never
supported. Pressuring for 0.117 means asking for twice the information that exists — which is
rank manufacture, banned by the doctrine, and which the λ = 1,000,000 saturation already
demonstrated is unreachable.

What is still true: the shape's variance is **unevenly weighted** (`pr_score` −2.749, mean pairwise
cosine 0.148), and the consumer legs are red. Direction *count* is at capacity; direction
*weighting* and *accessibility* are not.

**Decision needed (certification rule 8 — gate change):** re-derive `rank_target` from parallel
analysis (≈0.059 for a 256-dim shape) as a new protocol version, disclosed on the card as a
weaker claim. Rejected: keep 0.117 and raise λ (saturated), raise `geometry_boost` (same
mechanism), batch whitening (manufactures non-transferable rank).

**Next axis: accessibility, not rank.** A linear head on 15 real but unevenly-weighted
directions loses to a 48-column GBM — that is the measurement to make now.

**Not the blocker:** receipts (fresh), ledger (8 parts publish, G1–G4 pass), stability (governed,
no divergence), plumbing (products rebuild, config replay, UTC cut all fixed and committed).

## Next 3 actions

1. **Operator ruling on `rank_target`** — re-derive to ≈0.059 (parallel analysis) as a new
   protocol version, disclosed on the card as a weaker claim; stop geometry pressure. This is a
   gate change, so it is the operator's call, not the loop's.
2. **Measure accessibility** — with direction count at capacity, test whether the red legs come
   from uneven weighting/scale (`pr_score` −2.749, mean cosine 0.148) rather than missing rank.
3. **Re-run the four stale stage producers** (`measure_input_law`, `measure_views_law`,
   `measure_certification_law`) so S1/S2/S4/S6 carry the current behavior hash. S3's
   `clock_periodicity` and S5's three failures are separate, real defects.

## Operating facts

- **Repo:** `/home/austin_whaley/Documents/Repos/wonderland`, branch `foundation-rebuild`, venv `.venv`.
- **Artifacts:** products/registry under `/home/austin_whaley/wonderland_work/cfm_m8_counter/`;
  demo stream + truth under `/home/austin_whaley/wonderland_work/demo/`.
- **Card:** `python3 scripts/report_card.py` → `REPORT_CARD.md` (pure function of the receipts under
  `docs/report_cards/experiments/`).
- **Resources:** one GPU job at a time; 20 GB training container; system floor 8 GB; commits local
  until the operator asks to push.

## Open operator decisions

- **`rank_target`** — re-derive from parallel analysis (DEC-514, ≈0.059) as a new protocol
  version, or keep 0.117 and accept that the geometry gate is unreachable by construction.
- **P8 provenance** (`SPINE.md` Appendix B): the reference table contains random columns absent
  from the stream. Fix the generator and re-certify, or drop them from the fair baseline.
- **P2 testbed**: the synthetic stream may not be able to test the learned bet at all. A
  measure-insufficient stream is the decisive experiment and is not built.
