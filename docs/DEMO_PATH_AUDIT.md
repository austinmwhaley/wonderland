# Demo path audit — decisions vs principles

Scope: Layer A data (relational PATH A/B), Layer B encoder prototype, training loop, evaluation.
Legend: ✅ aligned · ⚠️ partial/fallback · ❌ misaligned (action) · 🔒 non-negotiable fixed.

| # | Decision | Principle(s) | Status | Principle-aligned action |
|---|---|---|---|---|
| E1 | Relational tables → unified stream + wide table | Universal, Honest | ✅ | — |
| E2 | PATH A point-in-time; leakage/balance gate | Honest | ✅ | — |
| E3 | PATH B exactly 4 canonical columns | Universal, Accessible | ✅ | — |
| E4 | as_of split / future-only labels | Honest, Lawful | ✅ | — |
| E5 | 2k/6k data scale | Efficient | ⚠️ testbed size; derive from budget | Capacity search |
| E6 | Action cadence `86400` declared | Universal, Adaptive | ❌ | Infer from action-event spacing; declared is fallback |
| E7 | Token fields hand-listed (product/category/brand/…) | Universal, Adaptive | ❌ | Generic schema-derived: all discovered fields; ID rule by cardinality |
| E8 | Signed `log1p` numeric transform | Learn, don’t teach | ⚠️ stable fallback | Learned monotone/quantile transform; keep log1p fallback |
| E9 | Fixed hour/dow periodic features | Universal, Adaptive | ✅ replaced | Data-derived spectral peaks (implemented in exp3: 12h/24h from data) |
| E10 | Step-index “fast/slow” decay | Lawful, Adaptive | ✅ replaced | Continuous-time learned `τ` (implemented in exp3) |
| E11 | Timescale count/`τ` init | No magic, Adaptive | ⚠️ | Count from spectrum (now 2); quantile bounds/caps are literals → derive/budget |
| E12 | Fixed width/depth | No magic, Efficient | ❌ | Capacity from data/validation/budget search |
| E13 | Python loop over T | Speed, Efficient | ✅ fixed | Vectorized `_scan` (compiled affine prefix scan) |
| E14 | Additive readout with learned positive weights | Dynamic, Accessible | ✅ | — |
| E15 | Fixed readout menu (last/mean/max/attention/magnitude) | Dynamic | ⚠️ | Menu is a capacity choice; learned weights; revisit multi-query pooling |
| E16 | Sampled cutoffs + data-derived horizons | Universal, Adaptive | ✅ | No fixed 90d curriculum |
| E17 | Mask rate / patience | Dynamic, Robust | ✅ measured (exp3) | Clip bounds are literals → derive or document fallback |
| E18 | Fixed LR `3e-4` | No magic, Robust | ❌ | Measured LR (short search/curvature) |
| E19 | Class weights from frequency | Adaptive | ✅ | — |
| E20 | No table/counter/summary/distill in `E` | Honest, Sequence-first | ✅ | — |
| E21 | Magnitude branch (small extensive sums) | Lawful, Accessible | ✅ | — |
| E22 | Objective coverage | Universal, Dynamic | ⚠️ | Currently next-event + horizons; add geometry (contrastive/JEPA) + action-conditioned outcome |
| E23 | Learned loss weights | Dynamic | ✅ | — |
| E24 | Measurement uses business GBM heads | Honest, Diagnosable | ⚠️ | Add head-free proxies first: linear/kNN, oracle gap, geometry, decision CI |
| E25 | Business heads only for final confirmation | Honest | ⚠️ | Keep them at milestones, not every fast loop |
| E26 | Block-aware serving | Accessible | ⚠️ design only | Implement block boundaries/scales; consumer views |
| E27 | Evaluation targets/folds/baselines/gates | Honest | 🔒 | Never dynamic |
| E28 | Certification integrity / anti-self-deception | Honest | ✅ | Continue frozen gates, receipts, audits |
| E29 | Prototype lives in `/tmp` | Auditable, Reproducible | ❌ | Move to `scripts/encoder_experiment.py`; audit/CI covers it |
| E30 | Pre-implementation check + principle audit | Honest, Diagnosable | ✅ | Run `scripts/principle_audit.py --strict` on changes |
| E31 | Generator realism (one item/order, no sessions) | Universal | ⚠️ documented testbed | Add sessions/hierarchy only if it answers a question |
| E32 | Reproducibility of prototype | Reproducible | ⚠️ | Seeds + hashes in the repo script; runtime receipt |

## Summary

**Aligned:** relational data, canonical stream, point-in-time labels, additive readout, sampled
horizons, continuous-time `τ`, data-derived periodicity, `_scan`, no table in `E`, learned loss
weights, fixed evaluation, principle audit.

**Misaligned / partial (actions):**

1. **Generic tokenizer** (E7): all discovered fields, IDs by cardinality, coverage vocab/hash.
2. **Capacity** (E12): width/depth/timescales/vocab by measured budget; no fixed `d/layers`.
3. **Measured LR / derived bounds** (E18, E17, E11): remove remaining literals.
4. **Head-free proxies** (E24): linear/kNN + oracle gap + geometry before business heads.
5. **Objective coverage** (E22): geometry and action-conditioned outcome.
6. **Block-aware serving** (E26).
7. **Move prototype into the repo** (E29) so the audit/CI applies.
8. **Infer action cadence** (E6).
9. **Learned numeric transform** (E8) after the rest.

## First principle-aligned fast result (exp3, 1 seed, 5 epochs)

Data-derived: periods `[12h, 24h]`, 2 timescales, `τ=[2.0, 24.2] days`, mask rate from
missingness. Readout weights uniform (no collapse). Vectorized `_scan`.

| Target | Table | E only | Hybrid | Table+seq | Table+seq+E |
|---|---|---|---|---|---|
| value90 R² | 0.349 | 0.208 | 0.384 | 0.417 | **0.420** |
| orders90 R² | 0.352 | 0.260 | 0.419 | 0.422 | **0.466** |
| churn90 AUC | 0.964 | 0.642 | 0.947 | 0.951 | 0.948 |

Signal: `E` adds **+0.044 R² on orders beyond the explicit sequence features**; value +0.003;
churn −0.003. One seed/5 epochs: **not confirmed**. Next: 3 seeds/20 epochs; then 6k scale.

## Pre-flight checklist for future demo runs

1. **Data:** canonical stream; as_of; action cadence inferred; schema-derived fields; dataset gate.
2. **Encoder:** generic additive token; continuous-time learned `τ`; `_scan`; capacity from
   budget; additive readout; no table/counter/summary in `E`.
3. **Objectives:** next event + sampled horizons + geometry + decision; learned weights/masks.
4. **Measurement:** head-free proxies first (linear/kNN, oracle gap, geometry, decision CI);
   business heads only at milestones.
5. **Governance:** `principle_audit.py --strict` on changes; fixed evaluation/causality; record
   decisions + runtime receipts.
6. **Scale:** confirm on 3 seeds; then 6k; then the certified card.
