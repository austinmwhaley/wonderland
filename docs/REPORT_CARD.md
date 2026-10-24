# REPORT_CARD — Layer-B (encoder) [GENERATED]

> **Generated** by `python3 scripts/report_card.py` — DO NOT EDIT. Gates live in `scripts/card_spec.py`; every row resolves to a receipt (`docs/report_cards/experiments/`). A missing receipt renders NOT MEASURED.

- **Generated:** 2026-10-24 · **commit:** `f64d013` · **branch:** primary-encoder
- **Data:** `relational_v3` synthetic reference (+ schema variant) · harness, no business data.
- **Latest demo receipt:** `7e32ebb84fc3` · seeds `3` · runtime `1009.2s`.
- **B mastery:** **NOT YET MASTERED** — PASS 27 · FAIL 1 · NOT MEASURED 0.

## Project goal (AGENTS.md)

One pretrained, **self-supervised, schema-agnostic encoder per stream** whose FROZEN output is the
**universal donor** every consumer builds on. **Layer 0 (now):** `E`-only matches/beats a fair table and
adds signal beyond `table+seq`; no table/counter/summary inside `E`. **Layer 1 (next):** certification --
sufficiency / minimality / decision-sufficiency via `U = 1 - worst-case oracle gap` vs full history.
**Anti-overfit:** Layer B is self-supervised only; known use cases may VETO, never SHAPE.

## Principles (charter; conflict order Robust > Universal > Adaptive > Lawful > Efficient > Honest)

1. **Universal** -- one architecture for any stream; a model is trained per stream, never cross-applied.
2. **Adaptive & Dynamic** -- everything resolves or is learned from the input; no magic literals.
3. **Lawful** -- compositionality, equivariance, and the correct algebra per quantity.
4. **Efficient** -- linear-time, constant-state, minimal capacity; slow or bloated is not done.
5. **Robust** -- fail safe, calibrated uncertainty, deterministic; divergence is a hard failure.
6. **Honest** -- held-out/falsifiable, label-free, never weakened to pass a gate or tuned to the judged metric.
7. **Accessible** -- information recoverable by simple consumers, not merely present.
8. **Diagnosable** -- attribute a shortfall to a specific axis with a matched control.

## Subgoals (decomposition)

Goal -> subgoals; a subgoal is MET only when all its rows PASS (or are EXCLUDED, reported). Verdicts are derived from the rows below.

| Subgoal | Principle | Rows | PASS | FAIL | EXCL | N/M | Status |
|---|---|---|---|---|---|---|---|
| SG-B1 Construction invariants | Lawful / Robust | 6 | 6 | 0 | 0 | 0 | **MET** |
| SG-B2 Donor contract | Universal / Honest | 5 | 5 | 0 | 0 | 0 | **MET** |
| SG-B3 Content axes | Adaptive / Lawful | 8 | 7 | 1 | 0 | 0 | **OPEN** |
| SG-B4 Self-supervised objectives | Adaptive | 10 | 9 | 0 | 1 | 0 | **MET** |
| **SG-B0 · Master the encoder (B)** | all | 29 | 27 | 1 | 1 | 0 | **OPEN** |

## Encoder architecture (Layer B)

- **Tokenizer** -- auto-discovers fields (categorical / numeric / ISO-time); additive per-field embeddings
  with learned gates; time features = `log1p(gap)` + data-derived Fourier periods (FFT peaks); raw numerics via signed `log1p`.
- **Dynamics** -- continuous-time gated SSM (learned decay, tau initialized from data gap-quantiles);
  affine prefix `_scan` (monoid-exact); width/layers derived from data.
- **State** -- contractive shape state (bounded) + additive mass state (extensive algebra: sums add, no `1-decay`).
- **Readout** -- ensemble (last / mean / max / attention-pooled / per-scale states / mass) with learned positive
  weights -> `E` (d-dim, frozen).
- **Self-supervised objectives (heads)** -- next-event; gap/rate; categorical + numeric reconstruction; identity;
  uncertainty (missingness); past-summary; future-summary (one head per data-derived horizon); relational/context.
- **Training** -- AdamW; trust-region adaptive lr (seed bound; expand/shrink on held-out response); no gradient
  clip; MAD-based early stop; deterministic kernels; learned loss weights.

## Construction — invariants by construction

| # | Item | Metric | Comparator | Gate | Value | Verdict |
|---|---|---|---|---|---|---|
| C1 | Field-permutation equivariance | max |E(x) - E(perm(x))| | exact (0) | <= 1e-6 | 1.19e-07 | **PASS** |
| C2 | Scan (monoid) exactness | max |scan(g o f) - scan(g) o scan(f)| | exact (0) | <= 1e-12 | 8.88e-16 | **PASS** |
| C3 | Mass additivity (extensive algebra) | |mass(a+b) - mass(a) - mass(b)| | exact (0) | == 0 | 0 | **PASS** |
| C4 | Determinism (same seed => identical) | val trajectory bit-identical across repeats | itself | identical | True | **PASS** |
| C5 | Schema-agnosticism (variant, zero code change) | E trains on renamed schema + new field | raw v3 | pass | True | **PASS** |
| C6 | Frozen + reproducible artifact | artifact hash recorded | present | hash present | d7d190e10924 | **PASS** |

## Donor — does not lose / adds / controls

| # | Item | Metric | Comparator | Gate | Value | Verdict |
|---|---|---|---|---|---|---|
| D1 | E vs raw table (6k) | value/orders/churn R2-AUC margin | raw table | not materially worse (all 3) | +0.079 / +0.079 / -0.001 | **PASS** |
| D2 | E vs fair table (6k) | value/orders/churn margin | fair table | E > fair on all | +0.246 / +0.304 / +0.225 | **PASS** |
| D3 | E adds over table+sequence | R2/auc gain of table+seq+E vs table+seq | table+seq | delta > 0 on all | +0.055 / +0.087 / +0.001 | **PASS** |
| D4 | Negative control: random E | same battery with random E | learned E | random < learned (all 3) | +0.437 / +0.448 / +0.386 | **PASS** |
| D5 | Negative control: destroyed E | same battery with shuffled E dims | learned E | destroyed < learned (all 3) | +0.447 / +0.521 / +0.402 | **PASS** |

## Content axes (8)

| # | Item | Metric | Comparator | Gate | Value | Verdict |
|---|---|---|---|---|---|---|
| A1 | Shape | retrieval / churn-nn / eff-rank | table (n/a) | retrieval > 0 and churn-nn > 0.6 | 0.417 | **PASS** |
| A2 | Mass | past-magnitude reconstruction corr2; additivity | oracle | corr2 >= 0.4 and additivity exact | 0.489 | **PASS** |
| A3 | Identity | kNN channel + category purity (from E) | table | both >= table | ch 0.297/0.244 · cat 0.180/0.185 | **PASS** |
| A4 | Rate | churn AUC | table | E >= table | E 0.9290 vs 0.9256 | **PASS** |
| A5 | Uncertainty | ECE; OOD retention | coverage target | ECE <= 0.1 and OOD retained | 0.0482 | **PASS** |
| A6 | Causal | IPS known-effect recovery | random; table | E recovers effect (CI excludes 0) | diff -215.3 [-413.1,-12.6] | **FAIL** |
| A7 | Order/Path | event-shuffle drop | table (invariant) | E drops, table does not | drop -0.2785 | **PASS** |
| A8 | Relational/Context | cross-entity purity | chance | > chance | 0.005 vs chance 0.0005 | **PASS** |

## Objectives (self-supervised battery)

| # | Item | Metric | Comparator | Gate | Value | Verdict |
|---|---|---|---|---|---|---|
| O1 | next | linear probe next-event skill | naive (marginal) | beats naive | skill +0.478 vs naive +0.289 | **PASS** |
| O2 | dt | gap/rate probe skill | naive (prev gap) | beats naive | skill +0.102 vs naive -0.000 | **PASS** |
| O3 | jepa | latent future prediction | naive (hist mean) | beats naive | skill +0.400 vs naive -0.001 | **PASS** |
| O4 | mask | masked-event recovery | naive (marginal) | beats naive | skill +0.910 vs naive +0.285 | **PASS** |
| O5 | value | value probe skill | naive (hist mean) | beats naive | skill +0.011 vs naive -0.000 | **PASS** |
| O6 | entity | entity/identity probe | naive (marginal) | beats naive | skill +0.878 vs naive +0.878 | **EXCLUDED** |
| O7 | order | temporal-order probe | naive (random) | beats naive | skill +0.905 vs naive +0.328 | **PASS** |
| O8 | agg | aggregate/window probe | naive (hist mean) | beats naive | skill +0.010 vs naive -0.000 | **PASS** |
| O9 | query | query-conditioned probe | naive (hist mean) | beats naive | skill +0.671 vs naive -0.000 | **PASS** |
| O10 | sf | successor-feature probe | naive (prev value) | beats naive | skill +0.220 vs naive +0.091 | **PASS** |

## Verdict

B is **not yet mastered**: 0 row(s) unmeasured, 1 failing. Unmeasured rows need receipts; failing rows need an axis fix (one axis per run).

## Disclosures

- Synthetic harness; no business-value or causal claim beyond the randomized holdout.
- Some axis receipts predate DEC-280/281 (future-objective + ladder); refresh required before claiming them.
- Objective battery (`next/dt/jepa/mask/value/entity/order/agg/query/sf`) is defined in `looking_glass/probes.py` and not yet wired to this encoder -> NOT MEASURED.
