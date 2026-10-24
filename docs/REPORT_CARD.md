# REPORT_CARD — Layer-B (encoder) [GENERATED]

> **Generated** by `python3 scripts/report_card.py` — DO NOT EDIT. Gates live in `scripts/card_spec.py`; every row resolves to a receipt (`docs/report_cards/experiments/`). A missing receipt renders NOT MEASURED.

- **Generated:** 2026-10-23 · **commit:** `397d02a` · **branch:** primary-encoder
- **Data:** `relational_v3` synthetic reference (+ schema variant) · harness, no business data.
- **Latest demo receipt:** `7e32ebb84fc3` · seeds `3` · runtime `1009.2s`.
- **B mastery:** **NOT YET MASTERED** — PASS 13 · FAIL 3 · NOT MEASURED 13.

## Construction — invariants by construction

| # | Item | Metric | Comparator | Gate | Value | Verdict |
|---|---|---|---|---|---|---|
| C1 | Field-permutation equivariance | max |E(x) - E(perm(x))| | exact (0) | <= 1e-6 | 1.19e-07 | **PASS** |
| C2 | Scan (monoid) exactness | max |scan(g o f) - scan(g) o scan(f)| | exact (0) | <= 1e-12 | 8.88e-16 | **PASS** |
| C3 | Mass additivity (extensive algebra) | |mass(a+b) - mass(a) - mass(b)| | exact (0) | == 0 | 0 | **PASS** |
| C4 | Determinism (same seed => identical) | val trajectory bit-identical across repeats | itself | identical | — | **NOT MEASURED** |
| C5 | Schema-agnosticism (variant, zero code change) | E trains on renamed schema + new field | raw v3 | pass | — | **NOT MEASURED** |
| C6 | Frozen + reproducible artifact | artifact hash recorded | present | hash present | d7d190e10924 | **PASS** |

## Donor — does not lose / adds / controls

| # | Item | Metric | Comparator | Gate | Value | Verdict |
|---|---|---|---|---|---|---|
| D1 | E vs raw table (6k) | value/orders/churn R2-AUC margin | raw table | E >= raw on all | +0.035 / +0.032 / +0.017 | **PASS** |
| D2 | E vs fair table (6k) | value/orders/churn margin | fair table | E >= fair on all | — | **NOT MEASURED** |
| D3 | E adds over table+sequence | R2/auc gain of table+seq+E vs table+seq | table+seq | delta > 0 (CI) | +0.040 / +0.065 / -0.003 | **FAIL** |
| D4 | Negative control: random E | same battery with random E | learned E | random < learned (all 3) | +0.437 / +0.448 / +0.386 | **PASS** |
| D5 | Negative control: destroyed E | same battery with shuffled E dims | learned E | destroyed < learned (all 3) | +0.447 / +0.521 / +0.402 | **PASS** |

## Content axes (8)

| # | Item | Metric | Comparator | Gate | Value | Verdict |
|---|---|---|---|---|---|---|
| A1 | Shape | retrieval / churn-nn / eff-rank | table (n/a) | retrieval > 0 and churn-nn > 0.6 | 0.417 | **PASS** |
| A2 | Mass | past-magnitude reconstruction corr2; additivity | oracle | corr2 >= 0.4 and additivity exact | 0.489 | **PASS** |
| A3 | Identity | kNN channel + category purity | table | both >= table | ch 0.504/0.251 · cat 0.172/0.265 | **FAIL** |
| A4 | Rate | churn AUC | table | E >= table | E 0.9290 vs 0.9256 | **PASS** |
| A5 | Uncertainty | ECE; OOD retention | coverage target | ECE <= 0.1 and OOD retained | 0.0482 | **PASS** |
| A6 | Causal | IPS known-effect recovery | random; table | E recovers effect (CI excludes 0) | diff -215.3 [-413.1,-12.6] | **FAIL** |
| A7 | Order/Path | event-shuffle drop | table (invariant) | E drops, table does not | drop -0.2785 | **PASS** |
| A8 | Relational/Context | cross-entity purity | chance | > chance | 0.005 vs chance 0.0005 | **PASS** |

## Objectives (self-supervised battery)

| # | Item | Metric | Comparator | Gate | Value | Verdict |
|---|---|---|---|---|---|---|
| O1 | next | linear probe next-event skill | naive (marginal) | beats naive | — | **NOT MEASURED** |
| O2 | dt | gap/rate probe skill | naive (prev gap) | beats naive | — | **NOT MEASURED** |
| O3 | jepa | latent future prediction | naive (hist mean) | beats naive | — | **NOT MEASURED** |
| O4 | mask | masked-event recovery | naive (marginal) | beats naive | — | **NOT MEASURED** |
| O5 | value | value probe skill | naive (hist mean) | beats naive | — | **NOT MEASURED** |
| O6 | entity | entity/identity probe | naive (marginal) | beats naive | — | **NOT MEASURED** |
| O7 | order | temporal-order probe | naive (random) | beats naive | — | **NOT MEASURED** |
| O8 | agg | aggregate/window probe | naive (hist mean) | beats naive | — | **NOT MEASURED** |
| O9 | query | query-conditioned probe | naive (hist mean) | beats naive | — | **NOT MEASURED** |
| O10 | sf | successor-feature probe | naive (prev value) | beats naive | — | **NOT MEASURED** |

## Verdict

B is **not yet mastered**: 13 row(s) unmeasured, 3 failing. Unmeasured rows need receipts; failing rows need an axis fix (one axis per run).

## Disclosures

- Synthetic harness; no business-value or causal claim beyond the randomized holdout.
- Some axis receipts predate DEC-280/281 (future-objective + ladder); refresh required before claiming them.
- Objective battery (`next/dt/jepa/mask/value/entity/order/agg/query/sf`) is defined in `looking_glass/probes.py` and not yet wired to this encoder -> NOT MEASURED.
