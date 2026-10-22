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

## Confirmed results — mass variants, pre-past-summary (superseded below)

| Δ from `table + seq` → `table + seq + E` | mean ± SE |
|---|---|
| explicit magnitude branch — value90 | +0.022 ± 0.023 |
| explicit magnitude branch — orders90 | +0.019 ± 0.011 |
| explicit magnitude branch — churn90 | −0.007 ± 0.002 |
| **high-purity `none`** — value90 | +0.021 ± 0.019 |
| **high-purity `none`** — orders90 | **+0.035 ± 0.005** |
| **high-purity `none`** — churn90 | −0.008 ± 0.003 |
| **high-purity `readout`** — value90 | +0.023 ± 0.017 |
| **high-purity `readout`** — orders90 | +0.032 ± 0.005 |
| **high-purity `readout`** — churn90 | −0.002 ± 0.007 |

- The pure SSM (`none`, no mass branch) adds the most on **orders** (+0.035, all seeds),
  no robust value gain, and slightly hurts churn.
- The learned attention mass (`readout`) is similar; the `state` variant was weak at 1 seed.
- At this stage (before the past-summary objective) E alone was far below the table
  (`none`: value 0.08, orders 0.08, churn 0.45). The past-summary section below supersedes
  this: E-only is now at parity with the principle-aligned table.

## Oracle ceiling (simple GRU on raw history)

| target | table | table + explicit seq | GRU sequence oracle |
|---|---|---|---|
| value90 R² | 0.337 | 0.415 | 0.226 |
| orders90 R² | 0.402 | 0.491 | 0.244 |

The GRU oracle is **weaker than the table**, so it is **not a valid ceiling**; it shows the
small GRU is underpowered, not that the data has no headroom. The practical sequence ceiling
is the explicit `table + seq` (0.415/0.491); the pure E adds +0.035 orders beyond it.
A stronger sequence model (or the custom SSM at larger capacity) is needed to establish the
true ceiling before deciding whether the reference data is the limit.

## Past-summary objective (all schema-derived) — 2k, 3 seeds, 20 epochs

Targets: **all** event-type counts, **all** categorical distinct counts, **all** numeric
count/sum/mean/std/min/max, timing/recency/tenure, dynamic query gap. **No fixed 30/90d
windows.** The table is a teacher only; no table/counter/summary inside `E`.

| 3-seed mean ± SE | E only | Table | E − table |
|---|---|---|---|
| value90 R² | 0.292 ± 0.032 | 0.344 | −0.053 |
| orders90 R² | 0.351 ± 0.021 | 0.388 | −0.036 |
| churn90 AUC | **0.959 ± 0.007** | 0.950 | **+0.009** |

`table + seq → table + seq + E` deltas: value **+0.024 ± 0.017** (not robust), orders
**+0.033 ± 0.011** (all seeds), churn **+0.009 ± 0.005** (all seeds). The E-only churn now
stands on its own (beats the table); value/orders are within ~0.04.

## Geometry + negative results (1 seed, 5 epochs)

- **Effective rank of `E` ≈ 2** (table ≈ 8). The E is a low-rank predictive manifold; not a
  full-rank representation. Outcome geometry is strong: kNN churn agreement 0.98 (random 0.88),
  value-neighbour rank 0.45, distance–churn Spearman 0.46.
- **Segment purity is not a valid geometry target**: `segment` is a **random** table column
  unused by the DGP. E purity 0.19 vs random 0.20 is the correct result; the table's 0.57 is
  fitting a random one-hot. Use the truth latents (`regime0`, `category_pref`,
  `price_sensitivity`) for geometry diagnostics instead.
- **Event-dropout contrastive (NCE, learned temperature): reject.** Downstream dropped
  (value 0.294→0.278, orders 0.361→0.344, churn 0.959→0.945) and rank barely moved (2.14→2.20).
- **Decayed multi-horizon past summary (data-derived τ = 2/24d): reject.** Value/orders dropped
  (0.294→0.262 / 0.361→0.325), churn flat. Many extra targets dilute rather than help.
- **VICReg (variance + decorrelation): reject as a rank fix.** 1 seed/5 epochs: E-only
  0.273 / 0.344 / 0.966, rank 2.09 (vs 2.14); no rank gain, value/orders down.
- Next geometry lever: latent-regime diagnostics; forcing rank is not working.

## Attribution sweep (1 seed / 5 epochs, same harness)

Base is `mass=explicit` (the argparse default): E-only 0.339 / 0.430 / 0.941. The prior
confirmed `mass=none` baseline (3 seeds/20) was 0.292 / 0.351 / 0.959, so the mass branch is a
real trade (value/orders up, churn down) and needs 3-seed confirmation.

| Change vs base | value90 | orders90 | churn90 | Interpretation |
|---|---|---|---|---|
| none (base) | 0.339 | 0.430 | 0.941 | — |
| `qgap` off | 0.150 | 0.257 | 0.620 | **critical** |
| Fourier periods off | 0.345 | 0.360 | 0.952 | periods drive orders (+0.070) |
| horizon heads off | 0.324 | 0.343 | 0.955 | horizon drives orders (+0.087) |
| next-event heads off | 0.327 | 0.415 | 0.977 | next drives value/orders; costs churn |
| past-summary off | 0.331 | 0.436 | 0.929 | past drives churn (+0.012) |
| `d=32` | 0.287 | 0.303 | 0.967 | capacity helps value/orders |
| `d=128` | 0.318 | 0.394 | 0.927 | no gain at this N |
| layers 1 | 0.247 | 0.324 | 0.972 | 2 layers help orders |
| layers 3 | 0.345 | 0.392 | 0.948 | 3 layers help value |
| `mass=readout` | 0.285 | 0.359 | 0.965 | worse than explicit |

Conclusions: `qgap`, data-derived periods, horizon/next-event objectives, and the past-summary
all have a measurable role; `d=64`/2 layers is a reasonable current point; `mass=explicit`
is the best value/orders screening arm but costs churn. The sweep is 1 seed, so treat deltas
<~0.02 as unresolved.

## Fair-table comparison — the E-only target is already met

Same holdout, table feature ablations (5-fold GBM):

| Table feature set | value90 | orders90 | churn90 |
|---|---|---|---|
| full 39 features | 0.349 | 0.352 | 0.964 |
| no random static columns | 0.364 | 0.377 | 0.961 |
| no fixed `value_30d/90d` | 0.274 | 0.355 | 0.963 |
| derivable + no fixed windows | 0.289 | 0.349 | 0.960 |
| **E only (3-seed)** | **0.292** | **0.351** | **0.959** |

- The raw table's **value edge is the fixed 30/90d windows**; without them the table is below
  the E-only (0.274 vs 0.292). The operator explicitly disallows fixed windows for `E`.
- The random static columns (`segment`, `loyalty_tier`) **hurt** the table on value/orders;
  removing them strengthens it (0.364/0.377/0.961). Against that stronger, noise-free table
  the E-only orders gap is +0.026 R² (3-seed 0.351 vs 0.377), value +0.072.
- Against the **principle-aligned** table (derivable, no hardcoded windows), E-only is at
  parity/above on all three (value +0.003, orders +0.002, churn −0.001).
- Honest claim: *E-only matches/beats the principle-aligned table; the raw-table value gap is
  fixed-window engineering, not a missing learned capability.*

## 6k Layer-0 confirmation (v3, 3 seeds / 20 epochs)

| Arm | value90 | orders90 | churn90 |
|---|---|---|---|
| raw table | 0.364 | 0.436 | 0.917 |
| fair table (no fixed windows) | 0.335 | 0.401 | 0.919 |
| hand `explicit` E | **0.388** | **0.457** | **0.939** |
| schema-agnostic auto-discovery E | **0.385** | **0.455** | **0.934** |

Both encoder arms beat the fair table on **all three** targets at 6k; the auto-discovery arm is
within noise of the hand-listed arm. E adds to `table+seq`: hand +0.021/+0.018/+0.017,
auto +0.021/+0.032/+0.012. The metric leg of Layer 0 is met at 6k. Remaining: v3 certificate,
repo prototype, transfer/geometry probes, derived capacity/LR, full generic heads/past.

## P8 — reference table is not fully derivable from the stream

- `segment` and `loyalty_tier` are **random** (`rng.integers`) customer columns; `segment` is
  absent from the stream, and the initial tier is not logged (only `loyalty_change` events).
- `signup_ts` is random 1–399 days **before** the event window, but `customer_signup` is emitted
  at simulation day 0. The table's `tenure_days = as_of − signup_ts`; the E's first-event tenure
  is understated by 1–399 days. The table therefore has hidden information (tenure ablation is
  small — value 0.349→0.344 — but the principle is violated).
- **Consequence:** in the original `relational_v1` stream, the table was a valid *predictive*
  baseline but not a **derived view of PATH B**; v3 fixes that. The certified Layer A receipt
  still refers to v1 and must be regenerated for v3.
- **Fixed in the generator (v3).** `customer_signup` now carries `segment`, `loyalty_tier`, and
  the true `signup_ts`; it stays at the window start so the 1–399 day pre-window gap is not
  treated as inactivity. PATH A is **byte-identical** (`sha256` match); PATH B now contains
  every table column. Rebuild: `/tmp/opencode/t0_loop/relational_v3`; re-certification pending.
- **Encoder inclusion.** The prototype now auto-discovers extra attribute fields (typed as
  categorical / numeric / timestamp) and applies a **learned per-field gate**; the data fix
  alone (v3) leaves the hand-listed E unchanged (value 0.296, orders 0.359, churn 0.959 at
  1 seed/5 epochs). Activating the extra table fields does **not** help at 2k/5 epochs
  (value 0.223–0.261, orders 0.303–0.345, churn 0.928–0.951), because `segment`/`loyalty_tier`
  are random and tenure is weakly predictive. Gates start off; the model may open them.
- Report the E-only claim against the **fair** table (derivable, no fixed windows); the raw-table
  value gap is fixed-window engineering, not hidden signal.
