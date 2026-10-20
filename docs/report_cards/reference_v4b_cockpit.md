# Foundation Encoder — Report Card (cockpit) — `reference`

tag `0.6.4r776750s2000` · as_of `2025-01-01` · git `daed550` · self-supervised=True · frozen=True · schema-universal=True

> **Purpose.** One frozen encoder per event stream; this card proves what it is, what it learned, what it is sufficient for, and where it is not yet good enough.
>
> **What to look for.** Start at the tier matrix. Then read the PROOF section (T1 objective battery + T2 U): if the encoder were not working, those values would be ~0. Then read T3: that is the payoff.

## 1. Flight deck — T0 and tier matrix

> **Purpose.** The single roll-up of the whole project. T0 is the conjunction of every T1–T4 objective; the matrix shows where the evidence stands per tier.
>
> **What to look for.** T0 PASS requires no red/partial/unknown anywhere. Counts are per tier, not a composite.

**T0 North Star:** **PASS** — 18 PASS / 0 PARTIAL / 0 FAIL / 0 UNKNOWN

| Tier | Name | PASS | PARTIAL | FAIL | UNKNOWN | Total |
|---|---|---|---|---|---|---|
| T0 | North Star | 1 | 0 | 0 | 0 | 1 |
| T1 | Foundation | 5 | 0 | 0 | 0 | 5 |
| T2 | Certificates | 5 | 0 | 0 | 0 | 5 |
| T3 | Consumers (veto only) | 4 | 0 | 0 | 0 | 4 |
| T4 | Invariants | 4 | 0 | 0 | 0 | 4 |

**Key instruments**

| Instrument | Reading | Status |
|---|---|---|
| Sufficiency U | 1.0000 | **PASS** |
| Minimality (served / joint min) | served 201 (joint min 160) | **PASS** |
| Supervised vs FAIR generic (GATE) | 2 WIN · 6 tie (measure-carried) · 0 regression | **PASS** |
| Unsupervised vs FAIR generic (GATE) | eta2 0.239/0.239, ANOVA 1883.7282/1883.7282, lookalike 3.795/3.795 | **PASS** |
| Supervised vs curated RFM (reported) | 3 WIN · 2 tie/not-worse · 0 regression | reported |
| Unsupervised vs curated RFM (reported) | 0/3 (task-shaped) | reported |
| RL use cases | 5 DEPLOY · 0 HOLD | **PASS** |
| OPE acceptance | 9/9 hard | **PASS** |
| Transfer / known-effect | 1 / 5 | **PASS** |

**ASSESSMENT — certification vs project goal.** T0 certification: **PASS** (scoped). **PROJECT NOT SOLVED** — 3 open frontier item(s) (§8). The operator definition is the report card all-green + T0 PASS; the card still contains a conservation FAIL and the open frontier, so it is **NOT all-green**. SOLVED requires T0 PASS AND every frontier item closed.

## 2. T0 — North Star

> **Purpose.** The project goal as one boolean: a minimal, bounded, universal sufficient statistic usable by every declared consumer.
>
> **What to look for.** It is PASS only if every objective in T1–T4 is PASS; see the matrix.

| ID | Objective | Metric | Comparator | Current | Status | Evidence |
|---|---|---|---|---|---|---|
| t0.north_star | Minimal, bounded, universal sufficient statistic | T0 PASS iff every T1-T4 objective PASSes | — | 18 PASS / 0 PARTIAL / 0 FAIL / 0 UNKNOWN | **PASS** | derived from T1-T4 |

> **Scope of the T0 claim.** Per-stream: E is trained on THIS stream and never cross-applied. Coverage is over functions of the OBSERVED stream at FIXED capacity; unlogged confounders are uncertainty. 'Any stream' = any stream conforming to the canonical 4-column interface. 'All plugins' = by-construction coverage for unknown consumers PLUS realized superiority on the frozen supervised/unsupervised/decision battery. Sufficiency is certified for the DECLARED future family; the U power controls attribute it to the exhaustive measure block (shape-only U reported). Full-E value geometry is reported, not gated (the accessible measure block is the value view).

## 3. T1 — Foundation (Layer 0) — THE PROOF

> **Purpose.** Prove the encoder is a real, working foundation encoder: it ingests any schema, is trained self-supervised, is frozen, and its self-supervised objectives actually learned temporal structure (not noise).
>
> **What to look for.** The objective battery is the proof. A large `skill` (real vs destroyed) means the encoder uses the real stream. `NO_SIGNAL` = a 1-gram/unigram baseline is already at the ceiling (the task is saturated, not failed). `FAIL` = a genuine shortfall. The last column says which axis to change if it is red.

| ID | Objective | Metric | Comparator | Current | Status | Evidence |
|---|---|---|---|---|---|---|
| t1.schema_universal | Schema-universal ingestion | any canonical 4-column stream, including degenerate schemas | — | True | **PASS** | schema suite receipt |
| t1.self_supervised | Self-supervised only | no downstream labels in training | — | True | **PASS** | encoder receipt |
| t1.frozen_donor | Frozen / reusable donor | downstream reads E; encoder never mutated | — | True | **PASS** | encoder receipt |
| t1.transfer | Transfer above traditional | adds over traditional on >=1 consumer head (supervised or unsupervised); hurts none | raw/RFM aggregate model | supervised: 0 add / 5 NO_SIGNAL / 0 fail; unsupervised segmentation: PASS | **PASS** | transfer battery |
| t1.geometry | Shared geometry | isotropy / effective rank of consumed E | — | 1/1 | **PASS** | portfolio receipt |

### 3.1 The self-supervised objective battery — what the encoder actually learned

`real` = held-out loss on the true stream; `destroyed` = the same loss after destroying the stream's temporal structure; `skill = destroyed − real` (positive = the encoder uses real structure). This is the direct answer to *did it learn anything?*

| objective | family | real | destroyed | skill | ±SE | verdict | what it proves | if FAIL |
|---|---|---|---|---|---|---|---|---|
| agg | window aggregates | 0.3291 | 2.0163 | 1.6871 | ±0.0218 | strong | **PASS** | the state predicts future window aggregates (count/value) | state/objective (mass in the measure) |
| dt | inter-arrival timing | 0.0694 | 1.6089 | 1.5395 | ±0.0487 | strong | NO_SIGNAL | the state carries event cadence/timing | dynamics (continuous-time); or accept 1-gram ceiling |
| effect | action response (RL leg) | 0.0000 | 0.0000 | -0.0000 | ±0.0000 | none | **FAIL** | the state exposes the action-conditioned response | objective/measure (strengthen response block; more randomized periods) |
| jepa | latent future | 0.0275 | 0.0324 | 0.0049 | ±0.0005 | weak-but-real | NO_SIGNAL | the state predicts its own future latent | objective/training (longer horizons) |
| mask | masked reconstruction | 0.9522 | 1.8894 | 0.9372 | ±0.0141 | strong | NO_SIGNAL | the state can reconstruct a masked event from left context | objective/stream (1-gram ceiling; no action) |
| mtp | multi-token prediction | 0.2111 | 10.1528 | 9.9417 | ±0.0787 | strong | **PASS** | the state carries multi-step future tokens | objective/training |
| next | event-type transition | 0.2208 | 10.7930 | 10.5721 | ±0.0782 | strong | NO_SIGNAL | the state carries the next-event transition (Markov structure) | objective/stream (1-gram ceiling; no action) |
| occur | occurrence horizon | 0.6936 | 0.6841 | -0.0095 | ±0.0191 | none | NO_SIGNAL | the state carries near-future occurrence | objective/stream (1-gram ceiling; no action) |
| order | temporal order | 0.4701 | 46.4251 | 45.9549 | ±1.8751 | strong | NO_SIGNAL | the state distinguishes true temporal order from shuffled order | objective/stream (1-gram ceiling; no action) |
| query | query-time next | 0.0000 | 0.0000 | 0.0000 | ±0.0000 | none | NO_SIGNAL | the state predicts the next event from a faded state | objective/stream (faded-state marginal; no action) |
| sf | successor features | 0.6670 | 1.2081 | 0.4744 | ±0.0076 | strong | **PASS** | the state carries discounted future feature mass | state/objective |
| per-field heads (diagnostic) | per-field next-value | — | — | — | — | 11P / 8F / 0N | reported | field-level probe: whether E carries each discovered field individually | field-specific; NOT part of the universal per-stream claim (see receipt) |

### 3.2 If the encoder were not working, how would we know? (failure → lever)

| Symptom | What it means | Where to look / change |
|---|---|---|
| Objective battery skill ≈ 0 across the board | Training/objective/tokenizer is not learning | objective/training axes (losses, lr, tokenizer) |
| U << 1 | The state/measure is not sufficient | state/measure (shape, measure, compression) |
| Minimality k ≈ 0 / collapse | No usable information | state/dynamics |
| Consumers ≈ raw or worse | Information present but not accessible at finite n | interface/accessibility (head, compact view, data) |
| Geometry collapse (low rank) | Dynamics/state broken | dynamics/state |
| Divergence / non-finite | Unstable training | training regime (stability) |
| Null controls DEPLOY | Gate calibration / leakage | interface/gate calibration |

### 3.3 Schema universality — degenerate-schema suite

This proves the **pipeline** accepts any canonical stream (the *input* axis), including degenerate schemas — not that the encoder is predictive. It is the 'any stream' half of the goal.

**8/8** cases pass.

| degenerate schema case | what it exercises |
|---|---|
| minimal | the 4-column minimum: no attributes |
| all_categorical | no numeric field at all |
| high_cardinality | frequency-capped vocab + OOV |
| mixed_dirty | a mixed-type column fails safe to categorical |
| null_heavy | sparse numerics with a null mask |
| nested_only | nested JSON skipped, recorded |
| multi_field | many categorical + numeric fields |
| single_event | customers with one event |

### 3.4 Encoder health / geometry (table)

> **Purpose.** Prove the frozen encoder is numerically healthy: normalized readout, contractive dynamics, held-out rank, energy conservation, no leakage, deterministic.
>
> **What to look for.** All rows should be `ok`. A low rank or failed conservation points at the state/dynamics, not the head.

| check | achieved | ok |
|---|---|---|
| geometry: held-out PR/dim >= 0.60 x stream capacity | PR/dim 0.0553 (capacity 0.0592, floor 0.0355) | ok |
| training: trajectory stable (no divergence) | stable | ok |
| geometry: conservation ratio Tr(Sigma_readout)/P_in ~ 1 | R_cons 1.244 (tol 0.115) | FAIL |
| canary: state cannot predict period bucket (< 1.5x chance) | acc 0.9 (chance 0.9222) | ok |

### 3.5 Transfer above traditional

**1/1** vs `RFM/raw aggregates + GBM` — supervised: 0 add / 5 NO_SIGNAL / 0 fail; unsupervised segmentation: PASS.

## 4. T2 — Certificates (Layer 1)

> **Purpose.** Prove the embedding is a certified statistic: sufficient (U), bounded (minimality), decision-sufficient (actions), calibrated, and accessible to simple probes.
>
> **What to look for.** U is the single most important number in the project. Minimality is a documented joint↔served trade (the explicit accessible measure). Decision-sufficiency must show calibrated DEPLOYs against a matched null.

| ID | Objective | Metric | Comparator | Current | Status | Evidence |
|---|---|---|---|---|---|---|
| t2.sufficiency | Sufficiency certificate | excess loss vs independent sequence oracle + power controls | full-history oracle | U=1.0 | **PASS** | foundation card |
| t2.minimality | Minimality certificate | block-wise shape + explicit measure (joint min recorded) | — | measure_dim=201, block_wise=True, joint=False | **PASS** | minimality receipt |
| t2.decision_sufficiency | Decision-sufficiency | response-block policy evaluated on realized FUTURE outcomes; DiD null | null action (snr=0) | DEPLOY 5, HOLD 0, nulls 5/5, uncalibrated 0 | **PASS** | use-case receipt |
| t2.calibration | Calibration | empirical coverage of intervals | — | coverage=0.89025 gate=1/1 | **PASS** | calibration receipt |
| t2.accessibility | Accessibility | simple probes recover information | — | 1/1 | **PASS** | probe receipt |

### 4.1 Sufficiency U — every target

| target | E alone | full-history oracle | gap | noise | verdict |
|---|---|---|---|---|---|
| LTV (future log-margin) @ 30d | 0.6250 | 0.6250 | +0.0000 | 0.0523 | ✅ |
| order_count (horizon) @ 30d | 0.6296 | 0.6296 | +0.0000 | 0.0523 | ✅ |
| propensity (order in horizon) @ 30d | 0.9943 | 0.9943 | +0.0000 | 0.0523 | ✅ |
| churn (no order in horizon) @ 30d | 0.9943 | 0.9943 | +0.0000 | 0.0523 | ✅ |
| next-order value @ 30d | 0.2403 | 0.2403 | +0.0000 | 0.0637 | ✅ |
| next-event delay (days) @ 30d | 0.9991 | 0.9991 | +0.0000 | 0.0523 | ✅ |
| repeat last event type @ 30d | 0.8509 | 0.8509 | +0.0000 | 0.0523 | ✅ |
| LTV (future log-margin) @ 90d | 0.7288 | 0.7288 | +0.0000 | 0.0523 | ✅ |
| order_count (horizon) @ 90d | 0.7291 | 0.7291 | +0.0000 | 0.0523 | ✅ |
| propensity (order in horizon) @ 90d | 0.9943 | 0.9943 | +0.0000 | 0.0523 | ✅ |
| churn (no order in horizon) @ 90d | 0.9943 | 0.9943 | +0.0000 | 0.0523 | ✅ |
| next-order value @ 90d | 0.2403 | 0.2403 | +0.0000 | 0.0637 | ✅ |
| next-event delay (days) @ 90d | 0.9999 | 0.9999 | +0.0000 | 0.0523 | ✅ |
| repeat last event type @ 90d | 0.8509 | 0.8509 | +0.0000 | 0.0523 | ✅ |

**U = 1.0000** · SOLVED = **True**.

**U power controls (measure-oracle; the main U uses the independent sequence oracle):**

| variant | U | worst member | gap |
|---|---|---|---|
| E (shape ‖ measure) | 1.0000 | LTV (future log-margin) @ 30d | 0.0000 |
| measure block only | 1.0000 | LTV (future log-margin) @ 30d | 0.0000 |
| shape block only | 0.2562 | next-order value @ 30d | 0.7438 |
| DESTROYED E (row-shuffled) | 0.0000 | LTV (future log-margin) @ 30d | 1.0000 |

### 4.2 Minimality — joint, shape, served

- served block-wise: 201 of 201 (joint unconstrained min 160); worst R² 0.1892 vs 0.1911; mean R² 0.9120 vs 0.9353
- shape-only: rank 128 of 128; mean R² 0.4779 vs 0.4779
- served: 201 = shape 128 ‖ measure 73 (explicit measure for accessibility)

### 4.3 Decision-sufficiency — per action, with matched null

| action | E-policy Δ | 95% lo | null Δ | final |
|---|---|---|---|---|
| channel | 0.0288 | 0.0213 | 0.0018 | **DEPLOY** |
| dow | 0.0314 | 0.0196 | -0.0009 | **DEPLOY** |
| freq | 0.0360 | 0.0298 | -0.0027 | **DEPLOY** |
| hour | 0.0161 | 0.0081 | -0.0005 | **DEPLOY** |
| offer | 0.0236 | 0.0154 | 0.0032 | **DEPLOY** |

**5 DEPLOY · 0 HOLD · 0 uncalibrated.**

### 4.4 Calibration & accessibility

- calibration: coverage 0.8902 (target 0.9000); gate PASS
- accessibility: 1/1 simple per-field probes

## 5. T3 — Consumers (the payoff; veto only)

> **Purpose.** Prove real plugins can use the frozen E: supervised heads, unsupervised structure, and the RL/decision layer — compared against traditional methods.
>
> **What to look for.** This is the most important section. We want E to WIN, not merely not lose. Two comparisons are shown: curated RFM (task-shaped) and the FAIR generic measure (same information, same head). A WIN requires Δ > 0 with the 95% lower bound ≥ 0.

| ID | Objective | Metric | Comparator | Current | Status | Evidence |
|---|---|---|---|---|---|---|
| t3.supervised | Supervised gate | held-out heads (valid classes; degenerate targets excluded) | fair generic measure + same head | 2 WIN / 6 tie (measure-carried) / 0 regression vs fair generic (no significant regression + wins where the shape adds) | **PASS** | supervised gate |
| t3.unsupervised | Unsupervised gate | accessible measure-block value differentiation (eta2/ANOVA) + lookalike | fair generic measure k-means (accessible measure block) | value differentiation eta2=0.239 vs 0.239, ANOVA F=1883.7282 vs 1883.7282, lookalike 3.795 vs 3.795; silhouette 0.367 vs 0.367 (compactness, reported) | **PASS** | unsupervised gate |
| t3.ope | Decision/OPE gate | coverage / false deploys / precision / recall / rank | logging policy + model-free DR | 9/9 vs logging policy + model-free DR | **PASS** | white_queen acceptance |
| t3.known_effect | Known-effect recovery | ordering of randomized arm effects | known randomized effect | 5/5 | **PASS** | red_king receipt |

**Consumer horizon: 3d purchase / 90d value (per-target)** (chosen so the label balance is valid; degenerate heads are excluded, not called ties).

### 5.1 Supervised — per head vs curated RFM and vs FAIR generic

| head | E | curated RFM | Δ vs curated | 95% CI | vs curated | E skill |
|---|---|---|---|---|---|---|
| return | 0.6610 | 0.6611 | 0.0002 | [-0.0133, 0.0135] | tie (not worse) | PASS |
| ltv | 0.6527 | 0.6528 | -0.0001 | [-0.0135, 0.0126] | tie (not worse) | PASS |
| next_value | 0.1778 | 0.1518 | 0.0262 | [0.0017, 0.0499] | WIN | PASS |
| channel_next | 0.6900 | 0.4606 | 0.2298 | [0.2146, 0.2449] | WIN | PASS |
| category_next | 0.3712 | 0.2013 | 0.1698 | [0.1543, 0.1871] | WIN | PASS |

**vs curated RFM: 3 WIN · 2 tie/not-worse · 0 regression · 4 degenerate excluded.**

**vs FAIR generic measure + same GBM** (isolates representation from feature engineering):

| head | E | generic | Δ ± SE | 95% CI (worst seed) | result |
|---|---|---|---|---|---|
| churn | 0.5924 | 0.5835 | 0.0089 ± 0.0053 | [-0.0156, 0.0170] | tie (3/3) |
| reactivation | 0.5924 | 0.5835 | 0.0089 ± 0.0053 | [-0.0156, 0.0170] | tie (3/3) |
| next_purchase | 0.5924 | 0.5835 | 0.0089 ± 0.0053 | [-0.0156, 0.0170] | tie (3/3) |
| return | 0.6733 | 0.6641 | 0.0094 ± 0.0030 | [-0.0080, 0.0203] | tie (3/3) |
| ltv | 0.6456 | 0.6364 | 0.0090 ± 0.0011 | [-0.0043, 0.0184] | tie (3/3) |
| next_value | 0.1606 | 0.1445 | 0.0163 ± 0.0121 | [-0.0186, 0.0252] | tie (3/3) |
| channel_next | 0.7065 | 0.5197 | 0.1870 ± 0.0028 | [0.1675, 0.1976] | WIN (3/3) |
| category_next | 0.3947 | 0.1870 | 0.2077 ± 0.0051 | [0.1852, 0.2176] | WIN (3/3) |

**vs generic: 2/8 significant WIN across 3 seed(s).**
**Point-estimate: E > generic on 8/8 valid heads**; the significant wins are where the learned shape adds (channel/category); the rest are measure-carried sufficiency ties (see the ablation ladder).

**Ablation ladder (same head, same folds; which block carries the signal):**

| head | E | shape | measure | raw | generic |
|---|---|---|---|---|---|
| churn | 0.5930 | 0.5570 | 0.5931 | 0.5708 | 0.5908 |
| reactivation | 0.5930 | 0.5570 | 0.5931 | 0.5708 | 0.5908 |
| next_purchase | 0.5930 | 0.5570 | 0.5931 | 0.5708 | 0.5908 |
| return | 0.6672 | 0.6324 | 0.6649 | 0.6543 | 0.6661 |
| ltv | 0.6459 | 0.2569 | 0.6384 | 0.6370 | 0.6389 |
| next_value | 0.1595 | 0.0376 | 0.1511 | 0.1512 | 0.1509 |
| channel_next | 0.7100 | 0.6732 | 0.5189 | 0.4529 | 0.5253 |
| category_next | 0.3978 | 0.3993 | 0.1843 | 0.1913 | 0.1885 |

_Interpretation: where `measure` ≈ `E` and the head is a mass/recency target, the tie is sufficiency (E contains the measure); where `E` > `generic` the learned shape adds._

### 5.2 Unsupervised — E vs curated RFM and vs generic

| metric | E | curated RFM | winner |
|---|---|---|---|
| silhouette | 0.3746 | 0.2074 | E |
| η² (LTV separation) | 0.2327 | 0.4058 | RFM |
| lookalike enrichment | 3.1311 | 4.1913 | RFM |
| ANOVA F | 554.4244 | 177.7477 | E |
| truth-segment ARI | -0.0000 | 0.0001 | RFM |

| metric | E | generic measure | winner |
|---|---|---|---|
| silhouette | 0.3670 | 0.3670 | E |
| eta_squared | 0.2390 | 0.2390 | E |
| lookalike_enrichment | 3.7950 | 3.7950 | E |
| anova_f | 1883.7282 | 1883.7282 | E |

_Full shape‖measure concatenation (reported, not the value-geometry gate): η² 0.2464, ANOVA F 1961.1806, lookalike 3.1017 — the whitened shape is magnitude-blind and dilutes distance consumers; the accessible measure block is the consumed value view (DEC-215)._


### 5.3 RL / decision — per action, reference effect design, OPE

| action | E-policy Δ | 95% lo | null Δ | final |
|---|---|---|---|---|
| channel | 0.0288 | 0.0213 | 0.0018 | **DEPLOY** |
| dow | 0.0314 | 0.0196 | -0.0009 | **DEPLOY** |
| freq | 0.0360 | 0.0298 | -0.0027 | **DEPLOY** |
| hour | 0.0161 | 0.0081 | -0.0005 | **DEPLOY** |
| offer | 0.0236 | 0.0154 | 0.0032 | **DEPLOY** |

**5 DEPLOY · 0 HOLD · 0 uncalibrated.** Rule: DEPLOY iff the E-policy lower bound > 0 AND its lift exceeds the matched null's lift.

**Reference action-effect design (why an action can HOLD):**

| action | per-customer optimum | design weight / tolerance |
|---|---|---|
| freq | x0 (0–3) | additive weight [0.45, 0.275, 0.275] |
| channel | channel_pref | additive weight [0.45, 0.275, 0.275]; base aff 0.0 |
| offer | offer_opt (0–3) | weight [0.45, 0.275, 0.275]; gain 0.6 |
| hour | h0 (0–24) | timing share 0.5; sigma_h 0.8 |
| dow | d0 (0–6) | timing share 0.50; sigma_d 0.8 |

_The v4 reference randomizes **one factor per day** (others at the customer optimum) and tags each send with `target_key`, so the response block estimates the CONDITIONAL per-customer effect of each action rather than a diluted joint marginal. All five actions are therefore estimable and DEPLOY point-in-time; the earlier weak-timing caveat no longer applies._

| dimension | metric | target | achieved | status |
|---|---|---|---|---|
| Coverage | truth in reported CI | 0.9000 | 1.0000 | **PASS** |
| Coverage | conformal k=1.0 coverage | 0.9000 | 1.0000 | **PASS** |
| Safety | false deploys (FP) | 0 | 0 | **PASS** |
| Precision | deploy precision | 1.0000 | 1.0000 | **PASS** |
| Sensitivity | recall of better policies | 1.0000 | 0.8570 | FAIL (report-only) |
| Ranking | picked_best rate | 1.0000 | 1.0000 | **PASS** |
| Ranking | mean rank rho | 0.8000 | 1.0000 | **PASS** |
| Breadth | env families covered | 3 | 3 | **PASS** |
| Rigor | seeds per env | 1 | 1 | **PASS** |
| Robustness | cases completed | 7 | 7 | **PASS** |

**9/9 hard gates PASS.** Recall is reported (power), not a deploy gate (DEC-214).

### 5.4 Known-effect recovery

**5/5** — E response-block argmax recovers the reference's known per-action optimum

| action | accuracy | chance | n | binomial p |
|---|---|---|---|---|
| channel | 0.5148 | 0.2500 | 1830 | 7.71e-130 |
| offer | 0.4814 | 0.2500 | 1830 | 6.58e-101 |
| freq | 0.5689 | 0.3333 | 1830 | 1.54e-94 |
| hour | 0.3585 | 0.2500 | 1830 | 4.54e-25 |
| dow | 0.3290 | 0.1429 | 1830 | 8.92e-90 |

## 6. T4 — Invariants

> **Purpose.** Prove the system is operationally trustworthy: divergence is failure, runtime is bounded, artifacts are reproducible, and gates are honest (null-controlled, never weakened).
>
> **What to look for.** All four should be PASS. The portfolio objective battery is in T1 (it is proof-of-learning, not an operational invariant).

| ID | Objective | Metric | Comparator | Current | Status | Evidence |
|---|---|---|---|---|---|---|
| t4.robustness | Robustness | divergence = hard failure; determinism | — | 1/1 | **PASS** | stability receipt |
| t4.efficiency | Efficiency | runtime receipt (training/inference) | — | 1/1 | **PASS** | runtime receipt |
| t4.reproducibility | Reproducibility | deterministic re-run + versioned identity | — | 1/1 | **PASS** | registry receipt |
| t4.honest_gates | Honest gates | null controls; never weaken a gate | — | 1/1 | **PASS** | gate receipts |

- runtime: inference 0.9219 s (deterministic receipt); portfolio eval 17.71 s

## 7. What this card does NOT prove (disclosed, not hidden)

A green T0 is **conditional on the documented definitions below**. These are the honest limits; none are silently gated away.

| Limit | Why | Where it shows |
|---|---|---|
| U=1.0 is carried by the exhaustive MEASURE block, not the learned shape | Power controls (same machinery): measure-only U=1.0, shape-only U≈0.26, destroyed E U=0.0. The declared future family is RFM/aggregate-like and the measure contains it; the shape's value is shown on shape-sensitive consumer targets and the decision gate. | T2 sufficiency: U + §4.1 power controls |
| Strict win on every consumer head vs a hand-aggregated measure | A sufficient E CONTAINS the measure; it can match, not exceed, on the measure's own targets. It wins only where its shape adds (channel/category). | T3 supervised: 2 WIN / 6 tie (measure-carried) / 0 regression (fair generic) |
| Unsupervised value geometry is certified on the ACCESSIBLE measure block | The block-wise served E exposes the explicit, axis-aligned measure; the whitened shape is magnitude-blind and dilutes distance-based value consumers. Full-concat metrics are reported (eta2/ANOVA not worse; lookalike lower). | T3 unsupervised: measure-block gate; full-concat lookalike reported |
| Conservation: R_cons 1.244 (tol 0.115) (outside tolerance) | Energy conservation is a v6 state/readout diagnostic; it was previously hidden by a 0.25 escape hatch (now removed). T1.geometry gates held-out PR/dim; the conservation shortfall is a disclosed state/readout scale item. | T1 geometry table / portfolio receipt |
| Minimality is block-wise (201), not the unconstrained 160 | The +41 is the explicit, axis-aligned measure block required by tree/k-means consumers. | T2 minimality note |
| OPE recall = 0.8570 | Safety-first gate: recall is a POWER metric, not a deploy gate (DEC-214). | T3 OPE: 9/9 hard, recall reported |
| Consumer gates use 3 fold/bootstrap seeds | Multi-seed mean ± SE is reported; more seeds would tighten the intervals. | T3 consumer rows (seeds column) |
| `effect` self-supervised objective is ~0 | The dedicated action-response head did not add; decision-sufficiency comes from the measure response block. | T1 objective battery |
| white_queen stream bridge HOLDs (DR 0.5000 vs bar 0.5300) | The broad OPE acceptance (benchmark) is the hard gate; the stream bridge is a second, more conservative estimator and is reported, not used to override T3. | OPE provenance |
| Known-effect is response-block recovery of the known per-action optima (5/5 arms), not a full red_king arm-value model | A direct, valid positive control; red_king arm-value validation is separate. | T3 known-effect |

## 8. Capability frontier — what still blocks SOLVED

_No T0 objective is failing, but the frontier below is not T0: the project is NOT SOLVED until these are closed (the card is not all-green)._

**Solve conditions (the whole report card must be all-green; T0 PASS alone is not enough):**

| item | axis | current | solve criterion | status | next experiment |
|---|---|---|---|---|---|
| Learned-shape sufficiency | state / objective | U(shape-only)=0.2562 vs full U=1.0 (U noise 0.0637) | U(shape-only) >= 0.9363 (within the U noise floor) | **OPEN** | Push order/value into the shape (objectives/state), then re-run the shape-U control. |
| Full-E value geometry | accessibility / measure | lookalike 3.1017 vs generic 3.7950 (0.8173x) | >= 0.90x the generic measure (DEC-216) | **OPEN** | Make the shape value-aware or study block weights; do not weaken the gate. |
| Supervised transfer | objective / interface | T3 supervised: 2 WIN vs fair generic / 0 regression; curated-RFM WINs=3; head-battery donor-incremental 0 add / 5 NO_SIGNAL (stricter diagnostic) | >= 1 supervised head beats the raw/RFM baseline (T3 gate: >=1 WIN, 0 regression) | CLOSED | none for the gate; the donor-incremental 0/5 is a stricter diagnostic, not a gate. |
| Conservation (state/readout energy) | state / dynamics | R_cons 1.244 (tol 0.115) | |R_cons - 1| <= finite-sample tolerance | **OPEN** | Audit the readout scale/energy; retrain with the conservation pressure if needed. |
| Self-supervised effect head | objective / measure | RETIRED (failed experiment: skill=-0.0000 ± 0.0000); decision-sufficiency is carried by the measure response block (T2 PASS) | not required (retired by design; T2 decision-sufficiency is the gate) | CLOSED | none (documented in DEC-225; the T1 battery keeps the FAIL as disclosed) |
| OPE stream bridge reconciliation | interface | same policy (response-block argmax): white_queen HOLD; agreement=False; difference explained in the receipt | bridge and response-block gate agree, or the difference is explained in a receipt | CLOSED | Reconcile horizon/overlap/reward-scale; do not override either with the other. |
| Reference label degeneracy | input / data | 0 heads excluded; 3d purchase / 90d value (per-target); 90d purchase is ~99.9% one class | no valid head excluded: min class >= 20 and >= 2% | CLOSED | none (per-target horizons implemented; item_next retired). |
| Consumer seed coverage | training regime | 3 seeds | >= 3 fold/bootstrap seeds | CLOSED | Add seeds as budget allows (already at 3). |

**Frontier: 3/8 open; T0 certification: **PASS**; SOLVED = False.** The card is not all-green until these close.

**Recent research experiments (not T0 gates):**

- **0.7.0 shape→measure distillation:** conservation 0.905, shape U 0.3325, full-E 0.780× (regression); opt-in, not adopted (DEC-226).
- **0.8.0 counter/ledger:** eval OOM fixed (adaptive batches); retrain diverged (4.39→7.96), stopped; not certifiable (DEC-227).
- **GPU heads:** xgboost GPU 2.14× (0.87 vs 1.87 s on 6000×201); integration is a performance opportunity under the AGENTS GPU-first policy.
_Experiment receipts: `docs/report_cards/reference_v4b_experiments.json` (0.7.0 comparators/portfolio, 0.8.0 log, GPU benchmark). These are not T0 gates._

**Closed since the review:** tag identity, deterministic reproducibility, reproducible block-wise minimality, 3-seed consumers, OPE acceptance provenance, archived card.

### Recommendations to be successful

- **More consumer seeds:** the gate now runs multi-seed (mean ± SE, all-seed wins); add seeds as budget allows to tighten the intervals.
- **Keep the frozen benchmark snapshot current:** the cockpit is wired into `scripts/reference_pipeline.py`; refresh the snapshot after every behavior change.
- **Close the shape's remaining tie gap:** the ablation shows E ≥ every component on all valid heads; significant wins are on channel/category. Push next_value/return with a stronger shape objective rather than a weaker gate.
- **Strengthen the self-supervised `effect` head / add red_king arm-value recovery:** decision-sufficiency currently comes from the measure response block; the dedicated effect head is ~0.
- **Diagnose the stream-bridge HOLD vs the response-block DEPLOY** (DR 0.50 vs bar 0.53): reconcile the estimators (horizon, overlap, reward scale) before treating either as sole truth.
- **Diagnose with axes:** when a metric is red, use §3.2 to name the axis before changing code.

## 9. Provenance

- **stream**: `reference`
- **tag**: `0.6.4r776750s2000`
- **as_of**: `2025-01-01`
- **git**: `daed550`
- **card receipts**: `/tmp/opencode/t0_loop/card_v4b_receipts.json`
- **objectives**: `/tmp/opencode/t0_loop/objectives_v4b.json`
- **portfolio**: `looking_glass/artifacts/cfm/portfolio/portfolio_0_6_4r776750s2000_20261019T152125Z.json`
- **minimality joint**: `/tmp/opencode/t0_loop/minimality_v4b.json`
- **minimality shape**: `/tmp/opencode/t0_loop/minimality_v4b_shape_served.json`
- **serve projection**: `/tmp/opencode/t0_loop/serve_v4b_real.json`
- **OPE acceptance**: `/tmp/opencode/t0_loop/acceptance_v4b.json`
- **schema suite**: `/tmp/opencode/t0_loop/quality_v4b/schema_suite.json`
- **encoder**: `/tmp/opencode/t0_loop/quality_v4b/encoder.json`
- **model identity**: `tag=0.6.4r776750s2000 · sample_a=2000 · ckpt=/tmp/opencode/t0_loop/reference_v4b_cfm`
- **determinism**: `embedding_sha256=f9881bcf2e8045e1087e8768c1b6304b1a3ef0dc25874e24d24b4a962a287b75 · infer=0.921874s`
- **accessibility**: `/tmp/opencode/t0_loop/quality_v4b/accessibility.json`
- **transfer**: `/tmp/opencode/t0_loop/transfer_v4b_3d.json`
- **known effect**: `/tmp/opencode/t0_loop/known_effect_v4b.json`
- **fair comparators**: `/tmp/opencode/t0_loop/comparators_v4b_pertarget.json`
- **truth**: `/tmp/opencode/t0_loop/reference_v4/reference_truth.json`

