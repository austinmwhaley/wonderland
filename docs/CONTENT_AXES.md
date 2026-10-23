# The content axes — one lens for the encoder

The encoder is judged as a **content** object: what information about the stream it carries.
There are six content axes. They are independent enough to measure and diagnose separately, and
every encoder/objective/readout decision should say which axis it serves.

The lens inside each axis is **direction × magnitude** (the polar factor).

| Axis | Question | Correct algebra | Mechanism |
|---|---|---|---|
| **Shape** | What is the normalized pattern/distribution? | bounded / contractive | contractive SSM, normalized readout |
| **Mass** | How much (counts, value, totals)? | additive / extensive | additive memory, raw-level readout |
| **Identity** | Who/what is the entity? | content-addressed | key/field embeddings, ID tokens |
| **Rate** | When/how often (gaps, recency, cadence)? | time/rate | continuous-time gates, learned τ, gaps |
| **Uncertainty** | How sure; what is missing/unlogged? | calibrated | masks, uncertainty heads, calibration |
| **Causal** | What happens under `do(a)`? | counterfactual | action-conditioned heads, OPE |
| **Order/Path** | In what order, with what transitions and memory? | sequence/order-sensitive | SSM order, transitions, long memory |
| **Relational/Context** | How do entities relate (product, campaign, order, session, support)? | relational/graph | cross-entity tokens, linkage, baskets |

**Eight content axes.** The earlier six were the information content; order/path and
relational/context are separate because the table's aggregates are order-insensitive and
relation-blind, while the stream carries both. Regimes are shape+rate+order; sessions/baskets are
relational.

These are **content axes**. They are not the foundation properties (task-free, universal/per-stream,
lawful, minimal, robust, artifact, donor, interpretable, efficient), which are the behavior contract.

## Axis definitions, metrics, and failure symptoms

### Shape
- **What:** event-type mix, category/brand mix, transition patterns, regime structure, normalized
  past/future summaries.
- **Metrics:** normalized next-event/future/past summary R²; regime NMI/ARI; retrieval and
  churn-neighbor agreement; shape-only oracle gap; JS/entropy of predicted distributions.
- **Failure:** low regime NMI, poor retrieval, predictions that only track the average mix.

### Mass
- **What:** extensive levels — order count, value, margin, qty, support/returns, recent totals.
- **Metrics:** R²/MAE on raw future/past totals (orders90, value90, margin); recency-weighted
  totals at data-derived horizons; mass additivity error; scale calibration.
- **Failure:** good shape/ranking but wrong level — exactly the current orders90 gap.

### Identity
- **What:** customer/entity identity, product/campaign/cohort identity, entity linkage.
- **Metrics:** kNN same-entity/product/campaign accuracy; cohort purity; static-attribute
  reconstruction where derivable; cross-customer retrieval.
- **Failure:** customers/products collapse; retrieval mixes unrelated entities.

### Rate
- **What:** inter-event gaps, frequency, recency, time-to-next-purchase, churn timing, seasonality.
- **Metrics:** time-to-next-event/next-purchase MAE; gap distribution (KS/Wasserstein); churn AUC;
  hazard/recency calibration.
- **Failure:** timing is averaged; recency and churn drift.

### Uncertainty
- **What:** missingness, unlogged events, confidence, OOD/novelty.
- **Metrics:** ECE/Brier; interval coverage; OOD AUC; performance under masking/dropout; correlation
  between uncertainty and error.
- **Failure:** confident wrong predictions; silent collapse under missing data.

### Causal
- **What:** action-conditioned outcomes, counterfactual effects, decision sufficiency.
- **Metrics:** action-conditioned/IPS/DR policy value with randomized holdout; known-effect
  recovery; decision-sufficiency gap.
- **Failure:** predicts the logged future but cannot answer “what if”.

### Order/Path
- **What:** event order, transition structure, path dependence, long memory.
- **Metrics:** transition-matrix NLL; event-shuffle sensitivity (E should degrade, table should not);
  long-memory autocorrelation; path/regime recovery.
- **Failure:** E matches a bag-of-events table, so order carries no signal.

### Relational/Context
- **What:** cross-entity relations — product/category/campaign/order/session/support, hierarchy.
- **Metrics:** entity linkage/retrieval; basket/session reconstruction; cross-entity prediction
  (e.g., campaign → product); relation-aware retrieval.
- **Failure:** entities collapse or relations are lost by per-customer aggregation.

## B report card layout

The B card is the **axis scorecard + foundation properties**. It is not the B+C utility card.

1. **Header:** data/version, E config, hash, N, seeds, runtime, synthetic-harness label.
2. **Axis scorecard:** one row per axis — primary metric, baseline/comparator, CI, status
   (working / partial / not working), current lever.
3. **Foundation properties:** sufficiency/oracle gap, accessibility (linear/kNN), minimality/
   boundedness, lawful composition/equivariance, stability/adaptability, efficiency, artifact,
   donor controls (raw/random/destroyed).
4. **Disclosures:** axes that are table-favourable or unmeasured.

The **B+C card** is separate: frozen E + consumer heads; utility vs the traditional table; paired
CIs. The B+C validates utility; it does not define the axes.

## Current status (honest)

| Axis | Status | Evidence |
|---|---|---|
| Shape | partial/good | geometry/retrieval; regime NMI below table |
| Mass | **not working** | learned E orders90 0.273 vs table 0.352; the hand summary was the margin |
| Identity | partial | churn-neighbor robust; product/campaign identity not measured |
| Rate | partial | churn 2k/6k; gaps; recency not isolated |
| Uncertainty | partial | dropout/ECE measured; OOD not |
| Causal | not working | RL channel table wins; offer point only |

## Next axis work

**Mass + rate** is the current bottleneck: additive mass channel with learned decay/timescales,
raw-level future/past totals, unnormalized readout, shape channel kept contractive. Then identity,
uncertainty, causal probes. If the table still dominates after mass+rate, the reference testbed is
the limit (P2), not the encoder.
