# Foundation model — everything it must do well

## 1. What it is
One self-supervised encoder trained **per stream**, frozen, that turns the raw event stream into a
universal donor. “Universal” = the *pipeline* adapts to any schema; the model is never cross-applied.
It must be a **sufficient, accessible, lawful, robust, transferable, decision-sufficient, minimal**
representation of the stream — judged against the **full history** and a **traditional wide table +
GBM/XGBoost**, never against hand-picked tasks alone.

## 2. The eight content axes (what it must carry)

| # | Axis | Meaning | Primary metric | Baseline | Failure |
|---|---|---|---|---|---|
| 1 | Shape | normalized patterns/mixes/regimes | normalized future/past R²; regime NMI | table | only averages |
| 2 | Mass | totals: counts/value/orders/margin | raw future/past R²/MAE | table | right shape, wrong level |
| 3 | Identity | customer/entity/product/campaign/cohort | kNN entity/product purity | table | entities collapse |
| 4 | Rate | gaps, recency, frequency, time-to-event | churn AUC; time-to-next MAE | table | timing averaged |
| 5 | Uncertainty | missingness, unlogged, confidence, OOD | ECE/Brier; OOD AUC; dropout | — | confident and wrong |
| 6 | Causal | `do(a)`, counterfactual, decision | action-conditioned IPS/DR; known effect | table/random | cannot answer “what if” |
| 7 | Order/Path | event order, transitions, path, long memory | transition NLL; shuffle delta | table | bag-of-events (table wins) |
| 8 | Relational/Context | product/campaign/order/session/support relations, baskets | entity linkage; basket/session | table | relations lost |

## 3. Cross-cutting dimensions (inside every axis)
- **Scale/granularity** — short vs long horizons, event vs customer vs cohort; multi-timescale.
- **Drift/non-stationarity** — regimes, seasonality, distribution shift; time-split stability.
- **Polarity** — direction × magnitude; signed/extensive vs bounded/shape.

## 4. Foundation properties (behavior contract)
| Property | Metric |
|---|---|
| Task-free / self-supervised | no labels/table in training; audit |
| Schema-agnostic / per-stream | same code on a schema variant + second stream |
| Universal / transferable | frozen E on held-out/unseen tasks; donor battery |
| Lawful / compositional | merge/scan exactness; permutation; time shift; per-quantity algebra |
| Minimal / bounded | retention vs E dimension; effective rank; learned-only |
| Robust / calibrated | masking/dropout; OOD; ECE; seeds; drift |
| Accessible | linear/kNN probes vs full history |
| Efficient | runtime, E dim, cost/event |
| Artifact / reproducible | frozen hash, config, one command, version |
| Donor contract | adds unique signal; never worse than raw; random/destroyed controls |
| Interpretable / diagnosable | geometry/regimes; axis attribution |

## 5. Evaluation
- **B card (guide):** the eight-axis scorecard + the foundation properties. No downstream heads.
- **B+C card (validation):** frozen E + consumer heads; E vs **raw table + GBM/XGBoost**; paired CIs;
  the fair table disclosed; the hand summary is not allowed inside E.
- **Claims:** only paired-CI wins are claimed; table wins are disclosed.

## 6. Baselines, honesty
- Traditional = wide table + normal feature engineering + GBM/XGBoost.
- Full-history oracle for sufficiency; random/destroyed E for donor controls.
- No hand summaries/counters/windows inside E; raw extensive algebra only.
- Synthetic harness is labelled; no business value claimed.

## 7. Run matrix
- Fast loop: 2k, 1 seed/5; confirmation 3 seeds/20.
- Full: 6k, 1 seed/5 in-loop, 3 seeds/20 in background.
- Axes: the eight probes; properties: sufficiency/accessibility/lawfulness/stability/efficiency.
- GPU-first: derived batch, compiled scan, vectorized data path.

## 8. Current status (honest)
| Axis | Status |
|---|---|
| Shape | partial/good |
| Mass | improved; 6k quick wins value/orders/churn |
| Identity | channel strong (0.727 vs 0.249); category partial |
| Rate | churn 2k below table |
| Uncertainty | ECE/dropout measured; OOD pending |
| Causal | channel table wins; offer point |
| Order/Path | **probe pending** |
| Relational/Context | **probe pending** |

Next: the order-shuffle and relational probes, then the B/C cards.
