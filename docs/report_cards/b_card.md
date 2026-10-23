# B card — content-axis scorecard (auto encoder)

Synthetic `relational_v3`; no business data. E: auto-tokenizer, learned states, frozen.

## Axis scorecard

| Axis | Primary metric | Baseline | Current | Status |
|---|---|---|---|---|
| Shape | retrieval / churn-neighbor / regime NMI | table | retrieval tie; churn-nn +1.4% robust; regime NMI below table | partial |
| Mass | raw orders90/value90 R² | table | 6k quick: value 0.383 vs 0.348; orders 0.456 vs 0.425 | working (point) |
| Identity | kNN channel/category purity | table | channel **0.727 vs 0.249**; category 0.217 vs 0.260 | strong (channel) / partial |
| Rate | churn AUC | table | E 0.934 vs table 0.964 | not working (rate) |
| Uncertainty | ECE / dropout / OOD | — | uncertainty block wired; ECE 0.089; dropout 0.966 | partial |
| Causal | action-conditioned IPS/DR | table/random | causal block wired; channel table wins; offer point | not working |

## Foundation properties

| Property | Metric | Status |
|---|---|---|
| Sufficiency | E vs full-history oracle gap | pending |
| Accessibility | linear/kNN probes | pending |
| Minimality / boundedness | effective rank / retention | partial (rank ~2.8) |
| Lawful composition | permutation / merge / additivity | pending |
| Stability / adaptivity | seeds, time-shift, schema variant | partial |
| Efficiency | E dim, train/inference | measured |
| Artifact | frozen hash, config, one command | partial |
| Donor controls | raw/random/destroyed | partial |

## Disclosures

- Mass is the main bottleneck; the hand summary previously supplied it and is removed.
- Anomaly is a table win.
- B+C utility is a separate card; this card does not judge by heads.
