# B card — content-axis scorecard (auto encoder)

Synthetic `relational_v3`; no business data. E: auto-tokenizer, learned states, frozen.

## Axis scorecard

| Axis | Primary metric | Baseline | Current | Status |
|---|---|---|---|---|
| Shape | retrieval / churn-neighbor / regime NMI | table | retrieval tie; churn-nn +1.4% robust; regime NMI below table | partial |
| Mass | raw orders90/value90 R² | table | orders90 0.320 vs 0.352 after additive mass | partial (improved) |
| Identity | kNN entity/retrieval | table | churn-nn robust; product/campaign pending | partial |
| Rate | churn AUC / time-to-next | table | churn 2k below table; 6k mixed | partial |
| Uncertainty | ECE / dropout / OOD | — | ECE 0.089; dropout cos 0.966; OOD pending | partial |
| Causal | action-conditioned IPS/DR | table/random | channel table wins; offer point only | not working |

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
