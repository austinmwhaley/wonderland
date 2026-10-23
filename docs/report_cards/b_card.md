# B card — content-axis scorecard (auto encoder)

Synthetic `relational_v3`; no business data. E: auto-tokenizer, learned states, frozen.

## Axis scorecard

| Axis | Primary metric | Baseline | Current | Status |
|---|---|---|---|---|
| Shape | retrieval / churn-neighbor / regime NMI | table | retrieval tie; churn-nn +1.4% robust; regime NMI below table | partial |
| Mass | raw orders90/value90 R² | table | orders90 0.393 vs 0.352 (+4.2% point) with separate mass block | partial (point win) |
| Identity | kNN entity/retrieval | table | identity block wired; churn-nn robust; product/campaign pending | partial |
| Rate | churn AUC / time-to-next | table | rate block wired; churn 2k -2.5% | partial |
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
