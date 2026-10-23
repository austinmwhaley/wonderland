# B+C report card — frozen auto encoder vs traditional XGBoost

Data: `relational_v3`; synthetic harness. E: self-supervised, all blocks, frozen. No hybrid.

## Supervised — 6k, XGBoost GPU

| Target | Traditional table | Frozen E | Margin |
|---|---|---|---|
| value90 | 0.358 | **0.401** | **+4.4%** |
| orders90 | 0.450 | 0.452 | tie |
| churn90 | 0.926 | 0.931 | tie |

## Supervised — 2k

| Target | Table | Frozen E | Margin |
|---|---|---|---|
| value90 | 0.378 | 0.379 | tie |
| orders90 | 0.392 | 0.409 | +1.7% |
| churn90 | 0.960 | 0.951 | −0.9% |

## Unsupervised

| Probe | Table | Frozen E | Result |
|---|---|---|---|
| value retrieval | 0.373 | **0.419** | +4.6% pt |
| churn-neighbor | 0.950 | **0.983** | **+3.3% robust** |
| anomaly AUC | **0.635** | 0.504 | table |

## RL (offline IPS/DR; logged randomized actions)

| Arm | Table | Frozen E | Result |
|---|---|---|---|
| channel | **518** | 480 | table |
| offer | 422 | 625 | E point, CI crosses 0 |

## Identity / Order

| Probe | Table | Frozen E |
|---|---|---|
| channel identity | 0.249 | **0.727** |
| category identity | 0.260 | 0.217 |
| event-shuffle drop (value/orders/churn) | 0 | **−0.091 / −0.078 / −0.030** |
| order_id event purity | n.a. | 0.005 (chance 0.0005) |

## Claim

The frozen E **beats XGBoost on 6k value (+4.4%)**, **matches** it on orders/churn at 6k, and is strong on **retrieval, churn-neighbor, channel identity, and order/path sensitivity**. It does **not** beat the strong XGBoost baseline on churn or anomaly, and the RL is not a robust win. No hybrid, no hand summary inside E.

## Caveats

- Synthetic harness; no business data.
- 6k quick is 1 seed/5; 2k is 1 seed/5 for probes.
- Uncertainty OOD, causal CI, 3-seed CIs, collate vectorization, and v3 certificate remain.
- Data card: `docs/LAYER_A_DATA.md`.
