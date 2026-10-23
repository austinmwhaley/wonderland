# B card — eight content axes (auto encoder)

Synthetic `relational_v3`; no business data. E: auto-tokenizer, learned shape + additive mass +
identity/rate/uncertainty/causal/order/relational blocks; frozen.

## Axis scorecard

| # | Axis | Metric | Frozen E | Table / baseline | Status |
|---|---|---|---|---|---|
| 1 | Shape | retrieval / churn-nn / regime | retrieval +8.5% pt; churn-nn +3.3% robust | table | partial/good |
| 2 | Mass | 6k value/orders/churn vs XGBoost GPU | **0.401 / 0.452 / 0.931** | 0.358 / 0.450 / 0.926 | **working (value +4.4%)**; orders/churn tie |
| 3 | Identity | kNN channel/category purity | **0.727** / 0.217 | 0.249 / 0.260 | strong (channel) / partial |
| 4 | Rate | churn AUC | 0.931 | 0.926 | tie |
| 5 | Uncertainty | ECE / dropout / OOD | **0.048 / 0.966 / variant churn 0.957** | — | **good (OOD PASS)** |
| 6 | Causal | IPS/DR policy value | channel E 480 vs table 686 (CI table wins); offer E 625 vs 580 (CI crosses) | table/random | not working |
| 7 | Order/Path | event-shuffle delta | value −0.091, orders −0.078, churn −0.030 | table 0 | **working** |
| 8 | Relational/Context | order_id event purity | 0.005 | random 0.0005 | partial (weak) |

## Foundation properties

| Property | Status |
|---|---|
| Task-free / self-supervised | PASS (audit) |
| Schema-agnostic / per-stream | partial (auto fields; variant pending) |
| Universal / transferable | partial (retrieval/identity; no held-out battery yet) |
| Lawful / compositional | field perm 1.2e−7; scan 8.9e−16; mass additivity 0.0 → **PASS** |
| Minimal / bounded | rank 2.4; E dim 128; past retention corr² 0.155; future 0.0 (future objective ineffective) |
| Robust / calibrated | partial (dropout/ECE; OOD pending) |
| Accessible | **shape E 0.365 vs table 0.394 (close); multi-scale blocks (5527d) −0.72 (FAIL); combined −0.64** |
| Efficient | partial (GPU train/eval; data/collate still CPU-bound) |
| Artifact / reproducible | frozen; hash d7d190; single command; v3 cert PASS |
| Donor contract | partial (unique signal vs random E) |
| Interpretable / diagnosable | partial (axis card, geometry) |

## Disclosures

- Stronger XGBoost GPU baseline removes the earlier sklearn-era orders/churn margins; value remains +4.4% at 6k.
- Anomaly is a table win; causal channel is a table win.
- Not 100% complete: uncertainty OOD, causal CI, 3-seed CI, collate vectorization, v3 certificate/one-command.
