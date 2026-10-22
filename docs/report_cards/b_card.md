# B card — auto encoder (B only)

Data: `relational_v3`, 2k/6k. Synthetic harness; no business data.
Option: auto-tokenizer, all discovered fields; d64/2 layers; generic next/future/past objectives; auto mass; E dim 64.

## Held-out objective
- 6k, 3 seeds: validation loss 24.8 (lower better); 2k fast: ~13.
- Objectives: next-event (all fields) + future-summary (all fields) + past-summary (all fields).

## Head-free geometry (2k holdout)
| Probe | Auto E | Table |
|---|---|---|
| effective rank | 2.32 | 7.98 |
| anomaly AUC | **0.759** | 0.635 |
| value retrieval (Spearman) | **0.455** | 0.373 |
| churn-neighbor agreement | **0.980** | 0.950 |
| regime NMI | 0.107 | 0.111 |

## Accessibility
Anomaly, retrieval, and churn-neighbor use no trained head; the auto E carries the signal.

## Efficiency
E = 64 dims; 2k fast run seconds-to-minutes; 6k 3 seeds/20 epochs.

## Adaptability
Same auto pipeline discovered renamed/new fields (`item_id`, `dept`, `maker`, `session_len`) in the schema variant.

## B+C
The B+C card is the validation check (`docs/report_cards/demo_bc_card.md`).
