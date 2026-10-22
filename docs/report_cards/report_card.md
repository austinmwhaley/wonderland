# Report card — auto encoder (B) + validation (B+C)

Data: `relational_v3`, 2k/6k. Synthetic harness; no business data.

## B — encoder (guide)

- **Option:** auto-tokenizer; all discovered fields; d64, 2 layers; generic next-event + future-summary + past-summary; auto mass; E dim 64.
- **Held-out objective:** 6k, 3 seeds, validation loss **24.8**; 2k fast **~13**.
- **Head-free geometry (2k):** effective rank 2.32 · anomaly AUC **0.759** (table 0.635) · value retrieval **0.455** (0.373) · churn-neighbor **0.980** (0.950) · regime NMI 0.107 (0.111).
- **Accessibility:** anomaly, retrieval and churn-neighbor use no trained head.
- **Efficiency:** 64 dims; 2k fast; 6k 3 seeds/20 epochs.
- **Adaptability:** same pipeline auto-discovered renamed/new fields (`item_id`, `dept`, `maker`, `session_len`).

## B+C — validation

### 6k E-only vs table

| Target | Auto E | Fair table | Raw table |
|---|---|---|---|
| value90 | 0.274 | 0.335 | 0.364 |
| orders90 | 0.320 | 0.401 | 0.436 |
| churn90 | **0.924** | 0.919 | 0.917 |

### Supervised (2k, 2 examples)

| target | table | E | hybrid | E-table | 95% CI |
|---|---|---|---|---|---|
| orders90 | 0.3516 | 0.3223 | 0.4021 | -0.0311 | [-0.1093, +0.0396] |
| churn30 | 0.7544 | 0.6952 | 0.7304 | -0.0618 | [-0.1225, -0.0121] |

### Unsupervised (2k, 2 examples)

| probe | table | E | E-table |
|---|---|---|---|
| value retrieval | 0.3733 | 0.4552 | **+0.0819** |
| churn-neighbor | 0.9500 | 0.9800 | **+0.0300** |

### RL (2k, 2 examples)

| arm | table IPS | E IPS | random | truth-acc (E) |
|---|---|---|---|---|
| channel | 518.37 | **605.01** | 511.20 | 0.294 |
| offer | 421.95 | 469.88 | 511.20 | 0.267 |

## Caveats

- 2k/5-epoch C; 6k/3-seed B. Synthetic harness, not business value.
- The auto E is **not** best on supervised value/orders; it wins churn, anomaly, retrieval, churn-neighbor, and the channel RL.
- RL is offline IPS/DR from logged randomized actions; not a shipped policy.
- Randomized static `segment` is not a target; fixed 30/90 windows are excluded from the fair table.
