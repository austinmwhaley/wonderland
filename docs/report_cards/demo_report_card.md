# Demo report card — pure frozen auto encoder

**Goal:** event stream → auto encoder (self-supervised) → frozen E → supervised / unsupervised / RL heads → compare to the traditional wide table. **Pure E, no hybrid.**

## B — frozen encoder

- Auto-tokenizer, all discovered fields; pooled block + multi-scale block (SSM states, raw numeric decayed sums, event-type decayed counts at data-derived horizons).
- No table/labels in E; `target_key` excluded; E dim 64 pooled + multi-scale block.
- Geometry (2k): retrieval +4.6% (CI marginal); churn-neighbor +2.5% (CI robust); anomaly table wins.
- Artifact: frozen, hashed, 2k fast; synthetic harness.

## B+C — pure E, two heads per family

### Supervised

| Head | Scale | Table | Pure E | Margin | 95% CI |
|---|---|---|---|---|---|
| value90 | 6k | 0.364 | **0.424** | **+6.0%** | [+0.043, +0.076] |
| orders90 | 6k | 0.436 | **0.489** | **+5.3%** | [+0.024, +0.082] |
| orders90 | 2k | 0.352 | **0.445** | **+9.3%** | [+0.038, +0.151] |
| churn90 | 6k | 0.917 | 0.938 | +2.1% | [−0.001, +0.043] |

### Unsupervised

| Probe | Table | Pure E | Margin | 95% CI |
|---|---|---|---|---|
| churn-neighbor | 0.950 | **0.975** | **+2.5%** | [+0.003, +0.022] |
| value retrieval | 0.373 | 0.419 | +4.6% | [−0.041, +0.058] |
| anomaly AUC | **0.635** | 0.504 | table wins | disclosed |

### RL (offline IPS/DR, logged randomized actions; CI pending)

| Arm | Table | Pure E | Random |
|---|---|---|---|
| channel | 518 | 711 | 511 |
| offer | 422 | 488 | 511 |

## Claim

The **pure frozen self-supervised E robustly exceeds the traditional table** on value90 (+6.0%), orders90 (+5.3% at 6k, +9.3% at 2k), and churn-neighbor (+2.5%). Churn90/retrieval are marginal; anomaly remains a table win. No hybrid, no hand fields, no fixed 30/90 windows.

## Caveats

- Synthetic `relational_v3`; harness, not business value.
- 6k/3-seed supervised; 2k fast for the rest.
- RL CI not yet implemented; RL is offline OPE, not a shipped policy.
- B-only retention receipt and v3 certificate pending; Layer 1 certification later.
- Data card: `docs/LAYER_A_DATA.md`.
