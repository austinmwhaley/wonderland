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
| value90 | 6k | 0.364 | **0.423** | **+5.9%** | [+0.029, +0.088] |
| orders90 | 6k | 0.436 | **0.485** | **+4.9%** | [+0.023, +0.076] |
| orders90 | 2k | 0.352 | **0.459** | **+10.7%** | [+0.051, +0.170] |
| churn90 | 6k | 0.917 | 0.943 | +2.5% | [−0.011, +0.061] |

### Unsupervised

| Probe | Table | Pure E | Margin | 95% CI |
|---|---|---|---|---|
| churn-neighbor | 0.950 | **0.980** | **+3.0%** | [+0.002, +0.025] |
| value retrieval | 0.373 | **0.477** | **+10.4%** | [−0.022, +0.065] |
| anomaly AUC | **0.635** | 0.504 | table wins | disclosed |

### RL (offline IPS/DR; test bootstrap)

| Arm | Table | Pure E | Random | E−table CI | Result |
|---|---|---|---|---|---|
| channel | **518** | 480 | 511 | crosses 0 | table wins |
| offer | 422 | 625 | 511 | [−16, +487] | E point win, not robust |

## Claim

The **pure frozen self-supervised E robustly exceeds the traditional table** on value90 (+5.9%), orders90 (+4.9% at 6k, +10.7% at 2k), and churn-neighbor (+3.0%). Retrieval is +10.4% point; churn90 +2.5% marginal; anomaly remains a table win. No hybrid, no hand fields, no fixed 30/90 windows.

## Caveats

- Synthetic `relational_v3`; harness, not business value.
- 6k/3-seed supervised; 2k fast for the rest.
- RL CI not yet implemented; RL is offline OPE, not a shipped policy.
- B-only retention receipt and v3 certificate pending; Layer 1 certification later.
- Data card: `docs/LAYER_A_DATA.md`.
