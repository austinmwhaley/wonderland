# Demo report card — pure frozen auto encoder

**Goal:** customer event stream → auto encoder (self-supervised) → frozen E → supervised / unsupervised / RL heads → compare to the traditional wide table. **No hybrid. Pure E only.**

## B — support (one page)

- Auto-tokenizer, all 22 discovered fields; d64/2 layers; E dim 64; no table/labels in E; `target_key` excluded. Audit **PASS**.
- Self-supervised: next-event + future-summary + past-summary over all fields.
- Geometry (2k): eff rank 2.77 · anomaly 0.504 · retrieval **0.390** (table 0.373) · churn-neighbor **0.973** (0.950).
- Robustness: 10% dropout E cosine 0.966; next-event 2.557 bits; ECE 0.089.
- Artifact: hash `c5ae38df2c0b`; d64/2; 2k runtime 70s.
- Donor: real vs random E — retrieval 0.390 vs −0.128; churn-NN 0.973 vs 0.900; anomaly 0.504 vs 0.455.

## B+C — pure frozen E (2 heads per family)

### Supervised

| Head | Scale | Table | Pure E | Result |
|---|---|---|---|---|
| orders90 | 2k | 0.352 | **0.358** | **E exceeds** |
| churn90 | 6k | 0.917 | **0.941** | **E exceeds** |
| value90 | 6k | 0.364 | 0.300 | table's fixed `value_30d/90d` — disclosed, not used by E |

### Unsupervised

| Probe | Table | Pure E | Result |
|---|---|---|---|
| value retrieval | 0.373 | **0.390** | **E exceeds** |
| churn-neighbor | 0.950 | **0.973** | **E exceeds** |
| anomaly AUC | **0.635** | 0.504 | table wins — disclosed |

### RL (offline IPS/DR, logged randomized actions)

| Arm | Table | Pure E | Random | Result |
|---|---|---|---|---|
| channel | 518 | **711** | 511 | **E exceeds** |
| offer | 422 | **488** | 511 | E exceeds table; random stronger |

## Claim

The **pure frozen self-supervised E** exceeds the traditional table on **orders90, churn90 (6k), retrieval, churn-neighbor, channel RL, and offer RL vs table** — no hybrid. The two disclosed table wins are **value90** (fixed 30/90d windows, which the E is not allowed to copy) and **anomaly**.

## Caveats

- Synthetic `relational_v3`; no business data; harness.
- 2k fast C; 6k churn from receipts.
- RL is offline OPE, not a shipped policy.
- B retention receipt pending; decision/minimality certification is Layer 1.
- Data card: `docs/LAYER_A_DATA.md`.
