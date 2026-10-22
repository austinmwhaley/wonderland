# Demo report card — frozen auto encoder + heads

**Goal:** customer event stream → auto encoder (self-supervised) → frozen E → supervised / unsupervised / RL heads → compare to the traditional wide table.

## B — support (one page)

- **Encoder:** auto-tokenizer, all 22 discovered fields, d64/2 layers, E dim 64; no table/counter/summary/labels inside E; `target_key` excluded. Training audit **PASS**.
- **Self-supervised objective:** next-event + future-summary + past-summary over all fields. 6k validation loss **24.8**; 2k fast **~13**.
- **Geometry (2k):** effective rank 2.13 · anomaly AUC **0.760** (table 0.635) · value retrieval **0.455** (0.373) · churn-neighbor **0.980** (0.950).
- **Robustness:** 10% numeric dropout E cosine **0.966**; next-event **2.557 bits**, ECE **0.089**.
- **Artifact:** hash `c5ae38df2c0b`; d64/2; 2k runtime 70s.
- **Donor controls:** real vs random E — retrieval 0.455 vs −0.128; churn-NN 0.980 vs 0.900; anomaly 0.760 vs 0.455.
- **Adaptability:** same pipeline auto-discovered renamed/new fields (`item_id`, `dept`, `maker`, `session_len`).
- **Pending:** retention standardization receipt; decision bridge and certified minimality are C/Layer 1.

## B+C — two heads per family

### Supervised

| Head | Scale | Table | Frozen E / hybrid | Result |
|---|---|---|---|---|
| churn90 | 6k | 0.917 | **E 0.941** | **E wins** (+0.024) |
| orders90 | 2k | 0.352 | **hybrid 0.402** | **E+table wins** (+0.050) |

Disclosed: pure E-only value/orders at 6k are **0.274/0.320 vs table 0.364/0.436** — the table wins; the 6k hybrid is a tie (orders 0.436 vs 0.436).

### Unsupervised (2k)

| Probe | Table | Frozen E | Result |
|---|---|---|---|
| anomaly AUC | 0.635 | **0.760** | **E wins** |
| value retrieval | 0.373 | **0.455** | **E wins** |

### RL (2k, offline IPS/DR; logged randomized actions)

| Arm | Table | Frozen E | Random | Result |
|---|---|---|---|---|
| channel | 518 | **605** | 511 | **E wins** |
| offer | 422 | 470 | 511 | E beats table; random stronger |

## Claim

One frozen self-supervised encoder supports all three head families. It beats the traditional table on **churn90, anomaly, retrieval, and channel RL**, and the **hybrid** beats it on **orders90 at 2k**. The pure E-only value/orders gap is stated, not hidden.

## Caveats

- Synthetic `relational_v3`; no business data; harness.
- 6k/3-seed for the supervised churn; 2k/5-epoch fast for the rest.
- RL is offline OPE, not a shipped policy.
- B retention receipt pending; decision/minimality certification is Layer 1.
- Data card: `docs/LAYER_A_DATA.md`.
