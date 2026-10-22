# B+C demo card (fast 2k; B reference 6k)

Synthetic relational reference `relational_v3`; no business data. Harness, not business value.

## Layer B — encoder

- Frozen E: dim 64, holdout 600, train/val 1260/140, best val 13.083.
- 6k explicit E-only: value 0.388, orders 0.457, churn 0.939.
- 6k none E-only: value 0.387, orders 0.448, churn 0.940.
- 6k generic E-only: value 0.385, orders 0.455, churn 0.934.
- 6k fair table: value 0.335, orders 0.401, churn 0.919; raw table 0.364/0.436/0.917.
- E adds to `table+seq` (6k): hand +0.021/+0.018/+0.017; auto +0.021/+0.032/+0.012.
- Feature ablations (1 seed/5): `qgap` critical; Fourier periods help orders; horizon helps orders; past helps churn; `mass=explicit` trades churn for value/orders.

- Schema-adaptability: same code auto-discovered renamed/new fields (`item_id`, `dept`, `maker`, `session_len`); orders90 +0.077 [+0.010,+0.143], churn90 -0.054. See `docs/report_cards/demo_bc_card_schema_variant.md`.
- Data card: `docs/LAYER_A_DATA.md`.

## Layer C — plugins

### Supervised

| target | table | E | hybrid | E-table | 95% CI |
|---|---|---|---|---|---|
| value30 | 0.1195 | 0.1419 | 0.1730 | +0.0215 | [-0.0730, +0.1112] |
| value90 | 0.3491 | 0.3414 | 0.3860 | -0.0070 | [-0.0777, +0.0548] |
| orders30 | 0.2312 | 0.3007 | 0.3108 | +0.0679 | [-0.0011, +0.1435] |
| orders90 | 0.3516 | 0.4436 | 0.4296 | +0.0932 | [+0.0275, +0.1588] |
| churn30 | 0.7544 | 0.8016 | 0.7875 | +0.0445 | [-0.0047, +0.0967] |
| churn90 | 0.9643 | 0.9605 | 0.9428 | -0.0025 | [-0.0341, +0.0375] |

### Unsupervised (truth-referenced)

| arm | NMI regime | ARI regime | NMI category | anomaly AUC | retrieval rank | eff rank |
|---|---|---|---|---|---|---|
| table | 0.1114 | 0.1017 | 0.0071 | 0.6347 | 0.3733 | 7.98 |
| E | 0.0734 | 0.0622 | 0.0078 | 0.54 | 0.4781 | 2.39 |

### RL (logged randomized actions; uniform-propensity IPS/DR)

| arm | state | arms | IPS | DR | random | truth-acc |
|---|---|---|---|---|---|---|
| channel | table | 4 | 518.37 | 462.27 | 511.20 | 0.250 |
| channel | E | 4 | 574.52 | 546.21 | 511.20 | 0.267 |
| offer | table | 4 | 421.95 | 531.37 | 511.20 | 0.206 |
| offer | E | 4 | 676.22 | 508.51 | 511.20 | 0.261 |

## Caveats

- 2k/5-epoch fast C; 6k/3-seed is the reference for the B card.
- RL uses the last logged `email_send`, future margin reward, uniform-propensity IPS/DR; not a shipped policy.
- Unsupervised uses truth latents; `segment` is random and is not used as a target.
- Low-rank E; hand-listed core + auto-discovered extensions; fixed windows excluded from the fair table.
