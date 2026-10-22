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

## Layer C — plugins

### Supervised

| target | table | E | hybrid | E-table | 95% CI |
|---|---|---|---|---|---|
| value30 | 0.1195 | 0.1429 | 0.1723 | +0.0226 | [-0.0744, +0.1129] |
| value90 | 0.3491 | 0.3345 | 0.3808 | -0.0141 | [-0.0833, +0.0508] |
| orders30 | 0.2312 | 0.3025 | 0.3048 | +0.0696 | [-0.0001, +0.1416] |
| orders90 | 0.3516 | 0.4347 | 0.4289 | +0.0842 | [+0.0196, +0.1496] |
| churn30 | 0.7544 | 0.7978 | 0.7868 | +0.0407 | [-0.0104, +0.0922] |
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
