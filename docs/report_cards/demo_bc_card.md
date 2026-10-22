# B+C demo card (fast 2k; B reference 6k)

Synthetic relational reference `relational_v3`; no business data. Harness, not business value.

## Layer B — encoder

- Frozen E: dim 64, holdout 600.
- 6k Auto-tokenizer + mass explicit (default): E-only value 0.300, orders 0.353, churn 0.941.
- 6k Hand tokenizer + mass explicit (ablation): E-only value 0.388, orders 0.457, churn 0.939.
- 6k fair table: value 0.335, orders 0.401, churn 0.919; raw table 0.364/0.436/0.917.
- E adds to `table+seq` (6k): hand +0.021/+0.018/+0.017; auto +0.021/+0.032/+0.012.
- Feature ablations (1 seed/5): `qgap` critical; Fourier periods help orders; horizon helps orders; past helps churn; `mass=explicit` trades churn for value/orders.

- Schema-adaptability: same code auto-discovered renamed/new fields (`item_id`, `dept`, `maker`, `session_len`); orders90 +0.077 [+0.010,+0.143], churn90 -0.054. See `docs/report_cards/demo_bc_card_schema_variant.md`.
- Data card: `docs/LAYER_A_DATA.md`.

## Layer C — curated demo (2 per class)

### Supervised (2)

| target | table | E | hybrid | E-table | 95% CI |
|---|---|---|---|---|---|
| orders90 | 0.3516 | 0.3225 | 0.4021 | -0.0309 | [-0.1089, +0.0403] |
| churn30 | 0.7544 | 0.6952 | 0.7304 | -0.0618 | [-0.1225, -0.0121] |

### Unsupervised (2)

| probe | table | E | E-table |
|---|---|---|---|
| value-retrieval (Spearman) | 0.3733 | 0.4552 | +0.0819 |
| churn-neighbor agreement | 0.95 | 0.98 | +0.0300 |

### RL (2)

| arm | table IPS | E IPS | random | truth-acc (E) |
|---|---|---|---|---|
| channel | 518.37 | 605.01 | 511.20 | 0.294 |
| offer | 421.95 | 469.88 | 511.20 | 0.267 |

## Appendix — full suite

### Supervised (6)

| target | table | E | hybrid | E-table | 95% CI |
|---|---|---|---|---|---|
| value30 | 0.1195 | 0.0618 | 0.1470 | -0.0600 | [-0.1575, +0.0307] |
| value90 | 0.3491 | 0.1901 | 0.3441 | -0.1608 | [-0.2423, -0.0792] |
| orders30 | 0.2312 | 0.1241 | 0.2722 | -0.1091 | [-0.2115, -0.0261] |
| orders90 | 0.3516 | 0.3225 | 0.4021 | -0.0309 | [-0.1089, +0.0403] |
| churn30 | 0.7544 | 0.6952 | 0.7304 | -0.0618 | [-0.1225, -0.0121] |
| churn90 | 0.9643 | 0.9449 | 0.9560 | -0.0195 | [-0.0646, +0.0105] |

### Unsupervised (4, truth-referenced)

| arm | NMI regime | ARI regime | NMI category | anomaly AUC | retrieval rank | eff rank |
|---|---|---|---|---|---|---|
| table | 0.1114 | 0.1017 | 0.0071 | 0.6347 | 0.3733 | 7.98 |
| E | 0.103 | 0.1071 | 0.0075 | 0.7601 | 0.4552 | 2.13 |

### RL (4, logged randomized actions; uniform-propensity IPS/DR)

| arm | state | arms | IPS | DR | random | truth-acc |
|---|---|---|---|---|---|---|
| channel | table | 4 | 518.37 | 462.27 | 511.20 | 0.250 |
| channel | E | 4 | 605.01 | 545.14 | 511.20 | 0.294 |
| offer | table | 4 | 421.95 | 531.37 | 511.20 | 0.206 |
| offer | E | 4 | 469.88 | 475.59 | 511.20 | 0.267 |

## Caveats

- 2k/5-epoch fast C; 6k/3-seed is the reference for the B card.
- RL uses the last logged `email_send`, future margin reward, uniform-propensity IPS/DR; not a shipped policy.
- Unsupervised uses truth latents; `segment` is random and is not used as a target.
- Low-rank E; hand-listed core + auto-discovered extensions; fixed windows excluded from the fair table.
