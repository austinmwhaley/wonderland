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
| orders90 | 0.3516 | 0.2898 | 0.3878 | -0.0635 | [-0.1282, +0.0090] |
| churn30 | 0.7544 | 0.7085 | 0.7311 | -0.0494 | [-0.1130, +0.0010] |

### Unsupervised (2)

| probe | table | E | E-table |
|---|---|---|---|
| value-retrieval (Spearman) | 0.3733 | 0.4542 | +0.0809 |
| churn-neighbor agreement | 0.95 | 0.9817 | +0.0317 |

### RL (2)

| arm | table IPS | E IPS | random | truth-acc (E) |
|---|---|---|---|---|
| channel | 518.37 | 516.83 | 511.20 | 0.256 |
| offer | 421.95 | 517.20 | 511.20 | 0.239 |

## Appendix — full suite

### Supervised (6)

| target | table | E | hybrid | E-table | 95% CI |
|---|---|---|---|---|---|
| value30 | 0.1195 | 0.0465 | 0.1587 | -0.0756 | [-0.1668, +0.0205] |
| value90 | 0.3491 | 0.1982 | 0.3647 | -0.1530 | [-0.2286, -0.0745] |
| orders30 | 0.2312 | 0.1508 | 0.2330 | -0.0822 | [-0.1626, +0.0069] |
| orders90 | 0.3516 | 0.2898 | 0.3878 | -0.0635 | [-0.1282, +0.0090] |
| churn30 | 0.7544 | 0.7085 | 0.7311 | -0.0494 | [-0.1130, +0.0010] |
| churn90 | 0.9643 | 0.9732 | 0.9759 | +0.0096 | [-0.0099, +0.0368] |

### Unsupervised (4, truth-referenced)

| arm | NMI regime | ARI regime | NMI category | anomaly AUC | retrieval rank | eff rank |
|---|---|---|---|---|---|---|
| table | 0.1114 | 0.1017 | 0.0071 | 0.6347 | 0.3733 | 7.98 |
| E | 0.1069 | 0.1048 | 0.0088 | 0.7587 | 0.4542 | 2.32 |

### RL (4, logged randomized actions; uniform-propensity IPS/DR)

| arm | state | arms | IPS | DR | random | truth-acc |
|---|---|---|---|---|---|---|
| channel | table | 4 | 518.37 | 462.27 | 511.20 | 0.250 |
| channel | E | 4 | 516.83 | 507.78 | 511.20 | 0.256 |
| offer | table | 4 | 421.95 | 531.37 | 511.20 | 0.206 |
| offer | E | 4 | 517.20 | 578.97 | 511.20 | 0.239 |

## Caveats

- 2k/5-epoch fast C; 6k/3-seed is the reference for the B card.
- RL uses the last logged `email_send`, future margin reward, uniform-propensity IPS/DR; not a shipped policy.
- Unsupervised uses truth latents; `segment` is random and is not used as a target.
- Low-rank E; hand-listed core + auto-discovered extensions; fixed windows excluded from the fair table.
