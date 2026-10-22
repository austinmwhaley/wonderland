# B+C demo card (fast 2k; B reference 6k)

Synthetic relational reference `relational_v3`; no business data. Harness, not business value.

## Layer B — encoder

- Frozen E: dim 64, holdout 600.
- 6k Auto encoder: E-only value 0.274, orders 0.320, churn 0.924.
- 6k fair table: value 0.335, orders 0.401, churn 0.919; raw table 0.364/0.436/0.917.
- E adds to `table+seq` (6k): hand +0.021/+0.018/+0.017; auto +0.021/+0.032/+0.012.
- Schema-adaptability: same code auto-discovered renamed/new fields (`item_id`, `dept`, `maker`, `session_len`); orders90 +0.077 [+0.010,+0.143], churn90 -0.054. See `docs/report_cards/demo_bc_card_schema_variant.md`.
- Data card: `docs/LAYER_A_DATA.md`.

## Layer C — curated demo (2 per class)

### Supervised (2)

| target | table | E | hybrid | E-table | 95% CI |
|---|---|---|---|---|---|
| orders90 | 0.3516 | 0.3583 | 0.4048 | +0.0080 | [-0.0608, +0.0818] |
| churn30 | 0.7544 | 0.6993 | 0.7224 | -0.0579 | [-0.1185, -0.0030] |

### Unsupervised (2)

| probe | table | E | E-table |
|---|---|---|---|
| value-retrieval (Spearman) | 0.3733 | 0.3898 | +0.0165 |
| churn-neighbor agreement | 0.95 | 0.9733 | +0.0233 |

### RL (2)

| arm | table IPS | E IPS | random | truth-acc (E) |
|---|---|---|---|---|
| channel | 518.37 | 710.67 | 511.20 | 0.328 |
| offer | 421.95 | 488.48 | 511.20 | 0.272 |

## Appendix — full suite

### Supervised (6)

| target | table | E | hybrid | E-table | 95% CI |
|---|---|---|---|---|---|
| value30 | 0.1195 | 0.1347 | 0.2035 | +0.0116 | [-0.0914, +0.1095] |
| value90 | 0.3491 | 0.2910 | 0.3712 | -0.0595 | [-0.1310, +0.0076] |
| orders30 | 0.2312 | 0.2254 | 0.2886 | -0.0055 | [-0.1005, +0.0810] |
| orders90 | 0.3516 | 0.3583 | 0.4048 | +0.0080 | [-0.0608, +0.0818] |
| churn30 | 0.7544 | 0.6993 | 0.7224 | -0.0579 | [-0.1185, -0.0030] |
| churn90 | 0.9643 | 0.9572 | 0.9589 | -0.0058 | [-0.0426, +0.0305] |

### Unsupervised (4, truth-referenced)

| arm | NMI regime | ARI regime | NMI category | anomaly AUC | retrieval rank | eff rank |
|---|---|---|---|---|---|---|
| table | 0.1114 | 0.1017 | 0.0071 | 0.6347 | 0.3733 | 7.98 |
| E | 0.0705 | 0.0732 | 0.0122 | 0.5035 | 0.3898 | 2.77 |

### RL (4, logged randomized actions; uniform-propensity IPS/DR)

| arm | state | arms | IPS | DR | random | truth-acc |
|---|---|---|---|---|---|---|
| channel | table | 4 | 518.37 | 462.27 | 511.20 | 0.250 |
| channel | E | 4 | 710.67 | 537.77 | 511.20 | 0.328 |
| offer | table | 4 | 421.95 | 531.37 | 511.20 | 0.206 |
| offer | E | 4 | 488.48 | 483.63 | 511.20 | 0.272 |

## Caveats

- 2k/5-epoch fast C; 6k/3-seed is the reference for the B card.
- RL uses the last logged `email_send`, future margin reward, uniform-propensity IPS/DR; not a shipped policy.
- Unsupervised uses truth latents; `segment` is random and is not used as a target.
- Low-rank E; hand-listed core + auto-discovered extensions; fixed windows excluded from the fair table.
