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
| orders90 | 0.3516 | 0.3402 | 0.4179 | -0.0120 | [-0.0808, +0.0599] |
| churn30 | 0.7544 | 0.7296 | 0.7737 | -0.0283 | [-0.0822, +0.0194] |

### Unsupervised (2)

| probe | table | E | E-table |
|---|---|---|---|
| value-retrieval (Spearman) | 0.3733 | 0.4076 | +0.0343 |
| churn-neighbor agreement | 0.95 | 0.97 | +0.0200 |

### RL (2)

| arm | table IPS | E IPS | random | truth-acc (E) |
|---|---|---|---|---|
| channel | 518.37 | 672.47 | 511.20 | 0.294 |
| offer | 421.95 | 598.27 | 511.20 | 0.306 |

## Appendix — full suite

### Supervised (6)

| target | table | E | hybrid | E-table | 95% CI |
|---|---|---|---|---|---|
| value30 | 0.1195 | 0.0257 | 0.1406 | -0.0947 | [-0.1986, -0.0059] |
| value90 | 0.3491 | 0.2696 | 0.3723 | -0.0812 | [-0.1524, -0.0027] |
| orders30 | 0.2312 | 0.2109 | 0.2773 | -0.0184 | [-0.1007, +0.0660] |
| orders90 | 0.3516 | 0.3402 | 0.4179 | -0.0120 | [-0.0808, +0.0599] |
| churn30 | 0.7544 | 0.7296 | 0.7737 | -0.0283 | [-0.0822, +0.0194] |
| churn90 | 0.9643 | 0.9646 | 0.9706 | +0.0017 | [-0.0355, +0.0413] |

### Unsupervised (4, truth-referenced)

| arm | NMI regime | ARI regime | NMI category | anomaly AUC | retrieval rank | eff rank |
|---|---|---|---|---|---|---|
| table | 0.1114 | 0.1017 | 0.0071 | 0.6347 | 0.3733 | 7.98 |
| E | 0.08 | 0.0666 | 0.0139 | 0.5187 | 0.4076 | 3.06 |

### RL (4, logged randomized actions; uniform-propensity IPS/DR)

| arm | state | arms | IPS | DR | random | truth-acc |
|---|---|---|---|---|---|---|
| channel | table | 4 | 518.37 | 462.27 | 511.20 | 0.250 |
| channel | E | 4 | 672.47 | 589.21 | 511.20 | 0.294 |
| offer | table | 4 | 421.95 | 531.37 | 511.20 | 0.206 |
| offer | E | 4 | 598.27 | 565.86 | 511.20 | 0.306 |

## Caveats

- 2k/5-epoch fast C; 6k/3-seed is the reference for the B card.
- RL uses the last logged `email_send`, future margin reward, uniform-propensity IPS/DR; not a shipped policy.
- Unsupervised uses truth latents; `segment` is random and is not used as a target.
- Low-rank E; hand-listed core + auto-discovered extensions; fixed windows excluded from the fair table.
