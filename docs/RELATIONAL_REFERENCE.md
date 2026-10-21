# Relational reference — PATH A / PATH B

The relational reference replaces the flat generator. It builds the tables a business
actually has, then exposes **two views of the same source data**:

- **PATH A — traditional supervised path.** One row per customer: features computed from
  history `<= as_of`, plus future labels `> as_of` (30d and 90d).
- **PATH B — customer event stream.** One row per event, exactly the canonical schema
  `customer_key, event_ts, event_type, event_attributes`. The full window is stored; the
  encoder must use events `<= as_of`. Context (product, brand, order, value, action keys)
  lives in the JSON `event_attributes`; the richer relational `events.parquet` keeps the
  typed columns.

Both join on **`customer_key`**. The relational tables join on their natural keys
(`order_id`, `product_id`, `ticket_id`, `return_id`).

## Build

```bash
python3 -m rabbit_hole.generators.relational_reference \
  --n-customers 2000 --window-days 540 --seed 17 --as-of 2025-01-01 \
  --out /tmp/opencode/t0_loop/relational_v1
```

Current build: 2,000 customers, 1,099,373 events, 123,793 orders, 14,951 returns.
The build runs the PATH A gate (`scripts/validate_path_a.py`) and writes
`path_a_validation.json`; it fails if the supervised view regresses (constant features,
leakage, degenerate labels, no signal, action imbalance). Use `--no-validate` only for
debugging.

## Files

| Path | Rows × cols | What |
|---|---|---|
| `path_a/customer_features.parquet` | 2000 × 41 | point-in-time features (`<= as_of`) |
| `path_a/customer_labels.parquet` | 2000 × 11 | future labels (`> as_of`, 30/90d) |
| `path_a/path_a_wide_table.parquet` | 2000 × 51 | features + labels (the supervised table) |
| `path_b/path_b_customer_event_stream.parquet` | 1,099,373 × 4 | the canonical event stream |
| `relational/customers.parquet` | 2000 × 6 | observable customer attributes |
| `relational/products.parquet` | 80 × 5 | product catalog |
| `relational/orders.parquet` | 123,793 × 8 | orders (time, channel, discount, total, margin) |
| `relational/order_items.parquet` | 123,793 × 6 | order lines (product, qty, price, discount) |
| `relational/events.parquet` | 1,099,373 × 16 | observed event log (pre-unification) |
| `relational/campaigns.parquet` | 24 × 5 | campaign calendar |
| `relational/returns.parquet` | 14,951 × 5 | returns |
| `relational/support.parquet` | 68,144 × 4 | support tickets |
| `manifest.json` | — | keys, counts, paths, as_of |
| `reference_truth.json` | — | known-truth latent receipt (action optima, segments) |

Compatibility aliases: `customer_event_stream.parquet` (= PATH B) and
`customer_wide_table.parquet` (= PATH A wide).

## What the data contains

- Latent **regimes** (browse / intent / loyal / at_risk) that switch over time.
- **Path dependence**: the next event/purchase depends on the last event (e.g. cart → purchase).
- **Interactions**: channel × regime, discount × price sensitivity.
- **Long memory**: a promotion shock decays over weeks.
- **Missing logs**: a fraction of organic events (view/search/cart) is dropped, so the table
  cannot recover the sequence.
- **Randomized daily actions**: one action key per day (`freq/hour/dow/channel/offer`),
  the others at the customer optimum, with `target_key` on `email_send`. The purchase outcome
  depends on action quality × regime × path × interactions × shock.

## Build gate

`python3 -m rabbit_hole.generators.relational_reference ...` runs
`scripts/validate_path_a.py` after writing. The gate checks: unique keys, label nulls,
A↔B join, no constant features, no near-duplicate features (|corr| > 0.999), point-in-time
recompute from PATH B, label balance, feature ranges, GBM signal vs a shuffled-label null,
temporal stability, and PATH B action-level coverage. Results land in
`path_a_validation.json`.

Current result: **ok, 0 failures** (churn90 AUC 0.966; value90 R² 0.367; null AUC 0.393;
temporal 0.960 / 0.352).

## Rules

- **PATH A features are point-in-time.** Never use `y_*` columns as features.
- **PATH B is the full window.** Cut at `as_of` before encoding; events after `as_of` are for
  labels/decision evaluation only.
- **The table is the baseline.** PATH A is the traditional supervised view; PATH B is the
  encoder view. The comparison is only meaningful on targets where PATH A is lossy.
- **No latent leakage.** `churn_day`/regime are in the truth receipt, not the observable
  customers table.

## Caveats (first version)

- One item per order; sessions are not yet modeled; campaign IDs are not yet attached to events.
- Send cadence varies with the `freq` action (probability 0.20 + 0.20 × freq); the old dense
  one-send-per-day stream was removed because it made `n_sends` and `recency` perfectly
  correlated.
- Campaign IDs are attached to a fraction of sends/purchases; sessions are not yet modeled.
- This is a testbed, not a claim that the encoder wins. The next check is whether a sequence
  oracle beats PATH A + GBM on the 90d value/order labels.
