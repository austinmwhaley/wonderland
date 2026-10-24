# Layer A — Data (reference)

Status: documentation for the **synthetic relational reference**. No business data is used.
This is a **harness**, not evidence of business value.

## What it is

`rabbit_hole/generators/relational_reference.py` builds the tables a business actually has
(customers, products, orders, order_items, events, campaigns, returns, support), then exposes
two views of the same source:

- **PATH A** — one row per customer, features `<= as_of`, future labels `> as_of` (30/90d).
- **PATH B** — the canonical 4-column event stream
  `customer_key, event_ts, event_type, event_attributes`.

Both join on `customer_key`. The encoder may use only events `<= as_of`.

## Build and scale

```bash
python3 -m rabbit_hole.generators.relational_reference \
  --n-customers 2000 --window-days 540 --seed 17 --as-of 2025-01-01 \
  --out /tmp/opencode/t0_loop/relational_v3
```

| Build | Customers | Events | Orders | Returns | PATH A | PATH B |
|---|---|---|---|---|---|---|
| 2k | 2,000 | 1,119,402 | 128,190 | 15,499 | 2000 × 51 | 1,119,402 × 4 |
| 6k | 6,000 | 3,355,797 | 385,166 | 46,045 | 6000 × 51 | 3,355,797 × 4 |

Event types: signup, view, search, add_to_cart, remove_from_cart, purchase, return,
email_send, support_ticket, review, loyalty_change. The stream carries product/category/brand/
channel/campaign/value/margin/price/qty/discount in JSON attributes plus the action metadata.

## Labels and truth

PATH A has 10 future labels (orders, value, margin, next_days, churn at 30/90d). The truth
receipt (`reference_truth.json`) exposes latent regimes (`regime0`, `regime_runs`), category
preference, price sensitivity, return proneness, channel preference, offer optimum, and churn
day. These are used for unsupervised diagnostics; the `segment` column is **random** and is not
used as a truth target.

Actions are daily, one-factor randomized (`channel`, `offer`, `freq`, `hour`, `dow`); the
decision reward is future gross margin. The randomized logging is the basis for the RL card's
IPS/DR evaluation; no policy is shipped.

## Derivability (P8)

The original v1 stream did not contain `segment`, initial `loyalty_tier`, or the true
`signup_ts`. v3 writes them into the `customer_signup` event at the window start. PATH A is
**byte-identical** to v1 (`sha256` match); PATH B now contains every table column, so the table
is a derived view of the stream. The pre-window gap is not stored as an event time, so it cannot
be mistaken for inactivity.

## Fair table

The raw 39-feature table includes fixed `value_30d/90d` windows and random static columns. The
**fair table** drops the fixed windows (and, in the diagnostic ablation, the random static
columns). At 2k: raw 0.349/0.352/0.964; fair 0.289/0.349/0.960. At 6k: raw 0.364/0.436/0.917;
fair 0.335/0.401/0.919. The encoder is compared against the fair table; the raw-table value edge
is fixed-window engineering.

## Certificate and gates

- `scripts/validate_path_a.py` — leakage/balance/signal checks (run by the generator).
- `scripts/validate_layer_a.py --dir <build> --seed 17` — Layer A validation; v3 passes.
- `docs/LAYER_A_CERTIFICATE.md`, `docs/report_cards/path_a_validation.json`,
  `docs/report_cards/layer_a_certificate.json` — reference receipts (v1; v3 validator pass
  recorded in the current docs).

## Limitations

- Synthetic; one item per order; no sessions/hierarchy; random static customer columns.
- The action design is a simplified one-factor daily randomization; the RL card is offline
  IPS/DR only.
- Missingness, long memory, regimes and action effects are simulated, not observed.
- Not business data; do not present as business value.

## Demo use

The demo is this reference stream run through the normal pipeline:
`python3 scripts/run_demo.py --generate --build --refresh` (or point `--data` at an
existing build). The primary card is `docs/REPORT_CARD.md`; the full audit is
`docs/AUDIT_CARD.md`.
