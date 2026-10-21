# Layer A certificate — PATH A (supervised table) + PATH B (event stream)

**Verdict: CERTIFIED for encoder development, at the scope below.** This certifies the
**data layer**, not the encoder, not business value, and not cross-stream universality.

- Build: `/tmp/opencode/t0_loop/relational_v1`
- As-of: `2025-01-01`; window: 2024-01-01 + 540 days; 2,000 customers
- Receipt: `docs/report_cards/layer_a_certificate.json`
- PATH A gate: `docs/report_cards/path_a_validation.json`
- Reproducibility: same seed → **byte-identical** PATH A, PATH B, truth, manifest
  (`layer_a_certificate.json → reproducible: true`)

## What is certified

### PATH A — traditional supervised table
- **2,000 × 51** = 39 model features (+ key/signup) + 10 future labels; one row per customer.
- **No leakage:** independent recompute from PATH B (events `<= as_of`) matches
  `n_orders`, `purchase_value`; future labels match `(as_of, as_of+30d]`.
- **No constant features; no near-duplicate features** (|corr| > 0.999).
- **Label balance:** churn90 7.95% (159 positives); 29–36 positives per segment (7.5–8.7%).
- **Signal:** churn90 AUC **0.936**, AP 0.891, Brier 0.012; value90 R² **0.374**,
  MAE 541.
- **No leakage null:** shuffled labels → churn AUC **0.476**, value R² **−0.146**.
- **Temporal stability:** train older signup cohorts → churn AUC **0.934**, value R² **0.406**.
- **Join:** all 2,000 A keys in B and all B keys in A.

### PATH B — customer event stream
- Exactly **`customer_key, event_ts, event_type, event_attributes`**; 1,119,402 rows,
  2,000 customers; **sorted** by customer/time.
- Sequence length: **min 117, median 556, max 975** events/customer.
- Prefix (`<= as_of`) 789,323 events; future window present for labels.
- **3 duplicate timestamps** out of 1.12M (0.0003%) — negligible.
- Event types present: signup, send, view, search, cart, remove, purchase, return,
  support, review.
- `target_key` is **excluded from the measure** (action-design metadata, not history).
- Latent **regime runs** and traits are in the truth receipt for diagnostics; not in the stream.

### Action design (usable for the decision layer)
All 5 keys present with ≥2 levels and ≥50 sends/level; observed conversion spreads:

| key | levels | conversion range | spread |
|---|---|---|---|
| freq | 0–3 | 0.106–0.130 | 0.024 |
| hour | 0–23 | 0.074–0.128 | 0.054 |
| dow | 0–6 | 0.100–0.119 | 0.019 |
| channel | 0–3 | 0.095–0.120 | 0.025 |
| offer | 0–3 | 0.125–0.145 | 0.019 |

### Encoder headroom (the critical test)
Adding **sequence features** from `<= as_of` (last/previous event types, last
category/touch, gaps, 7/30d counts) on top of PATH A:
- value90 R² **0.374 → 0.450** (**+0.076**) → the table is demonstrably lossy on value.
- churn90 AUC 0.936 → 0.930 (−0.006) → **churn is already table-solved**; do not
  expect or claim an encoder win on churn.

This establishes that Layer A supports the development goal: there is
**sequence-dependent value information the table does not capture**.

### Pipeline ingestion
The existing pipeline reads PATH B and builds the measure:
- 789,323 rows `<= as_of`; 2,000 customers; **103 measure columns**;
  **43 response columns**; all five actions discovered; `target_key` absent.

## Conditions and limits (do not overclaim)

- **Synthetic, per-stream, 2,000 customers.** No real-world or cross-stream claim.
- **One item per order; no sessions; campaign attachment is basic.** Fine for a
  reference, not a production replica.
- **Churn is table-solved.** Encoder development should target **value/order/next-product
  /decision** outcomes, where the headroom is measured.
- **Headroom is a feature proxy, not an encoder result.** The next gate is a trained
  sequence model beating PATH A on value/orders with a held-out CI.
- Use PATH A features `<= as_of` and labels `> as_of`; cut PATH B at `as_of` before
  encoding. Never put labels or latent regime fields in the features.
- The data are a **testbed**; the measure/sequence comparison is the experiment.

## Reproducibility

Same build command/seed produced byte-identical artifacts. Hashes (sha256):

| file | sha256 |
|---|---|
| PATH A wide | `6f0feafb710a68808ad631d32c62567e5759a8a911ec350d0fca11f9ddfda278` |
| PATH B stream | `1122c40a31a900933802cdba93aae37163dd3b9618016a75f926692f17a6da25` |
| truth | `b0bbed3556655263cdc3a127c5f8d33a7da6d3822a3174a0e53626a15415187e` |
| manifest | `c09f1cfd8de77dc73a8de6c2bb8d789e69ae47e1a0360e91feca800d4092857c` |

Build:
```bash
python3 -m rabbit_hole.generators.relational_reference \
  --n-customers 2000 --window-days 540 --seed 17 --as-of 2025-01-01 \
  --out /tmp/opencode/t0_loop/relational_v1
python3 scripts/validate_layer_a.py --dir /tmp/opencode/t0_loop/relational_v1
```

## What is now allowed

- Train the encoder on PATH B and compare against PATH A + GBM on **value/order**
  targets, same head and folds.
- Use the truth regime runs for representation probes (can the encoder recover the
  latent regime?).
- Treat **churn** as a table baseline, not an encoder target.
- The next decisive gate: a sequence model must beat PATH A on value/orders with a
  held-out CI; otherwise the learned-encoder track does not have a business case.
