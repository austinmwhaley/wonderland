# looking_glass

**Layer B — the frozen customer-foundation donor.** looking_glass reads the
canonical rabbit_hole event stream and trains one self-supervised state-space
encoder (`CFM`) whose frozen per-customer state is the feature table every
plugin trains on. It is the only writer of state and embedding tables; plugins
and downstream layers consume them read-only.

```
rabbit_hole stream → [ (0) ladder ] → (1) CFM train + products → (1b) daily state job → plugins
```

## The encoder

`CFM` (`cfm_model.py`) is a **selective multi-scale SSM**: input-dependent
Δ/B/C projections solved by a vectorized O(log T) affine scan, with a bank of
K timescale experts whose decay priors come from the measured inter-event
half-life (`autotune.py`). It trains self-supervised — next event type/entity/
time/value, occurrence, temporal order, contrastive, JEPA latent prediction,
successor features — with learned per-objective uncertainty weights, then
freezes. The single recurrence state is the public representation: constant-
time fade/absorb means the daily job advances every customer in seconds, not a
full re-forward.

## Why this shape (measured)

- **Sample decoupling:** the ladder picks training size from receipts
  (`artifacts/cfm_ladder/summary_<as_of>.json`); encoder cost is independent of
  customer count N. Only storage and the daily job scale with N.
- **Sufficiency battery:** donor beats raw aggregates on 6/8 targets, unique
  signal 75% — the frozen state is a universal donor (re-run:
  `python3 -m looking_glass.sufficiency_battery`).
- **One day boundary:** `as_of_epoch()` (`cfm_config.py`) is the single UTC
  midnight interpretation shared by the encoder cut, the daily-job window, and
  inference lookups. The daily job closes through **yesterday** only.

## Commands (run from repo root)

| Step | Command |
|---|---|
| size Sample A (ladder) | `python3 -m looking_glass.customer_foundation_model ladder --customers 25000 --as-of YYYY-MM-01 ...` |
| (1) train encoder + products (also closes day 1) | `python3 -m looking_glass.customer_foundation_model train --customers 25000 --anchors 6 --as-of YYYY-MM-01 --db rabbit_hole/data/duckdb/customer_event_stream.duckdb --out-dir looking_glass/artifacts/cfm` |
| rebuild products only (fast knob sweeps) | `python3 -m looking_glass.customer_foundation_model products --anchors 6 --out-dir looking_glass/artifacts/cfm` |
| (1b) daily state job, days 2..N | `python3 -m looking_glass.daily_states --as-of YYYY-MM-DD` |
| battery (donor vs raw) | `python3 -m looking_glass.sufficiency_battery` |
| dense weekly/daily states (red_queen input) | `python3 -m looking_glass.state_dense` |
| tests | `python3 -m pytest` (fast tier: `-m "not slow"`) |

## Data contract

The canonical stream is Apache Arrow queried with DuckDB (no SQLite, no
pandas): `customer_key, event_ts, brand, event_type, event_attributes` plus
`value`/`source_table`. Products live in `cfm_products.duckdb`:
`donor_embeddings` (training rows), `customer_state`, `state_embeddings`
(inference rows, materialized daily), `encoder_samples`, `state_job_receipts`.

## Package layout

```
cfm_config.py               config + as_of_epoch (the one day-boundary helper)
cfm_data.py                 point-in-time cuts, splits, samples, window reads
cfm_model.py                EventVocab, SelectiveSSM, MultiScaleSSM, CFM
cfm_training.py             governed training (governor, warm-start, receipts)
cfm_state.py                StateStore (only writer), fade/absorb, build_products
cfm_validation.py           validation probes (causal next-event, objectives)
customer_foundation_model.py  CLI entry (train / products / ladder / rebuild)
daily_states.py             (1b) daily job: closes yesterday, rematerializes embeddings
state_dense.py              dense daily/weekly states for red_queen's decision log
sufficiency_battery.py      E vs raw vs scrambled-E sufficiency gate
layer_b_proof.py            grouped-CV donor-vs-raw probes (battery helper)
autotune.py                 data-derived knobs (half-life from event gaps)
```

## History

This package previously exposed a second, non-production stack — the
`create_embedding_model` / `create_temporal_core_model` / `create_supervised_model`
factory API (EntityCore + SequenceEngine with Mamba-2/Samba backends). Nothing
downstream ever consumed it; the CFM path above is the only stack the system
uses. It was deleted (doctrine: prefer deletion over accretion) along with its
tests, demo, `lancedb`/`mamba-ssm`/`bitsandbytes` dependencies. Specs in
`specs/` that predate the deletion are historical records.
