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

`CFM` (`cfm_model.py`) is a **unified multi-timescale selective SSM**:
input-dependent Δ/B/C projections solved by a vectorized O(log T) affine scan,
with one wide state whose per-channel decay spans a continuous fast→slow
timescale spectrum (the learned Δ *is* the per-event time-constant). It trains
self-supervised on a portfolio of objectives — next event type/entity/time/value,
occurrence, temporal order, contrastive, JEPA latent prediction, successor
features, query, multi-horizon aggregation — alongside **structural invariant**
objectives, then freezes. The state is the public representation: constant-time
fade/absorb advances every customer in seconds, not a full re-forward.

## Design philosophy: mathematical self-governance

Every layer measures, enforces, and maintains its own invariant rather than
being tuned by hand. Structural problems are fixed with mechanisms, never with
per-run hyperparameter search or artificial bottlenecks.

1. **Invariants as native architecture, not post-hoc patches.** Isotropy,
   continuity, and recurrence stability are enforced *in the optimization
   dynamics* — self-throttling barrier functions and orthogonal gradient
   projections — not bolted on after a failure.
2. **Capacity over artificial constraints.** The trunk keeps its full 256
   channels. We never shrink the model to make a gate pass; we fix the
   objectives so the trunk uses its available capacity.
3. **Truth-first diagnostics.** Gates are honest, scale-free inspectors. A
   failing gate is an objective signal of an unaddressed task/geometry tension,
   not something to hide or bypass.
4. **Zero-leakage separation of concerns.** Fast event dynamics and slow
   long-horizon trends are isolated so they cannot interfere.

### Native mechanisms

| Concern | Mechanism |
|---|---|
| task vs structural gradients | **task-structural PCGrad** — invariant gradients projected onto the null space of the predictive-task gradient, so structural pressure can never reduce skill |
| manifold isotropy | **barrier `iso`** — hinge on participation-ratio/dim below the calibrated floor, with a sigmoid multiplier that self-throttles to 0 as it is met |
| trajectory continuity | **band-isolated `trajectory`** — acceleration penalty `‖Δ²h_slow‖²` applied only to the slow band (fast channels stay free) |
| boundary equalization | **global-EMA ZCA** at the donor boundary, entropy-weighted update rate, frozen at inference (train/serve share one map) |
| recurrence stability | decay `exp(−softplus(·)) ∈ (0,1)` — a strict contraction, stable by construction |
| evaluation | **scale-free gates** — participation ratio vs a *calibrated structured-manifold floor*; OOT via principal-angle subspace overlap (bounded [0,1]) |

## Measured result — v4.3 (Layer-B report card)

Run on the Instacart fixture, as_of 2025-11-01, reproduced across runs:

```
PORTFOLIO   13/13 self-supervised skills PASS
            geometry PASS   PR/dim 0.289–0.297  >= calibrated floor 0.25
            canary   PASS   (state does not leak assignment/period)
INTRINSIC   5/5 PASS
            slow-state trajectory cos +0.119
            OOT subspace overlap      0.970
            channel MI                0.012 nats
            Lipschitz p99             0.00014 / s
            information plane         16 losses @ 3 horizons
```

The geometry bar is honest, not white-noise: the data has an intrinsic
dimensional ceiling of ~0.30 PR/dim under linear readout (even an 8192-state
population bank cannot exceed it). The gate asks whether the readout uses the
*available* semantic manifold volume — which it does. Reproduce with
`python3 -m looking_glass.customer_foundation_model portfolio --out-dir <dir>`
and `python3 -m looking_glass.intrinsic --out-dir <dir>`.

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

## Test streams

The encoder is stream-agnostic; the same engine is graded on each fixture below
(all local-only, `.duckdb` is gitignored). Pass `--db <path> --as-of <date>` to
`train`/`portfolio`/`intrinsic`.

| stream | fixture | scale |
|---|---|---|
| **Layer A** (rabbit_hole synthetic) | `rabbit_hole/data/duckdb/customer_event_stream.duckdb` | 26.8M events / 25k customers |
| **Instacart** (external) | `rabbit_hole/data/instacart/customer_event_stream.duckdb` | 37.4M events / 206k customers |
| **ecommerce_2019** (external, Kaggle) | `rabbit_hole/data/ecommerce_2019/customer_event_stream.duckdb` | 4.4M events / 212k customers (Oct+Nov 2019; see `ecommerce_receipt.json`) |

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
portfolio.py                Layer-B grade: structure-skill + geometry receipt
intrinsic.py                intrinsic foundation proofs (disentanglement,
                            Lipschitz, trajectory, information plane)
```

Reference: **`specs/objectives_catalog.html`** (open in a browser) — every self-supervised
objective (the 13 we train, the 7-family operator menu, the full landscape),
what each forces into the state, and the capability-driven plan for turning
them on/off per stream (DEC-009/DEC-010).

## History

This package previously exposed a second, non-production stack — the
`create_embedding_model` / `create_temporal_core_model` / `create_supervised_model`
factory API (EntityCore + SequenceEngine with Mamba-2/Samba backends). Nothing
downstream ever consumed it; the CFM path above is the only stack the system
uses. It was deleted (doctrine: prefer deletion over accretion) along with its
tests, demo, `lancedb`/`mamba-ssm`/`bitsandbytes` dependencies. Specs in
`specs/` that predate the deletion are historical records.
