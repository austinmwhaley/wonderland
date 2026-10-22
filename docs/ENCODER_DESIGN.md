# Layer B encoder design — decisions and status

Status: **design + prototype**, not certified. The certified artifact remains the measure-first
0.6.4/0.11.0 card; this document tracks the sequence-first design we are testing.

## Pipeline, in order

```text
events → schema/roles → tokenization → token mixer → SSM → readout → E → freeze → consumers
                                            ↑
                                 self-supervised objectives
```

1. **Ingestion / schema** — canonical 4-column stream; field types inferred (numeric iff it parses);
   IDs by cardinality; actions/cadence declared or inferred.
2. **Tokenization** — event type + every discovered categorical field + every numeric field +
   data-derived periodic time + gap + static profile + missing mask.
3. **Token mixer** — project each field to `d`, **sum** (no concat), small MLP.
4. **SSM** — continuous-time gated state, learned per-channel `τ`, several timescales from the
   stream’s spectrum; full history; constant state; linear time.
5. **Readout** — last / mean / max / attention pool / magnitude branch, projected to `d`,
   learned positive weights, summed, MLP → **one `E`**.
6. **Objectives** — next event over all fields + multi-horizon value/orders at **sampled** cutoffs
   and horizons + optional geometry (contrastive/JEPA) + optional action-conditioned outcome;
   learned loss weights; field masking.
7. **Freeze / serve** — `E` at `as_of`, horizon-agnostic; block-aware scales (shape vs measure).
8. **Consumers** — supervised, unsupervised, RL heads read the same frozen `E`.
9. **Evaluation** — fixed gates; U vs oracle; power controls; learned-only attribution; consumer
   tests vs the table.

## Decisions

| Choice | Principle | Status |
|---|---|---|
| Continuous-time gated SSM: `h_t = exp(-Δt/τ)h_{t-1} + (1-exp(-Δt/τ))x_t` | Lawful, universal, adaptive | prototype next |
| `τ` learned per channel; initialized from gap/autocorrelation statistics | No magic numbers, adaptive | prototype next |
| Timescale count from spectral peaks (2–8) | Adaptive | prototype next |
| Data-derived periodicities (FFT/autocorrelation) instead of fixed hour/dow | Universal, adaptive | prototype next |
| Action cadence inferred from action-event spacing | Universal, adaptive | decision; implementation next |
| Capacity (width/depth/vocab/timescales) from data + budget | No magic numbers, efficient | decision; search next |
| Coverage-based vocab / hashing for high cardinality | Universal, efficient | decision; implementation next |
| Learned monotone numeric transform (log1p fallback) | Learn, don’t teach | decision; implementation next |
| Mask rate from field missingness/entropy; patience/tolerance from validation SE; LR measured | Dynamic, robust | prototype next |
| All readouts available; learned positive weights; no softmax gate collapse | Dynamic | prototype implemented (additive readout) |
| Sampled cutoffs + data-derived horizons; no fixed 90d curriculum | Universal, adaptive | prototype implemented |
| No table/counter/summary/distillation in `E` for the learned claim | Honest | prototype implemented |
| Block-aware serving scales | Accessible | design |

## Non-negotiable fixed things

- Evaluation targets, folds, baselines, gates.
- Causality: no future information, no `target_key`, no table in `E`.
- The table baseline.

## Current prototype results (2k reference)

- Donor-only `E` is far below the table on value/orders/churn.
- Hybrid (`table + E`) is consistently positive on value/orders with dynamic horizons, but the
  explicit sequence features (`table + seq`) are stronger.
- Decisive test running: does `E` add **beyond** `table + seq`?
- Diagnostics: additive readout avoids gate collapse; removing the magnitude branch hurts;
  churn is table-solved.

## Next

1. Implement continuous-time `τ`, data-derived periodicities/timescales, measured mask/patience.
2. Run the fast loop (2k, 1 seed, 5 epochs) one change at a time; confirm with 3 seeds.
3. Implement capacity search, vocab coverage/hashing, learned numeric transform.
4. Scale to 6k only after the 2k proxies confirm a gain.
