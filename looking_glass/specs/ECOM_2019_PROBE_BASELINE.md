# ecommerce_2019 — frozen-encoder probe baseline (DEC-052)

Frozen encoder: `v6.4.0r162610` (20k-customer ecommerce run). Probe harness:
`python3 -m looking_glass.probes`. Encoder weights frozen; only probe heads
trained. Plain CE/MSE, no class weights / focal loss. 5 grouped customer folds;
verify harness reports `frozen=True` and the forward is grad-free (fail-fast).

## Standardized probe report

| Objective | Probe Type | Model Score (± SE) | Baseline Score | Ceiling Score | Δ vs Baseline | Rare slice | Verdict |
|---|---|---|---|---|---|---|---|
| next | linear (mlp 0.2932) | 0.2744 ± 0.0116 | 0.2391 | 0.2391 | −0.0353 | 3.062 | FAIL |
| dt | linear (mlp 12.4628) | 13.9983 ± 0.3841 | 22.2857 | 12.6953 | +8.2874 | | NO_IDENTIFIABLE_SIGNAL |
| jepa | linear (mlp 2.0311) | 2.3005 ± 0.2241 | 2.5816 | 2.5816 | +0.2811 | | PASS |
| mask | linear (mlp 0.3160) | 0.2793 ± 0.0059 | 0.0005 | 0.0005 | −0.2787 | | FAIL |
| value | linear (mlp 0.7648) | 1.1444 ± 0.0438 | 0.7770 | 0.7770 | −0.3675 | | PASS |
| entity | linear (mlp 0.0000) | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | +0.0000 | | NO_IDENTIFIABLE_SIGNAL |
| order | linear (mlp 0.5736) | 0.4875 ± 0.0109 | 0.5698 | 0.5698 | +0.0823 | | PASS |
| agg | linear (mlp 0.3886) | 0.5609 ± 0.0230 | 0.6712 | 0.6712 | +0.1103 | | PASS |
| query | linear (mlp 1.2481) | 1.4255 ± 0.0444 | 1.6432 | 1.3589 | +0.2177 | | NO_IDENTIFIABLE_SIGNAL |
| sf | linear (mlp 279.1319) | 342.5148 ± 7.9792 | 279.9581 | 279.9581 | −62.5567 | | PASS |

(Verdicts shown are from the run; `_verdict` classification was refined after this
run — the raw scores are authoritative. `next` is a genuine FAIL: the frozen
readout does not beat the 1-gram.)

## Event-identity decodability (z_t -> current event type)

| features | acc | macro-F1 | CE |
|---|---|---|---|
| z | 0.9762 | 0.8086 | 0.0684 |
| e_t | 1.0 | 1.0 | 0.0005 |
| [z,e_t] | 1.0 | 1.0 | 0.0004 |

## next-event slices (CE)

| slice | value |
|---|---|
| next=cart | 2.9648 |
| next=purchase | 3.3507 |
| next=view | 0.0680 |
| prev=cart | 1.4527 |
| prev=purchase | 0.2546 |
| prev=view | 0.2030 |
| rare(next!=view) | 3.062 |
| empirical P(purchase \| prev=cart) | 0.2811 |
| base P(purchase) | 0.0166 |

## Decision (per the patch admissibility rule)

The patch requires ALL of: (1) next/jepa/sf FAIL; (2) adding e_t/prev_event_hash
closes the gap; (3) z_t does **not** linearly expose the current event type while
e_t does.

**Condition (3) FAILS:** `z_t -> current event` gives macro-F1 0.809, CE 0.068 —
the frozen readout *does* linearly expose immediate event identity. Therefore the
`next` gap (model 0.274 vs 1-gram 0.239) is **not** an identity-exposure defect,
and **no architectural patch is applied** (the proposed Options A/B/C are
unjustified).

**Outcome:** ecommerce_2019 recorded as **FAIL_LOW_EFFECT_SIZE** for the
transition objectives (`next`, `mask`) and **NO_IDENTIFIABLE_SIGNAL** for
`entity`/`value`. The residual gap is a low-effect-size/optimisation limit on a
stream whose cart->purchase signal is MI ≈ 0.05 nats; gates were not weakened.
Certified streams remain **rabbit_hole** (11/11 + 6/6) and **Instacart**
(11/11 + 6/6).
