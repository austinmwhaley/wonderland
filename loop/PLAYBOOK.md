# PLAYBOOK — how the autonomous loop works toward SOLVED

The loop's operational contract: **goal + principles are the operator's; everything here is the
agent's, decided within them and recorded in `DECISIONS.md`.** Doctrine: `AGENTS.md`.

## The goal predicate (read `loop/status.sh`)
`GOAL_MET` iff, on every stream, the **B card is GREEN**, the **B+C battery is GREEN**, and
`looking_glass.universality` prints **`SOLVED = True`** (Layer 1: `E` within the exhaustive
full-history oracle's own noise on every future-family member — DEC-126). The oracle is NOT
hand-RFM, so SOLVED cannot be earned by reproducing hand features.

`SOLVED = True` is the **stop**. It is derived from the goal, attainable (asks only that the
embedding be as good as the raw history), measurable, and un-gameable.

## Guardrails (Honest / Robust — non-negotiable)
- **No self-certification.** The loop optimizes *self-supervised* training; the gates (B, B+C,
  universality) are downstream instruments Layer B never sees. Never tune the encoder on the
  specific held-out rows used to certify. Keep an **independent** certificate: a fresh split /
  fresh stream the tuning did not use.
- **Never weaken a gate, baseline, threshold, or the oracle** to pass. If `SOLVED` is unreachable
  it stays unreachable — that is information.
- **Never hand-craft the oracle** further (no re-adding RFM/windows); the oracle is the *ceiling*,
  generic and schema-derived. A stronger oracle (learned sequence model) is allowed and encouraged.
- **No magic numbers** (derive/learn); **no domain literals**; **per-sample at serve**;
  **Layer B self-supervised only**; **divergence / non-finite = hard failure**, never masked.
- **Caps:** iters/hours/cost bounded by the driver (`LOOP_MAX_ITERS`/`LOOP_MAX_HOURS`); the loop
  runs on a dedicated branch.
- **Escalation rule:** decide everything except a change to the **goal** or a **conflict between
  principles** (or a genuine no-legitimate-path dead end) — those pause, one-line question to the
  operator, and stop the run.

## Diagnosis → the axis contract (Diagnosable)
Every shortfall must be **attributed to an axis**, the fix proposed, and the cross-axis trade
recorded. The universality gate's **worst member names the axis**:

| axis | ground-truth signal | current lever |
|---|---|---|
| input / tokenizer | runs on an unseen schema with no code change; cross-schema transfer | field/value universal encoder |
| dynamics | B-card `next`/`order` skill (shape/order/timing) | selective timescales; continuous-time |
| state | universality **mass-axis** gap (LTV, order-count, next-value) | the measured pathway (ledger) |
| objective | universality U (sufficiency) | self-supervised future prediction |
| training regime | stability (no divergence) + convergence | governed training |
| interface | accessibility (probe parity) + similarity (kNN label transfer) | usable geometry |

Coupling to record every time: tokenizer richness ↔ what the dynamics express; linear-time
dynamics ↔ expressivity; unbounded measure ↔ stability; objective ↔ what the state is pressured
to hold; universality ↔ task-freeness.

## Hypothesis queue (prioritized, falsifiable; DO NOT repeat recorded negatives)
1. **State axis (the bottleneck).** Make the **measure learnable**. Recorded negatives to avoid:
   a *linear* additive drive (DEC-120) and a diluted future-aggregate objective (DEC-121) both
   lost to the accumulator. Untried, principled next: a **nonlinear** drive + a sufficiency
   objective applied *to the measure readout itself* (not the whole state), and/or a
   **content-addressed associative memory**. Rung 0→1→2; matched control = the accumulator.
   If learning still loses, the honest fallback is a **complete** explicit measure (documented).
2. **Objective axis.** Find a self-supervised objective that forces sufficiency without diluting
   (per-objective normalization; the measure readout as the prediction target).
3. **Tokenizer axis.** Universal field/value encoder; test **cross-schema transfer** (train on
   stream A, evaluate on B) — the real universality falsifier.
4. **Dynamics axis.** Selective decay spectrum incl. the integrative limit; time-aware decay.
5. **Interface axis.** Accessibility (probe parity on a held-out family) + similarity (kNN
   label-transfer) as their own gates.
6. **Regime axis.** Stability + convergence receipts.

## Rung discipline (cheapest falsifier first)
Rung 0 = measure on an existing checkpoint (no training) → Rung 1 = tiny train
(`--customers 2000 --sample-a 1500 --set max_steps=250`) → Rung 2 = full. **Always a matched
control (one variable).** Keep the fast gate green each step
(`ruff check . && ruff format --check . && pytest -m "not slow"`). Keep what wins; delete dead
ends. Record a DEC entry (context / what / measured / verdict / principles served) + STATUS.md.
Commit code+docs only (never data `*.pt/*.parquet/*.duckdb`).
