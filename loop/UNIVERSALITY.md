# The UNIVERSALITY metric — how we measure the "universal embedding" goal

> The north star (see `AGENTS.md` "Project goal" and `STATUS.md` "Canonical
> decomposition"). `embedding(customer, t)` must be a **minimal, bounded,
> universal sufficient statistic of the stream**. This file defines what that
> means *measurably* and how the autonomous loop drives it.

## Definition

`E` is **universal** iff it is sufficient for the **full observable future** —
i.e. for **every** downstream function `τ` of the history,
`L*_τ(history) − L*_τ(E) ≈ 0` — and this holds at **bounded capacity**.

Equivalent, operational form (conditional independence):

> **`future ⊥ history | E`** — once you know `E`, the full history adds *nothing*
> about the future.

## The metric: `U`

```
oracle-gap_τ = ( excess loss of predicting the future from E )
             − ( excess loss of predicting it from the FULL HISTORY )      # Bayes-floor referenced
U            = 1 − worst-case_{τ∈family} ( oracle-gap_τ normalized to the oracle's achievable range )
U → 1  ⟺  universal at this capacity
```

Two legs:
- **predictive** (dense surrogate): future-prediction loss from `E` vs full history.
- **decision** (the real target): `V*(full history) − V*(policy from E)` in discounted
  incremental margin (the declared reward).

## The four things the measurement must get right
1. **Reference the oracle, not a task** — comparator is the full history reaching the
   **Bayes floor** (excess loss beyond the aleatoric noise floor; noise is not lost info).
2. **Family breadth = honesty** — "every function" is infinite; measure a broad family.
3. **Worst-case, not average** — `max_τ gap_τ`; averages hide blind spots.
4. **Capacity is a constraint** — sufficiency *at fixed `|E|`* (else "keep everything" wins).

## Design pieces (what to build)

1. **Oracle-gap / history-residual test** (core). Train a strong predictor of the future
   from `E` alone vs from `[E, full history]`; the difference is what the history adds.
   `gap = L(future | E) − L(future | E, history) → 0 ⟺ universal`. Task-free ("the future").
2. **Broad, self-supervised, adversarially-grown family.** Targets = future observables
   (moments, distributions, all horizons). Hold out part of the family for *selection* vs
   *grading*. Add a **growing adversary** that searches for the task where `E` loses most.
3. **Ground truth on the Layer-A lab** (`rabbit_hole/lab.py`, `lab_oracles.py`). On synthetic
   streams the true sufficient statistic is **known** ⇒ `gap` is **exact**; verify the
   *measurement machinery itself* here before trusting it on real data. The 32-use-case
   ontology is the "unknown-use-case" proxy set.
4. **Accessibility, measured separately from sufficiency.** Require that simple probes
   (linear / kNN) on `E` reach the oracle's performance on the family — *accessibility
   parity* — so other teams' simple models can consume it.
5. **Profile → scalar.** Report per-axis (shape, mass, identity, rate, uncertainty),
   per-horizon, per-family, and the worst-case gap; reduce to `U`.

Geometry (isotropy/rank/conservation) and **similarity-usability** (kNN label-transfer on a
held-out split) are **separate gates** from sufficiency.

## Anti-gaming (binding)
- **Select on held-out self-supervised loss; GATE on `U`.** Never train on `U`.
- Reference is the **full history** ⇒ the metric never encodes our tasks.
- Family stays **broader than any consumer**, adversarially grown.
- **Known use cases VETO, never SHAPE.**
- Never weaken a gate/baseline/threshold to raise `U`.

## Honest limits
- "Unknown use cases" is only *approximated* by a broad + adversarial + held-out family.
- Only **observable, predictable** future is coverable; unlogged events/confounders
  are carried as **uncertainty**, never faked.
- `U` is **capacity-relative** and **family-relative** — always report both.

## Where we are (2026-10-14)
| piece | have | gap |
|---|---|---|
| oracle-gap idea | `sufficiency_gate` (E-alone vs raw) | references a **hand-chosen** raw set + **fixed** tasks → make oracle-referenced + broad |
| donor test | `head_battery` | fixed/known tasks → not a universality measure |
| known-truth anchor | Layer-A lab | not yet wired to compute **exact `U`** |
| adversarial family | — | need a task-search adversary |
| accessibility parity | partial (`probes`) | not measured as oracle parity on a held-out family |
| similarity-usability | — | no kNN label-transfer gate |
| `looking_glass/universality.py` | — | **build this** |

## Acceptance for the metric itself
- On the Layer-A lab, `U` is **exact** and reproduces the built-in sufficiency of the truth.
- A deliberately lossy encoder (e.g. readout without the ledger/measure pathway) scores a
  **lower `U`** than a sufficient one — the metric separates them.
- Adding capacity beyond the minimal sufficient size **does not raise `U`** (saturation).
- An **adversary** can find a task-family member where `E` loses more than the average.
