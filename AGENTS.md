# AGENTS.md — Working Principles for `wonderland/`

This file governs every change to the system. It is a **doctrine**, not a
checklist of values. Where an example implies a constant or a procedure, treat
it as an *intent* and implement the mechanism that achieves it **adaptively**.

The system (one-way): `rabbit_hole → looking_glass → white_queen → red_king →
red_queen`, with `caterpillar` as the read-only interpretability layer.

---

## The one rule

**Never encode an answer the system could learn or derive.**

A hardcoded constant is a decision that has been moved away from where the
information is. Push every decision to the place with the most information: the
data, the task, or a learner. If a literal must exist, it is a **fallback** —
resolved at runtime, recorded in a receipt, and overridable — never the source
of truth.

## Data layer

The canonical stream is **Apache Arrow** (`.arrow`/`.feather`, uncompressed IPC)
— memory-mapped and zero-copy into Polars/DuckDB. **DuckDB** is the query engine
over it; **Parquet** is for compressed archival. No SQLite, no pandas. Read via
`rabbit_hole.stream.read_frame`; keep consumers DataFrame-native (no row-wise
dict materialization).

---

## Principles

1. **No magic numbers.** Every threshold, budget, rate, dimension, window,
   weight, or cap is (a) inferred from the data (distributions, quantiles,
   measured statistics), (b) derived from problem size and resources, or
   (c) learned. A surviving literal is a documented conservative fallback.
   Smell: fixed step counts, fixed patience, fixed thresholds, hardcoded
   network widths, fixed clip caps.

2. **Train to convergence, not to a count.** "Perfectly trained" means *best
   held-out performance, stopping when improvement plateaus within measurement
   noise*, under a resource-aware budget derived from data size and hardware —
   never a fixed epoch/step count. Use governed training (adaptive schedules,
   numerical guards, early stop on a **grounded** metric). Never select weights
   by the same self-referential metric they optimize; validate on held-out or
   ground-truth signal.

3. **Make it learn; do not teach it.** Prefer learned behavior, representations,
   and weights over hand-specified rules — but only where supervision or labels
   actually exist. Supply structure and objectives; let the system discover
   thresholds, mixtures, and policies. If a heuristic is unavoidable, make it
   **calibratable from data** (empirical/conformal calibration), not a human
   guess.

4. **Adapt to the input by construction.** Every component resolves its own
   configuration from what it sees — schema, dimensionality, horizon, action
   space, reward scale, episode length, cadence. No assumptions about domain,
   environment, or units. An unseen shape must work without code changes.

5. **Modularity and contracts.** One responsibility per module; explicit
   interfaces/dataclasses; swappable backends behind those interfaces;
   one-way dependencies; no cross-layer coupling. Adding a capability must not
   force edits in unrelated layers.

6. **Single source of truth; no duplication.** Each definition/artifact lives
   in exactly one place. Copies are bugs; link or import. If two things
   disagree, delete one. One generator, one schema, one implementation.

7. **Fail safe; reject, do not clamp.** Invalid, out-of-distribution, or
   uncertain results are refused or carry widened uncertainty — never silently
   substituted. A silent default converts a bug into false confidence.

8. **Uncertainty first.** Every estimate is reported with its uncertainty and
   coverage, calibrated empirically and verified against ground truth. A point
   estimate without an interval is unfit to drive a decision.

9. **Measure everything; gate on ground truth.** No capability is "done"
   without an independent validation gate with known-truth checks. Freeze the
   benchmark; make results reproducible; report per-capability metrics, not a
   single composite that hides failure.

10. **Reproducibility and versioning are identity.** Deterministic runs
    (seeded). Frozen, versioned artifacts. The version label encodes *behavior*
    (a change that can alter outputs bumps the version; a pure refactor does
    not) and a data/score revision. Every artifact records the inputs, config,
    and version that produced it.

11. **Observability.** Emit receipts: what was derived, measured, assumed, and
    which fallback fired. A decision that cannot be explained is not finished.

12. **Implement intent, not the example.** When given a specific value or
    procedure, ask what property it is meant to guarantee, and build the
    mechanism that guarantees it adaptively. Example: "train for N steps"
    means "train until genuinely converged, with a derived budget."

13. **Prefer deletion over accretion.** Remove dead code, superseded paths, and
    one-off special cases. Fewer moving parts is more adaptable.

14. **Use the best tool for the job.** Prefer a mature, vetted library over
    reimplementing it (e.g., use scikit-learn/scipy for probes and statistics
    rather than hand-rolling them). Install what is missing. Reimplementation is
    reserved for where no tool fits or the tool cannot meet the contract.

15. **Cost-aware capability.** Minimize training and operational cost
    *structurally* — incremental updates, caching, constant-time recurrence,
    bulk/columnar I/O — rather than by cutting quality. Optimize capability per
    unit cost.

16. **Speed is a first-class requirement.** Build for speed wherever it does not
    sacrifice capability, *from the start* (not as an afterthought). Vectorize by
    default: no per-item Python loops over rows, time, or examples when a
    tensorized/columnar/parallel-scan form exists; use Arrow/Polars bulk ops and
    associative scans over recurrences; derive budgets from data + hardware.
    Every component emits a **runtime receipt** and scales predictably with input
    size. A capability that is correct but unacceptably slow is **not done**.

---

## Anti-patterns → preferred

| anti-pattern | preferred |
|---|---|
| fixed step/epoch count | converge on held-out metric, derived budget |
| hardcoded threshold | learned or empirically calibrated threshold |
| per-environment `if/else` | schema/registry-driven dispatch |
| duplicated constants/logic | single resolver / one definition |
| silently clamping to a bound | reject the estimate, widen uncertainty |
| hand-written label rules | a learned head where labels exist |
| fixed cadence | event/time-driven, data-decided cadence |
| selecting weights by the training metric | held-out / ground-truth selection |
| domain-specific special case | adaptive resolution over the input |
| copying artifacts across projects | reference/link (single source of truth) |
| per-item Python loop over rows/time/examples | vectorized op / parallel scan / columnar expression |
| row-wise dict I/O at scale | Arrow/DataFrame zero-copy path |
| Python bootstrap/loop over data | tensorized / vectorized computation |
| runtime hidden / untested at scale | runtime receipt; test at 500 then scale up |

---

## Agent self-review before every change

- Did I add a literal that could be **derived or learned**? Can it move to
  runtime resolution?
- Does this generalize to an **unseen schema/dim/horizon/action space** with no
  code edits?
- Is the decision made **where the information is greatest** (data/learner)
  rather than in code?
- Is there **one source of truth**? Did I duplicate anything?
- Does it **fail safe** and carry **uncertainty**?
- Is there a **ground-truth validation** proving it, not merely that it ran?
- Can I **delete** something instead of adding?
- Am I implementing the **intent** behind the instruction, or literally its
  example?
- What is the **cost** (training and ongoing inference) per unit capability?
- Did I introduce a **Python loop over rows/time/examples**? Can it be a scan,
  gather, einsum, or a Polars expression instead?
- Does it **scale**? Test at 500 first, then scale up; record the runtime.

---

## Failure modes to watch

- Fixed budgets that under- or over-train while appearing successful.
- Self-referential selection that crowns degenerate snapshots.
- Cross-layer leakage (e.g., downstream labels influencing the upstream
  representation).
- Domain special-casing that silently excludes new input shapes.
- Silent defaults/clamps that masquerade as caution.
- Duplicated logic that drifts between copies.

---

## Direction of flow

Layers flow strictly one way. New requirements become **new upstream versions**,
never backward edges. Personalization happens in small downstream heads, never
by mutating a frozen foundation. `caterpillar` reads everything and changes
nothing.

---

## Objective (the one metric)

**Every RL/decision model optimizes the SAME reward: long-term INCREMENTAL gross
margin, discounted** (infinite-horizon, `G = Σ γ^k · Δgross_margin`). "Incremental"
= causal lift over the counterfactual, never raw/correlated margin. Train on
de-biased observed margin; evaluate on incremental counterfactual margin.

Corollaries:
- Any policy component (white_queen, red_king, red_queen heads) uses this reward.
- Plugin *prediction* heads may forecast components (margin, churn, timing); only
  the *decision* objective is the discounted incremental margin.
- Live exploration is offline-first; a **small randomized holdout** is permitted
  and is the ground-truth source for validating counterfactual models.

## Scope & cadence
- Actions: **email frequency first**, then promotions / timing / channel.
- Cadence: **hourly / daily / weekly** next-best-action.

## Definition of done (v1)
The system is v1-complete when ALL hold:
1. rabbit_hole emits a causally-identifiable stream (randomized arms + known effect).
2. looking_glass is a **universal donor** (battery: adds unique signal; never
   materially worse than raw) usable by every plugin.
3. Each plugin (supervised, unsupervised, white_queen) passes its **independent** gate.
4. red_king recovers **known** counterfactual effects and improves white_queen's
   certified decision **with vs without** it (never ships worse than logging).
5. red_queen composes next-best actions across cadences under constraints.
6. caterpillar answers "why" from artifacts only.
7. Every artifact versioned + reproducible; every run emits receipts.

## Agent operating mode

Work **autonomously and continuously**: chain the task list within a turn, update
`STATUS.md` after each phase, stop only for genuine decisions/blockers. Prefer
doing over asking. Maintain the ladder (small → scale) and the battery gate on
every change.

---

## Repository & git workflow

Source of truth: **https://github.com/austinmwhaley/wonderland** (branch `main`).

- **What is committed:** source code, specs, docs, small configs.
- **What is NEVER committed** (see `.gitignore`): data (`*.duckdb`, `*.feather`,
  `*.arrow`, `*.parquet`, `*.npz`), model checkpoints/artifacts (`*.pt`, `artifacts/`,
  `checkpoints/`, `results/`), virtualenvs (`.venv/`), caches (`__pycache__/`), logs.
  These are **regenerable** (rabbit_hole generates the stream; runs produce artifacts).
- **eighth_square owns** the shared `algorithms/` and `environments/`
  (`wonderland/algorithms` and `wonderland/environments` are symlinks into it).
- **Workflow:** `git pull --rebase` -> make changes -> `git add -A` -> `git commit`
  -> `git push`. One repo, one history; do not create nested `.git` directories.
- **Large data lives only locally** (or a separate storage/DVC), never in this repo.
