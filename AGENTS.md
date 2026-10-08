# AGENTS.md — Working Principles for `wonderland/`

This file governs every change to the system. It is a **doctrine**, not a
checklist of values. Where an example implies a constant or a procedure, treat
it as an *intent* and implement the mechanism that achieves it **adaptively**.

The system (one-way): `rabbit_hole → looking_glass → plugins → red_king →
red_queen`, with `white_queen` (OPE certification) and `caterpillar` (read-only
interpretability) hanging off `plugins`.

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
over it; **Parquet** is for compressed archival. **No SQLite, no pandas — there
is no SQLite anywhere in this project** (the colony store migrated to native
DuckDB; looking_glass loaders read DuckDB; `white_queen/tribunal/ope` ingests
`.duckdb`/Parquet/CSV/JSON). Read via `rabbit_hole.stream.read_frame`; keep
consumers DataFrame-native (no row-wise dict materialization).

Style: **spaces** (4) everywhere — `ruff format` is authoritative; no tabs.

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

Work **autonomously and continuously**: chain the task list within a turn,
prefer doing over asking, stop only for genuine decisions/blockers. The
sections below are the execution framework; where they disagree with the
engineering doctrine above, the doctrine above wins.

### Core rules

1. **Earn the right to scale** — smallest viable footprint first (a spike, a
   single script, one batch); validate, then modularize, then optimize. Never
   build for 10x before correctness is proven at 1x.
2. **Simple over complex** — the least complex solution that satisfies the
   current constraint; delete abstractions that a function could replace.
3. **Modular & loose coupling between components** — components talk through
   explicit interfaces; swapping one (model, store, tool) changes no neighbor.
4. **Dynamic & orchestrated** — behavior lives in config, not code: runtime
   toggles, paths, horizons, budgets are config/receipts, never hardcoded
   settings. (In this repo: `CFMConfig` + derived-resolution receipts are the
   config surface.)
5. **Maximize capability per cost** — treat time/compute/memory/API as
   budgets; cheapest tool that can do the sub-task reliably; cache
   deterministic intermediates.
6. **Robust & antifragile** — fail safe with the fixing command in the error;
   explicit fallbacks; failures improve the system (a failing gate is
   information, not noise).

### Trade-off hierarchy (resolve conflicts top-down)

1. Correctness & robustness  2. Simplicity  3. Execution speed & cost  4. Scalability & abstraction.
Never sacrifice 1 or 2 for speculative 4.

### Lifecycle (any new task)

`Stage 1: Spike` (hardcode freely, verify the concept) →
`Stage 2: Modularize` (extract to config, split responsibilities, add
fallbacks) → `Stage 3: Scale` (optimize only where measured data justifies).

### Operational artifacts (mandatory)

- **`STATUS.md`** — Macro (primary goal, current stage, progress %, blockers)
  + Micro (current task, last completed, next 3 actions, environment), kept
  current at every phase transition. Read it first to restore state; update
  before yielding.
- **`DECISIONS.md`** — append-only decision log (context / alternatives /
  decision+principle / trade-offs). Check it before non-trivial changes;
  log every non-trivial decision so choices are never re-litigated.

### Decision self-check (before non-trivial action)

Simplicity (standard-library instead of a layer?), Cost/Capability (cheapest
reliable tool?), Config (any parameter hardcoded in source?), Antifragile
(what happens on timeout/failure?), State (logged in DECISIONS.md, STATUS.md
updated?).

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
- **Quality gates (run before every commit):** `pytest` (full suite; acceptance
  tests skip when generated data is absent), `ruff check .`, `ruff format --check .`.
  CI runs the same gates on push/PR. Dependencies live in `pyproject.toml`
  (mirrored by `requirements*.txt` for pip).
- **Workflow:** `git pull --rebase` -> make changes -> `git add -A` -> `git commit`
  -> `git push`. One repo, one history; do not create nested `.git` directories.
- **Large data lives only locally** (or a separate storage/DVC), never in this repo.

---

## v6.0 algebraic law enforcement (DEC-036) — Layer B

Where a property can be expressed as an **algebraic law**, enforce it as a law,
never as a tuned threshold or a manufactured metric:

- **Monoidal functor**: events are morphisms on a continuous state space; the
  encoder `F` maps sequence-concatenation to composition of affine maps. The scan
  makes `F(g∘f)=F(g)∘F(f)` **exact** (a canary, not a loss).
- **Isometry over whitening**: the donor boundary is a per-sample orthogonal
  `R` (κ=1, invertible). A boundary must **never fake capacity the trunk lacks**;
  a low-rank diagnostic is an instruction to the trunk/data, not a defect to mask.
  (Batch-coupled ZCA amplified near-null directions to manufacture rank — banned.)
- **Lossy monoid, not group**: bounded memory ⇒ forgetting ⇒ no exact inversion
  in the encoder; inversion is a bounded-window Layer-C contract.
- **Derived invariants**: conditioning floor from Marchenko–Pastur
  `λ₊=(1+√(D/N))²`; conservation ratio `Tr(Σ_readout)/P_in≈1`; commutation on
  data-certified independent pairs; Markov sufficiency gap `Δ_suff` vs a shuffled
  null; scale-invariant Gram decorrelation `‖D_Σ^{-1/2}Σ_h D_Σ^{-1/2}−I‖_F²`.
- **Zero domain literals**: no business event names, no feature partitions;
  exogenous events are data/schema-declared; one whitening path (Newton–Schulz).

Macro filter for every future change: does it add a literal? → reject. Does it
rely on cross-sample batch statistics? → reject. Does it enforce an invariant law
over the continuous state manifold? → approve.

### Monotonic stability is a hard invariant (v6, DEC-039)

A self-governing engine must be **stable from step 0 to the final iteration**.
A run that diverges and is then "rescued" by keeping the best pre-divergence
state is a **FALSE GREEN** and must never be certified.

Rules:
- **Learning rate from the curvature law**: descent is stable iff `lr < 2/L`
  (`L` = local Lipschitz of the gradient). `lr` is measured, not tuned; no clip
  ceiling is allowed to do the work.
- **Divergence is failure, not a rescue**: a non-finite metric or a spike far
  above the measured noise floor marks the run `UNSTABLE`; the portfolio gate
  fails it. The governor does not launder divergence into a green.
- **No silent masking**: non-finite terms, swallowed exceptions, and clamps that
  hide instability are banned. Failures surface as receipts, not as green scores.
- **No unit-dependent constants**: every derived scalar must be dimensionless or
  derived from a data/hardware statistic — never `f(raw_units)` propped up by a
  clip.
