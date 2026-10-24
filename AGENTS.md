# AGENTS.md — Working Principles for `wonderland/`

This file governs every change to the system. It is a **doctrine**, not a
checklist of values. Where an example implies a constant or a procedure, treat
it as an *intent* and implement the mechanism that achieves it **adaptively**.

The system (one-way): `rabbit_hole → looking_glass → plugins → red_king → red_queen`,
with `white_queen` (OPE certification) and `caterpillar` (read-only interpretability)
hanging off `plugins`.

## The project — a universal foundation model for event streams

**Input an event stream → encode it → produce a foundation model that emits UNIVERSAL
FOUNDATIONAL EMBEDDINGS.** That is the current focus. Concretely: one pretrained,
self-supervised, domain-agnostic encoder whose FROZEN output is the "universal donor"
that everything downstream builds on.

Work in **two layers — do not conflate them**:

- **Layer 0 — the universal foundation embedding (NOW).** **One stream in → one foundation model
  out:** for any given event stream, one encoder (trained on *that* stream) maps its events to
  per-customer, per-time vectors, **universal** in the sense the pipeline adapts to any schema:
  - **self-supervised** (no labels in its training),
  - **reusable / frozen** (downstream reads it; never mutates it),
  - **schema- and domain-agnostic** (the *pipeline* handles any schema, no code changes; a model
    is trained **per stream** and never cross-applied),
  - **transferable** (many downstream tasks — including held-out/unseen ones — get signal
    from it above raw features),
  - **shared geometry** (similar customers land near each other — a general space).
- **Layer 1 — certification (NEXT).** Quality guarantees layered on top of a universal
  embedding: *sufficiency* (nothing that matters is lost), *minimality* (smallest such
  object), *decision-sufficiency* (counterfactual/action sufficiency), measured by `U`
  below. These are upgrades to **certify**, not part of the definition of a foundation model.

The aspiration the certification layer measures (Layer 1, precise):

> `embedding(customer, t)` = a **minimal, bounded, universal sufficient statistic** of the
> event stream — the smallest fixed-size object that loses nothing about the future, for
> prediction AND decisions, such that **any present or future consumer — ours or another
> team's, known or unknown — is covered BY CONSTRUCTION**, not by enumeration. It must be
> **predictive** (supervised), **structure-revealing** (unsupervised, usable geometry), and
> **decision-sufficient for the declared action/reward family** (RL loses nothing): a perfect
> distillation, not a lossy summary, *including magnitude/level* (which summaries like RFM
> keep and a contractive state must not lose).

**The certification metric** (`U`): `U = 1 − worst-case_{τ} oracle-gap_τ`, where oracle-gap is
the EXCESS loss (beyond the Bayes floor) of predicting the future from `E` vs from the **full
history**, over a **broad, adversarial, held-out, self-supervised** family, at **fixed
capacity**. `U → 1` ⟺ sufficient. Never judged against our known tasks. See
`loop/UNIVERSALITY.md`.

**Anti-overfit clause (binding).** To stay universal and not drift into designing for our known
use cases:
- Layer B is **self-supervised only**; no downstream labels/tasks in its training.
- The **reference is the full-history oracle** (Bayes floor), never a hand-chosen feature set or
  our task list.
- The task family stays **broader than any consumer** and is **adversarially grown** to expose
  blind spots; universality is a **worst-case (minimax)** property, reported at fixed capacity.
- **Known use cases may VETO (gates), never SHAPE the encoder.**
- Sufficiency guarantees information is *present*; **accessibility** (simple probes / usable
  distances) and **geometry** (isotropy/rank) are separate, gated requirements.
- Coverage is over **functions of the observed stream**; unlogged events/confounders are carried
  as **uncertainty**, never faked.

---

## Top-level principles (the operator's charter)

These are the project's **top-level principles** — deliberately **broad strokes**. **Every
decision must align to them**; a change that violates one is wrong regardless of any metric it
improves. The detailed engineering doctrine below ("Principles" 1–16) **elaborates** these — it
must not contradict them.

1. **Universal** — **one *architecture* for any event stream**: schema-agnostic, resolves its own
   configuration from whatever it sees. **A model is trained *per stream* and is NEVER applied to a
   stream it did not train on** — "1 stream in → 1 foundation model out." Universality is a property
   of the *pipeline*, not of shared weights across streams. (subsumes *adaptability-of-architecture*,
   *scale-invariance*.)
2. **Adaptive & Dynamic** — everything resolves or learns from the input; behavior lives in
   learned state / config, never in source; input-conditioned dynamics. (subsumes
   *adaptability*, *dynamism*, *self-supervision*.)
3. **Lawful** — built on universal laws and symmetries: **compositionality** (the state is a
   mergeable homomorphism over event sequences), **equivariance** (time-translation, entity-
   and field-permutation symmetry), and the **correct algebra per quantity** (additive for
   extensive mass, bounded/contractive for shape, idempotent for presence if ever needed).
   (subsumes *invariants by construction*. Laws and symmetries outrank convenience.)
4. **Efficient** — linear-time, constant-state, incremental, minimal capacity; a capability
   that is correct but unacceptably slow, or needlessly large, is **not done**. (subsumes
   *speed*, *incrementality*, *minimality*, *cost-awareness*.)
5. **Robust** — fail safe and reject rather than clamp; **calibrated uncertainty first**;
   deterministic and stable from step 0; divergence / non-finite is a hard failure, never
   masked or rolled back. (subsumes *determinism*, *calibration*, *stability*.)
6. **Honest** — grounded in held-out / ground-truth signal and **falsifiable**; label-free
   upstream; never tuned to the metric it is judged by; never weaken a gate to pass. (subsumes
   *generality over fitting*, *self-supervision*, *falsifiability*.)
7. **Accessible** — information must be **recoverable by simple downstream consumers**, not
   merely present; a sufficient-but-entangled representation is a failure of this principle.
8. **Diagnosable** — the loop must **attribute a shortfall to a specific axis, propose the
   change that resolves it, and reason about how axes interact** (see below). Without this the
   design space is inert: it is how we turn "the embedding isn't good enough" into "axis X is
   the bottleneck, here is the fix, and here is what it costs axis Y."

**Conflict order:** Robust → Universal → Adaptive → Lawful → Efficient → Honest, with
**Lawful (compositionality/equivariance) and Robust outranking convenience**.

### The loop ↔ axis contract (Diagnosable, in practice)

The design is decomposed into **axes** (input/tokenizer · dynamics · state · objective ·
training regime · interface; see the design-space notes). Each axis must be **independently
observable** so the loop can localize work:

- For every axis, keep an **axis scorecard**: a ground-truth metric + a *working / not* state +
  the current lever. A failing embedding must map to one (or more) axes.
- Attribute the bottleneck: run the cheapest experiment that **isolates one axis** on ground
  truth (matched control), and name the axis.
- Propose the **resolving solution** for that axis, cited to the principle it serves.
- Account for **coupling**: axes constrain each other, so a fix to one may cost another —
  e.g. tokenizer richness ↔ what the dynamics can express; linear-time dynamics ↔ expressivity;
  unbounded measure ↔ stability; objective ↔ what the state is pressured to hold; tokenizer
  universality ↔ objective task-freeness. Record both sides of every trade.

### Operating contract

The operator supplies principles; the agent aligns the solution to them. Every non-trivial
decision must cite the top-level principle(s) it serves, and `STATUS.md` / `DECISIONS.md`
record that alignment.

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

## Repository map & commands

Layers (one-way): `rabbit_hole` (data) → `looking_glass` (Layer B donor) → `plugins` →
`red_king` → `red_queen`, with `white_queen` (OPE certification) and `caterpillar`
(read-only interpretability) off `plugins`.

- **`looking_glass` = Layer B, the frozen customer-foundation donor.** One production
  stack: the `CFM` selective multi-scale SSM (`cfm_model.py`), trained self-supervised and
  frozen; plugins consume its `donor_embeddings` / `state_embeddings` tables read-only.
  There are **no factory functions and no alternate backends** (the old `create_*` /
  Mamba-2 `SequenceEngine` stack was deleted as dead code). Import submodules directly
  (`looking_glass.cfm_*`, `looking_glass.daily_states`, ...).
- **Run everything from the repo root** (`wonderland/`); the root `pyproject.toml` puts the
  package on `sys.path` (no install needed).
- Install deps: `python3 -m pip install -r requirements.txt`.
- Tests: `python3 -m pytest` (fast tier `-m "not slow"`); tests live in the repo-root `tests/`.
- Train + products: `python3 -m looking_glass.customer_foundation_model train --customers 25000
  --anchors 6 --as-of YYYY-MM-01 --db rabbit_hole/data/duckdb/customer_event_stream.parquet
  --out-dir looking_glass/artifacts/cfm`.
- Conventions: one responsibility per submodule, **no package-root re-exports** (import
  submodules); `scripts/` is not part of the package (it bootstraps the repo root onto `sys.path`).

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
- **Cockpit report card (PRIMARY steering artifact)** — `scripts/full_report_card.py`
  composed from real receipts (`foundation_report --receipts`, `objectives_receipts.py`,
  portfolio, minimality, comparators, transfer, known-effect, acceptance, schema suite).
  **The whole project is steered from the card:** every milestone/chunk ends by
  **printing the full card** (T0 + tier matrix + proof sections + caveats), and the
  card — not a hand summary — is the source of truth for what is PASS and what is next.
  Print it as **rendered Markdown** (never wrap the whole card in a fenced code block:
  that suppresses table rendering).

### Decision self-check (before non-trivial action)

Simplicity (standard-library instead of a layer?), Cost/Capability (cheapest
reliable tool?), Config (any parameter hardcoded in source?), Antifragile
(what happens on timeout/failure?), State (logged in DECISIONS.md, STATUS.md
updated?).

---

## Certification integrity (binding anti-self-deception protocol)

Every certification claim is a **pre-registered experiment**, not a narrative. The
following rules are binding: violating one invalidates the result regardless of the
numbers. They exist because the project twice produced a "SOLVED" label that a
subsequent adversarial review correctly overturned (DEC-228, DEC-234).

### 1. Freeze the gate before the result
- Gate definitions, thresholds, criteria, and selection rules are part of the protocol.
  Record them in `DECISIONS.md` and the receipt **before** running.
- A definition/criterion change after seeing a result is a **new protocol version**:
  stop, get explicit operator approval, re-run from scratch, and report both
  protocols. It may never be used to flip a verdict in the same cycle. The 0.11.0
  post-hoc "sufficiency" minimality criterion is the canonical violation (DEC-233/234).
- The same applies to the frontier: do not relabel an open item "closed" by changing
  what the item measures.

### 2. T0 is not SOLVED
- **T0** is the per-stream gate battery. **SOLVED** is T0 PASS **and** every frontier
  item CLOSED under the frozen definitions **and** the card all-green.
- Never print "SOLVED" unless the card's generated assessment says so from receipts.
  No hand-written "SOLVED" sentence and no post-result relabeling.

### 3. A receipt may not define its own criterion
- Receipts report measurements; gate criteria live in code (frozen). A receipt that
  introduces a new `criterion`/`rule` string is **UNKNOWN/PARTIAL** unless the
  criterion is pre-registered and the reader validates it.
- The objectives reader must reject `block_wise_certified=true` when the receipt's
  criterion is not the frozen reconstruction rule.

### 4. No circular evaluation
- The object under test must not contain the evaluation oracle's features. If E's shape
  includes deterministic summaries of the full-history measure, the claim is
  **"served-statistic sufficiency"**, not "learned-shape sufficiency"; report the
  learned-only attribution alongside.
- Report duplicated blocks in distance/value metrics (e.g. full-E = shape + measure
  when the shape already contains the measure).
- Required attribution controls: oracle independence, power controls (measure-only,
  shape-only, destroyed), **learned-only**, and for decisions a **random-donor**
  negative control.

### 5. Comparators must be finite and fair
- Every PASS needs a real comparator with the same head, a paired CI, and a finite
  baseline. `raw=nan`, `lo=hi`, or a hardcoded PASS is not evidence.
- Degenerate targets are excluded and reported, never called ties.
- OPE: the **stream gate** and the **broad acceptance contract** are separate
  results. Never substitute one silently for the other.

### 6. Reproducibility is part of the claim
- Every certified card must be regenerated by a **single repo command** at a clean,
  recorded commit, with artifact hashes and N recorded.
- A receipt under `/tmp` that no repo command regenerates is not evidence.
- Record N. If a run is OOM-killed at the target N, report the reduced N and why; do
  not compare across N without stating it.

### 7. The card must be current and self-consistent
- Regenerate the card after every behavior change. Hardcoded textual claims (e.g.
  "shape-only U≈0.26") are bugs.
- The card must not contradict its receipts (shape-U, minimality, full-E, calibration,
  OPE). Print tolerances and bands, not just nominal targets.
- The minimality row is the frozen reconstruction certificate. Serving a larger rank
  and relabeling it "minimal" is a violation.

### 8. Stop-and-ask triggers
Stop and ask the operator **before** running if a proposed change:
- changes a gate, threshold, criterion, or frontier definition;
- changes what "shape" contains or what "full-E" means;
- would move a verdict from FAIL to PASS by anything other than a real capability
  improvement;
- cannot be reproduced from the repo at a recorded commit.

### Certification checklist (print a card only after all are true)
1. Gate/criterion frozen **before** this run; no post-result edits.
2. T0 and SOLVED reported as separate claims; every frontier item has a measured status.
3. No circular evaluation: learned-only and power-control receipts present.
4. Every PASS has a finite, fair comparator with a CI; degenerate exclusions reported.
5. Stream OPE and broad acceptance shown separately.
6. One repo command reproduces the card at the recorded commit; hashes and N recorded.
7. The card has no stale/contradictory textual claims.
8. The minimality row is the frozen reconstruction certificate (served rank = k_min).

---

## Principle-first defaults (binding)

When two options both work, prefer the one whose **mechanism is learned or derived from the
stream** over a fixed literal. A literal is allowed only when:
(a) it is the **evaluation or causality contract** (those must stay fixed),
(b) a learned/data-derived version is unstable or unmeasurable at the current budget, or
(c) it is a **documented fallback** with a derivation receipt.

### Defaults

- **Time:** continuous-time gated SSM; `h_t = exp(-Δt/τ)·h_{t-1} + (1-exp(-Δt/τ))·x_t`;
  `τ` learned per channel, initialized from the stream’s gap/autocorrelation statistics.
- **Periodicity:** data-derived spectral peaks (autocorrelation/FFT); no fixed hour/day-of-week.
- **Action cadence/period:** inferred from action-event spacing; the declared value is a fallback.
- **Capacity:** width, depth, number of timescales, and vocab cap derived from data, validation,
  and the compute budget; no fixed `d/layers/experts`.
- **Vocab:** coverage-based cap (e.g. values covering ~99% of event mass) or hashing for
  high-cardinality fields; no fixed cap.
- **Numeric inputs:** learned monotone transform (quantile/spline) fit on train; signed `log1p`
  is the stable fallback.
- **Losses/masking/LR/patience:** learned task weights; mask rate from field missingness/entropy;
  LR measured; patience/tolerance from validation SE.
- **Readout:** all readouts available (last/mean/max/attention/magnitude); learned positive
  weights; no softmax gate that can collapse to one branch.
- **Horizons/cutoffs:** sampled from the data; no fixed 90d curriculum.
- **Serving:** block-aware scales derived from data (shape vs measure).

### Non-negotiable fixed things

- Evaluation targets, folds, baselines, gates.
- Causality: no future information, no `target_key`, no table/counter/summary in `E` for the
  learned-shape claim.
- The table baseline.

### Decision protocol

1. Name the fixed option and the learned/data-derived option.
2. Name the principle(s) served and the measurable benefit.
3. If a literal is needed for stability, record it as a **fallback** and show the derivation.
4. If the learned option cannot be evaluated cheaply or threatens robustness/cost, stop and ask.
5. Record the decision in `DECISIONS.md`.

### Pre-implementation principle check (binding)

Before writing code for a non-trivial change, answer these in the working notes or commit:

1. **Which principles does this touch?** (speed, adaptive, universal, lawful, honest, efficient,
   accessible, diagnosable).
2. **What are at least two options?** Include the existing primitive in the repo, not just a new
   implementation.
3. **Which option best satisfies the principles?** If the choice uses a fixed literal, a
   per-item Python loop, a new gate, or a new artifact, stop and justify it.
4. **For a hot path, time a smoke before scaling** and emit a runtime receipt.
5. **Record the choice and the benchmark** in `DECISIONS.md`.

If you cannot answer (2) from the repo, search first: `_scan`, `embed_batch_size`,
`derive_batch`, `generic_measure`, the generator, the existing tests.

### Hot paths: no per-item Python loops

Model, training, and serving hot paths must use scans, vectorized ops, token-budget batching, or
compiled kernels. A Python loop over `T` or rows is a red flag. Use the existing affine scan
(`looking_glass.cfm_model._scan`) or a vectorized equivalent; if a loop is unavoidable, record the
benchmark and the reason.

**Known primitives:** `_scan` (affine prefix scan), `embed_batch_size`, `derive_batch`,
`generic_measure`, `token-budget` batching, the existing SSM/scan paths.

### Principle audit (mechanized)

Run `python3 scripts/principle_audit.py` on changed files before committing. It flags:
- Python loops over time/rows in `looking_glass/` or `scripts/`,
- `86400`/`604800` and other cadence literals,
- fixed `dim`/`layers`/`epochs`/`seq_len` literals,
- new gate/criterion strings.

Use `--strict` in CI; a warning must be resolved or marked `# principle-ok: <reason>`.

---

## GPU-first execution (operator policy)

Use the **GPU for any hot path where a measured benchmark shows a material speedup**. Install the
CUDA-X / NVIDIA libraries needed (`cupy-cuda13x`, `cuda-python`, `nvidia-ml-py`, GPU-capable
`xgboost`, RAPIDS where it fits) rather than leaving performant work on the CPU.

- **Measure before adopting.** Record a CPU-vs-GPU benchmark (e.g. `scripts/bench_polars_gpu.py`) and
  keep the CPU fallback. A GPU path that is slower, numerically different beyond tolerance, or that
  regresses a gate is not adopted.
- **Respect the 8 GiB card.** One GPU job at a time; derive batches from VRAM/sequence length; use
  AMP/bf16 and compiled scans; never trade correctness for speed. cuPy/RAPIDS cannot allocate while
  a 7 GiB training job is resident.
- **Dependency conflicts are isolated, not ignored.** RAPIDS cu12 packages pin `cuda-bindings 12.9`,
  while torch cu13 needs `>=13.0.3`; install RAPIDS in a dedicated venv (as `cudf-polars` was) and
  use it through a process boundary. Do not downgrade the main env.
- **Record the runtime/GPU receipt** for every GPU-accelerated gate or benchmark.

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
- **Commit cadence (operator policy):** **every coherent chunk of work is committed** as soon as it
  is verified — code + docs only, never the artifacts above. Do not accumulate uncommitted chunks.
- **Push permission:** commits stay **local** until the operator explicitly asks to push. Pushing is
  the only git action that needs permission; `git add`/`git commit` do not.
- **Workflow:** make changes -> run the gates -> `git add` the intended code/docs -> `git commit`
  (one chunk, one commit) -> **wait for explicit permission to `git push`**. One repo, one history;
  do not create nested `.git` directories.
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

---

## Functional Realism v2.0 (project alignment — DEC-042)

System invariants are **algebraic construction inside the state update**, never
soft penalties, step tricks, or governor rescues. Each pillar is backed by a
measured receipt in this repo:

1. **Invariants by construction** — rank/energy live in `h_t=f(h_{t-1},x_t)`.
   Receipt: §4.3 spike — bilinear `h⊗x` raised *transferable* cross-sample rank
   1.79×; soft DWA penalties (§4.1 linear centering no-op; §4.2 gramian
   batch-fit manufactures) were **rejected by measurement**.
2. **Topological integrity (no silent rescues)** — divergence is a **hard
   failure**; reject the step, never roll back to fake a green. Receipt: DEC-039.
3. **Diagnostic honesty (no capacity faking)** — no batchnorm/ZCA dynamic-range
   manufacture; per-sample Cayley isometry `R` (κ=1). Receipt: §8.1/§4.2.
4. **Information conservation** — `R_cons = Tr(Σ_readout)/P_in ≈ 1` forbidden to
   discard stream entropy. Receipt: measured 0.90–1.05 across streams.

Cost doctrine: pay compute **once for a deterministically stable, self-verifying
system**, but build it fast (vectorized/scan-native; no per-step Python loops).
Non-linearity needed for rank is solved by **iterate-to-convergence scans**, not
sequential loops.
