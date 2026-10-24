# PLAN.md — wonderland execution prompt

> **Purpose:** this is the implementation brief for completing wonderland. Treat it as a
> prompt to an autonomous engineering agent, not as a description of work already done.
> Read `AGENTS.md` first. Where this plan and `AGENTS.md` disagree, `AGENTS.md` wins.
>
> **Current baseline:** the repository already has a synthetic 6k demo dataset, a frozen
> self-supervised encoder, supervised/unsupervised/RL measurements, and a generated
> `REPORT_CARD.md`. The baseline card is green for its declared Layer-0 scope. It is not
> yet proof that the learned 128-dimensional `E` alone is a universal sufficient statistic.
> Preserve that distinction throughout.

---

## 0. Mission

Build one system that accepts an event stream and emits a **frozen, versioned foundation
model artifact** that can serve as the common base for:

1. **Supervised heads** — value, orders, churn, margin, and future labels not known when the
   encoder is trained.
2. **Unsupervised heads** — retrieval, nearest-neighbor structure, clustering/segmentation,
   anomaly detection, and usable geometry.
3. **Reinforcement-learning heads** — action-conditioned response models and offline policy
   evaluation, using pre-action states and logged randomized actions.

The foundation model must be:

- **Per-stream:** one fitted artifact per stream; never apply weights to a stream they did
  not train on.
- **Universal as a pipeline:** the same code adapts to a new schema without business-specific
  edits.
- **Self-supervised upstream:** no downstream labels, targets, or task names shape Layer B.
- **Frozen downstream:** plugins read state; they never mutate the encoder.
- **Adaptive:** configuration, capacity, timescales, horizons, masking, and budgets are
  derived from data and hardware or learned, with documented fallbacks only when necessary.
- **Lawful:** use the correct algebra for each kind of information. Do not ask a neural
  network to rediscover exact addition or counting.
- **Accessible:** information must be recoverable by simple consumers, not merely present in
  an entangled hidden state.
- **Fast and bounded:** linear/scan-based sequence processing where possible, vectorized data
  preparation, predictable memory, measured runtime, and a CPU fallback.
- **Honest:** every claim has a comparator, uncertainty, provenance, and a reproducible command.

The goal is **not** to produce a single impressive number. The goal is a reusable foundation
interface whose components and limitations are measured consistently.

---

## 1. Non-negotiable operating rules

These rules override speed, convenience, and the desire to make a card green.

### 1.1 Principles and scope

Follow `AGENTS.md` completely, including its 16 engineering principles, the operator charter,
the anti-overfit clause, and the certification-integrity protocol.

The conflict order is:

```text
Robust → Universal → Adaptive → Lawful → Efficient → Honest
```

Lawful and robust choices outrank convenience.

### 1.2 Never change a gate to obtain a pass

- Freeze the evaluation protocol before a run.
- Do not weaken a threshold, comparator, split, seed, CI rule, or metric after seeing a result.
- A post-result criterion change is a new protocol version and requires explicit operator
  approval, a fresh run, and reporting of both protocols.
- Keep open research rows in `docs/AUDIT_CARD.md`; never hide them by editing the primary card.
- Do not call a result `SOLVED`, `MASTERED`, or `CERTIFIED` beyond the scope actually measured.

### 1.3 No downstream leakage

Layer B must not train on:

- `y_*` labels
- future business outcomes
- a downstream task list
- a hand-selected feature set chosen because it helps the demo
- policy rewards or deploy decisions

Known downstream use cases may **veto**, never **shape**, the encoder.

A model may use future events only to construct a self-supervised target at a causal cutoff.
Those events must never be present in the input at that cutoff.

### 1.4 One axis per iteration

Every experiment changes one meaningful axis:

- input/schema handling
- tokenizer
- dynamics
- state/measure interface
- objective family
- training regime
- serving/performance
- plugin interface
- evaluation infrastructure

Do not bundle a new architecture, a new objective, and a new capacity setting into one run.

### 1.5 Measure before adopting

Every proposed change gets:

1. a decision recorded before the run;
2. the principle(s) it serves;
3. at least two alternatives, including an existing repository primitive;
4. one acceptance metric and a comparator;
5. a small smoke test;
6. a full measurement only if the smoke is informative;
7. a runtime receipt;
8. accept or revert based on the pre-registered criterion.

A negative result is useful. Record it; do not keep re-running the same idea hoping for a
different seed.

### 1.6 Fast iteration is mandatory

Do not start with a large cumulative sweep.

Use this order:

```text
static checks → tiny smoke → small deterministic screen → one full confirmation → card
```

Use the small screen to eliminate bad ideas cheaply. Confirm only the best candidate at the
full demo scale. One GPU job at a time on the 8 GiB card. Do not launch overlapping runs that
write the same receipts.

### 1.7 Repository and artifacts

- Run from the repository root.
- Keep source, specs, and small receipts in git.
- Keep data, checkpoints, large arrays, virtual environments, and logs out of git.
- Do not create nested `.git` directories.
- Do not push unless the operator explicitly asks.
- Commit each coherent verified chunk locally.
- Update `STATUS.md` and append to `DECISIONS.md` at each meaningful phase transition.
- Use spaces and `ruff format`; run the principle audit on changed files.

---

## 2. Current system: what to keep and what to fix

### 2.1 Existing assets to preserve

The following are useful foundations, not disposable prototype debris:

- `rabbit_hole` canonical event stream and relational reference generator.
- `scripts/demo/stream_loader.py` for schema discovery, typed sequence arrays, and cached
  CPU/GPU preparation.
- `scripts/demo/auto_encoder.py` for the self-supervised SSM encoder and training loop.
- `looking_glass.cfm_model._scan` for the associative affine prefix scan.
- `looking_glass.autotune` for derived capacity and hardware-aware configuration primitives.
- `scripts/frozen.py` for train-once/frozen-cache evaluation.
- The supervised, unsupervised, and RL measurement scripts under `scripts/`.
- `scripts/report_card.py`, `scripts/audit_card.py`, and `scripts/run_demo.py`.
- The synthetic data generator and its known-truth receipt.
- Existing negative controls, construction checks, and held-out evaluation conventions.

### 2.2 Current baseline evidence

The baseline 6k run has approximately:

| Quantity | Current reading |
|---|---|
| Customers | 6,000 |
| Events | 3.36M |
| Window | 540 days |
| Learned `E` | 128 dimensions |
| Baseline published `BLK` | 2,137 dimensions |
| Published state | `E + BLK`, approximately 2,265 dimensions |

On the baseline holdout, the important attribution is:

| Target | Table | `E` only | Published state |
|---|---:|---:|---:|
| value90 | ~0.343 | ~0.285 | ~0.416 |
| orders90 | ~0.420 | ~0.377 | ~0.505 |
| churn90 | ~0.916 | ~0.926 | ~0.936 |

The published state is useful; the compact learned shape alone does not beat the table on
value/orders. The largest published blocks are per-timescale state/mass features and many
self-supervised forecast-head outputs.

Current known limitations include:

- category identity below the table;
- anomaly and regime segmentation below the table;
- next-event approximately tied with a naive baseline;
- RL policy value underpowered or not yet a valid deploy claim;
- churn approximately a ceiling tie;
- open A3/A6/O1 rows in the full audit.

These are engineering signals, not reasons to hide or relabel results.

### 2.3 Specific architectural issues to address

1. **Attribution ambiguity:** `E`, `BLK`, and `E+BLK` are used in different comparisons. Make
   the object under test explicit everywhere.
2. **Mass is not exact raw arithmetic:** the current mass is a learned, transformed,
   recency-weighted accumulation. Add a true typed measure state.
3. **Decay is not input-selective:** current retention depends on elapsed time and learned
   fixed `τ`, not on event content.
4. **Business-specific roles:** identity and relational fields are selected by names in the
   loader. Infer roles generically.
5. **Action prediction:** next-event training includes company-driven send events. Separate
   exogenous actions from customer dynamics.
6. **Accessibility gaps:** category identity, anomaly, segmentation, and next-event need
   explicit measurement and, where appropriate, direct state support.
7. **Performance:** the scan has `O(T log T)` total work; target construction contains Python
   loops; full-history forwards are repeated; padding and memory need profiling.
8. **Robustness:** missing, unknown, and observed-zero values need separate treatment;
   non-finite values must not be silently sanitized into a pass.
9. **Configuration:** some defaults remain fixed (`d`, layers, epochs, patience, bounds).
   Replace them with derived/learned settings or documented, receipted fallbacks.

---

## 3. Target architecture

Implement the following as one coherent production path.

### 3.1 `StreamAdapter`

Responsibilities:

- accept the canonical event stream directly;
- validate required columns and timestamp ordering;
- discover attribute fields and types;
- infer semantic roles from the manifest and training data;
- identify exogenous actions from declared metadata, not business-name conditionals;
- distinguish missing, unknown, and observed-zero values;
- create deterministic per-entity sequence arrays;
- resolve vocabulary, coverage, normalization, timescales, horizons, and capacity from the
  training portion of the stream;
- emit a complete schema/config receipt.

Suggested module:

```text
looking_glass/stream_adapter.py
```

Suggested data structures:

```python
@dataclass(frozen=True)
class SchemaSpec:
    fields: tuple[FieldSpec, ...]
    event_types: tuple[str, ...]
    exogenous_event_types: tuple[str, ...]
    action_fields: tuple[str, ...]
    categorical_sizes: tuple[int, ...]
    numeric_fields: tuple[str, ...]
    periods: tuple[float, ...]
    taus: tuple[float, ...]
    horizons: tuple[float, ...]
    config_hash: str
```

The wide table is an **evaluation baseline**, not an encoder-training dependency.

### 3.2 `FrozenFoundation`

Responsibilities:

- fit one model per stream;
- freeze the learned encoder and all state-build configuration;
- transform events up to an `as_of` boundary without future access;
- return named state views;
- save/load with schema map, seed, data hash, code version, and runtime metadata.

Suggested public interface:

```python
artifact = FrozenFoundation.fit(stream, config=AutoConfig())
state = artifact.transform(events, as_of=cutoff)
```

The artifact is more than a weight file: it includes the deterministic state builder and
its schema/configuration. That is necessary for reproducible serving.

### 3.3 Published state views

Use explicit names rather than an opaque `E + BLK` convention:

```text
Z_shape       learned sequence representation
Z_measure     lawful typed measure state
Z_forecast    generic self-supervised forecast features, if retained
Z_uncertainty missingness, unknown, support, and calibration features
Z_full        concatenation of the above, with a documented layout
```

Every view must have a fixed dimension, a version, and a declared consumer contract.

Do not claim that `Z_full` is a minimal sufficient statistic. Minimality is a separate Layer-1
research question.

### 3.4 Lawful measure state

Create a dedicated module, for example:

```text
looking_glass/measure_state.py
```

For every discovered field, compute only statistics appropriate to its inferred type:

- exact event/entity counts;
- nonzero counts;
- raw numeric sums;
- sums of squares where needed for variance;
- means and standard deviations derived from exact accumulators;
- min/max with explicit empty-history handling;
- first/last timestamps and recency;
- categorical value counts and normalized distributions;
- explicit missing and other/unknown buckets;
- recency-weighted variants only when their decay is data-derived and documented.

Accumulate raw values first. Apply normalization only at readout.

Use vectorized segment reductions, grouped `bincount`, associative scans, Arrow/Polars bulk
operations, or tensorized operations. Do not write a per-row Python loop in the hot path.

For an incremental/mergeable statistic, test the actual law:

```text
measure(history_a + history_b) = merge(measure(history_a), measure(history_b))
```

For normalized distributions, merge the raw counts and normalize after merging.

Keep the measure state separate from the learned shape so that attribution and circularity
controls remain possible.

### 3.5 Selective sequence dynamics

Retain the associative scan, but allow event-conditioned retention:

```text
τ_t = positive_function(x_t)
a_t = exp(-Δt / τ_t)
h_t = a_t h_{t-1} + (1-a_t) x_t
```

Requirements:

- positive, finite, numerically stable timescales;
- initialization from observed gap/span statistics;
- no unit-dependent constants presented as universal laws;
- preserve exact composition through the scan;
- no silent clamp that hides divergence;
- measure whether event-conditioned retention improves generic probes, not just one label.

Keep exact extensive measure outside this learned recurrence. A learned decayed projection is
not a substitute for exact additive statistics.

### 3.6 Self-supervised objective family

Use objectives generated from the schema and causal cutoffs:

1. next customer-driven event;
2. masked event/field reconstruction;
3. past-state reconstruction;
4. future aggregates over data-derived horizons;
5. prefix-to-future-state prediction;
6. missingness and support estimation;
7. optional query-conditioned objectives after the core state is sound.

Do not include company action prediction in the customer-dynamics objective. Actions may be
used as exogenous covariates or in downstream response heads.

Learn loss weights if useful, but do not let a weight collapse erase an entire content axis.
Select checkpoints using held-out self-supervised and accessibility evidence, never the
training objective alone and never a downstream demo label.

---

## 4. Plugin contracts

### 4.1 Supervised heads

Use one consistent head and split per comparison. Evaluate:

- table;
- shape only;
- measure only;
- shape + measure;
- full published state;
- table + full state where useful.

Report paired bootstrap confidence intervals for margins, not only point scores.

The encoder is frozen before labels are touched. Any supervised head may be small and
task-specific; personalization belongs downstream.

### 4.2 Unsupervised heads

Measure distinct capabilities:

- nearest-neighbor outcome/identity retrieval;
- clustering or segmentation against known synthetic latents;
- anomaly detection against a truth-referenced rare outcome;
- geometry/rank/isotropy and kNN usability.

Do not count two nearly identical purity rows as proof that the entire unsupervised family is
solved. Report weak probes honestly.

### 4.3 RL heads

Keep the donor action-free. A response plugin receives:

```text
pre-action frozen state + action + optional action metadata
```

It predicts post-action reward. Use the randomized design and logged propensity for IPS/DR.

Required controls:

- random policy;
- table-state response model;
- random-donor response model;
- grouped customer bootstrap;
- temporal/pre-action leakage check.

Known-effect recovery is not the same as IPS policy value. Report both separately.

For a meaningful RL demo, add a **new data revision** with explicit decision episodes:

```text
decision timestamp
pre-action history
randomized action
logged propensity
post-action reward
persistent control where applicable
```

Do not overwrite the existing data revision or retroactively change the meaning of an old
receipt.

---

## 5. Measurement and certification protocol

### 5.1 Fixed comparators

For every important comparison, use the same:

- customer split;
- evaluation holdout;
- folds;
- head class and hyperparameters;
- seeds;
- CI/bootstrap procedure;
- data revision;
- state version.

Do not compare a 2k result to a 6k result without stating the scale.

### 5.2 Required arms

At minimum, report:

| Arm | Question answered |
|---|---|
| table | What does the traditional baseline do? |
| fair table | What happens without fixed-window/hidden engineering? |
| shape | What does the learned encoder carry? |
| measure | What do lawful stream statistics carry? |
| shape + measure | Does the proposed foundation state work? |
| forecast/full state | Do derived forecast features add value? |
| random state | Is the result just dimensionality/noise? |
| destroyed state | Is customer-specific information required? |

A pass needs a finite comparator and a paired CI. Degenerate targets must be excluded and
reported, not called ties or passes.

### 5.3 B versus B+C

**B — foundation model:**

- construction laws;
- determinism and reproducibility;
- schema adaptation;
- donor comparisons;
- shape/measure/forecast attribution;
- accessibility and geometry;
- negative controls;
- self-supervised objectives.

**B+C — foundation model plus plugins:**

- at least two supervised targets;
- at least two genuinely different unsupervised capabilities;
- at least two RL/action evaluations with a valid randomized design;
- paired uncertainty and comparators for every claimed win.

**Layer 1 — separate research:**

- minimality;
- worst-case universality `U`;
- decision-sufficiency and deploy-grade OPE;
- calibration/coverage guarantees.

Do not use Layer-0 results to claim Layer-1 completion.

### 5.4 Negative and leakage controls

Include:

- random state;
- row-shuffled state;
- dimension-shuffled state;
- event-order-shuffled state;
- table invariant to event order;
- null/no-signal stream;
- schema-renamed/new-field variant;
- train-only versus all-data vocabulary/normalization check;
- non-finite output check;
- pre-action timestamp check for RL.

### 5.5 Report cards

`REPORT_CARD.md` at repository root is the primary B+B+C card.

`docs/AUDIT_CARD.md` is the complete research audit and must retain all open rows.

The card generator must:

- resolve every number to a receipt;
- identify the state arm used by each row;
- show data revision, N, seed, encoder config, and runtime;
- distinguish point ties from statistically supported wins;
- show open limitations;
- avoid hardcoded claims that can become stale.

---

## 6. Performance and iteration plan

### 6.1 Profile before optimizing

Record wall time and memory for:

1. stream reading;
2. JSON/attribute expansion;
3. vocabulary and normalization resolution;
4. sequence collation;
5. target construction;
6. forward scan;
7. backward pass;
8. validation;
9. state generation;
10. plugin measurement.

The biggest measured bottleneck gets fixed first.

### 6.2 Likely optimizations

- vectorized segment reductions instead of per-customer statistics loops;
- precompute targets once per cut, not repeatedly;
- avoid the second full-history forward where mathematically unnecessary;
- pack variable-length sequences by measured token budget;
- keep scan shapes stable to avoid repeated compilation;
- use the existing frozen cache for all measurements;
- use a small deterministic screen before full training;
- benchmark CPU versus GPU and retain the faster correct path;
- use AMP only after numerical and determinism tests;
- reject non-finite output rather than calling `nan_to_num` silently.

### 6.3 Iteration budget discipline

Do not run an eight-configuration full-scale sweep to discover that the baseline is best.
Use a small screen to rank candidates, then confirm one candidate at full scale. If two
consecutive small-screen results are neutral or worse, stop that axis and record the negative
result.

Do not keep long-running exploratory processes alive while starting new ones. Before each run:

```text
check GPU/process ownership → kill stale owned jobs → run one job → save one receipt
```

---

## 7. File-by-file implementation map

### Data and schema

- `rabbit_hole/generators/relational_reference.py`
  - preserve the existing reference;
  - add a separately versioned decision-episode revision for RL;
  - emit action/propensity metadata and truth receipts.
- `looking_glass/stream_adapter.py`
  - new canonical direct-stream adapter and role inference.
- `scripts/demo/stream_loader.py`
  - either refactor to use the adapter or make it a thin compatibility wrapper;
  - remove hand-listed identity/relational roles;
  - separate missing/unknown/zero;
  - derive training-only configuration.

### Encoder and state

- `looking_glass/cfm_model.py`
  - reuse `_scan`; add only well-tested selective dynamics if accepted.
- `scripts/demo/auto_encoder.py`
  - keep the baseline path reproducible;
  - remove hardcoded action prediction from the customer objective;
  - make `E` and published state layout explicit;
  - avoid business-specific encoder logic.
- `looking_glass/measure_state.py`
  - new exact typed measure state with vectorized reductions and merge tests.
- `looking_glass/foundation.py`
  - new frozen artifact API and versioned save/load.
- `looking_glass/state_layout.py`
  - one source of truth for state-view names, slices, dimensions, and versions.

### Plugins and measurement

- `plugins/`
  - consume the frozen artifact through the public state interface.
- `scripts/measure_donor.py`
  - report shape/measure/full-state arms with the same head and split.
- `scripts/measure_controls.py`
  - preserve negative controls and add shape/measure/full-state attribution.
- `scripts/measure_identity.py`
  - report identity by field role and by state view, not a hand-selected business field only.
- `scripts/measure_demo_plugins.py`
  - keep supervised/unsupervised/RL measurements on the frozen state.
- `scripts/measure_response.py`
  - retain pre-action, grouped, propensity-aware RL evaluation.
- `scripts/evaluate_foundation_state.py`
  - use a lean attribution battery for fast screens, not a duplicate source of truth.

### Cards and demo

- `scripts/report_card.py`
  - primary root `REPORT_CARD.md`; explicit state arm for every row.
- `scripts/audit_card.py`
  - full research audit; retain all open research rows.
- `scripts/run_demo.py`
  - one reproducible command from data generation through receipts and cards.
- `scripts/demo_show.py`
  - terminal presentation from the generated card, with no invented numbers.
- `README.md`, `STATUS.md`, `DECISIONS.md`, `AGENTS.md`
  - update whenever the public architecture, commands, or evidence scope changes.

### Tests

Add focused tests in `tests/` for:

- schema discovery on renamed/new fields;
- no business-name dependency;
- missing/unknown/zero separation;
- causal cutoff correctness;
- measure merge/additivity and normalization order;
- exact empty-history behavior;
- non-finite rejection;
- scan composition;
- deterministic save/load and state layout;
- no labels in encoder training inputs;
- pre-action RL state construction;
- card receipt resolution and no missing silent rows.

---

## 8. Execution loop for the implementation agent

Repeat this loop until the definition of done in section 9 is satisfied.

### Step 1 — Restore context

Read, in this order:

```text
AGENTS.md
PLAN.md
STATUS.md
DECISIONS.md
REPORT_CARD.md
docs/AUDIT_CARD.md
```

Check the working tree, current commit, active processes, and GPU ownership.

### Step 2 — Select one subgoal

Choose the smallest unmet subgoal with the highest leverage. Do not select a new axis just
because it is interesting. Prefer the current measured bottleneck.

### Step 3 — Record the decision before running

Append to `DECISIONS.md`:

- context and observed evidence;
- the single axis being changed;
- principles served;
- at least two rejected alternatives;
- the exact acceptance metric and comparator;
- the data/split/seed/protocol;
- expected runtime and resource cost;
- what would cause a revert.

### Step 4 — Implement minimally

Change only the files required for the axis. Preserve a baseline fallback. Avoid new
abstractions unless they remove duplication or are required by the public contract.

### Step 5 — Run static checks and a smoke test

At minimum:

```bash
python3 -m py_compile <changed-python-files>
ruff check <changed-python-files>
ruff format --check <changed-python-files>
python3 scripts/principle_audit.py <changed-python-files>
```

Run focused unit tests before any expensive training.

### Step 6 — Screen cheaply

Use a small deterministic data slice and the lean battery. Compare against the existing
baseline on the same split. Do not launch a full sweep.

### Step 7 — Confirm once if warranted

If the screen meets the pre-registered criterion, run the full 6k measurement and record
runtime, N, seed, data revision, and state config. Otherwise revert or keep the change behind
an off-by-default experimental flag and record why it is not adopted.

### Step 8 — Regenerate both cards

Run the real card generators. Do not hand-edit numbers or verdicts.

Inspect:

- B and B+C separately;
- shape versus measure versus full state;
- every FAIL and NOT MEASURED row;
- all disclosures and open audit items.

### Step 9 — Commit and update state

If accepted, run the relevant tests, commit the coherent chunk, update `STATUS.md`, and append
the outcome to `DECISIONS.md`. If rejected, revert the code and retain the negative receipt and
decision record.

Do not push without explicit operator permission.

---

## 9. Definition of done

The project goal is achieved only when all of the following are true.

### 9.1 Product behavior

- A canonical event stream can be supplied without a hand-built wide table for encoder fitting.
- The pipeline produces one frozen, versioned artifact per stream.
- The artifact can transform new events up to an `as_of` boundary without future leakage.
- The artifact exposes documented shape, measure, uncertainty, and full-state views.
- Supervised, unsupervised, and RL consumers can all use the same frozen artifact.
- A renamed/new-field schema works without business-specific code edits.

### 9.2 Evidence

- The root `REPORT_CARD.md` is generated from current receipts and shows all required B and
  B+C rows passing under the pre-registered protocol.
- Every claimed win has a finite comparator and a paired CI.
- `E`/shape, measure, forecast, and full-state results are not conflated.
- Random, destroyed, null, schema-variant, and leakage controls are present and reported.
- RL results use pre-action states and a valid randomized design; known-effect recovery and
  IPS policy value are not conflated.
- The full audit remains visible, including any open Layer-1 rows.

### 9.3 Engineering quality

- No non-finite output is silently sanitized into a pass.
- Deterministic reproduction at a recorded commit works from the documented command.
- Runtime and memory receipts exist for training and serving.
- The CPU fallback works; GPU use is benchmarked.
- No per-row Python hot-path loops remain where a vectorized reduction is practical.
- Focused tests pass; `ruff check`, `ruff format --check`, and the principle audit pass on
  changed files.
- Documentation and `STATUS.md` describe the actual current state.

### 9.4 Explicitly not required to fake

- A compact `E` need not beat every engineered table if the declared product is a richer
  frozen state. The shape-only result must still be reported honestly.
- Layer-1 minimality, universal `U`, and deploy-grade causal certification are not declared
  complete unless separately measured and green.
- A demo is not allowed to hide a weak anomaly, segmentation, identity, next-event, or RL
  result merely because another plugin passes.

---

## 10. Recommended first three loops

### Loop 1 — State contract and attribution

Implement `state_layout.py` and the attribution battery. Keep the baseline model unchanged.
Produce shape/measure/full-state comparisons and make the card name the arm used by every row.

**Accept when:** the same baseline can be reproduced, every row identifies its state view, and
no result is silently changed by the new reporting.

### Loop 2 — Exact typed measure state

Implement vectorized `measure_state.py` with training-derived vocabulary and merge tests.
Evaluate shape-only, measure-only, and shape+measure against the existing baseline.

**Accept when:** the measure state is lawfully correct, accessible, and improves at least one
measured weakness without materially regressing the published-state donor results. Otherwise
record the negative result and do not claim it as an improvement.

### Loop 3 — Selective retention and generic action/schema handling

As separate sub-loops, make retention input-conditioned and remove hand-listed identity/
relational roles plus company-action prediction from the customer objective.

**Accept each independently:** generic self-supervised/accessibility metrics improve or remain
neutral with no robustness regression, and all construction/leakage tests pass.

Only after these loops should the agent consider query-conditioned objectives, capacity search,
or a broader RL data revision.

---

## 11. What not to do

- Do not begin by increasing `d`, layers, or epochs.
- Do not run another long cumulative architecture sweep.
- Do not add another prediction head to solve an accessibility problem that requires a state
  interface change.
- Do not train Layer B on downstream labels or tune it to the report card.
- Do not hardcode `segment`, `product_id`, `channel`, `offer`, or any other business field as a
  universal architectural assumption.
- Do not call learned, transformed, decay-weighted mass “exact arithmetic.”
- Do not call `E + BLK` the same object as `E` in comparisons.
- Do not silently map unknown, missing, and observed-zero values together.
- Do not use `nan_to_num` to hide a divergent model.
- Do not edit a gate after seeing a result.
- Do not delete the audit card or its open rows.
- Do not leave overlapping GPU jobs or competing receipt writers running.
- Do not push commits unless the operator explicitly asks.

---

## 12. Final delivery narrative

When the work is complete, the demo should be explainable in five statements:

1. **Input:** here is a real event stream, with a schema discovered from the stream itself.
2. **Foundation:** here is the frozen per-stream artifact and its learned/lawful state views.
3. **Reuse:** the same frozen state drives supervised, unsupervised, and RL heads without
   retraining the foundation for each task.
4. **Evidence:** here are the comparators, confidence intervals, negative controls, and runtime
   receipts in `REPORT_CARD.md`.
5. **Honesty:** here is exactly what is not yet certified in `docs/AUDIT_CARD.md`.

The demo is complete when those statements are true from a clean reproducible run, not when a
single table of green labels has been produced.

---

## 13. Immediate agent instruction

Start with **Loop 1**, not another architecture sweep.

Do not ask for permission to make ordinary local code changes. Stop and ask only when a
proposed action would change a frozen gate/criterion, alter the meaning of the evaluation
protocol, require a new data revision that invalidates prior receipts, or require pushing to a
remote repository.

At every yield, leave:

- a current `STATUS.md`;
- an appended `DECISIONS.md`;
- a generated card whose numbers match its receipts;
- no unexplained running jobs;
- a clear statement of the next single axis to execute.
