# PLAN.md — wonderland: forward plan from the current baseline

> **Purpose:** a complete, principle-aligned plan for reaching the project goal from where the
> repository is now. It records the current evidence, the design alternatives considered, why
> they were accepted or rejected, and the remaining work in execution order.
>
> **Read before acting:** `AGENTS.md`, this file, `STATUS.md`, `DECISIONS.md`,
> `REPORT_CARD.md`, and `docs/AUDIT_CARD.md`. `AGENTS.md` always wins over this plan.
>
> **Checkpoint:** `971eeb2` on `primary-encoder`.

---

## 1. Mission and definition of done

Build one system that accepts an event stream and emits a **frozen, versioned foundation-model
artifact per stream**, reusable as the base for:

1. **Supervised learning** — value, orders, churn, margin, and labels unknown during encoder
   training.
2. **Unsupervised learning** — retrieval, nearest-neighbor structure, segmentation, anomaly
   detection, and usable geometry.
3. **Reinforcement learning** — action-conditioned response models and offline evaluation using
   pre-action state and logged randomized actions.

The pipeline must be:

- **Universal/generic:** one architecture adapts to new schemas without business-specific edits.
- **Per-stream:** one fitted artifact per stream; never apply weights to an untrained stream.
- **Self-supervised upstream:** no downstream labels or task-specific feature engineering in
  Layer B.
- **Frozen downstream:** plugins read state and never mutate the foundation.
- **Dynamic/adaptive:** configuration, timescales, horizons, masking, capacity, and budgets are
  learned or derived from data and hardware, with documented fallbacks only when needed.
- **Lawful:** exact algebra for exact quantities; learned representations for pattern and shape.
- **Accessible:** simple consumers can recover the information they need from named state views.
- **Efficient/fast:** vectorized preparation and sequence operations, bounded memory, measured
  runtime, and a working CPU fallback.
- **Honest/robust:** finite output, explicit uncertainty, reproducible receipts, fair comparators,
  and no gate changes made to obtain a pass.

### The workflow to complete

```text
raw canonical event stream
    ↓
schema discovery + causal sequence preparation
    ↓
self-supervised training
    ↓
frozen foundation artifact
    ↓
named state views
    ↓
supervised / unsupervised / RL plugins
    ↓
receipts → REPORT_CARD.md + docs/AUDIT_CARD.md
```

### Completion criteria

The product goal is met when all are true:

- A raw canonical stream can be fitted and transformed without a hand-built wide table.
- One deterministic, self-contained, versioned artifact exists per stream.
- The same artifact supports all three plugin families without retraining the foundation.
- Renamed and additional fields work without business-specific edits.
- State views and their dimensions are explicit and stable.
- B and B+C pass under the pre-registered protocol, or remaining failures are openly reported.
- Every claimed win has a fair comparator and paired uncertainty.
- There is no silent leakage, non-finite sanitization, or unidentifiable receipt.
- Training and serving have runtime and memory receipts and a working CPU fallback.
- The full audit remains visible, including open research questions.

Layer-1 minimality, worst-case universal `U`, and deploy-grade decision-sufficiency are separate
research goals and must not be implied by a green Layer-0 card.

---

## 2. Current baseline: preserve these assets

The following work is complete or already measured. Do not restart it without a new hypothesis.

### 2.1 Foundation and state contract

- `looking_glass/state_layout.py`
  - single source of truth for block order and named views;
  - views assembled by block name, with every `BLK` column used exactly once.
- `looking_glass/foundation.py`
  - `FrozenFoundation` with `fit`, `state`, `transform_events`, `describe`, `save`, `load`,
    content hash, and provenance;
  - schema/config, layout, and measure vocabulary persisted;
  - deterministic round-trip on the real cache;
  - refuses to emit state with missing `as_of` or non-finite output.
- `looking_glass/measure_state.py`
  - lawful typed measure state over all discovered fields;
  - exact accumulation before normalization;
  - bounded, data-derived vocabulary;
  - exact additivity of raw additive blocks, unit-tested.
- `looking_glass/stream_adapter.py`
  - frozen-schema encoding of raw canonical events.
- `scripts/frozen.py`
  - train-once cache and run manifest;
  - `assert_axes` prevents mixing artifacts built with different configurations.
- `scripts/select_views.py`
  - candidates declared up front, selected on training folds only, holdout evaluated once.
- `scripts/measure_response.py`
  - pre-action response evaluation with a valid uniform-policy comparator and guarded intervals.
- `scripts/measure_runtime.py`
  - cold and warm serving costs measured separately.
- Tests for measure algebra, schema roles, artifact lifecycle, stream encoding, run-manifest
  identity, response bootstrap validity, and fail-safe frozen loading.
- `REPORT_CARD.md` at repository root and `docs/AUDIT_CARD.md`.

### 2.2 Current evidence

The primary card reports:

```text
B:   28 PASS / 0 FAIL
B+C:  6 PASS / 0 FAIL
Total: 34 PASS / 0 FAIL
```

The full audit remains:

```text
26 PASS / 3 FAIL
```

The open research rows include A3 category identity, A6 policy value, and the strict next-event
point gate. Churn is a ceiling tie in the floor/ceiling analysis.

This is evidence for the declared Layer-0 scope and published state. It is not proof that the
learned 128-dimensional `E` alone is a universal sufficient statistic.

### 2.3 Measured multi-view behavior

The attribution battery shows complementary information rather than one universally best vector:

| View | Main measured strength |
|---|---|
| `shape` | Compact behavioral state; strong churn result |
| `learned_state` | Channel identity |
| `forecast` | Retrieval and churn-neighbor structure |
| `measure` | Category identity and best non-table anomaly view |
| `full` | Supervised value/orders |

Representative results:

- `shape` beats the table on churn (approximately 0.936 vs 0.911).
- `measure` nearly matches the table on category identity (approximately 0.261 vs 0.265).
- `full` reaches approximately 0.433 on value90 versus approximately 0.372 for the table.
- `shape` alone trails the table on value/orders.
- Naively concatenating every view can dilute raw-distance geometry and identity.

### 2.4 Lifecycle correctness already measured

- The frozen-schema adapter reproduces training sequences exactly for event types, categorical
  values, numeric transforms, and raw values; time features differ only by float32 rounding.
- Encoding the raw 6k stream reproduces the training `E` to about `4e-7` and `BLK` to about
  `1e-6`.
- Artifact save/load round-trips are bit-identical and the content hash is stable.
- Fitting works without `path_a/path_a_wide_table.parquet`; the wide table is only an evaluation
  baseline.
- Missing `as_of` or non-finite output is rejected instead of being silently converted into a
  valid-looking score.

### 2.5 Train-only view selection works

| Target | Selected view | Holdout selected | Table | Delta | 95% CI |
|---|---|---:|---:|---:|---|
| value90 | `full` | 0.4573 | 0.4240 | +0.0335 | [+0.0029, +0.0581] |
| orders90 | `full` | 0.5453 | 0.4938 | +0.0517 | [+0.0275, +0.0770] |
| churn90 | `forecast` | 0.9495 | 0.9452 | +0.0027 | [−0.0205, +0.0270] |

The training folds select different views per target. This supports the multi-view design and
avoids choosing the best-looking view on the test set.

### 2.6 RL evidence is valid but non-dominant

The response evaluator uses pre-action state, logged uniform propensity, customer-grouped
resampling, and a uniform-policy comparator:

- **Channel:** E beats uniform (`+1.92`, 95% CI `[+0.04, +3.95]`) but ties the table.
- **Offer:** E beats table (`+1.73`, 95% CI `[+0.40, +3.06]`) but ties uniform.

This is a working response interface, not a deploy-grade policy-value certificate.

### 2.7 Performance findings

- The earlier forward-pass “110 rows/s” figure was dominated by `torch.compile` warm-up.
- Warm state generation is approximately 1,900–2,000 rows/s; larger batches do not materially
  improve it.
- The meaningful measured costs are cold compilation and supervised head fitting.
- Length-sorted token-budget batching was slower (62.8 vs 110.3 rows/s) and was reverted.
- Further blind batching experiments are not justified.

### 2.8 Negative results to respect

| Experiment | Outcome | Rule going forward |
|---|---|---|
| `selective` retention | Some identity/anomaly gains but donor margins regressed at 6k | Do not retry unchanged; a new mechanism needs a new hypothesis |
| `noaction` action masking | Value/orders improved but churn add-over-sequence failed D3 | Do not adopt by weakening D3; diagnose or retain baseline |
| Long cumulative sweeps | Slow and mostly neutral or worse | Use focused small screens only |
| Larger model as first response | Not justified by the measured failures | Fix interface and accessibility first |

A rejected experiment is a recorded result, not a task to keep retrying without a new mechanism.

---

## 3. Remaining gaps

### 3.1 Public lifecycle is implemented but not yet fully unified

The cache-based path works, and fitting without a wide table has been verified. The remaining
work is to make one canonical production path:

- `FrozenFoundation.fit(...)` still needs to accept a raw canonical stream directly, not only a
  prepared bundle.
- `state(...)` still expects prepared `seqs`; `transform_events(...)` is the raw-event entry point.
- `scripts/demo/stream_loader.py` should become a compatibility wrapper around the adapter rather
  than a second schema implementation.
- Training and serving must use the same schema, vocabulary, missing/unknown rules, and
  configuration, with no rediscovery on a serving batch.

### 3.2 State views are not yet consistently consumed

- Some measurements still construct `E + BLK` manually.
- `full` needs one permanent, documented definition.
- `full_measure` must remain a separate named arm if evaluated.
- Consumers need a train-only view-selection protocol, not test-set selection.
- The card must name the view used by every row.

### 3.3 Encoder quality and adaptivity

- The frozen build still uses fixed `d=128`, two layers, and a fixed epoch budget in the default
  configuration.
- Default next-event training includes company-driven `email_send`; masking it was measured and
  rejected because it broke D3. The principle/metric tension is unresolved.
- Default retention is not input-conditioned; `selective` was measured and rejected at 6k.
- Relational roles remain declared rather than proven schema-derived.
- Missing, unknown, and observed-zero semantics need to be consistent across training and serving.
- Some legacy evaluation paths still sanitize non-finite values instead of rejecting them.

### 3.4 Performance

The runtime receipt identifies cold compilation and supervised head fitting as the real measured
costs. The scan has `O(T log T)` total work in its current Hillis–Steele form. Target construction,
padding, and duplicate full-history forwards also need profiling. No performance claim should be
made without before/after runtime, memory, and numerical-equivalence receipts.

### 3.5 RL evidence

Known-effect recovery on randomized synthetic action arms is valid but separate from policy-value
estimation. The repaired response evaluator is valid but non-dominant: channel beats uniform and
ties table; offer beats table and ties uniform.

Before any stronger policy-value claim:

- maintain arm alignment and independent controls;
- keep grouped bootstrap validity checks;
- use pre-action state and logged propensities;
- decide whether a new decision-episode data revision is necessary.

A new revision would invalidate prior receipts and must be introduced as a new version, not
silently substituted for the current demo data.

### 3.6 Research Layer 1

Minimality, worst-case universal `U`, and decision-sufficiency remain separate research goals.
They are not required to pretend Layer 0 is complete, and they are not claimed by the current card.

---

## 4. Non-negotiable principles for all future work

1. Follow `AGENTS.md` in full, including its conflict order and certification-integrity rules.
2. Change one meaningful axis per experiment.
3. Record context, alternatives, principle, acceptance metric, runtime, and revert condition before
   running.
4. Never change a gate, split, comparator, seed, or CI rule after seeing a result.
5. Never train Layer B on downstream labels or choose features because they help the demo.
6. Do not conflate `E`, measure, forecast, and full-state results.
7. Accumulate exact quantities exactly; normalize at readout.
8. Do not silently map missing, unknown, and observed-zero values into one ambiguous signal.
9. Reject non-finite output rather than calling `nan_to_num` to make it look valid.
10. Derive or learn configuration where practical; if a literal remains, document and receipt the
    fallback.
11. Use held-out convergence and evidence, not a fixed epoch count presented as convergence.
12. Use small deterministic screens before full-scale runs. No more long cumulative sweeps.
13. One GPU job and one receipt writer per run; no competing writers or stale processes.
14. Keep the CPU fallback and benchmark accepted GPU/AMP changes for speed and numerical agreement.
15. Commit coherent verified chunks locally. Do not push without explicit operator permission.
16. Preserve rejected results in `DECISIONS.md` so they are not repeated blindly.

---

## 5. Strategic choices and rationale

### 5.1 One opaque vector versus named state views

**Options:**

- **A. Ship only compact `E`.** Simpler API and closer to the original “foundation embedding”
  ambition, but current `E` loses to the table on value/orders and hides the measured value of
  the other blocks.
- **B. Concatenate everything into one large vector.** Simple consumer code, but measured
  dilution of geometry and identity and no way to explain which component carries the signal.
- **C. Publish named views with a stable layout.** Supports the measured multi-view reality,
  attribution, and task-specific consumption while keeping `full` available.

**Decision: C.** `shape`, `learned_state`, `forecast`, `measure`, `uncertainty`, `heads`, and
`full` are the product contract. This is aligned with Accessible and Diagnosable. It does not
require pretending that `E` alone is sufficient.

### 5.2 Learn every statistic versus compute exact statistics

**Options:**

- **A. Ask the network to reconstruct all counts and magnitudes.** Attractive as a pure learned
  model, but unnecessarily difficult and loses the exact additive law.
- **B. Use only hand-built aggregates.** Strong on this reference, but domain-specific and unable
  to represent sequence structure well.
- **C. Combine learned shape with lawful measure.** Learn patterns; compute counts, sums,
  distributions, and other exact statistics directly from the stream.

**Decision: C.** Exact arithmetic belongs in the measure state, not in a prediction head. Report
learned-only and measure-only results so the combination is not mistaken for learned-shape
sufficiency.

### 5.3 Always use `full` versus select a view per consumer

**Options:**

- **A. Always use `full`.** Convenient, but geometry and identity can degrade when irrelevant
  blocks dominate distances.
- **B. Pick the best view on the holdout.** Produces optimistic, leaky results.
- **C. Select on training folds only, then freeze and evaluate once.** Honest and compatible with
  the measured differences between targets.

**Decision: C.** A consumer may choose `full` if training evidence supports it, but it may not
choose a view because it looks best on the test set.

### 5.4 Mask company actions versus keep them in the next-event objective

**Options:**

- **A. Keep actions in the objective.** This is the current baseline and preserves the measured
  D3 result, but it means the model learns some company-driven transition structure.
- **B. Mask exogenous actions.** Better matches the exogenous-action doctrine, but the measured
  6k change improved value/orders while making churn add-over-sequence negative.
- **C. Separate customer and action dynamics structurally, with a new mechanism.** Potentially
  resolves the tension, but is unproven and must be tested as a new axis.

**Decision: retain A as the baseline and investigate C only with a new, justified mechanism.**
Do not simply rerun B and do not weaken D3. This is an open principle/metric tension, not a solved
problem.

### 5.5 Input-conditioned retention versus fixed learned timescales

**Options:**

- **A. Fixed learned `τ` per channel.** Current baseline; stable and measured.
- **B. Input-conditioned `τ(x_t)`.** More adaptive, but the first 6k implementation regressed donor
  margins.
- **C. A gated residual selective mechanism initialized to the baseline.** A possible future
  attempt, only if it has a specific reason to preserve donor margins.

**Decision: keep A; try C only with a new hypothesis and a small screen.** Do not claim that the
rejected selective approach is an improvement because a few secondary metrics improved.

### 5.6 Bigger encoder versus better interface

**Options:**

- **A. Increase `d`, layers, or epochs first.** More capacity and cost, but the measured failures
  are largely accessibility, attribution, and interface problems.
- **B. Improve state construction, schema handling, objectives, and consumer access first.**

**Decision: B.** Capacity changes come later, through a small nested, cost-aware search. Do not use
model size to hide a representation or evaluation problem.

### 5.7 Existing RL data versus a new decision-episode revision

**Options:**

- **A. Keep using the current reference.** Supports known-effect recovery and the repaired
  pre-action response evaluation; no need to disturb the demo.
- **B. Create a new decision-episode data revision.** Required if the goal is a defensible,
  deploy-style policy-value claim with clean episodes and adequate power, but it invalidates
  prior receipts and requires full re-certification.

**Decision: keep A for the current demo; use B only as a separately versioned RL workstream.**
Known-effect recovery and IPS policy value must never be conflated.

### 5.8 Batch optimization versus cold-start/head-fit optimization

**Options:**

- **A. More batching changes.** Already measured poorly; warm throughput is not batch-limited.
- **B. Reduce compile warm-up, stabilize compiled shapes, and reduce repeated work/head-fit cost.**

**Decision: B.** Measure cold and warm separately. Optimize only measured costs.

### 5.9 Layer-1 now versus after the product contract is stable

**Options:**

- **A. Start universal `U` and minimality claims now.** Risks mixing a research certificate with
  an unstable public interface.
- **B. Finish the lifecycle, consumers, and evidence first, then certify separately.**

**Decision: B.** Layer-1 is a separate research phase and is not implied by the current green card.

---

## 6. Forward roadmap

## Phase 0 — Evidence identity and baseline preservation

**Status:** mostly complete; harden and verify rather than redo.

Remaining tasks:

1. Keep the run manifest as the source of encoder configuration: data, revision, seed, axes,
   layout, dimensions, cache hash, commit, and runtime configuration.
2. Make every measurement writer consume that manifest and the same artifact.
3. Use run-specific receipts and atomic publication; prevent competing writers.
4. Assert cache axes, manifest axes, and card config agree.
5. Regenerate the primary and audit cards from that consistent run.
6. Keep a regression test for config/receipt mismatch.

**Why:** a previous `noaction` battery left config metadata that did not describe the baseline
receipts. That is a provenance failure, not a model result.

**Accept when:** every card row resolves to an identified artifact and no row depends on an
unidentified cache.

## Phase 1 — Complete the public stream-to-artifact lifecycle

**Purpose:** finish the actual product interface, not just the cache-based path.

Tasks:

1. Make `looking_glass/stream_adapter.py` the single canonical input path for both training and
   serving.
2. Accept canonical Arrow/Parquet/DataFrame streams directly.
3. Validate columns, timestamps, entity keys, and ordering.
4. Infer types and identity/static roles behaviorally, without business field names.
5. Use declared action metadata where available, with a documented structural fallback.
6. Separate missing, unknown, and observed-zero values.
7. Build causal per-entity sequences and derive configuration from training data only.
8. Make `scripts/demo/stream_loader.py` a compatibility wrapper rather than a second adapter.
9. Complete `FrozenFoundation.fit(...)` so raw streams can be fitted without a prepared bundle.
10. Complete `transform(events, as_of=...)` so callers never construct `seqs` manually.
11. Persist all schema, vocabulary, measure, layout, and provenance needed to serve in a fresh
    process.

Required tests:

- direct stream fitting and transformation without a wide table;
- renamed and additional fields;
- missing, unknown, and observed-zero semantics;
- empty stream and single-entity stream;
- long gaps and duplicate timestamps;
- causal cutoff correctness;
- save/load determinism;
- no labels in encoder training inputs;
- serving schema not rediscovered from a serving batch.

**Accept when:** a new canonical stream can be fitted, saved, loaded, and transformed with no
business-specific edits and no wide-table dependency for training.

## Phase 2 — Unify state consumption and view selection

**Purpose:** make the state contract useful to all consumers without leaking test information.

Tasks:

1. Keep `full` permanently defined in `state_layout.py` and document its composition.
2. Keep `full_measure` separate if it is evaluated.
3. Migrate remaining measurement and serving paths from hand-built `E + BLK` bundles to the
   artifact.
4. Use the train-only view-selection protocol for each consumer:
   - declare candidates before scoring;
   - select using training-fold CV only;
   - freeze the selection;
   - evaluate the holdout once with paired uncertainty.
5. Report shape-only, measure-only, full, and selected-view results separately.
6. Do not force measure into every consumer. Use it where training-only evidence supports it.
7. Add tests proving view selection does not use holdout labels.

**Accept when:** every B and B+C row names its state view and no view is selected because it looks
best on the test set.

## Phase 3 — Encoder quality and adaptive behavior

Handle each item as its own experiment; do not bundle them.

### 3A. Action/objective semantics

- Preserve declared exogenous-action metadata.
- Diagnose why masking actions improved value/orders but hurt churn.
- Try only a new, mechanistically justified solution that preserves the donor gate.
- Do not simply rerun `noaction` and do not weaken D3.
- If no acceptable solution is found, retain the baseline and document the tension.

### 3B. Selective retention

- Do not repeat the rejected implementation unchanged.
- A new attempt must explain why it should preserve donor margins while improving retention.
- Confirm at 6k after a small screen; a good 2k screen alone is not evidence of transfer.

### 3C. Relational roles

- Keep the declared relational configuration until a genuinely general rule is validated.
- Test proposed rules on schema variants and unrelated fields, not only on the reference pair.
- Do not invent a heuristic merely to reproduce the existing pair.

### 3D. Missing/unknown/zero and non-finite behavior

- Preserve separate missing, unseen, and observed-zero channels.
- Reject non-finite state before scoring.
- Replace legacy silent sanitization in production paths with explicit failure receipts.

### 3E. Adaptive capacity and convergence

- Replace fixed `d`, layers, and epoch budgets with derived settings or a documented, receipted
  fallback.
- Use a small nested held-out capacity search with a cost penalty, not a broad sweep.
- Use held-out convergence with a noise-derived tolerance.
- Never select weights by training loss or by the demo card.

## Phase 4 — Performance and scalability

Profile first, then change one measured bottleneck at a time.

Candidate work:

1. Vectorize target and measure construction.
2. Remove redundant full-history forwards where mathematically safe.
3. Reduce padding without creating many tiny batches.
4. Stabilize compiled scan shapes and reduce compile warm-up.
5. Cache deterministic preparation and targets.
6. Benchmark CPU/GPU and AMP only with numerical and determinism checks.
7. Measure head-fit cost separately from serving cost.

For every candidate record runtime, peak memory, numerical difference, and gate effect. Reject a
change that is slower or changes results beyond the declared tolerance.

## Phase 5 — RL evidence and, if needed, a new data revision

First keep the repaired evaluator correct:

1. Verify policy/logged-action/reward alignment.
2. Verify random-donor and uniform-policy controls are independent and well-defined.
3. Recompute the paired statistic correctly on every grouped bootstrap resample.
4. Reject impossible intervals, including zero-width CI with nonzero mean.
5. Use customer-grouped resampling and report IPS and DR separately.
6. Keep pre-action states and logged propensities.
7. Add tests for bootstrap validity and no-future-action leakage.

Only then decide whether a new decision-episode Layer-A revision is necessary. If so:

- give it a new version/name;
- pre-register the protocol;
- regenerate all affected receipts;
- compare against, rather than silently replace, the old data results.

Known-effect recovery and policy-value evaluation remain separate claims.

## Phase 6 — Final verification and demo

1. Generate the demo data through the documented command.
2. Build the frozen artifact once from the final code.
3. Generate all receipts from that exact artifact in one owned process.
4. Generate `REPORT_CARD.md` and `docs/AUDIT_CARD.md`.
5. Run focused tests, the fast suite, `ruff check`, `ruff format --check`, and the principle audit.
6. Run the full suite if the environment permits; disclose any incomplete run.
7. Present the terminal/Markdown demo with `scripts/demo_show.py`.
8. Update `README.md`, `STATUS.md`, `DECISIONS.md`, and this plan if the public contract changes.
9. Remove obsolete duplicate scripts/cards only after checking documented references.
10. Commit coherent chunks locally; do not push without explicit permission.

An interactive dashboard is optional and comes only after the terminal/Markdown demo is correct.

## Phase 7 — Layer-1 research, separately from product completion

If pursuing the research north star, schedule separately:

- minimality and served-rank certification;
- worst-case universal `U` against a full-history oracle;
- learned-only versus served-statistic attribution;
- decision-sufficiency and deploy-grade OPE;
- calibration and coverage guarantees.

Do not use a Layer-0 green card to imply these are complete.

---

## 7. Measurement protocol for every future loop

Unless a new protocol is explicitly approved, keep fixed:

- data revision;
- customer split and holdout;
- folds and seeds;
- head class and hyperparameters;
- CI/bootstrap method;
- state-view definition.

Where applicable compare:

- raw table;
- fair table;
- shape only;
- measure only;
- forecast/full state;
- shape + measure;
- selected view;
- random state;
- destroyed/shuffled state.

Every claimed win needs a finite comparator, paired CI, named state view, N, seed, data revision,
runtime, and a receipt that a repository command can regenerate.

Required controls where applicable:

- scan composition;
- field-permutation equivariance;
- determinism;
- schema variant;
- event-order shuffle;
- null/no-signal stream;
- non-finite rejection;
- pre-action RL timestamp check;
- no labels in encoder training.

Do not compare across different N without saying so. Do not treat a point tie as a statistically
supported win. Exclude and report degenerate targets rather than forcing a verdict.

---

## 8. Execution loop for the implementation agent

At every iteration:

1. **Restore context:** read `AGENTS.md`, this plan, `STATUS.md`, `DECISIONS.md`, and both cards.
   Check working tree, commit, running jobs, and GPU ownership.
2. **Choose one subgoal:** the smallest unmet item with the highest leverage, not the most
   interesting architecture.
3. **Record before running:** append context, axis, principles, alternatives, acceptance metric,
   protocol, expected runtime, and revert condition to `DECISIONS.md`.
4. **Implement minimally:** preserve the baseline fallback and avoid duplicate sources of truth.
5. **Run static checks:** compile, ruff, format check, principle audit, focused tests.
6. **Screen cheaply:** use a small deterministic slice and the lean battery. No full-scale sweep.
7. **Confirm once if warranted:** run the full measurement only if the screen supports it. Record
   N, seed, data revision, runtime, and state config.
8. **Regenerate both cards:** never hand-edit numbers or verdicts.
9. **Accept or revert:** apply the pre-registered criterion, record the outcome, and update
   `STATUS.md` and `DECISIONS.md`.
10. **Commit locally:** only after verification. Do not push without explicit operator permission.

Stop and ask the operator only for:

- a gate, threshold, criterion, split, or protocol change;
- a new data revision that invalidates prior receipts;
- a claim beyond the measured scope;
- a remote push.

---

## 9. File ownership for the remaining work

### Canonical public path

- `looking_glass/stream_adapter.py` — direct input, schema/role inference, causal preparation.
- `looking_glass/foundation.py` — fit/transform/save/load and artifact metadata.
- `looking_glass/state_layout.py` — state-view definitions and dimensions.
- `looking_glass/measure_state.py` — exact typed measure state.

### Compatibility and migration

- `scripts/demo/stream_loader.py` — compatibility wrapper only.
- `scripts/demo/auto_encoder.py` — baseline training path; no business-specific architecture.
- `scripts/frozen.py` — bridge from existing cache to artifact, not a second source of truth.
- `scripts/measure_*.py` — consume the same artifact and run manifest.
- `scripts/report_card.py` — primary card with config derived from the run manifest.
- `scripts/audit_card.py` — full audit and open research rows.

### Tests

Extend measure, role, and artifact tests with:

- direct stream lifecycle;
- schema variant;
- missing/unknown/zero semantics;
- causal cutoffs;
- run-manifest/receipt identity;
- train-only view selection;
- response bootstrap validity;
- numerical equivalence for accepted performance changes.

---

## 10. Definition of done for this plan

### Product

- A raw canonical event stream can be fitted and transformed without a wide table.
- One frozen, versioned artifact exists per stream.
- The artifact is reproducible, deterministic, and self-contained.
- Supervised, unsupervised, and RL consumers use the same frozen artifact.
- A renamed/new-field schema works without business-specific code edits.
- State views and dimensions are documented and stable.

### Evidence

- The primary card is generated from one consistent run manifest.
- B and B+C rows pass under the pre-registered protocol, or remaining failures are explicitly
  reported rather than hidden.
- Every claimed win has a comparator and paired CI.
- Shape, measure, forecast, and full-state results are not conflated.
- Known-effect RL and IPS policy value are not conflated.
- The audit card retains all open research rows.

### Engineering

- No silent non-finite sanitization remains in production/serving paths.
- Runtime and memory receipts exist for training and serving.
- The CPU fallback works; accepted GPU changes are benchmarked.
- Tests and static checks pass on changed files; incomplete full-suite runs are disclosed.
- No unexplained jobs or competing receipt writers remain.
- Documentation matches the actual public contract.

### Not implied by Layer-0 completion

- Layer-1 minimality;
- worst-case universal `U`;
- deploy-grade decision-sufficiency;
- a new decision-episode data revision, if required for valid policy-value evidence.

---

## 11. Immediate next three loops

### Loop A — Canonical lifecycle unification

Finish the direct raw-stream `fit`/`transform` path and make the loader a compatibility wrapper.
Do not retrain the baseline solely to prove it; use sequence-equivalence and artifact round-trip
tests first.

### Loop B — Artifact-consistent consumers

Move the remaining measurement writers onto the artifact and manifest. Keep train-only view
selection and make every card row name its state view.

### Loop C — One measured quality improvement

Choose the highest-leverage open item from measured evidence, not from a new architecture wish:

- if accessibility is the target, use the lawful measure/view interface;
- if action semantics is the target, diagnose the churn trade before changing the objective;
- if adaptivity is the target, derive capacity/convergence settings with a small nested screen;
- if RL is the target, repair/validate the evaluator before considering a new data revision.

Do not begin with another architecture sweep, another capacity increase, or a repetition of the
rejected `selective`, `noaction`, or batching experiments.

The immediate objective is to make the **public lifecycle, state contract, and evidence identity**
as reliable as the current measured baseline, then improve the encoder on top of that foundation.

---

# Accomplished — 2026-10-25

This section records what was actually delivered against the plan above. It is a status
report, not a new plan.

## Public lifecycle (Loop A) — done

- `looking_glass/stream_adapter.discover()` derives a complete schema from a **raw canonical
  event stream**: event vocabulary, categorical vocabularies and hashing, numeric and
  time-delta fields, behavioural identity roles, exogenous event types, Fourier periods,
  timescales, horizons, and masking rate.
- Sequences for training are built by calling the **same `transform()` used at serving**, so the
  two paths cannot drift apart.
- `FrozenFoundation.fit_stream(frame, as_of_ts)` fits an artifact directly from raw events.
  No prepared bundle and no wide table are required.
- `scripts/demo/stream_loader.load_data(engine=None)` now delegates to discovery; the GPU path
  remains the prepared-cache builder.
- `look+_glass/foundation.py` is importable without callers configuring `sys.path`.

Proof it is equivalent, not merely plausible (6k reference, compared against the GPU-built
prepared artifact):

| Array | Max difference |
|---|---|
| event types | 0.0 |
| categorical indices | 0.0 |
| numeric transform | 0.0 |
| gaps | 0.0 |
| raw values | 0.0 |
| time features | 6e-08 (float32) |
| timestamps | 2.4e-07 |

Keys, `et_v`, categorical sizes, periods, taus, and identity/relational/exogenous indices were
identical.

Four real bugs were found and fixed while verifying this:

1. int-coded categoricals gained a spurious extra vocabulary bucket;
2. periods were computed on naive timestamps instead of local time, which reordered the spectral
   peaks and swapped the Fourier columns;
3. `fit_stream` did not set `AS_OF_TS`, producing invalid gap features;
4. the validation split could consume the entire training set on a small stream.

## State contract (Loop B) — done

- `state_layout.FULL_ORDER` and `compose_full()` now define the `full` view permanently:
  `shape → learned_state → forecast → uncertainty → heads`.
- `FrozenFoundation.state()` uses that single definition instead of assembling `E + BLK` ad hoc.
- Tests assert the declared order, the declared dimensions, and that every `BLK` column is used
  exactly once across the views.

## Robustness (Loop C) — done

- `frozen.require_finite()` now guards the frozen state in the donor, controls, ceiling, battery,
  and attribution writers. Non-finite encoder output fails loudly instead of being converted to
  zeros.
- `nan_to_num` remains only for baseline feature tables, where NaN is a legitimate artefact of
  one-hot construction.
- Verified behaviour-neutral: `measure_donor` was re-run after the change and the receipt is
  identical in every reported metric.

## Evidence integrity — done

- Every cache writes `<cache>.manifest.json` (data, revision, seed, axes, layout, dimensions,
  measure vocabulary, field-role sources, best validation, git SHA, build time, cache SHA-256).
- `assert_axes()` refuses to mix a cache with a differently-configured process.
- `encoder_config.json` and the card header derive from that manifest, so a card row can no
  longer be labelled with a configuration that did not produce it.

## RL evidence — repaired

- The previous zero-width IPS interval was traced to a **degenerate control**: the shuffled-state
  model converged to the same policy, so the IPS difference was identically zero.
- The comparator is now the **uniform policy**, whose IPS value under uniform logging is exactly
  mean reward.
- `ci_or_fail()` rejects zero-width intervals with a nonzero mean and any non-finite interval.
- Result: channel beats uniform (+1.92, 95% CI [+0.04, +3.95]) and ties the table; offer beats
  the table (+1.73, +0.40 to +3.06) and ties uniform. Reported as a working response interface,
  not a deploy certificate.

## Performance — corrected

- The earlier “~110 rows/s” forward figure was `torch.compile` warm-up, not steady state. Warm
  state generation is about **1,900 rows/s**; batch size is not the lever.
- Length-sorted token-budget batching was measured at 62.8 rows/s versus 110.3 and reverted.
- Cold compilation and supervised head fitting are the real measured costs.

## Current evidence

```text
REPORT_CARD.md      B 28/28 PASS · B+C 6/6 PASS · 34 PASS / 0 FAIL
docs/AUDIT_CARD.md  26 PASS / 3 FAIL (open research rows preserved)
```

Measured highlights:

- `shape` beats the table on churn (~0.936 vs 0.911).
- `measure` nearly matches the table on category identity (~0.261 vs 0.265) and is the best
  non-table anomaly view.
- Train-only view selection picks `full` for value/orders and `forecast` for churn; all three
  holdout comparisons are reported with paired CIs, and churn is honestly a tie.
- Fitting no longer requires the wide table; the wide table is only an evaluation baseline.

Validation: 33 focused tests pass, the fast suite passes, `ruff check`/`ruff format --check` are
clean on changed files, and the principle audit reports no findings on the new modules.

## Still open (unchanged by this work)

- **3A** action-objective semantics: masking `email_send` improved value/orders but made churn
  add-over-sequence negative, failing D3. Retained the baseline; the tension is recorded, not
  hidden.
- **3B** selective retention: rejected at 6k because donor margins regressed. A new attempt
  needs a new hypothesis.
- **3D/3E** missing/unknown/zero channel separation, and derived capacity/convergence.
- **Relational roles** remain a declared configuration; no principled rule reproduces the current
  pair.
- **Phase 5** a decision-episode data revision, if a stronger policy-value claim is wanted.
- **Phase 7** Layer-1 minimality, universal `U`, and decision-sufficiency are not started and are
  not implied by the green Layer-0 card.

## Commits

```text
0bd8d29 loopA: canonical discovery, raw-stream fit_stream, loader delegates on CPU path
384bdb0 loopB/C: pin full-view composition; remove silent state sanitization
396f579 card: regenerate at current commit
```

Local only; not pushed (awaiting explicit permission per `AGENTS.md`).
