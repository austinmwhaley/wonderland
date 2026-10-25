# PLAN.md — wonderland: completion and hardening plan

> **Status:** rewritten after the Layer-0 implementation and measurement loops.
>
> This is the complete forward plan for the project, not a reaction to one experiment or one
> metric. It covers the public product contract, encoder quality, generic/adaptive behavior,
> lawful state construction, supervised/unsupervised/RL consumers, performance, evidence,
> and the demo.
>
> Read `AGENTS.md` first. It always overrides this document. Also read `STATUS.md`,
> `DECISIONS.md`, `REPORT_CARD.md`, and `docs/AUDIT_CARD.md` before changing anything.

## Phase status (as executed)

| Phase | State |
|---|---|
| 0 — Evidence identity (`frozen.py` run manifest, provenance guard) | **DONE** |
| 1 — Direct stream → fit/transform artifact (`stream_adapter.py`, `FrozenFoundation`) | **DONE** |
| 2 — Train-only view selection (`scripts/select_views.py`) | **DONE** |
| 3 — Encoder quality/adaptivity | **PARTIAL**: 3C fail-safe load and 3D receipted config fallback done; 3A/3B measured and **rejected** at 6k, recorded as open quality items |
| 4 — Performance | **DONE (measurement corrected)**: forward was cold-compile-bound, not steady-state |
| 5 — RL evidence | **DONE**: degenerate control replaced by the uniform policy; intervals guarded |
| 6 — Final verification/demo | **DONE**: card B 28/28, B+C 6/6; audit 26/3 preserved |
| 7 — Layer-1 research (minimality, `U`, decision-sufficiency) | **NOT STARTED** — separate research |

Checkpoint commit: `e0196e0`.

---

## 1. Mission and success definition

Build one system that accepts an event stream and emits a **frozen, versioned foundation-model
artifact per stream**. The artifact must provide a reusable state for:

1. **Supervised learning** — value, orders, churn, margin, and labels not known during encoder
   training.
2. **Unsupervised learning** — retrieval, nearest-neighbor structure, segmentation, anomaly
   detection, and usable geometry.
3. **Reinforcement learning** — action-conditioned response models and offline evaluation using
   pre-action state and logged randomized actions.

The required properties are:

- **Universal as a pipeline:** one architecture adapts to any supported event-stream schema.
- **Per-stream fitting:** one artifact per stream; never apply weights to an untrained stream.
- **Self-supervised upstream:** no downstream labels or task-specific feature engineering in
  Layer B.
- **Frozen downstream:** plugins read state and never mutate the foundation.
- **Dynamic/adaptive:** configuration, timescales, horizons, masking, capacity, and budgets are
  learned or derived from data and hardware, with documented fallbacks only when necessary.
- **Generic:** no business-specific field names or per-domain special cases in the architecture.
- **Lawful:** exact algebra for exact quantities; learned representations for pattern/shape.
- **Accessible:** simple consumers can recover the information they need from named state views.
- **Efficient/fast:** vectorized preparation and sequence operations, bounded memory, measured
  runtime, and a working CPU fallback.
- **Honest/robust:** finite output, explicit uncertainty, reproducible receipts, fair comparators,
  and no gate changes made to obtain a pass.

The product goal is not “a small vector must beat every engineered table.” The product goal is a
**reliable frozen foundation interface** whose components and limitations are measured
consistently.

---

## 2. Current baseline: preserve these assets

The Layer-0 loops have already produced real, useful work. Do not restart the project or repeat
completed loops without a new hypothesis.

### 2.1 Completed implementation

- `looking_glass/state_layout.py`
  - single source of truth for block order and named state views;
  - views are assembled by block name, not incorrect contiguous slices;
  - verification showed every `BLK` column is used exactly once.
- `looking_glass/measure_state.py`
  - lawful typed measure state over all discovered fields;
  - exact event counts, categorical distributions, missing fractions, per-entity distinct counts,
    and numeric count/sum/mean/std/min/max;
  - raw accumulation before normalization;
  - bounded, data-derived vocabulary;
  - exact additivity of raw additive blocks, unit-tested.
- `looking_glass/foundation.py`
  - `FrozenFoundation` load/state/describe/save/load;
  - provenance, schema/config, layout, measure vocabulary, and content hash;
  - deterministic round-trip on the real cache;
  - refuses to emit state with missing `as_of` or non-finite output.
- `scripts/evaluate_foundation_state.py`
  - attribution battery separating shape, learned state, forecast, measure, and full state.
- `scripts/measure_runtime.py`
  - serving and runtime receipt.
- Schema-derived identity roles in `scripts/demo/stream_loader.py`
  - selected behaviorally rather than by business names;
  - on the reference data the selection exactly matches the previous hand-listed identity set.
- `REPORT_CARD.md` at repository root and `docs/AUDIT_CARD.md`.
- New tests across measure state, schema roles, and the artifact.

### 2.2 Current evidence

The primary card reports:

```text
B:   28 PASS / 0 FAIL
B+C:  6 PASS / 0 FAIL
Total: 34 PASS / 0 FAIL
```

The full audit remains intentionally open:

```text
26 PASS / 3 FAIL
```

The open research rows include category identity, IPS policy value, and the strict next-event
point gate. Churn is a ceiling tie in the supervised floor/ceiling analysis.

The current card is valid evidence for its **declared Layer-0 scope and published state**. It is
not proof that the learned 128-dimensional `E` alone is a universal sufficient statistic.

### 2.3 Measured multi-view behavior

The attribution battery established that the views are complementary rather than redundant:

| View | Measured strength |
|---|---|
| `shape` | Strong churn result; compact behavioral representation |
| `learned_state` | Channel identity |
| `forecast` | Retrieval and churn-neighbor structure |
| `measure` | Category identity and best non-table anomaly view |
| `full` | Strongest supervised value/orders view |

Representative current-state results:

- `shape` beats the table on churn (approximately 0.936 vs 0.911).
- `measure` nearly matches the table on category identity (approximately 0.261 vs 0.265).
- `full` reaches approximately 0.433 on value90 versus approximately 0.372 for the table.
- `shape` alone trails the table on value/orders; the stronger published result depends on the
  larger state, not just the compact embedding.
- Concatenating every view naively can dilute raw-distance geometry and identity. Consumers need
  a declared view-selection protocol, not an assumption that more dimensions always help.

### 2.4 Important bugs and integrity findings

These are part of the current engineering baseline and must not be forgotten:

1. **Latent NaN behavior:** `embed()` could return all-NaN when `AS_OF_TS` was unset, and legacy
   measurement paths silently converted that to zero with `nan_to_num`. `as_of_ts` is now
   persisted in the schema config and the new artifact rejects the invalid state. Legacy paths
   still need migration to the same fail-safe contract.
2. **Degenerate RL bootstrap:** the response receipt reported `E_minus_random` as exactly zero
   with a zero-width CI alongside non-zero IPS/DR values. That is not a valid paired bootstrap.
   The IPS row is excluded from the primary card until the evaluator is fixed.
3. **Receipt provenance:** multiple measurement processes can write the same receipts with
   different `WQ_AXES` settings. A config receipt must be tied to the exact cache used by every
   other receipt, not to whichever process wrote last. A prior `noaction` battery left stale
   config metadata after a baseline restore; this must be checked and prevented.

### 2.5 Measured negative results

Do not repeat these unchanged:

| Experiment | Outcome | Disposition |
|---|---|---|
| Input-conditioned `selective` retention | Improved some identity/anomaly metrics but regressed donor margins at 6k | Rejected for the default path; retained behind a flag |
| `noaction` masking of company actions | Improved value/orders but made churn add-over-sequence negative and failed D3 | Rejected for the default path; retained behind a flag |
| Length-sorted token-budget batching | 62.8 rows/s versus 110.3 rows/s for existing batching, with equivalent numerics | Reverted |
| Long cumulative architecture sweeps | Slow and mostly neutral or worse | Not an acceptable default exploration method |

A rejected experiment is a recorded result, not a task to keep retrying without a new mechanism.

---

## 3. What remains incomplete

The remaining work is broader than the RL bootstrap.

### 3.1 Public input/output lifecycle

The complete public workflow is not yet implemented:

```text
raw canonical stream → fit → save → load → transform(raw events, as_of)
```

Specific gaps:

- `FrozenFoundation.fit(...)` is not implemented.
- `state(...)` expects prepared `seqs`, not raw events.
- There is no single `looking_glass/stream_adapter.py` that owns direct canonical input.
- `scripts/demo/stream_loader.py` still reflects the generated dataset layout and evaluation
  needs rather than a clean public training/serving interface.
- Training must not require the wide table; the wide table is an evaluation baseline only.

### 3.2 State consumption

The artifact exposes views, but the rest of the system does not yet use them consistently:

- some measurements still construct `E + BLK` manually;
- `full` needs one permanent, documented definition;
- `full_measure` must remain a separate named arm if evaluated;
- consumers need a train-only view-selection protocol, not test-set selection;
- the card must name the view used by every row.

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

The runtime receipt identifies the forward pass as the main serving bottleneck:

- cached stream load: approximately 0.26 s;
- artifact load: approximately 0.02 s;
- measure transform: approximately 3,000 rows/s;
- frozen forward: approximately 110 rows/s;
- five-fold head fit: approximately 66 s;
- peak RSS: approximately 2.2 GB.

The scan has `O(T log T)` total work in its current Hillis–Steele form. Target construction,
padding, and duplicate full-history forwards also need profiling. No performance claim should be
made without before/after runtime, memory, and numerical-equivalence receipts.

### 3.5 RL evidence

Known-effect recovery on randomized synthetic action arms is a valid, separate result. It is not
a deploy-grade offline policy-value estimate.

Before any IPS/DR policy claim:

- fix response arm alignment;
- verify independent random-donor control;
- recompute the paired statistic correctly on every grouped bootstrap resample;
- reject impossible zero-width CIs with nonzero point estimates;
- keep pre-action state and logged propensities;
- add bootstrap-validity and no-future-leakage tests.

A new decision-episode data revision may be necessary for a defensible policy-value claim. It
would invalidate prior receipts and must be introduced as a new version, not silently substituted
for the current demo data.

### 3.6 Research Layer 1

Minimality, worst-case universal `U`, and decision-sufficiency remain separate research goals.
They are not required to pretend that Layer 0 is complete, and they are not claimed by the current
card. They must be scheduled explicitly if the research north star is to be pursued.

---

## 4. Target architecture and public contract

The stable public API should be:

```python
artifact = FrozenFoundation.fit(canonical_stream, config=AutoConfig())
artifact.save(path)

artifact = FrozenFoundation.load(path)
state = artifact.transform(events, as_of=cutoff)

state.shape
state.learned_state
state.forecast
state.measure
state.uncertainty
state.heads
state.full
```

The artifact must contain:

- frozen weights;
- schema and field-role map;
- tokenizer and vocabulary;
- missing/unknown/zero rules;
- data-derived temporal and horizon configuration;
- state layout and dimensions;
- lawful measure vocabulary;
- data hash/revision, code version, seed, and artifact hash;
- provenance and runtime metadata.

`full` must have one permanent definition. Do not silently change it to include or exclude
measure. If an additional `full_measure` view exists, name and measure it separately.

### State construction principles

- **Shape:** learned sequence representation for behavioral patterns.
- **Measure:** exact or lawfully derived statistics for extensive quantities and identity/rate.
- **Forecast:** generic self-supervised future features, retained only when they add measured value.
- **Uncertainty:** missingness, unseen values, support, and calibration information.
- **Full:** a documented composition, not an opaque bundle assembled differently by each script.

### Training principles

- Learn what requires learning; compute what is lawfully computable.
- Derive field roles from behavior and declared metadata, not business names.
- Keep actions exogenous to the customer foundation model.
- Generate self-supervised objectives from schema and causal cutoffs.
- Use held-out self-supervised and accessibility evidence for checkpoint selection, never the
  training metric alone and never the demo card.

---

## 5. Execution roadmap

## Phase 0 — Evidence identity and reproducibility

Do this before further model changes.

1. Create a run manifest from the exact frozen artifact/cache used by all measurements:
   - data path and data revision;
   - seed and axes;
   - artifact/cache hash;
   - code commit;
   - dimensions and state layout;
   - training and runtime configuration.
2. Make `encoder_config.json` derive from that manifest, not from the last battery process.
3. Assert that cache axes, manifest axes, and card config agree.
4. Use run-specific receipt paths and atomic publication to prevent competing writers.
5. Add a regression test for config/receipt mismatch.
6. Regenerate the primary card and audit card from the consistent manifest.

**Accept when:** every card row resolves to one identified artifact and no config is inherited
from an unrelated run.

## Phase 1 — Direct stream adapter and artifact lifecycle

Create `looking_glass/stream_adapter.py` as the single canonical input path.

It must:

- accept canonical Arrow/Parquet/DataFrame event streams;
- validate columns, timestamps, entity keys, and ordering;
- expand attributes and infer types;
- infer identity/static roles behaviorally;
- use declared action metadata, with a documented structural fallback;
- distinguish missing, unknown, and observed-zero values;
- construct causal per-entity sequences;
- derive configuration from training data only;
- emit a schema/config receipt.

Then:

- make `scripts/demo/stream_loader.py` a compatibility wrapper;
- implement `FrozenFoundation.fit(...)`;
- implement `transform(events, as_of=...)` so callers never construct `seqs` manually;
- persist everything needed to serve in a fresh process.

Required tests:

- direct stream input without a wide table;
- renamed fields and an additional field;
- missing/unknown/observed-zero semantics;
- empty stream and single-entity stream;
- long gaps and duplicate timestamps;
- causal cutoff correctness;
- save/load determinism;
- no labels in encoder training inputs.

**Accept when:** an unseen canonical stream can be fitted, saved, loaded, and transformed with no
business-specific edits and no wide-table dependency for training.

## Phase 2 — Unified state consumption and view selection

1. Freeze the definition of `full` in `state_layout.py`.
2. Keep `full_measure` separate if it is evaluated.
3. Migrate measurement and serving code from hand-built `E + BLK` bundles to the artifact.
4. Implement train-only view selection:
   - declare candidate views before evaluation;
   - select with training-fold cross-validation only;
   - freeze the selected view before the holdout;
   - evaluate the holdout once with paired uncertainty.
5. Report shape-only, measure-only, full, and selected-view results separately.
6. Do not force measure into every consumer; use it where training-only evidence supports it.

**Accept when:** every B and B+C row names its state view and no view is chosen because it looks
best on the test set.

## Phase 3 — Encoder quality and adaptive behavior

Handle each item as a separate experiment.

### 3A. Action/objective semantics

- Preserve declared exogenous-action metadata.
- Diagnose why action masking improved value/orders but hurt churn.
- Try only a new, mechanistically justified single-axis solution.
- Do not simply rerun `noaction` and do not weaken D3.
- If no acceptable solution is found, retain the baseline and document the unresolved tension.

### 3B. Selective retention

- Do not repeat the rejected implementation unchanged.
- A new attempt must explain why it should preserve donor margins while improving retention.
- Compare against the exact baseline at the same scale before considering adoption.

### 3C. Relational roles

- Keep declared relational configuration until a genuinely general rule is validated.
- Test any proposed rule on schema variants and unrelated fields, not only the reference pair.
- Do not invent a heuristic just to reproduce the old pair.

### 3D. Missing/unknown/zero and non-finite behavior

- Preserve separate missing, unseen, and observed-zero channels.
- Reject non-finite output before scoring.
- Replace legacy silent sanitization in production paths with explicit failure receipts.

### 3E. Adaptive capacity and convergence

- Replace fixed `d`, layers, and epoch budgets with derived settings or documented fallbacks.
- Use a small nested held-out capacity search with a cost penalty, not a broad sweep.
- Use held-out convergence with noise-derived tolerance.
- Never select weights by training loss or by the demo card.

## Phase 4 — Performance

Profile first. Optimize the largest measured cost, one change at a time:

1. vectorize target and measure construction;
2. remove redundant full-history forwards where mathematically safe;
3. reduce padding without creating many tiny batches;
4. stabilize compiled scan shapes;
5. cache deterministic preparation and targets;
6. benchmark CPU/GPU and AMP with numerical and determinism checks.

For every candidate record runtime, peak memory, numerical difference, and gate effect. Reject a
change that is slower or changes results beyond the declared tolerance.

## Phase 5 — RL evidence

Fix the evaluator before changing the encoder or the dataset:

1. Verify policy/logged-action/reward alignment.
2. Verify the random-donor arm is independent.
3. Recompute the paired statistic correctly on every grouped bootstrap resample.
4. Fail validation on impossible intervals such as zero-width CI with nonzero point estimate.
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

## Phase 7 — Research Layer 1 (separate from Layer-0 product completion)

If pursuing the research north star, schedule separately:

- minimality and served-rank certification;
- worst-case universal `U` against a full-history oracle;
- learned-only versus served-statistic attribution;
- decision-sufficiency and deploy-grade OPE;
- calibration and coverage guarantees.

Do not use a Layer-0 green card to imply these are complete.

---

## 6. Measurement protocol for every loop

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

## 7. Execution loop for the implementation agent

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

## 8. File ownership for the remaining work

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

## 9. Definition of done for this plan

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

## 10. Immediate sequence

Start here, in this order:

1. **Evidence identity repair:** implement the run manifest, validate config/cache agreement,
   and add the mismatch regression test. No retraining.
2. **Direct stream lifecycle:** implement `stream_adapter.py`, `FrozenFoundation.fit`, and
   `transform(events, as_of=...)` with direct-stream and schema-variant tests.
3. **Artifact-consistent measurement:** move measurement writers onto the artifact and manifest,
   implement train-only view selection, and regenerate both cards.
4. **RL evaluator repair:** fix `measure_response.py` arm alignment and bootstrap validity; do not
   change the encoder or demo dataset for this loop.
5. **Performance profiling:** choose the next optimization from measured cost, not from another
   batching hypothesis.
6. **Only then:** new encoder-quality experiments or a new decision-episode data revision.

Do not begin with another architecture sweep, another capacity increase, or a repetition of the
rejected `selective`, `noaction`, or batching experiments.

The immediate objective is to make the **public lifecycle, state contract, and evidence identity**
as reliable as the current measured baseline, then improve the encoder on top of that foundation.
