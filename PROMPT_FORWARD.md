# PROMPT_FORWARD — the forward operating prompt

> **State:** run this prompt while the project is **not** all-green, or after
> `PROMPT_BACKWARD.md` returned at least one **BREAKING** finding.
> **Job:** move the project toward its goal by one measured axis per run, obeying the
> principles. This is not a coding prompt; code is only the instrument.
> **Every run ends with results written to `REPORT_CARD.md`.** When its gate-of-record
> section is all green, stop forward work and run `PROMPT_BACKWARD.md`.

---

## 0. Read order (do not skip)

1. `AGENTS.md` — doctrine, principles, certification integrity, workflow.
2. `docs/SPINE.md` — the constitution: goal, four consumers, judge, S1–S6, T/A/C-series
   laws, bucket table, ledger guardrails.
3. `STATUS.md` — live state (macro + micro). Read first to restore context.
4. `DECISIONS.md` — append-only decision log. Read the tail; never re-litigate a
   settled decision without new evidence.
5. `REPORT_CARD.md` — the ONE card: run report, gate-of-record, and every open row (see §4).
6. `AUDIT.md` — the principle-alignment audit written by BACKWARD.
7. `docs/report_cards/experiments/` — the receipts every number must trace to.
8. `docs/SPINE.md` Appendix A — the Layer-1 `U` contract.
9. §11 below — the A/B/C pipeline contract (layers, schema, plugin surface, gates).

If a file contradicts another, the order is: **AGENTS.md doctrine → SPINE laws →
frozen gate definitions → receipts → STATUS/DECISIONS narrative**. Narrative never
overrides a receipt.

---

## 1. The goal (one sentence, from SPINE §0)

> Input any event stream → output a **frozen** foundation model (one per stream) whose
> state supports **supervised**, **unsupervised**, **RL**, and **simulator** heads using
> simple consumers.

Layer 0 is the frozen universal donor. Layer 1 is certification: sufficiency,
minimality, decision-sufficiency, measured by `U`. Do not conflate them.

The judge is **PATH B (event stream) > PATH A (wide table)** on the consumer battery,
with paired CIs and controls. The two-number rule is binding: **brain alone = progress;
brain + ledger = delivery.** Ledger points never count as a brain win.

---

## 2. Principles that bind every run

- **Conflict order:** Robust → Universal → Adaptive → Lawful → Efficient → Honest.
  Lawful and Robust outrank convenience.
- **The one rule:** never encode an answer the system could learn or derive.
- **Anti-overfit clause:** Layer B is self-supervised only; known use cases may VETO
  (gates), never SHAPE the encoder; never train on the graded metric.
- **Accessibility:** information must be recoverable by simple consumers (linear for
  supervised shipping; kNN for unsupervised). MLP is diagnostic only.
- **Honesty:** pre-register before the run; grade against frozen acceptance; report
  FAIL; never weaken a gate to pass.
- **Efficiency:** one axis per run, cheapest diagnostic first, linear-time and
  vectorized, no per-row Python loops in hot paths.
- **Robustness:** divergence is a hard failure; reject or widen uncertainty, never
  clamp; deterministic and seeded.

---

## 3. The forward loop (binding)

For every run:

1. **Generate the card** — run the one command that regenerates it (see §6). Read the
   gate-of-record section and the blocking rows.
2. **Pick the highest-leverage unmet subgoal** that is achievable now; cheapest first;
   exactly **one axis**.
3. **Diagnose before protocol** — if a cheap read-only test can falsify the premise,
   run it first and record it as a diagnostic (DEC-462/464 rule). A diagnostic may not
   adopt anything.
4. **Pre-register the decision in `DECISIONS.md` before the run.** It must name:
   context; the principle(s) served; ≥2 rejected alternatives; the single axis changed;
   the named metric; the frozen acceptance threshold; guards; the falsifier; the cost;
   and what is **not** changed (gates, comparator, probes, E's content, ledger).
5. **Implement the one change** and run the one command that emits the receipt. The run
   is the measurement.
6. **Grade against the frozen acceptance before any verdict.** Adopt only if the named
   axis moved. Otherwise revert, archive the receipts under
   `docs/report_cards/experiments/protocols/<name>/`, and restore the live receipts of
   the recorded artifact.
7. **Regenerate the card at a clean, recorded commit**; write the run result to
   `REPORT_CARD.md`; update `STATUS.md`; append the measured outcome to `DECISIONS.md`;
   commit code + docs (never data/artifacts).
8. **If the gate-of-record is not all green, repeat.** If it is all green, run
   `PROMPT_BACKWARD.md`.

A run that cannot name its axis, its frozen acceptance, or its receipt is not a run.

---

## 4. The single card and the completion contract

**Operator ruling (DEC-497, extended): `REPORT_CARD.md` is the one and only card.** There is
no second card file: the old judge card was retired and its file deleted, and the separate
`docs/AUDIT_CARD.md` research audit was removed — the single renderer
`scripts/report_card.py` now carries the 8 leg-columns, the 6-stage ladder, certification
hygiene, freshness, the anti-false-PASS rules, S6, **and** every open row. If a row is open it
appears on the card; nothing may be parked in a side document.

### Required `REPORT_CARD.md` shape (generated, never hand-edited)

```text
# REPORT_CARD — PATH B > PATH A
  header: generated_at · commit · branch · clean/dirty · encoder behavior hash
## Assessment
  the generated completion condition (NOT DONE / ENCODER DONE)
## The judge — four legs × two numbers
  8 rows: supervised/unsupervised/rl/simulator × brain alone / brain + ledger
## The stage ladder
  S1..S6, each with its working-state proof and verdict
## Certification hygiene
  H1..H5
## Backward requirements
  the structured findings from `AUDIT.md`
## What is blocking
  exact missing receipts and open backward requirements
```

The renderer reads receipts under `docs/report_cards/experiments/` and
`AUDIT.md`; it never invents a number. A run that edits the card
by hand is not evidence.

### Backward requirements are card requirements

`PROMPT_BACKWARD.md` writes **one file, `AUDIT.md`**: the narrative, the principle-alignment
scorecard, and the `BACKWARD_REQUIREMENTS_JSON` block at the end. BACKWARD writes no other file.
Each requirement has an `id`, `severity`
(`BREAKING`/`MAJOR`/`MINOR`/`INFO`), `principle`, `status` (`open`/`closed`/`accepted`),
`finding`, `forward_subgoal`, `gate_touched`, `evidence`, `audit_ref`, and `date`. The
renderer reads that block and displays the requirements in `REPORT_CARD.md`; an open
`BREAKING` or `MAJOR` requirement blocks the completion condition. `MAJOR` may be closed by a forwarding protocol or
explicitly `accepted` by the operator; `BREAKING` may never be accepted, only closed by
evidence.

### "All green" — the only definition that counts

All of the following, simultaneously:

1. `REPORT_CARD.md`: **0 FAIL, 0 UNKNOWN**; every one of the 8 leg-columns is PASS or
   **EXCLUDED-and-reported** under a frozen, operator-approved exclusion rule; all
   **6/6 stages** PASS.
2. Hygiene 5/5 (environment, behavior hash, N, controls, finite/fair comparator);
   S6 6/6 (one command, deterministic render, clean recorded commit, artifact hashes,
   every row reports N).
3. Freshness: the receipts' `behavior_hash` equals the current tree; no stale block;
   no unadopted receipt on the card.
4. **No open BREAKING/MAJOR backward or principle requirement** in `AUDIT.md`'s
   `BACKWARD_REQUIREMENTS_JSON` block; every `MAJOR` is closed or operator-accepted. This
   includes every open finding in the same file.
5. No open operator ruling blocks a row; every stop-and-ask item is resolved or
   explicitly recorded as a non-blocking disclosure.
6. The generated assessment says the completion condition. A hand-written "SOLVED",
   "universal", or "PASS" is a bug.

EXCLUDED-and-reported is not a free pass: it applies only to a target with no variance
or a baseline that already saturates, and only under the frozen exclusion rule. Proposing
a new exclusion is a **gate change** → operator.

---

## 5. Evidence and certification integrity (binding)

Follow AGENTS.md's certification integrity rules 1–8. In practice:

- **Freeze the gate before the result.** Definitions, thresholds, criteria, and
  selection rules are in `DECISIONS.md` and the receipt **before** the run.
- **A receipt may not define its own criterion.** New criterion strings are UNKNOWN
  unless pre-registered.
- **No PASS from a point estimate.** A PASS needs a finite, fair comparator, a paired
  CI excluding zero, a recorded N, and controls below the brain.
- **No circular evaluation.** The object under test must not contain the oracle's
  features. Report learned-only, ledger-only, shape-only, random, destroyed controls.
- **No post-hoc gate edits.** A criterion change after seeing a result is a new
  protocol version: stop, get operator approval, re-run from scratch, report both.
- **Reproducibility is part of the claim.** One repo command at a clean, recorded
  commit; artifact hashes and N; receipts in the repo, not `/tmp`.
- **Freshness is mechanical.** A behavior change marks dependent receipts stale →
  UNKNOWN until regenerated.
- **Never train on the graded metric.** Known tasks may veto, never shape the encoder.
- **Audit what you RUN.** Any executing code is in scope, not just what you wrote or changed;
  a measurement from un-audited code is not evidence. `--all --strict` is the gate.

---

## 6. Documentation and artifact upkeep (mandatory)

| Artifact | What to keep current |
|---|---|
| `STATUS.md` | macro (goal, stage, progress, blockers) + micro (current task, last completed, next 3, environment). Update **before yielding**. |
| `DECISIONS.md` | append-only. Pre-registration before every run; measured outcome after; context / alternatives / decision+principle / trade-offs. Never edit old entries. |
| `REPORT_CARD.md` | the run report + gate-of-record section (§4) and **every open row** — it is the only card; no row may be hidden. Regenerated each run; no hand-written numbers. |
| `AUDIT.md` | the single BACKWARD file: principle audit + findings + `BACKWARD_REQUIREMENTS_JSON` block rendered into `REPORT_CARD.md` (§4). |
| `docs/report_cards/experiments/` | receipts; `protocols/<name>/` archives for non-adopted runs. |

**Commit discipline:** every coherent verified chunk is committed; code + docs only;
data, checkpoints, logs, and virtualenvs are never committed. Commits stay **local**
until the operator explicitly asks to push. The card is regenerated at a clean recorded
commit; a receipt that no repo command regenerates is not evidence.

### Commands (read the current paths from `STATUS.md`; never guess an artifact)

- One-command chain + card:
  `python3 scripts/measure_certification_law.py --products <recorded artifact> --demo <demo> --train-log <artifact>/train.log`
- Report card render (the single card): `python3 scripts/report_card.py`
- Principle audit (ALL code we run, not just changed files):
  `python3 scripts/principle_audit.py --all --strict` · summary: `--all --summary`

The chain command is the only S6-certifying path. A hand-rendered card is not evidence.

**Evidence scripts:** any script committed as evidence must pass
`python3 scripts/principle_audit.py --strict` with **0 findings**, be vectorized (no
per-row Python loops in hot paths), and emit a runtime receipt.

---

## 7. Resources and operating constraints

- **One GPU job at a time.** Derive budgets from VRAM/hardware; keep the CPU fallback.
- **Caps are the operating model:** vLLM 11 GiB RAM / 15 GiB total, GMU ≤ 0.70
  (refuse-above); training container 20 GB; system floor 8 GB. The watchdog is opt-in
  (`--watchdog`), not the default.
- **Co-tenancy:** validate the trainer's budget against the resident LLM before a run;
  never run two LLM containers at once; record the low-water/peak memory receipt.
- **No code edits to `looking_glass/` while a run's lazy imports are pending.**
- **Never run destructive deletes, material spend, customer-facing actions, or live
  exploration without explicit operator authorization.**
- **Never push** without explicit permission.

---

## 8. End-of-run checklist (print it, then commit)

1. The one command ran; every producer exited 0; the receipt exists.
2. Frozen acceptance was graded **before** the verdict; the named axis is stated.
3. Adopted → keep; not adopted → revert + archive receipt + restore live receipts.
4. `STATUS.md` updated; `DECISIONS.md` outcome appended; open `AUDIT.md` findings are
   either closed with their receipt or explicitly carried as the next subgoal.
5. `REPORT_CARD.md` run sections written from generated cards; gate-of-record section
   agrees with the renderer's own generated bytes; no contradiction.
6. Clean-tree commit recorded; hashes/N/controls present; no `/tmp`-only evidence.
7. `python3 scripts/principle_audit.py --all --strict` clean: **0 new blocking findings on the
   whole surface**, not merely on the files touched. The baseline may only shrink. ruff/tests per
   the repo's gates (or the known-failure baseline disclosed).
8. Print the rendered `REPORT_CARD.md` (never fenced) as the run's source of truth.
9. If all green → stop and run `PROMPT_BACKWARD.md`. Else → next FORWARD run.

---

## 9. Stop-and-ask triggers (operator only)

Stop **before** running if a proposed change:

- changes a gate, threshold, criterion, frontier definition, or exclusion rule;
- changes what "shape" contains, what "full-E" means, or what the comparator is;
- would move a verdict from FAIL to PASS by anything other than a real capability
  improvement;
- introduces a new consumer, artifact contract, bucket, or architecture;
- cannot be reproduced from the repo at a recorded commit;
- requires material spend, destructive deletes, customer-facing exposure, or a push.

Ask once, with the measured evidence and a recommendation. Silence is never assent for
a gate-touching or authorization item.

---

## 10. Forbidden shortcuts (the failure modes this project has already paid for)

- Declaring a point estimate a win when the paired CI includes zero.
- Selecting a hyperparameter on the scored split (e.g., a whitening rank swept on B).
- Training on the target the card grades (circularity), or letting labels reach Layer B.
- Adding an unadopted receipt to the card; leaving a stale behavior hash; hiding a row.
- Weakening a gate, redefining a target, or excluding a row to turn a verdict green.
- Narrating a number instead of generating it; editing the card to improve a verdict.
- Hope-driven reruns of a falsified ladder rung ("512-d next") without a new mechanism.
- Per-row Python loops in hot paths; batch-coupled serving statistics; silent clamps.
- Running a second LLM container, exceeding the caps, or training while the co-tenant
  is unbounded.

---

## 11. The A/B/C pipeline contract

*(merged from the retired `docs/ABC_CONTRACT.md` — this is the only copy)*

One canonical event stream in, one frozen foundation model out, plugins on top. Everything
flows one way:

```text
Layer A: rabbit_hole (canonical events)
    -> Layer B: looking_glass (one frozen CFM per stream, E(customer,t))
        -> Layer C: plugins / gates / white_queen (read E at train AND serve)
```

Authoritative code: `rabbit_hole/schema.py`, `looking_glass/customer_foundation_model.py`,
`looking_glass/cfm_training.py`, `looking_glass/cfm_state.py`, `looking_glass/layer_b.py`,
`looking_glass/plugin_contract.py`, and the gates named below.

### Layer A — canonical event stream

`rabbit_hole` emits one append-only stream. Onboarding writes, and the CFM reader accepts, the
nine-column `looking_glass.onboard.CANONICAL_COLUMNS`:

```text
customer_key, event_ts, event_type, brand, entity_type, entity_id,
source_table, value, event_attributes
```

`rabbit_hole/schema.py` draws a stricter, deliberately minimal guarantee: every stream has
`customer_key, event_ts, event_type, source_table, event_attributes` (`CANONICAL_FIELDS`).
`brand`, `entity_type`, `entity_id`, `value` are *not* assumed universal there;
`canonicalize_row()` folds any non-canonical column into the `event_attributes` JSON map and
`canonical_view_sql()` exposes a wide table as the five-field canonical view
(`customer_events_canonical`). `cfm_data._read_stream` accepts both forms — top-level content
columns when present, otherwise extracted from `event_attributes` — and discovers every JSON
attribute key so arbitrary schemas are consumed without code edits.

- `customer_key` — customer identifier (string in canonical output).
- `event_ts` — event time; canonical output is UTC ISO text. `cfm_config.as_of_epoch()` is the
  one day-boundary rule: naive ISO dates mean UTC midnight.
- `event_type` — event name.
- `brand`, `entity_type`, `entity_id` — optional content dimensions; in the minimal
  `rabbit_hole.schema` contract these live inside `event_attributes`, and onboarding promotes
  them to top-level when detectable.
- `source_table` — stream/table origin; defaults to the source name during onboarding.
- `value` — monetary value (float; 0.0 when absent).
- `event_attributes` — JSON object holding every field not promoted above. Action keys
  (`freq`, `hour`, `channel`, `offer`, ...) are JSON keys on send-like events.

`rabbit_hole.schema.COLUMN_ALIASES` defines accepted aliases (`customer_id`, `timestamp`,
`event_name`, `amount`, ...). `canonicalize_row()` folds remaining columns into
`event_attributes`; no information is dropped.

### Layer B — one frozen CFM per stream

```text
state_c(t)   = CFM_version(events of customer c with timestamp <= t)
E(customer,t)= learned shape || learned compression of the full-history measure
```

Self-supervised (no labels), frozen before any plugin sees it. The full-history measure is
schema-derived from the whole stream (`layer_b.generic_measure`, DEC-197), not from the queried
customer subset; its compression is fit on training customers and frozen for serving. `E` is
produced by `cfm_training.forward_embedding()` and loaded at serve by
`cfm_state.load_frozen_encoder()`.

```bash
python3 -m looking_glass.customer_foundation_model train \
  --db <stream.parquet> --as-of <date> \
  [--set use_measure=True \
   --set effect_actions=('freq','hour','channel','offer') \
   --set action_fields=('freq','hour','channel','offer','discount_pct')]
```

Flags: `--customers`, `--sample-a`, `--anchors`, `--out-dir`, `--set`. `--as-of` cuts the raw
stream to events `<= as_of` before keys, split, anchors, and training
(`cfm_data._cut_as_of`) — point-in-time and leak-free.

**A/B split.** `cfm_data.assign_split()` hashes each customer into population A (default 70%,
`split_a_frac`) or B. The encoder trains only on sample A; A and B never overlap. `--as-of`
also rotates the split seed monthly (`cfm_config.monthly_split_seed`), so membership and anchor
dates refresh on each monthly rebuild. Everything downstream — plugin fitting, anchor
embeddings, gates, the card — uses held-out population B.

**Anchors.** For sample-B customers, `build_sequences(..., with_anchors=True)` draws `n_anchors`
(default 6) uniform-random anchor days over the customer's history. At each anchor the model
encodes history up to that day, then fades the state to the anchor boundary (the same rule used
at serve). These point-in-time embeddings are written to `anchor_embeddings` by
`cfm_state.build_products()`; plugins train on them.

**`E` composition (DEC-198).** At serve, `load_frozen_encoder()` sets
`model.accumulator_on = False`. The consumed embedding is

```text
E = shape (donor readout; 256 dims in the certified runs) || learned measure (82 dims)   ~338
```

The **accumulator is the hand-RFM baseline** (token-drive sum/mean/last at multiple timescales),
not part of the production pipeline; it is switched on only for the card's "hand features"
comparison row. `measure_dim=0` (default) keeps the measure **full/lossless** (DEC-195); a
positive value is explicit compression. `scripts/reference_pipeline.py` requests
`measure_dim=128`; with 82 available measure columns `LayerBEmbedder.fit()` caps components at
82, so the served `E` remains ~338. `use_measure=True` activates the learned measure;
`effect_actions` activates the action-conditioned response objective; `action_fields` excludes
logged action columns from the generic measure so `E` does not carry the treatment itself.

### Layer C — the plugin contract

A plugin reads the frozen `E` for **both** training and inference: it fits on held-out
B-customer anchors, is then frozen, and serving never retrains it.

`looking_glass/plugin_contract.py` defines one surface, pickle-persisted, dependency-light
(numpy + scikit-learn):

| member | meaning |
|---|---|
| `Plugin.fit(E, targets)` | learn from frozen `E` and family-specific targets |
| `Plugin.infer(E)` | apply the fitted artifact (no retraining) |
| `Plugin.save(path)` / `Plugin.load(path)` | pickle persistence |
| `Plugin.freeze()` | make immutable; later `fit` raises |
| `fit_and_freeze(plugin, E, targets, out_dir)` | fit, freeze, save, return the frozen plugin |

Families:

- `SupervisedPlugin(estimator, target=...)` — any scikit-learn regressor/classifier on `E`;
  `infer` returns predictions, `proba`/`predict_proba` where available.
- `UnsupervisedPlugin(n_clusters=..., scaler=..., kmeans=...)` — KMeans on `E` (optional
  scaler); `infer` returns cluster labels, `transform` returns distances.
- `DecisionPlugin(actions=..., estimator=...)` — value model `V(E,a)`; `fit` takes
  `targets=(A, y)` with numeric actions, `values(E)` returns per-action values, `infer` returns
  `argmax_a V(E,a)`.
- `PluginRegistry` — name-keyed collection; `save_all(dir)`, `load_all(dir)`, `from_dir(dir)`.

A plugin is trained against the frozen `E` and is never given a path back into the encoder.

### Certification gates

**`U` — universality (`looking_glass/universality.py`).**

```text
U        = 1 - max_{informative tau} normalized oracle-gap_tau
gap_tau  = score([E, exhaustive full-history measure]) - score(E)
normalized = gap_tau / range_tau
```

The oracle is the schema-derived exhaustive full-history measure (not hand-RFM), cross-checked
by an independent sequence model. Evaluation is on held-out population B. `SOLVED` means `E` is
within the oracle's own noise on every informative member.

```bash
python3 -m looking_glass.universality --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> [--customers 25000] [--horizons 30,90] [--require-solved]
```

`--synthetic` runs the known-sufficient-statistic machinery check.

**Use-case gate (`looking_glass/use_case_gate.py`).** `discover_action_keys(cfg)` finds
randomized discrete action keys (JSON keys on send events, preferring a declared `action_keys`
receipt). `run_use_case(model, cfg, action_key)` reports the empirical dose-response and a
per-customer policy from `E`: a joint value model `V(E,a)` is cross-fitted by customer, the
candidate is chosen on one half of customers, and its value is estimated on the other half by
**g-computation** (no importance reweighting, no winner's curse). Verdict is `DEPLOY` iff the
paired bootstrap 95% CI lower bound on the policy gain is `> 0`, else `HOLD`.

```bash
python3 -m looking_glass.use_case_gate --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --actions freq,hour
```

**Null controls.** The card re-runs the same use-case and decision gates on a signal-free
stream (`snr=0`) trained into a separate null checkpoint. Every real-stream `DEPLOY` must be
paired with a null `HOLD`; a null that also deploys invalidates the claim.

The card's default B+C arm uses `looking_glass.decision_certify.certify()` (doubly-robust value
with the logging marginal propensity; `N/A` when there is no per-customer arm structure). The
real tribunal below is the optional, stronger witness.

**Real white_queen (`looking_glass/white_queen_bridge.py`).** Puts
`white_queen.tribunal.ope.api.evaluate` directly on the `E`-policy:

- `certify(model, cfg, horizon_days=90)` — candidate policy reads `E`; reward is the logged
  future monetary outcome (`sum(log1p|value|)` on purchases); no logged propensity is passed
  (white_queen estimates it).
- `certify_period(model, cfg, action_key="channel")` — per-period randomized action (channel,
  offer, freq, ...); context is the customer's frozen `E`.
- `oracle(...)` — a known-truth policy (e.g. `arm_pref`/`channel_pref`) as a positive control.

DEPLOY/HOLD is white_queen's conservative verdict: never ship worse than logging.

**Supervised gate (`looking_glass/supervised_gate.py`).** `run_supervised()` fits heads on
held-out `E` for churn, reactivation, next purchase, return, LTV, next value, next
channel/category/item. Per target it reports skill, a bootstrap 95% CI, and `PASS`/`FAIL`
(`PASS` iff the CI lower bound is above the model-free baseline: 0 for regression,
majority/0.5 for classification). With `null=(db, tag, ckpt_dir)`, the same targets run on the
null encoder and `null_ok` is true only if every null target `FAIL`s.

**Unsupervised gate (`looking_glass/unsupervised_gate.py`).** `run_unsupervised()` certifies
segmentation and lookalike geometry on held-out `E`: silhouette `> 0`, value-differentiation
ANOVA `p < 0.05`, and lookalike enrichment `> 1.0` are all required for `PASS`. With a null
encoder, `null_ok` is true only if the null verdict is `FAIL`.

**Reference card (`looking_glass/foundation_report.py`).** `build_card()` emits the one-pager:
U verdict, per-target rows, architecture, use cases, B+C decision certification, optional
white_queen/supervised/unsupervised sections, and null controls. This is the *old* reference
pipeline's card — the project's steering card is `REPORT_CARD.md` (§4).

```bash
python3 -m looking_glass.foundation_report --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --out card.md [--fast] [--white-queen --supervised --unsupervised]
```

`--fast` skips the bare/accumulator comparison U runs (the dominant cost); `--full` is **not** a
`foundation_report` flag — it is the `scripts/reference_pipeline.py` flag that adds white_queen,
supervised, and unsupervised to the card. Without it the reference card runs the fast core
(B + use-cases + null controls).

### One command

```bash
python3 scripts/reference_pipeline.py --as-of <date> [--full] [--dry-run]
```

Stages in `scripts/reference_pipeline.py`:

1. generate the REFERENCE stream (real signal);
2. generate the REFERENCE null control (`--snr 0`);
3. train the foundation model on the real stream;
4. train the foundation model on the null stream;
5. emit `card_reference.md` from the B & B+C gates, resolving the data-derived tags from the
   newest registry in each checkpoint directory.

`--full` adds white_queen/supervised/unsupervised and is slow; `--dry-run` prints the commands
without running them. Outputs live under `rabbit_hole/data/reference_pipeline` by default.

---

*The forward run ends when the card is all green. The project is not done until
`PROMPT_BACKWARD.md` returns no BREAKING findings against that green card.*
