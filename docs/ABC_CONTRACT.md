# The A/B/C pipeline contract

One canonical event stream in, one frozen foundation model out, plugins on top.
Everything flows one way:

```
Layer A: rabbit_hole (canonical events)
    -> Layer B: looking_glass (one frozen CFM per stream, E(customer,t))
        -> Layer C: plugins / gates / white_queen (read E at train AND serve)
```

The authoritative code is `rabbit_hole/schema.py`, `looking_glass/customer_foundation_model.py`,
`looking_glass/cfm_training.py`, `looking_glass/cfm_state.py`, `looking_glass/layer_b.py`,
`looking_glass/plugin_contract.py`, and the gates named below.

## Layers and seams

**Layer A — canonical event stream.** `rabbit_hole` emits one append-only event stream.
Onboarding writes, and the CFM reader accepts, the nine-column
`looking_glass.onboard.CANONICAL_COLUMNS`:

```
customer_key, event_ts, event_type, brand, entity_type, entity_id,
source_table, value, event_attributes
```

`rabbit_hole/schema.py` draws a stricter, deliberately minimal guarantee: every stream has
`customer_key, event_ts, event_type, source_table, event_attributes`
(`CANONICAL_FIELDS`). `brand`, `entity_type`, `entity_id`, and `value` are *not* assumed
universal there; `canonicalize_row()` folds any non-canonical column into the
`event_attributes` JSON map, and `canonical_view_sql()` exposes a wide table as the five-field
canonical view (`customer_events_canonical`). The CFM reader (`cfm_data._read_stream`) accepts
both forms: it reads top-level content columns when present, otherwise extracts them from
`event_attributes`; it also discovers every JSON attribute key so arbitrary schemas can be
consumed.

**Layer B — `looking_glass`.** Train **one** frozen foundation model per stream. It is a
causal state function:

```
state_c(t) = CFM_version(events of customer c with timestamp <= t)
E(customer,t) = learned shape || learned compression of the full-history measure
```

The encoder is self-supervised (no labels) and frozen before any plugin sees it. The
full-history measure is schema-derived from the whole stream (`layer_b.generic_measure`,
DEC-197), not from the queried customer subset; its compression is fit on training customers
and frozen for serving. `E` is produced by `cfm_training.forward_embedding()` and loaded at
serve by `cfm_state.load_frozen_encoder()`.

**Layer C — plugins.** A plugin reads the frozen `E` for **both** training and inference. It fits
on held-out B-customer anchors and is then frozen; serving never retrains it.

## Layer A canonical schema

- `customer_key`: customer identifier (string in canonical output).
- `event_ts`: event time; canonical output is UTC ISO text. `cfm_config.as_of_epoch()` is the
  one day-boundary rule: naive ISO dates mean UTC midnight.
- `event_type`: event name (e.g. `purchase`, `send`, `open`, `return`).
- `brand`, `entity_type`, `entity_id`: optional content dimensions. In the minimal
  `rabbit_hole.schema` contract these live inside `event_attributes`; onboarding promotes them
  to top-level columns when detectable.
- `source_table`: stream/table origin. Defaults to the source name during onboarding.
- `value`: monetary value of the event (float; 0.0 when absent).
- `event_attributes`: JSON object holding every field not promoted above. Action keys
  (`freq`, `hour`, `channel`, `offer`, ...) are JSON keys on send-like events.

`rabbit_hole.schema.COLUMN_ALIASES` defines accepted aliases (`customer_id`, `timestamp`,
`event_name`, `amount`, ...). `canonicalize_row()` folds all remaining columns into
`event_attributes`; no information is dropped.

## A -> B: train the frozen foundation model

```bash
python3 -m looking_glass.customer_foundation_model train \
  --db <stream.parquet> --as-of <date> \
  [--set use_measure=True \
   --set effect_actions=('freq','hour','channel','offer') \
   --set action_fields=('freq','hour','channel','offer','discount_pct')]
```

Useful accompanying flags: `--customers`, `--sample-a`, `--anchors`, `--out-dir`, `--set`.
`--as-of` cuts the raw stream to events `<= as_of` before keys, split, anchors, and training
(`cfm_data._cut_as_of`), so the run is point-in-time and leak-free.

**A/B split.** `cfm_data.assign_split()` hashes each customer into population A (default 70%,
`split_a_frac`) or B. The encoder trains only on sample A; A and B never overlap. `--as-of`
also rotates the split seed monthly (`cfm_config.monthly_split_seed`), so membership and
anchor dates refresh on each monthly rebuild. Everything downstream — plugin fitting,
anchor embeddings, gates, the report card — uses held-out population B.

**Anchors.** For sample-B customers, `build_sequences(..., with_anchors=True)` draws
`n_anchors` (default 6) uniform-random anchor days over the customer's history. At each
anchor the model encodes history up to that day, then fades the state to the anchor boundary
(the same rule used at serve time). These point-in-time embeddings are written to the
`anchor_embeddings` table by `cfm_state.build_products()`; plugins train on them.

**`E` composition (DEC-198).** At serve, `load_frozen_encoder()` sets
`model.accumulator_on = False`. The consumed embedding is:

```
E = shape (donor readout; 256 dims in the certified runs) || learned measure (82 dims)
```

i.e. ~338 dims. The **accumulator is the hand-RFM baseline** (token-drive sum/mean/last at
multiple timescales), not part of the production pipeline; it is switched on only for the
report card's "hand features" comparison row. `measure_dim=0` (default) keeps the measure
**full/lossless** (all measure columns; DEC-195). A positive value is explicit compression.
`scripts/reference_pipeline.py` requests `measure_dim=128`; with 82 available measure columns,
`LayerBEmbedder.fit()` caps components at 82, so the served pipeline `E` remains ~338.
`use_measure=True` activates the learned measure; `effect_actions` activates the
action-conditioned response objective; `action_fields` excludes logged action columns from
the generic measure so `E` does not carry the treatment itself.

## B -> C: the plugin contract

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

- `SupervisedPlugin(estimator, target=...)`: any scikit-learn regressor or classifier on `E`;
  `infer` returns predictions, `proba`/`predict_proba` where available.
- `UnsupervisedPlugin(n_clusters=..., scaler=..., kmeans=...)`: KMeans on `E` (optional
  scaler); `infer` returns cluster labels, `transform` returns distances.
- `DecisionPlugin(actions=..., estimator=...)`: value model `V(E,a)`; `fit` takes
  `targets=(A, y)` with numeric actions, `values(E)` returns per-action values, `infer` returns
  `argmax_a V(E,a)`.
- `PluginRegistry`: name-keyed collection; `save_all(dir)`, `load_all(dir)`, `from_dir(dir)`.

A plugin is trained against the frozen `E` and is never given a path back into the encoder.

## Certification

**`U` — universality (`looking_glass/universality.py`).**

```
U = 1 - max_{informative tau} normalized oracle-gap_tau
gap_tau = score([E, exhaustive full-history measure]) - score(E)
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
randomized discrete action keys (JSON keys on send events, preferring a declared
`action_keys` receipt). `run_use_case(model, cfg, action_key)` reports the empirical
dose-response and a per-customer policy from `E`: a joint value model `V(E,a)` is cross-fitted
by customer, the candidate is chosen on one half of customers, and its value is estimated on
the other half by **g-computation** (no importance reweighting, no winner's curse). Verdict is
`DEPLOY` iff the paired bootstrap 95% CI lower bound on the policy gain is `> 0`, else `HOLD`.

```bash
python3 -m looking_glass.use_case_gate --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --actions freq,hour
```

**Null controls.** The report card re-runs the same use-case and decision gates on a
signal-free stream (`snr=0`) trained into a separate null checkpoint. Every real-stream
`DEPLOY` must be paired with a null `HOLD`; a null that also deploys invalidates the claim.

The card's default B+C arm uses `looking_glass.decision_certify.certify()` (doubly-robust
value with the logging marginal propensity; `N/A` when there is no per-customer arm
structure). The real tribunal below is the optional, stronger witness.

**Real white_queen (`looking_glass/white_queen_bridge.py`).** This puts
`white_queen.tribunal.ope.api.evaluate` directly on the `E`-policy:

- `certify(model, cfg, horizon_days=90)`: candidate policy reads `E`; reward is the logged
  future monetary outcome (`sum(log1p|value|)` on purchases); no logged propensity is passed
  (white_queen estimates it).
- `certify_period(model, cfg, action_key="channel")`: per-period randomized action (channel,
  offer, freq, ...); context is the customer's frozen `E`.
- `oracle(...)`: a known-truth policy (e.g. `arm_pref`/`channel_pref`) as a positive control.

DEPLOY/HOLD is white_queen's conservative verdict: never ship worse than logging.

**Supervised gate (`looking_glass/supervised_gate.py`).** `run_supervised()` fits heads on
held-out `E` for churn, reactivation, next purchase, return, LTV, next value, next
channel/category/item. Per target it reports skill, a bootstrap 95% CI, and `PASS`/`FAIL`
(`PASS` iff the CI lower bound is above the model-free baseline: 0 for regression,
majority/0.5 for classification). With `null=(db, tag, ckpt_dir)`, the same targets run on
the null encoder and `null_ok` is true only if every null target `FAIL`s.

**Unsupervised gate (`looking_glass/unsupervised_gate.py`).** `run_unsupervised()` certifies
segmentation and lookalike geometry on held-out `E`: silhouette `> 0`, value-differentiation
ANOVA `p < 0.05`, and lookalike enrichment `> 1.0` are all required for `PASS`. With a null
encoder, `null_ok` is true only if the null verdict is `FAIL`.

**Report card (`looking_glass/foundation_report.py`).** `build_card()` emits the one-pager:
U verdict, per-target rows, architecture, use cases, B+C decision certification, optional
white_queen/supervised/unsupervised sections, and null controls.

```bash
python3 -m looking_glass.foundation_report --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --out card.md [--fast] [--white-queen --supervised --unsupervised]
```

`--fast` skips the bare/accumulator comparison U runs (the dominant cost); `--full` is **not**
a `foundation_report` flag. It is the `scripts/reference_pipeline.py` flag that adds white_queen,
supervised, and unsupervised to the card. Without it the reference card runs the fast core
(B + use-cases + null controls).

## One command

```bash
python3 scripts/reference_pipeline.py --as-of <date> [--full] [--dry-run]
```

Stages in `scripts/reference_pipeline.py`:

1. generate the REFERENCE stream (real signal);
2. generate the REFERENCE null control (`--snr 0`);
3. train the foundation model on the real stream;
4. train the foundation model on the null stream;
5. emit `card_reference.md` from the B & B+C gates, resolving the data-derived tags from the newest
   registry in each checkpoint directory.

`--full` adds white_queen/supervised/unsupervised and is slow; `--dry-run` prints the commands
without running them. Outputs live under `rabbit_hole/data/reference_pipeline` by default.
