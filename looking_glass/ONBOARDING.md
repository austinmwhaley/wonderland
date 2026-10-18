# Onboarding a new / real stream

This is the "point it at your logs" path. Layer A is produced by
`looking_glass/onboard.py`, Layer B is trained by
`looking_glass/customer_foundation_model.py`, and Layer C plugins consume the frozen `E`.

Run everything from the repository root.

## Onboard

The source is a **positional argument** (there is no `--source` flag):

```bash
python3 -m looking_glass.onboard <source> --out-dir <dir> \
  [--action-keys freq,hour] [--cadence week]
```

`<source>` may be a parquet/`.pq` file, a directory of parquet files, CSV/TSV, Arrow/IPC
(`.arrow`, `.feather`), JSON/JSONL, a DuckDB file, or a polars DataFrame through the
`onboard()` API. Supported inputs and detection live in `onboard._read_source`,
`onboard._pick_column`, and `onboard.onboard`.

What it does:

- auto-detects `customer_key`, `event_ts`, `event_type`, `value`, and, when present,
  `brand`, `entity_type`, `entity_id`, `source_table` using aliases and name hints;
- folds every unmapped column into `event_attributes` JSON (existing JSON attribute columns
  are parsed and merged);
- normalizes timestamps to UTC ISO text, coerces values to floats, drops rows with a null or
  unparseable customer/time (and reports the count);
- discovers integer-valued **action keys** on send-like events and derives a **reward event**
  (the event type with the highest total monetary value);
- writes `<out-dir>/customer_event_stream.parquet` and `<out-dir>/onboard_meta.json`.

The receipt contains `canonical_path`, `n_rows`, `n_customers`, `columns_mapped`,
`action_keys`, `reward_event`, `cadence`, `value_col`, and `warnings`. Read it before training;
warnings mean a detection was filled with a default (`event_type` -> `"unknown"`, `value` ->
`0.0`) or that a timestamp was not monotone per customer.

### Overrides

Use `--columns FIELD=COLUMN` (repeatable, or one JSON object) when auto-detection is wrong:

```bash
python3 -m looking_glass.onboard events.csv --out-dir data/acme \
  --columns customer_key=user_id --columns event_ts=created_at \
  --columns value=order_total --columns event_type=event_name
```

Shortcuts also exist: `--customer-col`, `--time-col`, `--event-col`, `--value-col`. Mapping
targets are `customer_key`, `event_ts`, `event_type`, `brand`, `entity_type`, `entity_id`,
`source_table`, `value`. A missing customer or time column is a hard error; a missing event
or value column is filled with the warnings above. `--action-keys` overrides discovery, and
`--reward-event` overrides the derived reward event.

## Declared metadata from the receipt

Read `onboard_meta.json` and carry its declarations into training and certification:

- **Action keys** (`action_keys`) are JSON keys on send events. Pass them to the encoder so
  it learns the action-conditioned response and excludes the logged action from the measure:

  ```bash
  python3 -m looking_glass.customer_foundation_model train \
    --db data/acme/customer_event_stream.parquet --as-of <date> \
    --out-dir artifacts/acme --set use_measure=True \
    --set effect_actions=('freq','hour') \
    --set action_fields=('freq','hour')
  ```

  `effect_actions` activates the `effect` objective (`cfm_training`).
  `action_fields` is the exclusion list for the generic measure; include every action key,
  plus other logged action columns you do not want leaking into `E`. For parquet/DuckDB
  sources the CFM reader exposes JSON attribute keys as columns, so declared JSON keys resolve
  correctly (the onboarding output is parquet).

- **Reward event** (`reward_event`) is a receipt for the use-case/decision gates. Training does
  not take it as a `--set`. The use-case gate's period builder defaults to `reward_event="purchase"`
  (`use_case_gate._period_rows`); if your receipt names a different reward event, call the gate
  with that name or expose the reward under the expected event type. The encoder's `effect`
  objective uses the stream's monetary `value` signal rather than an event name.

- **Cadence** (`cadence`) is recorded in the receipt. The current action bucketing is weekly
  (`cfm_data.ACTION_PERIOD_S = 604800` and the same period in `use_case_gate._period_rows`);
  a non-weekly declared cadence is metadata, not yet a change to the gate's period length.

## Randomized holdout note

Decision certification is **only identifiable if actions were randomized** in the log (or a
small randomized holdout exists that can supply an unbiased action→reward sample). The gates
do not require a logged propensity: `white_queen_bridge` lets white_queen estimate it, and
`decision_certify` uses the empirical marginal only because randomized arms make it valid.
Confounded observational logs can still be scored, but the resulting `DEPLOY`/`HOLD` is not a
causal certification. `decision_certify.certify()` returns verdict `N/A` when there is no
per-customer randomized-arm structure; per-period actions are certified by the use-case gate
instead.

## Run the plugins

All of these read the frozen `E` and require `--db`, `--ckpt-dir`, `--tag`, `--as-of`.

**Supervised** (`looking_glass/supervised_gate.py`): held-out heads for churn, next purchase,
LTV, next value, return, and next channel/category/item. `run_supervised()` returns per-target
skill and `PASS`/`FAIL` (bootstrap 95% CI lower bound above the model-free baseline). Add
`--null-db`, `--null-tag`, `--null-ckpt-dir` to run the null control; `null_ok` requires every
null target to `FAIL`.

```bash
python3 -m looking_glass.supervised_gate --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> [--null-db <null_stream> --null-tag <null_tag> --null-ckpt-dir <null_dir>]
```

**Unsupervised** (`looking_glass/unsupervised_gate.py`): segmentation plus lookalike geometry.
`PASS` requires silhouette `> 0`, segment-value ANOVA `p < 0.05`, and lookalike enrichment
`> 1.0`. The same null flags are supported; `null_ok` requires the null verdict to be `FAIL`.

```bash
python3 -m looking_glass.unsupervised_gate --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> [--max-k 8]
```

**Decision / policy** (`looking_glass/use_case_gate.py`,
`looking_glass/white_queen_bridge.py`, `looking_glass/candidate_roster.py`):

```bash
python3 -m looking_glass.use_case_gate --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --actions freq,hour

python3 -m looking_glass.white_queen_bridge --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --horizon-days 90 [--truth <truth.json>]

python3 -m looking_glass.candidate_roster --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --action channel
```

**Custom plugin** (`looking_glass/plugin_contract.py`): fit, freeze, and persist on `E`:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from looking_glass.plugin_contract import SupervisedPlugin, fit_and_freeze

plugin = fit_and_freeze(
    SupervisedPlugin(HistGradientBoostingRegressor(), target="ltv"),
    E, y, "artifacts/plugins",
)
```

**The card** (`looking_glass/foundation_report.py`):

```bash
python3 -m looking_glass.foundation_report --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --out card.md [--fast] \
  [--white-queen --supervised --unsupervised] \
  [--null-db <null_stream> --null-tag <null_tag> --null-ckpt-dir <null_dir>]
```

`--fast` skips the bare/accumulator comparison U runs. `--full` belongs to
`scripts/apex_pipeline.py`; it adds white_queen, supervised, and unsupervised to the card.

For a full one-command demo on generated data:

```bash
python3 scripts/apex_pipeline.py --as-of <date> [--full]
```

## Checklist

- [ ] Canonical columns present: `customer_key` and `event_ts` non-null; `event_type`,
      `source_table`, and JSON `event_attributes` populated.
- [ ] Timestamps are UTC (onboarding normalizes them; `as_of_epoch` treats naive dates as UTC
      midnight).
- [ ] The receipt's `columns_mapped` and `warnings` were reviewed; no silent `"unknown"` event
      type or `0.0` value unless intended.
- [ ] Action keys declared (`action_keys` in the receipt) and passed as both `effect_actions`
      and `action_fields` to training.
- [ ] Reward event defined and consistent with `use_case_gate._period_rows` (default
      `purchase`).
- [ ] A randomized assignment or randomized holdout exists for OPE/decision certification.
- [ ] Training used point-in-time `--as-of` and an explicit `--out-dir`.
- [ ] Gates and the card ran on held-out B customers; real `DEPLOY`s were paired with null
      `HOLD`s.
