# rabbit_hole — Unified Customer Event Stream (Layer A)

The **data layer**. rabbit_hole's job **ends at producing the customer event
stream** — the single canonical, append-only table every downstream layer reads.
Nothing downstream (tokenization, embeddings, models) lives here; those live in
`looking_glass`.

It transforms **hundreds of wide customer tables** into:

```
customer_key, event_ts, brand, event_type, event_attributes
```

- Faithful raw: no labels, no rewards, no embeddings, no irreversible filtering.
- One row per event, ordered per customer by time (cross-domain ordering intact).
- `event_attributes` is a schema-evolved attribute map for the rest of the
  context — extension without a DDL per source.

## Contents

```
rabbit_hole/
  schema.py       canonical fields, COLUMN_ALIASES, EventRecord,
                  canonicalize_row, validate_event, parse_attributes
  stream.py       read_events / write_events / CustomerEventStream
                  (Arrow IPC primary + DuckDB + Parquet)
  generators/
    generate_data.py   THE generator: builds the wide customer tables, then
                       materializes them into the unified event stream
    instacart_stream.py  public-data FIXTURE: Instacart CSVs -> the same
                       canonical stream (see "Instacart fixture")
data/
  arrow/customer_event_stream.feather   the canonical stream (Arrow primary)
  duckdb/customer_event_stream.duckdb   the source DuckDB stream (Arrow feather is derived from it)
  instacart/                            fixture build (local-only, gitignored)
    src/*.csv                           downloaded source tables
    customer_event_stream.duckdb        canonical stream + `orders` contract table
    instacart_receipt.json              sha256s, counts, timeline receipts
  logs/
```

## The contract (persistent)

Every backend and every read carries the **five canonical fields**:

    customer_key, event_ts, brand, event_type, event_attributes

Extra columns are allowed and preserved (`event_id, entity_type, entity_id,
source_table, value`), but the five are always present. Backends by extension:
`.arrow`/`.feather`/`.ipc` -> Arrow (primary data layer), `.duckdb`/`.ddb` ->
DuckDB (query engine over the Arrow stream), `.parquet`/`.pq` -> Parquet
(archival). No SQLite, no pandas.

## Boundary

- **rabbit_hole ends here:** the customer event stream + the code that produces
  and validates it.
- **looking_glass owns everything downstream:** event-payload tokenization,
  the sequence/state-space backbone, task heads, evaluation, and the embedding
  vector store.
- **One generator, one stream:** `generate_data.py` is the only *generator*
  (wide tables -> `customer_events`); the retired direct simulators
  (`generate_full.py`, `gen_toy.py`) and their datasets are gone.
  `instacart_stream.py` is a *fixture adapter* (real CSVs -> same contract),
  not a second generator of synthetic data.
- **Deterministic:** the data window is a fixed reference date, so the same seed
  reproduces the same stream.
- **No duplication:** each artifact has exactly one home.

## Usage

```python
import rabbit_hole as rh

stream = rh.CustomerEventStream.load("data/duckdb/customer_event_stream.duckdb")
for row in stream:  # canonical rows
    attrs = rh.parse_attributes(row["event_attributes"])
    ...

rh.write_events("out.duckdb", rows)  # producer -> canonical stream
```

Column names vary across sources; incoming rows are canonicalized through
`COLUMN_ALIASES` (e.g. `customer_id`/`customer_key`, `timestamp`/`event_ts`,
`event_payload_json`/`event_attributes`).

## Instacart fixture (public real-data validation)

Runs the whole pipeline on real observational data — no arms/propensity/known
effect, so `generate_data.py` remains the known-truth fixture for causal/OPE
gates; this one validates layers A→C the way production would see them.

```bash
# 1. download (~197MB, no Kaggle auth needed for public datasets)
curl -sL -o /tmp/instacart.zip \
  "https://www.kaggle.com/api/v1/datasets/download/psparks/instacart-market-basket-analysis"
mkdir -p rabbit_hole/data/instacart/src && unzip -o /tmp/instacart.zip -d rabbit_hole/data/instacart/src

# 2. build (canonical stream + `orders` table; ~2.5 min, DuckDB out-of-core)
python3 -m rabbit_hole.generators.instacart_stream \
  --src rabbit_hole/data/instacart/src --out rabbit_hole/data/instacart

# 3. any layer can point at it
python3 -m looking_glass.customer_foundation_model train --customers 500 --anchors 6 \
  --as-of 2025-11-01 --db rabbit_hole/data/instacart/customer_event_stream.duckdb \
  --out-dir /tmp/insta_cfm
python3 -m plugins.head_template supervised_purchase_propensity_30d --as-of 2025-11-01 \
  --products /tmp/insta_cfm/cfm_products.duckdb \
  --stream rabbit_hole/data/instacart/customer_event_stream.duckdb \
  --out-dir /tmp/insta_plugin
```

Known properties (all receipted in `instacart_receipt.json`):

- **No absolute dates in the source** — timelines are reconstructed from
  `days_since_prior_order` + `order_hour_of_day` with a deterministic per-user
  end-stagger (`OFFSET_WINDOW=180`d before 2026-01-01).
- **Gaps are right-censored at 30d** (source maximum) — inter-order gaps are
  lower bounds; a 30d-forward purchase label gets its negatives from the
  staggered inactivity tails.
- **`order_dow` is ~14% consistent** with gap arithmetic — kept as an
  attribute only, never used for time.
- **No monetary data** — `orders.gross_margin` is NULL by construction: binary
  targets (purchase propensity) work; margin/CLV targets are unsupported.
- `products`/`aisles`/`departments` are joined wide into each
  `add_to_cart` event's attributes (product name, aisle, department; department
  also rides the canonical `brand` field).
