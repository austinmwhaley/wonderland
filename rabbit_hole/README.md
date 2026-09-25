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
data/
  arrow/customer_event_stream.feather   the canonical stream (Arrow primary)
  duckdb/customer_event_stream.duckdb   query-engine copy over the Arrow stream
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
  entity embeddings, the sequence/state-space backbone, task heads, evaluation,
  and the embedding vector store. (e.g. `tokenizer.py` is in looking_glass.)
- **One generator, one stream:** `generate_data.py` is the only generator
  (wide tables -> `customer_events`); the retired direct simulators
  (`generate_full.py`, `gen_toy.py`) and their datasets are gone.
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
