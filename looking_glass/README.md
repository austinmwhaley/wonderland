# looking_glass

**sklearn for the new paradigm of machine learning: embedding models, state-space sequence models, and task heads. Point it at your business event stream and train churn, LTV, and any other supervised target — without writing a single feature engineering function.**

## Who This Is For

looking_glass is for **data scientists and data engineers** at companies that have a business event stream and want to build predictive models on top of it — without the traditional per-project feature engineering tax.

If your company records things customers do — clicks, orders, signups, support tickets, sessions, returns — you have an event stream. looking_glass turns that stream into a predictive system: churn models, LTV predictors, propensity scores, whatever supervised task your business needs next. The key is that you train the backbone once, and every new task is just a new head.

---

## The Problem With The Old World

The traditional approach to business ML looks like this: for every new prediction task, a separate project. Churn? A team writes rolling 30-day order counts, days-since-last-purchase, RFM scores, session depth averages. LTV? A different team writes different aggregations. Propensity-to-buy? Another set of rolling windows, another flat feature table, another round of judgment calls about which patterns are worth capturing.

Each of these projects carries the same original sin: **you are deciding in advance which patterns in the behavioral data are worth preserving.** A column called `rolling_30d_orders` throws away the rhythm, the timing, and the sequential context of the 30 orders it summarizes. The feature engineering becomes the ceiling on what the model can ever learn.

The old world is not just slow — it is structurally limited. You cannot engineer your way out of a lossy representation.

---

## The New Paradigm

The new paradigm starts from a different premise: **the event stream is already the rich representation.** The question is not which features to compute from it — it is how to train a model that can read it directly.

This is what state-space models make possible. An SSM trained on an event stream does not need hand-crafted features because it learns the sequential structure of behavior at every scale simultaneously. The model's internal state at each point in a customer's timeline is a learned compression of everything that happened up to that moment, calibrated to be causally predictive of what comes next.

Once you have that representation, every downstream task is just a head on top of it. You train the SSM once. Every new prediction task is a new head on the same shared backbone. No new feature engineering. No new data pipeline. No new modeling project from scratch.

This is the paradigm looking_glass implements.

---

## What looking_glass Is

looking_glass is a Python library with three composable primitives that implement this pipeline end to end: an embedding model for your dimensional entities, a state-space sequence model for your event stream, and supervised task heads for your prediction targets.

The analogy to sklearn is deliberate. Just as sklearn gives you `fit` / `transform` / `predict` as a stable interface over arbitrary estimators, looking_glass gives you three factory functions that each accept raw records and return learned representations or predictions. The models decide what signal matters. You describe the schema.

```python
from looking_glass import (
    create_embedding_model,
    create_temporal_core_model,
    create_supervised_model,
)
```

That is the entire public API you need to know.

All three factory helpers accept `sequence_backend="mamba2"` or `sequence_backend="samba"` when you want to choose between the pure recurrent and hybrid state-space backbones.

---

## The Three Primitives

### 1. `create_embedding_model` — Dimensional Representation

Before you can read event sequences meaningfully, every entity in those sequences needs a dense learned representation. What *is* a product? What *is* a customer? You need vectors that capture the attributes of each entity in a continuous space — so that similar entities cluster together and dissimilar ones are separated.

`create_embedding_model` does this via a self-supervised reconstruction objective. You give it a list of records — one dict per entity — and tell it which fields are categorical and which are numeric. It trains a model to compress each entity down to a fixed-size vector by trying to reconstruct all of its fields from that vector alone. No labels needed.

```python
embedding_model = create_embedding_model(
    id_field="product_id",
    categorical_fields=["category", "brand"],
    numeric_fields=["base_price", "unit_cost"],
    hidden_dim=128,
    epochs=4,
)

result = embedding_model.fit_transform(product_records)
# result.vectors: dict[product_id -> np.ndarray shape (128,)]
```

The output is a lookup table of vectors — one 128-dimensional float array per entity. This mirrors sklearn's `fit_transform` pattern exactly.

(Design note: the embedding model encodes each entity as a one-token sequence through the full ``TemporalStack`` + ``SequenceEngine``.  The temporal tracks are constant at t=0 and the SSM passes through trivially, so the cost is only a handful of linear projections — but it keeps one shared architectural backbone throughout the library so QDoRA, checkpointing, and the entity composition path are uniform.)

---

### 2. `create_temporal_core_model` — Sequence Representation

This is the central primitive. You feed it your unified event stream — every customer's activity in chronological order, with each event pre-enriched with the entity vectors from the previous step. The model reads those sequences and learns, event by event, what each moment in a customer's timeline means in context.

The training objective is next-event prediction with causal masking: given the first *n* events in a customer's timeline, predict the embedding of event *n+1*. This forces the model to build internal states that are causally meaningful. The hidden state at position *n* is a learned compression of everything that happened up to that moment.

```python
temporal_model = create_temporal_core_model(
    sequence_id_field="customer_id",
    event_id_field="event_id",
    timestamp_field="event_ts",
    categorical_fields=["event_type", "entity_type", "source_table"],
    numeric_fields=["value"],
    vector_fields=["customer_vector", "product_vector"],
    hidden_dim=128,
    epochs=6,
)

result = temporal_model.fit_transform(enriched_event_records)
# result.event_embeddings: dict[event_id -> list[float]]
# result.customer_records: list[{"customer_id", "core_last_vector", "core_mean_vector", "core_event_count"}]
```

The output is two things: a vector for every individual event (its meaning in sequential context) and a per-customer summary record containing both a last-state and mean-state vector. Those per-customer summaries replace the rolling-window features you would have otherwise computed by hand.

---

### 3. `create_supervised_model` — Task Heads

With a per-customer behavioral representation in hand, any supervised task becomes straightforward. You supply records annotated with outcome labels — churned yes/no, LTV amount, propensity score — plus whatever scalar fields and vector fields you want the head to consume. A task-specific head is trained on top.

The train/validation split is deterministic from `seed`, and categorical vocabularies plus normalization statistics are fit on the training rows only. Leakage control for time-dependent targets is handled by how you construct outcome rows upstream; the reference smoke pipeline does this with fixed history and future windows before calling `create_supervised_model`. For classification, the decision threshold is optimized on the training split to maximize F1, then evaluated on held-out rows.

```python
churn_model = create_supervised_model(
    task="classification",
    id_field="customer_id",
    target_field="churn_label",
    categorical_fields=[],
    numeric_fields=["order_count", "recency_days", "core_event_count"],
    vector_fields=["customer_vector", "core_last_vector"],
    epochs=30,
)

result = churn_model.fit_predict(outcome_records)
# result.predictions: dict[customer_id -> float]  (churn probability)
# result.report.metrics: {accuracy, precision, recall, f1, threshold}
```

No features. No joins. No rolling windows. The sequence representation *is* the feature — for every task.

---

## Embedding Tables Created as Byproducts

Every entity referenced in the event stream produces a dense, reusable embedding. These are not intermediate artefacts to throw away — they are valuable data products that other teams query directly from the vector store.

### Static Embeddings (pre-computed once, versioned, persisted to LanceDB)

These are trained via `create_embedding_model` on each entity's dimensional table. The vector captures the intrinsic attributes of the entity — what it *is*, independent of behaviour.

| Embedding table | Trained from | Resolved in the stream via |
|---|---|---|
| **Product vectors** | Catalog records (category, brand, price, margin) | `product_id` in `event_payload_json` |
| **Store vectors** | Store records (type, region, city, square footage) | `store_id` in `event_payload_json` |
| **Campaign vectors** | Campaign records (type, channel, discount, segment) | `campaign_id` in `event_payload_json` |

Each table is independently versioned and retrainable. Retraining product embeddings does not invalidate store or campaign vectors. All are resolved at tokenize time by the `PayloadSchema` and summed element-wise into the event token. A missing entity ID contributes the zero vector — the identity element under addition.

### Time-Dependent Embeddings (produced by the temporal core)

These are the Mamba backbone's output. They evolve with every new event a customer performs.

| Embedding | What it is | Stored as |
|---|---|---|
| **Event embeddings** | Each individual event's meaning in its sequential context — the encoded state at that point in the customer's timeline. | `event_id → 128-dim vector` in LanceDB |
| **Customer behavioral states** | Per-customer summary vectors: `core_last_vector` (recency-weighted, the state after the most recent event) and `core_mean_vector` (stability-weighted average across the full timeline), plus `core_event_count`. | `customer_id → {vectors, core_event_count}` in the outcome frame |
| **Point-in-time state records** | The customer state evaluated as of a specific timestamp, queryable via `get_state_as_of(customer_id, date)`. Enables downstream teams to ask "what did we know about this customer as of March 1st?" without leaky joins. | `(customer_id, as_of_ts, vector, backbone_version)` in LanceDB |

The customer state is the primary time-dependent output. It replaces the hand-crafted features you would otherwise compute — recency, frequency, monetary value — with a learned, high-dimensional summary that captures the same information plus sequential patterns those flat aggregates discard. External teams query it from the point-in-time store with a single call, exactly the way they would query an RFM table today.

---

## Input Data Contract

looking_glass accepts Python dicts. There is no required storage layer — bring data from Postgres, Snowflake, BigQuery, Spark, pandas DataFrames, or anything else you can turn into a list of dicts. The library only cares about field names and types.

This is the same contract as sklearn's array interface: sklearn expects `(n_samples, n_features)` arrays with consistent column ordering. looking_glass expects dicts with consistent field names. You are responsible for producing records that conform to the shapes below. The library handles everything from that point forward.

### Entity Records (for `create_embedding_model`)

One dict per entity — one per product, one per customer, one per whatever dimension you are embedding. Required field:

| field | type | description |
|---|---|---|
| `<id_field>` | str | unique identifier for this entity, passed as the `id_field` parameter |

All additional fields passed as `categorical_fields` or `numeric_fields` are consumed as features. The exact names are yours to choose — there is no required column naming convention.

### Event Records (for `create_temporal_core_model`)

One dict per event in your unified event stream. Required fields:

| field | type | description |
|---|---|---|
| `<sequence_id_field>` | str | groups events into per-entity sequences (e.g., `customer_id`) |
| `<event_id_field>` | str or int | unique event identifier |
| `<timestamp_field>` | str | ISO-8601 timestamp — used to sort events and compute inter-event time deltas |

All other fields are passed through. Before feeding events to the temporal core, you enrich each event with the entity vectors produced by `create_embedding_model`. Those pre-embedded vectors are what the temporal core reads as the semantic content of each event.

### Outcome Records (for `create_supervised_model`)

One dict per sequence (e.g., one per customer). Required fields:

| field | type | description |
|---|---|---|
| `<id_field>` | str | sequence identifier |
| `<target_field>` | int or float | label to predict: 0/1 for classification, continuous for regression |
| `<vector_fields>` | np.ndarray per field | pre-computed vectors from the embedding and temporal core stages |

---

## Architecture

### EntityCore — The Composition Primitive

`EntityCore` is the low-level wiring primitive for users who want to construct and train models directly rather than through the factory functions. All three components are defined by strict abstract base classes (`TemporalEncoderBase`, `SequenceEngineBase`, `TaskHeadBase`) — any component can be replaced with a custom implementation as long as it satisfies the interface contract.

```python
from looking_glass import EntityCore, TemporalStack, SequenceEngine, MultiTaskBusinessHead

model = EntityCore(
    temporal_encoder=TemporalStack(hidden_dim=256),
    sequence_engine=SequenceEngine(hidden_dim=256, recurrent_steps=4, backend="samba"),
    task_head=MultiTaskBusinessHead(hidden_dim=256, task_output_dims={"churn": 1, "ltv": 1}),
)

out = model(hydrated_events, delta_t)
# out.encoded_states:  (batch, seq_len, hidden_dim)
# out.task_outputs:    dict[task_name -> (batch, output_dim)]
```

---

### Tokenizer — Element-Wise Entity Resolution

Every event in the stream carries an ``event_payload_json`` field. The library's
tokenizer reads that payload and independently projects each signal source to
the model's hidden dimension, then sums them element-wise:

```
E_event = Σ E_cat + Σ E_num + Σ E_vector
```

* **Categorical projections** — payload keys like ``payment_method`` and
  ``device`` go through embedding lookups.
* **Numeric projections** — keys like ``order_value`` and ``rating`` are
  projected through a learned linear layer.
* **Vector lookups** — keys declared in ``PayloadSchema.vector_id_keys``
  (``product_id``, ``store_id``, ``campaign_id``) are resolved at tokenize
  time against their respective pre-computed static embedding tables. All
  resolved vectors are summed element-wise. A missing entity ID contributes
  the zero vector — the identity element under addition — so an
  ``email_send`` event without a product reference is handled naturally.

This factorization isolates the dependency between static entities and the
event stream: you can retrain store embeddings without touching the
product or campaign tables and without rewriting any historical events.
The stream carries IDs; the tokenizer resolves them at read time.

---

### TemporalStack — Making Time Legible

The gap from a page view to the next click might be 3 seconds. The gap from a last order to churn might be 180 days. A model treating these on a linear scale will be dominated by the large values and miss short-timescale dynamics entirely.

`TemporalStack` solves this through three parallel tracks that are fused together:

**Track 1 — Time2Vec:** Maps each delta to `[linear(t), sin(w₁t + φ₁), …, sin(wₙt + φₙ)]`. The linear term preserves magnitude ordering; the sin projections let the model learn periodic patterns at learned frequencies — weekly purchase cycles, monthly billing rhythms.

**Track 2 — TAPE:** Sinusoidal positional encoding keyed to absolute wall-clock time. Where Time2Vec encodes *how long it has been*, TAPE encodes *when in the calendar we are*. The model can learn that events near month-end or during holiday seasons behave differently.

**Track 3 — NeuralODE Latent Drift:** Between events a customer's latent state does not freeze — it drifts. A NeuralODE applies Euler integration over the inter-event interval, so a customer who went silent for 90 days arrives at their next event with a hidden state that has explicitly evolved through those 90 days of silence. This makes the model's sense of time continuous rather than discrete.

All three outputs are concatenated and projected back to `hidden_dim` via an MLP with a LayerNorm residual connection.

---

### SequenceEngine — Depth Without Parameters

`SequenceEngine` achieves depth through **weight sharing**: a single `UniversalTransformerBlock` is applied `recurrent_steps` times with identical weights at every step. This gives expressive depth without multiplying parameter count — the right tradeoff when data efficiency matters more than raw capacity.

It exposes two explicit state-space backends:

- **`mamba2`** — a pure recurrent Mamba-2 stack with residual feed-forward mixing.
- **`samba`** — a hybrid block that combines Mamba-style state dynamics, Flash Attention, and an FFN under recurrent weight sharing.

There is no GRU fallback. When `mamba_ssm` imports cleanly, the engine uses its optimized kernels; otherwise it uses an internal portable PyTorch state-space implementation of the same backends.

The shared block combines:

- **Mamba2 SSM layer** — selective state-space processing with linear-time sequence complexity.
- **Flash Attention** — `scaled_dot_product_attention` for global context mixing.
- **FFN with pre-norm and residuals** — pointwise position-wise transformation.

---

### QDoRALinear — Quantized Adaptation

`BusinessHead` and the shared adapter path inside `MultiTaskBusinessHead` use `QDoRALinear` to keep fine-tuning cheap when you build multi-task heads directly. The lightweight `ClassificationHead` and `RegressionHead` used by the factory helpers currently use standard MLP projections. QDoRA separates the base weight from the adaptation:

- **Base weight (frozen, 4-bit):** Quantized via `bitsandbytes`. Falls back to `nn.Linear` if unavailable.
- **LoRA update (trainable):** Low-rank decomposition `A @ B` — only the small factors update during fine-tuning.
- **Magnitude scaling (trainable):** A per-output learned scalar that controls the L2 norm of each output direction, decoupled from the direction itself.

The forward pass computes `base(x) + m * normalize(lora_update) * scaling` where `normalize` is per-sample direction normalisation of the adaptation delta. At initialization `lora_b` is zero, so the layer starts at exactly the base behaviour.

---

### Backbone Checkpointing

"Train the backbone once" requires the backbone to be serializable.  All three model classes expose `save_pretrained(path)` / `load_pretrained(path)` methods that persist the full training artefacts (weights, vocabularies, normalisation statistics) to a single file:

```python
core_model.save_pretrained("backbone.pt")
loaded = TemporalCoreModel.load_pretrained("backbone.pt")
# loaded.trained_core is ready for downstream heads

supervised.save_pretrained("churn_head.pt")
loaded_head = SupervisedModel.load_pretrained("churn_head.pt")
predictions = loaded_head.predict(new_customers)
```

The version tag defaults to a content hash of the actual weights (``derive_backbone_version``), so it can never drift silently.

### Daily-Cadence Updates

The pretrained backbone processes new events without weight changes.  ``incremental_forward`` seeds a saved prior state from the ``PointInTimeStateStore``, runs the frozen core over the new event window, and writes the updated per-entity state.  ``incremental_state_update`` wraps this into a single call:

```python
from looking_glass import incremental_state_update

incremental_state_update(
    core,
    store,
    entity_ids=["cust_1", "cust_2"],
    new_hidden_states=today_tensor,
    new_delta_t=today_deltas,
    new_valid_mask=today_mask,
    seed_as_of=yesterday,
    as_of_ts_list=[today_ts, today_ts],
    backbone_version="v1",
)
```

### Causality

FlashSelfAttention defaults to causal masking when no explicit mask is provided, so direct ``EntityCore`` compositions automatically respect next-event prediction semantics.

---

## Package Layout

```
looking_glass/
    interfaces.py     — TemporalEncoderBase, SequenceEngineBase, TaskHeadBase (strict ABCs)
    entity_core.py    — EntityCore: wires temporal encoder → sequence engine → task head
    temporal.py       — Time2Vec, TAPE, NeuralODELatentDrift, TemporalStack
    sequence.py       — Mamba2SequenceLayer, MambaBlock, SambaBlock, FlashSelfAttention, UniversalTransformerBlock, SequenceEngine
    tokenizer.py      — PayloadSchema, parse_event_payload (multi-entity vector resolution from event_payload_json)
    adaptation.py     — QDoRALinear, AdapterFusion
    heads.py          — BusinessHead, MultiTaskBusinessHead, ClassificationHead, RegressionHead
    embeddings.py     — create_embedding_model, EmbeddingModel, LanceDB persistence helpers
    temporal_core.py  — create_temporal_core_model, TemporalCoreModel, TemporalCoreOutputs, to_state_records
    supervised.py     — create_supervised_model, SupervisedModel, PredictionResults
    outcomes.py       — LabelSpec, build_outcomes, assert_no_leakage (leakage-safe label construction)
    state_store.py    — PointInTimeStateStore, InMemoryStateStore, LanceDBStateStore (as-of-correct vector queries)
    metrics.py        — roc_auc, pr_auc, lift_table, ranking_metrics (threshold-free evaluation)
    baselines.py      — GBTBaseline, compare (xgboost baseline for head-to-head comparison)
    checkpoint.py     — save_core_checkpoint, load_core_checkpoint, derive_backbone_version (core serialization)
    pretrain.py       — extract_final_states, incremental_forward, incremental_state_update, wrap_core_with_qdora (state save/load, LoRA retrofit, daily cadence)
    robustness.py     — RejectionSampler (batch-level data quality filtering)
```

The `scripts/` directory holds runnable examples against looking_glass: the
full-pipeline smoke test (``smoke_test.py`` — entry façade over
``smoke_pipeline.py`` / ``smoke_support.py`` / ``smoke_config.py``), the example
demo (``example.py``), toy training and sweep helpers, and the legacy benchmark
runner ``run_full.py`` (requires the retired ``simulacrum.db`` — its generator
was removed; rabbit_hole is the canonical data source). These are not part of
the installable package.

---

## Installation

```bash
python3 -m pip install -r requirements.txt
```

**Optional kernel dependency:**

- `mamba_ssm` — enables the optimized Mamba kernels used by both `mamba2` and `samba` when compiled successfully.

If you have a local CUDA toolchain with `nvcc` available, install the upstream package with:

```bash
python3 -m pip install torch>=2.1.0
python3 -m pip install mamba-ssm --no-build-isolation
```

If you do not have `nvcc`, the project still runs with its internal portable state-space implementation.

**Optional dependency:**

- `bitsandbytes` — enables 4-bit quantized base weights in `QDoRALinear`. Falls back to standard `nn.Linear` without it.

Device selection is automatic: CUDA → MPS → CPU.

---

## Example: End-to-End Run

The `scripts/` directory contains a runnable reference application of looking_glass against a synthetic retail dataset: 120,000 customers, 1,200 products, and 2,000,000 events spanning 3 years. The following is a stage-by-stage walkthrough of an actual run.

### Stage 1: Product Embedding

**Input:** 1,200 product records — id, category, brand, base price, unit cost.

**What happens:** The model compresses 1,200 products into 128-dimensional vectors via self-supervised reconstruction. Products in the same category and price bracket end up near each other in the vector space — not because you told it to cluster them, but because the reconstruction objective forces it to preserve that information.

**Output:** 1,200 vectors × 128 dimensions, one per product.
**Time:** 2.5 seconds on CUDA.
**Quality:** `mean_abs_cosine=0.714` — well-distributed, not collapsed.

---

### Stage 2: Customer Embedding

**Input:** 120,000 customer profile records — demographic fields, loyalty tier, income band, lifecycle stage, acquisition channel.

**What happens:** The same embedding architecture applied to the customer entity. The model learns a 128-dimensional representation for each customer from their profile attributes alone, before any behavioral data is considered. This becomes the "who is this person" signal that gets attached to every event they touch.

**Output:** 120,000 vectors × 128 dimensions, one per customer.
**Time:** 11.3 seconds on CUDA.
**Quality:** `mean_abs_cosine=0.207` — well-separated across 120K entities.

---

### Stage 3: Event Enrichment

**Input:** 300,000 raw event records from the unified event stream, with a cutoff timestamp to prevent label leakage.

**What happens:** Every event is augmented with two vectors: the customer's 128-dim profile vector and the product's 128-dim vector (on product-related events). This is a pure lookup — no model runs here.

**Output:** 300,000 enriched records carrying original fields plus pre-embedded entity context.
**Coverage:** 100% customer vector coverage, 100% product vector coverage.

---

### Stage 4: Temporal Core Training

**Input:** 300,000 enriched events grouped into 32,849 distinct customer timelines. Longest sequence: 57 events. Mean: 9.1 events per customer.

**What happens:** The model reads each customer's event history as a time-ordered sequence with explicit inter-event time deltas. TemporalStack encodes the time dimension across three parallel tracks (Time2Vec, TAPE, NeuralODE latent drift). SequenceEngine refines the representation through recurrent shared-weight passes. The loss is next-event prediction with causal masking.

**Output:** 300,000 per-event context vectors + 32,849 per-customer summary vectors.
**Time:** 91.7 seconds on CUDA. Training loss: 2.478. Quality gate: passed.

---

### Stage 5: Outcome Construction

**Input:** The 32,849 customer timelines and their temporal summary vectors.

**What happens:** A cutoff date splits behavioral history (before) from the outcome window (after). A customer is labeled churned if they had no qualifying activity after the cutoff within the label window. Target mode is `auto` — the pipeline detected that order-based churn labels were degenerate (98.4% churned; the outcome window post-dated most orders in a 3-year dataset) and automatically switched to event-mode targets.

**Output:** 32,159 eligible customers. Churn rate: 48.2% — a balanced, learnable split.

---

### Stage 6: Churn Head

**Input:** 32,159 customers — with the temporal summary vector, the customer profile vector, and baseline aggregate features (event count, recency, etc.) as a lightweight tabular complement.

The benchmark report (`run_full.py`) now prints a three-way ablation — vectors-only, aggregates-only, combined — alongside the xgboost baseline on aggregates.  This lets you see exactly what the learned representation contributes *on top of* and *instead of* hand-crafted features:

| Variant | Features | Purpose |
|---|---|---|
| vectors-only | `core_last_vector`, `customer_vector` | Test the core thesis — can the backbone replace feature engineering? |
| aggregates-only | RFM-style count/value/recency columns | Same information as the xgboost baseline; measures head quality when vectors are unavailable |
| combined | vectors + aggregates | The pragmatic production setting — the model gets everything and the xgboost serves as a lower bound |

| Metric | Value |
|---|---:|
| Accuracy | 79.1% |
| Precision | 73.1% |
| Recall | 90.2% |
| F1 | 80.8% |
| Threshold | 0.42 |

The model gets the learned backbone vectors plus a handful of generic aggregates (event count, recency days, etc.).  The ablation table in the benchmark report (`run_full.py`) breaks out how much the vectors contribute independently of those aggregates.

---

### Stage 7: LTV Head

**Input:** Same 32,159 customers, same vectors, continuous LTV labels.

| Metric | Value |
|---|---:|
| R² | 0.534 |
| RMSE | $764.60 |

---

### Overall Quality Determination

```python
{
    "ok": True,
    "product_embeddings": True,
    "customer_embeddings": True,
    "temporal_event_embeddings": True,
    "temporal_core": True,
    "churn_head": True,
    "ltv_head": True,
}
```

Total wall time from raw events to trained predictors: approximately 2.5 minutes on a single GPU.

See `scripts/smoke_test.py` (entry façade; pipeline implementation in
`scripts/smoke_pipeline.py`) for the full pipeline source.
