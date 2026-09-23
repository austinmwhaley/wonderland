# The Work — Unified Customer Decisioning System

A single, end-to-end platform that turns **raw customer data** into **next-best-action
decisions**, learned entirely **offline-first** (no live exploration until a small
controlled holdout is justified). Built for a large omni-channel retailer; designed
to optimize total company gross margin / LTV.

Everything here is one system with a **strict one-way flow**. Each layer freezes
its output and hands a versioned artifact to the next; nothing reaches backward.

```
        rabbit_hole            looking_glass          white_queen
  wide customer tables  ->  customer foundation  ->  offline policy learning
  -> unified event stream     model (embeddings)       + OPE certification
                                                            |
                                                            v
        red_queen         <-         red_king        <-  (policy candidates)
  next-best-action engine    counterfactual world model
  (hourly/daily/weekly)      (simulator)
```

---

## The layers

### 1. `rabbit_hole` — Unified Customer Event Stream (Layer A / ingestion)
Transforms **hundreds of wide customer tables** into **one unified customer event
stream**:

```
[customer_key, event_ts, brand, event_type, event_attributes]
```

- Faithful raw, append-only, one row per event, ordered per customer by time.
- Identity resolution, canonical taxonomy, source QC (landing checks only — no
  cleaning here; cleaning lives downstream and is versioned).
- **Data layer: Apache Arrow** (`.arrow`/`.feather`, uncompressed IPC) is the
  primary stream — memory-mapped and zero-copy into Polars/DuckDB; **DuckDB** is
  the SQL query engine over it and **Parquet** is for compressed archival. No
  SQLite, no pandas.
- Storage + serving with **snapshot pinning** so any downstream version can
  recreate exactly the rows it trained on.
- Status: **implemented** — `rabbit_hole/` holds `schema.py`, `stream.py`, the
  generators, and the event-stream data (`data/arrow/`, `data/duckdb/`).
  rabbit_hole *ends* at the event stream; everything downstream lives in
  looking_glass.

### 2. `looking_glass` — Customer Foundation Model (Layer B)
Builds a **customer foundation model** over the rabbit_hole event stream: a
sequence encoder (SSM / Mamba-2) that produces a customer state embedding

```
S_c(t) = embedding(customer, history <= t)      # 128-d, frozen, versioned
```

- Pre-trained (masked-event + contrastive), frozen after release, served daily as
  `customer_embedding_daily` (+ a private wide state cache).
- Includes automated **HVA mining** (high-value-action discovery) beside the model.
- Downstream consumers read **versioned embeddings via API only** — the foundation
  is bit-identical for every consumer; no fine-tuning, no backward edges.
- Status: **implemented** at `wonderland/looking_glass/` (specs preserved in
  `wonderland/looking_glass/specs/`). It **points at rabbit_hole** for the event stream:
  `data/{arrow,duckdb,logs}` and the generator are symlinks into `rabbit_hole/`,
  and it reads the **Arrow** stream (`data/arrow/customer_event_stream.feather`)
  zero-copy by default. Tokenization and all model code live here, not in
  rabbit_hole.

### 3. `white_queen` — Offline RL + Off-Policy Evaluation (learning layer)
Given a **bucket of logs**, trains offline RL policies and returns **models plus a
trustworthy DEPLOY / HOLD verdict with confidence intervals**. The product is one
line:

```
(value, [lo, hi], behavior_value)  ->  ship iff lo > behavior_value
```

- Ingests any log format (dict / Arrow / Polars / pandas / DuckDB / Parquet / CSV /
  DB); discrete or continuous; bandit or sequential.
- Trains IQL / CQL / BC (and continuous IQL), evaluates with an OPE panel
  (FQE, model-based rollouts, DR/WIS, LSTDQ, ensembles), and certifies each
  candidate — **never ships a model worse than the logging policy**.
- Measured against a frozen, reproducible **acceptance contract + scorecard**
  (coverage, safety, precision/recall, ranking).
- **Generalizable by design**: works on any logs. Intended to optionally consume
  `looking_glass` embeddings as the state representation, while remaining usable
  standalone on arbitrary logs.
- Status: **working** (99 tests; deterministic benchmark). See `white_queen/README.md`.

### 4. `red_king` — Counterfactual World Model (simulator)
A learned **dynamics + reward model** of the customer world: given state and
action, predict next state and outcome. Bootstrap-ensemble, uncertainty-aware,
used for **counterfactual evaluation** ("what would have happened if…") and
**model-based offline RL** (MOPO-style pessimistic rollouts).

- Enables learning and comparing policies where logged action coverage is thin and
  confounded — the core reason this platform is model-based first.
- Status: **planned / empty**.

### 5. `red_queen` — Next-Best-Action Engine (tip of the pyramid)
The decision engine that consumes the learned value/world models and emits the
**next best action per customer**, handling **multiple decision cadences**
(hourly, daily, weekly) and hard business constraints.

- Orchestrates policy heads (HVA selection first; personalization and delivery
  after), enforces constraint middleware, and produces the deployable decision.
- Status: **planned / empty**.

### 6. `caterpillar` — Interpretability Engine
A **plain-language interface over the whole system**. You ask questions in
natural language and it answers them using the parts we have built — the event
stream, the foundation-model embeddings, the learned policies, the world model,
and the OPE results.

- Examples: *"Why was this customer sent this action?"*, *"What drove the lift
  for this segment?"*, *"What happens if we raise the email cap?"*, *"Why did the
  model HOLD this policy?"*
- Answers are grounded in the platform's own artifacts and receipts (estimates,
  intervals, certificates, embeddings, counterfactuals) — not invented.
- Sits **beside** the pipeline (reads everything, changes nothing).
- Status: **planned / empty**.

---

## Shared principles (from the specs)

- **One-way flow:** data → representation → policy → world model → decisions.
  Downstream never reaches upstream; new needs become new upstream versions.
- **Frozen foundation:** the base encoder is immutable per release and consumed
  via a versioned API. Consumers add small heads, never fine-tune the base.
- **Defensive by default:** assume NaN/Inf/out-of-distribution; fail safe.
- **No ML fallback:** neural failure triggers infrastructure rollback to
  business-as-usual, never heuristic degradation.
- **Split train vs. evaluation rewards:** train on de-biased observed margin;
  evaluate on **incremental counterfactual margin**.
- **Offline-first:** prove value offline (temporal replay + rollouts + OPE +
  sensitivity + constraint audits) before any live pilot.
- **Everything is versioned and measured:** data snapshots, model versions,
  embeddings, and the acceptance scorecard are all pinned and reproducible.

## How the pieces connect

- `rabbit_hole` produces the **event stream** (the ground truth of behavior).
- `looking_glass` turns it into **frozen embeddings** (the state).
- `white_queen` learns and certifies **offline policies** from logs — consuming
  embeddings when available, generic logs otherwise.
- `red_king` provides the **counterfactual simulator** for model-based learning
  and evaluation.
- `red_queen` composes the final **next-best actions** across cadences and
  constraints.

## Shared libraries

`algorithms/`, `environments/`, `OFFSET/`, and `scripts/` are shared runtime
dependencies used by `white_queen` (and later `red_king`/`red_queen`). They live
at the `wonderland/` root so any component can import them (`from environments.registry
import make_env`, `from algorithms....`).

## Status at a glance

| component   | role                          | status              |
|-------------|-------------------------------|---------------------|
| rabbit_hole | wide tables → event stream    | implemented         |
| looking_glass | customer foundation model   | implemented (CFM validated) |
| white_queen | offline RL + OPE certification | working (99 tests) |
| red_king    | counterfactual world model    | planned (empty)     |
| red_queen   | next-best-action engine       | planned (empty)     |
| caterpillar | interpretability engine       | planned (empty)     |
