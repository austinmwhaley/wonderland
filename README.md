# wonderland — Unified Customer Decisioning System

An **offline-first** platform that turns historical customer data into
**next-best-action decisions** aimed at **long-term *incremental* gross margin**.

Deployment is **not assumed**. The system's job is to **train offline, analyze
offline, and prove — with randomized persistent holdouts — that it has earned the
right to act** (and to keep proving it). Nothing here requires live operation to
make the case.

Everything is one system with a **strict one-way flow**; each layer freezes a
versioned artifact and hands it downstream. Nothing reaches backward.

```
rabbit_hole ──▶ looking_glass ──▶ plugins ──▶ red_king ──▶ red_queen
 (A: stream)    (B: donor)        (C: heads)   (world model)  (NBA engine)
                                        └──▶ white_queen  (OPE certification)
                                        └──▶ caterpillar  (explain, read-only)
```

---

## The foundation embedding — headline evidence (v8)

**Goal (per stream):** one event stream in → one foundation model out. `embedding(customer, t)`
must be a **sufficient statistic of the history** — everything about the past that matters for the
future is in it; nothing that matters is lost.

**Architecture:** `E = [ learned SHAPE (self-supervised SSM) ‖ learned-compressed generic
FULL-HISTORY measure ]`, plus an in-net **anchored additive counter** (exact sums of raw quantities;
learned gating; content-addressed registers). The net does *selection* and *usage*; the counter does
*exact arithmetic* — **addition is a law, not something we ask the net to learn**.

**Metric:** `U = 1 − worst-case oracle-gap`, where the oracle is the **exhaustive full-history
measure** (**not** hand-built RFM), cross-checked by an independent sequence model; held-out
customers; **no leakage**; `SOLVED` ⟺ every target within the oracle's own noise.

**Certified results** (real, frozen consumed embedding):

| stream | bare (shape) | hand RFM | **this project** | verdict |
|---|---|---|---|---|
| instacart (hard) | 0.05 | 0.26 | **0.94** | ✅ SOLVED |
| ecommerce | 0.45 | 0.71 | **0.87** | ✅ SOLVED |
| **magnitude (controlled)** | **0.00** | **0.00** | **0.94** | ✅ SOLVED |
| L2 / L4 / L6 (easy) | ~0.95 | ~0.95 | ~0.95 | ✅ SOLVED (match) |
| L0 null (no signal) | — | — | **0.00** | ✅ correct (nothing to certify) |

**Controlled proof the measure is *necessary* (DEC-167):** on the `magnitude` stream — the latent is
written **only into an early transient + the future**, **activity is independent of it**, and
**histories ≫ the window** — a windowed encoder **even with RFM scores U = 0.00** (it cannot recover
a full-history cumulative the recent window does not contain), while the **full-history measure
scores U = 0.94, SOLVED**. It is a *causal*, by-construction demonstration, not a correlation.

**Not yet:** **decision-sufficiency (RL)** — the embedding is certified for **prediction**, not yet
for **counterfactual decisions** (`P(outcome | do(action), history)`).

---

## The layers

### A. `rabbit_hole` — Unified Customer Event Stream (ingestion)
Turns wide customer tables into one append-only event stream:
`[customer_key, event_ts, brand, event_type, event_attributes]` (+ entity, value).

- **Storage: Apache Arrow** (`.arrow`/`.feather`, uncompressed IPC) primary,
  memory-mapped/zero-copy; **DuckDB** for SQL; **Parquet** for archival. No SQLite,
  no pandas.
- Order economics (`revenue`, `cogs`, `gross_margin`) and email send/open/click.
- **Measurement design:** a **persistent holdout** — 5% of customer-periods receive
  **no marketing** — the randomized control that makes **incrementality
  identifiable offline**.
- Status: **implemented**; acceptance 25/25.

### B. `looking_glass` — Customer Foundation Model (representation)
A **frozen, self-supervised, action-free** sequence encoder (selective SSM) that
produces a per-customer state. Horizon-free successor features + JEPA + multi-task
objectives; company actions are **exogenous covariates** (never predicted).

- **Universal donor**: on the sufficiency battery it adds **unique** signal beyond
  raw features (signal 100 / standalone 100 / unique 62% (20k) and 75% (30k)
  customers) and
  **beats raw on 6/8 targets**.
- Publishes **state tables only**: `encoder_samples` (strict A/B split),
  `anchor_embeddings` (sample-B past-anchor states), `customer_state` (`h`+`as_of`
  for `fade()`/`absorb()` inference), `state_dense` (weekly states).
- Version = pre-1.0 semver `MAJOR.MINOR.PATCH` (code, e.g. `0.6.4`) + an `r`
  data-revision suffix → tag `<version>r<revision>` (e.g. `0.6.4r162610`); the
  historical `v6.x` line is now `0.6.x`; derived config; deterministic;
  vectorized training.
- Status: **implemented**; gate 15/15.

### C. `plugins` — Task heads (independent)
Each plugin reads the frozen state and computes its **own** target against
rabbit_hole. All pass their gates:
- **supervised** (CLV: point/two-part/quantile/baseline) 6/6,
- **unsupervised** (value-aware segmentation) 4/4,
- **white_queen policy** plugin 5/5.

### `white_queen` — Offline RL + OPE certification
Given logs, learns candidate policies and returns a **DEPLOY/HOLD** verdict with a
certificate `(value,[lo,hi],behavior)`; ships only if it can **reject
"not-better-than-logging"**. Hardened: **corroboration is required** (≥1 witness) —
no certificate-only deploys. 99 tests.

### `red_king` — Counterfactual world model (analyst tool)
A latent **RSSM** ("what happens if we act?") over frozen donor states, trained
causally (clipped-IPW) and **validated as a population estimator** (ordering
1.0, calibration 0.99). **Not in the decision path**: the decisive A/B
(`red_king/ab_witness.py`, 25 candidates against exact truth, with/without as
a white_queen MB witness) changed **zero** certified decisions — under the
pre-committed ship-or-delete rule it was removed from decisions and is kept
for offline analysis only (locked by tests). red_queen runs the validated
population path.

### `red_queen` — Next-Best-Action engine (the product)
Consumes the donor (+ optional red_king/white_queen) and emits decisions:
- **multi-cadence** (daily/weekly/monthly) and **multi-action** (counts + composition,
  spread within epochs);
- **constraint middleware** (send budget, per-customer caps — reject, not clamp);
- **fail-safe** (act only on positive incremental value);
- **certification-gated** (HOLD when nothing is certified);
- **uplift-targeted** (per-customer response model decides *whom* to market).
- Status: **implemented**; validated mode beats the logging policy.

### `caterpillar` — Interpretability (read-only)
Answers "why" from artifacts: recommended action, nearest customers in donor space,
provenance. v1.

### `eighth_square` — Standalone RL library
Owns the shared **`algorithms/`** and **`environments/`** (`wonderland/algorithms`
and `wonderland/environments` are symlinks into it). Single source of truth.

---

## The measurement that matters
- **Persistent holdout** ⇒ randomized control ⇒ **causal incrementality offline**.
- Current result: **ATE +11.62/period, 95% CI [11.19, 12.08]** (significant);
  per-customer uplift is **predictable** and monotone across predicted quintiles
  (−8.3 → +27.6); top-20% targeting gain **+18.95**.
- Every claim carries a CI; where evidence is weak we **fall back to population or
  HOLD**.

## Principles
See `AGENTS.md` (16 principles + v1 definition of done + repository/git workflow).
In short: no magic numbers, learn don't teach, adapt by construction, fail safe
(reject not clamp), uncertainty first, gate on ground truth, versioning = identity,
speed-first, and **honesty on unkind real-world data**.

## Repository & workflow
Source of truth: **https://github.com/austinmwhaley/wonderland** (`main`).
Code/specs/docs are committed; **data, checkpoints, artifacts, and venvs are
git-ignored** (regenerable). Workflow: `git pull --rebase` → change → `git add -A`
→ `git commit` → `git push`. No nested `.git`.

## Development
```bash
# provision (pip)
pip install -r requirements.txt -r requirements-dev.txt
# or with uv (lockfile: uv.lock)
uv sync
# quality gates (also enforced by CI on every push/PR)
pytest -m "not slow"       # inner loop: 202 tests in ~1 min
pytest                     # full suite (slow gates + acceptance included)
pytest --cov=white_queen.tribunal --cov-fail-under=80   # CI gate (80.7% source-only)
ruff format . && ruff check .
pre-commit install         # format + lint on commit
```
Dependency source of truth: `pyproject.toml` (`requirements*.txt` mirror it for
pip users). Indentation is **spaces** everywhere; `ruff format` is authoritative.

**Production operating loop** (monthly encoder + plugin retraining, daily state
job, weekly read-only inference): see [`RUNBOOK.md`](RUNBOOK.md) — the cadence,
commands, and hard rules (no peeking, forward-only states, read-only inference,
encoder-pin versioning).
`eighth_square` is standalone: `pip install -e eighth_square/` before importing
it as a package from outside its directory.

## Status at a glance
| Layer | Role | Status |
|---|---|---|
| rabbit_hole | stream + holdout design | ✅ |
| looking_glass | universal donor | ✅ |
| plugins | supervised / unsupervised / white_queen | ✅ |
| white_queen | OPE certification (hardened) | ✅ |
| red_king | counterfactual world model (analyst) | 🧰 analyst-only (A/B: 0 decision changes) |
| red_queen | multi-cadence NBA + uplift targeting | ✅ |
| caterpillar | interpretability | 🟡 v1 |

## Demo — one dataset, one report card

The demo is not a separate artifact. It is the project running on a synthetic
6k event stream, judged in the same **B (encoder alone) + B+C (plugins)** card:

```bash
# full run: generate data, freeze the encoder, recompute receipts, render the card
python3 scripts/run_demo.py --generate --build --refresh

# instant: re-render from the saved receipts
python3 scripts/run_demo.py
```

The saved card is checked in and renders instantly; a full rebuild is compute-heavy
(encoder training + GBM/XGBoost batteries). The dataset is deterministic for seed 17.

Rebuild the interactive page itself with `python3 scripts/build_demo_site.py`; the
tab payloads and regeneration order are documented in [`demo/README.md`](demo/README.md).

The output is [`REPORT_CARD.md`](REPORT_CARD.md). The full research audit
(including the deliberately open Layer-1 causal/IPS rows and unclaimed probes) is
[`docs/AUDIT_CARD.md`](docs/AUDIT_CARD.md).

## Frozen foundation artifact

```python
from looking_glass.foundation import FrozenFoundation

art = FrozenFoundation.load("/tmp/opencode/artifact_6k")
keys, state, receipt = art.transform_events(raw_events)   # raw canonical stream
state["shape"], state["measure"], state["full"]           # named views
```

`transform_events` applies the artifact's **frozen schema** — vocabulary, typing, and
periods are reused, never rediscovered from the serving batch. `fit`, `save`, `load`,
`describe`, and a content hash are part of the same interface. Non-finite state or a
missing `as_of` fails loudly rather than being sanitized.

State views: `shape` (learned, 128-d), `learned_state`, `forecast`, `uncertainty`,
`heads`, `measure` (lawful exact statistics), and `full`. Which view a consumer should
use is decided on training folds only (`scripts/select_views.py`).

## How to run (offline)
```bash
# generate the stream
python -m rabbit_hole.generators.generate_data --num-customers 25000 ...
# train the donor + state tables
python -m looking_glass.customer_foundation_model all --customers 25000 --anchors 6
# donor gate / sufficiency battery
python -m looking_glass.sufficiency_battery
# plugins
python -m plugins.run --window 365
# incrementality + per-customer targeting
python -m red_queen.response_model
```
