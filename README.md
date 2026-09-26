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
  raw features (signal 100 / standalone 100 / unique 75% at 50k customers) and
  **beats raw on 6/8 targets**.
- Publishes **state tables only**: `encoder_samples` (strict A/B split),
  `anchor_embeddings` (sample-B past-anchor states), `customer_state` (`h`+`as_of`
  for `fade()`/`absorb()` inference), `state_dense` (weekly states).
- Version = `vNrN` (`v` code, `r` data-revision hash); derived config; deterministic;
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
pytest -m "not slow"       # inner loop: 225 tests in ~1 min
pytest                     # full suite (slow gates + acceptance included)
pytest --cov=white_queen.tribunal --cov-fail-under=80   # CI gate (85% today)
ruff format . && ruff check .
pre-commit install         # format + lint on commit
```
Dependency source of truth: `pyproject.toml` (`requirements*.txt` mirror it for
pip users). Indentation is **spaces** everywhere; `ruff format` is authoritative.
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
