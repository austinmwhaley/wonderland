# White Queen

An offline reinforcement-learning and off-policy-evaluation (OPE) library that
takes a **bucket of logged data** and returns **offline RL models plus a
trustworthy DEPLOY / HOLD verdict** for each — with a confidence interval.

The product is one line:

```
(value, [lo, hi], behavior_value)  ->  ship iff lo > behavior_value
```

You point it at logs; it trains offline policies, decides which are confidently
better than the policy that produced the logs, and never ships one that is worse.

---

## What it does

1. **Ingest** any log format — a dict of arrays, Arrow, Polars, pandas, a
   DuckDB relation, Parquet/CSV, or a colony DuckDB file. Discrete or continuous
   actions; bandit or sequential.
2. **Train offline-RL candidates** on the pool: IQL, CQL, BC (and a
   continuous-action IQL for continuous logs). No live environment is needed.
3. **Evaluate** each candidate against the data with an OPE panel
   (FQE, model-based rollouts, DR/WIS, LSTDQ, ensembles).
4. **Certify** each candidate with a confidence interval and decide
   DEPLOY/HOLD by comparing the interval's lower bound to the behavior value.

Safety is a consequence of calibration: if the true value lies in `[lo, hi]`
and `lo > behavior`, the model is truly better. Recall is interval tightness;
abstention is an honestly wide interval.

---

## Core design

### Estimator contract (`ope/contracts.py`)
Every estimate is a tagged object: `(value, interval, support, estimand,
sources, reliable)`. Nothing in the decision layer sees a bare number.

- **Estimand** — what is estimated (`V(s0)`, marginalized `V(d_mu)`,
  advantage `E[(rho-1)A]`). The judge never compares across estimands.
- **Sources** — where the information comes from (`qnet`, `dynamics`,
  `weighting`, `td`). Two estimates may corroborate only if their source sets
  are **disjoint**. Soft-FQE and FQE-argmax share one Q net, so they are one
  source; DR shares the Q net with FQE, so its independent partner is the
  dynamics model.
- **Reliability** — out-of-range, diverged, under-budget, or low-support
  estimates are **rejected**, never silently clamped.

### Certificate (`ope/certificate.py`)
Builds the value interval from one representative per independent source
(FQE, dynamics, TD), plus the FQE ensemble spread, an overlap penalty for
unsupported candidates, and conformal width calibration. Decision:
`lo > behavior`.

### Decision (`ope/decide.py`, `judge.py`)
Superiority doctrine: DEPLOY is the default whenever "not better than
behavior" can be rejected by independent, estimand-matched evidence; HOLD
only when the evidence is insufficient (wide interval / no support /
disagreement).

---

## Install / requirements

- Python 3.12
- PyTorch (CUDA optional; GPU used when available)
- NumPy, DuckDB, PyArrow
- Gymnasium environments (`environments/`), plus the optional external OFFSET
  package for the OFFSET ladder (not vendored in this repo).

Tests need only CPU (a GPU is used automatically if present).

---

## Quick start

### One pool -> trained models -> decisions

```python
from white_queen.tribunal.ope import pipeline

report = pipeline.run(
    "path/to/logs.duckdb",  # any accepted source
    algorithms=("iql", "cql", "bc"),
    gamma=0.99,
    fast=True,
)

print(report["deployed"])  # candidates certified safe to ship
print(report["rank"])  # best-first by deployable value
for name, dec in report["decisions"].items():
    print(name, dec["deploy"], dec["certificate"])
```

### A single candidate

```python
from white_queen.tribunal.ope.api import evaluate

rep = evaluate(logs, my_policy, nA=4)  # logs: dict/Arrow/etc.
print(rep["deploy"], rep["bar"])
```

### Continuous-action logs

Register/point at a continuous policy object (with `action_mean` and
`log_prob_fn`). A continuous IQL trainer is bundled as `iql_cont`:

```python
from white_queen.tribunal.ope import pipeline

pipeline.run(continuous_logs, algorithms=("iql_cont",))
```

---

## Acceptance contract and scorecard

The library is measured against a falsifiable contract
(`white_queen/tribunal/bench/acceptance.py`) over a fixed environment suite
(one pooled log per environment):

- **Coverage** — fraction of cases where the true value lies in the reported CI.
- **Safety** — zero deployments of a worse-than-behavior model.
- **Precision / Recall** of the deploy decision.
- **Ranking** — top-ranked candidate is the true best.
- **Breadth / Rigor / Robustness / Engineering**.

Print the current scorecard:

```bash
python -m white_queen.tribunal.bench.scorecard --print --tests <N>
```

Run / refresh an environment (writes results under
`white_queen/tribunal/bench/results/`; override with `WQ_BENCH_RESULTS`):

```bash
python -m white_queen.tribunal.bench.scorecard --run cartpole --seed 0 --force
```

Cost control: select cases with `--tests` / `--seed`, reuse cached results
between runs, and let the library's autotune derive FQE budgets (no magic step
counts). The scorecard reads no other environment knobs.

The scorecard is the project's instrument: every estimator change is judged by
whether coverage stays at target and recall rises while false deploys stay 0.

---

## Tests

```bash
python3 -m pytest white_queen/tribunal/ope/tests -q   # from the repo root
# or bare `pytest` to run the whole repository suite
```

Covers: estimator contract, decision inference, certificate, judge, gate
receipts, data ingestion, schema adapters, continuous E2E, bandit E2E,
candidate registry, drift, sensitivity, model-based, autotune, cache,
training, and the benchmark harness.

---

## Repository map

```
white_queen/
  config.py            presets (quick / publication / smoke)
  db.py                colony DB schema (obs-dim agnostic)
  colony/              online behavior generation ("the colony")
  tribunal/
    ope/
      api.py           one-call evaluate(source, candidate)
      pipeline.py      pool -> train candidates -> evaluate -> decide
      data.py          agnostic ingest to canonical form
      schema.py        declared source-schema adapters
      contracts.py     estimator contract (estimand/source/reliability)
      certificate.py   the value certificate (the product)
      decide.py        decision as inference
      judge.py         gate/judge + receipts
      estimators.py    panel: FQE, DR/WIS, MB, LSTDQ, ensembles
      continuous.py    continuous-action OPE panel
      autotune.py      derived budgets + gate/bar resolution
      model_based.py   learned dynamics + rollout
      sensitivity.py   hidden-confounding sensitivity (Gamma*)
      behavior.py      propensity estimation
      receipts.py      bootstrap CIs/p-values, behavior stats
    bench/
      acceptance.py    the acceptance contract
      harness.py       ground-truth scoring (FP/FN/precision/recall/CIs)
      scorecard.py     run/reprint the contract table
  verdicts/            frozen artifacts per version (gitignored, regenerable)
algorithms/            online + offline algorithms (DQN, PPO, IQL, CQL, ...)
environments/          gym + custom env registry (OFFSET ladder envs need the
                       external OFFSET package — not vendored in this repo)
scripts/               surveys, benchmarks, frozen reruns
```

---

## Design principles

- **Truth-blind decisions.** The judge reads only offline quantities; it never
  reads ground-truth returns. Ground truth is used only to *score* the
  library, never to make a decision.
- **Independence is structural.** Corroboration requires disjoint information
  sources, not a hand-tuned witness count.
- **Reject, do not clamp.** An out-of-support estimate is invalidated, not
  substituted — silent clamping hides failures.
- **Fail safe.** No support -> wide interval -> HOLD, with an actionable
  diagnosis rather than a guess.
- **Everything is measured.** A frozen, reproducible ground-truth harness
  scores the system; the scorecard is the single source of truth on progress.

---

## Status

- Deterministic, reproducible benchmark (frozen per-environment logs).
- Safety is absolute on the measured suite: **zero false deploys**, precision 1.0.
- Ranking is correct (picks the true best model when it ships).
- Coverage reaches target under conformal calibration.
- Recall is the active work item: it withholds some genuinely-better models
  where the independent estimators disagree, and on a few large environments
  the full-budget measurement has not yet completed.

The remaining work is estimator accuracy (a calibrated, low-bias value
function and a reliable independent partner), tracked and measured by the
scorecard — not additional decision logic.
