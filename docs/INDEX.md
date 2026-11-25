# INDEX — where everything lives

*The map of this repository. Doctrine is `AGENTS.md`; this file is orientation only. If the
two disagree, `AGENTS.md` wins.*

## Read in this order

| # | File | Why |
|---|---|---|
| 1 | `AGENTS.md` | doctrine: goal, principles, the trunk axis, the binding reflection, gates |
| 2 | `PLAN.md` | the goal cut backwards into atomic problems (generated from `scripts/plan_spec.py`) |
| 3 | `STATUS.md` | live state: stage, blockers, next actions |
| 4 | `DECISIONS.md` | append-only decision log (pre-registration → outcome → addendum) |
| 5 | `REPORT_CARD.md` | the single measurement surface (generated; SOLVED/FAILED/OPEN) |
| 6 | `AUDIT.md` | BACKWARD review + machine-readable requirements |
| 7 | `PROMPT_FORWARD.md` | the loop prompt (monotone frontier: beat the best-known record) |
| 8 | `PROMPT_BACKWARD.md` | the adversarial review prompt |
| 9 | `docs/SPINE.md` | the constitution: goal, consumers, judge, S1–S6, `U` contract |

## Code — the one-way pipeline

```
rabbit_hole  →  looking_glass  →  plugins  →  red_king  →  red_queen
                            ↘ white_queen (OPE)   ↘ caterpillar (explainability)
```

| Path | What it owns |
|---|---|
| `rabbit_hole/` | Layer A — canonical event stream (`schema.py`: five required fields, aliasing, canonical view SQL) |
| `looking_glass/onboard.py` | onboarding any source onto the canonical nine-column frame |
| `looking_glass/cfm_config.py` | `CFMConfig` — the only config surface (runtime toggles, derived resolution, tag signature) |
| `looking_glass/cfm_data.py` | stream reader, point-in-time cut, A/B split, anchors |
| `looking_glass/cfm_model.py` | `CFM` — the encoder; trunk swap-in point (`trunk='ssm'` default, `'looped_gpt'` opt-in) |
| `looking_glass/cfm_looped_trunk.py` | the Dory-style looped transformer trunk (DEC-613/651) |
| `looking_glass/cfm_training.py` | objectives, DWA balancer, governors, the training loop |
| `looking_glass/cfm_state.py` | frozen serve path + products (embeddings, anchors, ledger, outlets) |
| `looking_glass/universality.py` | the `U` certification metric |
| `looking_glass/*_gate.py` | supervised / unsupervised / decision / use-case gates |
| `plugins/` | Layer C — plugin contract, acceptance, ladders, white_queen bridge |
| `red_king/`, `red_queen/`, `caterpillar/`, `white_queen/` | decision, composition, explanation, OPE |
| `scripts/` | producers (measurements), renderers (card/plan), launchers (`safe_run.py`) |
| `eighth_square/` | a **vendored standalone RL library** (DQN/SAC/PPO/IQL/CQL/DT…), not part of the one-way pipeline; used by `white_queen`'s OPE tribunal candidates. Top-level `algorithms/` and `environments/` are symlinks into it. |
| `deploy/` | `Dockerfile.wonder`, `wonderland.slice.in` (the memory ceiling unit) |
| `demo/` | JSON fixtures used by the demo deployment path |
| `tests/` | the gates that keep all of it honest |

## Evidence — `docs/report_cards/experiments/`

| File | Producer |
|---|---|
| `encoder_win.json` | `measure_encoder_win.py` — B9.1: the encoder vs the 1-gram (the outcome gate) |
| `best_known.json` | `best_known.py` — the monotone frontier record (generated, only improves) |
| `scripts/build_port_artifact.py` | rebuilds the 307-dim delivery port from repo data (B3.2/B3.4) |
| `scripts/compute_policy.py` | the compute-tier policy + worker count |
| `judge_stages.json` | `measure_input_law.py` (S1/S2), `measure_dynamics_law.py` (S3), `measure_views_law.py` (S4), `measure_battery_law.py` (S5), `measure_certification_law.py` (S6) |
| `judge_leg_*.json` | `measure_bc_bank.py` — the four judge legs × two columns |
| `encoder_*.json` | `measure_b_props.py`, `measure_sufficiency_law.py`, `measure_robustness_law.py`, … |
| `lab/` | the DEC-611/615/633 development instruments (multi-domain, multi-tier) |
| `protocols/` | archived non-adopted runs |

A receipt is evidence only if its `_meta.behavior_hash` matches the current tree **and** its
population matches the demo manifest (`measure_encoder_win._population_check`).

## Canonical commands

```bash
# the one-command certification chain + card (S6)
python3 scripts/safe_run.py -- python3 scripts/measure_certification_law.py \
    --products looking_glass/artifacts/cfm_v2k_vf2_bank --demo looking_glass/artifacts/demo_v2

# the outcome gate (encoder vs the pooled 1-gram)
python3 scripts/safe_run.py -- python3 scripts/measure_encoder_win.py --head logistic

# the best-known record
python3 scripts/best_known.py            # report
python3 scripts/best_known.py --update   # promote strictly-better values

# docs
python3 scripts/plan.py                  # regenerate PLAN.md
python3 scripts/plan_spec.py --check     # PLAN.md must be current
python3 scripts/report_card.py           # regenerate REPORT_CARD.md

# gates (parallel: ~70 s instead of ~4 min)
python3 scripts/principle_audit.py --all --strict
ruff check . && ruff format --check . && pytest -m "not slow" -n 8
```

### Fast iteration vs certification — pick deliberately

| goal | command | cost |
|---|---|---|
| iterate on training | add `--set skip_products=true` | ~65 s (40 steps, 500 customers) vs ~183 s |
| test gate | `pytest -m "not slow" -n 8` | ~70 s |
| re-stamp one B row | `python3 scripts/measure_<name>.py` | seconds to minutes |
| **full certification** | the one-command chain | ~20 min, all 13 B producers included |

`skip_products=true` writes no served artifacts and prints *"iteration run, not a receipt"* —
never cite a number from such a run.

### Compute policy (`scripts/compute_policy.py`)

* **PyTorch/CUDA is mandatory** for training and every model forward.
* **joblib parallelism is the default** for independent fits (`LG_JOBS`, default 8).
* **GPU Polars is opt-in** (`LG_POLARS_GPU=1`): proven to run on this GB10 but measured 0.9× on
  the real demo query, so it is not the default — re-measure before adopting for a new shape.
* **DuckDB is CPU-only**: it has no GPU execution backend.

## Training a stream

```bash
# SSM trunk (default, the certified path)
python3 scripts/safe_run.py -- python3 -m looking_glass.customer_foundation_model train \
  --db <stream.parquet> --as-of <date> --anchors 6 --customers <N> --out-dir <dir>

# looped transformer trunk (opt-in; same downstream stack)
python3 scripts/safe_run.py -- python3 -m looking_glass.customer_foundation_model train \
  --db <stream.parquet> --as-of <date> --anchors 6 --customers <N> --out-dir <dir> \
  --set trunk=looped_gpt --set n_loops=2
```

`--customers` must match the demo manifest's population or the comparison receipt is written
`complete=false` — the population guard exists because two architecture campaigns were once
measured at 1/10 the population and produced meaningless verdicts.

## Vocabulary (the confusable pairs)

| Term | Meaning |
|---|---|
| `trunk='ssm'` | unified selective SSM (default; certified) |
| `trunk='looped_gpt'` | Dory-style looped transformer (attention, weight-tied latent loop) |
| `looped_recurrence` | **different**: a small weight-tied MLP block inside the SSM — a trunk-algebra flag, not the transformer |
| `bilinear_recurrence` / `drive_looped_recurrence` | the other two trunk-algebra flags (mutually exclusive with each other) |
| `bank` | the served 179-dim block object (`cfm_v2k_vf2_bank`) the judge reads |
| `E1.a/b` | served *signature* block (next-event / intent) |
| `E1.c` | delivery/ledger port (bank + measure PCA-128) — not a trunk-only capacity claim |
| M1 | the milestone "encoder beats the 1-gram" (achieved 2026-11-25) — a floor, not the finish |
| record | a value in `best_known.json` that every run must beat |

## Invariants (do not break)

1. Gates live in code (`scripts/b_spec.py`, `scripts/judge_spec.py`) and are frozen before a run.
2. `REPORT_CARD.md` and `PLAN.md` are generated; hand-editing either is a bug.
3. Receipts are regenerated, never edited; a stale hash is UNKNOWN, not PASS.
4. One GPU job at a time, under the 32 GB slice (`scripts/safe_run.py`).
5. `DECISIONS.md` is append-only; pre-register before the run, record the outcome after.
6. A run that only re-reads the same receipts is a thinking loop — record the negative once.
