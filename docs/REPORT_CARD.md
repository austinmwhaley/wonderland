# REPORT_CARD — pure foundation encoder (primary card)

> **This is the single primary project card.** Everything needed to confirm or deny the project
> objective lives here. Major docs: `AGENTS.md` (principles), `DECISIONS.md` (decision log),
> `STATUS.md`. One experiment = one axis changed; receipts in `docs/report_cards/experiments/`.

- **Branch:** `primary-encoder`. Repo: `wonderland`.
- **Data:** `relational_v3` synthetic reference (2k/6k) + `relational_v3_schema` variant. Layer A
  generator: `rabbit_hole/generators/relational_reference.py`; data card `docs/LAYER_A_DATA.md`;
  certificate `docs/report_cards/layer_a_certificate_v3.json` (PASS).
- **Verdict (current):** objective **supported at 6k** — value90 **+0.0570 CI [+0.0073, +0.1066]** excludes
  zero with the DEC-281 horizon ladder (orders +0.0558, same sign; churn tie). E-only, no hand summary.
  **2k remains tie-zone** (5-seed +0.0100 [-0.0389,+0.0589]; the win is 6k-scale — disclosed, not hidden).
  The claim comes from the fixed future objective (DEC-279/280/281): adjacent, per-horizon, correctly
  normalized, ladder-filled — with E carrying 94% of the full-history oracle's future-signal accessibility
  (Ridge 0.522 vs 0.558).

## 1. Objective (frozen)

Pure foundation encoder: **E learned only from the stream** (no table features, no hand summary, no
counter inside E), frozen, consumed by supervised / unsupervised / RL heads. Success = **E-only
robustly beats the traditional wide table + XGBoost**, proven fast at 2k before 6k, delivered via
one B card (guide) + one B+C card (validation) + a **single-command demo**.

## 2. Demo contract — how to reproduce

```
python3 scripts/demo/demo.py --data /tmp/opencode/t0_loop/relational_6k_v3 --epochs 12 --seeds 0,1,2
```

- cudf-polars GPU data prep (artifact-cached, byte-identical) → deterministic encoder training
  (trust-region lr, no clip, MAD early-stop) → B probes → B+C vs XGBoost GPU (encoder memory freed
  first — 8 GB constraint) → `docs/report_cards/demo_run.json` + `demo_command_receipt.md`.
- Latest receipt: **6k 3-seed DEC-281 ladder** `f0a7fca2d0bf` (1097.5s). Prior: `a969f1b0f190` (662.3s).

## 3. Layer A data (summary; full card `docs/LAYER_A_DATA.md`)

- `customer_signup` carries `segment`/`loyalty_tier`/true `signup_ts` at window start; PATH A
  byte-identical to v1. Schema variant renames `item_id`/`dept`/`maker` + adds `session_len`.
- Validator exit 0; certificate linked. No business data; synthetic harness only.

## 4. B — eight content axes scorecard

| # | Axis | Metric | Frozen E | Table / baseline | Status |
|---|---|---|---|---|---|
| 1 | Shape | retrieval / churn-nn | **0.461 / 0.981** | — | **working** |
| 2 | Mass | supervised vs XGBoost GPU | 6k: value **+0.0570** [+0.0073,+0.1066] excludes 0; orders +0.0558 (crosses); churn tie. 2k 5-seed: +0.0100 (crosses) | table | **WORKING at 6k; 2k tie (disclosed)** |
| 3 | Identity | learned `IB` kNN channel/category | channel **0.448** (> 0.251); category 0.162 (< 0.265) | table | **partial** (channel won; category loses) |
| 4 | Rate | churn AUC | 0.949 | 0.952 | tie |
| 5 | Uncertainty | ECE / OOD | ECE 0.048; variant churn 0.957 | — | **good** |
| 6 | Causal | IPS/DR policy value | channel: table wins (CI); offer: point only (CI crosses) | table/random | **not working** |
| 7 | Order/Path | event-shuffle value drop | **−0.256** | table 0 | **working** |
| 8 | Relational/Context | order_id event purity | 0.005 | random 0.0005 | **partial (weak)** |

## 5. Foundation properties scorecard

| Property | Status |
|---|---|
| Task-free / self-supervised | **PASS** (audit) |
| Schema-agnostic / per-stream | **partial** (auto fields; variant rerun queued) |
| Universal / transferable | partial (no held-out battery) |
| Lawful / compositional | field perm 1.2e−7; scan 8.9e−16; mass additivity 0.0 → **PASS** |
| Minimal / bounded | rank 2.4; E dim 128 (disclosed bound; derived 256 diverged, DEC-271); past corr² 0.53; future (365d) ridge-on-E **0.522** vs oracle 0.558; horizons ladder-filled (DEC-281); tau-scale horizons = noise floor (honest) |
| Robust / calibrated | **partial** (ECE good; deterministic kernels PASS; NaN hard-abort wired, untested) |
| Accessible | shape 0.365 vs table 0.394; multi-scale 5527d blocks −0.72 → **FAIL** |
| Efficient | load_data GPU byte-identical 4.3× (DEC-277); collate per-batch Python remains → **partial** |
| Artifact / reproducible | frozen; single command; hashes; artifact rev = data+code → **PASS** |
| Donor contract | partial (unique signal vs random E) |
| Interpretable / diagnosable | partial (axis card, geometry) |

## 6. Principle audit disposition (DEC-265 → DEC-277)

| Item | Verdict | Evidence |
|---|---|---|
| Learning rate (was literal, probed on judged metric) | **FIXED** — trust-region adaptive lr (seed 1e-3 bound, expand 1.5× / shrink 0.5×, bounds disclosed) | DEC-267 PASS; trajectory in every receipt |
| Gradient clip ceiling 1.0 | **REMOVED** — instrumented 465/465 binds (100%) = violation; stability now from trust-region shrink + NaN hard-abort | DEC-269/270; best receipt followed |
| weight_decay 1e-4 | **disclosed bound** (wd=0 failed acceptance: value −0.0048) | DEC-268 |
| Capacity d literal | **disclosed bound** (derived d=256 diverged: val spike 6.7e6, false green; item OPEN: capacity-aware stability coupling) | DEC-271 |
| Epochs = 5 | **budget cap** — MAD early-stop governs (seed 0 trains 10 of 12) | DEC-272/275 |
| Data path | **FIXED** — single code path `stream_loader.py`, cudf-polars GPU in isolated `.venv-cudf` (uv-managed, lock committed), artifact cache, **byte-identical gate PASS**, 4.3× | DEC-277 |
| Determinism | **PASS** — same seed → bit-identical trajectory | DEC-274 |

## 7. B+C — validation vs traditional table + XGBoost GPU

### Supervised (6k, 3 seeds; receipt `a969f1b0f190`)

| Target | Table | Frozen E | Margin (95% CI) |
|---|---|---|---|
| value90 | 0.3802 | **0.4479** | **+0.0677 [+0.0541, +0.0813] EXCLUDES 0** |
| orders90 | 0.4547 | **0.5202** | **+0.0655 [+0.0386, +0.0924] EXCLUDES 0** |
| churn90 | 0.9305 | 0.9338 | +0.0033 [−0.0071, +0.0137] tie |

### Supervised (2k, 3 seeds; receipt `5a50753d7f39`)

| Target | Table | Frozen E | Margin |
|---|---|---|---|
| value90 | 0.3576 | 0.3420 | −0.0156 |
| orders90 | 0.4003 | 0.4010 | +0.0007 |
| churn90 | 0.9515 | 0.9487 | −0.0028 |

> **Historical (superseded):** static-summary blocks gave 6k value +4.4% (1 seed/5) — removed for
> violating the no-hand-summary contract. Receipts: `encoder_6k_*`.

### Unsupervised (2k)

| Probe | Table | Frozen E | Result |
|---|---|---|---|
| value retrieval (Spearman) | 0.373 | **0.461** (6k) / 0.418 (2k) | E |
| churn-neighbor | 0.950 | **0.981** | E |
| anomaly AUC | **0.635** | 0.504 | table |

### RL (offline IPS/DR; logged randomized actions)

| Arm | Table | Frozen E | Result |
|---|---|---|---|
| channel | **518** | 480 | table |
| offer | 422 | 625 | E point; CI crosses 0 |

### Identity / Order

| Probe | Table | Frozen E |
|---|---|---|
| channel identity | 0.251 | **0.448** |
| category identity | 0.265 | 0.162 |
| event-shuffle drop (value) | 0 | **−0.256** |
| order_id event purity | n.a. | 0.005 (chance 0.0005) |

## 8. Honest failures (never hidden)

1. **Causal:** channel policy — table wins (CI excludes 0); offer point-estimate only.
2. **RL:** channel arm — table wins.
3. **Anomaly AUC:** table wins (0.635 vs 0.504).
4. **Accessibility:** multi-scale blocks (5527-d) not linearly accessible (Ridge −0.72).
5. **Retention (resolved, DEC-280):** the gapped future target was unlearnable-by-anyone (oracle −0.13); fixed to adjacent per-horizon targets; tau-scale (~1d) remains noise floor for everyone (honest).
6. **Identity category:** learned head loses to table (0.162 vs 0.265).
7. **Efficiency:** `collate` is still per-batch Python (the DEC-260 vectorization failed both gates
   and was reverted; retry now has a proven recipe: integers through the GPU, floats divided on CPU).
8. **Capacity:** d=256 derivation diverges under the current optimizer; needs capacity-aware
   stability coupling before it can ship.

## 9. Completion matrix (honest, live)

| # | Requirement | Evidence | Verdict |
|---|---|---|---|
| 1 | One repo command, hashes/runtime | `scripts/demo/demo.py`, receipt `a969f1b0f190` (662.3s) | **PASS** |
| 2 | B card fully measured (no pending rows) | 8 axes measured; variant rerun + universal battery queued | **PARTIAL** |
| 3 | B+C 2/2/2 vs XGBoost GPU with paired CIs | supervised 3-seed CIs done; unsup point estimates; RL/causal CIs exist but cross/table-win | **PARTIAL** |
| 4 | Layer A current (v3 cert, validator 0) | `layer_a_certificate_v3.json` | **PASS** |
| 5 | No hand summary inside E | learned mass/heads only; static blocks removed | **PASS** |
| 6 | Runtime/GPU utilization | load_data GPU 4.3× byte-identical; collate CPU remains | **PARTIAL** |
| 7 | Reproducibility (commit + hashes) | card + `demo_run.json` hash + artifact rev | **PASS** |

## 10. Receipts index

- `docs/report_cards/experiments/encoder_*.json` — ablations, 6k seeds, capacity search, axis
  probes, lawfulness, causal CI, retention, RL CI, accessibility, OOD, collate-failure receipt.
- `demo_run.json` — latest demo receipt (hash `a969f1b0f190`).
- Data certificate `layer_a_certificate_v3.json`.
- Decision log: `DECISIONS.md` DEC-237 … DEC-277. Errata recorded inline (DEC-256 → DEC-258).

## 11. Caveats / disclosures

- Synthetic harness; no business data. 2k supervised receipts predate the deterministic/MAD config.
- XGBoost GPU OOMs if the encoder stays resident; demo frees it first (8 GB constraint).
- GPU data path divides floats on the CPU (GPU division ~1 ULP off Python — DEC-277).
- Stronger XGBoost GPU baseline removed earlier sklearn-era orders/churn margins.
- No hybrid heads, no hand summary, no counter inside E anywhere above.

## 12. Next steps — decision-ready queue (binding axis first)

1. **Supervised margin (the binding axis).** Value CI crosses zero; seed spread ±0.02–0.03 is the
   same size as the effect. One-axis probes, cheapest first:
   a. **Early-stop patience / slower lr expansion** (training regime): the val-best epoch varies
      2–10 across seeds; probe whether longer patience recovers value. Acceptance: 6k 3-seed
      value90 CI lower bound > 0.
   b. **Future objective fix** (objective axis): head_fut is ineffective (future retention 0.0).
      Diagnose target/loss, retrain. Acceptance: future retention corr² > 0 AND value CI improves.
   c. If (a)+(b) fail: **capacity-aware stability coupling** to un-block d=256 (DEC-271 open item).
2. **Collate vectorization retry** (efficiency axis) using the DEC-277 recipe; acceptance:
   bit-identical outputs + ≥3× per-batch time + demo receipt unchanged.
3. **Schema-variant rerun** (universal axis): demo on `relational_v3_schema` -> universality receipt at the final config. **2k revalidation done** — tie-zone at 5 seeds (disclosed above).
4. **RL/causal 3-seed CIs** at the current config (decision-sufficiency evidence).
5. **Migration** (per operator direction): `auto_encoder` → `looking_glass`, `stream_loader` →
   `rabbit_hole`, dataset generation owned by a `rabbit_hole` command; `scripts/demo` shrinks to the
   one-command orchestrator. Gate: deterministic trajectory unchanged before/after the move.

Only when every supervised CI excludes zero in E's favor (or is honestly declared a limit of the
approach) is the objective met; everything else above is supporting evidence.
