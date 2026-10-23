# REPORT_CARD — pure foundation encoder (primary card)

> **This is the single primary project card.** Everything needed to confirm or deny the project
> objective lives here. Older cards (`docs/report_cards/b_card.md`, `demo_bc_card.md`,
> `demo_report_card.md`, `demo_command_card.md`, `report_card.md`) are superseded and now point
> here. Major docs remain: `AGENTS.md` (principles), `DECISIONS.md` (decision log), `STATUS.md`.
> One experiment = one axis changed; receipts in `docs/report_cards/experiments/`.

- **Branch:** `primary-encoder` (cut from `main` @ 3498bd5). Repo: `wonderland`.
- **Data:** `relational_v3` synthetic reference (2k/6k) + `relational_v3_schema` variant. Layer A
  generator: `rabbit_hole/generators/relational_reference.py`; data card `docs/LAYER_A_DATA.md`;
  certificate `docs/report_cards/layer_a_certificate_v3.json` (PASS).
- **Verdict (current):** objective **NOT yet met**. E wins retrieval/identity/order axes; with learned
  heads (the aligned config) supervised is tie-to-loss at 2k and 6k (6k 3-seed value −0.0022). The
  historical 6k +4.4% came from static summary blocks that violate the no-hand-summary contract and
  were removed. Every table win disclosed.

## 1. Objective (frozen)

Pure foundation encoder: **E learned only from the stream** (no table features, no hand summary,
no counter, no hand summary inside E), frozen, consumed by supervised / unsupervised / RL heads.
Success = **E-only robustly beats the traditional wide table + XGBoost**. Proven fast at 2k before
scaling to 6k. Delivered via one **B card** (guide) + one **B+C card** (validation) + a
**single-command demo**.

## 2. Demo contract — how to reproduce

```
python3 scripts/demo/demo.py --data /tmp/opencode/t0_loop/relational_v3 --epochs 5 --seeds 0,1,2
```

- Builds/validates v3, trains the frozen auto encoder (GPU), runs B probes, runs B+C heads vs the
  wide table + XGBoost GPU (encoder memory freed before XGBoost — 8 GB constraint), writes
  `docs/report_cards/demo_run.json` + `demo_command_card.md` with N, hashes, runtime.
- Latest receipts: 2k 3-seed `5a50753d7f39` (428.7s); 6k 3-seed `e94ab16cbf2b` (875.9s); 6k trust-region adaptive-lr `738880279c0b` (351.0s).

## 3. Layer A data (summary; full card `docs/LAYER_A_DATA.md`)

- `customer_signup` carries `segment`/`loyalty_tier`/true `signup_ts` at window start; PATH A
  byte-identical to v1. Schema variant renames `item_id`/`dept`/`maker` + adds `session_len`.
- Validator exit 0; certificate linked. No business data; synthetic harness only.

## 4. B — eight content axes scorecard

| # | Axis | Metric | Frozen E | Table / baseline | Status |
|---|---|---|---|---|---|
| 1 | Shape | retrieval / churn-nn / regime | retrieval 0.418; churn-nn 0.980 | — | **working** |
| 2 | Mass | supervised vs XGBoost GPU | 2k 3-seed: −0.016/+0.001/−0.003; 6k 3-seed learned heads: −0.0022/−0.0154/+0.0017 | table | **tie-to-loss (both scales)** |
| 3 | Identity | learned `IB` kNN channel/category | channel 0.335 (> table 0.249); category 0.177 (< table 0.260) | table | **partial** (learned head weaker than static 0.727) |
| 4 | Rate | churn AUC | 0.949 | 0.952 | tie |
| 5 | Uncertainty | ECE / dropout / OOD | ECE 0.048; variant churn 0.957 | — | **good** |
| 6 | Causal | IPS/DR policy value | channel: table wins (CI); offer: point only (CI crosses) | table/random | **not working** |
| 7 | Order/Path | event-shuffle value drop | −0.257 (learned) | table 0 | **working** |
| 8 | Relational/Context | order_id event purity | 0.005 | random 0.0005 | **partial (weak)** |

## 5. Foundation properties scorecard

| Property | Status |
|---|---|
| Task-free / self-supervised | **PASS** (audit) |
| Schema-agnostic / per-stream | **partial** (auto fields; variant card pending rerun) |
| Universal / transferable | partial (no held-out battery) |
| Lawful / compositional | field perm 1.2e−7; scan 8.9e−16; mass additivity 0.0 → **PASS** |
| Minimal / bounded | rank 2.4; E dim 128; past retention corr² 0.155; **future retention 0.0** |
| Robust / calibrated | partial (ECE good; divergence policy untested) |
| Accessible | shape E 0.365 vs table 0.394; multi-scale 5527d blocks −0.72 → **FAIL** |
| Efficient | GPU train/eval; collate CPU-bound (vectorization failed acceptance, DEC-261); lr now adaptive (DEC-267) → partial |
| Artifact / reproducible | frozen; single command; hashes; cert PASS → **PASS** |
| Donor contract | partial (unique signal vs random E) |
| Interpretable / diagnosable | partial (axis card, geometry) |

## 6. B+C — validation vs traditional table + XGBoost GPU

### Supervised (2k, 3 seeds, XGBoost GPU; latest receipt 5a50753d)

| Target | Table | Frozen E | Margin |
|---|---|---|---|
| value90 | 0.3576 | 0.3420 | **−0.0156** |
| orders90 | 0.4003 | 0.4010 | +0.0007 |
| churn90 | 0.9515 | 0.9487 | −0.0028 |

### Supervised (6k, 3 seeds/5 — learned heads, XGBoost GPU; receipt e94ab16cbf2b)

| Target | Table | Frozen E | Margin |
|---|---|---|---|
| value90 | 0.3802 | 0.3780 | −0.0022 (tie) |
| orders90 | 0.4547 | 0.4393 | −0.0154 |
| churn90 | 0.9305 | 0.9321 | +0.0017 (tie) |

> **Historical (superseded):** static-summary blocks gave 6k value +4.4% (1 seed/5), but static
> summaries violate the no-hand-summary contract and were removed; the learned-head config is the
> claimed E. Receipts: `encoder_6k_*`. Val loss plateau-noisy at 5 epochs → training regime is the
> next one-axis probe (DEC-263).

### Unsupervised (2k)

| Probe | Table | Frozen E | Result |
|---|---|---|---|
| value retrieval (Spearman) | 0.373 | **0.418** | +4.5 pt |
| churn-neighbor | 0.950 | **0.980** | +3.0 pt |
| anomaly AUC | **0.635** | 0.504 | table |

### RL (offline IPS/DR; logged randomized actions)

| Arm | Table | Frozen E | Result |
|---|---|---|---|
| channel | **518** | 480 | table |
| offer | 422 | 625 | E point; CI crosses 0 |

### Identity / Order (2k)

| Probe | Table | Frozen E |
|---|---|---|
| channel identity | 0.249 | **0.335** (learned; static was 0.727) |
| category identity | 0.260 | 0.177 |
| event-shuffle drop (value) | 0 | **−0.257** |
| order_id event purity | n.a. | 0.005 (chance 0.0005) |

## 7. Honest failures (never hidden)

1. **Causal:** channel policy — table wins (CI excludes 0); offer only point-estimate.
2. **RL:** channel arm — table wins.
3. **Anomaly AUC:** table wins (0.635 vs 0.504).
4. **Accessibility:** multi-scale blocks (5527-d) not linearly accessible (Ridge −0.72).
5. **Retention:** future objective ineffective (future retention 0.0); past corr² 0.155.
6. **Identity regression:** learned head underperforms the removed static pooled features.
7. **Efficiency:** `load_data` ≈58s + `collate` CPU-bound; GPU idles during data prep.

## 8. Completion matrix (honest, live)

| # | Requirement | Evidence | Verdict |
|---|---|---|---|
| 1 | One repo command, hashes/runtime | `scripts/demo/demo.py` (3-seed 428.7s, GPU XGBoost) | **PASS** |
| 2 | B card fully measured (no pending rows) | 8 axes have metrics; OOD/causal CI/lawfulness/access measured; 6k learned-head rerun pending | **PARTIAL** |
| 3 | B+C 2/2/2 vs XGBoost GPU with paired CIs | supervised 3-seed done; unsup point estimates; RL CI crosses | **PARTIAL** |
| 4 | Layer A current (v3 cert, validator 0) | `layer_a_certificate_v3.json` | **PASS** |
| 5 | No hand summary inside E | learned mass/heads only | **PASS** |
| 6 | Runtime/GPU utilization | collate CPU-bound; GPU idles | **FAIL** |
| 7 | Reproducibility (commit + hashes) | this card + `demo_run.json` hash | **PASS** |

## 9. Receipts index

All receipts: `docs/report_cards/experiments/encoder_*.json` (ablations, 6k seeds, capacity search,
axis probes, lawfulness, causal CI, retention, RL CI, accessibility, OOD). Data certificate:
`layer_a_certificate_v3.json`. Decision log: `DECISIONS.md` (DEC-237..DEC-258). Errata: DEC-256
claim was wrong (see DEC-258): the encoder had always been training; DEC-256's "fix" was cosmetic.

## 10. Caveats / disclosures

- Synthetic harness; no business data. 6k supervised is 1 seed/5 (3-seed pending on learned heads).
- XGBoost GPU OOMs if the encoder stays resident; demo frees it first (8 GB constraint).
- Stronger XGBoost GPU baseline removed earlier sklearn-era orders/churn margins.
- No hybrid heads, no hand summary, no counter inside E anywhere above.
