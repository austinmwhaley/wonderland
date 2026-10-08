# looking_glass — Architecture & Design Card (Layer B)

> A self-governing, self-supervised **foundation encoder for event streams**.
> This card is written to be self-contained: given it, an LLM should be able to
> re-implement the system and understand *why* every part exists.
> Canonical source: `wonderland/looking_glass`. Version: v4.6 (DEC-035).

---

## 0. One-line purpose

`looking_glass` reads a raw customer **event stream** and produces a **frozen
per-customer state / embedding** — the universal donor feature table every
downstream model trains on. It is judged **only on self-supervision quality
(Layer B)**, never on a downstream label.

System flow (one-way):
```
rabbit_hole → looking_glass → plugins → red_king → red_queen
                     │
                     ├── white_queen (OPE certification)
                     └── caterpillar (read-only interpretability)
```

---

## 1. Philosophy (the doctrine that governs every decision)

**The one rule:** *Never encode an answer the system could learn or derive.*
Every threshold, rate, cap, window, dimension, and weight must be
**(a) inferred from the data, (b) derived from problem size/hardware, or
(c) learned.** A surviving literal is only a *documented conservative fallback*,
overridable and recorded in a receipt.

**Pillars**

1. **Invariants as native architecture, not post-hoc patches.** Smoothness,
   isotropy, stability are enforced *in the optimization dynamics* (barrier
   functions, orthogonal gradient projections), not bolted on after a failure.
2. **Capacity over artificial constraints.** The trunk keeps its full width
   (`dim=256`). Never shrink the model to make a gate pass; fix the objectives
   so the trunk uses its capacity.
3. **Truth-first diagnostics.** Gates are honest, scale-free inspectors. A
   failing gate is an objective signal of a task/geometry tension — never
   hidden or bypassed.
4. **Zero-leakage separation of concerns.** Fast event dynamics and slow
   long-horizon trends must not interfere; layers flow strictly one way.
5. **Fail safe, reject, do not clamp.** Invalid/OOD results are refused or carry
   widened uncertainty — never silently substituted.
6. **Measure everything; gate on ground truth.** No capability is "done" without
   an independent, reproducible validation gate with a null baseline.
7. **Reproducibility is identity.** A change that can alter outputs bumps the
   version; every artifact records inputs, config, and version.
8. **Cost-aware & fast.** Vectorized/columnar by default; derive budgets from
   data + hardware; every component emits a runtime receipt.

**Anti-patterns → preferred:** fixed epochs → converge on held-out; hardcoded
threshold → learned/calibrated; per-stream `if/else` → schema/registry dispatch;
static gate constant → scale-free statistic; selecting weights by the training
metric → held-out selection.

---

## 2. The encoder — `CFM` (customer foundation model)

### 2.1 Input: per-event token

Each event becomes a fused embedding (width `chan = dim / n_experts`; for the
unified trunk `chan = dim`):
```
x_t = emb_et(event_type) + emb_brand(brand) + emb_ent(entity_type)
      + w_val(value) + w_dt(log1p(Δseconds)) + w_co(covariates)
```
- `event_type, brand, entity_type/entity_id` → categorical embeddings.
- `value` (log-monetary / magnitude), `dt` (inter-event seconds) → learned
  linear projections.
- `dt` is *also* what makes the recurrence continuous-time: the decay is
  input-dependent, so non-uniform gaps are handled natively.

### 2.2 Trunk: unified multi-timescale **selective SSM**

One wide state-space model, recurrences solved by a **vectorized O(log T)
affine prefix scan** (`_scan`, Hillis–Steele):
```
δ_t   = softplus(W_δ x_t + δ_bias)      # per-channel, input-dependent, ≥0
decay = exp(−δ_t)                        # ∈ (0,1) — a strict contraction (stable by construction)
h_t   = decay ⊙ h_{t−1} + (1−decay) ⊙ (W_B x_t)
y_t   = W_C h_t
```
- **`δ_bias` is a per-channel LEARNED vector** initialised as a log-uniform
  spectrum from fast (short memory) to slow (long memory). One trunk carries all
  timescales; there is **no hand-split expert bank** (the early v3.0 dual-expert
  scaffold was removed).
- **Stability is structural**: `decay = exp(−softplus(·))` is always in `(0,1)`.

### 2.3 Input low-pass (stream-profiled spectral bottleneck)

Before the recurrence, the token passes through a **per-channel learned EMA
low-pass** (`IntentFilter`, solved by the same affine scan):
```
x̃_t = r ⊙ x̃_{t−1} + (1−r) ⊙ x_t
```
- `r` (retention) is a **learned per-channel parameter**, initialised from the
  **stream's own statistics**: the content change probability `p` gives an
  average run length `L = 1/(1−p)`, and the slow-band retention is
  `exp(−ln2 / L)` (`stream_lowpass_retention`).
  - Layer A (77% type-change) → `r ≈ 0.85` (strong smoothing)
  - Instacart (19%) → `r ≈ 0.57`
- **Why:** a state that consumes raw per-event content is a first-order filter of
  an alternating signal, so its *velocity direction* flips every step. The
  low-pass converts discrete spikes into continuous local **rate densities**, so
  the slow state is a smooth trend that still carries long-horizon content
  information (which `agg` needs).
- **No binary feature partition.** Content and trend are NOT routed to different
  bands by hand; every feature enters every band and is low-passed by its own
  learned cutoff.

### 2.4 Band isolation

`W_δ`/`W_B` are made **block-diagonal between the fast and slow timescale bands**
(the split is derived from the *learned* `δ_bias` spectrum, not a feature list).
This keeps the slow state from receiving high-frequency content through the
mixing weights, so slow-state continuity is **structural**.

### 2.5 Donor boundary (the consumed representation)

```
z = ZCA( proj(h) )      # proj: Linear(dim,dim); whitening applied AFTER proj
```
- `proj` is a learned linear readout; whitening is applied **after** it (the true
  boundary — applying it before `proj` re-collapses rank, because `proj` is
  ill-conditioned; measured).
- **Global-EMA ZCA** (Newton–Schulz): maintain an EMA of the projected mean and
  covariance; approximate `Σ^{−1/2}` with the coupled Newton–Schulz iteration
  `Y←½Y(3I−ZY), Z←½(3I−ZY)Z`, **spectral-norm normalised** (power iteration) and
  **eps-ridged** so the condition number is bounded. Frozen at inference;
  train/serve share one map.
- **Entropy-weighted EMA rate**: the ZCA update rate scales with the batch's
  spectral entropy, self-stabilising against bursty activity windows.
- Donor readout feeds all downstream plugins as `donor_embeddings` /
  `state_embeddings` (read-only).

### 2.6 Inference: constant-time state advance

The recurrence state is the public representation. `fade/absorb` advance every
customer to an anchor time in seconds (constant-time per customer) — no full
re-forward. Materialised daily by `daily_states.py`.

---

## 3. Objectives (self-supervised)

All objectives are trained jointly. Weights are **learned** (DWA, see §5). The
`objectives` tuple:

| objective | forces into the state |
|---|---|
| `next` | next event **type** (cross-entropy) |
| `entity` | next **entity type** (cross-entropy) |
| `dt` | next inter-event **time** (log1p MSE) |
| `value` | next event **value/magnitude** (log1p MSE) |
| `occur` | whether the next event occurs within a data-derived horizon |
| `order` | temporal **order** of event vs a random alternative (BCE) |
| `mask` | reconstruct a masked event's type (masking rate from config) |
| `contrast` | InfoNCE over projected states (uniformity) |
| `redundancy` | off-diagonal covariance penalty (decorrelation) |
| `jepa` | latent future prediction from the context state (EMA target encoder) |
| `sf` | successor features: discounted future value/count at sampled γ |
| `query` | read the **faded** state at an arbitrary time between events (the serving path) |
| `agg` | exact multi-horizon window **count + value-sum** (long-horizon integration) |
| `variance` | per-dim std floor (VICReg hinge; treats scale collapse) |
| `rank` | participation-ratio pressure (PR = tr(C)²/‖C‖²) |
| `spectrum` | soft-spectrum isotropy: penalise `var(log per-dim variance)` |
| `volume` | log-det trunk-volume barrier (trace-normalised) |
| `iso` | large-sample isotropy: PR loss on **per-step** states (B·T rows) with a self-throttling barrier |
| `trajectory` | slow-band smooth-velocity: penalise `‖Δ²h_slow‖²` |
| `ortho` | fast/slow cross-covariance → zero interference (scale-free, standardized) |

**Why these, not fewer:** each targets a distinct statistic raw RFM hand-feeds
(list price, churn timing, category affinity, occurrence, order, long-horizon
sums…). The ensemble is the "portfolio"; a combination map lives in
`specs/objectives_catalog.html`.

---

## 4. Invariants (enforced, not just measured)

| invariant | mechanism |
|---|---|
| recurrence stability | `decay = exp(−softplus(·)) ∈ (0,1)` — structural |
| slow-trajectory continuity | per-channel input low-pass (stream-profiled) + block isolation + `trajectory` loss |
| manifold isotropy | `iso` barrier (hinge below `τ_mp`, self-throttling multiplier) + `spectrum` + `volume` |
| trunk volume | GeometryBank log-det barrier on the population (8192-state FIFO) |
| expert independence | `ortho` cross-covariance (fast vs slow halves) |
| task/geometry non-interference | **task-structural PCGrad** (§5) |
| OOT stability | principal-angle subspace overlap (dimensionless) |
| reproducibility | deterministic init (`seed_everything`), frozen artifacts, receipt identity |

---

## 5. Optimization engine

- **DWA (Dynamic Weight Average)** — default weighting. Scale-free: each loss is
  divided by its own EMA, so weights respond to *improvement rates*, not raw
  scale. (`weight_mode="dwa"`, `dwa_temp=2.0`.)
- **Task-structural PCGrad** — objectives are split into **task** (predictive
  skills) and **invariant** (`GEOMETRY_FAMILY` + `trajectory`). When the
  invariant gradient conflicts with the task gradient, it is projected onto the
  null space of the task gradient — structural pressure can refine the
  representation but **can never reduce predictive skill**.
- **Adaptive Information Bottleneck** — a bounded compression term (variance-ratio
  rate) whose multiplier is tuned by **dual ascent** on the batch participation
  ratio (raises compression when rank exceeds target, eases when it falls).
- **GeometryBank governor** — an 8192-state FIFO of projected population states;
  a log-det barrier + redundancy penalty with a closed-loop λ (PID-like,
  EMA-damped) pushes the population rank toward its target.
- **Grouped gradient** — the non-AMP path computes two grouped backwards
  (task / invariant) rather than 13 pairwise.

---

## 6. Evaluation gates (scale-free, per-stream)

### 6.1 Layer-B portfolio (`portfolio.py`)

- **Structure-skill** for each *gated* objective: the held-out self-supervised
  loss on **real** data vs **destroyed** data (column/order-destroyed), grouped
  cross-validation; skill = `loss_destroyed − loss_real` (positive ⇒ uses
  structure), reported with fold SE. Gated exclusions (no structure-skill):
  `contrast, redundancy, variance, spectrum, volume, iso, trajectory`.
- **Geometry gate (self-calibrating):** the consumed representation's
  participation-ratio fraction `PR/dim`, compared to a **coverage fraction of the
  stream's own achievable whitened PR** (from the checkpoint's whitening
  receipt). Per-stream, so a naturally lower-dimensional stream is not punished.
  Reports the white-noise Marchenko–Pastur null for reference.
- **Canary probes:** the frozen state must NOT predict assignment/arm or period
  bucket (< 1.5× chance) — a leak detector.

### 6.2 Intrinsic foundation proofs (`intrinsic.py`)

Five representation-space proofs, no downstream probe:
1. **Disentanglement / OOT** — channel MI (kNN estimator vs shuffled null) and
   **out-of-time subspace stability** via principal-angle overlap of temporal
   splits (bounded [0,1]); gate `< 1.5× null` for MI, `> 0.85` for OOT.
2. **Lipschitz** — perturb an event (±1 day) and bound `‖Δz‖/Δ`.
3. **Slow-state trajectory continuity** — directional cosine of the slow band's
   velocity over growing prefixes; gate `> 0`.
4. **Information plane** — the multi-horizon `agg` skills across horizons.

---

## 7. Training protocol

- **Budget derived from data + hardware** (`derive_budget`); govern until the
  held-out metric plateaus within a measured noise floor (no fixed epoch count).
- **Best-state kept strictly** (lowest held-out loss ever evaluated).
- **Warm-start / continual** supported with a version/arch guard.
- **whitening computed before save** so the checkpoint ships WITH the boundary
  transform; products rebuilt from the checkpoint.
- Config surface: `CFMConfig` + derived-resolution receipts; CLI `--set KEY=VAL`
  for any override; registry records inputs/config/version.

---

## 8. Streams tested

| stream | fixture | scale |
|---|---|---|
| **Layer A** (rabbit_hole synthetic) | `rabbit_hole/data/duckdb/customer_event_stream.duckdb` | 26.8M events / 25k cust |
| **Instacart** (external) | `rabbit_hole/data/instacart/customer_event_stream.duckdb` | 37.4M events / 206k cust |
| **ecommerce_2019** (Kaggle, external) | `rabbit_hole/data/ecommerce_2019/customer_event_stream.duckdb` | 4.4M events / 212k cust |

The canonical schema (DuckDB table `customer_events`):
`customer_key, event_ts, brand, event_type, event_attributes, entity_type,
entity_id, source_table, value`. The encoder is stream-agnostic; required column
semantics only — the input tokenizer adapts via vocabulary + schema.

---

## 9. Measured state (Layer-B report card)

Instacart v4.3 (weights), v4.4-gates: **portfolio 15/15 (13 skills + geometry +
canary); intrinsic 5/5** (trajectory +0.119, OOT 0.971, MI 0.012, Lipschitz,
info-plane). Geometry reproducible (0.289–0.297 ≥ per-stream floor).

Layer A (rabbit_hole): the hardest stream — conviction case; needed the
input-level continuous low-pass to make slow-state continuity structural.

Open derivation items (in flight): derive the last gate constants (`mp_floor`,
`geom_coverage`, `traj_tau_mult`) from stream statistics; `aib`/`whiten_cond`
literals remain documented fallbacks.

---

## 10. Decision log (why the current shape)

- **DEC-006** teach the state, no hand-fed features.
- **DEC-008** encoder judged on Layer B only.
- **DEC-009/010/011** objectives default-ON, opt-out; agnostic φ.
- **DEC-014** DWA replaces Kendall (scale-free dynamic weighting).
- **DEC-016/017/018** rank trains on the graded metric; geometry as an explicit
  Pareto coordinate; sf gated by R².
- **DEC-019/020** bank + closed-loop governor; barrier + grouped PCGrad.
- **DEC-022** whitened donor readout (boundary).
- **DEC-023** intrinsic proofs.
- **DEC-024** trajectory zigzag is structural to a first-order recurrence.
- **DEC-025** dual-velocity experts (superseded by unified trunk).
- **DEC-026/028** boundary whitening order + condition-capped whitening
  (reproducible gates); soft-spectrum isotropy.
- **DEC-027** slow-expert low-pass intent filter.
- **DEC-029/030/031/032** macro shifts: unified multi-timescale SSM, global-EMA
  ZCA, AIB, log-det volume; trainable iso/trajectory; calibrated geometry floor;
  dual-target volume.
- **DEC-033** cross-stream validation (Layer A): robustness fixes.
- **DEC-034** per-stream data-derived geometry floor (coverage × capacity).
- **DEC-035** continuous stream-profiled input low-pass (removes the binary
  content/trend partition); task-structural PCGrad; stream-agnostic profiling.

---

## 11. File map

```
cfm_config.py                 config + as_of_epoch (single day boundary)
cfm_data.py                   point-in-time cuts, splits, samples, sequences
cfm_model.py                  EventVocab, _scan, SelectiveSSM, IntentFilter, ns_zca, CFM
cfm_training.py               governed training, objectives, DWA, PCGrad, GeometryBank,
                              whitening, stream_lowpass_retention, slow_channel_mask
cfm_state.py                  StateStore (only writer), fade/absorb, build_products, load
customer_foundation_model.py  CLI (train / products / ladder / portfolio)
portfolio.py                  Layer-B grade (structure-skill + self-calibrating geometry + canary)
intrinsic.py                  intrinsic foundation proofs
autotune.py                   data-derived budgets / half-life / horizons; governor
daily_states.py               (1b) daily job
specs/objectives_catalog.html every objective, its inductive bias, failure modes
```

Run:
```
python3 -m looking_glass.customer_foundation_model train --customers 25000 --anchors 6 \
  --sample-a 20000 --as-of YYYY-MM-01 --db <fixture>.duckdb --out-dir <dir>
python3 -m looking_glass.customer_foundation_model portfolio --out-dir <dir>
python3 -m looking_glass.intrinsic --out-dir <dir>
```
