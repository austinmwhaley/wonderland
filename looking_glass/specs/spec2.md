# Customer Foundation Model & Hierarchical Offline RL - Complete Specification v1.1

## Core Principles
- **Defensive by Default:** Every tensor/operation assumes NaN, Inf, or OOD possible.
- **No ML Fallback:** Neural failures trigger infrastructure rollback to BAU (no heuristic degradation).
- **One-Way Flow:** `A → B → C → D → Execution`. C has zero knowledge of D.
- **Maximize Modularity & Dynamism:** Static values replaced with data-driven distributions or adaptive mechanisms.
- **Split Training vs. Evaluation Rewards:** Train on observed margin (de-biased); evaluate on incremental counterfactual margin.


## 1. Data Layer (A)
**Raw Input:** `[customer_key, event_ts, brand (6 total), event_type, event_attributes]`.


**Winsorization & Filtering (Applied sequentially):**
1. **Customer-Level Filtering (Drop top 1%):** Compute per-customer aggregates; drop if > 99th percentile of:
   - `total_margin_dollars`
   - `total_events`
   - `max_events_per_hour` (computed via `MAX(COUNT events per hour)` per customer)
2. **Event-Level Clipping (Applied after filtering):**
   - **Margin Dollars:** Per-brand fixed 99.5th percentile (hard cap).
   - **Event Velocity:** Global 99th percentile per hour (cap).
   - **Time Deltas:** Quantile Transformation (Empirical CDF scaled to [-1, 1]) instead of raw log clipping.
   - **Sequence Length:** Dynamically set to `min( P_99(global_event_distribution), GPU_MEM_LIMIT )`. Recalculated quarterly. Recency bias applied (most recent events).
3. **Historical Action Capping (RL Dataset only):** Cap historical actions for training: `email ≤ 14`, `sms ≤ 7`, `discount ≤ 0.30`.


**Counterfactual Builder Output:** For each decision epoch, pre-compute `r_cf` (Incremental Gross Margin Dollars) using the existing project pipeline. This is used *exclusively* for OPE (Evaluation) and deployment gating—**not** for Q-learning gradients.


## 2. Foundation Model (B)
**Backbone:** Mamba-2 (O(n) complexity).
**Projector:** 2-layer MLP compressing Mamba output to `S_t` (128-dim, normalized).


**Pre-training (Phase 2 - Dual Loss):**
- **Loss A (Predictive):** Masked event prediction (Brand + Type + Attributes).
- **Loss B (Contrastive - InfoNCE):**
  - Positives: Temporal sub-sequences from the *same* customer (shifted ±7 days).
  - Negatives: Sub-sequences from different customers in batch.
  - Temperature `τ=0.07`, Loss Weight `λ=0.5`. Monitor uniformity; increase τ to 0.1 if collapse detected.
- **Feature Crossing (Static Context):** 5 explicit customer stats (Return Rate, Inter-purchase SD, Category Diversity, Open Rate, Discount Sensitivity). **Concatenated directly to RL Critic/Actor inputs** – NOT passed through Mamba.


**Inference:** Projection head discarded. `S_t` cached in Redis/Vector DB (daily batch precomputation).


## 3. Plugin Layer (C - Specialists)
**Definition:** Independent IQL agents. One plugin = one discrete capability (e.g., `email_dose`).
**Action Spaces (Global/Cross-Brand):**
- `email_dose`: 15-class logits (0–14).
- `sms_dose` / `push_dose`: 8-class logits (0–7).
- `discount_dose`: Continuous scalar (Sigmoid * 0.30).
- `category_preference` / `division_preference`: Categorical logits.


**Training (Propensity-Weighted IQL):**
- **Dataset:** `(S_t, A_plugin_hist, r_obs, S_{t+1})`.
  *(Note: No context other than S_t. State handles historical confounding, not simultaneous actions).*
- **Propensity Model (`π_hat`):** A small supervised model (XGBoost or Tiny MLP) trained to predict `P(A_hist | S_t)`. Serves as the behavior policy for debiasing.
- **Critic:** Input `[S_t, A_plugin]` → Q-value. The MSE loss is **weighted** by Clipped Inverse Propensity Scores (IPS): `weight = min(1 / π_hat(A_hist | S_t), ρ_max=10)`. This forces the Critic to pay 3-4x more attention to the 25% exploratory actions, correcting for selection bias in the 75% deterministic BAU data.
- **Actor:** Input `[S_t]` → Global proposal. Uses standard IQL Advantage-weighted regression against the **debiased Critic**.
- **Losses:** Weighted IQL + BC regularization (`λ=0.5`) + Entropy bonus (`β=0.01`).
- **Inference State:** Frozen after training.


**Brand Exception Handling:** Default = Global only. If constraints differ (e.g., discount cap), attach **LoRA adapter (rank r=4)** to the final layer. Freeze base; train only LoRA.


**Registry:** Plugins register via `@register_plugin("namespace")`. Publish outputs to shared context.


## 4. Engine Layer (D - Orchestrator)
**Definition:** The Generalist Optimizer. Handles per-brand allocation, interaction effects, and hard constraints.
**Action Space (Output):** Flattened per-brand vector.
- Email: 6 brands × 15 logits.
- SMS/Push: 6 brands × 8 logits.
- Discount: 6 brands × continuous scalars.
*(Category is separate, not in Marketing Dose Engine).*


**Training (Propensity-Weighted IQL):**
- **Dataset:** `(S_t, Proposals_All_C, A_final_hist, r_obs, S_{t+1})`.
- **Propensity Model (`π_hat`):** Same as above, but trained on the multi-dimensional `A_final_hist`.
- **Critic Input:** `[S_t, A_final_hist]` → Q-value. Loss is **weighted** by the same clipped IPS (ρ=10) to debias confounding.
- **Actor Input:** `[S_t, Proposals_All_C]` → Final per-brand actions. Uses standard IQL Advantage-weighted regression against the debiased Critic.
- **Architecture:**
  - **Causal Action Embeddings:** Discrete doses (lookup tables, 16-dim); continuous discount (MLP + raw scalar).
  - **MoE (Mixture of Experts):** 4 experts, Top-2 router. **Spawn & Prune:** Prune if routing weight < 1% (merge weights). Spawn if router entropy high (copy most activated expert + σ=0.01 noise).
  - **DeepFM/DCN:** Explicit pairwise interactions between action embeddings (ANOVA-style).
- **Losses:** Weighted IQL + BC (`λ=0.1`) + Entropy (`β=0.01`).


**Constraint Enforcer Middleware (Decoupled):**
- Post-D, pre-execution deterministic layer.
- Projects raw D output onto valid simplex enforcing **14 marketing events/week** cap.
- Business rules updated in YAML/ConfigMap → **no retraining required**.


## 5. Offline RL (IQL) & Reward (Updated)
**Training Reward (`r_obs`):** Observed `Margin_Dollars` (Revenue - COGS) generated in the lookahead window.
*(No discount subtraction – avoids double-counting. No send costs – negligible).*
**Evaluation Reward (`r_cf`):** Pre-computed **Incremental Gross Margin Dollars** from the Counterfactual Builder. Used strictly for OPE validation and deployment gating, **not** for Q-learning updates.


**Standardization:** Both `r_obs` and `r_cf` are normalized by `(Brand, Customer_Tier)` mean/std to prevent gradient domination. De-normalized at inference.


**MDP:**
- **Type:** Continuous Infinite-Horizon (no hard churn terminal).
- **Discount Factor (`γ`):** `0.995` (weekly). **Active:** Dynamically adjusted based on effective churn window (if churn accelerates, lower γ).
- **Decision Epochs:** Varied per plugin. MDP handles via timestamps.


**Active Hyperparameters (Must):**
- **`λ_BC` (BC weight):** Track rolling OPE score. If plateauing, reduce λ to escape local optimum. If OPE variance spikes, increase λ.
- **Gradient Clipping:** Global norm = 1.0. Dynamic LR reduction if gradient norm > 3x rolling average.


## 6. OPE & Validation (Defensive Baseline - Updated)
**Method:** Doubly Robust (DR) estimation using **`r_cf` (Incremental Gross Margin)** as the reward metric for evaluation, not `r_obs`.


**Propensity Correction:** Uses the same `π_hat` model from training.
`DR = (1/N) * Σ [ (weight) * (r_cf - Q_hat(S_t, π_new)) + Q_hat(S_t, π_new) ]` where `weight = min(π_new / π_hist, ρ=10)`.


**Effective Sample Size (ESS):** Compute `ESS = 1 / sum(π_new / π_hist)^2`. Block deployment if `ESS < 0.1 * N` (insufficient historical overlap).


**Action Drift Monitor:** During Shadow Mode, compute KL Divergence between `π_hist` and `π_new`. Flag if `KL > 0.5` for >10% of requests without corresponding OPE lift.


**Deployment Gate (The Ultimate Decider):**
- Shadow Mode (3 days) → Compute DR estimate using `r_cf`.
- **Deploy to production ONLY IF:** The DR estimate shows a statistically significant positive lift (p < 0.05) in incremental gross margin dollars. If lift is negative, flat, or statistically insignificant, the policy is **discarded** and the system rolls back to BAU at the infrastructure level (K8s/Feature Flag).


## 7. System Architecture (Communication)
**Pub/Sub:** Plugins publish tensors to shared context. Engines subscribe via **Manifest**.
**Registry:** Central decorator-based Plugin Registry.
**Data Flow:** One-way. C has no knowledge of D.
**Serialization (Future):** Protobuf/Avro flagged for later (when plugins/engines > 10).


## 8. Deferred / Future Items (V2)
- **Speed Optimizations:** Hash-based MoE Router (LSH), Batch State Flush (write Redis every 10 min).
- **Semantic Cold-Start:** Copy nearest neighbor LoRA weights for new brands/categories.
- **Automatic Kill Switch:** Circuit breaker logic (placeholder exists).