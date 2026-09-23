# Offline RL Prototype Specification — Predictive Lifecycle Management (HVA → LTV)

**Status:** Draft for review
**Owner:** [Data Science]
**Version:** 0.1

---

## 1. Context

We are a large US omni-channel clothing retailer (stores + e-commerce + email/marketing channels). We have customer-level transaction data back to 2021, which will be consolidated into a single long-format **Customer Event Stream** table containing every event a customer has taken (purchase, email click, etc.) or has had happen upon them (email send, demographic update, etc.). This event stream is the foundation for all modeling described here.

The organization operates across three pillars, each of which will eventually use reinforcement learning / adaptive solutions:

1. **Predictive Lifecycle Management** — selecting the highest-value action (HVA) a customer can take to increase their Lifetime Value (LTV) the most
2. **Marketing Personalization** — pricing, promotion, etc. per individual customer
3. **Marketing Delivery** — channel, frequency, time-of-day optimization

**Scope — all three pillars are minimum scope for this project.** This is not a Pillar 1 prototype; it is the **decision-optimization platform** that serves all three pillars:

1. **Predictive Lifecycle Management** — HVA selection per customer per week (the first policy head to be built)
2. **Marketing Personalization** — pricing/promotion decisions per customer
3. **Marketing Delivery** — channel/frequency/time-of-day decisions per customer

The shared foundation (event stream → episode builder → causal encoder → simulator → offline evaluation harness → operations) is built **once** and serves all three pillars. Each pillar is a **policy head** consuming the same customer state representation, trained and evaluated by the same machinery. Pillar 1 ships first because it has the cleanest measurement; Pillars 2 and 3 layer onto the platform. The RL approach optimizes **total company gross margin** (via incremental gross margin / LTV growth).

**Why offline-first:** we must prove the approach works *before* deployment to earn leadership buy-in for moving from the current "traditional" world to the "optimized" world. Future state includes a small (5–10%) standing holdout group for online incrementality measurement.

---

## 2. Purpose

Build and validate the decision-optimization platform that:

1. Learns a **customer state representation** (`embedding(customer, t)`) from the event stream using a causal transformer encoder
2. Learns a **customer simulator** (dynamics + reward model) from historical data — the "world model"
3. Uses that simulator to train and compare **policy candidates** under hard guardrails — one policy head per pillar (HVA selection first; personalization and delivery heads follow on the same platform)
4. **Proves value offline** via a rigorous offline evaluation harness (temporal holdout replay, simulator rollouts, off-policy evaluation, sensitivity analysis, constraint audits)
5. Produces a **leadership-ready report** and a design for a constrained online pilot (5–10% holdout)
6. Establishes the **RL maturity path**: model-based offline RL (simulator) kickstarts the platform now; **IQL/CQL model-free offline RL is the stated goal** once the deployed policy generates richer logged data (Section 8.7)

---

## 3. Success Criteria

The prototype is successful when:

- [ ] **Offline proof:** temporal-holdout replay (train 2021–2024, replay 2025) shows the winning policy beats the historical behavior policy on incremental net gross margin, with confidence bands and a constraint-violation audit
- [ ] **Cross-validation:** simulator rollouts, OPE estimators, and temporal replay directionally agree (no contradictory stories)
- [ ] **Guardian metrics flat:** opt-outs, deliverability, returns, and engagement proxies show no silent damage under the winning policy
- [ ] **Robustness:** policy's edge survives sensitivity analysis (perturbation of simulator parameters)
- [ ] **No failure modes:** policy-collapse and reward-hacking checks pass; a human sanity review finds no degenerate decisions
- [ ] **Leadership artifact:** a single report presenting candidates vs. baselines, dollars-and-cents, risk column, and the pilot design
- [ ] **Pilot ready:** a constrained online A/B design with pre-agreed graduation criteria (incremental GM lift at 95% confidence, guardian metrics flat, kill-switch plan)
- [ ] **Operational baseline:** reproducible training (model registry, experiment tracking), drift monitoring defined, retraining cadence documented
- [ ] **Platform reusability:** the shared encoder/simulator/eval stack demonstrably serves a second policy head (Pillar 2 or 3) without re-architecture

---

## 4. Architecture Overview

```
Customer Event Stream (long-format, all events)
        │
        ▼
Episode builder (weekly timesteps: state, action, reward)
        │
        ▼
Causal Transformer Encoder ──► embedding(customer, t)   [state representation]
        │
        ▼
Dynamics Model (simulator): P(next-state, reward | state, action)
        │
        ├── Policy heads (trained/evaluated IN the simulator):
        │       • Pillar 1: HVA selection (first to ship)
        │       • Pillar 2: personalization (pricing/promotion)
        │       • Pillar 3: delivery (channel/frequency/time)
        │       • Candidate algorithms per head: return-conditioned sequence
        │         policy, uplift/bandit baseline, greedy-with-guardrails
        │
        ▼
Offline Evaluation Harness (temporal replay + rollouts + OPE + sensitivity + audits)
        │
        ▼
Leadership report ──► Constrained online pilot (5–10% holdout) ──► Graduation / full rollout
```

### 4.1 Recommended approach and why

**The recommendation (locked):** a custom-built, conservative **model-based offline RL** system:

1. Causal transformer encoder over the event stream → `embedding(customer, t)` (Section 6)
2. Bootstrap-ensemble dynamics + reward model — the simulator (Section 8)
3. DQN-style discrete Q-learning trained **inside** the simulator, with MOPO-style uncertainty-penalized rollouts (Sections 8.5–8.6)
4. Guardrails as action masking + budget Lagrangian, enforced in-sim and in production with the same code path (Section 9)
5. The offline evaluation harness as the gatekeeper — nothing deploys without passing it (Section 10)
6. Maturity path: IQL (primary), CQL/COMBO (candidates) in the same harness once pilot data enriches coverage (Section 8.7)

**Why this approach (in order of importance):**

1. **Coverage dictates the algorithm family.** Historical action coverage is thin and confounded (mostly "no action" + targeted campaigns). Model-free offline RL (IQL/CQL) needs rich coverage to learn trustworthy values directly from logged transitions — that is not our data yet. A model-based approach is robust to this: the dynamics model borrows strength across segments and provides the counterfactual rollouts needed for proof
2. **The action space is small and discrete** (3–5 HVAs + "no action"). This points to value-based discrete-action RL (DQN-style) over the continuous-control methods the offline RL benchmark literature favors (TD3+BC, SAC-based variants)
3. **Exploration is free inside the simulator**, so the policy optimizer can be simple and well-understood; conservatism is injected where it matters — distrusting uncertain simulator regions via ensemble disagreement (MOPO-style) — rather than as penalties on unseen actions
4. **Interpretability is the buy-in strategy.** The ensemble model and per-action values are auditable; the harness produces "here's what this policy would have done in 2025, dollar-for-dollar, with a constraint audit." A black-box critic (off-the-shelf COMBO/CQL) cannot be interrogated the same way for leadership
5. **The harness decides, not the algorithm choice.** Offline RL algorithm rankings are benchmark-specific and brittle. Candidates — including COMBO, IQL, CQL, DT-style, and uplift baselines — all run the same gauntlet (temporal replay, OPE, sensitivity analysis), and only the empirical winner deploys
6. **This approach builds the platform anyway.** Everything IQL/CQL will later need (encoder, episode builder, reward model, evaluation harness) is produced by the model-based kickstarter. The maturity path is a graduation, not a rewrite

**What was deliberately rejected (and why):**

| Rejected | Why |
|---|---|
| IQL/CQL today (model-free offline RL) | Coverage too thin to trust direct value learning on logged transitions; deferred to the maturity path (Section 8.7) |
| Off-the-shelf COMBO/MOPO | Validated on continuous-control benchmarks with dense rewards and full coverage — a different problem shape (discrete actions, sparse/confounded rewards, hard constraints); still requires all the custom machinery; black-box for leadership |
| Decision Transformer as primary | Return-conditioned imitation cannot beat the status quo by itself; kept as a candidate |
| One-step bandit/uplift | Horizon-blind — cannot capture LTV dynamics; kept as a baseline to answer "does temporality actually add value?" |
| Online RL | Months per learning cycle at 90-day reward horizons, live-exploration margin cost, brand risk |
| Behavioral cloning | Imitates the status quo — can never beat it |

---

## 5. Data: Customer Event Stream

### 5.1 Conceptual schema (long format)

| Column | Example | Notes |
|---|---|---|
| customer_id | 1002341 | tokenized / anonymized |
| event_timestamp | 2024-03-14T18:22:00Z | |
| event_type | purchase, email_send, email_open, email_click, sms_send, store_visit, demographic_update, offer_redemption, ... | single taxonomy to be finalized |
| event_channel | ecom, store, email, sms, direct_mail | |
| action_tag / campaign_id | welcome_series, winback_v1, birthday_offer | critical for action extraction |
| event_amount / GM | 45.60 | GM attributed to the event |
| item/category context | outerwear, denim, ... | for cross-sell signals |
| device / platform | ios, web, pos | |

### 5.2 Defensive data engineering (MUST before any modeling)

The following questions must be answered with data audits before building the encoder or simulator. Treat unresolved items as blockers, not nice-to-haves.

1. **Event taxonomy:** is there a single canonical event type list? What is the full enumeration today, and who owns it?
2. **Coverage:** which event types exist with meaningful volume per channel back to 2021? Which began mid-stream (schema/backfill gaps)?
3. **Timestamp integrity:** single timezone-normalized timestamp? Are send/click timestamps reliable (delivery vs. actual receive)?
4. **Action attribution:** how are campaign/HVA sends tagged today? Are tags consistent across email/SMS/direct-mail systems? This determines the action set (Section 9)
5. **Nulls / dedupe:** duplicate events (retries, re-sends), null customer_ids, test accounts — what are the known data-quality issues and their expected rate?
6. **Offline vs online unity:** are store purchases and e-commerce events in the same stream with the same fidelity?
7. **PII scope:** what PII lives in the stream, and what is the anonymization/tokenization requirement before the stream is used in training (see Section 6.4)?
8. **Data gravity:** where does the stream physically live (Databricks Unity Catalog today, GCP BigQuery after migration)? Confirm read access + compute availability.

### 5.3 Episode construction (shared substrate for all algorithms)

- **Timestep:** weekly, per customer (aligned with campaign cadence)
- **State at time t:** `embedding(customer, t)` — see Section 6
- **Action a_t:** the HVA given in week t (or "no action"); overlapping touches per week should be treated as bundles or resolved by the attribution rule defined below
- **Reward r_t:** incremental net GM realized within the following 90-day horizon, discounted — see Section 7. Attribution and counterfactual-baseline rules must be defined and documented before training
- **Causality:** reward windows must not overlap training states in a way that leaks the future; causal masking throughout (Section 6.2)
- **Calendar context:** week-of-year, promo calendar, holiday flags as global features (Section 8.2)

### 5.4 Data prep rules (agreed)

- **Downweight 2021–2022** (COVID-era distortion) in training
- **Exclude or flag anomalous windows** (systems outages, promotion anomalies) — maintain a known-issues calendar
- Split conventions: train 2021–2024 / temporal holdout 2025 (used for replay only, never for training or model selection)

---

## 6. Encoder: Causal Transformer over the Event Stream

The state of a customer is a learned embedding: `embedding(customer, t)` = output of a causal sequence encoder applied to that customer's event history up to time t.

### 6.1 Design

- Tokenize events (event type + channel + item context + amount bins + time gap)
- Train a transformer with **causal masking** over per-customer event sequences
- The representation must be a **function** `enc(customer, t)` evaluable at *any* timestamp t — not a lookup over "last event" — to support simulation rollouts and historical replay
- Include **inter-event time-gap encoding** (irregular event timing is fundamental to retail event streams; standard positional encoding alone will misrepresent sparse vs. dense activity)

### 6.2 Causality (non-negotiable)

Only events with timestamp ≤ t may influence `enc(customer, t)`. Enforce at the data level (masking) and verify at the eval level (ablation checks that future events do not change past embeddings). Future leakage will silently invalidate every offline evaluation result.

### 6.3 Cold start / short sequences

New customers have very few events. Requirements:

- Learned `<start>` token and padding strategy for short sequences
- Explicit validation that encoder quality does not degrade on new-customer cohorts (this is the highest-LTV moment of the lifecycle; it must not be the weak point)
- Verify at training: embedding quality evaluated **by cohort age**

### 6.4 PII / privacy

- Tokenize/anonymize the event stream before encoder training
- Review with data governance: embeddings trained on raw customer data can leak personal information
- Confirm retention/deletion policy for the trained artifact

### 6.5 Encoder quality gate (MUST pass before simulator work)

The encoder is only as good as what it can predict. Before building the dynamics model:

- Probe tasks: predict next purchase (next 7/30 days), predict churn, predict next event type — measured on held-out customers
- Compare against a hand-featured (RFM) baseline; the encoder must beat it or justify not being used
- Check embedding sanity: does distance/clustering separate known segments (new vs. loyal vs. lapsed, category affinities)?

### 6.6 Versioning

- Every encoder version: data snapshot, training config, eval results → recorded in the model registry (Section 11.1)
- Embedding drift monitoring defined (Section 11.2)

---

## 7. Reward Engineering (lock these early)

### 7.1 Definition

The reward is **incremental net gross margin** of the action versus "no action":

```
reward = GM(state, action) − GM(state, no-action)
```

- Net of action cost (offer margin cost, send cost)
- Measured over a 90-day horizon from the action, discounted
- Margin, not revenue: the agent must not optimize revenue at the expense of margin

### 7.2 The counterfactual problem (single hardest estimation issue)

`GM(state, no-action)` is **unobserved** in historical data — we only ever see one branch. The prototype must pick a documented strategy (or a combination):

1. **Uplift-style modeling:** explicitly model the increment (action vs. control branches where they exist)
2. **Propensity weighting (IPW):** reweight logged actions by their inverse probability of being taken, to de-confound targeted sending
3. **Natural-experiment windows:** restrict training to periods where campaigns were sent broadly/near-randomly (e.g., sitewide sends)

Decision required before reward-model training. The choice will be validated in the temporal holdout.

### 7.3 Confounding risk (accepted and managed)

Historical actions were targeted, not randomized. Naive models will attribute purchases to actions that "would have happened anyway." Mitigations: Section 7.2, holdout replay (Section 10.1), and labeling all results as "historically-consistent projections" rather than causal claims until the online pilot proves them.

### 7.4 Finance alignment (pending)

- Reward definition must ultimately be signed off by FP&A (margin definitions, cost allocation, discounting)
- **Open item:** schedule FP&A engagement after prototype value is demonstrated

### 7.5 Guardian metrics (monitored alongside the reward — NOT part of the optimization objective)

| Metric | Why |
|---|---|
| Unsubscribe / opt-out rate | silent list damage |
| Email deliverability / spam rate | infrastructure damage |
| Return rate | margin quality |
| Engagement proxies (open/click per send) | channel health |
| NPS / brand sentiment | brand equity |

Requirement: every evaluation report must show guardian metrics alongside lift numbers. The reward can't see silent damage; these catch it before it compounds.

### 7.6 From per-customer reward to total company GM

The unit of optimization is per-customer incremental GM; **the company objective is its sum**. Total company GM = Σ baseline GM + Σ incremental GM — and since baseline GM is unaffected by the policy (customers who receive no action contribute zero *incremental* GM but keep their baseline GM), optimizing the sum of per-customer increments **is** optimizing total company GM.

The sum works because:

1. **Baseline GM cancels** — untouched customers contribute zero increment; the policy only moves the incremental layer
2. **Attribution is disjoint** — every GM dollar is attributed to exactly one customer (requires the attribution rule, Section 5.3), so summing counts each dollar exactly once
3. **Horizons and discounting compose** — discounted 90-day per-customer rewards sum to a consistent company-level discounted objective

**The two things that break naive additivity (both must be handled):**

1. **Overlapping time windows** — adjacent 90-day reward windows for the *same customer* can double-count a purchase. Rule required: reward credit is assigned to the triggering action week, not to all overlapping windows
2. **Global coupling constraints** — budget envelopes and channel capacity couple customers together. Independently maximized per-customer rewards can exceed the budget. Solution: a global allocation layer (e.g., Lagrangian on the budget: effective reward = incremental GM − λ·cost) so per-customer maximization is budget-aware — the same mechanism as Section 9 guardrail enforcement

**Empirical anchor:** the 5–10% holdout A/B measures **total** incremental GM directly at the company level (treatment GM − control GM). Model validation requirement: Σ per-customer predicted incremental GM must approximately reconstruct the measured aggregate; if not, apply a calibration adjustment (reweighting/scaling). The holdout is the ground truth for both the per-customer decomposition and the total.

**What the sum does NOT capture** (accepted approximations, monitored elsewhere): cross-customer network effects (referrals, word of mouth), competitor/macro effects, and brand equity — the latter covered by guardian metrics (Section 7.5).

---

## 8. Simulator (Option A: Model-Based RL / World Model)

### 8.1 Dynamics model

Learns `P(next_state, reward | state, action)` from the event stream:

- Input: `(embedding(customer, t), action a_t, calendar context)`
- Outputs: next-week state change and reward distribution
- Implementation options: per-outcome gradient-boosted response/hazard models (tractable, interpretable) or a small neural net; decision documented in design phase

### 8.2 Seasonality (must be in the model)

Retail behavior is deeply seasonal (Q4, holidays, back-to-school). Calendar features (week-of-year, holiday flags, promo calendar) are **required global inputs** — otherwise the temporal-holdout evaluation will simulate a 2025 without holiday peaks and mislead us.

### 8.3 Reward calibration validation

The simulator's *reward* predictions need their own validation against historical periods where actions actually happened — not just state-transition accuracy. An uncalibrated reward model poisons every downstream number.

### 8.4 Guardrails live inside the simulator

Budget caps, frequency caps, and margin floors must be enforced during simulated rollouts, not just in production — otherwise offline numbers overstate achievable value (Section 9.3).

### 8.5 Policy candidates evaluated in the simulator

1. **Return-conditioned sequence policy** (Decision-Transformer style): sequence model that conditions on a target return; uses the same event-token vocabulary as the encoder
2. **One-step uplift / bandit policy:** argmax over predicted incremental GM per customer-week (no temporality; kept as a cheap, strong baseline to answer "does temporal modeling actually add value?")
3. **Behavior policy** (what the brand historically did) — the baseline every candidate must beat
4. **Greedy-with-guardrails:** simple, explainable, hard to beat in early versions

All candidates are scored by the same offline evaluation harness (Section 10).

### 8.6 Known limits

- Model-based RL is bounded by simulator accuracy; sim-to-real gap is expected → design the pilot so the learning loop closes (log, evaluate, retrain), and communicate expected degradation to leadership up front
- Policy collapse (agent finds a degenerate action exploiting simulator blind spots) and reward hacking (GM-maximizing but brand-destroying exploits) are tested explicitly (Section 10.5)

### 8.7 RL maturity path — model-based now, model-free as the goal

- **Stated goal:** IQL/CQL model-free offline RL as the end-state decision engine, operating directly on logged (state, action, reward, next-state) transitions
- **Why we don't start there:** model-free offline RL requires rich coverage of the action space in the logged data. Today the behavior policy is mostly ad-hoc campaigns and "no action" — thin coverage makes model-free estimates untrustworthy. The model-based (simulator) approach is robust to this: the dynamics model borrows strength across segments and provides the counterfactual rollouts needed for proof
- **The kickstarter:** the model-based approach builds everything model-free will later need — the causal encoder, the episode builder, the reward model, and above all the **offline evaluation harness** (the trust layer). It also produces the initial policies that run the pilot
- **The graduation:** as the deployed (pilot) policy logs richer, nearer-optimal experience into the event stream, action-space coverage improves → train IQL/CQL on the accumulated data → compare against the model-based policy in the same evaluation harness → graduate the winner
- **Simulator as safety net:** the simulator pre-trains the model-free critic (off-policy RL starts warm and conservative instead of cold) and remains the evaluation harness for model-free policies — same machinery, no re-tooling

---

## 9. Guardrails (hard constraints, enforced everywhere)

### 9.1 Constraint set (to be finalized with stakeholders)

- **Action masking:** which HVAs are eligible for which customer segments (rules-based exclusions, e.g., already-received-recently)
- **Frequency caps:** max touches per customer per week/month
- **Budget envelopes:** total offer spend per campaign/period
- **Margin floors:** min expected margin per action-customer pair

### 9.2 Enforcement points

1. **Planner/policy:** constraints encoded as action masking + constrained optimization (Lagrangian or projection) at decision time
2. **Simulator:** constraints enforced inside rollouts (Section 8.4)
3. **Production:** same enforcement code path as the simulator — no divergence
4. **Audit:** every evaluation report includes a constraint-violation count alongside every lift number

---

## 10. Offline Evaluation Harness (the leadership-proof layer)

This is the most important engineering deliverable. Ranked by convincingness:

### 10.1 Temporal holdout replay (most convincing)

- Train encoder + dynamics + policies on 2021–2024 only
- Replay 2025: at every historical decision point, score what each candidate policy would have chosen and what it would have produced (dollars, not units)
- Output: "here's what this policy would have generated against what you actually did last year," with uncertainty bands
- Strict rule: the holdout year is never touched during training or model selection

### 10.2 Simulator rollouts

- Long-horizon counterfactual runs with confidence bands
- Constraint-violation audit beside every number
- Guardian metric projections

### 10.3 OPE estimators

- Importance sampling / doubly-robust estimation on logged events to cover one-step claims
- Reports both the estimate and the effective sample size (the honesty column)

### 10.4 Sensitivity analysis

- Perturb dynamics model parameters (elasticities, response rates) within plausible ranges
- If the winning policy's edge survives perturbation, it is real; if not, it's an artifact

### 10.5 Failure-mode tests (required)

- **Policy collapse check:** review the chosen policy distribution — degenerate (single-action, zero-action) outputs fail the review
- **Reward-hacking check:** search for actions that maximize reward while violating business sanity (margin floor, brand equity); a human sanity review of the top-10 strangest decisions per week runs during the pilot
- **Coverage analysis (step 0):** before any training, map which HVAs occurred for which segments historically. Thin coverage kills model-free RL and constrains the action set (Section 9.1, Section 12)

### 10.6 Baseline set (everything must beat these)

- Behavior policy (historical)
- One-step uplift / bandit
- Greedy-with-guardrails
- "No-action"

### 10.7 Evaluation report format

Every run produces: candidates vs. baselines table (lift, uncertainty, constraint violations, guardian metrics), sensitivity results, failure-mode checks, and the temporal-replay headline number. This is the artifact shown to leadership.

### 10.8 How OPE and offline RL work together

OPE (off-policy evaluation) estimates how well a candidate policy would perform **using only logged data** — no deployment. It is the offline analog of the A/B test: cheap and fast, but an estimate; the online holdout is the expensive, definitive answer. The trust chain escalates: OPE estimate → temporal replay (dollars vs. what actually happened) → simulator rollouts → small online pilot.

OPE families in use here:

- **Direct method:** reward model predicts the outcome of the policy's choices — simple, but biased by model error
- **Importance sampling (IS):** reweight logged transitions by ρ = π_target(a|s) / π_behavior(a|s) — unbiased in expectation, but variance explodes at long horizons (hence clipped/per-step variants)
- **Doubly robust (DR):** IS correction on top of a model baseline — the practical workhorse for one-step claims
- **Model-based rollouts:** the simulator (Section 8) rolls the policy forward over long horizons — the sequential, horizon-agnostic option
- **FQE / HCOPE:** fitted-Q evaluation learns the target policy's value from logged transitions (modern standard for sequential OPE); high-confidence variants produce "≥X% uplift with 95% confidence" statements — the exact language leadership needs

Why this pairing is essential: offline RL **trains** policies from logged data; OPE **scores** them on logged data. They share the same distribution-shift enemy, and offline RL's conservatism (refusing to over-trust out-of-distribution actions) is exactly what makes the OPE numbers credible. OPE is the trust layer that converts "a policy we trained" into "a policy we can bet margin on."

---

## 11. Operations

### 11.1 Model registry & experiment tracking

- Every encoder, dynamics model, reward model, and policy version: data snapshot, config, eval results (MLflow-style)
- "Why did the pilot underperform?" must be answerable for any version

### 11.2 Drift monitoring & retraining cadence

- Monitors: embedding distribution shift, sim-vs-actual response gap, KPI drift
- Retrain triggers defined; default cadence (e.g., monthly retrain, quarterly full re-encoder) to be finalized

### 11.3 Data pipeline

- Event stream → episode builder → feature/embedding store → training → evaluation, orchestrated (Airflow on Databricks today; GCP Composer/BigQuery after migration)
- Full reproducibility of any reported number

### 11.4 Human oversight

- Weekly human sanity review of policy decisions (top-10 strangest) during pilot
- Kill switch: immediate rollback path, pre-tested

### 11.5 Tooling environment

- Databricks (today) → Google Cloud Platform (soon)
- GPU availability for transformer training: to be confirmed
- **Open item:** confirm compute budget and ML runtime versions

---

## 12. Pilot Design (future, spec'd now)

- **Population:** customers in scope for the chosen HVAs
- **Assignment:** 5–10% standing holdout (random), remaining eligible customers to policy
- **Metrics:** primary = incremental net GM (90-day, discounted); secondary = guardian metrics (Section 7.5)
- **Graduation criteria (pre-agreed before results exist):**
  - Statistically significant incremental GM lift (95% confidence) vs. holdout
  - Guardian metrics flat (or better)
  - Constraint violations below threshold
  - Drift monitors green
- **Risks:** sim-to-real degradation expected → pilot sized accordingly; kill switch available
- **Learning loop:** every pilot decision and outcome logged back into the Customer Event Stream, enabling retraining and (later) model-free RL (IQL/CQL) once the deployed policy is richer

---

## 13. Roadmap

| Phase | Deliverable | Exit criteria |
|---|---|---|
| 0 | Data audit, event taxonomy, coverage analysis, action set definition | All Section 5.2 questions answered; HVA list confirmed with brand team |
| 1 | Episode builder + causal transformer encoder + quality gates | Probe tasks beat RFM baseline; causality verified; cold-start validated |
| 2 | Simulator + reward model + calibration validation | Reward calibration passes; seasonality handling verified |
| 3 | Policy candidates + offline evaluation harness | Temporal replay produces headline lift number; all failure-mode checks pass |
| 4 | Leadership report + pilot design | Sign-off to run constrained pilot |
| 5 | Pilot (5–10% holdout) + learning loop | Graduation criteria met → full rollout decision |
| 6 | Pillar 2 head (personalization) on shared platform | First personalization policy evaluated in the harness |
| 7 | Pillar 3 head (delivery) on shared platform | Delivery policy evaluated; shared platform serving all three pillars |
| 8 | Graduate to model-free offline RL (IQL/CQL) | Model-free policy beats the model-based policy in the same evaluation harness → deployed |

---

## 14. HVA Brainstorm (candidate action set — to be narrowed with brand team)

For v1, the action set should be **3–5 well-defined HVAs with good historical coverage and clean tagging**. Candidate list:

1. **Onboarding / welcome series** — welcome email, style quiz, loyalty program enrollment
2. **Second-purchase activation** — day-30 "complete your look" offer / cross-sell into adjacent category
3. **Win-back** — lapsed-customer offers at 90/180 days
4. **Cross-sell / adjacent category** — e.g., outerwear to denim buyers
5. **Loyalty** — program enrollment, tier upgrade, points push
6. **Birthday / anniversary offer**
7. **Abandoned cart / checkout recovery**
8. **Store-visit incentive** (BOPIS / in-store pickup — omni-channel leverage)
9. **Referral program invite**
10. **Review / rating request** (engagement proxy)
11. **Restock / price-drop alerts** (item-level interest signal)
12. **Post-purchase care / fit education** (retention/returns angle)

Selection criteria: (a) exists in the event stream with consistent tagging, (b) sufficient historical volume, (c) measurable downstream GM, (d) meaningful action space for the policy to learn from.

---

## 15. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Confounded historical actions (selection bias) | Propensity weighting / uplift modeling / natural-experiment windows (Section 7.2); honest labeling |
| Thin action coverage | Coverage analysis first (Phase 0); scope action set to what's logged |
| Sim-to-real gap | Uncertainty bands, expected-degradation messaging, pilot sized accordingly, learning loop |
| Reward hacking / policy collapse | Failure-mode tests, constraint audits, human sanity review, kill switch |
| Data quality (taxonomy, timestamps, dedupe) | Defensive data engineering gate (Section 5.2) is a blocker, not nice-to-have |
| Encoder leakage / weak representation | Causality verification, probe-task quality gates, cohort-age validation |
| Guardian-metric damage invisible to reward | Guardian metrics in every report; graduation criteria require them flat |
| Organizational (finance sign-off, metric setting) | Reward definition documented; FP&A engagement scheduled after value demonstrated |

---

## 16. Open Items (need decisions)

1. **Final HVA list** — brand team to provide; candidate brainstorm in Section 14
2. **Event taxonomy / tagging audit** — who owns the event-type list; campaign tagging consistency across channels
3. **Reward definition sign-off** — FP&A engagement timing (proposed: after Phase 3 value demonstrated)
4. **Counterfactual baseline strategy** — final choice among Section 7.2 options, after Phase 0 coverage analysis
5. **Compute environment** — GPU availability on Databricks/GCP; orchestration tooling
6. **Constraint set finalization** — frequency caps, budget envelopes, margin floors with stakeholders
7. **Attribution rule** — how overlapping touches in a week resolve to an action/reward pair
8. **90-day horizon & discounting** — confirm with finance; sensitivity to horizon to be tested
9. **Pillar 2/3 decision logging** — offer-exposure and send metadata requirements for the personalization and delivery policy heads (what must be logged to the event stream for those heads to train)

---

## Appendix A — Approach Rationale: "Why not just…?"

**Why not supervised prediction models?** They answer "what happens?" — we need "what should I do?" A model that predicts purchase probability doesn't decide whether to offer a discount, when, or to whom. RL is a decision paradigm, not a prediction paradigm.

**Why not online RL?** With a 90-day reward horizon, online learning means months per learning cycle, plus the margin cost of live exploration and brand risk. Offline RL compresses years of experience into one training run.

**Why not just uplift/lift models?** They are step-wise and horizon-blind — they pick the best action *now* but cannot model that decisions compound over a customer's lifecycle, which is the entire premise of LTV optimization.

**Why not just heuristics/rules?** They encode what we already know; the whole point is to discover what we don't — systematically, measurably, and provably better than the status quo.

**Why offline RL isn't magic:** it inherits the biases of its data (confounded historical targeting — Section 7.2) and can over-claim on thin coverage. Hence Phase 0 coverage analysis and the offline evaluation harness are hard gates, not decor. Offline RL doesn't remove uncertainty — it moves it to places we can measure before committing margin.

---

## Appendix B — Capability gained vs. the current state

**Current state:** 40+ supervised propensity models (customer lifetime value, discount sensitivity, churn propensity, etc.), each built from scratch with its own separate data pipeline.

### B.1 The shift: from prediction to decision

Today's models answer **"what will happen?"** — P(purchase), P(churn), LTV, discount sensitivity. They produce inputs; humans apply them via rules and tribal knowledge.

The platform answers **"what should I do about it, and what happens if I do?"** — and optimizes that answer against a single objective (incremental GM) over time, under hard constraints, with the decision logged and measured.

### B.2 Capability comparison

| | Today (40+ propensity models) | After (RL platform, validated) |
|---|---|---|
| Question answered | Who will churn? Who is discount-sensitive? | Exactly what action to take for each customer, this week |
| Optimization target | 40+ separate targets, often uncoordinated, sometimes in conflict | One objective: total company incremental GM, with guardrails |
| Incrementality | Not built in — cannot distinguish "would have bought anyway" from "bought because we acted" | Reward is incremental GM vs. no-action (Section 7) — optimizes what spend actually buys |
| Horizon | One-step predictions | Sequential, multi-step: when to hold an offer so it's more effective later |
| Decisions | Humans apply models via rules; drifts by person and over time | One consistent policy, guardrails enforced in code, every decision logged |
| Feedback loop | Static until manually retrained; never learns from outcomes of decisions | Closed loop: every decision + outcome logs back → policy improves; compounding asset |
| Counterfactuals | None | "What if we'd never sent win-backs at all?" — the simulator/what-if layer |
| Proof | Model AUC against history | Incrementality proof: offline replay + pilot holdout — marketing contribution finally measured |
| Pipelines | 40+ bespoke data pipelines, retrain cycles, drift monitors | One event stream → one encoder → one platform |
| Scale across pillars | A model zoo per pillar, built separately | One substrate; lifecycle/personalization/delivery are policy heads on it |

### B.3 The three upgrades that matter most

1. **Incrementality becomes the objective.** This is the fundamental one. Propensity models cannot tell you whether a discount *caused* a purchase; the platform's reward is engineered to measure exactly that (Section 7). It's the difference between predicting behavior and buying it
2. **Sequencing.** LTV is a sequence of decisions — a win-back sent at week 10 vs. week 16 has different value, and the policy learns that trade-off. No propensity model can see it
3. **The decision layer finally exists.** Today the models produce inputs and humans decide. The platform *is* the decision — consistent, constraint-aware, auditable, and provably better before it spends a dollar of margin

### B.4 What is kept

The propensity models do not vanish — they get absorbed: their predictions become features in the event stream / encoder inputs, they serve as baselines in the evaluation harness, and their outputs feed the simulator. Note that consolidation is as much an organizational project as a technical one (40 models have 40 stakeholders), and per-model interpretability is partially traded for decision-level explainability — the simulator's per-action values give better decision-level explanation than a zoo of propensity scores ever did.
