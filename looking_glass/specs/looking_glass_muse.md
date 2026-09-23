# Looking Glass Spec (Muse v0.10)

**Status:** Draft v0.10. Scope limited to this conversation. Companion files are reference only.
**System:** Large online+retail retailer. Layers A → B → C, unidirectional.
**Layer A status:** A1–A4 done (sources landed, keys resolved, 5-col table, taxonomy). A5 source-faithfulness done; cleaning lives in B0. A6 clarified as read paths + snapshot pinning.
**Cadence [PREF]:** B trains monthly (1st), infers daily; C trains any time against a pinned B version (§6).
**Notation:** `CFM_v = train(stream ≤ data_through, B0_w)`; `S_c(t) = CFM_v(c, hist_c ≤ t)`; store in `customer_embedding_daily` (§6). Versions are `vNrN`: `N` bumps only on training-code changes (architecture, objective, breaking B0/taxonomy/vocab change); `rN` bumps on monthly data retrains (e.g. `v1r1 → v1r2`).

Conventions: **[LOCKED]** = non-negotiable (version bump + written reason to change). **[PREF]** = agreed default (changeable with reason + gates passing). Each decision includes a one-line Why / Why not.

---

## 1. Architecture constraints [LOCKED]

1. **Unidirectional flow.** A feeds B, B feeds C, C feeds D. No gradients, retraining triggers, or feature requests upstream. New requirements → new upstream version, never a backward edge.
   - Why: reproducibility of shared substrate. Why not bidirectional: downstream experiments become breaking changes.
2. **Frozen foundation.** B weights immutable after release. C consumes versioned embeddings via API only.
   - Why: comparability across consumers. Why not mutable: results become unanswerable.

```
A: Event Stream  →  B: Foundation (§3 modeling + §6 serving halves)
  →  C: Plugin heads (options + values)  →  D: Marketing Dose composer (§7: weekly 24-dim card)
```

---

## 2. Layer A — Customer Event Stream (faithful raw) [LOCKED principle]

**Definition.** Faithful representation of source data. Append-only log, one row per event, ordered per customer by time. No cleaning beyond faithful landing — no labels, rewards, embeddings, or irreversible filtering [LOCKED].
- Why faithful: preserves optionality; any cleaning is information loss. Multiple B versions can retry without re-ingesting.
- Why not clean in A: bakes one team's thresholds into shared substrate permanently.

**Components.**
- A1. Source connectors — done.
- A2. Identity resolution — done (`customer_key` stable; side handling of unresolvable rows as implemented).
- A3. Canonical table — done: `[customer_key, event_ts, brand, event_type, event_attributes]`.
- A4. Taxonomy registry — done (canonical `event_type` set + per-source maps as implemented).
- A5. Source QC (in A) — landing checks only: contract validation counts (null key, bad ts, unknown type version), volume/freshness monitoring. No outlier removal, no dedupe-as-deletion here.
- A6. Storage + serving — two read paths + snapshot pinning (see below).

**Sources [PREF].** All (list in A1).
- Why all: cross-domain sequencing is the signal. Why not transactions-only: loses intent, response, and affinity signals C needs.

**Platform [PREF].** Databricks now, GCP later; paths portable (config-only switch).
- Why: current gravity vs. planned migration. Why not single-lock: avoids rewrite debt or a stalled flag-day move.

**Schema [PREF, as built].**
```
[customer_key, event_ts, brand, event_type, event_attributes]
```
`event_attributes` is a schema-evolved map for all remaining context.
- Why 5 columns: extension without DDL per source. Why not wide columns: null-heavy, breaks per source. Why not per-domain tables: destroys cross-domain ordering.

**A6. Read paths + snapshot pinning [LOCKED requirements, mechanism as built].**
- Bulk scan (training): all events in [t0, t1] without full-table scans.
- Point lookup (inference/replay): one customer's history ≤ t without full scans.
- Snapshot pinning: (table version + time predicate) recreates exactly the rows any B version trained on.
- Why: bulk-only starves inference; lookup-only starves training; unpinnable inputs void the B freeze. Why not a third copy per pattern: doubles storage + drift surface; partitioning/clustering should serve both.

**Acceptance gates [LOCKED, thresholds TBD].** Source coverage + freshness; key integrity rates; snapshot recreatable from version + predicate. B does not start until these pass.

---

## 3. Layer B — Customer Foundation Model

**Definition.** `embedding(customer, history ≤ t)`, evaluable at any `t`. Trained on A, frozen, consumed downstream [LOCKED].

**Components.**
- B0. Ingest / preprocessing — versioned cleaning on consume (see below) [LOCKED: versioned + shared with C].
- B1. Dataset builder — time-bounded, versioned snapshots from A (via A6 paths).
- B2. Input prep — event tokenization (design TBD).
- B3. Backbone — sequence model [PREF: SSM, e.g. Mamba-2].
- B4. Training objective — pretraining loss (TBD).
- B5. Freeze + registry — immutable versioned releases recording data snapshot + B0 version + config + gate results [LOCKED].
- B6. Serving — two tables, both versioned with the model [LOCKED]: `customer_embedding_daily` (public 128-d `S_t`, one row per customer per day, §6) and `customer_state_cache` (private wide `H`, same grain, rewritten only for customers with new events; the sole basis for incremental updates — yesterday's `S_t` cannot substitute). Versioned embedding API refreshed daily under the monthly-frozen weights (§6) (store mechanism TBD).
- B7. Evaluation — causality, quality, and cold-start gates (metrics TBD).

**B0. Cleaning on consume [LOCKED principle; PREF defaults].** All cleaning lives here, not in A. B0 is deterministic, versioned, and imported by C task builders so state-view and label-view never diverge.
- Why B0-shared, not clean-in-A: A stays faithful; thresholds stay revisable per B version. Why not per-consumer cleaning: N cleanings = N incomparable results (same failure as fine-tuning).
- B0.1 Validate on read: count (not silently drop) contract violations; train on valid only.
- B0.2 Dedupe: exact (key, ts, type, campaign-in-attributes) retry/re-send collapse [PREF].
- B0.3 Outliers via quantiles, not fixed values: per-brand amount caps, velocity caps, quantile time-gaps, dynamic sequence cap (P99 vs. memory, most-recent kept) [PREF]. Why quantiles: fixed caps rot with prices/volume.
- B0.4 Bad windows flagged, not deleted: downweight/exclude downstream, preserve ordering + replayability [PREF].
- B0.5 Logged: B0 version + thresholds in registry with every B release [LOCKED].

**Backbone [PREF].** SSM (e.g. Mamba-2).
- Why: O(n) training + O(1)-per-new-event updates — addresses the stated training-cost and daily-refresh constraints without losing long-range capacity.
- Why not Transformer: O(n²) training + growing inference state; wrong cost envelope for full-history daily refresh.

**Release.** Versioned, immutable; registry records data snapshot + config + gate results [LOCKED]. Serving mechanism/cadence TBD (daily batch is the leaning, unagreed).

**Rejected: Transformer + per-task fine-tuning [LOCKED].**
- Why not: N tasks → N encoders (incomparable, version explosion); each copy re-pays training + refresh cost; one task's gradients reshape another's state space. Permitted instead: frozen base + small C-side heads; base bit-identical for all consumers.

**Acceptance gates [LOCKED, metrics TBD].** Causality (future must not affect past embeddings, verified by ablation); quality (beats agreed baseline on agreed probes); cold-start reported by customer-age cohort. No freeze until these pass.
- Note on encoder overlap (not an issue, guarded): the encoder trains on all history, so it has seen the customers and periods it later scores (e.g. v1 trained through Dec 2025 scoring ABC-Jan-2025). This is accepted because (a) weights aggregate millions of customers toward general patterns — per-customer memorization is marginal, not headline; (b) the strict per-row input cutoff means no future events ever enter a historical vector; (c) all gate metrics run on held-out customers the encoder never trained on, so memorization cannot flatter reported numbers; (d) the online holdout remains ground truth. Train on everything; let only strangers vote.
- **Encoder vs. embedding [LOCKED].** The encoder is the machine, trained on all time. The embedding is one output, made by feeding that machine only events ≤ t. To get `S_c(July-1)`, pull raw events ≤ July-1 from A and run them through frozen `vN` — fetch the July-1 daily partition or recompute on demand. Today's vector is never used for historical training. The encoder having seen that customer in training is accepted (see overlap note) — it learned general patterns, not a copy of their future, and gates on strangers prove it.
  - Plain version: we train the encoder (SSM, not Transformer) with data through TODAY. When we train the plugins we need embeddings for input and we use the encoder to produce those but we produce those with history through D and not TODAY. Inference is through TODAY.
  - Why: simple cutoff kills state leakage; weight overlap is second-order. Why not retrain encoder per plugin window: destroys amortization for no measurable gain.
- Open: objective, `H` width, `S_t` size (sweep candidates, freeze smallest passing with margin), tokenizer, compute, serving store, baseline + probes.

### 3.x B-side analytic: HVA-mining (fully automated) [LOCKED shape]

Sits beside B (reads frozen CFM + A, writes nothing back). Zero human input: one RUN produces the ranked HVA table per CFM revision.
1. **Enumerate** candidate customer journeys + singular actions (2–4 step sequences over customer-side event types, composed programmatically from the taxonomy).
2. **Match** same-pre-state trajectories (nearest neighbors in `S_t` at `t₀`): took-journey vs. didn't.
3. **Derive incremental GM** per candidate (matched margin difference + path-propensity correction from the CFM's own predictive distributions).
4. **Rank and publish** table: `(journey, support_n, incremental_$_per_customer, CI, CFM_revision)` — support-weighted, uncertainty attached, revision-stamped. Refreshes automatically per CFM revision.
- Output feeds C/D design (which journeys the dose engine nudges toward) and validates against the OPE ensemble. Freely-chosen journeys carry maximal selection bias — matching + propensity + CIs are the defense; non-survivors don't ship.

---

## 4. Decisions (this conversation only)

| # | Decision | Status | Why / Why not |
|---|---|---|---|
| D1 | Unidirectional A→B→C, strict freeze, versioned API | [LOCKED] | Reproducibility/comparability; not backward edges or fine-tuning (explosion + coupling) |
| D2 | A sources = all | [PREF] | Sequencing signal; not transactions-only (blind to intent) |
| D3 | Databricks now, GCP later, portable | [PREF] | Gravity without stall; not single-lock (migration debt) |
| D4 | A = faithful raw 5-col; no cleaning in A | [LOCKED] | Optionality, no baked-in loss; not clean-in-A (irreversible, inherits one team's thresholds) |
| D5 | B backbone = SSM | [PREF] | Train + daily-inference cost with capacity; not Transformer (O(n²)) |
| D6 | Reject Transformer + fine-tuning | [LOCKED] | Comparability + cost; LLM playbook breaks frozen substrate |
| D7 | C supports Unsupervised, Supervised, Online RL, Offline RL + OPE on frozen embeddings | [LOCKED] | One foundation amortized; not single-paradigm (fragments into N foundations) |
| D8 | Cleaning = versioned B0 on consume, shared with C task builders; registry logs B0 version | [LOCKED] principle / [PREF] defaults (B0.1–B0.5) | State-view = label-view; not per-consumer cleaning (divergence) |
| D9 | B trains monthly, infers daily; C pins a B version and trains any time; no auto-migration on new B release | [PREF] cadence / [LOCKED] pinning | Freshness without churn; not retrain-B-per-C-task or silent upgrades (unreproducible) |
| D10 | One pinned embedding space per training build; cross-version windows recomputed under it, pay-per-re-pin | [LOCKED] | Comparability + reproducibility; not cross-version training (shifting languages) or monthly world-recompute (unamortized) |
| D11 | Backfill grid: BEST = daily over lookback + label window; GOOD = sampled within same bounds, recipe logged | [PREF] | Full fidelity vs. economy per build; not sparse-without-recipe (irreproducible) or latest-only (leakage) |
| D12 | Versions `vNrN` (N = code change, rN = data retrain); parallel-online revisions, never truncate on release; retire at zero pins | [LOCKED] | Safe per-plugin rollout + instant rollback; not flag-day migration (breaks all plugins at once) or truncate (destroys rollback + history) |
| D13 | Supervised template = tiny MLP (`128 → 64 → 1` + regularization + early-stop + calibration gate); XGB as challenger baseline | [PREF] | One net stack into RL incl. critic warm-start; not XGB-template (second stack, no composition) |
| D14 | Plugins' SOLE feature input is the N-dim CFM output — no side inputs | [LOCKED] | One point-in-time-correct contract; not statics concat (second versioned pipeline + hidden foundation weakness) |
| D15 | Plugin interface generality (action schema + reward fn + cadence; policy/value on 128-only); first family = weekly per-brand doses (email/push/sms 0–14, discount 0–40 step 5, factorized) | [LOCKED] interface / [PREF] action space | Example never leaks into platform; not bespoke-per-plugin (N stacks) or flat cross-product (coverage death) |
| D16 | Engine = IQL default + permanent bandit baseline ("does sequential pay?") | [PREF] | Sequencing premise + thin-coverage tolerance; not bandit-engine (horizon-blind) or cloning (status-quo ceiling) |
| D17 | Propensity-as-supervised-plugin + layered bias defense (clipped IPS, support floor, coverage receipt, natural-experiment checks) | [LOCKED] principle / [PREF] knobs | Targeted history lies without debiasing; not weighted-only (no veto) or standalone-matched (faith-based) |
| D18 | Reward v1: incremental core, no cost term, γ=0.99 infinite horizon; train bounded-attributable (12w default), gate longest measurable | [LOCKED] core+direction / [PREF] windows | Caused-not-hogged margin, long-aimed without mushy credit; not raw-margin (credit theft) or 365d labels (unattributable) |
| D19 | Baseline ladder (opportunistic holdout > natural experiments > matched controls > model-based; rung cited per segment) + per-brand robust-norm add-on; time-decay attribution default | [PREF] | Honesty-graded numbers with no holdout assumed; not holdout-dependent design or unscaled multi-brand training |
| D20 | OPE ensemble gate (IS + direct + DR + replay + sensitivity + receipt; explicit vetoes, coverage floor = auto no-go) | [LOCKED] shape / [PREF] specifics TBD | No single estimator trusted; not single-number gating (variance/bias blind) |
| D21 | Layer D = hierarchical meta-controller over C option-proposers; head contract = (dose, value_at_dose); D-rules v1 → D-learner on trigger, same reward throughout, learner trains on saved C outputs, rule projection retained as guardrail | [LOCKED] layer+contract+sequencing | Composition is the product; not joint-monolith (unlearnable) or headless clipping (value left behind); learner promotes only by beating rules |
| D22 | D scoring = ANOVA-factorized (mains from heads + sparse pairs); v1 additive (pairs armed-but-zero, ordered promotion queue) | [LOCKED] doctrine / [PREF] shortlist | Learn where data is, zero where it isn't; not full-joint (hallucination) or permanent-additive (interaction blindness) |
| D23 | HVA-mining = automated B-side analytic (RUN → ranked journey/action table with support, incremental $, CIs, per CFM revision; zero human input) | [LOCKED] shape | Customer journeys discovered from the foundation alone; not hand-picked HVAs (bias) or human-in-the-loop mining (unscalable, unversioned) |
| D24 | Layer D generalization (Super NBA: any plugins × any cadence → cards, MANY:MANY) guides design, NOT locked; Marketing Dose = the engine under construction | [DISCUSSION] | Single-engine vs many-engines-by-cadence + shared coordination still open; not prematurely locked |
| D25 | Timing scheduled post-composition in D (timing head picks day×daypart per prescribed send, same shared reward; trains as C plugin, runs as D-stage-2) | [LOCKED] placement | Dose space × timing space is coverage death jointly, learnable staged; not joint dose-timing heads or click-proxy timing |

---

## 5. Layer C — Consumers

**Contract [LOCKED].** C consumes frozen versioned embeddings and builds its own labels/actions/rewards from A. No writes back into A or B. One interface serves all five paradigms (D7). Heads take the 128-d `S_t` as their SOLE feature input — no side inputs, no concatenated statics; anything predictive must come through the event stream and earn its place in the embedding.

**Components.**
- C0. Task builders — labels, actions, and rewards derived from A **through B0** (same cleaning as B), with time cutoffs (designs TBD) [LOCKED: must import B0].
- C1. Unsupervised — e.g. lifecycle clustering on embeddings; outlier flagging. Why embeddings: usable distance metric.
- C2. Supervised — e.g. small heads predicting future purchase/churn/LTV labels. Why frozen + small head: minutes to train, shared versioning. Template [PREF]: tiny MLP (`128 → 64 → 1`, dropout + weight decay, early-stop on held-out, calibration gate); XGBoost runs alongside as challenger baseline — XGB beating MLP is a bug signal, not a ship decision. Warm-start path (supervised head → RL critic init) stays open by keeping one net stack.
- C3. Online RL — e.g. bandit over send-time/channel under guardrails; all decisions + outcomes logged to A. Why metered: live exploration is budgeted risk that enriches offline logs.
- C4. Offline RL — policies from logged histories + C-built rewards, no live exploration. Why offline-first: proof before margin spend.
- C5. OPE + gating — value estimates of candidate policies from logged data; owns the deployment gate. Why separate: converts "trained" into "bettable."
- C6. Constraint handling — mechanism TBD (in-model vs. decoupled open).
- C7. Execution logging — every C decision + outcome back into A (no gradient path into B).

**Open (remaining TBD).** IPS cap, support floor, attribution-rule finalization, training-window default, OPE estimator specifics + graduation thresholds, constraint handling (in-model vs. decoupled).

### 5.1 Plugin interface generality [LOCKED]

Every Offline RL plugin — present and future — meets one contract: it declares (action schema, reward fn, decision cadence) and implements `policy(S_t) → action` + `value(S_t, action) → expected reward` on 128-only inputs (D14). The weekly dose family below is the first tenant, not the architecture; future families (send-time, affinity, pricing) rent the same interface with no platform change.
- Why interface-first: the example's specifics (weekly, per-brand, doses) must never leak into shared machinery. Why not bespoke-per-plugin platform: N plugins → N stacks, the 40-pipeline failure mode repeated.

### 5.2 First action space — weekly per-brand doses [PREF]

Per customer, per week, per brand: `email_dose ∈ [0,14]`, `push_dose ∈ [0,14]`, `sms_dose ∈ [0,14]` (ints); `discount_depth ∈ [0,40 step 5]`. Dimensions factorized (joint ≈ 30k combos — intractable for propensity; independence-given-state assumed, documented, revisited on interaction diagnostics).
- Why doses, not campaigns: campaigns churn monthly; dose + preference composes across them without redefining the space. Why not flat cross-product: combinatorial coverage death.

### 5.3 Engine — IQL default + permanent bandit baseline [PREF]

IQL family (conservative, thin-BAU-coverage-tolerant). Sequencing is the premise (send-now-vs-hold is invisible to one-step methods). A one-step bandit/uplift baseline runs beside every plugin permanently as the standing test "does sequential thinking pay here?" — IQL failing to beat it is information, not failure.
- Why not bandit-as-engine: horizon-blind by construction. Why not cloning BAU: ceiling = status quo.

### 5.4 Bias defense — propensity as supervised plugin [LOCKED principle, PREF knobs]

History is targeted, not randomized — naive values learn merchant habits. Defense in layers: (1) factorized propensity heads (`P(action|S_t)` per dose dimension) built as supervised plugins on the same template/pins, registry-linked as each policy's twin; (2) clipped IPS-weighted critic (cap TBD); (3) explicit support floor per segment (never prescribe untried doses); (4) coverage receipt (effective sample size) at the gate; (5) natural-experiment cross-checks where targeting was suspended.
- Why propensity-as-plugin: same template, same grids, same pins — debiasing reuses the whole supervised apparatus instead of inventing a second one. Why layered, not weighted-only: weights debias, floors + receipts veto.

### 5.5 Reward v1 [LOCKED core, PREF estimators]

```
r = robust_norm( attributed_margin − baseline ),  γ = 0.99, infinite horizon. No cost term.
```
- **Incremental core [LOCKED].** The minus-baseline aims optimization at caused margin, not hogged credit. Send/push/SMS unit costs ≈ 0 (dropped); discount cost already lives inside margin.
- **Horizon [LOCKED direction].** γ = 0.99 weekly (≈2yr effective reach): far-future incremental GM drives today's Q-values through bootstrapping. Training labels stay bounded-attributable (realized increments; default 12w window [PREF], knob); the gate scores the longest honestly measurable window (up to the 365d standard). Short-trainable, long-aimed.
- **Baseline ladder [PREF order, rung cited per segment]:** (1) opportunistic micro-holdout if ever granted — calibrates everything below; (2) natural experiments (broad sends, outages, phased rollouts) as pseudo-holdouts — the primary workhorse, no business ask required; (3) matched controls via the same IPS machinery (assumption-heavy, never standalone — must agree with rung 2 where both exist); (4) model-based densifier for thin segments (aggregate trust only). No standing holdout assumed, ever.
- **Normalization [LOCKED add-on].** Per-brand robust scaling (median/MAD — margin tails are savage), clipped critic targets; de-normalize to dollars at OPE/reporting. Stability infra, not a fourth objective.
- **Overlap attribution [PREF default].** Time-decay credit to recent action weeks for purchases inside multiple windows; winner-take-all and even-split kept as challenged alternatives with sensitivity runs.

### 5.6 OPE ensemble gate [LOCKED principle, PREF specifics TBD]

No single estimator votes alone: importance-weighted returns + direct-model estimates + doubly-robust blends + temporal replay in dollars + sensitivity runs (edge must survive plausible model error) + coverage receipt. One gate, explicit vetoes — coverage below floor is automatic no-go regardless of lift. Estimator specifics, floors, shadow protocol, and graduation thresholds to be designed; the ensemble-with-vetoes shape is locked.

---

## 6. Layer B (serving half) — Training + Serving Embeddings [PREF cadence, LOCKED pinning]

This section is Layer B's operational core: how frozen models become daily state, how training builds get correct history, and how versions roll over without flag-days. §3 covers the modeling half; everything about producing, storing, retaining, and retiring embeddings lives here.

Monthly training, daily inference, any-time C training. Worked example: B releases on the 1st, a plugin trains on the 15th.

1. **1st — B trains + freezes.** B trains on a pinned snapshot (e.g. data through prior month-end, via A6 bulk scan through B0), passes §3 gates, freezes as `vN`. Registry logs data snapshot + B0 version + config. `vN` weights do not change until next month.
   - Why monthly, not per-demand: training is the dominant cost; monthly amortizes it. Why not less frequent: representation rots as behavior/taxonomy drift.
2. **Daily — B infers** into `customer_embedding_daily` with grain one row per customer per day (appended date partitions, never in-place overwrite). Column order: `customer_key, CFM_version, CFM_as_of_date, cust_as_of_date, S_t` — where `CFM_version` (e.g. `v1`) names the frozen weights (static all month), `CFM_as_of_date` is the model release date (static all month), and `cust_as_of_date` advances daily. Batch job runs frozen weights forward over new/changed histories only (SSM state carried forward, O(1) per new event) plus the cheap time-conditioned readout for quiet customers (§3 drift rule). Serving reads the latest partition; training reads historical partitions (point-in-time fetch, no recompute).
   - Why daily vectors under frozen weights: decisions need fresh state; fresh state must not require fresh weights.
3. **15th — plugin trains against pinned `vN`.** The plugin pins `version=vN` and fetches embeddings **as-of each training decision time**, not as-of the 15th (point-in-time correct via stored history or recompute through B0 + `vN`). Labels/actions come from A through B0 with strict time cutoffs. Registry logs `vN` + B0 version + training window + snapshot.
   - Why as-of-decision-time, not latest: training on 15th-state for a 1st-decision leaks the future. This is the same causality rule as §3, enforced at fetch time.
   - Why 14-day-old weights are fine: `vN` is a general representation; the plugin head learns the task mapping. Staleness costs a future re-pin, not a B retrain.
4. **Serving.** The trained plugin serves on the latest daily `vN` vectors + its frozen head. No B involvement per request.
5. **Next 1st — `vN+1` releases; nothing auto-migrates [LOCKED].** The 15th-trained plugin keeps serving on `vN` until its owner revalidates and re-pins to `vN+1`. Skipped versions are allowed.
   - Why no auto-migration: silent upgrades make "why did this decision change?" unanswerable and couple B releases to C incidents.

---

### 6.1 Retention — serve latest, retain window [PREF]

Downstream plugins serve on the latest partition per customer (one lookup: max `cust_as_of_date` for pinned `CFM_version`). Serving NEVER reads older partitions — no plugin serves on a stale vector, ever. History partitions serve training fetches only (decision-time features, §6 step 3 / §6.4). History partitions are retained for a window TBD with floor = longest C training-decision lookback + longest label window (a head training on 12 months of past decisions with 90-day labels needs ≥15 months; audit/replay margin on top).
- Why not latest-only: training rows need decision-time state (`S_c(t_decision)`, §6 step 3). Latest-only forces full-history recompute per training row or, worse, trains on leaked future state. Storage (~512B/row/day) is cheap; destroyed causality is not.

### 6.2 Worked example — ABC, v1, Jan 01 → Jan 03 [LOCKED process]

Formulas: `CFM_2026_01_01 = train(stream ≤ 2025-12-31, B0_v3)`; `S_c(t) = Readout_v1(Scan_v1(B0_v3(events_c ≤ t)), t − last_ts_c(t))`.
- **Jan 01 (release + first inference):** freeze v1; append `(ABC, v1, 2026-01-01, 2026-01-01, [0.22, …])`; cache `(ABC, v1, 2026-01-01, through 2025-12-31, H)`.
- **Jan 02 (quiet):** A query empty → cache untouched → readout with clock+1 → append `(ABC, v1, 2026-01-01, 2026-01-02, [0.21, …])` (inactivity drift, §3 function).
- **Jan 03 (late Jan-02 purchase surfaces):** A lands it faithfully → cache folds one gated `update()`, `H_updated_through` → 2026-01-02 → append Jan-03 embedding reflecting the purchase. Published Jan-02 row stands (partitions immutable); true Jan-02 state recomputable on demand through B0_v3 + v1.
- Daily rhythm (all cases): (1) cache: fold events since `H_updated_through`, advance marker; (2) embeddings: readout + append. Late arrivals need no special case.

### 6.3 One embedding space per training build [LOCKED]

All rows in a training build MUST come from a single pinned CFM version. Retained daily partitions span many monthly versions and are incomparable across version boundaries (same 128 slots, different meanings). Any training window crossing versions therefore requires a recompute pass under the pinned version — deterministic B0 + frozen weights over faithful A, identical to what would have been published. Prefer versions whose training data covers the window (recompute outside the trained era is extrapolation).
- Recompute is cohort-scoped (that consumer's customers, often a subsample) and follows re-pin cadence, not release cadence: pay per upgrade. Months with no re-pins pay no backfills. No auto-migration — old versions keep serving pinned plugins while backfills run.
- Why not train across versions: heads cannot learn across shifting embedding languages; results differ per version mix and reproduce never.

### 6.4 Backfill grid standard [PREF]

Bounds are fixed (§6.1 floor: decision lookback + label window, e.g. 365 + 365 = 730 days). Density within those bounds is a per-build choice:
- **BEST = daily** across the full bounds. Full fidelity, max storage/compute.
- **GOOD = sampled** within the same bounds: dense near decision and label-resolution dates (state moves fast, leakage hides there), sparse across quiet deep history — or event-anchored pairs (decision date, decision + label date) only.
- All sampled dates stay point-in-time correct (history ≤ t, pinned version, shared B0 — no exceptions for sparse grids). The sampling recipe (which dates, why) is logged with the dataset build, or the run is not reproducible.
- Serving grid stays daily regardless — grid freedom applies to training backfills only.

### 6.5 Ops cadence [PREF]

**Monthly (1st week).**
1. Clamp training snapshot + pin B0 version. Window is expanding-fixed-start: `[2019-02-01, last-day-of-prior-month]` — genesis never moves, head advances monthly (month-end, not run-date, so reruns clamp identically).
2. Train candidate CFM (SSM) on snapshot through B0. Revision bump (`rN+1`); version bump (`N+1`, reset `r1`) only on training-code changes.
3. Run acceptance gates (§3: causality ablation, quality probes vs. baseline, cold-start by cohort).
4. Freeze pass → `vNrN`; registry logs snapshot + B0 + config + gates. Fail → previous revision stands, no partial release.
5. Full state-cache rebuild under the new revision (all customers, full history re-scan — old `H` rows are a different space and unreusable) + remake `plugin_training_table` (materialized BEST grid: daily over lookback + label window, single pinned revision) for re-pinning consumers. Training grids are produced rolling: every monthly cycle remakes the grid over the slid window, so training-ready data under the current revision always exists — no plugin ever waits on a backfill to start training.
6. Parallel online [LOCKED]: daily jobs write the new revision alongside every revision with pinned plugins. Old revisions keep inferring fresh until their last pin moves. Tables are NEVER truncated on release; a revision's rows retire only after zero pins + retention floor (§6.1).
7. Re-pin window: plugin owners revalidate against `plugin_training_table` and flip pins on their own cadence. Pin-back is the instant rollback. Version retirement review: zero-pin revisions stop daily jobs, partitions drain per retention.

**Daily.**
1. Land + validate new A rows (contract counts, freshness).
2. Per customer with events since `H_updated_through`: incremental SSM state step(s) → rewrite `customer_state_cache` row, advance marker.
3. Per active customer under every live version (current + versions with pinned plugins): time-conditioned readout → append `customer_embedding_daily` partition. Quiet customers cost the readout only.
4. Publish latest partitions; run reconciliation (row counts, null/dup rates) + drift monitors. Page on breach, never degrade silently.
5. Drain C execution logs (decisions + outcomes) back into A — continuous, reconciled daily.

### 6.6 Two embedding categories [LOCKED]

Training and serving embeddings share a format but have separate lifecycles — never mix their retention:
- **Serving embeddings** (`customer_embedding_daily` partitions): written daily per live revision. Readers take latest only (e.g. CFM retrained on the 1st, daily partitions for the 1st–15th, plugin serving on the 15th reads the 15th — the 1st–14th are invisible to it). Retained per §6.1 floor because today's serving rows are next quarter's training history.
- **Training embeddings** (per-build grids incl. `plugin_training_table`): materialized under one pinned revision over the build's date grid (BEST daily or GOOD sampled, §6.4). Read during that build's training only. Droppable after the build — deterministically rebuildable from (revision + grid recipe + A) — though the recipe must stay logged.
- Why separate: serving optimizes for latest-freshness, training for point-in-time depth. One lifecycle rule for both either deletes training's past or drowns serving in it.

---

## 7. Layer D — Marketing Dose composer (composition layer; Super NBA generalization under discussion)

Layer D is the general composition layer — the Super NBA: it takes in as many C plugins as exist and, per decision cadence, picks the one/many actions that maximize the shared reward under constraints. MANY:MANY throughout: any engine subscribes to any plugin subset via manifest; any plugin feeds any number of engines. The Marketing Dose composer below is engine one (hierarchical meta-controller; the weekly Next Best Action); future engines (pricing, delivery-time, real-time surfaces) reuse the layer unchanged.

C heads propose per-dimension options; D commits the single weekly 24-dim card (6 brands × [email, push, SMS, discount]) the brands execute — that card IS the Next Best Action: per customer, per week, the best action-combination under constraints, with runner-ups and reasons attached. No separate NBA system exists or is planned; "NBA" names this output, not a new component. (A future interaction-time NBA — sub-second per-request actions for site/app — would be a serving *mode* on the same frozen vectors + heads, not new intelligence; gated on a latency-demanding consumer.) Hierarchy: C = option proposers, D = meta-controller. One reward (§5.5) at every level — no per-level shaping, so no level profits by gaming another.

C heads propose per-dimension options; D commits the single weekly 24-dim card (6 brands × [email, push, SMS, discount]) the brands execute. Hierarchy: C = option proposers, D = meta-controller. One reward (§5.5) at every level — no per-level shaping, so no level profits by gaming another.

**Components.**
- D0. Head contract — each C head emits `(dose, value_at_dose)`: its chosen dose plus predicted incremental margin (D's common arbitration currency). Runner-up gaps deferred.
- D1. D-rules (v1) — deterministic constrained argmax over the 24 pairs: eligibility masks, per-customer caps, margin floors, Lagrangian λ for brand budgets. Constraint config in YAML: rule changes ship without retraining anything. Every card records pins + runner-ups + binding constraints (the card explains itself).
- D2. Scoring doctrine — ANOVA-factorized `V(card) = base + Σ mains + Σ pairs`; mains arrive free from heads; v1 ships additive (pairs armed-but-zero) with ordered promotion queue (within-brand dose×discount → same-channel cross-brand → discount cross-brand), each term needing holdout survival.
- D3. D-learner (graduated) — same IQL stack, inputs = frozen state + saved 24 pairs, output = final card, trained on the D-rules operation log (proposals, values, constraints, cards, outcomes — assembled free). Promotes only by beating D-rules in the §5.6 ensemble; the rule projection stays as output guardrail permanently.
- D4. Decision card — 24 ints + audit tail (C pins, D version, values, bindings), logged to A with outcomes. Weekly batch, human-readable aggregate before send.
- D5. Timing scheduler (second stage, high interest) — per prescribed send, a timing head picks `(day_of_week, time_of_day)` (~28 slots) from `S_t` at scheduling time + prescribed dose + channel/brand context, maximizing the same shared reward (§5.5, γ=0.99) under slot masks (quiet hours, caps, cross-loop suppressions). Trains like a C plugin (same template/pins/rhythm); runs inside D after D1 composition, so layers stay unidirectional (C proposes, D composes then schedules). Card audit tail extends per send (timing pin + slot). Rationale: history is rich in quasi-randomized timing variation (batch hours, timezones, throttling jitter) where it is poor in dose variation — timing is the most learnable dimension in the system. Short-proxy click-timing stays as permanent challenger baseline.

---

## 8. Build order

1. A6 verification (bulk + lookup + snapshot pinning) → A declared done.
2. B0 + foundation pipeline (SSM PREF) → clean end-to-end sample run.
3. Foundation gates (§3) → versioned freeze (snapshot + B0 version logged).
4. Daily inference job + point-in-time fetch (§6) → any-date `S_t` retrievable under a pinned version.
5. Minimal C pilots via B0-shared task builders → all five run on frozen contract, no backward edges.

---

## 9. Open items

A6 mechanism confirmation. B: objective, `H` width, `S_t` size sweep, tokenizer, compute, serving store + daily-job owner + retention window, baseline + probes, B0 threshold finalization. C: per-paradigm v1 scope, reward/OPE/constraint designs, re-pin policy on new B versions. Platform: portability mechanism.
- Deferred (noted, not designed): Mixture-of-Experts (MoE — v2 capacity lever on measured per-brand residual trigger).
- B-side analytic family (beside B, revision-stamped, off the live path): HVA-mining (§3.x, locked) + world model / learned dynamics (OPE panelist + what-if tool + D-learner imagination fuel; graduates from standby the day DR and replay disagree on a long-horizon call).

---

*End v0.10 (Layers A–D). Next: constraint config + OPE estimator specifics.*
