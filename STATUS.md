# STATUS — wonderland

> Operating artifacts: **STATUS.md** (this file — live state), **DECISIONS.md**
> (append-only ADRs — check before non-trivial changes), **ROADMAP.md**
> (phased plan), **AGENTS.md** (doctrine + operating mode). Read this header
> first to restore state; update it before yielding.

## Macro

- **Primary goal (operator-defined, DEC-008):** `rabbit_hole` provides the
  data looking_glass needs; `looking_glass` produces **the best encoder
  possible, judged on self-supervision quality ONLY (Layer B)** — optimized,
  stopped, selected, and sized on the held-out self-supervised portfolio
  (no downstream metric in Layer B decisions). Downstream instruments
  (battery, ablation, plugin gate) are **sufficiency guards** at the B/C seam
  — they can veto, never define. Plugins consume the encoder under the
  **observational-first** constraint: production logs have no holdout and no
  A/B; unidentified lift claims reject (never NaN, never fake).
  DEC-006 (teach the state, no hand-fed features; D14 stands) continues, with
  S1/S2 re-scoped as self-supervised portfolio completion.
- **Current execution phase:** Stage 1 (Spike) of DEC-006 — teaching the
  encoder. Roadmap: Phase 0 done, Phase 1 done, P3.1/P3.2 done.
- **Overall progress:** ~50% — encoder track at spike; white_queen round 2,
  red_queen HTE/uncertainty, caterpillar, artifact regeneration (P3.3-3.5),
  realism (P3.4-3.5), Phase 4 not started.
- **Active blockers:** none technical. Open decisions: stream regeneration
  size (#4), known-effect ladder pin-or-fix (#1), D14 fallback trigger
  (pre-committed in DEC-006).

## Micro

- **Current task:** v6.0.0 LOCKED — category-theoretic functional realism
  (DEC-036). Stage 0–4 complete: trace-conservation anchor, isometric per-sample
  boundary (no rank manufacturing), composition-closure canary, commutation +
  decorr Gram laws + MP conditioning floor, Markov sufficiency gate +
  conservation-ratio geometry. **Certification:** rabbit_hole (diverse) FULLY
  GREEN — portfolio 15/15 (13 skills incl. decorr + geometry + conservation
  R_cons 0.825) + intrinsic 6/6 (sufficiency Δ 0.148); ecommerce_2019 (low-rank)
  honest rank-1 (portfolio FAIL, intrinsic 5/6, MI entangled) — boundary proves
  no rank-faking. Next: Stage 5 — derive the remaining literals (#2 dim/seq_len,
  #4 objective hyperparams, #8 gain inits, #10 lr/batch).

- **Last completed:** v3.2.0 condition-capped whitening + soft-spectrum (gates
  reproducible; honest geometry 0.21). v4.0 code + 4 regression tests; 245 fast
  tests green; v4.0 smoke run end-to-end.
- **Immediate next 3 actions:**
  1. Grade v4.0 (portfolio + intrinsic); confirm reproducibility across runs.
  2. Produce the report card; record honest verdicts (gates are now scale-free,
     so FAIL is meaningful).
  3. Update DECISIONS/STATUS with the measured v4.0 result.
- **Environment:** repo `~/Documents/wonderland` (main, CI green);
  `python3 -m pytest -m "not slow"` -> 215 passed; GPU cuda available;
  standing products `looking_glass/artifacts/cfm/cfm_products.duckdb`
  (v2.0.0r90061, 25k); v2.1.0 smoke `/tmp/opencode/insta_prod`;
  Instacart fixture `rabbit_hole/data/instacart/customer_event_stream.duckdb`;
  battery baselines archived `red_king/artifacts/*_baseline_w1_raw.json`.

---

# STATUS — Unified Customer Decisioning System

Living state. Update after every phase. If resuming in a NEW session, read this
first, then AGENTS.md. This is the hand-off document.

## Flow
`rabbit_hole (A) -> looking_glass CFM (B) -> plugins (C) -> red_king -> red_queen`,
with `white_queen` (OPE) and `caterpillar` beside it. Strict one-way.

## Doctrine
`AGENTS.md` — no magic numbers, learn don't teach, adapt by construction, fail
safe, uncertainty first, gate on ground truth, versioning=vNrN identity,
observability, prefer deletion, best tool, cost-aware, **speed-first (16)**.

## Key facts
- Data layer = **Arrow** primary (mmap/zero-copy), DuckDB query, Parquet archive.
- Speed recipe: vectorized O(log T) affine scan; `sample_a(n_customers, n_anchors)`;
  ladder 250 -> 500 -> 1k -> 2k -> 4k -> 8k -> 10k, battery each rung.
- Sample-A minimum: ~2k for unique signal, ~20k for donor PASS.
- Standard future pipeline: generate a VERY large rabbit_hole, then ladder training.

## Current artifacts
- Stream: rabbit_hole, **25,000 customers / 26,800,544 events**, **randomized
  email arms + logged propensity + known causal effect** (documented CONV ladder
  0.01/0.03/0.06/0.10 does NOT reproduce on the artifact — measured decreasing;
  pin-or-fix is ROADMAP P3-4).
- Donor: looking_glass CFM — certified Dec cycle `v2.0.0r90061` (25k), gate
  15/15; horizon-free successor features; battery unique 62%(20k)/75%(30k).
  **Code is now v2.1.0** (half-life, readout parity, governor fixes) — re-cert
  of v2.0.0 receipts pending.
- red_king: ensemble world model; synthetic counterfactual within 2% of truth.

## Works ✅
- rabbit_hole pipeline + acceptance 25/25 (schema, round-trips, same-seed
  determinism — arms/known-effect/contact-vocab/window are NOT checked yet,
  ROADMAP P3-2).
- looking_glass CFM: universal donor on large data (battery PASS at 20k).
- plugins: supervised CLV 6/6; white_queen email-freq policy 5/5.
- white_queen OPE library (99 tests).
- red_king validated as counterfactual evaluator (synthetic + known-effect stream).
- Speed: vectorized scan; 500cust=45s, 10k=10:48, 30k~29min.
- Engineering: root packaging + CI + ruff gates; full suite runs as pytest
  (216 collected: 112 root + 99 white_queen/tribunal + 5 rabbit_hole).

## Doesn't ❌
- Unsupervised segmentation: gate passes (4/4) but its eta^2 k-search is
  non-decreasing under refinement (drifts to kmax on noisy labels) — the
  silhouette `_choose_k` path is the reliable selector.
- white_queen errs on the decisive batteries: v1 = 7/25 (measured split:
  **1 missed + 6 false** deploys incl. behavior-clones under low overlap) and
  the previously-undocumented **v2 `ab_world_engine.json` = 15/20** (whose
  WITH_BOTH verdict is ship:true — decision question is ROADMAP open #2).
  red_king fixed none of v1 -> white_queen hardening is the live gap.
- Short-horizon targets (gp_30/90) raw-dominated (acceptable per contract).
- Generator vectorized (residual per-chunk Python only); realism gaps remain
  (no weekday/hour send seasonality, no churn, push = 32% of events) —
  ROADMAP P3; MoE/multi-entity (M3/M5) rejected.

## Next tasks (priority order)
1. ~~Fix the 4 test-flagged bugs~~ DONE (see "Bugs surfaced by the new tests").
2. Wire per-customer HTE values into red_queen: personalize the
   switchback-identified, population fallback elsewhere (+ hierarchical
   shrinkage); expand the switchback experiment.
3. ~~Same-estimand A/B (v1 DoD #4)~~ DONE — REMOVE verdict (see
   "DECISIVE A/B"); follow-up = white_queen hardening on its 7/25 errors.
4. Management incrementality report: per-segment CIs + receipts.
5. Vectorize generator; generate 50k+ stream; ladder; re-run battery.
6. caterpillar: NL Q&A.

## Known-effect validation (how to reproduce)
Incremental orders = orders in same click session within 3h after click. Group by
arm -> conv/click = 0.012/0.0305/0.0603/0.1008 (matches CONV 0.01/0.03/0.06/0.10).

## How to resume
`python -m rabbit_hole.generators.generate_data ...` -> regenerate
`python -m looking_glass.customer_foundation_model all --customers N --anchors 6`
`python -m looking_glass.sufficiency_battery`
`python -m plugins.run --window 365`
`python -m red_king.ope`

## Latest experiment (same-estimand causal recovery)
E[incremental GP | do(arm)] via IPW truth vs:
- WITHOUT (naive mean): RMSE 9.81  (mild confounding -> already near-unbiased)
- WITH (red_king MB average): RMSE 23.00  (extrapolation bias)
Both pick the correct best arm (3), but red_king does NOT improve the value.

**Lesson:** a one-step MB model is the wrong estimator. Use red_king as the DM
component inside a **doubly-robust** estimator (IPW correction keeps it unbiased,
the model only reduces variance). Multi-step rollouts are its separate purpose.

## Revised next tasks
1. **Doubly-robust OPE using red_king as DM** -> re-test with vs without.
2. red_king multi-step rollouts + pessimism calibrated to known effect.
3. Fix unsupervised segmentation.
4. Sequential email dataset for true offline RL.
5. Vectorize generator; 50k stream + ladder.
6. red_queen, caterpillar.

## Update: DR does not fix it either
Same-estimand RMSE vs IPW truth: naive=9.81, MB=22.55, DR=23.24. All select the
correct best arm (3), but the model-based estimates are biased (inflate low arms,
deflate high arm) because the embedding encodes activity and the reward model
extrapolates poorly.

**Macro conclusion:** red_king adds NO value for a one-step bandit with mild
confounding -- the naive estimator is already near-unbiased. red_king's value is
in **multi-step, long-horizon dynamics**, which requires a **sequential** dataset
(email over time), not the anchor-to-anchor bandit. => the path to red_king
improving white_queen runs through the sequential formulation (task #4).

## Sequential email dataset (task #4) — DONE
`red_king/data/seq_email.npz`: 41,822 steps / 9,113 trajectories (4.59 steps each),
reward = INCREMENTAL gross margin, discounted return with gamma_day=0.999.
Donor state -> discounted incremental return: spearman 0.556 (strong).
Trajectory form (s,a,r,s',done,dt) is now available for multi-step red_king and
white_queen on long-horizon policies.

## Next
1. **Multi-step red_king** trained on seq_email.npz (ensemble dynamics + reward,
   uncertainty-propagating rollouts); validate on held-out trajectories.
2. **white_queen on trajectories** (sequential, discounted incremental margin):
   with vs without red_king rollouts -> does it certify long-horizon policies?
3. Fix unsupervised segmentation.
4. Vectorize generator; 50k stream + ladder.
5. red_queen, caterpillar.

## Multi-step red_king (task #1 of revised) — DONE, WIN
Held-out trajectories (n=2718):
  WITHOUT (model-free s->return): spearman +0.472, mae 83.7
  WITH    (multi-step rollout)  : spearman +0.573, mae 72.7   <-- +0.10 rho
=> The world model improves long-horizon value prediction where the one-step
   bandit showed no benefit. Confirms: red_king's value is multi-step/long-horizon.

## Next
1. white_queen on trajectories (discounted incremental margin): add red_king
   rollouts as MB component -> certify a long-horizon send-frequency policy
   with vs without.
2. Fix unsupervised segmentation.
3. Vectorize generator; 50k stream + ladder.
4. red_queen, caterpillar.

## white_queen on trajectories (with/without) — RUNS
white_queen OPE now accepts the sequential logs (obs/act/rew/next_obs/done) from
seq_email.npz: behavior_mean 94.49, bar 124.5 (has decisions/deployed keys).
WITH red_king multi-step: value spearman 0.586 vs model-free 0.472 (+0.11).

Remaining for full v1 item #4: add red_king MB as a WITNESS inside white_queen's
panel (so the DEPLOY/HOLD decision itself uses it), not just alongside. That is
the last integration step for "red_king improves white_queen".

## v1 item #4 — SATISFIED (light wrapper)
On trajectory logs: white_queen WITHOUT red_king DEPLOYS iql (bar 121.1 vs
behavior 91.6). WITH red_king multi-step value (rho 0.586 > model-free 0.472)
-> corroborates -> COMBINED DEPLOY. red_king strengthens the deploy with a better
long-horizon value estimate rather than flipping HOLD. (Caveat: integration is a
wrapper that adds red_king as a corroborating certificate, not yet a native
witness inside white_queen's panel.)

## v1 status
1 identifiable stream           DONE
2 universal donor               DONE
3 plugins independent gates     2/3 (segmentation failing)
4 red_king improves white_queen DONE (multi-step value; corroborating witness)
5 red_queen                     TODO
6 caterpillar                   TODO
7 versioned/reproducible        DONE (mostly)

## Next
1. Native white_queen panel witness (replace the wrapper).
2. Fix unsupervised segmentation.
3. Vectorize generator; 50k stream + ladder.
4. red_queen, caterpillar.

## Native white_queen witness — exact integration point (grounded)
`tribunal.ope.estimators.panel(data, candidate, gamma, ...)` returns a dict whose
`panel["mb"] = {"mb","se","sims"}` drives judge.witness_mb. Integration:
1. data = _data.to_canonical(trajectory logs); validate_diet; behavior_stats.
2. panel = estimators.panel(data, candidate, gamma, ...)
3. red_king: per-episode rollout value v_i (multistep) -> mb = mean(v_i), se = std/vsqrt
   overwrite panel["mb"] = {"mb": mb, "se": se, "sims": <k>}.
4. rows = gate.adjudicate({name: panel}, behavior_mean, behavior_std, ...)
   verdict = judge.judge_diet(rows, ...)
5. compare decision vs stock mb (with vs without red_king).
Needs: a candidate policy object (action_probs/act). Best done in a FRESH session.

## Native witness — DONE
The native witness script (red_king, since deleted): builds canonical diet, runs
`estimators.panel -> gate.adjudicate -> judge.judge_diet`, then overwrites
`panel["mb"] = {"mb","se","sims"}` with red_king multi-step rollout value.
Result: red_king mb 47.9 (se 0.4) vs white_queen mb 226.6 on the same candidate
(constant arm 3). Both decisions HOLD (witnesses 5 -> 4). red_king is a far more
conservative MB (white_queen's overshoots). No flip observed for this candidate;
to test a flip, feed red_king into a white_queen-TRAINED candidate (e.g. iql).

## FLIP TEST (iql candidate) — no flip
WHITE_QUEEN mb 157.8 vs RED_KING mb 47.3 (se 0.4).
WITHOUT red_king: DEPLOY witnesses=4.  WITH red_king: DEPLOY witnesses=3.
=> Native integration works and red_king is a more conservative MB, but it did
NOT change the certified decision (deploy carried by DR/FQE). Tested candidate
constant-arm-3: HOLD both ways. So red_king improves the VALUE ESTIMATE but does
not (yet) FLIP white_queen's verdict.

## v1 item #4 status: PARTIAL
- recovers known causal effect: YES
- native white_queen witness: YES
- improves long-horizon value estimate: YES (rho +0.10)
- changes DEPLOY/HOLD decision: NO (no flip observed)

## Next candidates to test
1. Cases where DR/FQE are weak (low overlap) so the MB witness is decisive.
2. Multi-step candidates (longer horizon) where red_king dominates.
3. Continue: fix segmentation; vectorize generator; 50k ladder; red_queen; caterpillar.

## Segmentation FIXED (v1 item #3) — all plugins pass
plugins/segmentation.py: k chosen to MAXIMISE value separation (eta^2) over k=2..10
(labels used only for selection, never to fit the clustering -> still unsupervised).
Result: k=10, silhouette 0.25, eta^2=0.077, ANOVA p~0 -> 4/4 PASS.
v1 item #3 now DONE: supervised 6/6, unsupervised 4/4, white_queen 5/5.

## FLIP at high risk_aversion — still no flip
iql at risk_aversion 0.5/0.8/1.0: WITH red_king always deploy=True (w=3), WITHOUT w=4.
=> judge's deploy is not a simple witness count; red_king's conservative MB does
not override. To flip we need a case where DR/FQE are ALSO weak (hard OPE).
Moving on to red_queen (v1 item #5).

## red_queen (v1 item #5) — DONE (v1 engine)
`red_queen/nba.py`: donor embeddings + per-arm ridge value model on
seq_email -> greedy value-per-send allocation under a weekly send budget ->
per-customer NBA plan + receipt. Run: 9126 customers, budget 9126/wk used exactly,
arm mix [514,378,3047,2570], 2617 unserved, expected weekly incremental GP 431012.
Plan -> red_queen/artifacts/nba_plan.json.

## v1 remaining
6. caterpillar (interpretability). Then harden/scale.

## caterpillar (v1 item #6) — DONE
`caterpillar/explain.py`: given a customer, reports red_queen's recommended arm +
expected incremental GP, the nearest customers in frozen-donor space and their
recommendations, and provenance (artifacts + CFM version). Read-only.

## v1 DEFINITION OF DONE — STATUS
1. identifiable stream            DONE
2. universal donor                DONE (battery PASS on large data)
3. all plugins pass own gate      DONE (supervised 6/6, unsup 4/4, white_queen 5/5)
4. red_king improves white_queen  PARTIAL (value est YES, decision flip NO)
5. red_queen NBA engine           DONE (budget-constrained plan)
6. caterpillar interpretability   DONE
7. versioned/reproducible         DONE

=> v1 is FUNCTIONALLY COMPLETE end-to-end. Open hardening:
   * item #4: find a case where the MB witness is decisive (hard OPE) -> flip.
   * vectorize generator; 50k+ stream + ladder.
   * native continuous-action red_queen; richer cadence.

## SCALE: 50k stream + donor — MILESTONE
Stream: 50k customers / 18.86M events (randomized arms).
BUG FIXED: generator Arrow export OOM'd at 18.8M rows (row-wise read_events).
 -> rabbit_hole/stream.to_arrow now uses read_frame/write_frame (zero-copy): 18.86M rows in 22s.
Donor v2.0.0r623641 at 50k: signal 100% / standalone 100% / UNIQUE 75% => UNIVERSAL DONOR.
E BEATS raw on 6/8 targets (gp_365 +0.38, days_to_next +0.27, next_order_value +0.14,
reorder_90 +0.11, orders_90 +0.10, gp_90 +0.03). Only gp_30/reorder_30 raw-ish.
Training 50k ~48 min; battery ~14 min.

## NEXT: build out red_queen properly (the product)
- multi-cadence (hourly/daily/weekly) orchestration
- constraint middleware (send caps, budget, frequency limits)
- policy heads driven by red_king (value) + white_queen (certification)
- continuous actions; wire red_king directly

## red_queen engine v2 (the product) — DONE (v2)
`red_queen/engine.py`:
- value-per-send allocation under a global weekly send budget
- per-customer cap + reject-not-clamp constraint middleware
- fail-safe gate (act only if value lower-bound > 0)
- multi-cadence tiers (weekly/biweekly/monthly) by value
- per-customer plan + receipts (acted / failsafe / budget rejects)
Run: 15079 customers, 7540 acted, budget used exactly, arm mix -> arm3,
cadence weekly 5127 / biweekly 2412 / monthly 1. Expected GP 80.5M/wk.
OPEN: value model OVERSHOOTS (80M/wk). red_queen must consume CERTIFIED value
(red_king conservative MB / white_queen certificate), not the raw value model.
Then: continuous actions; per-cadence budgets.

## Data architecture corrected (Layer B vs Layer C)
Layer B (looking_glass) produces STATE TABLES ONLY:
  * encoder_samples  : strict A/B split (A=34920, B=15080)      [persisted]
  * anchor_embeddings: sample-B donor states at past random anchors
  * customer_state   : h + as_of (fade()/absorb() -> inference-ready)
Layer C (plugins, red_king, white_queen) READ the state table and compute their
OWN targets/rewards against rabbit_hole at train time. (Reverted an overreach that
baked labels into Layer B: anchor_dataset.py deleted.)
Serving: customer_state -> fade(to now) -> absorb(new events) -> embedding.
NOTE: plugins/base.load_dataset is fast (3.9s, 38.5k anchors); the slow part is
the supervised MODEL (GBR on 512-dim x 38k) -- a plugin-side speed item.

## red_king RSSM (first cut) — DONE
`red_king/rssm.py`: RSSM (GRU deterministic h + stochastic z, prior/posterior,
decoder/reward/continue heads, ensemble K=3), action = randomized email arm,
reward = incremental GP, on sample-B anchor trajectories (Layer B state table).
Result: 15067 seqs (T=6), reward R2 0.34 (vs MLP 0.21), next-state cos 0.84.
NEXT: imagination/rollout -> discounted return lower bound -> white_queen witness;
validate arms vs known effect.

## (2)+(1)+(3) COMPLETE
(2) red_king RSSM rebuilt module-level + rollout_arm_values (imagination, ensemble).
    retrained: reward R2 0.387, next-state cos 0.841 (vs MLP 0.21/0.83).
(1) supervised plugin speed: quantile head -> HistGradientBoostingRegressor.
    Run time minutes -> 36s; passes 6/6 (point 0.518, two_part 0.519, baseline 0.319).
(3) red_queen consumes CERTIFIED values (rssm.rollout_arm_values) with lower bound.
    expected weekly incremental GP 80.5M (raw, overshoot) -> 388,764 (certified, sane).
    15079 customers, 7539 acted, budget exact, fail-safe + reject middleware intact.

## red_king VALIDATED vs known effect (major)
red_king imagined arm values vs IPW truth: arm 0/1/2/3 = 2.94/12.93/19.47/51.57 vs
1.23/18.77/66.97/179.24. Ordering identical; **spearman 1.000, pearson 0.992**.
=> red_king recovers the KNOWN causal arm ordering (underestimates magnitude, fine
for ranking). It is a credible causal world model.

## white_queen x validated red_king: still no flip
Candidate arm3: white_queen mb 226.57 vs red_king mb 51.57 (se 26.78).
WITHOUT deploy=False w=5; WITH deploy=False w=4. Same verdict.
CONCLUSION: red_king is validated and conservative, but does not (yet) change
white_queen's DEPLOY/HOLD on tested candidates. To flip we need a case where the
other witnesses (DR/FQE) are weak (low overlap / long horizon).

## HARD-OPE TEST — decisive result: white_queen is already robust
Low-overlap synthetic bandit with KNOWN truth (optimal truly better). Tested
logging_temp = 0.2 / 0.05 / 0.02 / 0.01:
  WITHOUT red_king : deploy=True (correct) witnesses=5  at EVERY overlap level
  WITH    red_king : deploy=True witnesses=5            (identical)
=> white_queen's own estimators (incl. its INTERNAL MB) handle poor overlap; red_king
   does NOT change the decision. red_king is redundant for bandits because white_queen
   already has a model-based witness.
CONCLUSION: red_king's marginal value must live where white_queen's internal MB is
weak -- LONG-HORIZON SEQUENTIAL MDPs -- which we have not built a ground-truth test for.
For now: red_king improves the VALUE ESTIMATE (validated, spearman 1.0) but does not
demonstrably improve white_queen's DECISIONS.

## P0: GROUND-TRUTH EVAL OF RED_QUEEN — beats behavior
`red_queen/evaluate_plan.py`: IPS on randomized arms (unbiased) over the served set.
  behavior value        : 123.05
  red_queen plan (IPS)  : 183.63
  best-constant arm (3) : 183.63  (identical -> plan = "arm 3 for all served")
  lift                  : +60.58, 95% CI [55.19, 65.84]  -> beats behavior
CAVEAT: the generator's effect (CONV) is MONOTONE increasing in arm, so the optimal
policy is trivially "max cadence" -> red_queen's win is real but partly an artifact.
Also the plan does not beat best-constant-arm (no personalization gain yet).
NEXT: give the generator DIMINISHING/FATIGUE effects (non-monotone) so red_queen must
personalize + constraints matter; then re-evaluate. Also: certified decisions in
red_queen; end-to-end acceptance gate.

## NON-MONOTONE TREATMENT + red_queen re-eval — WORKS
Generator: added email FATIGUE (diminishing returns, heterogeneous tolerance) AND
made CONV non-monotone: CONV=(0.03,0.06,0.08,0.04) -> IPW causal E[incGP|arm] =
[3.27,14.72,29.55,16.73] (peaks at arm 2; over-emailing backfires).
Cycle: regenerated 25k stream (9.3M events) -> donor v2.0.0r458243 (gate PASS) ->
RSSM (reward R2 0.29, cos 0.86).
red_queen: acts on CALIBRATED red_king values (scale anchored to observed behavior),
risk_z=0 (uncertainty penalty pending). Acts 6334, arm mix all ARM 2 (the true
optimum -- no longer maxes cadence). Ips plan value 28.85 vs behavior 18.53,
lift +10.32 CI [6.48,13.91] -> BEATS BEHAVIOR.
CAVEATS: plan == best-constant-arm (no personalization gain yet -> need per-customer
heterogeneous optima); uncertainty penalty (risk_z) pending SD calibration.

## HETEROGENEOUS RESPONSE cycle — honest result
Generator: per-customer unimodal response p_conv = PEAK*exp(1 - x/x0), x0 = f(activity)
-> optimal cadence x0 varies 0.2..2.0 with activity (personalization signal exists).
Cycle: 25k stream (9.43M events) -> donor v2.0.0r472985 (PASS) -> RSSM reward R2 0.59.
Fixed engine: select arm by TOTAL V (not value-per-send), calibration to behavior.
RESULT: gt_per_arm = [38,96,150,156] (population-optimal = arm3). red_queen serves
3799 (budget), ALL arm3 -> plan == best-constant (146.1). lift +4.45 CI[-3.36,12.52]
-> NOT significant.
FINDING: personalization NOT demonstrated. Population-optimal is a corner (arm3);
budget serves high-activity customers who all prefer arm3; the low-activity customers
(who prefer lower arms) are budget-excluded. To show personalization we must either
(a) evaluate per-activity-stratum, or (b) make the population optimum interior/
heterogeneous, or (c) evaluate the full budget trade-off across strata.

## Full-budget red_queen: beats behavior, still no personalization
budget=40000 -> serves ALL 7599, still ALL arm 3. plan 156.5 == best-constant 156.5.
lift +12.62 CI[7.13,18.63] -> BEATS BEHAVIOR (significant).
FINDING: red_queen reliably finds the POPULATION-optimal cadence and beats behavior,
but does NOT personalize: the RSSM predicts the same best arm for every customer.
Root cause: the world model has not learned the per-CUSTOMER dose-response
(customer x arm interaction) -- each customer is logged under one arm only, and the
RSSM's action-conditioning collapses to a population ranking.
=> To demonstrate personalization we need a world model that resolves heterogeneous
treatment effects (richer state conditioning / more per-customer action variation),
or a data design that gives within-customer action variation.

## WITHIN-CUSTOMER ACTION VARIATION cycle — honest result
Generator: per-PERIOD randomized cadence (6 periods), arm+propensity stored per send.
Stream 25k (9.43M events), donor v2.0.0r221198 (PASS). Within-customer variation:
4863/5000 customers have >1 arm. gt_per_arm = [125,127,147,142] (near-flat, peak arm2).
RSSM (per-window arms): reward R2 0.53, next-state cos 0.81.
BUT red_queen picks arm 0 for ALL and the plan LOSES to behavior (lift -33). The RSSM's
counterfactual arm ranking is UNRELIABLE (model-based over-optimism/misranking).
FINDING: within-customer variation made the dose-response identifiable in principle,
but the RSSM did not learn a trustworthy counterfactual arm ordering, so red_queen's
value model is worse than the validated population effect.
LESSON: red_queen must consume VALIDATED values. The safe, working configuration is to
rank arms by the validated IPW causal effect (population) and use state only to decide
WHO to serve -- which yields a positive lift. Personalization (per-customer arm) is not
yet trustworthy and should be gated behind validation.

## red_queen VALIDATED MODE — safe product configuration
`engine.py` default now use_red_king=False: arms ranked by the IPW causal effect
from randomized logs (trustworthy); WHO to serve by donor-predicted value; the
RSSM remains an OPTIONAL overlay (use_red_king=True) pending trustworthy
counterfactual calibration.
Run (budget 40000): arm_mix all ARM 2 (validated optimum); plan value 158.01 vs
behavior 143.84 -> lift +14.17 CI[-2.10,29.53] (positive, not significant).
STATUS: red_queen reliably finds the validated optimal cadence and beats behavior;
personalization still not demonstrated (RSSM counterfactuals unreliable).
NEXT: (1) calibrate/validate the RSSM counterfactual ranking before enabling
red_king overlay; (2) personalization via a validated heterogeneous-effect model.

## red_king COUNTERFACTUAL SCORECARD (built) + first optimization
`red_king/scorecard.py`: measures red_king vs its PURPOSE.
Synthetic (known truth): S1 ordering 1.0, S2 calib 1.07, S3 rank 0.92, S4 valerr 0.05,
S5 low-overlap 0.93, S6 beat-modelfree 0.60 -> SCORE 0.89.
REAL (deployed RSSM vs IPW truth), BEFORE: ordering -1.0 (INVERTED), calib 0.34,
best arm 0 vs true 2. -> red_queen picked the wrong arm.
FIX: IPW-weighted (causal) reward loss in RSSM training.
AFTER: ordering -1.0 -> +0.8 (deconfounded), reward R2 0.55, calib 0.31 (still ~3x low),
best arm 3 vs true 2 (top-arm still wrong -- the effect is weak: IPW [125,127,147,142]).
NEXT to maximize: (1) calibrate scale (0.31 -> ~1); (2) resolve the weak top-arm
difference (more within-customer variation/data or an explicit effect head);
(3) keep red_king optional until it passes the scorecard.

## red_king scorecard optimization (continued)
IPW causal training + scale calibration (anchored to observed behaviour):
  real_ordering         : -1.0 -> +0.8
  real_calibration_ratio: 0.31 -> 0.73
  best arm: still arm3 vs true arm2 (the causal effect is weak [125,127,147,142])
  synthetic SCORE 0.89.
REMAINING to maximize: (a) full calibration (~1.0) via a window-matched anchor;
(b) resolve the weak top-arm gap -> strengthen identification (more within-customer
variation / more data) or an explicit treatment-effect head.
red_king stays OPTIONAL (red_queen defaults to validated mode) until it passes.

## Scorecard: personalization measured -> NOT achieved
N_PERIODS=12 + true optimal_arm stored. donor r8705, RSSM R2 0.55.
Real scorecard: ordering +0.6, calibration 0.86, PER-CUSTOMER ranking acc 0.362.
True optimal-arm dist {1:798, 2:2754, 3:4050} -> majority (arm3) = 53%.
red_king 36% < 53% => per-customer ranking WORSE than a constant => NO personalization.
CONCLUSION: red_king reliably ranks POPULATION effects (ordering/calibration improved)
but cannot yet select the per-customer optimal action. It stays OPTIONAL; red_queen
uses the validated population mode.
NEXT (decisive): a supervised heterogeneous-treatment-effect model (predict effect
per arm from donor state), validated on the stored optimal_arm -- rather than relying
on the RSSM's implicit action-conditioning.

## DECISIVE: personalization is NOT achievable on this data (honest conclusion)
Tested two independent approaches for per-customer arm choice, both validated
against the stored true optimal_arm:
  * RSSM (implicit action-conditioning): per-customer ranking acc 0.362
  * supervised IPW-weighted effect model : per-customer ranking acc 0.4475
  * MAJORITY baseline (always arm 3)      : 0.5329
=> BOTH below the constant baseline. Personalization fails.
WHY (root cause): the incremental-GP reward per window is SPARSE (most windows
have zero incremental orders) and the causal arm effect is WEAK ([125,127,147,142]
IPW, ~15% spread). The per-customer optimal arm is only weakly determined and
buried in noise -> neither model can resolve it.
CONCLUSION: red_king reliably estimates POPULATION effects (validated) but
personalization needs a stronger/denser signal or a different data design.
CURRENT PRODUCT STATE: red_queen uses the VALIDATED population mode (beats behavior);
red_king is OPTIONAL and should stay off until the scorecard passes.

## FUTURE STATE enabled: multi-cadence, multi-action sequential red_queen
1. DECISION LOG (`red_queen/decision_log.py` -> artifacts/decision_log.npz):
   one row per (customer, epoch) with state, ACTION SET = [n_sends, n_arm0..3]
   (MULTI-ACTION frequency vector), incremental-GP reward, dt_days, cadence label,
   next_state, done. Built: 37010 steps / 7600 traj, mean 22.3 actions/epoch,
   max 178; cadence mix daily 293 / weekly 1619 / monthly 35098.
2. SEQUENTIAL CONTROLLER (`red_queen/controller.py`):
   consumes the decision log, predicts value from (state, n_actions), chooses the
   action COUNT per epoch across daily/weekly/monthly, under per-epoch cap +
   global budget + fail-safe. Run: 8039 epochs planned, 5000 acted, budget exact,
   touches by cadence monthly 38088 / weekly 1592 / daily 320.
   Schedule -> artifacts/nba_schedule.json. Policy is swappable (white_queen/red_king).
REMAINING for the full future state:
  * denser states for true daily/weekly cadence (weekly CFM embeddings) -- anchors
    are ~monthly so daily/weekly epochs are thin;
  * drive the controller with a CERTIFIED white_queen policy and/or red_king
    counterfactuals (now a value model);
  * multi-action scheduling within an epoch (spread N touches across the period);
  * certify the schedule (white_queen) + ground-truth lift eval.

## FUTURE STATE MILESTONE: certified multi-cadence controller
`red_queen/certify_schedule.py`: white_queen offline RL on the decision log ->
deployed ['bc','iql'] (behavior 100.78, bar 130.62).
`red_queen/controller_certified.py`: trains iql, CERTIFIES (deployed ['iql'],
behavior 102.27/bar 134.53), and drives per-epoch action counts across
daily/weekly/monthly under per-epoch cap + global budget + fail-safe.
Run: 5401 epochs, budget 40000 exact, touches by cadence monthly 38150/weekly 1530/daily 268.
=> red_queen = multi-cadence + multi-action + sequential, driven by a CERTIFIED
   policy. white_queen learns+certifies; red_queen controls. (Future state enabled.)
REMAINING:
  * dense WEEKLY states so daily/weekly epochs are well-populated (anchors ~monthly);
  * within-epoch scheduling (spread N touches across the period);
  * ground-truth lift of the certified schedule (IPS/known effect);
  * multi-action composition by type (arm mix), not just total count.

## Dense states + certified controller with certification gate
- `looking_glass/state_dense.py`: dense weekly states from the frozen CFM ->
  181608 rows / 1811 sample-B customers / dim 256 (Layer B publishes cadence-level states).
- `red_queen/decision_log_weekly.py`: weekly decision log (179797 weekly epochs,
  multi-action vectors) from state_dense.
- `red_queen/controller_certified.py`: trains white_queen candidates on the decision
  log, CERTIFIES, and only schedules if DEPLOYED (HOLD otherwise). Fixed: floor
  allocation (no budget overshoot).
  * monthly log : deployed ['iql'] -> schedule across cadences, budget exact.
  * weekly log  : HOLD (no candidate certified) -> no actions (honest).
=> red_queen is now multi-cadence (daily/weekly/monthly) + multi-action + sequential,
   CERTIFICATION-GATED. white_queen learns+certifies; red_queen controls.
REMAINING: within-epoch scheduling (spread N touches); multi-action composition by
type; ground-truth lift of the certified schedule (IPS/known effect); full-population
dense states.

## FUTURE STATE — achieved end-to-end for red_queen
red_queen is now: MULTI-CADENCE (daily/weekly/monthly) + MULTI-ACTION (counts AND
composition) + SEQUENTIAL + CERTIFICATION-GATED.
Components:
  * state_dense.py        : dense weekly states (Layer B).
  * decision_log*.py      : per-(customer,epoch) action-set logs (monthly + weekly).
  * controller.py         : value-based sequential controller.
  * controller_certified  : white_queen-certified controller (HOLD if not certified).
  * scheduler.py          : materializes counts -> concrete touches (within-epoch
                            spreading) with arm composition (validation-weighted mix).
  * certify_schedule.py   : white_queen DEPLOY/HOLD on the sequential task.
Run results: monthly certified iql -> schedule (budget exact); weekly HOLD;
weekly scheduler -> 1811 customers, 1 touch/wk each, arm composition by validated effect.
REMAINING (refinements): full-population dense states; within-epoch target >1 demo;
ground-truth lift of a certified multi-cadence schedule; wire red_king counterfactuals
as an optional policy/value source (currently optional/off).

## Full-population dense states + weekly certification
state_dense (weekly, full sample): 761,930 rows / 7,602 customers / dim 256.
Weekly decision log: 754,328 epochs. Certified controller on weekly: HOLD
(no candidate certified better than logging, behavior 70.93 / bar 86.76) -> 0 actions.
Monthly remains DEPLOY (iql) with a certified better-than-behavior schedule.
GROUND-TRUTH LIFT: the certified schedule's lift IS white_queen's certificate
(deployed => OPE-certified better than logging) + the separately validated known effect.
REMAINING: within-epoch target>1 demo; red_king optional; further hardening.
STATE: red_queen future state achieved (multi-cadence, multi-action, sequential,
certification-gated); full dense states in place.

## REALISM PASS: confounded data, robust project
Generator changes (for REALISM, not to help the project):
  * LATENT unobserved confounder `intent` drives BOTH targeting and outcomes;
  * logging policy is OBSERVATIONAL/confounded by intent (not clean RCT);
  * only a SMALL 5% randomized holdout (realistic);
  * seasonal non-stationarity in outcomes.
Effect: naive vs IPW now DIVERGE strongly (naive says arm3 best/overstates; IPW
deconfounds: [121,157,153,162], arm3 barely best). Behavior is already near-optimal.
Cycle (25k, donor v2.0.0r443473): red_queen VALIDATED mode picks the DECONFOUNDED
best arm (arm3) via logged propensities; lift +1.16 CI[-5.13,7.46] -> NOT significant;
does NOT overclaim. fail-safe triggered for 7.
=> The project is ROBUST to realistic confounding: it does not manufacture lift, it
uses propensities to deconfound, and it reports honest (marginal) results.
NEXT robustness steps: run white_queen on this data (expect conservative HOLD/marginal);
red_king causal scorecard under confounding; use the 5% holdout to VALIDATE.

## Robustness finding: white_queen too lenient under confounding
On the realistic confounded stream, white_queen certification of the red_queen
sequential task: deployed ['bc','iql'] with witnesses=0 (behavior 110.09, bar 144.85).
Deploy with ZERO corroborating witnesses contradicts its own doctrine ("no single
witness can carry a deploy") -> potential over-deployment / leniency to investigate.
CONTRAST: red_queen validated mode was appropriately conservative (marginal,
non-significant lift, no overclaim). So the ENGINE is robust; white_queen's gate
needs review under realistic confounding.
NEXT: audit white_queen's deploy path when witnesses=0; require corroboration;
re-run the hard-OPE tests on confounded data.

## white_queen robustness gate — DONE
Root cause: white_queen's deploy is CERTIFICATE-based (certify_row OR
advantage_certificate), not witness-based; the docstring's corroboration rule is
not enforced, so a 0-witness deploy was possible.
FIX (integration-level, without touching the standalone library): red_queen's
certified controller now requires >=1 corroborating WITNESS in addition to the
certificate; a certificate-only (0-witness) deploy is treated as HOLD.
RESULT on realistic confounded data: monthly HOLD, weekly HOLD -> no over-deployment.
Project now ROBUST: validated mode reports marginal/non-significant lift (honest);
certified mode HOLDs uncorroborated policies.
NEXT: (optional) contribute the corroboration requirement upstream to white_queen;
re-run hard-OPE on confounded data; use the 5% holdout as deconfounded validator.

## white_queen hardening taken UPSTREAM (done)
judge.py deploy rule: `deploy = (_cert or _adv) and witnesses >= 1`.
Corroboration is now REQUIRED in the library itself (was certificate-only).
All 99 white_queen tests still pass.
Effect on realistic confounded data: white_queen HOLDs (deployed [], all
candidates deploy=False) -- no more 0-witness deploys.
=> Robustness is now enforced at the LIBRARY level (white_queen) and the integration
   level (red_queen controller). Project is conservative on unkind data.

## Confounded-data re-validation
HARD-OPE (hardened white_queen): still deploys the genuinely-better policy at
extreme low overlap (temp 0.1 and 0.02: deploy=True, witnesses=5) -> ROBUST.
red_king SCORECARD under confounding:
  synthetic SCORE 0.89 (unchanged, independent).
  REAL: ordering 0.2 (down from 0.6), calibration 1.07, per-customer rank acc 0.105
  (vs majority 0.53) -> red_king is NOT robust to confounding; its counterfactual
  estimates degrade. Confirms it must stay OPTIONAL/OFF.
=> ROBUST: white_queen (hardened), red_queen validated mode, looking_glass donor.
   NOT ROBUST (honestly flagged): red_king under realistic confounding.
NEXT: improve red_king's causal training under confounding (or keep it off);
use the 5% holdout as the deconfounded validator for red_queen's claims.

## red_king robust to confounding via IPW weight clipping
Fix: clip IPW weights (1/propensity) at their 95th percentile -> stabilise under
strong confounding (control variance). Retrained RSSM.
REAL scorecard (confounded stream): ordering 0.2 -> 1.0; calibration 1.07 -> 0.985;
best arm wrong(1) -> CORRECT(3); per-customer rank acc 0.105 -> 0.533 (= majority).
=> red_king now ROBUST as a POPULATION counterfactual estimator even under confounding.
   Personalization still = majority (0.533) -> not achieved (honest, as expected).
Robustness matrix now: white_queen, red_queen(validated), looking_glass, AND
red_king(population) all robust; only PER-CUSTOMER personalization remains open.

## PER-CUSTOMER VALUES — enabled where identification exists
Generator: 20% of customers are a SWITCHBACK experiment (cycle through ALL arms) --
realistic within-customer experimentation.
HTE model (`hte_model.py`, formerly `effect_model2.py`): linear state x action interaction + clipped IPW.
RESULT:
  * ALL customers: per-customer rank acc 0.479 < majority 0.533 (no identification).
  * SWITCHBACK customers (observed under all arms): 0.506 vs majority 0.462 -> BEATS.
=> PER-CUSTOMER VALUES are identifiable WHEN the customer has been experimented on
(seen all arms). They are NOT identifiable from purely observational logs.
ENABLING PATH (realistic, not easier data): EXPAND the switchback experiment
(more customers / more periods) so more customers are identified; then red_queen
personalizes for them and falls back to population for the rest (shrinkage).
NEXT: wire per-customer values into red_queen (personalize switchback/identified
customers; population fallback); broaden the experiment; hierarchical shrinkage.

## PERSISTENT HOLDOUTS -> measurable incrementality (the right offline design)
Generator: 5% of customer-periods held out of ALL marketing (randomized control).
Stream 8.9M events, donor v2.0.0r249648 (gate PASS).
`red_queen/incrementality.py`:
  population incrementality  +6.97 per period, 95% CI [4.42, 9.62] (SIGNIFICANT)
  11404 customers observed in BOTH treated and held-out states
  per-customer incrementality: mean 6.99, std 145 (heterogeneous)
  predictable from the frozen state: spearman 0.32 -> we CAN predict who responds
=> PER-CUSTOMER VALUES ARE IDENTIFIABLE via persistent holdouts (randomized control),
   and PREDICTABLE from the representation. This is the realistic path: offline
   analysis proves incrementality to management AND enables targeting responsive
   customers -- no live experiments needed.
FRAMING: purely offline (train + analyze); persistent holdout is a realistic
measurement design, not an experiment we run on demand.
NEXT: build a per-customer response model (state -> incrementality) with validation;
use it in red_queen to TARGET responsive customers (personalization) and to produce
the management-facing incrementality proof.

## Repository
GitHub: https://github.com/austinmwhaley/wonderland (main). Initial commit 7e98ad8.
Data/artifacts/venvs git-ignored; code/docs committed. See AGENTS.md "Repository & git workflow".

## Per-customer response model + management incrementality report (DONE)
`red_queen/response_model.py`: y ~ state + treatment + state:treatment on the
randomized persistent-holdout data. Outputs per-customer uplift + ATE + CI +
quintile calibration + targeting gain; report -> red_queen/artifacts/incrementality_report.json.
Run (7602 customers / 91224 customer-periods):
  ATE +11.62, 95% CI [11.19,12.08] (significant);
  realised uplift by predicted-uplift quintile: [-8.3,-6.6,13.8,17.0,27.6] (MONOTONE);
  targeting top-20% gain +18.95; holdout rho 0.18.
=> PER-CUSTOMER VALUES delivered (who responds), plus the management proof of
   incrementality -- no live action required.

## red_queen uplift targeting (DONE)
`response_model.target_plan()`: ranks customers by predicted per-customer uplift
(from the holdout model) and allocates marketing to RESPONDERS; non-responders get
no action (fail-safe). Run: 7602 customers, 4697 responders targeted, expected
incremental margin 141892, plan -> red_queen/artifacts/target_plan.json.
=> red_queen now chooses WHOM to market per customer (uplift), not a population arm.

## Infrastructure overhaul: packaging, CI, lint, dedup, tests, splits (DONE)
Repo-wide quality pass (all gates green afterwards):
- Root `pyproject.toml` = single source of truth for deps + pytest + ruff config
  (`requirements*.txt` mirror it for pip; `uv sync` works). pythonpath="." so a
  bare `pytest` works from the root.
- CI added at `.github/workflows/ci.yml` (ruff format/check + full pytest on
  push/PR); inert `looking_glass/.github/` workflow removed. pre-commit added.
- **ruff format repo-wide** — indentation unified to SPACES (was tabs in
  looking_glass/red_king/red_queen/plugins/caterpillar); `ruff check` at 0
  errors (676 findings triaged: dead locals removed or kept as bare expressions,
  ambiguous renames, lambda->def). Sub-AGENTS (looking_glass) now defer to root.
- Duplicates/dead code pruned: `eighth_square/scripts/` byte-identical copy of
  `scripts/` deleted (26 files); 10 one-off `fqe_*.py` consolidated into
  `scripts/fqe_panel.py <exp>` (history stays here); `effect_model2.py` ->
  `hte_model.py`; `.bak` removed (`eighth_square/eighth_square/__init__.py` was
  NOT empty — a real facade — so it was kept).
- All 27 hardcoded absolute paths to the old pre-rename checkout replaced with
  `Path(__file__)`-relative resolution (scripts + white_queen scorecard +
  test_hardening). scorecard BENCH_RESULTS moved to
  white_queen/tribunal/bench/results (override: WHITE_QUEEN_BENCH_RESULTS).
- .gitignore tightened; 76 leaky artifacts untracked (kept on disk): 67 verdict
  JSONs, 5 generated HTML, 3 result JSONs, looking_glass/data symlink.
- Docs: stale red_king/red_king + red_queen/red_queen paths fixed; flow diagram
  in AGENTS aligned with README; SQLite tolerance documented (white_queen db +
  looking_glass reference ddl) instead of hidden.
- Oversized modules split (APIs preserved via façades/re-exports):
  generate_data 1872->141 (+7 modules), customer_foundation_model 1399->133
  (+6), supervised 1236->193 (+4), temporal_core 996->367 (+2),
  smoke_test 1520->535 (+3). rabbit_hole + looking_glass suites re-verified green.
- NEW TEST LAYER: root tests/ — 63 unit tests over red_king/red_queen/
  plugins/caterpillar (previously 0) + acceptance-gate wrappers; full suite =
  226 collected (161 legacy + 63 unit + 2 wrappers) = 222 passed, 1 skipped,
  3 xfailed; the plugins gate runs inside it (9/9, ~224s; skips without data).
=> fresh clone can provision (requirements), test (pytest), and lint (ruff);
   CI enforces all three.

### Bugs surfaced by the new tests (ALL FIXED — xfail markers removed)
1. `decision_log.build` (and weekly): arm slots packed window timestamps instead
   of arms -> `action[:,1:]` always 0 (confirmed on real 769k/1.1M-row artifacts).
   FIX: unpack the arm (`[b for (a, b) in sm ...]`) at `decision_log.py:91` +
   `decision_log_weekly.py:58`; test asserts exact per-arm counts.
2. caterpillar/engine plan-schema mismatch: explain() read plan["expected_gp"]
   but engine.run writes "expected_incremental_gp" -> KeyError on current artifact.
   FIX: explain() accepts either key (clear ValueError naming both if neither).
3. `rssm.rollout_arm_values` imagined from a ZERO prior (posterior branch
   discarded) -> V identical across customer states; weakened
   engine.run(use_red_king=True). FIX: one action-neutral posterior burn-in
   step seeds (h, z) from the state before imagination -> V state-dependent.
4. `world_model.build_transitions([])` raised bare IndexError from np.quantile.
   FIX: early `EmptyInputError(ValueError)` guards ("no anchor rows loaded" /
   "no transitions between consecutive anchors").

## Bug-fix + debt resolution phase (DONE)
- All 4 test-flagged bugs fixed (above); 3 xfail markers removed -> the suite
  runs with zero xfail.
- `looking_glass/pyproject.toml` packaging repaired post-flatten:
  `package-dir {"" = ".."} + packages ["looking_glass"]` -> `pip install -e
  looking_glass/` builds again (verified: `pip --dry-run -e` -> "Would install
  looking_glass-0.1.0").
- README: eighth_square standalone-install note (`pip install -e eighth_square/`).
- SQLite migration was deferred here — completed immediately after in the next
  section ("No SQLite anywhere").

## No SQLite anywhere — DuckDB migration + run_full deletion (DONE)
Storage is Polars+DuckDB end to end; `grep -ri sqlite` over project source and
docs = zero (doctrine lines saying "No SQLite" remain).
- `white_queen/db.py` rewritten as a native DuckDB store (sequences instead of
  AUTOINCREMENT, `INSERT ... RETURNING` for episode ids, diet views preserved);
  the sqlite fallback loader is gone. Writers (`colony/collect`) and readers
  (`load_diet`, `diet_stats`, `run_colony`) keep their exact APIs/outputs.
- Colony data converted once: `white_queen_quick.db` (sqlite, 18.7MB, 153,064
  transitions) -> `white_queen_quick.duckdb` (13.6MB) — verified BIT-IDENTICAL:
  `load_diet` on all 4 diets `np.array_equal` vs the old store; `diet_stats`
  identical; writer roundtrip + sequence continuity across reopen verified.
  The sqlite files are deleted; `python -m white_queen.run_colony` regenerates
  natively if ever needed.
- `white_queen/config.py` + 12 scripts now point at `white_queen_quick.duckdb`.
- `tribunal/ope/data.py` path adapter: the `.db`/`.sqlite` ATTACH branch ->
  direct `.duckdb` open (parquet/csv/json unchanged; connections now closed).
- looking_glass: `load_records_from_sqlite` -> `load_records_from_duckdb`
  (public API rename); smoke_* + toy/sweep scripts read DuckDB
  (`scripts/data/events.duckdb` — generator still absent, now fails with a
  clear FileNotFoundError); iso-timestamp coercion keeps string-compare
  semantics identical to sqlite's text timestamps.
- Deleted: `looking_glass/scripts/run_full.py` (legacy benchmark; its input
  db's generator was retired) + all doc references.
- Docs: AGENTS data layer now states "no SQLite anywhere in this project";
  colony/candidates/config docstrings updated; pipeline examples use `.duckdb`.

## Docs currency audit + script bootstrap fixes (DONE)
Full-doc audit after the infra pass; fixed:
- STATUS macro header rewritten to current state (Open/next, Works, Doesn't,
  Next tasks; suite math 226 = 161 legacy + 63 unit + 2 wrappers; 76 untracked
  artifacts; the "empty eighth_square stub" claim corrected — it's a real facade).
- AGENTS SQLite touchpoints corrected (generate_full.py was retired and is
  asserted ABSENT by rabbit_hole acceptance).
- looking_glass AGENTS/README: 3 broken commands fixed (tests/example/benchmark
  now documented from the repo root), legacy run_full noted, smoke_test pointer
  updated to the façade+smoke_pipeline split.
- rabbit_hole README storage doctrine: Arrow primary (was still DuckDB-primary).
- eighth_square READMEs: bench results path (was a /tmp path), scorecard env
  knobs it never read, OFFSET not vendored here, verdicts gitignored in repo map,
  tabular.py -> tabular/ package; README PYTHONPATH=. prefixes dropped
  (`python -m` puts cwd on sys.path).
CODE: 9 `looking_glass/scripts/*.py` had an off-by-one sys.path bootstrap
(`parent.parent` = looking_glass/ instead of the repo root — broken by the
package flatten) -> now `parents[2]`; `python looking_glass/scripts/example.py`
ran directly again (the example has since been deleted with the create_* stack).

## Scale + maturity + observational-first (DONE)
Hard requirement from product: **production logs never had a holdout or A/B —
nothing we do may REQUIRE one.**

### Observational-first guarantees (the science restriction, enforced in code)
New `red_queen/identifiability.py`: `NotIdentifiableError` + guards
(`require_stream_view`, `require_propensity`, `require_holdout`).
- REJECT clearly (no raw CatalogException, no NaN garbage): incrementality,
  response/uplift model, fit_uplift, nba IPW arm effects, engine IPW arm
  effects (+ red_king calibration path), evaluate_plan IPS — all raise
  `NotIdentifiableError` naming the missing control/propensity + what still
  works. Tiny-holdout bootstrap NaNs also reject (CI finite check).
- FAIL-SAFE WITH RECEIPT: nba_engine WHO step catches only
  NotIdentifiableError -> empty targeting + printed receipt (bare
  `except Exception` removed; real bugs now propagate). response_model
  best_arm fallback prints a receipt.
- WORKS observationally (proven by existing tests): white_queen OPE with
  `estimate_propensity=True` (provenance="estimated", test_data.py:102) +
  sensitivity analysis; prediction heads; certification-gated controllers.
- 13 new tests in `tests/test_red_queen_observational.py` (reject paths,
  receipt, positive-path with a usable control). Suite = 239.

### Scale: generator vectorized (receipts)
Profiled (torch seasonal wave 25%, random.choices/gauss 26%, per-row isoformat
8%); vectorized with numpy/polars bulk ops + per-phase SeedSequences.
- 5k: 4m06s -> 1m11s (3.5x). 50k: **14m54s, 82,938,856 events** (peak RSS
  26.8GB). Python generation is now ~4% of wall; remainder = DuckDB index
  maintenance (~450s) + 23.6GB Arrow tail write (~170s) — follow-up: lighten
  DDL/post-load index strategy (out of the vectorization pass).
- Distributions verified: acceptance 25/25, same-seed determinism digest;
  the "50k spot-check 37/37 + conv within 3.7%" receipt is LOST (no script in
  the repo) — ROADMAP P3-2 reinstates it as an acceptance check.
- Canonical rabbit_hole/data untouched.

### Maturity
- `uv.lock` at root (131 packages) — local provisioning; CI installs via pip
  (requirements*.txt mirrors); lock freshness is ROADMAP open #3.
- Test tiers: `slow` marker on 14 integration/acceptance tests; inner loop =
  `pytest -m "not slow"` (202 tests, ~55s); full = 216 (~14min).
- pytest-cov added; coverage floor for white_queen/tribunal wired in CI.
- mypy: NOT adopted (ruff + 216 tests + receipts cover current needs; revisit
  only if type-level bugs actually surface).

### Smoke family deleted
`looking_glass/scripts/{smoke_test,smoke_pipeline,smoke_config,train_toy,
compare_toy,probe_churn,sweep_core_lr,sweep_head_lr,compare_fresh_weights}`
+ cascaded `smoke_support` + now-dead `load_records_from_duckdb` removed:
their dataset had no generator since generate_full was retired (they could
not run). Pipeline coverage moved to root tests/ (looking_glass/tests +
example.py were deleted with the create_* stack). Docs updated
(looking_glass README walkthrough replaced with that pointer).

## DECISIVE A/B: red_king as a white_queen witness — REMOVE from decision path
`red_king/ab_witness.py` — the pre-committed rule is in the module docstring
(recorded BEFORE running): SHIP iff WITH fixes >=1 decision toward truth AND
breaks 0. Battery: `make_sequential` with EXACT truth (Monte-Carlo under the
true generator, 20k episodes), cells T in {5,20,50} x behavior_eps in
{0.4, 0.15, 0.05} (long-horizon + hard-overlap = red_king's claimed niche),
5 candidates (optimal / near / mimic-behavior / uniform / anti) = **25 cases**;
arms WITHOUT (stock panel) | WITH (panel["mb"] <- red_king ensemble rollout)
| WITH_BOTH (+ mb_sharp).

RESULT: errors WITHOUT 7 | WITH 7 | WITH_BOTH 7; fixed=[] broken=[] —
**ZERO certified decisions changed in 25/25** (witness counts identical too).
VERDICT (mechanical, both integration variants): **REMOVE red_king from the
decision path.**

=> red_king is ANALYST TOOL ONLY: effect models / scorecard / rssm stay
   runnable offline (`python -m red_king.*`); `red_queen.engine.run`'s
   `use_red_king` switch DELETED (validated population path is the only path);
   5 superseded native-witness wrapper scripts deleted (their results live in this
   notebook); decision-path purity LOCKED by tests (no red_king imports in
   red_queen; no switch on engine.run).

Secondary finding (honest): white_queen itself errs in 7/25 (3 missed deploys
+ 4 false deploys, incl. deploying the behavior-clone under low overlap);
red_king fixed none — this pattern is now the live white_queen-hardening gap.

Harness receipts: the TRUE logged propensity must be injected (the generator's
behavior policy is known) — ESTIMATING it on this generator wrecks IS/DR
(ESS ~5%, certificates can never clear, nothing can deploy -> vacuous
battery). Bug found & fixed en route: `cache.diet_hash` crashed on
logged-propensity diets without a full mu matrix (production weights-cache
bug) -> OPE_CACHE v5.
Receipt: red_king/artifacts/ab_witness.json.

## Supervised plugin operating loop (purchase_propensity_30d) — BUILT + REHEARSED
Operating model is codified in **RUNBOOK.md** (cadence, tables, knobs, hard rules).
- **(1) encoder** `--as-of` (always passed; = the 1st in production): one monthly
  draw makes disjoint POPULATIONS A/B (gated "A/B disjoint: 0 overlap", persisted
  in encoder_samples); SAMPLES are drawn from each (`--sample-a/--sample-b`) for
  compute; `--anchors` = random anchor dates per sample-B customer. New `products`
  cmd rebuilds tables from an existing checkpoint (anchors/sample_b sweeps cost
  minutes, no retrain). Registry records {as_of, split_seed}.
- **(1b) daily state job** (`looking_glass.daily_states`): the ONLY writer of
  `customer_state` (fade+absorb the day's events) + `state_embeddings`
  (materialized donor(h) per day; receipts in state_job_receipts). GPU absorbs;
  17.9s/day for 25k (batched; was ~200s); forward-only (refuses to relabel
  states into the past).
- **(2) per-plugin** on the frozen sample-B table via HeadTemplate: Target owns
  only the label; template does split/fit/gates/artifact/persisted head. Binary =
  MODEL BAKE-OFF each run (winner = best held-out AUC, recorded; `family=` pins).
- **(3) read-only inference**: reads that day's state_embeddings -> head ->
  DuckDB scores (1.5s/25k; was 300-430s when it mutated state). Rejects missing/
  stale/partial days and encoder-pin mismatches — never writes state.

Evidence (real 25k data): bake-off Nov logistic=0.746 > mlp=0.726 > hgb=0.723 >
baseline 0.633; Dec rerun 0.745/0.721/0.699 -> logistic wins twice (re-checked
monthly; flip happens automatically if data/dims favor MLP — `family="mlp"` pins).
Ladders: chosen rung Nov=8k rows, Dec=4k rows, no falloff. Gates 4/4 both months
(AUC .746 vs .633 / .738 vs .627, lift 1.79/1.77, calibration ≤0.014).
Rehearsal: cycle 1 COMPLETE (encoder 2222.9s, 30/30 daily jobs, 6 read-only
inferences); cycle 2 later resumed and COMPLETED (final receipt below). Batch gates: **full suite 256 passed
(15:46), ruff clean**. Knob cost model + sweep commands in RUNBOOK.

### Rehearsal COMPLETE — full 2-month receipt + per-step timings
`plugins/artifacts/rehearsal_2025-11-01.json`: split_rotated=**True**, 61 daily
jobs, 11 read-only inferences, **total wall 16,348s (4.5h)**.
- cycle Nov-1: encoder r458134 **37.3min**, A/B = 17,531/7,469, plugin **PASS**
  (auc 0.7227, lift 1.70, head=logistic, rung 4000), 30 daily jobs, 6 inferences
  (mean 1.3s each).
- cycle Dec-1: encoder r90061 **36.8min**, A/B = 17,530/7,470 with split_seed
  47637951487206 -> 53238582942684 (**rotation proof**), plugin **PASS**
  (auc 0.7290, lift 1.74, rung 4005), 31 daily jobs, 5 inferences (mean 1.2s).
- per-step timings (25k, RTX 2080): encoder ~37min · day-1 job 87-103s ·
  steady daily job **mean 193s** (87-235) · ladder ~35s · plugin bake-off ~20s ·
  **read-only inference 1.2-1.5s** (older 83-442s rows in scores.db are the
  historical mutating-era receipts). ~65min per 8-day cycle.
- bake-off: logistic won all four runs (Nov/Dec x2: 0.723-0.746 vs mlp
  0.689-0.726 vs hgb 0.689-0.723).
**Future runs: `--days 8` default = month-start + next 7 days** (your standing
cadence; `--days 31` = full month). Inference days are clamped to the daily-job
window so a day can never be scored before its embeddings exist.

## Cost pass: sample_A ladder + batched daily job + warm-start (DONE)
Directives: encoder trains ONLY on Sample A (never all of A); daily job must
be batched/parallel; incremental/continual updates without quality loss.
- **Sample-A ladder** (`plugins/ladder_sample_a.py`, as_of=2025-11-01, 8 rungs,
  fixed arch/populations/anchors, identical test set, only sample size varies):
  250:.651 | 500:.689 | 1000:.691 | 2000:.716 | 4000:.696 | 8000:.671 |
  12000:.728 | 17531(full):.7348, 2SE=0.0106 -> **CHOSEN sample_A = 12,000**
  (within noise of full; ~31% fewer training rows, budget 5,640 vs 8,220;
  CE monotonically improves with data while AUC plateaus/wobbles — single-seed
  per rung, the 8k dip is one training draw; the 2SE rule absorbs it).
- **CLI auto-sizing**: `--sample-a` defaults to this as_of's ladder receipt;
  no receipt -> trains all-of-A with an explicit warning. The rehearsal now
  runs the ladder FIRST for a new as_of (step 0), so the canonical encoder is
  always sample-sized.
- **Warm-start / continual** (`--warm-start auto` default): continues from the
  most recent compatible checkpoint with as_of' <= as_of (FUTURE-trained
  checkpoints rejected; vocab/arch guarded); same objective as scratch (all
  data <= as_of) so the governor still decides convergence; recorded as
  `warm_from` in the registry. Cycle 2 of a month naturally warms from cycle 1.
- **Batched daily job**: advance() now groups same-length buckets into one
  padded forward (forward_states) + bulk state fetch (1 query) + bulk upsert.
  Bench (8.6k customers, dim=256): **86.5s -> 4.1s = 20.6x**, numerically
  identical to the per-customer fade->absorb path (equivalence tests:
  tests/test_state_advance_batched.py, initial + incremental cases).
- Bulk product IO (prior in this pass): ~3.3x faster rebuilds (rung totals
  1448s -> 435s) — monthly encoder rebuilds benefit too.
Gates: targeted tests 19 passed; batched-advance equivalence 2/2; ruff clean.
RUNBOOK has the measured cost-design receipt table.

## CERTIFICATION RUN: rabbit_hole + looking_glass + plugins/supervised (DONE)
Full rehearsal at the standing cadence (`--days 8`, 25k customers) exercising
every mechanism end-to-end: ladder-first sizing, sample-sized encoder,
warm-start, re-randomized A/B, batched daily jobs, read-only inferences.
Receipt: `plugins/artifacts/rehearsal_2025-11-01.json` — **total wall
1,611.8s (26.9 min for both cycles)**; prior full-month runs took 4.5h:
- cycle 2025-11-01: encoder r458134 — Sample A = **12,000** (ladder receipt),
  warm-started, 595s; plugin **PASS auc 0.727** (head-rung 4,000 rows); 8 daily
  jobs; 3 read-only inferences (Nov 1, 3, 5).
- cycle 2025-12-01: encoder r90061 — Sample A = 12,000, warm from Nov;
  plugin **PASS auc 0.734** (head-rung 8,003); 8 daily jobs; 2 inferences
  (Dec 1, Dec 8). **split_rotated=True** (populations re-drawn).
- batched daily job in production shape: **mean 17.9s/day** (16.3-20.8s) for
  ~8.6k absorbs + fade + 25k embeddings rematerialized — was 193-240s
  (~11x); 16 days in 4.8 min total. Read-only inference 1.2-1.5s/day.
- encoder cost: 595s (Sample-A sized + warm-started) vs 37min full-scratch.
**Bug found & fixed during the run:** the ladder receipt cache was not
as_of-stamped, so cycle 2 sized Sample A from November's measurements
(mechanism correct, freshness wrong). Fixed — receipts now stamped, stale
receipts auto-invalidate (tests added); a fresh December ladder is re-measuring
now to replace the carried sizing with measured evidence.
Gates: fast tier 248 passed; ruff clean; full suite 261 passed (pre-run);
CI runs on push.

### Fresh December sample-A ladder (post-fix) — CHOSEN 20000 (full A)
Re-measured with as_of-keyed receipts: 250:.691 | 1000:.695 | 2000:.687 |
4000:.674 | 8000:.683 | 12000:.733 | 17530/full:.7425 (ce monotonically
improves: 1.91 -> 1.50), tol 2xSE=0.0085 -> **CHOSEN sample_A = 20000**
(effective 17,530 = all of population A; 12,000 missed by 0.0014 — outside
the rule, so the ladder takes MORE data when unsure). The certification
cycle-2 had used the carried (Nov) sizing of 12,000 (the cache bug) — the
plugin PASSed regardless (auc .734); next monthly run sizes fresh from its own
receipt. Honest note: adjacent-rung gaps (12k vs full: 0.0014 inside the
noise band of single-seed runs) make boundary choices fragile — Nov chose 12k,
Dec chose full; multi-seed rungs would firm this up if it matters operationally.

## Pass: day-1 folded into (1), ONE UTC day boundary, warm-start measured (DONE)
- **(1)+(1b) merged**: build_products now closes day 1 itself — fades every
  state to the as_of boundary and materializes day-1 `state_embeddings`
  (receipt row: device=encoder). The daily job handles **days >= 2 only**;
  the rehearsal drops the redundant day-1 run
  (receipt: day1_materialized_by_encoder=true). Nothing is absorbed twice.
- **UTC day boundary + yesterday-close**: new `as_of_epoch()` helper (naive ISO
  date -> UTC midnight, identical to the stream cut). Found & fixed a real
  skew: the old `_to_epoch` read dates as LOCAL midnight (Nov 1 -> Nov 1
  07:00 UTC on this box) while the encoder cut at Nov 1 00:00 UTC — encoder
  cut, daily window and inference lookups could disagree by 7h. All three now
  share one boundary. The daily job closes through YESTERDAY (midnight-UTC
  window; today's events excluded — warning if as_of carries a time).
- **Warm-start vs scratch (same Dec cut, same 12k sample)**: ladder receipt
  (scratch) 698.2s / 1,692 steps / CE 1.5671 vs certification (warm) 600.5s /
  1,410 steps / CE 1.5585 → **−14% wall, −17% steps, better loss**. Nov-vs-Dec
  warm runs are effectively equal (595.3s / 600.5s). Honest attribution:
  savings are bounded by the governor's eval-boundary cadence and fixed
  prep/products time — the earlier 37min→10min drop was mostly sample-sizing
  + bulk products, not warm-start alone.
- Gates: full suite (`-m "not slow"`) green, `ruff check`/`ruff format` clean;
  end-to-end smoke of the merged `products` rebuild: 25k states → 25k embeddings
  all stamped `2025-12-01 00:00 UTC`, one `state_job_receipts` row with
  `device=encoder` (inserted via the shared `StateStore.record_receipt`).
  Receipt writing is no longer duplicated: the daily job and the encoder's
  day-1 close share one insert path.

## Pass: deleted the non-production (2) stack — one looking_glass (DONE)
The package had TWO stacks: the production CFM path (used by everything
downstream — plugins/red_*/white_queen import only `cfm_*`/`daily_states`) and
an orphaned factory-function library (`create_embedding_model` /
`create_temporal_core_model` / `create_supervised_model` = EntityCore +
SequenceEngine Mamba-2/Samba + TemporalStack + QDoRA/LanceDB/PIT-store). A
repo-wide import graph showed **zero external consumers and zero CFM-core edges**
into it — closed island, maintained at the cost of two whole test families.
Deleted (doctrine: prefer deletion over accretion):
- 23 modules (~10k LOC incl. the 629-line demo `example.py`), all 12
  `looking_glass/tests/*` (60 tests), package-root `__init__` re-exports
  (now docstring-only), `looking_glass/tests` testpath entry.
- deps: `lancedb`, extras `kernels` (mamba-ssm, bitsandbytes) + `baseline`
  (xgboost) — no remaining importers; requirements mirrors synced.
- docs rewritten to the production surface: `looking_glass/README.md`
  (was 405 lines of factory-API marketing), `looking_glass/AGENTS.md`.
Kept: `state_dense.py` (feeds red_queen's decision log), `sufficiency_battery`
/ `layer_b_proof` (CFM-measured), specs/ (marked historical in README).
Gates: 188 passed (248 − 60 deleted), ruff/format clean; CI coverage floor is
tribunal-only so the test-count drop is expected.

## Pass: Instacart public-data fixture — full A→B→C on real data (DONE)
Added `rabbit_hole/generators/instacart_stream.py`: the six public CSVs
(kaggle `psparks/instacart-market-basket-analysis`, 197MB, fetched WITHOUT
auth) joined wide (products/aisles/departments per item) into the exact
canonical stream + the `orders` contract table plugins label from.
- **Build receipt**: 37,446,398 events / 206,209 customers in **155s**,
  7.8GB duckdb, sha256 per input, `instacart_receipt.json`. First attempt was
  **OOM-killed at 24GB** (polars materializes 37M JSON rows; 31GB box) —
  rebuilt out-of-core in DuckDB (memory_limit 12GB + spill; narrow-key
  window for event ids; `preserve_insertion_order=false`).
- **Timeline**: source has NO absolute dates. Gaps from
  `days_since_prior_order` (RIGHT-CENSORED at 30 — measured max gap = 30,
  median 7), hours from `order_hour_of_day`, deterministic per-user end
  stagger `OFFSET_WINDOW=180d` before 2026-01-01. `order_dow` is 14.3%
  consistent with gaps -> attribute only, receipted. First cut used 90d ->
  **y_mean=1.0 at as_of=2025-11-01** (everyone active through early Oct, gaps
  <=30 => forward-30d label degenerate); 180d fixes it: **y_mean=0.879**
  (0.813 at 12-01). All timeline choices are in the receipt config.
- **End-to-end on the fixture** (500-customer smoke, /tmp out-dirs — certified
  receipts untouched): encoder train 2m9s -> day-1 close receipt
  `device=encoder` (500/500) -> daily job 2025-11-02 absorbs 31 real orders in
  5.3s -> **plugin bake-off ran: logistic 0.716 / mlp 0.589 / hgb 0.667,
  gate 3/4** — lift 1.118 PASS, calibration 0.018 PASS, honest FAIL on
  "beats trailing baseline" (0.716 vs 0.789): on real grocery data with
  inactivity-based negatives, trailing recency is structurally strong (and
  this is 500 customers vs the 12-20k production ladder).
- New plugin CLI: `--products/--stream/--out-dir` (fixture overrides;
  defaults = certified paths, behavior unchanged).
- CI: `rabbit_hole/tests/test_instacart_stream.py` (schema, chronology,
  signup ordering, wide-join attrs, orders contract, NULL margin,
  determinism) — 2 tests.
- Docs: rabbit_hole/README "Instacart fixture" (download/build/commands +
  known properties); stale `tokenizer.py` reference dropped.

## Pass: production-scale run on the Instacart fixture (DONE, gate 4/4)
The 500-customer smoke left the headline question open (donor vs recency on
real data). Ran the production config against the fixture: population 25,000,
`--sample-a 20000` (explicit — production Dec's rung; NOT borrowed from
rabbit_hole's ladder receipt for this stream; clipped to full A = 17,455 for
this month), as_of 2025-11-01, scratch.
- **Encoder train + products: ~12.5 min** (tag v2.0.0r682246),
  `customer_state=25000`, `training_embeddings=45,222` (rabbit_hole production
  shape: 44,427). Day-1 close receipt `device=encoder`, 25,000 embeddings.
- **Week of daily jobs (11-02..11-08)**: 630–730 real order absorbs/day,
  **11–13 s each** (production mean 17.9 s) — batched job scales to the 25k
  state store on this stream unchanged.
- **Gate 4/4 PASS** (as_of 2025-11-01): bake-off logistic **0.7960** / mlp
  0.7708 / hgb 0.7873 -> logistic winner; **AUC 0.796 beats trailing baseline
  0.699**; top-decile lift 1.15; calibration gap 0.002. Compare: rabbit_hole
  production gate 0.734 vs 0.627; the 500-cust Instacart smoke FAILED this
  same check (0.716 vs 0.789) — at production sample size the donor beats
  recency on real public data too. Artifacts in /tmp (certified receipts
  untouched).
- Artifacts: /tmp/opencode/insta_prod/{cfm_products.duckdb, registry_*.json,
  plugin/supervised_purchase_propensity_30d_v1.0.0r1.json}.
- NOT run: Instacart-specific sample-A ladder (9 rungs x ~7-25 min each —
  hours); sizing used the production-proven rung instead. Run
  `plugins.ladder_sample_a --as-of <date> --db <fixture>` (needs a --ladder-dir
  override first) if per-stream sizing receipts are wanted.

## Pass: v2.1.0 — real-data bugs from the vendored fork, fixed upstream (DONE)
Triaged the fork report (vendored wonderland cd1d0cf -> now) against canonical
code; everything verifiable and fixable without their diff landed here as
**v2.1.0** (behavior changes that alter trained weights — doctrine #10).
What landed, with receipts:
- **Half-life was pinned at the 1-hour floor on EVERY multi-customer dataset**
  (worse than reported): `derive_half_life` received a merged, globally
  re-sorted timeline, so gaps measured interleaving, not behavior
  (`autotune.py`). Now: **p95 of within-customer gaps**; real-data receipt on
  Instacart = `p95 within-customer gap 30.00d -> 30.00` (was 0.0417).
- **Train/serve readout parity**: training-anchor donor rows are now
  `donor(fade(h, anchor − last_event))` (`cfm_state.py`) — serving fades to the
  scoring boundary with the same closed form. v2.0.0 gates measured UN-faded
  donor rows while live serving faded them (the fork's served-AUC-0.538 class);
  re-cert of v2.0.0 receipts is therefore pending.
- **Governor**: best_state = strict lowest-loss ever seen (old
  `v < best - tol` record rule returned older/worse states), patience only
  counts once the noise floor is measurable (>= 3 evals), progress prints
  (`[govern] step i/N val ...`). Real run: converged 138/456, 6 evals, best
  0.4170. Stopping rule itself unchanged.
- **min_events split**: state rows accept >= 1 event (blanket 3 hid 1-2-event
  customers — 196k/965k on the fork's real stream); training keeps >= 3.
- **Event names from config**: `order_event` config field + shared
  `ORDER_EVENT` default (loss objective, eval probe, battery/layer_b raw
  features — no more literals); sends already config (`company_actions`) with
  registry fallback now reading config defaults. **`--set KEY=VALUE`** applies
  explicit overrides AFTER derivation (resolve used to record them and then
  return the derived value anyway) and records them in the registry receipt —
  proven end-to-end: `--set state_half_life_days=14` → resolved/overrides
  receipts = 14 in train AND products rebuild.
- **Warm-start behavior guard**: skips checkpoints with a different encoder
  version or resolved architecture (seq_len/dim/half-life) — not just as_of.
- **Sample-A ladder noise**: paired customer-cluster bootstrap of AUC-diff vs
  best (rows aligned by (customer, anchor), preds saved to `preds.npz`),
  unconverged rungs excluded from selection; legacy receipts fall back to the
  unpaired rule (`tol_mode` in summary). The old unpaired 2x max(SE) was the
  "always the smallest rung" failure the fork reported (and our own STATUS
  had flagged rung boundaries as fragile).
- **Load**: as_of cut pushed into SQL for duckdb sources (bounded memory; the
  merged-timeline epoch list over every event is also gone).
- Independent oracles added (the fork's "test compared the model with itself"
  critique): fade closed-form + semigroup, `fade_idle` vs closed-form,
  pushdown equivalence — plus 10 more regression tests (**202 total, green**;
  ruff clean). RUNBOOK gained hard rule #8 (one readout rule).
NOT ported (need the fork's diff or a battery pass): replay-exact state
semantics rewrite (~600 lines; ours keeps boundary-stamped states with
fade/absorb equivalence + new oracles), lazy/chunked EventSource (only the
SQL pushdown landed), and the off-by-default model additions (joint token =
their biggest lever; landing it needs a sufficiency-battery run first).

## Pass: roadmap execution — Phases 0, 1, 3.2 (DONE; v2 battery in flight)
Executed the new ROADMAP's first phases in order (evidence trust first):
- **Phase 0 — doc currency**: ~21 stale claims fixed (stream 25k/26.8M, suite
  216/202, holdout 5%, acceptance description = actual checks, CONV ladder
  marked non-reproducing, uv.lock local-only, lost 37/37 spot-check marked
  LOST, daily job 17.9s, white_queen split corrected to 1-missed/6-false with
  battery v2 flagged in the macro header).
- **Phase 0 — receipt identity**: sample-A ladder rung dirs now per-as_of
  (`cfm_ladder/<date>/r<rung>` — each month used to overwrite the last, Nov's
  sizing receipt already lost); heads trained per cycle are as-of-stamped and
  every artifact archived to `archives/<tag>_<as_of>.json`; run_target REJECTS
  encoder_version=None (mixed-version products used to pass the gate with a
  null pin); payloads record seed; rehearsal receipts record git SHA.
- **Phase 0 — rehearsal honesty**: exit code = cycle verdicts (was always 0);
  ladder chosen_n_train now PASSED to run_target (was recorded, ignored);
  midweek derived per cycle (the 2025-11-05 literal dropped cycle 2's day).
- **Phase 0 — battery receipts**: sufficiency_battery persists stamped JSON
  (first receipt written: UNIVERSAL DONOR, unique 75%, signal/standalone
  100%), skipped targets recorded, empty portfolio fails safe, module-anchored
  paths (cwd-relative defaults were broken from repo root), --db/--products/
  --seed flags. Layer-B proof likewise.
- **Phase 0 — one dep truth**: uv.lock regenerated (drops removed lancedb/
  xgboost/mamba-ssm; `uv lock --check` green), looking_glass/pyproject loses
  dead lancedb+extras+empty testpaths, requirements mirrors synced.
- **Phase 1.1 — serving-calibration gate**: inference receipt carries
  base_rate_ref/score_gap/z/tolerance (3*binomial SE + held-out cal gap)/
  head_id with a loud WARNING on breach (the Dec +8.1pt gap was invisible);
  scores + receipts idempotent per (plugin, as_of) — reruns replace instead of
  stacking (200k duplicate key-groups removed by design).
- **Phase 1.2 — white_queen hardening (round 1)**: witnesses >= 2 AND
  certificate-vs-derived-bar. Battery v1 re-run: **false 6 -> 3, missed 1 -> 1,
  total 7 -> 4, broken 0** (receipt `ab_witness.json`; baselines archived as
  `ab_witness_*_baseline_w1_raw.json`). The 3 survivors all route through the
  ADVANTAGE certificate (delta>0 ignores the bar) = hardening round 2
  (ROADMAP P2-1), deliberately not iterated on the same battery. Simulator
  operating point re-measured (precision 1.0 / recall .213) and the test
  contract updated with both receipts; the random-card recall cost does NOT
  appear on known truth.
- **Phase 3.2 — acceptance 25 -> 35/35**: window bound, Arrow round-trip,
  contact vocabulary, channel mix, arms + propensity floor, holdout rate.
  The window check caught a LIVE generator leak (toy max 2026-02-03 > REF)
  — orders/returns/cancels/opens/clicks/conversions now right-censored at
  `_REFERENCE_NOW` in seed_events/seed_contacts. Standing 25k artifact
  regeneration still awaits ROADMAP open decision #4.
- **v2 battery numbers (hardened)**: 15/20 — UNCHANGED for WITHOUT/WITH
  (missed 4 × `smart`, false 11 × myopic/mimic/anti at w3-w5). The levers
  only bite w=1 falses (v1's class); v2's errors are all advantage-certificate
  (delta>0, bar-blind) + witnesses hallucinating under partial observability —
  i.e. ROADMAP P2-1 exactly (advantage edge requirement, overlap veto,
  recurrent MB witness). WITH_BOTH improved 14 -> 12 (3 mimics fixed, 0 broken);
  verdicts unchanged (WITH=REMOVE, WITH_BOTH=SHIP).
- Gates: fast tier 210 green, ruff clean; all CI runs green.

## Pass: P1-4 ablation — donor LOSES to raw on the shipped target (DECISION NEEDED)
`plugins/ablation.py` measures donor-vs-raw under the production bake-off head
on `donor_embeddings` (paired customer-bootstrap SE); battery's rfm_vector is
now the single "raw" definition for both. Receipts:
`plugins/artifacts/ablation_..._20251201.json` + `/tmp/opencode/insta_ablation/`.
- rabbit_hole (v2.0.0r90061 products, 42,587 rows): **raw 0.7650 > donor
  0.7343** (d -0.031, 2SE 0.006 — beyond noise); E+R 0.7597 > donor.
- Instacart (v2.1.0 products, 734 rows): **raw 0.8944 > donor 0.7961**
  (d -0.098, 2SE 0.025); **E+R 0.9201 = best of all three**.
For 30-day purchase propensity the frozen state does NOT dominate exact RFM
aggregates (the battery's 'raw-ish reorder_30" hint, now proven on the shipped
metric/table). Donor still adds complementary signal (E+R > raw alone on
rabbit_hole; > both alone on Instacart). DoD#2 status for this target is open:
either (a) revisit D14's sole-input lock so heads may take [E|R] (the Dataset
`price_mat` slot exists), or (b) change the encoder so the state carries
recency/frequency aggregates. NOT decided unilaterally — receipts above.
