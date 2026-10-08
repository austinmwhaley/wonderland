# DECISIONS — wonderland (append-only)

Architectural decision record. Read before non-trivial changes; append after
every non-trivial decision so choices are never re-litigated. Format:
Context & problem → Alternatives considered → Decision & justification
(principle) → Trade-offs accepted.

---

## DEC-001 — One production stack: the CFM path only

- **Date:** 2026-09-28 (commit `009edef`)
- **Context:** looking_glass carried two stacks — the production CFM path
  (used by every downstream consumer) and an orphaned `create_*` factory
  library (EntityCore + Mamba-2 SequenceEngine). Import-graph proof: zero
  external consumers, zero CFM-core edges into the island.
- **Alternatives:** (a) maintain both; (b) finish the factory stack as the
  public API; (c) delete the island.
- **Decision:** delete it (23 modules, demo, 60 tests, `lancedb`/`mamba-ssm`/
  `bitsandbytes` deps). Principle: *simple over complex* + doctrine
  "prefer deletion over accretion".
- **Trade-offs accepted:** lost a generic library surface nobody used; specs
  in `looking_glass/specs/` are now historical records.

## DEC-002 — v2.1.0 encoder semantics (correctness over continuity)

- **Date:** 2026-09-30 (commit `22a689b`)
- **Context:** vendored-fork audit found: half-life derived from a merged
  timeline pinned it at the 1h floor on every dataset; training-anchor
  readouts were unfaded while serving fades; governor returned a non-minimal
  state; `min_events=3` hid sparse customers from the state store.
- **Alternatives:** (a) ignore (numbers were "stable"); (b) patch silently;
  (c) fix and bump the behavior version.
- **Decision:** fix all four, bump encoder `v2.0.0 → v2.1.0`, re-cert later.
  Principle: *correctness & robustness is tier 1*; stable-but-wrong is the
  worst outcome.
- **Trade-offs accepted:** certified v2.0.0 receipts are now known to measure
  an un-faded readout — re-cert is pending and blocks nothing until run.

## DEC-003 — Evidence identity (receipts survive their writers)

- **Date:** 2026-10-01 (commit `ad6c861`)
- **Context:** ladder rung dirs were not as_of-scoped (Nov's sizing receipt
  destroyed by Dec's run); monthly head/artifact overwrites under a constant
  tag; null encoder pins could pass the gate; no seed/git stamps.
- **Alternatives:** (a) accept overwrites as "regenerable"; (b) archive
  everything (costly); (c) scope by as_of + archive the run's artifacts.
- **Decision:** per-as_of rung dirs, as-of-stamped heads, `archives/` copies,
  reject null pins, record seed + git SHA. Principle: *correctness* +
  doctrine #10/#11 (receipts are identity).
- **Trade-offs accepted:** artifact dirs grow (archived copies retained on
  purpose); some legacy paths became orphans.

## DEC-004 — white_queen hardening: pre-committed rules, no iterating on the
same battery

- **Date:** 2026-10-01 (commit `0cf06a4`)
- **Context:** known-truth battery v1 = 7/25 errors (1 missed, 6 false), all
  false deploys ≤2 witnesses, several behavior-clones with truth == bar.
- **Alternatives:** (a) witnesses ≥2 alone; (b) certificate-vs-bar alone;
  (c) both; (d) keep tuning until 0 errors on the same battery.
- **Decision:** (c), measured once: 7→4 errors, 0 broken, missed unchanged;
  the 3 survivors (advantage-certificate route) are scoped as round 2 with
  NEW cells rather than tuned against this battery. Principle: *robust &
  antifragile* — a gate tuned on its own test is not a gate.
- **Trade-offs accepted:** simulator recall .39 → .213 on random cards (real
  battery shows no recall loss); errors are 4, not 0.

## DEC-005 — Ship/no-ship gates use measured noise, not literals

- **Date:** 2026-10-01 (commit `85019f0`)
- **Context:** plugin gates were literals (`lift>1.0`, `gap≤0.05`, zero-
  margin AUC-vs-baseline, winner = single-split argmax) — doctrine #1/#8
  violated at the exact point decisions are made.
- **Alternatives:** (a) keep literals (stable); (b) statistical gates
  everywhere (over-engineered); (c) paired-SE where rows are shared, fold-SE
  for singles, derived tolerances, roster-order tie-breaks.
- **Decision:** (c). Principle: *correctness* (a tie must not ship) with
  *simple over complex* (reuse existing `_paired_auc_se`/`_fold_se`; no new
  framework). Verified 5/5 on certified products with headroom.
- **Trade-offs accepted:** gates can now FAIL on statistically-tied data
  that the old rules passed — intended; thresholds are data-dependent so
  receipts must travel with every gate report.

## DEC-006 — Primary goal & the encoder-augmentation path (teach, don't feed)

- **Date:** 2026-10-01 (this entry)
- **Context & problem:** the E-vs-R ablation (P1-4) proved raw RFM beats the
  frozen donor on the shipped target — rabbit_hole 0.765 vs 0.734 and
  Instacart 0.894 vs 0.796 (both beyond paired noise) — while donor+raw beats
  either alone. The user's goal: rabbit_hole → looking_glass → propensity
  must work at its best, by making the encoder LEARN what raw has, without
  bolting hand-built features onto it (D14's sole-input lock stands for now).
- **Alternatives considered:**
  1. Concat raw features into the head input (fastest AUC; violates D14 and
     the "give it things that allow it to learn" directive).
  2. Teach-the-state, in three steps: (S0) decompose raw into components to
     measure WHICH statistic wins; (S1/B) query-time readout — train the
     fade-to-anchor path the serving actually uses (it has never seen a
     gradient); (S2/A) multi-horizon future-aggregate objectives (counts /
     value / days-to-next over gap-quantile horizons) so history-integration
     becomes a learning requirement rather than an input.
  3. Widen the dynamics family (unit-root/integrator expert + timescale span
     from gap quantiles) — capacity, deferred until S0/S1 say it's needed
     (the old "multi-scale gave no gain" verdict was measured under the 1h
     half-life bug and does not transfer).
- **Decision:** option 2, staged per the lifecycle — spike first (S0),
  then correctness (S1/B unconditionally: the serving path must be trained),
  then learning pressure (S2/A, with S3/C only if S0 points at counting and
  S1+S2 miss). **Pre-committed acceptance:** every step must flip
  `plugins.ablation` to `donor_beats_raw=True` on BOTH streams with the
  sufficiency battery (unique ≥75%) and plugin gate (4/4) unchanged.
  **Pre-committed fallback:** if S1+S2 fail the acceptance on both streams,
  the D14 question (heads may take `[E|R]`) is reopened with these receipts —
  no re-litigation. Principles: *correctness first* (S1), *simple over
  complex* (smallest change that could flip the measurement; no new
  framework), *earn the right to scale* (spike before architecture).
- **Trade-offs accepted:** slower than concatenating RFM; encoder version
  bumps again (re-cert); S2 adds loss terms (new hyperparameters must be
  derived + receipted, not tuned by hand).

## DEC-007 — The ablation is the encoder's regression test

- **Date:** 2026-10-01
- **Context:** donor quality was previously "proven" by Spearman-of-Ridge on
  `anchor_embeddings` — a metric/table the product never ships.
- **Decision:** `plugins/ablation` (paired ΔAUC on `donor_embeddings` under
  the production bake-off head) is the standing acceptance instrument for
  encoder changes (see DEC-006), with the sufficiency battery and plugin gate
  as unchanged guards. Principle: *measure everything; gate on ground truth*.
- **Trade-offs accepted:** every encoder change now costs 2×~2min ablation
  runs — cheap insurance.
## DEC-006a — Stage-1 spike results (amends/validates DEC-006 sequencing)

- **Date:** 2026-10-01 (receipts: `plugins/artifacts/ablation_..._20251201.json`,
  `/tmp/opencode/insta_ablation/ablation_..._20251101.json`)
- **Context:** DEC-006 ordered S0 (decompose raw) before aiming S1/S2. Ran
  `plugins/ablation --components` on both streams (component vs donor, paired
  2*SE):
  - **rabbit_hole:** frequency +0.0227 BEATS donor, event_mix +0.0195 BEATS;
    recency −0.1237 (worst single statistic). Winners = history-integration
    statistics (counts / mean gap / per-type histograms over history).
  - **Instacart:** recency +0.0652 BEATS donor (frequency/event_mix/monetary
    all lose). Its negatives are inactivity-defined (30d-censored gaps), so
    time-since-last-event is the label.
- **Decision (validates the DEC-006 order, gives each stage a proven target):**
  1. **S1 (query-time readout)** = the Instacart lever + correctness — the
     serving fade path has never seen a gradient, and recency beats the donor
     exactly where recency defines the label.
  2. **S2 (multi-horizon aggregate targets)** = the rabbit_hole lever —
     counting/histogram prediction is precisely what frequency+event_mix
     require.
  3. **S3 (integrator dynamics / timescale span)** stays conditional on S2.
  No single statistic wins both streams — the state must be able to represent
  BOTH gap-since-last-event and long-window integration; that is the encoder
  work, not a feature list.
- **Trade-offs accepted:** two-stage encoder work instead of one; each stage
  pays a version bump and re-cert.

## DEC-008 — The goal, stated by the operator: encoder judged on Layer B ONLY

- **Date:** 2026-10-01
- **Context & problem:** the operator fixed the division of labor:
  **rabbit_hole's goal = provide the data looking_glass needs; looking_glass's
  goal = produce the best encoder possible, judged on self-supervision quality
  ONLY (Layer B).** Audited against that definition, current Layer B has
  three violations/gaps (receipts):
  1. **Sample-A ladder sizes encoder training data by DOWNSTREAM
     purchase-propensity AUC** (`plugins/ladder_sample_a.py` rule; consumed by
     `customer_foundation_model`) — a Layer C judgment inside a Layer B
     config decision. (The ladder already records per-rung held-out `ce` as
     "corroboration" — the self-supervised number is already measured.)
  2. **Selection/verification is narrower than the objective**: the governor
     stops and selects on next-event CE ALONE (`cfm_training._val_loss`);
     `validate()` gates on argmax ACCURACY over the first 200 sequences (8 for
     causality), with an unseeded `torch.randint` probe — no CE/NLL gate,
     no uncertainty, no portfolio view.
  3. **The self-supervised portfolio is local** (next-step type/time/value,
     occurrence at median gap, near-JEPA) — nothing in the exam demands
     long-horizon integration or states read at query times. The DEC-006a
     spike (frequency/event-mix win downstream on rabbit_hole; recency wins
     on Instacart) is the downstream shadow of that missing exam content.
- **Alternatives:** (a) keep judging the encoder with downstream AUC (ladder,
  battery, ablation as primary); (b) strict Layer-B judging — held-out
  self-supervised portfolio is the ONLY definition of "best encoder";
  downstream instruments stay as sufficiency GUARDS at the Layer B/C seam;
  (c) merge everything into one score.
- **Decision:** (b). The encoder is optimized, stopped, selected, and sized
  (sample_A) on held-out self-supervised loss only. Consequences:
  - sample_A ladder re-sizes rungs on paired self-supervised CE (per-row
    log-loss diff, same machinery as the AUC pairing), downstream AUC demoted
    to a recorded guard, not the chooser;
  - `validate()` grows CE/NLL rows over the full held-out set, seeded probes;
  - S1 (query-time readout) and S2 (multi-horizon aggregate prediction) are
    re-scoped as **completing the self-supervised portfolio** (self-supervised
    tasks at derived horizons — no labels, no downstream metric in the loss);
  - battery + ablation remain sufficiency guards (DoD#2 / DEC-007) — they can
    VETO a release, never define or optimize the encoder.
  Principles: *correctness* (a metric that lies about the goal is a bug),
  *simple over complex* (use the `ce` receipts the ladder already writes).
- **Trade-offs accepted:** give up downstream-tuned data sizing (product
  alignment moves to the guards); re-cert must show the portfolio, not just
  one CE; ladder rework is real work in front of S1/S2.

## DEC-009 — Objectives audit: keep 13, fix what's broken, catalog everything

- **Date:** 2026-10-01
- **Context & problem:** operator asked (1) do we care about each trained
  capability and (2) train/grade on ALL of them, rigorously (geometry
  included), defensible/robust/reliable. Audited every objective against the
  code, not the intent.
- **Alternatives:** (a) trim to a minimal set; (b) keep all + fix defects +
  build the missing grade; (c) redesign from scratch.
- **Decision:** (b), recorded in
  `looking_glass/specs/objectives_catalog.html` (the operator's 7-family menu
  + full landscape + status marks). Findings that changed code this cycle:
  - `mask` claimed bidirectional — actually CAUSAL (prefix scan); worse, the
    true brand/entity/value leaked into masked positions → redacted; frac
    from config. Causality now test-locked.
  - `contrast` has NO second view (uniformity-only; `gamma_contrast` dead) →
    status 🔧; views/augmentation design is the fix candidate.
  - `sf` phi named the purchase event → `sf_mode=event_types` (agnostic) is
    the default; checkpoint truth governs loads.
  - gaps closed: `query` (read-at-time) + `agg` (exact windows at derived
    horizons) added as objectives; selection now on the held-out COMBINED
    objective; portfolio grade (structure-skill vs destroyed data + geometry
    vs permuted nulls) is the Layer-B receipt.
- **Trade-offs accepted:** bigger objective set = slower steps (measured,
  acceptable at current scale); contrast stays ON-but-marked for continuity
  until views land; a few known defects ship as documented 🔧 not silent.
- **Guard:** nothing enters the default tuple without passing the §5 entry
  protocol in the catalog (capability probe → config → grade → guards).

## DEC-010 — Universal encoder: objectives are capability-driven, config-gated

- **Date:** 2026-10-01
- **Context & problem:** operator goal: "pass ANY event stream and the
  encoder trains the optimal universal set of objectives." Objectives have
  hard prerequisites (money fields, side actions, seasonality, vocab size)
  that differ per stream (rabbit_hole vs Instacart vs future real data).
- **Alternatives:** (a) one fixed tuple for all streams (breaks or wastes on
  every mismatch); (b) hand-tuned per stream (doesn't scale, unrecorded);
  (c) a capability probe derives an **objective plan receipt** per stream,
  config `--set` overrides win and are recorded, portfolio grades only what's
  enabled.
- **Decision:** (c). Design + probe fields + default profiles live in the
  catalog §5. Core set = anything with a sequence (`next, dt, occur, mask,
  order, jepa, redundancy, query, agg, sf`); everything else activates on
  proven capability. Principles: *adapt to the input by construction*,
  *dynamic & orchestrated* (behavior in config/receipts), *fail safe*
  (disabled-with-reason, never silent).
- **Trade-offs accepted:** plan machinery still to build (probe → receipt →
  wiring into resolve/registry); grades differ across streams (comparisons
  are per-stream, not cross-stream); until the probe lands, the default tuple
  is the fallback and is documented as such.

## DEC-011 — Default-ON objectives, opt-out only; catalog ships as HTML

- **Date:** 2026-10-07
- **Context & problem:** operator set two policies: (1) *every objective is ON
  by default; we opt OUT of objectives* (supersedes DEC-010's capability
  opt-in phrasing — the probe still records missing capabilities, but as
  automatic recorded auto-outs, not as preconditions for enabling), and (2)
  the objectives catalog exists **only as HTML**
  (`looking_glass/specs/objectives_catalog.html`) — no .md version.
- **Alternatives:** (a) keep opt-in plans; (b) default-on with explicit +
  data-driven opt-outs; (c) default-on with no auto-outs at all.
- **Decision:** (b). Enabling is the non-decision; disabling is always
  explicit or data-proven and always recorded (registry/plan receipt). The
  catalog entry protocol rewritten accordingly: new objectives land
  default-ON after tests + portfolio grade + version bump; `--set
  objectives=(...)` is the only operator surface; missing capability →
  auto-out with reason, never silent, never half-working. HTML chosen for the
  catalog (operator preference; self-contained, no repo .md), generated from
  the same content and linked from the README. Principles: *dynamic &
  orchestrated* (behavior in config/receipts), *fail safe* (recorded reasons).
- **Trade-offs accepted:** on streams lacking a capability, degenerate-input
  training is possible until the probe lands (documented; probe is build
  item (e) in the catalog's build order); HTML is not diff-friendly in
  review — mitigated by keeping DECISIONS.md as the textual rationale log.
- **Status marks:** ✅ implemented = default-ON; 🔧 implemented = default-ON
  with a scheduled fix (contrast stays ON); ⏸ not yet implemented (build
  candidates — landing means default-ON); ⛔ doctrine-excluded.

## DEC-012 — Portfolio composition: all 13 stay; combination map + meta-rule

- **Date:** 2026-10-07
- **Context & problem:** operator asked which objectives to keep and which
  combinations are good vs bad (JEPA and SF considered important; many others
  uncertain).
- **Alternatives:** (a) trim to a small core now; (b) keep all 13 with the
  combination contract written down + a measured keep/drop meta-rule.
- **Decision:** (b). Full combination map published in the catalog
  (§1.5: synergistic clusters = TPP core / hazard stack / future stack /
  geometry pair / structure regularizers / readout; managed tensions; bad
  combos banned: anything without `redundancy`, `query` without its heads,
  aggregates without intensity, regularizer-only portfolios). Minimal viable
  portfolio = 6 (`next, dt, sf, jepa, redundancy, query`). Cut order under
  pressure: contrast (until views) -> entity (vocab-dependent) -> occur
  (subsumed). **Meta-rule:** keep = portfolio structure-skill > 0 at re-cert;
  skill ~0 across re-certs = flagged drop-candidate — measured, not argued.
  Principle: *simple over complex* applied to TRAINING (uncertainty weights
  handle 13-way conflict) + *gate on ground truth* applied to the objectives
  themselves.
- **Trade-offs accepted:** larger default compute per step; two 🔧/new items
  (contrast views, portfolio CLI wiring) must land before the meta-rule is
  fully automatic.

## DEC-013 — Fielding decisions: six fielded combos, explicit not-fielded list

- **Date:** 2026-10-07
- **Context:** operator wanted the fielded combinations and the explicit
  not-fielded list with reasons/triggers (the §1.5 map made operational).
- **Decision:** field six combos by default (TPP core; clock stack; future
  stack; geometry pair; discipline drills; readout) = all 13 objectives;
  everything else stays unbuilt/unfielded with a NAMED TRIGGER (vocab scale,
  seasonality probe, periods, basket targets, sampling use case...).
  Promoted: the canary leak-detector objective to next cycle. Published as
  catalog §9 (HTML-only per DEC-011).
- **Trade-offs accepted:** several high-value candidates (quantile head,
  hazard) wait on triggers; the fielded set is judged by the portfolio grade
  whose sf yardstick still needs one fix.

## DEC-014 — Balancer: DWA replaces Kendall as default; FAMO/CAGrad = Pareto upgrades

- **Date:** 2026-10-07
- **Context & problem:** operator proposed DWA/GradNorm/FAMO-CAGrad to replace
  uncertainty weighting. The production v2.2.0 run CONFIRMED the defect
  empirically: Kendall's optimum s*=ln(L) sent redundancy's s to ~-9 (tiny
  geometric loss), giving a NEGATIVE combined contribution (0.5 - 4.5) the
  optimizer could improve by s-drift, not learning; held-out geometry was
  collapsing (eff-rank 0.23x null, redundancy 26x null) while the s-terms
  hid it.
- **Alternatives:** (a) keep Kendall (bounded-likelihood assumption violated
  by our geometric losses); (b) DWA (rate-of-change weights; scale-free; few
  lines); (c) GradNorm (gradient-norm balancing; needs shared-layer choice +
  aux optimizer); (d) FAMO/CAGrad (gradient-direction; directly Pareto).
- **Decision:** (b) DWA as DEFAULT (cfg.weight_mode="dwa", temp 2.0 = the only
  knob, paper default); Kendall kept as "uncertainty" mode for likelihood-pure
  stacks; "equal" mode for ablations. Weight trajectory recorded per eval
  (the frontier readout); portfolio per-task contributions are mode-aware.
  FAMO/CAGrad = catalogued upgrade with trigger: DWA's equal-rate
  approximation leaves an objective starved (portfolio skill ~0 while its
  weight is high) -> switch to FAMO. GradNorm declined for now (invasive,
  same benefit class as DWA at our scale). Principles: *dynamic &
  orchestrated* (weights learned, zero hand-tuning), *robust* (non-bounded
  losses can't exploit), *measure everything* (trajectory + per-objective
  skills = the frontier evidence).
- **Trade-offs accepted:** DWA approximates equal learning speed, not a true
  Pareto step (FAMO reserved for that); warmup = 2 evals at equal weights;
  v2.2.0's grade is superseded (geometry collapse finding stands as the
  motivation).

## DEC-015 — Geometry fix #1: variance-floor objective (VICReg-style hinge)

- **Date:** 2026-10-07
- **Context:** two consecutive production grades measured held-out geometry
  collapse (eff-rank 0.23-0.26 x null; held-out redundancy 18-26x the
  marginal null) while train-batch redundancy looked fine — batch-local
  decorrelation never treats (a) scale collapse and (b) population-global
  correlation.
- **Alternatives:** (a) variance-floor hinge (VICReg); (b) cross-batch
  redundancy bank; (c) contrast views; (d) accept collapse (heads only need
  2-3 dims).
- **Decision:** (a) now — new `variance` objective, hinge
  `mean_j relu(1 - std(proj(h)_j))`, default-ON (DEC-011), graded by the
  geometry gate (its OUTCOME) rather than a destroyed-null (it is not a
  predictive loss — same treatment as redundancy). (c) stays queued (pairs
  into the full VICReg recipe); (b) is the follow-up if population-global
  correlation persists; (d) rejected — headroom is the point of a UNIVERSAL
  donor (future unknown heads need unused dims).
- **Trade-offs accepted:** one more loss term (DWA balances it); the gate may
  still fail if the null's bar (0.3 x) is met but headroom stays thin — the
  ratio is reported either way.

## DEC-017 — geometry_boost: the explicit Pareto coordinate + frontier sweep

- **Date:** 2026-10-07
- **Context:** three falsified attempts at raising held-out eff-rank via
  batch-local losses (variance hinge, eigh rank, trace-form rank — all
  measured, geometry ratio 0.23/0.26/0.18) while ALL predictive skills stay
  positive. Conclusion: the predictive objectives genuinely prefer a
  low-rank state — this is a Pareto tension, not a bug.
- **Decision:** add `geometry_boost` (multiplier on variance/rank/redundancy
  losses; default 1.0; recorded in the registry) and sweep {5, 20} on
  Instacart to map the skills-vs-headroom frontier. The "best recipe" is then
  a CHOICE on a measured curve, not an opinion.
- **Trade-offs accepted:** predictive skills may dip at high boost; the sweep
  quantifies exactly how much.

## DEC-020 — Barrier geometry + grouped PCGrad (v2.8.0): the autonomous loop

- **Date:** 2026-10-07
- **Context:** operator's autonomous-architecture blueprint (dimensionless
  losses / bank+PID governor / eigenvalue barrier / PCGrad / R² yardsticks).
  Components 1, 2, 5 already landed (v2.6.0/v2.7.0); 3 and 4 were missing.
- **Decision:** (a) log-det barrier on the bank covariance
  (-log det(C + eps·I), eps scaled to the trace — infinite wall at collapse;
  the soft tau-hinge stays as a gentle floor); (b) GROUPED PCGrad: geometry-
  family gradient projected onto the predictive gradient's plane on conflict
  (2 grouped backwards, not 13 pairwise — affordable at our batch sizes;
  geometry can never destroy predictive learning; applied via manual param
  update, no .backward()). Full GradNorm declined (cost 13 backwards/step;
  the EMA-normalized DWA covers the same benefit class); pairwise-13 PCGrad
  reserved as the escalation if grouped proves insufficient.
- **Trade-offs accepted:** the projection is asymmetric (protects predictive
  from geometry, not vice versa — intended: the guard must never block
  learning); barrier's eps is a documented literal scaled to the trace.

## DEC-023 — Intrinsic foundation proofs (representation-space, no probes)

- **Date:** 2026-10-07
- **Context:** operator's intrinsic-proof blueprint: demonstrate the encoder
  is foundational from the representation space alone, without training any
  downstream probe (which would make the claim circular).
- **Decision:** `looking_glass/intrinsic.py`, four proofs, each gate-row +
  receipt, runnable on any frozen checkpoint:
  1. **Disentanglement** — channel mutual information (kNN estimator) vs a
     shuffled null; OOT (temporal-split) covariance invariance vs a
     within-period null. Proves per-channel independence + temporal stability
     of the geometry.
  2. **Local Lipschitz** — perturb a realistic event time by ±1 day,
     re-encode, measure ‖Δz‖/‖Δt‖. Proves a smooth manifold, not memorization.
  3. **Trajectory smoothness** — per-step velocity/acceleration vectors; mean
     directional continuity cos(v_t, v_{t+1}). Proves z is a continuous
     dynamical state variable.
  4. **Information plane** — multi-horizon predictive losses (the agg/next/dt/
     sf skills) as the measured lower bound on I(z; future); the compression
     axis is proof 1's MI + eff-rank. Descriptive.
- **Trade-offs accepted:** MINE/InfoNCE would need a trained estimator (a
  probe → circular); we use proxy/estimator bounds and say so in the receipt.
  The Lipschitz and OOT gates are deliberately conservative literals,
  documented. Principles: *gate on ground truth*, *measure everything*.

## DEC-024 — Half-life experiment: trajectory zigzag is structural, not tunable

- **Date:** 2026-10-07
- **Context:** the intrinsic trajectory proof measured cos(v_t, v_{t+1}) =
  -0.29 (raw state) on the v2.9.0 encoder. Hypothesis: short half-life makes
  the state too reactive; doubling it should smooth the trajectory.
- **Experiment:** retrained with `--set state_half_life_days=60.6` (doubled
  from the derived 30.3d). All 14 portfolio rows PASS (geometry 0.817).
  Trajectory cos: **-0.298** — unchanged from -0.29 at 30.3d.
- **Conclusion:** the zigzag is NOT caused by decay rate. It is caused by the
  event content embeddings: alternating event types (view/order/view) have
  orthogonal token directions, so the state tracks *content transitions*,
  not smooth behavioral trends. The behavioral trend information is present
  (sf/agg pass) but the trajectory traces the event sequence's actual
  jaggedness. Doubling the half-life is a dead end for trajectory smoothing.
- **Implication:** if trajectory smoothness is required for a downstream
  consumer, it must come from the readout (e.g., an EMA-smoothed donor
  output) or from a dual-velocity architecture — not from the half-life.
- **v2.9.0 is LOCKED** as the production encoder: all 13 portfolio objectives
  PASS, geometry PASS by construction (whitened readout, ratio 0.82),
  Lipschitz smooth (p99 0.0016), MI low (0.044 nats), canaries clean.

## DEC-025 — v3.0.0: Dual-Velocity Multi-Horizon Encoder

- **Date:** 2026-10-07
- **Context:** the half-life experiment (DEC-024) proved trajectory zigzag is
  structural to a single-state recurrence: event-content embeddings alternate
  (view/order/view), and one state tracks content, not smooth behavioral
  trends. The architecture fix is dual-velocity: two independent SelectiveSSM
  experts with spread delta_bias init (fast = -1.5, slow = +1.5), each
  tracking a different timescale.
- **Alternatives:** (a) dual-velocity MultiScaleSSM (K=2); (b) dual-
  architecture (two separate models); (c) accept the zigzag.
- **Decision:** (a) — the existing MultiScaleSSM bank already supports K>1
  experts; spreading the delta_bias init breaks the symmetry so the two
  experts discover different timescales (fast tracks token transitions for
  next/dt/query/mask; slow accumulates behavioral trends for sf/agg/value/
  entity). Ortho-loss (cross-covariance between expert state components)
  prevents collapse into the same subspace. Trajectory measured on the SLOW
  state (the last 1/K of the state vector). Rolling EMA whitening (per-eval,
  not post-hoc). GPU-aware load_frozen_encoder.
- **Trade-offs accepted:** 2× trunk compute (2 experts × half channels = same
  total params, but 2 ssm scans); ortho-loss adds a batch-matmul; the
  trajectory proof needs 64 prefix forwards per trajectory (measured: ~10 min
  on CPU, seconds on GPU — GPU-aware loading fixes this).

## DEC-026 — Donor-boundary whitening must be fit & applied AFTER `proj` (v3.0.1)

- **Date:** 2026-10-07
- **Context:** v3.0.0's portfolio geometry gate failed (eff_rank/dim 0.06 vs bar
  0.30) despite the checkpoint shipping a whitening transform whose own receipt
  reported `pr_after 0.85`. Diagnosed by direct measurement on the loaded
  checkpoint: `donor_batch` was `proj(_whiten(h))` — whitening fit on `h` and
  applied *before* `proj`, but `proj` is itself ill-conditioned (measured
  singular values 0.0018..7.9, ratio ~4380), so it re-collapsed the full-rank
  whitened states back to PR/dim 0.06. Whitening the *consumed* boundary
  (`_whiten(proj(h))`) restores PR/dim 0.761 (matches v2.9.0's 0.82). The stored
  transform was additionally stale (fit on train `a_seqs`, implied eigenvalue
  floor 0.01, while held-out states reach 1e-11).
- **Alternatives:** (a) grade `_whiten(h)` (pre-proj) rather than the consumed
  readout; (b) constrain `proj` to be orthogonal; (c) whiten the true consumed
  representation (after `proj`) and refit at grade time.
- **Decision:** (c) — DEC-022 says the transform lives *at the donor boundary*,
  and the boundary is what downstream reads (`proj(h)`). `donor`/`donor_batch`/
  `embed` now apply `proj` first then `_whiten`; `compute_whitening` collects
  `proj(h)` (whitening disabled) to fit the transform. The portfolio + intrinsic
  graders grade the **frozen** transform on their held-out split (the honest
  holdout number, no refit-on-the-graded-split); only a checkpoint that ships no
  transform at all gets one derived there. Fail-safe: if the fit does not
  *increase* effective rank (degenerate/untrained states), keep the transform a
  no-op rather than ship a worse boundary.
- **Trade-offs accepted:** (a) rejected as gaming — it grades a representation no
  head consumes. (b) rejected — constraining proj fights the trained trunk.
  A grader that *refits* on the graded split was tried and rejected: it makes
  geometry pass by construction (measured 0.76) while the shipped transform's
  honest held-out number is 0.65 — grading the frozen artifact is the real gate.
- **Also fixed in this change:** `task_losses_chunked` (portfolio + intrinsic
  `_info_plane` ran `_task_losses` on all ~2.6k val sequences in one batch; the
  `agg` objective's T×T windows OOM'd a 7.6 GB GPU — measured 2.56 GB alloc
  failure). Regression test:
  `tests/test_intrinsic.py::test_donor_boundary_whitening_is_self_consistent`.
- **Result:** v3.0.1 — old v3.0.0 weights, checkpoint patched in place to refit
  the boundary transform (`pr_before 0.12 → pr_after 0.76`), registry receipt +
  products rebuilt. Honest held-out grade (frozen transform): **portfolio PASS
  14/14 + geometry eff_rank 167.0/256 = 0.652**, **intrinsic 4/5** (OOT 1.545→
  **1.43** PASS, MI 0.013 PASS, Lipschitz PASS, info plane PASS; only the known
  structural slow-state trajectory zigzag FAILs, per DEC-024).

## DEC-027 — v3.1.0: derived slow expert + low-pass intent filter (DEC-024 fix)

- **Date:** 2026-10-07
- **Context:** the one open intrinsic gap was the slow-state trajectory cos
  (v3.0.x ~-0.34 < 0). Two distinct problems were found by direct measurement:
  (1) the trajectory proof (and the ortho split) hardcoded slow = the last
  1/K of the state, but `decay = exp(-softplus(W_delta(x)+delta_bias))` makes
  `delta_bias=+1.5` the FAST expert — the LAST half was measured to be the fast
  expert (mean decay 0.409 vs 0.847 for expert 0), so the proof was grading the
  wrong channels; (2) even the true slow expert's velocity zigzags (cos -0.026),
  because it is a first-order filter of the raw token stream whose event-type
  component flips every event (DEC-024).
- **Alternatives:** (a) only fix the selector; (b) smooth the slow expert's
  contents ("intent filter"); (c) both.
- **Decision:** (c). `slow_expert_index()` derives the slow expert as
  `argmin(e.delta_bias)` (architecture-derived, never "the last half"); the
  trajectory proof and ortho split now use it. The slow expert receives a
  **low-pass intent filter**: a learned `W_intent` projection then a
  per-channel EMA whose retention is learned and initialized from that expert's
  own decay (`exp(-softplus(delta_bias))`). Its state is therefore a
  second-order low-pass of the token stream, so its velocity is smooth. Measured
  on a synthetic alternating sequence: slow-state directional cos -0.78 →
  +0.79. Config: `slow_intent_filter` (default on; opt out via `--set`).
  Checkpoint records `slow_intent` for backward-compatible loading.
- **Trade-offs accepted:** the EMA alone does not smooth velocity (a first-order
  filter's velocity tracks its input) — the cascade with the SSM is what does;
  this is exactly why the fix is architecturally placed on the SLOW expert.
  Cost: one extra linear + one scan on the slow expert only.
- **Open follow-up (recommended, not yet implemented):** a Jacobian
  orthogonality loss (cross-subspace input-sensitivity) to push geometry PR
  toward >0.75; the existing state cross-covariance `ortho` loss already PASSes.

## DEC-028 — v3.2.0: condition-capped whitening + soft-spectrum isotropy

- **Date:** 2026-10-07
- **Context:** the boundary whitening Sigma^{-1/2} (DEC-022) used an absolute
  1e-4 eigenvalue floor. On an ill-conditioned state covariance this amplified
  near-null directions ~1e3x, so FP32 CUDA matmul nondeterminism became large
  swings in the measured geometry: the IDENTICAL command/checkpoint gave
  eff_rank 21.6 then 56.2 (and the v3.1.0 OOT proof 1.09 vs 2.36). The geometry
  and OOT gates were therefore not reproducible — the v3.0.1 "0.652" passed on
  noise, and the model's true consumed rank is only ~0.084 of dim.
- **Alternatives:** (a) measure geometry on the raw (unwhitened) representation;
  (b) condition-capped whitening; (c) raise the true rank by training.
- **Decision:** (b) + (c).
  - **(b)** `compute_whitening` now floors eigenvalues at
    `whiten_cond_floor * max_eigenvalue` (config `whiten_cond_floor`, default
    1e-2 -> kappa(W) <= 10). Measured: geometry is now stable under 1e-4 state
    perturbations (0.145 -> 0.145), and honest held-out PR/dim is 0.18 (tau=1e-2)
    to 0.24 (tau=1e-3) — below the 0.30 bar, which is the true value.
  - **(c)** new `spectrum` objective: penalize `var(log(per-dim variance))` so
    no single direction dominates the participation ratio (`rank` maximizes PR
    but is insensitive to a dominant direction; `variance` only lower-bounds
    std). Together with `redundancy` (decorrelation) this pushes the covariance
    toward isotropic. Folded into GEOMETRY_FAMILY and GATED_EXCLUSIONS.
- **Trade-offs accepted:** honest geometry is below the 0.30 gate; the gate
  itself was measuring amplified noise before. Reproducibility (doctrine:
  identity = behavior) is prioritized over a passing number. `rank` already
  trained on raw `proj(h)` (not the whitened space), so no re-target was needed.
- **Expected:** geometry/OOT reproducible; 13/13 objective skills; slow-trajectory
  5/5 preserved; geometry value honest (target >0.30 via the spectrum loss).
- **Measured (v3.2.0r682246):** geometry is now REPRODUCIBLE — three identical
  CLI runs gave eff_rank 54.7 / 53.2 / 52.4 (PR/dim 0.213 / 0.207 / 0.204),
  versus 21.6 → 56.2 before. Honest geometry is **0.21 < 0.30 (FAIL)** — the
  gate was previously passing on amplified noise. The `spectrum` loss lifted the
  skills sharply (agg +38.6, occur +11.2, sf +3.5, order +1.37, value +0.28) and
  the slow-trajectory cos to **+0.267** (was +0.013); MI/Lipschitz/info-plane
  PASS. OOT is only partially reproducible (**2.15, 2.52** — still FAIL vs 1.5),
  so that gate also needs a scale-free reformulation (see v4.0 plan).

## DEC-029 — v4.0: self-governing encoder (all four macro shifts)

- **Date:** 2026-10-07
- **Context:** the v3.2.0 finding — the whitening-dependent gates were passing on
  amplified numerical noise (identical command gave eff_rank 21.6 then 56.2) and
  the honest consumed rank was ~0.08-0.21 of dim. The fix is not another static
  patch but making isotropy, calibration, and control intrinsic.
- **Decision (recipe, all four shifts, config-gated, defaults ON for v4.0):**
  1. **Unified multi-timescale SSM** (`unified_ssm`): one wide SelectiveSSM with a
     per-channel log-uniform `delta_bias` spectrum (fast→slow), replacing the
     hand-split 2-expert bank + DEC-027 intent filter. `slow_state_slice` selects
     the slow channel half DERIVED from the learned spectrum.
  2. **Differentiable Newton-Schulz ZCA** (`zca`): `ns_zca` in the donor forward
     during training (isotropy by construction; gradients shape it). Spectrum
     normalized by the spectral norm (power iteration) so NS converges; `eps`
     ridge caps the condition number (no noise amplification). Inference uses the
     frozen condition-capped transform, so train/serve agree.
  3. **Log-det trunk-volume barrier** (`volume` objective, Step 3): maximize the
     volume of the raw projected covariance (data-derived eps) so ZCA cannot fake
     rank from zero-variance directions. Complements `spectrum`/`rank`/`variance`.
  4. **Scale-free self-calibrating gates** (Step 4): geometry gate is now the
     Marchenko-Pastur-normalized isotropy score
     `(PR/dim − PR_MP(γ))/(1 − PR_MP(γ)) > 0` (0 = pure noise, 1 = isotropic;
     dimension/sample-invariant). OOT gate is the mean canonical correlation of
     temporal splits over the effective rank (`> 0.85`), replacing the covariance
     ratio.
  5. **Adaptive Information Bottleneck + self-paced controller** (Step 5): a
     Gaussian-KL compression term on projected codes with its multiplier
     `_aib_beta` tuned by dual ascent on the batch participation ratio, on top of
     DWA + grouped PCGrad.
- **Trade-offs:** more moving parts and one more retrain; all shifts are
  config-gated so any can be disabled. Honest gates may still read FAIL if the
  trunk's true rank is genuinely low — that is the point (no more noise-passing).
- **Measured:** _pending full v4.0 run (`/tmp/opencode/insta_v40`)._
