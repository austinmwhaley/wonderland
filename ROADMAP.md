# ROADMAP — wonderland

Written after a full-project scan (decisioning layers, data layer, plugins,
measurement, hygiene). Suite: **216 tests** (202 fast + 14 slow; fast tier
`pytest -m "not slow"` ≈ 55s). Receipts cited below are on disk unless marked
"projected".

---

## Hard constraint (product)

**Production logs have no holdout and no A/B — nothing we do may require one.**
Enforced in code, not just documented:

| | On observational logs (production) | On lab/rabbit_hole data (has control) |
|---|---|---|
| OPE DEPLOY/HOLD (white_queen) | ✅ `estimate_propensity=True` (provenance receipt) + sensitivity bounds | ✅ (logged propensity) |
| Prediction heads (CLV, churn, propensity) | ✅ pure prediction, no causal claim | ✅ |
| Certification-gated controllers | ✅ reads frozen artifacts | ✅ |
| Incrementality / uplift / response model | ❌ **rejects**: `NotIdentifiableError` | ✅ randomized control |
| IPW arm effects | ❌ rejects without logged propensity | ✅ |
| nba WHO step | empty targeting **with a printed receipt** | ✅ uplift responders |

## Where we are (DoD)

| DoD | State |
|---|---|
| 1 stream + identifiable design | ✅ lab data; production = observational by constraint |
| 2 universal donor | ✅ — but the receipt is the weak spot: battery measures **Spearman on `anchor_embeddings`** while production heads consume **AUC on `donor_embeddings`** (fix: P1-4) |
| 3 plugin gates | ✅ 4/4 on rabbit_hole and Instacart at production sample size — thresholds are literals on one split (fix: P1-3) |
| 4 red_king improves white_queen | ✅ resolved: REMOVE, locked by tests |
| 5 red_queen multi-cadence NBA | ✅ + observational guards; personalization/uncertainty still open (P2-2) |
| 6 caterpillar "why" | 🟡 v1 per-customer only (P2-3) |
| 7 versioned artifacts + receipts | ⚠️ receipts exist but evidence is overwritten and partly stale (Phase 0 exists for this) |

---

## North star — Phase U: the UNIVERSAL embedding (`loop/UNIVERSALITY.md`)

The goal (v8-sharpened) is a **minimal, bounded, universal sufficient statistic** of the stream:
sufficient for the **full observable future**, so any present/future consumer (ours or another
team's, known or unknown) is covered **by construction**. Judged by
`U = 1 − worst-case_τ oracle-gap_τ` over a **broad, adversarial, held-out, self-supervised** family
at **fixed capacity** (oracle = full history / Bayes floor). `U → 1` ⟺ done.

Anti-overfit (binding): Layer B self-supervised only; reference = full-history oracle, never our
tasks; family broader than any consumer + adversarially grown; **known uses VETO, never SHAPE**;
**accessibility** (probe parity) and **geometry/similarity-usability** are separate gates;
unlogged events carried as uncertainty.

**Phase U workstream (sequenced after Phase 0's evidence fixes for the real-stream legs):**

- **U1. Build the universality gate** — `looking_glass/universality.py`: the oracle-gap /
  history-residual test (`future ⊥ history | E`), a broad self-supervised future family, and the
  `U` scalar. Rung-0 on existing checkpoints. **DONE (DEC-117):** Rung-0 instacart **U=0.2227**,
  worst-case = mass axis.
- **U2. Wire the Layer-A lab to exact `U`** — on synthetic streams the true sufficient statistic
  is known ⇒ `U` is exact; verify the *machinery* and that a deliberately-lossy encoder scores
  lower while an over-capacity one does not score higher (saturation). **DONE (DEC-118, machinery):**
  sufficient 0.00 · magnitude-dropped 0.65 · over-capacity 0.00. Remaining: repeat on a generated
  ladder stream with *trained* lossy/sufficient encoders (Rung 1/2).
- **U3. Adversarial family** — a growing task-search adversary that finds where `E` loses most
  (worst-case objective).
- **U4. Accessibility + similarity gates** — probe-parity on the held-out family; kNN
  label-transfer (lookalike/segmentation usability) — separate from sufficiency.
- **U5. Close the gap with the architecture** — the shape×measure decomposition (contractive
  shape memory + content-addressed additive measure memory; `AGENTS.md` DEC-106 root cause),
  driven by the future-prediction objective; re-measure `U` each step.
- **U6. Cross-consumer validation** — train plugins on `E` for held-out, *never-used* tasks and
  confirm parity (the empirical "unknown use case" test).

---

# The roadmap

Sequencing matters: **Phase 0 first** — several numbers we currently cite don't
exist or contradict their own receipts, and two runs already destroyed their own
evidence. Nothing downstream is trustworthy until that's fixed.

## Phase 0 — Make the evidence trustworthy (mostly S; do first)

1. **Doc/receipt currency sweep (~21 stale claims).** Big ones: suite count
   appears as 239/226/256/225/239 in STATUS alone → 216; stream is
   **25k/26.8M**, not "30k/11.3M"; holdout is **5%**, STATUS says 10% ×4;
   "acceptance 25/25 (Arrow, margin, email, arms)" — none of those four are
   among the 25 checks; "generator still per-event Python" contradicted by the
   same file; the CONV ladder receipt `0.012/0.0305/0.0603/0.1008` does not
   reproduce (measured **0.221/0.217/0.194/0.157 per click, decreasing**);
   "37/37 spot-check" has no script; `uv.lock` "used by CI" — CI uses pip.
2. **Reconcile the white_queen battery story.** Documented "7/25" is battery
   **v1** (`red_king/artifacts/ab_witness.json`, actual split **1 missed +
   6 false**, not 3+4) and an **undocumented battery v2 exists:
   `ab_world_engine.json` = 15/20 errors** (4 missed + 11 false), whose
   `WITH_BOTH` verdict is `ship: true`. Decide the red_king question (see open
   decisions #2) and put both batteries on the roadmap where they're visible.
3. **Receipt identity — stop destroying evidence.**
   - Sample-A ladder rung dirs are **not as-of-scoped** (`cfm_ladder/r250/…`):
     each month overwrites the last; the receipt for what the Nov cycle
     actually trained on is already gone.
   - Heads/artifacts overwrite monthly under a constant tag `v1.0.0r1` —
     the 2025-11-01 head no longer exists; no seed, no git SHA in payloads.
     Bump `revision` on behavior change; as-of-stamp head files; record seed.
   - Reject `encoder_version = None` at train time (today a mixed-version
     products table trains a **null-pinned head that passes the gate**).
4. **Rehearsal honesty (4 one-liners).**
   - Exit code reflects gate verdicts (`rehearsal.py` returns 0 unconditionally
     — FAIL is invisible to any runner; violates RUNBOOK rule 6).
   - Pass the ladder's `chosen_n_train` into `run_target(max_train=…)` —
     today the ladder runs and its answer is recorded but never used.
   - Stamp receipts with git SHA; derive midweek instead of the
     `2025-11-05` literal (cycle 2 silently loses an inference day).
   - Wire `cfm_validation.validate()` (seeded — its `order` probe currently
     uses unseeded `torch.randint`) into the loop; add a held-out CE row.
5. **Battery receipts.** `sufficiency_battery` / `layer_b_proof` print and
   discard; no JSON anywhere, targets with <100 rows vanish from the
   denominator silently, empty portfolio → NaN pass. Persist full rows +
   coverage + skipped list.
6. **Dependencies: one truth.** `uv.lock` is stale (pins the removed
   `lancedb`, misses `pytest-cov`); `looking_glass/pyproject.toml` still
   declares `lancedb` + dead `baseline`/`kernels` extras + a `testpaths` for
   the emptied `looking_glass/tests`; three dependency roots exist. Regenerate
   the lock, add `uv sync --locked` to CI (or delete the lock and say "pip"),
   delete the nested declarations.
7. **Coverage honesty.** Claimed 85%/5pt headroom is inflated by test files:
   source-only tribunal coverage is **80.7% (0.7pt headroom)**. Measure
   excluding tests, set the floor accordingly, decide whether looking_glass
   core joins the floor.

## Phase 1 — Close the loop on what ships (S→M)

1. **Serving calibration gate + realized-label closer.** Nothing reads what
   inference writes: Dec receipts show `mean_score 0.516 vs base_rate 0.435`
   (+8.1pt, >25σ binomial SE at n=25k) while the training gate passes. Add:
   a same-day derived-tolerance check (`|mean_score − base_rate|` vs binomial
   SE + held-out calibration gap), `head_id` + dedup on `plugin_scores`
   (200k duplicate key-groups today, up to 4 indistinguishable receipt rows
   per as_of), and a post-window closer joining persisted scores to `orders` —
   **4 of the 5 rehearsal inference days already have closed labels locally**.
2. **white_queen quick levers** (projected from receipts, needs re-run):
   - witnesses ≥ 2 (`judge.py:362`): projects v1 7→4 errors, v2 15→6, no
     correct deploys lost.
   - certificate must clear the adaptive **bar**, not the raw behavior mean
     (`certificate.py:159`) — 4/6 v1 false deploys are behavior-clones whose
     truth *equals* the bar.
   - explicit behavior-clone / no-material-gain rejection (HOLD).
   Then re-run both batteries (≈25min + 51min) and **persist cells into
   `tribunal/bench/results/` as a pytest gate** (`false_deploys=0`,
   `recall=1.0` contract exists but has never been scored; `CONFORMAL_K=1.0`
   is raw while `calibrate_k()` exists).
3. **Gate uncertainty.** `_binary_rows` thresholds are literals
   (`lift>1.0`, `gap≤0.05`, winner = max AUC on the *same* single split
   `seed=0`). Add fold/paired SE to AUC/lift/calibration rows, Brier (already
   computed) + ECE rows, paired-noise winner selection (machinery already
   exists in `ladder_sample_a._paired_auc_se`), and derived tolerances.
4. **The donor claim's missing receipt.** Run the E-vs-R-vs-E+R ablation
   *under the production bake-off head on `donor_embeddings`* with paired
   ΔAUC (lift the battery's RFM builder into `load_dataset`). Today the
   universal-donor verdict rests on Spearman-of-Ridge over a table production
   never reads — and `reorder_30` (≈ this plugin's label) is the known
   "raw-ish" case.
5. **Scores/head schema** (enables 1 and Phase 3): `head_id` column,
   upsert-not-insert, as-of-stamped head archive.

## Phase 2 — Capability gaps (the open DoD/next-tasks) (M)

1. **white_queen hardening round 2.**
   - Overlap as a first-class veto: propagate propensity quantiles /
     `clipped_frac` into `gate.adjudicate` (4/6 v1 false deploys occur under
     ε∈{0.15,0.05}; diagnostics exist, gate on them).
   - Recall side: v2's 4 missed deploys have w=0 because white_queen's internal
     MB witness is a one-step MLP blind to partial observability (its MB value
     −51.7 vs truth 16.7) — replace/down-weight it, or admit the
     currently-dead `decide.decide` independent-pair rule as an alternate
     deploy route (doctrine #13 says wire it or delete it).
2. **red_queen: make decisions personalized and uncertain.**
   - Wire per-customer HTE (`red_king/hte_model.py` exists; **10,716/25k
     customers are all-arm identified and unused**) with hierarchical
     shrinkage; population fallback elsewhere — STATUS next-task #2.
   - `value_sd` is **all zeros** today (`risk_z` is inert; violates doctrine
     #8): bootstrap/ensemble SD so the fail-safe means something.
   - `_best_joint` picks arms from the **raw** mean while a deconfounded IPW
     version exists next door — switch or refuse on disagreement.
   - Vectorize `decision_log_weekly` (row-wise loop over 762k rows), add its
     missing test, and produce an **independent IPS lift** for the certified
     schedule (today the lift *is* white_queen's certificate).
   - Per-segment incrementality report (STATUS next-task #4).
3. **caterpillar.** First: grounded artifact Q&A over receipts we already
   have (certification HOLD/DEPLOY, plans, incrementality, certificates) —
   template router, no model, receipted. Second: NL surface on top with the
   provenance dict as citations and a refuse path (open decision #5).
4. **Generator 50k → ladder → battery re-run** (STATUS next-task #5) —
   sequenced *after* Phase 3's perf + acceptance fixes so the run is pinned.

## Phase 3 — Data-layer truth & realism (S→M)

1. **Generator perf (cheap first):** move `CREATE INDEX` out of the DDL to
   post-load (`business_tables.py:267-270` + duplicate at
   `event_stream.py:49-51`) — this is the ~450s lever; delete dead work
   (redundant DELETEs, the LanceDB flag); add phase wall-time receipts.
   Note: the "170s Arrow tail" is likely DuckDB's close-time checkpoint
   (measured 24s IPC write on 25k) — measure before optimizing.
2. **Acceptance +6 checks, data-derived thresholds:** window bound
   (**measured leak: 15,727 events after `_REFERENCE_NOW`, max
   2026-03-04**), arms/propensity/holdout + known-effect pin, contact event
   vocabulary (89% of rows unchecked), Arrow round-trip, channel-mix bounds
   (push = 32.3% of events), propensity-overlap floor (min 0.00026).
3. **Regenerate the canonical artifact** — the standing 25k stream predates
   the vectorized generator (log shows 5 materialize stages vs code's 8) and
   violates the window bound; ship it with a build receipt and decide its
   size (open decision #4).
4. **Known-effect: pin it or fix it** (open decision #1): the documented
   increasing CONV ladder does not exist in code and does not reproduce on
   the artifact (measured decreasing). Either reparameterize `p_conv` to the
   intended ladder or write the regression test against the truth you want.
5. **Realism (M each, gated by the battery):** weekday/hour seasonality for
   sends (flat today), campaign bursts tied to promo windows, churn/attrition
   (lifecycle is frozen at signup), drift/regime scenarios for battery
   robustness, channel-mix rebalance, propensity temperature for overlap.

## Phase 4 — Scale & the fork's big levers (M→L; only after Phases 0–1)

1. **Land the fork's joint token** (their reported biggest lever) — as an
   off-by-default `token_mode` validated by the sufficiency battery before it
   can be considered done; then the rest of their off-by-default set
   (`state_mode`, EMA `weight_average`, `sf_mode=event_types` — note our SF
   objective still names the purchase event, `cfm_training.py` `phi`, which
   the agnosticism rule objects to — `attr_dim`, history objective).
2. **Replay-exact state semantics + lazy/chunked EventSource** from the
   fork (our only landed part is the as_of SQL pushdown; a 206k-customer
   Instacart full load is 18.4GB RSS — the OOM they hit at 1M is real).
3. **Storage contract**: whole-stream feather (7.5GB/25k) → parquet
   partitions when 50k+ lands; requires touching every reader.
4. **CI**: split slow tier, an acceptance-vs-artifact job (data cached),
   pinned linters, lock enforcement (from P0-6).

## Open decisions

1. **Known-effect ladder**: pin the measured (decreasing) truth, or
   reparameterize the generator to the documented increasing ladder?
2. **Battery v2**: `WITH_BOTH → ship:true` — does that reopen red_king for
   the decision path? (Recommendation: no — REMOVE stands on the same-
   estimand A/B; v2's 15/20 feeds white_queen hardening instead.)
3. **One dependency root** (uv+lock vs pip, which of the 3 pyprojects) and
   coverage scope/floor.
4. **Canonical stream size** (keep 25k vs generate 50k) and when to
   regenerate (leak fix alone justifies it).
5. **caterpillar NL**: add an LLM dependency, or stay template-only?

---

## Appendix — 100M customers (analysis closed; not planned)

Kept as reference (`STATUS.md` "Scale path"): stream ~20 TB (107 B/row),
state+embeddings ~210 GB fp32, encoder/plugin compute **unchanged**
(sample-capped); what breaks first = generation (21 days single-node → shard),
then the daily job (4.5h/day 1-GPU → shard the state store), then the 210 GB
daily embeddings rewrite. All shardable; not needed at current scale.

---

## v8.0 — RL World Model Extensions (conceptual; not implemented)

Deferred roadmap items 9–11 (Capabilities 1–8 are the v7.0 baseline). Do not
implement until the v7.0 plan lands (fix `uncertainty`, confirm the
`event_residual` transition fix). All three must obey Guardrails A–E.

### Capability 9 — Intervention & Action Conditioning (Causal RL)
Separate observational user behavior from interventional system actions by
conditioning the SSM transition on an action embedding (gate the input
projection `B`, or route user vs intervention events through a MoE). Probe:
`intervention_effect` (Δ conversion given an action, controlling for pre-action
state).
- **Feasibility (per-sample norm + contractive bounds):** compatible *if* the
  conditioning is a **bounded multiplicative gate** (e.g. `σ(W_a·a)` in `[0,1]`
  applied to `B·x`, or an additive action embedding like the existing `w_co`
  exogenous-covariate channel) — the decay stays contractive (`ρ<1`) because
  gating `B` scales the *drive*, not the transition operator, and the readout
  stays per-sample RMS-normalized. Two hard caveats: (a) **MoE routing is a
  known-failed pattern here** — context-gated softmax/top-k mixtures were
  rejected earlier for collapsing channel diversity (M2/M5); (b) causal
  identification needs **randomized/interventional** data, which ecommerce_2019
  lacks (it is purely observational). This capability is only meaningful on
  streams with logged arms (rabbit_hole's randomized email arms).

### Capability 10 — Hierarchical Intent Abstraction (Macro-States)
Pool micro-states into a slowly-updating macro-state at detected session/regime
boundaries; dual readout `z_micro` (per event) + `Z_macro` (per boundary). Probe:
`macro_order` (session outcome from `Z_macro` alone).
- **Feasibility:** a macro-state that is a **convex/EMA pool** of `z_micro` is a
  bounded, contractive accumulator (`Z_macro ← (1−g)·Z_macro + g·pool`, `g∈[0,1]`)
  and is fully per-sample — compatible. The risk is the **trigger**: using the
  *predicted* boundary (Capability 7, AUROC only 0.669 on ecommerce) as a
  structural pooling gate is (i) error-propagating and (ii) **discrete/non-
  differentiable** if hard-gated. It must be a **soft, bounded gate** (continuous
  in the boundary logit) to keep gradients; the weak boundary signal caps how
  much `Z_macro` can add on this stream.

### Capability 11 — Nuisance Invariance (Robustness)
Structural contrastive alignment: pass the same sequence through two augmented
tokenizers (drop <100 ms dwell; jitter Δt ±10%) and align the two `z_t` with a
plain unweighted InfoNCE.
- **Feasibility:** the most compatible of the three. The augmentations are
  **input-side** (not loss hacks), the contrastive term is a plain objective
  (the model already has a `contrast` instance-discrimination objective), and
  because both views' `z_t` are **per-sample RMS-normalized** the alignment is
  scale-free; the contractive SSM is untouched. One honest caveat: InfoNCE's
  **in-batch negatives are batch-coupled** during training (acceptable — the
  frozen readout is per-sample, so this does not fake rank at serve time), and
  the augmentation must stay *structural* (fixed, data-derived thresholds) rather
  than tuned, or it becomes a soft hyperparameter.

---

## v8 — scorecard hardening + Layer-A laboratory + capability frontier (2026-10-13)

**Done:** Layer-B report card + Layer-C head battery hardened (held-out customers, strong
baselines, bootstrap 95% CI, far-exceed tier) and made stream-agnostic; measurement bugs fixed
(oracle baseline, mask leak, scaling, multi-output null, channel-MI rank-vs-entanglement,
conversion event, `sf` standardization); Layer-A laboratory (32-use-case ontology + parameterized
data spec + complexity ladder + oracle/null/adversarial controls + coverage gate); the A<->B loop
proven on an anchor (both cards GREEN, 4/5 far-exceed); §20.2 value encoders (entity_id +
attributes + model-free product co-occurrence embeddings); new objectives (recency, foresight) +
magnitude-preserving readout.

**Scoreboard (B | B+C):** rabbit_hole GREEN|GREEN · anchor GREEN|GREEN · instacart GREEN|NOT ·
ecommerce NOT|GREEN(vacuous).

**Frontier (the next real work):** the encoder's unique signal over a strong RFM+GBM baseline is
~+0.04 on RFM-reconstructable targets (instacart LTV/order-count), vs a rich model-free baseline's
+0.08. The signal is CARRIED (so NO_SIGNAL would be dishonest); the encoder under-exposes
recency/velocity at the readout. Every readout/data lever tried (rank, attributes, half-life,
recency, foresight, magnitude-preserving readout, SF-product, 8 seeds) lands ~+0.04. Closing it
needs a genuinely new capability (e.g., a dedicated recency-window readout fed by the raw event
clock), not another tweak.
