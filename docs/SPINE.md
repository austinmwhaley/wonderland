# SPINE — the single first-principles structure (living document)

> The one steering document for the foundation-model re-think (started 2026-10-27; merges
> and supersedes `docs/FIRST_PRINCIPLES.md`, whose argument log is folded into §10).
> Nothing here is implemented; this is the structure we are building by conversation.
> The operator reads the chat, not this file — this file is the record the chat produces.
> Vocabulary adopted: **the brain** = the learned views; **the ledger** = the `measure`
> view (exact computed tallies, built outside the network).

## 0. The goal, one sentence

Input any event stream → output a **frozen** foundation model (one per stream) whose state
supports **supervised**, **unsupervised**, **RL**, and **simulator** heads using simple
consumers.

## 1. The four consumers come first (tracing rule — operator ruling 2026-10-27)

Everything in the system must trace to the consumer set as a whole ("no feature without a
trace"): a component exists only if it can name a consumer need it serves or a gate it
satisfies. Consumers map to project layers: supervised/unsupervised/RL heads live in
`plugins` (RL certified by `white_queen`); the simulator is the `red_king` role
(counterfactual effect recovery / decisions), also certified via `white_queen` OPE.

| Consumer | The question it asks | What it needs from the frozen state |
|---|---|---|
| **Supervised** (predict churn, value, orders…) | "What will happen to this customer?" | The state must **contain** everything predictive about the customer, and a plain line must be able to **read** it |
| **Unsupervised** (segments, lookalikes, anomalies) | "Who is like this customer?" | **Distance = behavioral similarity**: neighbors genuinely share futures |
| **RL** (email frequency, offers…) | "What should we do?" | Customer state **just before** an action + response structure, where the logged data tested actions |
| **Simulator** *(added 2026-10-27, DEC-384 — the red_king role)* | "What would happen IF we did X to this cohort?" | An **action-conditioned** future model in the state: outcomes given actions, with uncertainty; causal claims only where the stream randomized |

## 2. The constitution (ACCEPTED 2026-10-27)

**Axioms are gates** — violate one and the run is void. **Maxims are ranked** — between two
designs that pass the gates, the higher maxim wins.

### Axioms (gates)
| # | Axiom | The test |
|---|---|---|
| A1 | **Robust** | deterministic; divergence = hard failure (never rescued); reject or widen uncertainty, never clamp; calibrated uncertainty everywhere |
| A2 | **Honest** | no downstream labels in encoder training; never trained on the graded metric; fair comparators with paired CIs; reproducible receipts at a recorded commit |
| A3 | **Strict-frozen** *(LOCKED)* | after freeze the encoder is immutable; consumers ≤ {linear, kNN, small MLP}. The encoder must pre-adapt to consumers it will never meet — accessibility is an encoder requirement, not a consumer problem |

### Maxims (ranked core)
| Rank | Maxim | The test |
|---|---|---|
| M1 | **Universal / generic** | zero domain knowledge in code; any schema resolves from the input; test = renamed-schema + new-field training, zero code edits |
| M2 | **Adaptive / dynamic** | every resolvable quantity measured or learned at fit time; surviving literals are receipted fallbacks only |
| M3 | **Lawful** | correct algebra per quantity; exact symmetries (field order, time shift, sequence-merge); laws rank above efficiency because the laws are the *mechanism* of speed |
| M4 | **Efficient / fast** | linear time, constant state, O(1) incremental update, vectorized; runtime receipts |

### Standing tie-breaker — OVERMATCH (operator doctrine, DEC-385)
Between "build the correct thing structurally **now**" and "defer until a receipt proves
the need," choose **now** where costs are comparable. Correctness is not a cost to
postpone — and under strict-frozen, a deferred interface error costs a full refit, so
"wait and see" is the *expensive* option at any interface the encoder controls.

### The parameter trichotomy
Every parameter lives in exactly one column; wrong column = bug.
| Column | Meaning | Examples |
|---|---|---|
| **Contract** (fixed on purpose) | rules of the game | laws, evaluation oracles, label firewall, strict-frozen, per-stream rule, negative controls |
| **Derived** (measured at fit time) | read from the stream | field types, vocab caps, transforms, timescales τ, horizons, mask rates, capacity, loss weights |
| **Learned** (fitted) | the model's own knowledge | embeddings, gates, decays, prediction heads |

## 3. THE JUDGE — PATH B > PATH A (operator definition of progress, 2026-10-27)

| Path | Data shape | Trains | Meaning in the judge |
|---|---|---|---|
| **PATH A** | **wide table — one row per customer** | one set of models (table + GBM baseline) | the incumbent: everything must beat this |
| **PATH B** | **event stream — one row per customer-event** | the encoder (our build) | progress = PATH B beats PATH A on the consumer battery, paired CIs |

Everything below serves this judge. The stage ladder (§7) is the diagnostic layer beneath
it: when B fails to beat A, the ladder names which stage is the bottleneck.

**Corollary (derived 2026-10-27, from the ledger discussion):** any score contributed by
the ledger cannot count as a B-*win*, because the ledger is mechanically what PATH A's
table already is (receipt: ledger category purity 0.261 vs table 0.265 — same object).
The ledger can only prevent B from *losing*; **the brain earns the win, the ledger
prevents the loss.** Hence the two-role rule (see §5): certify the foundation on the
brain alone; serve with brain ⊕ ledger.

## 4. The derivation spine (S1 → S6), each stage forced by a gap

```
 RAW events (time, customer, typed attributes)          — the only given
   ▼ S1 INTERPRET  read the stream's ontology off the stream            [M1 · M2]
 Typed fields + measured statistics → a resolution receipt
   ▼ S2 SYMBOLIZE  event → vector, correct algebra per field            [M3 · M2]
 Event vectors
   ▼ S3 ACCUMULATE the sufficient-statistic recurrence of the process   [M3 · M4]
   ▼                class: decay between events, jumps at events,
   ▼                two pathways (shape·contractive | mass·additive), learned timescales
 Customer state h(customer, t) — bounded, fixed-size, incremental
   ▼ S4 EXPOSE     named views + decodability contract ρ + uncertainty  [A3 ⇒ accessibility]
 FROZEN STATE (views, ρ)
   ⟲ S5 PRESSURE   self-supervised objectives — the only force that fills the state
   ◆ S6 FREEZE     hash + provenance + per-view controls + card
```

| Stage | Gap that forces it | Failure if faked |
|---|---|---|
| **S1 Interpret** | No downstream operation is type-agnostic; the *data* must declare what each field is (M1 forbids being told) | new schema breaks it; ids treated as numbers |
| **S2 Symbolize** | S3 needs one continuous object per event; mapping must preserve each field's algebra; **time is not a feature** (timestamp belongs to S3; binning banned) | clamped unseen values; hand-written templates; cadence literals |
| **S3 Accumulate** | Fixed-size state ≈ full history + constant memory + O(1) updates ⇒ only a recurrence can do it; *which* recurrence is derived (fade + jump = sufficient statistic of the event-process family). Architecture = ansatz about the process class; S5 tests it | hand-set τ/width; divergence rescued; mass decayed (level lost) |
| **S4 Expose** | A3: sufficiency survives scrambles that kill simple heads, and no consumer may adapt ⇒ the encoder pre-adapts to consumers it never meets; named views; abstention channel | views picked on holdout; one blended vector; "present but unreadable" sold as sufficiency |
| **S5 Pressure** | Architecture is capacity, not content — nothing enters the state unless a gradient demands it; only self-supervised objectives admissible (A2); accessibility/distance failures feed back here as objective-mix changes, never clamps | hand-weighted losses; no distance pressure (today's geometry gap); training on graded metric |
| **S6 Freeze** | Without hash + provenance + per-view controls + regenerable card, every claim is narrative | card newer than receipts; stale claims |

### What the encoder is NOT
- Not a consumer of labels/downstream tasks (firewall; PATH A's table is a
  **teacher/baseline**, never resident inside the brain — see ledger guardrail G4).
- Not cross-stream (1 stream → 1 model; unseen stream = refuse).
- Not batch-coupled at serving (per-sample contract).
- Not allowed to clamp (out-of-support → abstain/widen).

## 5. THE BUCKETS — the complete view table (real names, per `looking_glass/state_layout.py`)

**LEDGER DECISION (2026-10-27, DEC-383): Option 3 (companion) ADOPTED** with guardrails
G1–G5 mandatory, and the two-number card (brain-vs-A = the judge; brain+ledger-vs-A =
delivery). **Option 4 preserved**: serving on the brain's learned mass channel remains a
live target, and ledger aggregates remain admissible S5 auxiliary targets (self-supervised
pressure from the visible past; G4 still bans the ledger as a brain *input*). Serving
ledger retires only when the brain earns those rows outright.

The frozen artifact exposes **named views** (buckets) so simple consumers select what they
need. Real names below; "brain vs ledger" = learned-by-the-network vs computed-outside-it.

| Real name | Brain/ledger | Dims (6k artifact) | What it represents | Primary consumer use |
|---|---|---|---|---|
| `shape` | brain | 128 | the behavioral profile — "who is this customer like" (the compact learned state `E`) | all distance/neighbor tasks (unsupervised), supervised |
| `learned_state` | brain | 1152 | the raw per-timescale memory contents + learned mass (what the recurrence held) | fine-grained supervised; carries channel identity (0.72 purity) |
| `forecast` | brain | 835 | learned past/future summaries (self-supervised summary heads) | prediction heads (best churn view 0.925) |
| `uncertainty` | brain | 23 | missingness / support / gap — "what did I actually see" | anomaly detection; careful heads abstaining |
| `heads` | brain | 127 | head outputs: identity / next-event / relational predictions | identity probes, next-event consumers |
| `measure` | **LEDGER** | data-derived (~734, coverage 0.99) | exact lawful tallies computed **outside the network** (counts, totals, shares per category, first/last…) | category identity, anomaly — the only place these currently work |
| `response` | brain | **approved, not built** (DEC-384) | action-conditioned outcome predictions (given the logged action + pre-action state) — conditional predictions with uncertainty, never causal claims; causal reading only where the stream randomized | RL + simulator (red_king) |
| `full` | brain bundle | 2265 | **permanently defined** concatenation of shape + learned_state + forecast + uncertainty + heads (order fixed; **`measure` excluded by definition**; `response` joins only if the audit earns it) | strongest supervised view |

Facts pinned by code (single source of truth):
- `FULL_ORDER = (shape, learned_state, forecast, uncertainty, heads)` — `full` is the brain
  bundle; the ledger is NOT inside it ("full + measure" is a separately-named arm).
- `pre-action` is **not a bucket** — it is a **cut**: any bucket read at a timestamp
  strictly before an action. Buckets are what; cuts are when. (RL consumes `shape`/
  `learned_state` as pre-action cuts.)
- Every card row must name the bucket it used (the layout exists for that).
- **Yes, that list (6 content + 1 bundle) is ALL the artifact publishes today.** Each
  bucket exists because a component of the machine emits it:

| Bucket | Emitted by (origin in the machine) |
|---|---|
| `shape` | the core recurrence's output `E` |
| `learned_state` | the per-timescale memory cells + learned mass (internal contents, exposed) |
| `forecast` | the past/future summary heads from the training battery (kept at serve) |
| `uncertainty` | the missingness/gap heads |
| `heads` | the identity / next-event / relational head outputs |
| `measure` | the ledger builder — accumulation **outside the network** (`measure_state.py`), vocabulary resolved once at fit time and persisted (no re-scan at serve) |
| `full` | nobody — it is the permanently-fixed concatenation of the five brain buckets (a bundle, not new content) |

### Bucket audit — is the set derived or accidental? (open, queue item 3)
Derived-from-consumers checklist vs the current set:

| Content class a consumer provably needs | Bucket serving it | Verdict |
|---|---|---|
| predictive shape ("who") | `shape` (+`learned_state`, `heads`) | ✓ |
| magnitude ("how much") | `learned_state` mass, `forecast`, ledger | ✓ (brain side partially — corr² 0.568) |
| support ("what did I see") | `uncertainty` | ✓ |
| exact counts | ledger | ✓ (as companion, pending decision) |
| **response ("what if we do X")** | **DECIDED: `response` bucket approved (DEC-384)** — outlet + training pressure; conditional predictions + uncertainty; causal claims only under logged randomization | ✅ closed by decision; build pending |
| redundant? | `learned_state` (internals dump, but empirically earns its keep: channel purity 0.72); `heads` (network's own guesses — consumer bucket or training scaffolding?) | audit |

### The ledger question — OPTIONS (OPEN; operator deciding)

The worry (operator, 2026-10-27): "if we add the table back we're right back to PATH A."
Correct and empirically true — the ledger ≈ the table. Options:

| # | Option | In one line | For | Against |
|---|---|---|---|---|
| 1 | **Brain only** | no ledger in the artifact; consumers needing exact counts compute them from events | purest judge; smallest artifact | ledger-served tasks (category identity, anomaly) get no answer from the foundation; breaks "never materially worse than table"; each consumer rebuilds its own PATH A |
| 2 | **Ledger blended in** (status-quo direction) | ledger is just another bucket inside E, summed into headline scores | one-stop; parity guaranteed | contaminates the judge — ledger-backed "wins" are table-vs-table; slippery slope to hand-picked columns |
| 3 | **Companion (recommended)** | the **foundation is certified on the brain alone**; the artifact **ships brain ⊕ ledger as named buckets side by side**; ledger can never count toward a B-win | judge stays honest (B wins only via brain); ledger guarantees never-shipped-worse-than-table; consumers still single-stop | more bookkeeping; attribution discipline must be enforced by the card, not by trust |
| 4 | **Learned ledger** | no ledger; force the brain to learn exact totals via training pressure (already half-present: mass-reconstruction corr² 0.568 PASS) | purest artifact; brain carries magnitude content | brain burns capacity on arithmetic a CPU does exactly (never exact, slower, bigger) — violates minimality/efficiency |

**Guardrails — MUST if the ledger ships at all (operator ruling 2026-10-27: mandatory):**
| # | Guardrail | In plain English |
|---|---|---|
| G1 | **Mechanical only** | the ledger is a fixed recipe (count, sum, last, first, distinct-count, share-per-category) applied to **every field S1 discovered** — no human ever hand-picks columns or windows; hand-picked = PATH A smuggled back in |
| G2 | **History-only** | computed from events visible before the as-of cut; no peeking forward |
| G3 | **Target-blind** | never contains the definition of the quantity being predicted |
| G4 | **Never feeds the brain's training** | the brain trains on events only; if the brain could read the ledger, we could never tell learned from copied |
| G5 | **Attribution always** | every reported row states brain / ledger / both, with learned-only + ledger-only + destroyed controls |

Recommended: **Option 3 (companion) + all five guardrails.** Slogan: *the brain earns the
win; the ledger prevents the loss.*

### How Option 3 works at inference (verified against `measure_state.py`, 2026-10-27)

The ledger is **not a stored wide table** — nothing to keep, nothing to join. It is an
accumulation channel, like an odometer: the recipe (which fields/values to tally — the
"vocabulary") is resolved once at fit time and rides inside the frozen artifact; each
customer's ledger numbers are produced by ticking counters as their events stream through
(+1, += value), in one vectorized columnar pass, and extend per new event thereafter.
Measured (6k report card): ledger pass 0.19 s for 600 customers vs warm brain forward
0.316 s — same order of magnitude, simple arithmetic, no network. Storage: +~734 floats per
customer next to the ~2265 the brain already keeps. If option 4 replaced it, the same
arithmetic would be hidden inside the network — paid for in **capacity** (the scarce
resource the U metric holds us to), approximate instead of exact (violates the exact-additive
mass law, receipt C3 = 0), with a bigger slower forward. **Counting is the cheapest thing the
engine does; teaching a neural net to count is the expensive move.**

## 6. The consumer-backward trace

| Layer | Supervised | Unsupervised | RL | Simulator (red_king) |
|---|---|---|---|---|
| Needed property | predictive sufficiency | behavioral metric (distance = shared futures) | decision-usefulness (v1 bar) | action-conditioned future + uncertainty |
| Buckets that serve it (real names) | `full`, `forecast`, `learned_state`, `shape` | `shape` geometry; `uncertainty` for anomaly; ledger for category identity | pre-action cuts of `shape`/`learned_state`; `response` (approved) | `response` (approved) + pre-action cuts |
| Stage that creates it | S3 + S5 + S4 | S3 + **S5 distance pressure (missing today)** + S4 | S3 + S5 response pressure + pre-action cut + randomized data (Layer A) | S3 + S5 action-conditioned pressure + S4 + randomized data (Layer A) |
| Progress gate (receipt) | beats PATH A with paired CI; ≥50% of ceiling gap | distance↔truth-similarity beats table/random; kNN transfer > table | known-effect + IPS beats table & random with CI | known-effect recovery of built-in effects; counterfactual agreement vs randomized arms with CI |

Single-object record: the three properties are projections of ONE object — the stream's
future-predictive law (action-conditioned where identified). Supervised = determine it;
unsupervised = distance equals discrepancy between laws; RL = use it under actions.

## 7. Progress mechanics (the diagnostic ladder beneath the judge)

Judge = PATH B > PATH A (§3). When B doesn't beat A, this ladder names the bottleneck
stage; each state moves only on a receipt. "Mastered" = all six working.

| Stage | Working-state proof | Not-working looks like |
|---|---|---|
| S1 | renamed-schema + new-field pass with full resolution receipt, zero code edits | any field typed by hand; any cadence literal |
| S2 | unseen-value path exercised; transform receipts; permutation invariance ≤ tol | clamped unseen values; templates in code |
| S3 | merge/equivariance/mass-additivity exact on data; τ receipted from spectra; stable from step 0 | divergence rescued; τ/d hand-set |
| S4 | per-view accessibility parity + per-view negative controls | views selected on holdout; no decodability contract |
| S5 | objective battery beats naive (or excluded+reported); distance-pressure row exists | hand-weighted losses; no geometry pressure |
| S6 | one command regenerates the card at a recorded commit + hashes | card text older than last behavior change |

## 8. Decision protocol (how any change is processed)

1. Which stage (S1–S6) does it touch? **Which consumer(s) does it trace to? Does it move
   the judge (B > A)?** (no trace = no)
2. Which maxim does it serve? Does it violate an axiom? (axiom violation = stop)
3. Which column is its parameter in (contract / derived / learned)?
4. What receipt moves that stage's state? What negative control catches the failure?
5. Record in `DECISIONS.md`; regenerate card; accept or revert.

## 9. Literature anchors (compressed; full citations in git history of FIRST_PRINCIPLES.md)

- Event-stream SSL & foundation models: JAIR 2025 unified-SSL survey (no adopted universal
  ES foundation model exists); MLEM 2024 (no single objective dominates; generative wins
  timing; aligned mixtures most universal); ESGPT; PyTorch-Lifestream IJCAI 2025;
  HT-Transformer; RecSys Challenge 2025 (fixed-vector event autoencoder + normalized view
  concatenation); LLM4ES CIKM 2025; PinnerFormer KDD 2022 (frozen user embeddings at scale).
- Sufficiency & prediction: informative statistics (NeurIPS 1999); predictive information
  (Bialek 2001; Ver Steeg & Galstyan 2015); DAPC; CPIC; slow-feature principle
  (Wiskott & Sejnowski 2002; Creutzig & Sprekeler 2008).
- Control-relevance: state abstraction (Li–Walsh–Littman 2006; Du et al. 2019);
  bisimulation metrics (Ferns–Precup–Pan; Castro et al. 2020; robust BSM NeurIPS 2021;
  RAP NeurIPS 2022; DBC 2019); ATC decoupled representations (Stooke 2020);
  action-sufficient reps (ICML 2022); predictive ≠ control sufficiency (Latent State
  Design under Sufficiency Constraints); frozen-encoder limits (response-aware planning,
  Shang 2026).
- Accessibility/probing: Hewitt & Manning 2019; Vo & Rogers 2020; Probing with Noise 2022
  (norm = separate information container); Present but Not Remembered 2026.

## 10. Argument log (from the merged FIRST_PRINCIPLES discussion)

1. **Freezing is the defining commitment** — all learning moves from consumer to encoder;
   presence ≠ decodability. The object is a pair: content + exposure contract.
2. **Three consumers, three sufficiencies, one generator** (the future-predictive law).
   Decision sufficiency for a rich reward family subsumes prediction; geometry = isometry
   to the metric the law induces.
3. **No free lunch** — "universal" is relative to (a) the stream's real predictive content
   (Bayes floor), (b) capacity (right size ≈ predictive-information rate; measured by a
   retention-vs-d curve, never tuned), (c) a minimax over an adversarial held-out family.
   Task count is decoupled from information content. "1 stream → 1 model" is the easier,
   defensible contract the outside literature keeps failing at by chasing cross-domain.
4. **Freezing strictly costs something** — frozen encoders can provably miss what
   fine-tuning recovers; accessibility must be designed + certified, not hoped for.

## 11. Conversation log

- 2026-10-27 s1 — framing (three sufficiencies; impossibility; U-metric continuity).
- 2026-10-27 s2 — operator: Q3 LOCKED (strict frozen), Q5 decision-useful; wants structured
  raw→frozen derivation + consumer-backward tracing + simple language; SPINE v1.
- 2026-10-27 s3 — operator: single .md; constitution ACCEPTED; feature-by-feature begins.
- 2026-10-27 s4 — operator: **judge defined = PATH B (event stream) > PATH A (wide table)**;
  full bucket table pinned from `state_layout.py` (real names; `full` excludes `measure`;
  pre-action = a cut, not a bucket); ledger guardrails G1–G5 made MANDATORY; ledger
  OPTIONS 1–4 tabled with companion recommendation (Option 3) — operator deciding.
  Vocabulary adopted: brain / ledger.
- 2026-10-27 s5 — operator leans Option 3 but asks the inference-time cost question
  (fearing a stored table). Verified vs code: ledger = accumulation channel (odometer),
  recipe persisted in artifact, no table kept; measured pass 0.19s/600 customers vs
  0.316s warm forward → Option 3 remains recommended; Option 4 re-pays the cost in
  capacity + inexactness. Bucket origin table + derived-set audit added: current 6+1 set
  is ALL that publishes; **`response` bucket gap** found (RL has cuts, no bucket);
  `learned_state`/`heads` flagged for the minimality audit (queue item 3).
- 2026-10-27 s6 — operator adopts recommendations: **Option 3 + G1–G5 + two-number card
  ACCEPTED (DEC-383)**; Option 4 explicitly preserved (brain absorbs counting over time;
  ledger as S5 reconstruction target, never as input). Next: queue item 3 — bucket audit,
  headline finding = missing `response` bucket.
- 2026-10-27 s7 — operator: **Option B chosen** (new `response` bucket directly, skipping
  pressure-first C); **Q6 YES — simulator declared fourth consumer** (maps to the
  `red_king` role; certified via `white_queen`). Recorded as DEC-384. Consumer set now
  four; trace table and bucket table updated (`response` approved-not-built). Next:
  queue item 4 — the shape/mass split.
- 2026-10-27 s8 — operator protocol adopted: **every feature presented as options table
  (why / why not / recommendation); operator decides.** Feature 4 (shape/mass split)
  options A/B/C/D presented; recommendation A (split as law, engine + polar interface
  rule). Awaiting ruling.
- 2026-10-27 s9 — operator: **Feature 4 = A ratified (DEC-385)** + **Overmatch doctrine**
  adopted as standing constitutional tie-breaker (build the correct thing now, not when
  needed; freeze makes deferral expensive). Feature 5 (the engine) options presented:
  A fade+tick continuous-time recurrence (recommended), B attention-over-history,
  C fixed-window binning, D event-indexed discrete recurrence, E ledger-only (PATH A).
