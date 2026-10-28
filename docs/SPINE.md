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
| M4 | **Efficient / fast** | linear time, constant state, O(1) incremental update, vectorized; runtime receipts; **GPU-first execution (DEC-389)** |

### Operational platform (DEC-389, operator directive)
Hardware: **NVIDIA GB10 (DGX Spark)** — 6,144 CUDA cores, 128 GB coherent unified memory
@ 273 GB/s, CUDA 13.0, aarch64. Directive: **anything that can be GPU-accelerated should
be.** Strategy: *move nothing, batch everything* (bandwidth-bound chip, compute ≈ RTX-5070
tier) — streams resident in unified memory, bf16 on Tensor Cores, parallel scans, GPU
heads (xgboost `device=cuda`), GPU columnar ledger/data-prep (RAPIDS via process boundary),
GPU FFT for spectral periods. Binding: CPU-vs-GPU benchmark receipt per adoption, CPU
fallback always kept, one GPU job at a time. First GPU work item when implementation opens:
rebuild the environment (torch aarch64/CUDA-13) — no importable torch exists on the box now.

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

### S5 detail — THE BATTERY (complete by derivation, DEC-387)

Coverage checklist = the six content classes. Every objective hangs on a class; a class
without a hanging objective is a gap; an objective hanging on nothing is deleted.

| Content class | Objectives pressuring it | Status |
|---|---|---|
| **who** (predictive shape) | next-event type · JEPA latent-future · order/path · entity/identity | passing |
| **how much** (mass) | aggregates/past-totals (incl. ledger-reconstruction — preserved Option 4) · numeric marks (values) | shaky (corr² 0.57) — needs more pressure |
| **when** (timing) | gap/rate · next-event time | passing |
| **what it's like** (distance) | **distance regression — NEW** (exact truth-distance on the lab; predicted-future discrepancy on real streams) | missing → approved |
| **what if** (response) | **action-conditioned outcome — NEW** · successor features | missing → approved (DEC-384) |
| **what I saw** (support) | missingness/support prediction | present (the `uncertainty` heads) |
| anti-collapse | masked recovery (any field of any past event) | passing |
| accessibility pressure | query-conditioned probe-as-objective | passing |

Discipline (binding): trace-or-cut · learned weights, held-out-SSL selection only · naive
comparator per objective or pruned.

### T-series — the internals map (between raw event and frozen state)

S1–S6 are the stages; each contains internal decisions. Status legend:
**law/receipt** = already settled by a ratified law or existing receipt · **open** = due at
the table as a T-feature (same protocol: options, why/why not, recommendation, ruling).

| # | Internal decision | Where | Status |
|---|---|---|---|
| T1 | **Token unit** — what one "tick" of the engine is | S2 | **LAW (DEC-391): one event = one tick** |
| T2 | Categorical vocab policy — caps, hashing for id-like fields, unseen-value path | S2 | **LAW (DEC-393): measured head + hashed tail + profile id-typing** |
| T3 | Numeric encoding — learned monotone transform vs fallback | S2 | **open — presenting now** |
| T4 | **Missing/unknown/zero semantics** — three-way field presence | S2 | **LAW (DEC-392): not-applicable / unknown / present, data-pruned** |

**Canonical input schema (contract column, DEC-391):** `entity_id | event_ts | event_type |
event_attributes` — entity_id hashed to identity view (never numeric); event_ts feeds
continuous-time physics; event_type conditions and is predictable; event_attributes is the
open bag S1 interprets. Schema-agnosticism applies inside the bag and to column names
(C5 renamed-schema test keeps its teeth).
| T5 | Composition law — how fields combine into one event vector | S2 | **law** (permutation-equivariant symmetric composition; C1 receipt) |
| — | Time is not a feature; periodicity from data-derived spectral peaks | S2/S3 | **law** (DEC-386) + receipt (Fourier-off hurts orders) |
| T6 | Sequence construction — chunking long histories, budgets, as-of cuts | S3 | **open — presenting now** |
| T7 | Timescale allocation — how many τ channels, from what statistic | S3 | **open — presenting now** |
| T8 | Nonlinearity placement — gating/bilinear coupling where rank needs it | S3 | **open — presenting now** |
| T9 | Capacity d | S3 | open — deferred to retention-vs-d curve (DEC-390) |
| T10 | Bucket membership — which head outputs live in which bucket | S4 | **open — presenting now** |
| T11 | Weight-learning mechanism, horizons, mask rates | S5 | **LAW (DEC-399): all derived from measurement; C shelved as fallback** |
| — | Training regime (convergence not counts, stability law, early stop) | S5 | doctrine (AGENTS.md) — needs spine receipts |
| — | Freeze semantics | S6 | **law** (A3 lock) |

### T1 — Token unit (options table, presented 2026-10-27)

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | **One event = one tick** — the engine reads each event as-is, fields encoded by their measured kind, timestamp feeds the physics | Native fidelity | Matches the engine law exactly (fade between, tick AT events); no cadence literals; irregular timing preserved; O(1) per event | Field-heavy events make one tick carry a lot (fine: composition is symmetric and learned) | ✅ recommended |
| B | **Text-ify** — render each event as a sentence, feed a text model | Translate to English first | Reuses LLM machinery; free linguistic priors | Lossy paraphrase (numbers drift when written as words); tokenizer bloat; external dependency violates minimal self-contained stack; LLM-style ES models remain single-domain per the JAIR survey | ✗ |
| C | **One day = one tick** — aggregate the day's events into a token | Daily digest | Compact; smooths noise | It is a **cadence literal** and destroys intra-day order/timing — the banned anti-pattern (DEC-386 rejected this family) | ✗ |
| D | **One field = one tick** — spread an event's fields across separate steps | Micro-tokens | Slower, finer reads | Breaks the tick-at-event physics (an event no longer arrives at a time); multiplies state updates by fields with no law to justify it | ✗ |

Ratified: **A (DEC-391)**. T2/T3/T4 proceed as their own tables.

### T4 — Missing / unknown / zero semantics (presented 2026-10-27)

The problem: a field can be (1) **not applicable** (email_subject on a purchase — structurally
absent), (2) **recorded-but-unknown** (tracking failed: value exists, we don't have it), or
(3) **genuinely zero** (discount = $0.00 — a real value). Collapsing these means "didn't
measure" and "measured zero" become the same number: confident-wrongness by construction,
and it starves the `uncertainty` bucket of content. Missingness is itself signal (tracking
failures correlate with cohorts/channels).

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | **Zero-fill everything** (typical default) | missing → 0 / sentinel | simplest | erases the distinction; engine learns "unrecorded" as "zero" — wrong patterns, no abstention, uncertainty faked (A1 violation) | ✗ |
| B | **Explicit presence semantics** | every field carries its true status — not-applicable / unknown / present(value); categorics get dedicated unknown slots, numerics an indicator beside the transformed value; **only statuses the data actually exercises get channels** (S1-derived pruning) | truth in the vector; `uncertainty` bucket finally has content; missingness becomes usable signal; cheap (bits + indicator, vectorized) | slight state growth, pruned by derivation when a status never occurs | ✅ recommended |
| C | **Mask-and-ignore** (BERT-style) | treat missingness only as a training-noise device | the masking objective is a fine *pressure* (stays in the battery) | not a semantics answer — says nothing about what missing means at serving; complementary, not an alternative | as pressure only |
| D | **Drop high-missing fields** | schema hygiene | cleaner input | throws away signal; the missing pattern was the signal; M1 (deletion decided by data anyway) | ✗ (data-derived pruning of *constant* fields stays in S1) |

Ratified: **B (DEC-392)**. Presence profile is S1 receipt content; accessibility gains the
unknown-vs-zero probe gate; ledger counts present-values.

### T2 — Categorical vocabulary policy (presented 2026-10-27)

Which values get their own learned embedding, what happens to the long tail and to
id-like fields (sku, campaign_id), and what happens when a never-seen value arrives at
serving. (T4 governs *unrecorded*; T2's unseen path governs *recorded but new* — they
compose.)

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | **Full vocab** — every observed value gets a row | everyone gets a name | simple; no collisions | parameter mass dominated by long tails (99% of values carry ~1% of events); singleton embeddings are memorized noise (overfit by construction); no principled unseen path | ✗ |
| B | **Measured head + hashed tail + id-typing + seen-before discipline** ⭐ | S1 measures each field's value-mass curve: **head** values (the measured mass cover) get dedicated embeddings; **tail** hashes into a budgeted shared region sized from measured tail cardinality (collision-rate receipt); **id-like fields** (high cardinality + heavy tail, discovered by profile not by name) route to identity-view hashed embeddings, never polluting category space; **unseen at serve** → hash-routed tail + support flag; if unseen mass exceeds the calibrated coverage bound, the stream is flagged (reject/widen, never silent) | capacity spent where event mass is; ids don't collide with categories; unseen handled without clamping or crashing (A1); every cap is a measurement (M2) — the ledger already practices 0.99 coverage | more machinery than A; needs collision + coverage receipts kept honest | ✅ |
| C | **Hash everything** (fixed tables, no measured vocab) | one big hat, name anyone in | simplest fixed memory | head values — the most important categories — eat collisions while the tail eats space; throws away the measured distribution S1 just paid to compute (M2 regression) | ✗ |
| D | **Drop high-cardinality fields** | complexity control by deletion | simple | identity-bearing signal discarded (channel/identity receipts show it matters); deletion must be data-derived (lives in S1), not a schema verdict | ✗ |

Ratified: **B (DEC-393)**. Receipts required and kept current: vocab-coverage per field,
collision budget, serve-time unseen-mass monitor wired to support.

### T3 — Numeric encoding (presented 2026-10-27)

Raw numbers are hostile learners: heavy tails ($0.01…$10,000 with most purchases under
$50), fields in incompatible units (dollars vs seconds vs counts), stream-specific scales.
Exactness is NOT at risk — exact totals live in the ledger (DEC-383) and polar separation
holds (DEC-385) — the brain's numerics are free to be *meaning positions*.

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | Raw + standardize (mean/std) | center and scale | familiar | outliers still dominate after z-scoring; mean/std are **batch statistics** (banned coupling at serve); unit-inconsistent across fields/streams | ✗ |
| B | Fixed signed log1p squash | one guess that usually works | cheap heavy-tail taming, sign-safe, deterministic | the shape is a **guess** — right for dollars, not for everything | ⚠️ documented **fallback** only (per doctrine), with fallback-fired receipt |
| C | **Measured monotone transform** (quantile/spline fit at fit time, frozen in artifact) ⭐ | a number becomes its *rank in its own field's measured distribution* — "what does $40 mean in THIS stream" | unit-free by construction (re-scale dollars→cents and embeddings are identical — a testable law); strictly order-preserving (monotone: 2 always ≥ 1 — a law, not a hope); deterministic per-sample at serve (no batch coupling); OOD values clip + support-widen (A1); zero-inflated fields get an explicit **zero atom** so "how much above zero" stays readable | fit-time bookkeeping (knots stored in the artifact — negligible) | ✅ primary |
| D | Free-learned transform per field (small MLP end-to-end) | let the net invent the scale | most flexible | **monotonicity not guaranteed — order can be violated (lawlessness)**; gradient fragility at scale boundaries; memorizes noise; uninspectable | ✗ |

Ratified: **C primary + B as documented fallback (DEC-394)**. Unit-invariance law test
wired into construction gates; zero atoms standard; fallback-fired receipts required.

### T6 — Sequence construction / chunking (presented 2026-10-27)

Problem: a customer can have 50,000 events over a decade; the engine keeps constant
state and one batch holds finite memory. What feeds the scan?

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | **Truncate to last N events** | forget the ancient past | simple, bounded memory | N is a **fixed literal**; the deepest past is exactly what the slowest timescales (tenure/loyalty) exist to carry — loss is arbitrary, detectable only by receipts | ✗ as doctrine |
| B | **Chunked streaming, exact composition** ⭐ | split history at *arbitrary* chunk boundaries; scan chunk 1 → state feeds chunk 2 → … The merge law (DEC-386; receipt 8.88e-16) makes this **exact, not approximate**. Chunk size = token budget derived from VRAM/hardware, never an event count. Ledger composes the same way. State checkpoints at derived cadence so any past as-of answers from the latest checkpoint, not a full replay | no truncation, no loss; the engine never "knows" a sequence length — only "state so far"; memory bounded honestly; serve = O(1) ticks after first pass | checkpoint cadence + anchor density must stay derived (receipts), not hardcoded | ✅ |
| C | Sliding windows with overlap | overlapping context windows | smoother context | window size literal, overlap double-counts, fights tick-at-event physics (same family DEC-386 rejected) | ✗ |
| D | Resample / downsample long histories | thin the events | cheap | destroys the irregular timing the project exists to model (banned binning family) | ✗ |

Ratified: **B (DEC-395)**.

### T7 — Timescale allocation (presented 2026-10-27)

The engine remembers at several speeds (fast channels for "next event", slow channels for
tenure/loyalty). How many speeds, and which ones?

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | Human ladder: hours/days/weeks/months | a watchmaker's dial | readable; familiar | business-clock **cadence literals** (banned family); an hourly-gig stream and a seasonal-retail stream would get the same dial | ✗ |
| B | **Measured rhythm spectrum → allocate** ⭐ | S1 already measures each stream's autocorrelation/FFT spectrum + gap quantiles; **number of channels = number of significant spectral peaks** (significance vs a measured shuffle-noise floor), each τ **initialized at its peak** and then **learned** onward | the stream picks its own rhythms (M2 exactly); doubles as the DEC-386 ansatz-detector (flat/odd spectrum = receipted warning); peak count is data, so it generalizes to any stream | spectral estimation needs a noise-floor calibration (shuffle null) — one more receipt | ✅ primary |
| C | Overbuild a geometric ladder, prune the quiet ones | 16 log-spaced τs, delete channels whose measured contribution < noise | robust; no spectral machinery | wastes capacity until pruned; pruning threshold must itself be noise-calibrated | ⚠️ documented **fallback** (small/flat-spectrum streams), pruning receipt required |
| D | Free-learned τ, no measured init | every channel invents its own speed | maximal adaptivity | unmotivated channels collapse onto each other and leave holes (known degenerate behavior); throws away the spectrum S1 already paid for | ✗ as init policy (the *learning* part of D is kept: τ stays learnable after peak-init) |

Ratified: **B with fallback C (DEC-396)** — spectrum receipt (peaks, powers, shuffle-noise
floor) required; τ learnable after peak-init; layer count = derived convergence budget.
Operator challenge on B-vs-D answered in the open and recorded in DEC-396.

### T10 — Bucket membership (presented 2026-10-27)

Which outputs deserve to BE a bucket? Rule from the Feature-3 audit: a bucket = a content
class a consumer provably needs. Today's six content buckets, examined: shape (who) ·
learned_state (memory internals — empirically earned: channel purity 0.72) · forecast
(when/summaries — earned: best churn view) · uncertainty (what-I-saw) · heads (identity/
next-event/relational **predictions** — the suspect grab-bag) · measure (ledger) · plus
`response` approved (what-if, DEC-384) and `full` (fixed brain bundle).

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | Freeze current membership | what ships is what stays | zero rework; receipt continuity | blesses `heads` (a bag of network predictions, not a content class) without an earned-keep verdict — accretion by inheritance | ✗ |
| B | **Content class + earned-keep** ⭐ | a bucket exists iff (1) it maps to a content class and (2) at least one card row currently WINS in it (attribution receipt); buckets that stop earning are dissolved (prefer deletion). Concretely: shape/learned_state/forecast/uncertainty/measure/response stand (each earns today); **`heads` is PROVISIONAL** — next-event/identity outputs must win a card row by attribution or the bucket dissolves (consumers re-derive predictions from `shape` via their own heads; `response` stays separate — different class, different disciplines) | membership is decided by receipts, not habit; keeps the bucket set exactly as big as the consumers' needs | bookkeeping: per-row attribution must stay current (already required by DEC-390) | ✅ |
| C | One giant blob; consumers slice | just `full` | fewest names | measured harm — concatenation dilutes distance probes; breaks per-bucket decodability contracts (DEC-388) | ✗ |
| D | Consumer-shaped buckets (`churn_bucket`, `value_bucket`…) | a bucket per known downstream task | maximally convenient for today's consumers | **forbidden**: known use cases would SHAPE the foundation (veto-never-shape); also guarantees the next consumer is unaccommodated | ✗ |

Ratified: **B (DEC-397)**. `heads` provisional pending attribution; `response` separate by
discipline boundary; earned-keep is permanent — buckets that stop winning dissolve.

### T8 — Where the engine may bend (nonlinearity placement; presented 2026-10-27)

The core is a linear spring: exponential fade between events + additive ticks. Pure
linearism caps expressivity (your own v6 spike measured bilinear content×state coupling
raising usable rank 1.79× — pure linear was measurably not enough). But every bend is a
bet, and a bend in the wrong place breaks a ratified law. Where may nonlinearity live?

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | **Bend at write and read; never between** ⭐ | nonlinearity only (1) **at event ticks**: learned gates (which events write, how strongly) + bilinear content×state coupling, and (2) **at readout** (per-bucket ensemble, learned positive weights — no softmax collapse). The fade between events stays **exactly linear** | keeps merge exactness (receipt 8.88e-16 → T6 chunking stays free), stability law (contractive linear core), determinism; spends bends exactly where content meets state — where the v6 receipts say rank comes from | bilinear coupling costs compute at ticks (GPU: fine) | ✅ |
| B | No bends (pure linear) | lawfulness maximal | provable everything | **measured failure**: rank/expressivity too low; soft linear workarounds (centering) were measured as no-ops — rejected by experiment | ✗ |
| C | Bend the fade itself (nonlinear decay ODEs, MLPs running between events) | richer dynamics | could express non-exponential memory | **voids the merge law** (chunked scanning no longer exact — T6's free lunch revoked), stability law at risk, determinism degraded | ✗ as edit to the core; admissible ONLY as a **second, parallel mechanism** added by battery receipt (DEC-386 ansatz-detector path), each mechanism carrying its own merge test |
| D | Batch-coupled bends (batchnorm/ZCA family) | normalize by the batch | — | **banned absolutely**: per-sample contract; batch statistics manufacture capacity (the ZCA scandal this repo has already lived once); train≠serve divergence | ✗ |

Ratified: **A (DEC-398)**. The spring between events is sacred; the pen (write: gates +
bilinear content×state) and the voice (readout: learned positive weights) may bend.
Richer dynamics admissible only as receipted parallel mechanisms with own merge tests;
batch-coupled bends banned absolutely.

### T11 — The last derived knobs (presented 2026-10-27 — closes the internals arc)

Three knobs left that are traditionally typed into code. The same rule applies: measure
what measurement can see.

| Knob | Derived from (if A) | The usual literal it replaces |
|---|---|---|
| **Mask rate** (masked-recovery objective) | per-field measured missingness + entropy — mask what THIS stream actually withholds, so the objective becomes "predict this stream's own blank patterns" | BERT's fixed 15% |
| **Horizon set** (future-summary targets) | sampled from measured behavioral-time statistics (event-gap quantiles, per-stream activity spans), receipted | the 30/60/90-day curriculum (a cadence literal) |
| **Objective weights** (13-way battery mix) | learned balancing from measured per-objective noise/gradient scales; weights decaying to zero trigger the prune pass (deletion receipt); selection on held-out self-supervised loss ONLY (DEC-387) | equal weights, or expert tuning |

Folded in from training doctrine, pinned to the Derived column: **learning rate measured
from curvature (stability bound lr < 2/L), patience from validation SE** — convergence,
never counts.

| # | Option | Verdict |
|---|---|---|
| A | **All three derived at fit time, receipted** (table above) | ✅ recommended |
| B | Community defaults (15%, 30/60/90, equal) | ✗ a bag of literals |
| C | Grid-search on held-out SSL loss | admissible **fallback** if the learned mechanism proves unstable (search over selection-on-graded-metric stays banned); documented, not doctrine |
| D | Expert-tuned weights | ✗ M2 violation |

Ratified: **A with C shelved as documented fallback (DEC-399)**. **The T-series is
complete — internals arc closed.**

### T-series — CLOSED (2026-10-27)

| # | Decision | Status |
|---|---|---|
| T1 | Token unit | **LAW** — one event = one tick (DEC-391) |
| T2 | Vocab policy | **LAW** — measured head + hashed tail + profile id-typing + flagged unseen (DEC-393) |
| T3 | Numeric encoding | **LAW** — measured monotone transform; log1p fallback (DEC-394) |
| T4 | Presence semantics | **LAW** — not-applicable / unknown / present, data-pruned (DEC-392) |
| T5 | Composition | **LAW** — symmetric, learned gates (permutation receipt) |
| T6 | Sequence construction | **LAW** — chunked streaming via merge law; derived budgets (DEC-395) |
| T7 | Timescale allocation | **LAW** — spectrum peaks vs shuffle-noise floor; peak-init then learn; geometric-ladder fallback (DEC-396) |
| T8 | Nonlinearity placement | **LAW** — bend at write + read; fade exactly linear; batch-coupled banned (DEC-398) |
| T9 | Capacity d | **deferred by design** — retention-vs-size curve → k_min (DEC-390; implementation step 7) |
| T10 | Bucket membership | **LAW** — content class + earned-keep; heads provisional (DEC-397) |
| T11 | Final knobs | **LAW** — measured mask rates / horizons / weight balancing; search shelved as fallback (DEC-399) |
| — | Training regime | **doctrine-pinned** — convergence not counts, lr < 2/L, patience from SE (receipts at implementation) |
| — | Schema | **CONTRACT** — `entity_id \| event_ts \| event_type \| event_attributes` (DEC-391) |

### A-series — encoder microarchitecture (opened 2026-10-27)

DEC-386 settled the memory *family* (lawful continuous-time gated recurrence) and T8 the
*bend locations*; the layer interior remains. Closest prior art: the **Mamba-shaped**
engine — but ours is law-constrained, not off-the-shelf: **Δt is measured from the clock**
(Mamba *learns* per-token Δ — time is data-truth here, not something to learn), the
**mass pathway skips decay entirely** (Mamba decays everything), and merge exactness is a
gate, not a hope. Mamba-1-vs-Mamba-2 internals (e.g., SSD kernel formulations) are
therefore **implementation** choices: pick the fastest lawful kernel on the GB10 by
benchmark receipt (DEC-389), not a design debate.

| # | Question | Status |
|---|---|---|
| A1 | **Block anatomy** — what is one layer? | **LAW (DEC-400): recurrence + channel-MLP, two sub-blocks, attention-free** |
| A2 | **Depth** — how many layers | **LAW (DEC-400): derived from held-out convergence; card's `layers 2` = inherited, pending re-derivation** |
| A3 | **Gates** — which gates exist, where, measured init | **LAW (DEC-401): full lawful set — write + event-locked persistence + read gates; mass never gated; measured init; step-0 stability receipt** |
| A4 | **Normalization** — allowed where? (BANNED on state paths — mass additivity C3 + contractivity analysis; per-sample norms only on bend paths) | **LAW (DEC-402): per-sample, bend-path inputs only, admitted by receipt; unnormalized magnitude lane mandatory** |
| A5 | **Residuals** — skip structure (legal in channel space; never through state) | **LAW (DEC-403): channel-space skips + read taps, learned neutral-init gains (depth fairness); dense concat parked** |
| A6 | **Attention's place** — allowed only as read-side pooling; may never be the state (DEC-386) | **LAW (DEC-404): read-side only, resident objects only — residency law** |

### A1 — Block anatomy (presented 2026-10-27)

Transformer discipline says a layer = two sub-blocks: a **time-mixer** (attention) + a
**channel-mixer** (MLP). Our lawful recurrence IS the time-mixer. What else belongs in a
layer?

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | Pure recurrence (one sub-block) | maximum law purity | fewest parts | channel mixing must live only in write/read gates — cramped; the measured rank win (1.79×) came from content×state interaction that wants dedicated room | ✗ |
| B | **Recurrence + channel-MLP** ⭐ (two sub-blocks, attention-free) | keep the transformer's *discipline* (mix across time, then across channels), replace its *instruments* (recurrence for time, MLP for channels); bends all sit at write/read (T8-legal) | matches where the measured rank lives; depth scales the interaction; per-sample ops only; kernel-friendly for GPU | slightly more params than A (cheap — capacity is certified by curve anyway, T9) | ✅ |
| C | Recurrence + attention over a recent bounded window (true hybrid) | keep attention as a spice | sharpens short-range patterns | the window is a **literal** and history-dependent state in miniature — DEC-386's problem re-imported at small scale; park behind a receipt (T8's second-mechanism pattern) | ⏸ receipted candidate |
| D | Recurrence + short convolution pre-stage (Mamba-style conv) | smooth before integrating | cheap, standard | kernel width = fixed-range literal; fast-τ channels already cover short-term structure (T7) | ⏸ receipted candidate |

Ratified: **B (DEC-400)** — two sub-blocks (time-mix = lawful recurrence; channel-mix =
per-layer MLP), attention-free. C/D parked as receipted candidates per DEC-398's
second-mechanism pattern. "Two" = count of mixing AXES (derived), not rounds (A2, measured).

### A3 — Gates: what may the engine decide to remember, hold, or say? (presented 2026-10-27)

Gates are the engine's attention budget. Already law from T7/T8: **write gates + bilinear
content×state** at ticks, **positive-weight ensembles** at readout, **static per-channel τ**
(spectrum-initialized, then learned). The genuine open fork: may content ALSO steer
*persistence* (a forget gate)?

Plain meanings: **write gate** = "is this worth recording?"; **persistence (forget) gate**
= "how long do I keep this thought?" (rereading something important should make it last
longer); **read gate** = "how loudly do I speak about this channel when asked?"

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | Minimal set (write gates + readout ensembles; static τ only) | fewest dials | maximal stability surface | can't express "this event makes memories stick longer/shorter" — persistence is one-size per channel | ⚠️ baseline |
| B | **Full lawful gate set** ⭐ | write gates + bilinear (law) + **event-locked persistence gate**: at each event the engine may retune each channel's holding time; **between events decay is exact exponential at locked parameters → merge law survives, T6 chunking stays free**; τ stays inside **bands derived from the spectral peak's bootstrap uncertainty** (no literals); per-layer read gates; positive-weight ensembles | expressive where it matters (content-dependent memory), lawful between events, band + step-0 stability receipts guard the known instability (content-driven decay can be pushed to 0/∞) | needs band-derivation + stability receipts kept current | ✅ |
| C | Maximal: decay steered continuously between events | fully adaptive fade | — | **voids the merge law** (T6's free lunch revoked) + stability endangers (DEC-398's sacred spring) | ✗ |
| D | Bare recurrence, no extra gates | simplest | — | contradicts the measured rank receipts (T8) | ✗ |

Folded rulings if B (each already law-shaped, restated here):
1. **The mass pathway never gets gates** — gating a sum falsifies sums (mass-additivity
   gate C3 = exact 0 error).
2. **Measured gate init** (T7's rule): write-gate biases from measured event rate/redundancy;
   persistence gates initialized at the static-τ equivalent (spectrum); random init ✗.
3. **Determinism (A1 axiom):** gates are deterministic functions; no stochastic gates at
   serve; any training-time noise is seed-reproducible under the C4 receipt.
4. Stability law binds: lr from curvature, divergence kills the run — persistence gates must
   ship a **step-0 stability receipt**.

Ratified: **B (DEC-401)**. Event-locked persistence gate; exact exponential between events;
τ bands from spectral bootstrap uncertainty; mass never gated; measured gate init; step-0
stability receipt. Option A (minimal set) retained as a receipted ablation, not doctrine.

### A4 — Normalization: where is it allowed to sit? (presented 2026-10-27)

Two parts are already law: batch-coupled norms are **banned absolutely** (DEC-398), and no
norm may sit **on the state or mass paths** — a norm there rescales the state
non-affinely, which voids the merge law (`F(g∘f)=F(g)∘F(f)` dies), breaks additive sums
(mass gate C3 = exact zero error), and destroys level/magnitude (a first-class requirement
the contractive core must not lose). What remains genuinely open: are **per-sample** norms
allowed on the *bend paths* (the standard "pre-norm" habit: normalize the input to the
channel-MLP and the gate projections)?

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | None anywhere | measured init (T7/T11) + lr-from-curvature may make norms unnecessary | fewest parts; zero risk of erasing magnitude; purity | if a deep bend path destabilizes, we discover it at training time rather than having a lawful instrument ready | ⚠️ start here |
| B | **Per-sample norms on bend-path inputs only, admitted by receipt** ⭐ | RMSNorm-style (per-sample statistics, no batch coupling) on the channel-MLP input and gate projections — **never** on state, mass, or readout; any rescaling must leave a **parallel unnormalized magnitude lane** so level survives; each norm enters only when a training-behavior receipt shows instability on that path | standard deep-training stability without touching any law; per-sample ⇒ train=serve; placement decided now (no future drift), presence decided by evidence (prefer-deletion) | two receipts to keep: the placement audit + the instability justification | ✅ |
| C | Norms on the state too (unit-sphere / traced-state SSM) | normalize the hidden state each step | bounded-state proofs get easy | **voids merge exactness** (state-normalization is not affine in the state), **destroys level** (the thing RFM-style summaries keep and we are forbidden to lose), breaks mass additivity | ✗ |
| D | Batch-coupled norms (batchnorm/ZCA family) | — | — | **already banned** (DEC-398): capacity fabrication, train≠serve divergence | ✗ closed |

Ruling pending (recommend **B**, which is "A, plus a lawful instrument held in reserve").
Plain summary: **norms may wash the pen and the voice, never the inkwell.** Per-sample is the
hard condition — anything computed across customers is the ZCA scandal waiting to happen again;
and a norm that rescales must never be the only path, because magnitude is information.

Ratified: **B (DEC-402)** — per-sample norms on bend-path inputs only, evidence-gated;
parallel unnormalized magnitude lane mandatory; state/mass/readout norms banned; batch-coupled
closed. Placement audit + per-norm instability receipts owed.

### A5 — Residuals: which skips may the engine have? (presented 2026-10-27)

A residual ("skip") is `output = input + f(input)` — the shortcut that lets depth train.
Already-law boundaries: skips live in **channel space**, never route around the recurrence,
and never mix the mass channels with shape channels through a shared path (mass additivity
gate C3 must keep measuring exact zero). Note the lock with DEC-402: the mandatory
*unnormalized magnitude lane* beside any norm **is** the residual path — a "no residuals"
ruling would force that lane to be reinvented unlawfully.

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | No residuals (pure stack) | fewest parts | maximal contractivity | deeper derived stacks (A2) starve gradients; the need would be discovered mid-training with no lawful form ready; also strands DEC-402's magnitude lane | ⚠️ |
| B | **Channel-space skips + per-layer read taps, learned neutral-init gains** ⭐ | (1) skip around the channel-MLP sub-block and layer-to-layer input skips **in channel space**; (2) every layer exposes a **read tap** into the readout ensemble (A3's learned positive weights — already law); (3) each skip carries a **learned scalar gain initialized at the neutral element** (zero contribution ⇒ adding a layer cannot change the function at step 0 — ReZero-style) | **depth fairness**: the A2 depth sweep is only honest if a deeper model *contains* the shallower one at init, so improvement is attributable to depth, not to re-randomization; skips earn their contribution by gradient (M2, no `0.5`/`1/(N+1)` literals); step-0 function is identity-preserving (stability, A1) | step-0 zero-gain skips can train slowly at first (mitigated: measured init elsewhere; gradient flows through the other paths); placement audit owed | ✅ |
| C | Skip **around** the recurrence (`s_t = fade(s_{t-1}) + Bx_t + s_{t-1}`) | bypass the fade | easy gradient path | algebraically it is `(A+I)·s_{t-1}+Bx_t` — an **un-faded second copy of history**: fast-τ channels can no longer forget (bounded-memory law defeated, measured τ falsified), and `A+I` can be expansive (stability law endangered) | ✗ |
| D | Dense multi-connectivity (concat every layer into the readout — ConnectNet style) | everything connects to everything | multi-resolution features at read | duplicates a mechanism we already ratified: A3's read gates ARE learned positive weights over sources — dense concat is the same idea with extra dimensions (prefer-deletion); admissible only by receipt | ⏸ receipted candidate |

Ruling pending (recommend **B**). Plain summary: **the past may be re-read, the present may be
short-circuited, the fade may never be bypassed** — and every shortcut has to *earn* how much
it contributes.

Ratified: **B (DEC-403)** — channel-space skips + per-layer read taps, learned gains at the
neutral element (zero contribution at step 0) for **depth fairness**. Dense concat parked with
a **pre-registered admission trigger** (separable-access failure demonstrated by matched
control) — never relitigated by opinion.

### A6 — Attention's place (presented 2026-10-27 — closes the A-series)

Family ruling (DEC-386) and A1 already settled that attention may **never be the memory**.
What remains: may it be a *read* behavior? Note one sub-case is already occupied ground —
"which layer's tap should I trust?" is exactly A3's read gates (learned positive weights), so
a separate layer-attention mechanism would be a duplicate (prefer-deletion). The genuine
question is content-adaptive aggregation on the read side.

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | Banned outright (fixed poolings: last/mean/max/magnitude + A3 tap ensemble) | zero risk | fewest parts; nothing can re-import history | discards a capability the readout doctrine lists as available (content-adaptive "summarize the ticks that matter") | ⚠️ baseline |
| B | **Read-side only, resident objects only** ⭐ | pooling weights computed from the customer's own current state, scoring what the **fixed-size state already carries** (taps + state-resident summaries); no KV cache; no raw-history access; positive weights (no softmax that can collapse to one branch) | content-adaptive reading without touching any law: state stays constant-size, merge law untouched, serve stays O(1) | needs the **residency receipt** kept current (below) or it quietly becomes D | ✅ |
| C | Also over a bounded recent window of raw ticks (serve-side KV cache) | sharper short-range reads | — | window = literal + a per-customer history buffer ⇒ constant-state contract broken at serve; train≠serve | ✗ |
| D | Full-history attention at read (retrieve over the whole event log) | "just look it up" | — | **PATH A in disguise**: the brain starts depending on raw material the ledger owns; O(history) serving cost; the frozen-donor contract (consumers read *embeddings*) is void | ✗ |

Folded rulings:
1. **Residency law** (the load-bearing one): any read path may read **only what the fixed-size
   state carries**. Receipt: the readout is re-implemented against the frozen artifact alone —
   no raw events reachable — and must be bitwise-equal to the training-time readout. If a read
   needs anything else, it is rejected as D in disguise.
2. Training-time attention obeys the same residency rule, never reads the future, never
   crosses customers (per-sample contract, DEC-402).
3. **Positive weights, no collapsing softmax**; gate init measured (A3's rule).
4. **Attention weights are diagnostics**: emitted as a receipt of *where the voice looked*
   (caterpillar reads them), never used to gate mass, never fed back into training signal.

Ruling pending (recommend **B**). Plain summary: **attention may be a way of looking, never a
way of remembering.**

Ratified: **B (DEC-404)** — **residency law**: any read path reads only what the fixed-size
state carries; proven by re-implementing the readout against the frozen artifact alone
(bitwise-equal). Attention weights are read-only diagnostics. Layer-trust attention deleted as
a duplicate of A3's read gates.

### A-series — CLOSED (2026-10-27)

| # | Decision | Status |
|---|---|---|
| A1 | Block anatomy | **LAW** — two sub-blocks: lawful recurrence (time-mix) + channel-MLP (channel-mix), attention-free (DEC-400) |
| A2 | Depth | **LAW** — derived from held-out convergence; card's `layers 2` = inherited, pending re-derivation (DEC-400) |
| A3 | Gates | **LAW** — write + event-locked persistence + per-layer read gates; τ bands from spectral bootstrap; mass never gated; measured init; step-0 stability receipt (DEC-401) |
| A4 | Normalization | **LAW** — per-sample, bend-path inputs only, admitted by receipt; unnormalized magnitude lane mandatory (DEC-402) |
| A5 | Residuals | **LAW** — channel-space skips + read taps, learned neutral-init gains (**depth fairness**); dense concat parked with pre-registered trigger (DEC-403) |
| A6 | Attention | **LAW** — read-side only under the **residency law**; weights are diagnostics (DEC-404) |
| — | Kernel formulation (Mamba-1/2-style SSD etc.) | **implementation** — fastest lawful kernel on the GB10 by benchmark receipt (DEC-389), not a design debate |
| — | Parked receipted candidates | recent-window attention · short-conv pre-stage (A1) · nonlinear parallel dynamics (T8) · dense concat (A5) — each admitted only by battery receipt, never smuggled into the core |

Doctrine lines from the A-series: *the spring between events is sacred; the pen and the voice
may bend* · *norms may wash the pen and the voice, never the inkwell* · *the past may be
re-read, the present may be short-circuited, the fade may never be bypassed* · *attention may
be a way of looking, never a way of remembering*.

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

## 6C. The C-series — consumer-side contracts (opened 2026-10-27)

The encoder is specified; the **consumer side of the boundary** is not yet. Derived from the
four declared consumers (DEC-384) by asking, for each: what does it receive, what may it
build, what is it forbidden to do, how is it gated, what covers it if nobody imagined it,
how does it know the artifact changed under it, and where does personalization live.

| # | Question | Status |
|---|---|---|
| C1 | **The read surface** — what exactly does a consumer receive, and how is "brain-only" made mechanical? | **LAW (DEC-405): artifact = the interface; dumb consumers; no ability to talk back** |
| C2 | **Permitted head classes** | **LAW (DEC-407): linear is the only shipping supervised head; kNN → unsupervised; MLP → diagnostic; head-ladder receipt tests it** |
| C3 | **Firewall rules** | **LAW (DEC-412): no mutation, no backward edges, labels never reach B, no future info at serve — plus the data-mediated leakage law: no derived/consumed table may be an event source** |
| C4 | **Per-consumer gates** | **LAW (DEC-412): gates computed by the card from receipts, never self-declared; thresholds frozen pre-run; uncertified is advertised** |
| C5 | **The unknown-consumer guarantee** | **LAW (DEC-410): the coverage lattice — proven form, measured content, hunted unknown, per-consumer falsification; "universal" retired until green; v1 stub hunter** |
| C6 | **Version & compatibility contract** | **LAW (DEC-412): version = behavior; additive-only within a line; deprecation ledger; pin major+hash and refuse mismatch; geometry epoch for distance consumers** |
| C7 | **Personalization** | **LAW (DEC-412): adapter, never rewiring — per-segment LINEAR heads at derived capacity with a derived support floor; below it, fall back to global with widened uncertainty** |
| C8 | **The factory** — supervised plugin generation from a target spec (operator directive, DEC-406) | **LAW (DEC-408): spec-interpreted engine + gated `custom` escape** |
| C9 | **Unsupervised instruments** — dumb instrument set for structure/geometry/anomaly, template-generated | **LAW (DEC-408): ratified set, derived counts, table/random/destroyed comparators** |
| C10 | **Decision & RL consumers** — decision-platform contract only; RL internals live in peer mini-projects | **open arc (DEC-409)** — invariants adopted, 4 open items named |

### C1 — The read surface — **RATIFIED by operator directive (DEC-405): the artifact *is* the interface**

Operator's ruling, which superseded my first draft: **plugins are dumb consumers; they must not
have the *ability* to talk to B; there may be thousands of them.** My original C1-B ("single
typed accessor at read time") is **withdrawn** — anything standing in the consumption path is a
dependency waiting to happen, and with fan-out to thousands of consumers, per-consumer coupling
is fatal. Fan-out, not tidiness, is the argument.

**The law:**

| Rule | Content |
|---|---|
| **1. Artifact = interface** | `customer_key × as_of` tables (embeddings, buckets, ledger, uncertainty) written by **one** producer path. A consumer is anyone who can run a SELECT — that is what "accommodates unknown consumers **by construction**" means: the contract is a **schema**, not a function signature |
| **2. No talking back** | Banned shapes: imports in either direction between B and C; consumer-invokable compute in B (no request-driven recompute, no callback, no gateway B answers); shared mutable state; consumer-specific columns/tables/logic in B's namespace; consumers write only their own namespace. Enforced by import-graph + call-site scan (`principle_audit.py --strict` / CI) |
| **3. Read-only in fact** | Output immutable and content-addressed (hash + version + build commit); consumers get read-only access. Physical enforcement, not discipline |
| **4. Self-describing artifact** | *Required by fan-out:* the artifact carries its own metadata (version, hash, `as_of`, freshness bound, bucket inventory) so each consumer **self-validates and locally refuses** a stale/out-of-support artifact without asking B. Reject, never clamp, at the boundary |
| **5. Producer-side materialization** | Anything derived from B's own guarantees (pre-action cuts, ledger companion views) is **materialized by the producer** — thousands of consumers each re-deriving a cut is guaranteed drift, and it escapes the encoder's certification. B stays oblivious; it just publishes more columns |
| **6. Judgment outside both sides** | "Which buckets a consumer actually used" comes from the judge's **read-only** observations (declared manifest + static column-reference extraction + **knob-turn receipt**: zero/permute a bucket, see if the score moves) — never a gate inside B |

**Existing violations (one cleanup work item):** `looking_glass/white_queen_bridge.py` (×4) and
`looking_glass/candidate_roster.py` import `white_queen.tribunal.ope.api` (**B → downstream**);
`plugins/{inference,ablation,rehearsal}.py` import `looking_glass.*` (**C → B**) — including
`rehearsal.py → daily_states.update_day`, a consumer that can **produce its own input**, the
worst form of the loop. Shared constants → a leaf `contracts/` module that imports nothing;
certification call sites relocate to the judge/card layer (which may read everything, the
`caterpillar` pattern).

**Disclosed costs:** declarations are self-reported (honesty enforced by the knob-turn receipt —
concealing a ledger read buys a consumer nothing); version pinning is matched by the judge
rather than observed in a call path, hence rule 4's stamped metadata.

Plain summary: **B publishes to a wall. Thousands may read the wall. Nobody can ask the wall a
question, write on it, or get a wall built just for them.**

### C8 — The supervised plugin factory (operator directive, DEC-406)

The observation that makes this mechanical: **the supervised dataset already exists.** One row
per `customer_key × as_of`; features = the published columns; the label joins by key from the
label store. So a supervised plugin is not software — it is a **target + discipline**.

| # | Option | In one line | For | Against | Verdict |
|---|---|---|---|---|---|
| A | Generate editable plugin **code** per target (scaffold) | familiar | consumers can customize anything | correctness diffuses into thousands of editable copies; the first edit that breaks an as_of split silently voids the encoder's certification claim, and the card can't see it | ✗ |
| B | **Spec-interpreted factory** ⭐ | a plugin **is** a declarative spec (`target`, `entity grain`, `label window`, `buckets allowed`, `task type`, `cadence`) + trained artifact + receipt, produced by **one** engine | one implementation of every discipline (single source of truth); guarantees are **structural, not reviewed**; the card enumerates plugins from specs, so a claim with no receipt simply cannot be displayed; one engine fix re-certifies all consumers — with generated code it never propagates | exotic needs must use the escape hatch | ✅ |
| C | **B + gated custom escape** ⭐ | hand-written plugins allowed, but must pass the **same** gates and emit the **same** receipts, labeled `custom` on the card | keeps the factory honest (it must express real needs) without letting anyone skip discipline; a growing `custom` count is itself a card signal of factory deficiency | the escape must stay rare and visible | ✅ (adopted with B) |
| D | Free-form consumer code, human-reviewed | trusts experts | — | review does not scale to thousands; unreviewed paths become unfalsifiable claims about the encoder | ✗ |

**The discipline the engine owns on every consumer's behalf** (the actual product): as_of-anchored
time-ordered splits (train strictly before validation — no future information); features
restricted to published columns and to the buckets the spec declared; head class = the locked dumb
set with **derived** capacity (C2); calibration with measured coverage; **both** comparators
(capacity-matched control **and** the strong PATH A incumbent) with paired CIs; degenerate targets
**excluded and reported**, not passed; residency re-check (the trained scorer runs from the
published table alone, DEC-404); pinning to the artifact's version + hash (DEC-405 rule 4); receipt
with N, commit, hashes. `sklearn`/`scipy` supply estimators, splits, calibration, and statistics
(principle 14) — none of it hand-rolled.

**Firewall (C3, enforced here):** the spec names a *label source*; labels join by key, read-only,
and **never** flow into B. Encoder selection stays on held-out self-supervised loss (DEC-387) —
the factory consumes the encoder, it does not train it.

Ratified: **B + C (DEC-408)**. Plain summary: **you supply the question; the factory supplies
the question's honesty.**

### C2 — The head ceiling — **RATIFIED (DEC-407): linear is the shipping head**

The trio is not a menu; the head follows from what the foundation is *for*.

| Instrument | Role | Why |
|---|---|---|
| **Linear** | **the shipping supervised head** | accessibility means information is recoverable by *simple* consumers; if a linear head cannot read a target off the published columns, that is an **encoder defect**, not a cue to strengthen the head. Strengthening the head is how a foundation gets relieved of its obligations |
| **kNN / distance** | **the unsupervised product** (C9) | here distance *is* the deliverable (similarity = shared futures); it is not a supervised head in this architecture |
| **Small MLP** | **diagnostic tier only** — labeled, read-only, never ships, never earns a card row | attribution: linear fails + MLP succeeds ⇒ **accessibility gap** (trunk work item); linear fails + MLP fails ⇒ information genuinely **absent** (sufficiency gap) |

**Head-ladder receipt (new card row, the test the operator asked for):** every certified supervised
target reports all three instruments at derived capacity, identical splits, paired CIs. MLP
beating linear beyond noise renders as an **accessibility gap against the encoder** — never as
permission to ship the MLP. A `custom` consumer may request an MLP head; its row is labeled
`custom/MLP` and does **not** count as evidence for the donor.

Capacity is **derived** from the consumer's own data (no typed unit counts; `sklearn` defaults
overridden with derived values, emitted in a derived-values receipt), trained to convergence,
calibrated with measured coverage, and **capacity is a card column**. Comparators: a
capacity-matched linear row on PATH A features (the fair fight) **and** the strong PATH A
incumbent GBM (the practical bar) — a PASS needs both. The shipping set is closed but
re-openable by receipt (locked trio cannot reach a certified target *and* the candidate stays
dumb: published columns only, no reach into B).

### C9 — Unsupervised instruments (sklearn-grade, same template discipline)

Same published rows, no labels anywhere in fitting. Dumb instrument set: **distance/kNN**
(similarity = shared futures, S5 distance pressure) · **PCA / whitening** (rank & isotropy
diagnostics) · **k-means / HDBSCAN** (segments) · **IsolationForest / Mahalanobis** (anomaly, read
with the `uncertainty` bucket) · **cross-cohort transfer** (fit on one cohort, tested on another).
Every count that would otherwise be typed — clusters, neighbors, contamination, components — is
**derived and receipted**; comparators are always **table / random / destroyed**, and a structure
claim that beats neither is excluded, not celebrated. Geometry gates (isotropy, effective rank,
decay) attach to the **artifact**, not to each consumer's re-derivation.

Ratified (DEC-408).

### C5 — The unknown-consumer guarantee (presented 2026-10-27 — the deepest row)

The claim to be carried: any present or future consumer — ours or another team's, **known or
unknown** — is covered **by construction, not by enumeration**. Stated plainly first, because the
honest answer is not a single mechanism: **no available mechanism proves it.** What we can build is
a stack in which (a) part of the claim is genuinely proven, (b) the rest is measured over a
*declared* family, (c) the unknown is *hunted*, and (d) any consumer can falsify the claim in
seconds — and in which **the word "universal" is never printed unless the card derives it.**

| # | Option | What it claims | Verdict |
|---|---|---|---|
| A | Enumerate consumers/tasks (a benchmark) | "we tested N targets and they passed" | ✗ — not universality; gameable by flattering tasks; the next consumer is unaccounted for. This is what most "foundation model" claims actually are |
| B | The algebra proves it (law only) | the merge homomorphism + contractivity + fixed capacity make the state a sufficient statistic | ⚠️ **true but partial** — the algebra guarantees *form* (it is a state of the stream, not a feature vector for our tasks) and *consistency*, never *content*: bounded memory means forgetting, and how much is lost is a measurement, not a theorem |
| C | Measure worst case over a declared family (the `U` metric) | `U = 1 − worst-case oracle-gap` vs full-history at fixed capacity | ✅ necessary, but coverage = the family's width; a published family is honest, an unpublished one is marketing |
| D | Hunt adversarially (the task-hunter) | search over functions of the future for a target the donor cannot serve | ✅ the only mechanism that reaches a consumer we have not met — and it inherits its own blind spot (the hunter searches its own hypothesis class) |
| E | **The coverage lattice** — B + C + D + per-consumer falsification, with the claim retired until earned | ⭐ recommended (below) | ✅ |

**The lattice — four legs, each with its guarantee type declared:**

| Leg | What it carries | Guarantee type |
|---|---|---|
| **1. Form** | bounded, mergeable homomorphism; explicitly parameterized forgetting; per-sample; schema-agnostic; two pathways | **proven by construction**, verified by receipts (merge 8.88e-16, equivariance, mass additivity) |
| **2. Content** | battery's six content classes; `Δ_suff` vs shuffled null; oracle-gap `U` at fixed capacity over a self-supervised family whose members are **published on the card** | **measured**, family-bounded and visibly so |
| **3. Adversary** | synthesized targets (random functionals of the future, programmatic targets, "find me a target the donor can't serve"); each blind spot becomes a new battery pressure with its own row | **minimax estimate**, hunter-class disclosed |
| **4. Boundary falsification** | every consumer — known or unknown — gets a machine verdict on *"is my target served?"* from the factory's head-ladder + matched comparators + knob-turn receipt, at zero cost to us; NOT-SERVED is a defect attributed to the **encoder** | **immediate, per-consumer, falsifying** |

**What is genuinely by construction (claim these, they are free credibility):**

| Promise | Why it is a law, not a hope |
|---|---|
| **Additive/aggregate targets** (counts, sums, recency/frequency/value style features) | the mass pathway is exact-additive and never decayed/gated/normalized (DEC-401/402) ⇒ every additive aggregate of the window is **linearly recoverable** — a whole commercially important target class is covered by construction, with the linear shipping head (DEC-407) |
| **Consistency under chunking / time-translation** | merge exactness + equivariance receipts ⇒ a consumer reading a cut at `t` gets the same answer whether the history was streamed or batched |
| **Usable distances** for kNN-style consumers | geometry gates (isotropy, effective rank, decay) live on the **artifact**, so similarity is certified, not rediscovered per consumer |
| **Refusal over clamping** | out-of-support → abstain with widened uncertainty (axiom), so an unaccommodated consumer is told so rather than misled |

**What we cannot promise (stated in the artifact's metadata, not in a footnote):** coverage is over
**functions of the observed stream**. Unlogged events, hidden confounders, and covariates never
entering the stream are carried as **uncertainty**, never faked. A consumer needing them is not
covered by any encoder — it is covered by getting data.

**Honest disclosures required with E:** (1) v1 measures leg 2 over a *declared* family and (if the
operator approves the stub below) hunts with a cheap class, so the v1 card's wording is "**no
unserved target found among battery ∪ registered consumers ∪ stub-hunter**, as of receipt/commit"
— which is strictly weaker than "all consumers," and must be printed that way; (2) the full
task-hunter stays a **Layer-1** item (DEC-390), disclosed as an open frontier row; (3) the word
**"universal" never appears in a generated artifact unless the coverage rows are green.**

Ruling pending (recommend **E**, plus the **cheap stub hunter in v1**: fixed random projections + a
few programmatic functionals of the future, self-supervised, GPU-cheap — so C5's row is non-empty
on day one). Note the stub is a *receipt addition*, not a gate change, but it touches DEC-390's
deferral, so it is named for explicit approval.

### C10 — Decision & RL consumers — **re-scoped (DEC-409): a decision-PLATFORM contract, not an RL design**

Operator correction accepted: `white_queen` is **pure offline RL** and is its own mini-project,
exactly like `red_king`. The spine does not own RL internals. It specifies only **(a)** what a
decision consumer may **read**, **(b)** what it must **emit**, **(c)** the **safety envelope** it
may not leave. Algorithm choice — offline OPE, counterfactual modeling, future online RL — is
internal to each mini-project, which carries its own gate and receipts. All are peers: each reads
the published artifact, each writes to its **own** namespace, none may touch B (DEC-405).

**Invariants adopted now** (cheap to honor later, expensive to retrofit):

1. **No gradients upstream, ever.** An agent owns its head, its replay, its policy state.
2. **Everything artifact-mediated** — decisions, transitions, evaluations are tables; no agent
   calls B and B never calls an agent.
3. **Agents pin the encoder version** (hash + commit). The freshness law therefore applies to
   *policies*: **a re-fit of B invalidates every policy trained on its embeddings** — they go
   UNKNOWN and must be re-derived and re-certified. Named here because it is the operational
   hazard most likely to be discovered in production instead.
4. **Guardrails live outside the learner** (a learner cannot police itself): shadow mode,
   per-customer caps, budget caps, kill switch, rollback to the logging baseline — which is also
   the standing rule *never ship worse than logging*.
5. **Simulator-first**: where a certified world model exists (`red_king`, validated on known
   effects), policy learning happens there before any live exposure; live exploration is
   offline-first and requires explicit operator approval.
6. **One objective**: discounted **incremental** gross margin — causal lift, never correlated
   margin.

**Open items — parked as their own arc, not silently dropped:** (1) the **decision/transition log
schema** (`entity_key`, `as_of`, artifact hash, policy version, action, propensity,
reward-at-time) — the learning data currently has no home; (2) **actuation feedback** (was the
action delivered, and when) — without it `(s,a,r,s')` cannot be closed honestly; (3) **incremental
reward online** — incrementality requires randomization or a certified simulator, so
online/explorative learning is gated on the randomization budget rather than on the agent's
sophistication; (4) **invalidation & re-certification workflow** for policies on encoder bump.

The tier ladder (offline analysis → certified offline policy → constrained online learning →
approved randomized exploration) is retained as the **evidence ladder** — how much proof a
deployment needs — and is explicitly *not* a design of RL internals or of which algorithm a
mini-project must use.

### The Outlet Standard — what "plug in and be powered" obligates us to build (presented 2026-10-27)

Operator's intent, adopted as the C-series capstone: **any plugin plugs into the wall outlet and
is powered and works.** An outlet achieves that with five properties, none of which is
"the utility knows your appliance." Translated into obligations, with the ruling that already
guarantees each:

| Clause | The outlet property | Our obligation | Guaranteed by |
|---|---|---|---|
| **O1 Shape** | the socket is a standard, not a bespoke harness | one published schema: `entity_key × as_of` + named bucket columns + faceplate; never a function signature | DEC-405 |
| **O2 Power** | voltage delivered to spec, at every socket | the content is there **and recoverable by a simple plug** (linear head / kNN): the coverage lattice's content leg + geometry gates on the artifact | DEC-407, DEC-410 |
| **O3 Tester** | you find out in one second, alone | a **conformance kit** ships *with* the artifact: schema check, residency check (scorer runs from the table alone), knob-turn (zero/permute a bucket → does the score move?), head-ladder, both comparators → verdict **SERVED / GAP-accessibility / GAP-sufficiency / NOT-SERVED** | DEC-408 (mechanisms), **new as a shipped kit** |
| **O4 Spec sheet on the faceplate** | no phone call to the utility | the artifact self-describes: version + hash + commit, column/bucket inventory, measured geometry stats, freshness bound, pointer to the served-target registry, and the non-promise text (functions of the observed stream only) | DEC-405 rule 4, DEC-410 |
| **O5 Breakers** | it fails safe instead of burning the house | out-of-support → refuse with widened uncertainty; stale → UNKNOWN; uncertified → advertised as uncertified | axioms, DEC-390, **C4 open** |
| **O6 The wall is live** | outlets are on when you need them | a **published cadence** and a visible "last good publish" stamp — a dead outlet is worse than no outlet | **open (serving & refresh arc)** |
| **O7 Adapters, not rewiring** | weird plugs are legal, walls are not rebuilt | `custom` consumers and the parked receipted mechanisms are adapters: same gates, labeled, and never a per-consumer wall | DEC-408, DEC-390/398 |
| **O8 Anyone may plug in** | no permission slip | **read access is unconditional; certification is opt-in** — registration exists to be *judged*, never to be *allowed* (oblivious producer, thousands of consumers) | DEC-405, DEC-406; **new as a stated clause** |

**"Works," defined mechanically** (otherwise the sentence is a slogan): a conforming plug whose
target is a function of the observed stream receives (a) a **verdict** from the kit, (b) if
served, performance within a bounded gap of the **full-history oracle** at its declared capacity,
and (c) if not served, an **attributed cause** — accessibility gap (our trunk's defect) or
sufficiency gap (the information is not in the window) — never a silent shrug.

**What a wall cannot do, stated to every plug-in author:** it cannot make a broken appliance work
(a plugin that ignores as_of discipline still fails its own gate, and the card says so), and it
cannot power a device that needs gas (a target requiring unlogged covariates is a data problem,
not an encoder problem — the faceplate says so up front, so nobody discovers it after building).

Ratified (DEC-412). O3's shipped conformance kit and O8's unconditional-read clause are new;
the rest restate ratified law as one promise. **O6 (the wall is live) is the sole undesigned
clause → the serving & refresh arc.**

### O6 — Keeping the wall live (serving & refresh) — **RATIFIED (DEC-414)**

Operator inputs: stale = *"when the embedding fails to do its job anymore"*; 30–90 days between
re-fits is tolerable but cadence must be **measured from performance degradation, not static**;
and training/retraining/inference should be **as cheap as possible**, with shortcuts welcome if
they change the answer.

**The insight that reconciles them: freshness of embeddings and re-fitting the encoder are
different problems.** Conflating them is what makes refresh expensive.

| Clock | What it does | Cost | Exactness |
|---|---|---|---|
| **A — scoring** (continuous) | publish embeddings for **changed** customers on the **frozen** encoder | O(1) per tick; unchanged customers never recomputed (content-addressed by input-window hash) | **exact** — O(1) incremental state (DEC-395) + merge law (chunk-exact), so delta scoring is *not* an approximation |
| **B — re-fit** (rare) | re-train the encoder when the **degradation detector** fires | expensive, event-triggered | champion/challenger; never ship worse than the champion |

**Degradation detector — "fails to do its job," measured per consumer leg:**

1. **Label-free tier (cheap, early).** Re-run the **self-supervised battery** on fresh data: if
   held-out SSL losses drift beyond the measured noise floor, the encoder is going stale *before*
   any label proves it. Companions: input drift on the S1/S2 statistics already collected,
   **unseen-value mass** (T2), geometry drift (isotropy / effective rank).
2. **Label tier (confirms, per leg).** A **backtest ledger**: every published cohort is scored at
   `as_of` and resolved when labels mature → an honest time series of the brain-vs-PATH-A
   advantage *by cohort age*. Stale = the advantage decayed by more than **N × measured SE** (the
   convergence rule of "plateau within measurement noise", aimed at time instead of steps).

Staleness is **per-leg and per-bucket**, never global (churn may be live while value is dead). A
fired leg renders **UNKNOWN, never PASS**, and the faceplate publishes the verdict, last-good
publish, next check, and which trigger fired. Serving a refused artifact is a **refusal**, not a
degraded pass.

**No static interval.** The backstop is derived by fitting the **utility-decay curve** (metric vs
cohort age) from the backtest ledger across historical re-fits — the backstop is the age at which
the trigger would have fired anyway. The operator's 30–90 days is recorded as the **outer
envelope**, not the schedule; until enough re-fits exist to fit the curve, a conservative derived
fallback runs **with a fallback-fired receipt**.

**Cost law — "exactness is free; approximation must be bought with a receipt."**
*Exact, adopted outright:* delta scoring from state, content-addressed caching, batch-everything
GPU execution (DEC-389), O(1) serve. *Certified shortcuts (they change outputs):* **warm-started
re-fits** (validated against a **cold-start control** on the card, with periodic cold starts so
successive warm starts cannot accumulate drift), bf16/fp16 within stated tolerance, sampled
re-fit cohorts, partial-battery re-training (full battery is the certified fallback). Every
speedup is benchmarked before adoption and keeps the slow-but-certified path as fallback.

Cost budgets are **derived from data + hardware**, and the concrete re-fit compute ceiling is
presented for **authorization** at implementation go-time (material spend).

### C-series — CLOSED (2026-10-27), one open arc

| # | Decision | Status |
|---|---|---|
| C1 | Read surface | **LAW** — the published artifact *is* the interface; no ability to talk back (DEC-405) |
| C2 | Heads | **LAW** — linear ships; kNN is the unsupervised product; MLP diagnoses; head-ladder receipt (DEC-407) |
| C3 | Firewall | **LAW** — incl. no derived table as an event source (DEC-412) |
| C4 | Gates | **LAW** — computed not claimed; frozen thresholds; uncertified advertised (DEC-412) |
| C5 | Unknown-consumer guarantee | **LAW** — the coverage lattice; "universal" retired until green; v1 stub hunter (DEC-410) |
| C6 | Version & compatibility | **LAW** — behavior-versioned, additive-only, deprecation ledger, geometry epoch (DEC-412) |
| C7 | Personalization | **LAW** — adapter heads, derived support floor, fallback with widened uncertainty (DEC-412) |
| C8 | The factory | **LAW** — spec-interpreted engine + gated `custom` escape (DEC-408) |
| C9 | Unsupervised instruments | **LAW** — dumb set, derived counts, table/random/destroyed comparators (DEC-408) |
| C10 | Decision & RL | **open arc** — decision-platform contract + 6 invariants; 4 items parked (DEC-409) |
| — | **The Outlet Standard** | **LAW** — O1–O8; "works" defined mechanically (DEC-412) |
| **O6** | Serving & refresh | **LAW (DEC-414)** — two clocks; degradation-triggered re-fit; exactness-free-of-cost |
| — | Cleanup owed | backward imports (B→white_queen ×5, C→B ×3 incl. `rehearsal → update_day`); leaf `contracts/` module |

## 7. Progress mechanics (DEC-390) — the card IS the judge

**Card top line (the judge):** four consumer legs × two numbers —
**brain vs PATH A = progress** · **brain+ledger vs PATH A = delivery**, each with paired
CIs and random/destroyed controls. Beneath: the stage ladder, states derived only from
receipts. Unmeasured or stale rows render **UNKNOWN, never PASS**.

**Freshness law (binding):** any behavior change to the encoder or serving path marks
dependent receipts stale → UNKNOWN until regenerated. One command regenerates the card at
a recorded commit with hashes.

**Encoder done (v1), mechanically:** (1) all six stage states working by receipt;
(2) all four consumer legs green vs PATH A (paired CIs + controls); (3) card fully fresh,
one command, recorded commit + hashes; (4) capacity certified — retention-vs-size curve
measured, k_min recorded (settles the width question with data). Adversarial task-hunter
deferred to Layer 1 **by design, disclosed on the card** — layering, not a softened gate.

When B doesn't beat A, this ladder names the bottleneck stage; each state moves only on a
receipt. "Mastered" = all six working.

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

### Decision rights (DEC-413) — who decides what

The operator supplies **principles and context**; the agent aligns the solution to them (AGENTS.md
operating contract). Since the operator will usually take the agent's recommendation, the check on
self-deception must not depend on the operator evaluating mechanisms — so the *questions* change:

| Class | Who decides | Examples |
|---|---|---|
| **[PRINCIPLE]** | **operator only** | principles, the objective, what "good" means for the business, risk appetite, live customer exposure, whether to continue or stop |
| **[AUTHORIZATION]** | **operator's explicit word** | push, destructive deletes, material spend/compute, anything customer-facing, any live exploration |
| **Gate-touching** | **operator only, always surfaced** | any change to a gate/threshold/criterion/frontier definition, anything touching the judge or frozen definitions, any FAIL→PASS movement, any completion claim ("SOLVED", "universal", PASS language) |
| **[TECHNICAL]** | **agent decides**, citing the principle served, logged in `DECISIONS.md` with rejected alternatives | architecture, mechanisms, derived thresholds, library choice, experiment design, sequencing — **recommendation adopted unless the operator objects** |

Every question is tagged. The operator's veto on **[TECHNICAL]** items is *principle-fit* ("this
doesn't fit the principles"), which is evaluable without domain expertise — not mechanism review.
Multi-option tables are reserved for genuine forks touching the objective or risk; routine rows are
presented as **decision · principle served · what was rejected and why**. Silence is never treated
as assent for any gate-touching or authorization item.

### Implementation order (recorded; **awaiting operator "go" — no code until then**)
One axis per loop; each step = one commit chunk + card regeneration; GPU work follows
DEC-389 (benchmark receipt per adoption, CPU fallback kept). `PLAN.md` gets reconciled to
this order when implementation opens.

| # | Step | Why this order |
|---|---|---|
| 0 | **Environment rebuild** — torch aarch64/CUDA-13, xgboost-cuda, numpy; runtime-receipt smoke test on the GB10 | ✅ **done 2026-10-28** (DEC-416): torch 2.14.1+cu130 on the GB10, compute cap 12.1, warm bf16 matmul ≈15.2 TFLOPS; `scripts/setup_env.sh` + `scripts/env_receipt.py`. One operator action outstanding: `python3-dev` for `torch.compile` |
| 1 | **Judge-first card skeleton** — two-number × four-leg top line, freshness law, UNKNOWN default (existing receipts feed it; missing rows render UNKNOWN) | ✅ **done 2026-10-28** (DEC-416): `scripts/judge_spec.py` + `scripts/judge_card.py` → `JUDGE_CARD.md` (0/8 legs, 0/6 stages, NOT DONE); 18 tests prove it cannot be talked into a PASS; freshness enforced via `behavior_hash()` |
| 2 | **S1/S2 receipts** — renamed-schema + new-field pass, resolution receipt, unseen-value path, transform receipts | cheapest; proves M1/M2 claims are real |
| 3 | **S3 compliance receipts** — homomorphism/equivariance/mass-additivity on data; τ-from-spectrum; stability step 0 | ratifies DEC-386 with paper |
| 4 | **Battery completion** — distance-regression + action-conditioned outcome + ledger-reconstruction; naive comparators; prune pass | the two known gaps; expected biggest progress jump (DEC-387) |
| 5 | **`response` bucket + decodability contracts** — build bucket (DEC-384 disciplines); per-bucket ρ + accessibility parity (DEC-388), tolerances frozen pre-run | completes bucket set + pays the frozen tax |
| 6 | **Ledger companion formalized** — G1–G5 as code-enforced tests | DEC-383 made mechanical |
| 7 | **Capacity curve** — retention-vs-d sweep → k_min certified; settle d=128 (derived or replaced) | DEC-390 done-item (4); answers Feature 7's width question |

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
- 2026-10-27 s10 — operator: **Feature 5 = A ratified (DEC-386)** with both safeguards
  (learned-from-spectrum timescales; ansatz mismatch surfaces as a receipt).
- 2026-10-27 s11 — operator: **Feature 6 = A ratified (DEC-387)**, demands ALL objectives
  (battery pinned as complete-by-derivation against the six content classes, S5 detail
  table above) + all three discipline conditions. Feature 7 (frozen tax) options
  presented: A zero-tax ship-as-is, B accessibility-parity guarantee (recommended),
  C overbuild width (measure, don't doctrine), D adapters (ruled out by A3 lock).
- 2026-10-27 s12 — operator: **Feature 7 = B ratified (DEC-388)** (accessibility-parity
  guarantee + decodability contracts; tolerances frozen pre-run; width question deferred
  to the capacity curve). **GPU-first directive issued (DEC-389)** — hardware verified as
  NVIDIA GB10 / DGX Spark; policy recorded, execution sequenced post-structure. Feature 8
  (receipts & progress) presented — the final feature.
- 2026-10-27 s13 — operator: **Feature 8 = C ratified (DEC-390)**. **Structure phase
  CLOSED.** All eight features decided: constitution (axioms/maxims/overmatch), four
  consumers + judge (B > A), S1–S6 spine, buckets + ledger companion + `response` bucket,
  split-as-law engine, derived battery, accessibility guarantee, judge-first card with
  freshness law and encoder-done (v1). Implementation order recorded (§8), **awaiting
  operator "go"** — step 0 is the CUDA environment rebuild. DEC trail: DEC-383…390.
- 2026-10-27 s14 — operator opened the **internals arc (T-series)**: map of internal
  decisions between stream and state pinned (T1–T11 with statuses; laws marked).
  T1 (token unit) options presented, recommendation A (one event = one tick). Pending:
  T1 ruling, then T2 vocab policy, T3 numerics, T4 missing/unknown/zero semantics.
- 2026-10-27 s15 — operator: **T1 = A ratified (DEC-391)**; **canonical schema recorded**
  (`entity_id | event_ts | event_type | event_attributes`, contract column). T4
  (missing/unknown/zero) options presented A–D, recommendation B (explicit presence
  semantics, data-pruned). Pending: T4 ruling; then T2, T3, T6.
- 2026-10-27 s16 — operator: **T4 = B ratified (DEC-392)** (three-way presence,
  data-pruned; unknown-vs-zero accessibility gate added). T2 (vocab policy) options
  presented A–D, recommendation B (measured head + hashed tail + id-typing + unseen
  discipline with coverage/collision receipts). Pending: T2 ruling; then T3, T6.
- 2026-10-27 s17 — operator: **T2 = B ratified (DEC-393)**. T3 (numeric encoding)
  options presented A–D: recommendation C (measured monotone transform, quantile/spline,
  frozen in artifact; unit-invariance law test; zero atoms) with B (signed log1p) as the
  documented fallback. Pending: T3 ruling; then T6 (chunking/sequence construction).
- 2026-10-27 s18 — operator: **T3 = C+B ratified (DEC-394)**. T6 (sequence construction)
  presented: recommendation B (chunked streaming with exact composition via the merge law;
  hardware-derived token budgets; derived anchors/checkpoints). Pending: T6 ruling; then
  T7 (timescale allocation), T10 (bucket membership), T8/T11 micros.
- 2026-10-27 s19 — operator: **T6 = B ratified (DEC-395)**. T7 (timescale allocation)
  presented: recommendation B (measured spectrum peaks allocate channels; τ peak-initialized
  then learned), fallback C (geometric ladder + noise-calibrated pruning). Pending: T7
  ruling; then T10 (bucket membership), T8 (nonlinearity placement), T11 (mask rates,
  horizons, weight-learning mechanism).
- 2026-10-27 s20 — operator challenged B-vs-D on T7 (maximal adaptivity); answered in
  open (start position vs learning; gradients can't teach what the channel can't hear;
  collapse failure mode) — recorded in DEC-396. **T7 = B ratified (DEC-396)** with C
  fallback. T10 (bucket membership) presented: recommendation B (content class +
  earned-keep; `heads` provisional; `response` separate by discipline boundary).
- 2026-10-27 s21 — operator: **T10 = B ratified (DEC-397)**. T8 (nonlinearity placement)
  presented: recommendation A (bend at write — gates + bilinear content×state — and at
  readout — learned positive weights; fade between events stays exactly linear; nonlinear
  decay only as a receipted second mechanism; batch-coupled bends banned absolutely).
  Pending: T8 ruling; then T11 (last derived knobs) closes the internals arc.
- 2026-10-27 s22 — operator: **T8 = A ratified (DEC-398)**. T11 (mask rates / horizons /
  objective weights — last internals feature) presented: recommendation A (all three
  derived from measured statistics at fit time; C = search-on-held-out-SSL as documented
  fallback). Pending: T11 ruling → internals arc closes; T9 capacity already deferred to
  the retention-vs-size curve (DEC-390).
- 2026-10-27 s23 — operator: **T11 = A with C fallback ratified (DEC-399)**.
  **INTERNALS ARC CLOSED** — T-series complete (T1–T8, T10, T11 law; T9 deferred to the
  capacity curve by design). DEC trail now DEC-383…399. Remaining table items: consumer-
  side contracts (plugins/RL/simulator interfaces), serving & refresh policy (incremental
  updates, drift, retrain cadence), GPU work-mapping per component, Layer-1 certification
  (U-metric/adversary), and the implementation go/no-go (step 0 = CUDA env rebuild).
- 2026-10-27 s24 — operator asked whether **actual encoder architecture** (SSM vs
  transformer, 2 vs 1, gating) had been ruled. Answered honestly: the *family* is law
  (DEC-386), the *microarchitecture* is not. **A-series opened** (A1 block anatomy · A2
  depth · A3 gates · A4 normalization · A5 residuals · A6 attention's place). A1 presented
  (recommend B). Mamba positioned as closest prior art, not adopted wholesale: we take Δt
  from the clock (Mamba learns it) and never decay the mass pathway (Mamba decays all);
  Mamba-1-vs-2 kernel formulations demoted to **implementation** choices settled by GB10
  benchmark receipt (DEC-389), not design debate.
- 2026-10-27 s25 — operator challenged **"why 2 blocks, why not 5?"** Real ambiguity
  caught and resolved in the open: **kinds-per-round vs rounds are separate dials** —
  A1's "2" is the derived count of mixing *axes* (time, channels; a third kind does not
  exist, so any "more blocks" request is really depth), while rounds = A2 depth, measured.
  Also named out loud: the card's `layers 2` is **inherited from screening, pending
  re-derivation** (same disease as `d 128`). **A1 = B and A2 = A2a ratified (DEC-400)**.
  A3 (gates) presented: recommendation B — full lawful gate set (event-locked persistence
  gate with exact-exponential decay between events so the merge law survives; τ bands from
  spectral bootstrap uncertainty; per-layer read gates; mass pathway never gated; measured
  gate init; step-0 stability receipt). Pending: A3 ruling; then A4–A6 close the arc.
- 2026-10-27 s26 — operator: **A3 = B ratified (DEC-401)**. A4 (normalization placement)
  presented: recommendation B — per-sample norms (RMSNorm-style) admitted on bend-path
  inputs only (channel-MLP input, gate projections), never on state/mass/readout, any
  rescaling paired with a parallel unnormalized magnitude lane, each norm entering only on
  a training-instability receipt; state-normalizing SSMs rejected (voids merge exactness,
  destroys level); batch-coupled norms already closed. Pending: A4 ruling; then A5
  (residuals), A6 (attention's place).
- 2026-10-27 s27 — operator: **A4 = B ratified (DEC-402)**. A5 (residuals) presented:
  recommendation B — channel-space skips (around the channel-MLP, layer-to-layer) plus
  per-layer read taps, every skip carrying a learned neutral-init (zero-contribution) gain
  so a deeper model *contains* the shallower one at init — **depth fairness**, which is what
  makes the A2 depth sweep honest. Rejected-as-illegal: skipping around the recurrence
  (algebraically `(A+I)·s_{t-1}` — an un-faded second copy of history: falsifies measured τ,
  endangers contractivity); dense concat readout parked as a receipted duplicate of A3's
  read gates. Noted lock with DEC-402: the mandatory unnormalized magnitude lane IS the
  residual path. Pending: A5 ruling; then A6 (attention's place) closes the A-series.
- 2026-10-27 s28 — operator asked to understand **B vs D on residuals** (skips+read taps vs
  dense concat). Answered in the open: add-in-place vs keep-everything; D's only genuine
  advantage is *separable* per-source access (per-source linear maps vs scalar-gain sums),
  which is why it is parked with a **pre-registered admission trigger** rather than banned;
  D also breaks the depth-fairness nesting property and grows wiring with depth² and served
  width with depth × width. **A5 = B ratified (DEC-403)**. A6 (attention's place) presented:
  recommendation B — read-side only, **resident objects only**, with the residency law
  (a read path may read only what the fixed-size state carries; receipt = readout re-implemented
  against the frozen artifact alone, bitwise-equal), positive weights, attention weights as
  diagnostics. Pending: A6 ruling → **A-series closes**.
- 2026-10-27 s29 — operator: **A6 = B ratified (DEC-404)**. **A-SERIES (MICROARCHITECTURE)
  CLOSED** — A1 two-subblock recurrence+MLP · A2 depth measured · A3 full lawful gate set ·
  A4 per-sample norms on bend paths · A5 channel-space skips w/ neutral-init gains · A6
  read-side attention under the residency law. Kernel formulations demoted to implementation
  (GB10 benchmark receipt). DEC trail now DEC-383…404; 45+ commits. Remaining arcs: consumer-
  side contracts (plugins/RL/simulator), serving & refresh policy, GPU work-mapping per
  component, Layer-1 certification (U-metric/adversary), implementation go/no-go (step 0 =
  CUDA env rebuild).
- 2026-10-27 s30 — operator chose **Arc 1: consumer-side contracts**. **C-series opened**
  (C1 read surface · C2 permitted head classes · C3 firewall rules · C4 per-consumer gates ·
  C5 unknown-consumer guarantee · C6 version & compatibility · C7 personalization).
  C1 presented: recommendation B — declared read-set verified by a single typed accessor, with
  the brain-credit rule (a card row is "brain" iff its read-set excludes ledger buckets) making
  DEC-383's two-number card mechanical rather than narrated, and freshness scoped to the
  declared read-set. Pending: C1 ruling.
- 2026-10-27 s31 — operator overrode my C1 draft with a stronger framing: **plugins are dumb
  consumers; they must not have the *ability* to talk to B; there may be thousands of them**;
  write embeddings to a table, plugins consume directly. First draft's "typed accessor" gate
  **withdrawn** (anything in the consumption path becomes a dependency; fan-out makes
  per-consumer coupling fatal). **C1 ratified as DEC-405**: artifact = the interface; imports
  banned both directions; immutable content-addressed read-only output; self-describing
  artifact so thousands of uncoordinated consumers can self-validate and locally refuse;
  producer-side materialization of derived cuts; attribution by the judge's read-only
  observations (manifest + static column extraction + knob-turn receipt). Repo audit found the
  rule already broken (B→white_queen ×5; C→B ×3 incl. `rehearsal.py → update_day`, a consumer
  that can produce its own input) — cleanup owed as one work item. Pending: C2 (head-class
  ceiling — what bounds "small MLP").
- 2026-10-27 s32 — operator issued the **Layer C charter** (DEC-406): supervised plugins
  **spawned from a template** given essentially just the target (the published artifact already
  *is* the supervised dataset); unsupervised the same; **sklearn-level quality and simplicity**;
  `white_queen` as the primary RL instrument with **online RL admitted**; "all of Layer C is
  dedicated to creating plugins given this encoder." Recorded as a **certified factory +
  registry** (not bespoke scripts). C8/C9/C10 presented: factory = spec-interpreted engine +
  gated custom escape (B+C, rejecting generated-code and free-form review); unsupervised =
  dumb instrument set with derived counts and table/random/destroyed comparators; RL = tiered
  ladder T-0 offline → T-1 certified offline policy → T-2 constrained online learning → T-3
  live randomized exploration, all under the single discounted-incremental-margin objective and
  with gradients never reaching B. Pending: rulings on C2, C8, C9, C10.
- 2026-10-27 s33 — operator rulings. **C2**: recommendation accepted with the operator's
  amendment "only do 1 of the trio, happy to test" → derived to **linear as the sole shipping
  supervised head**, kNN reassigned as the unsupervised product, MLP demoted to the diagnostic
  tier, plus a **head-ladder receipt** (all three reported per target; MLP beating linear = an
  accessibility gap against the *encoder*, never permission to ship it) as the test **(DEC-407)**.
  **C8** ratified (spec-interpreted engine + gated `custom` escape) and **C9** ratified
  **(DEC-408)**. **C10**: operator corrected the scope — `white_queen` is pure offline RL and is
  its own mini-project, like `red_king`; standard online RL must be supported but the enabling
  system is unknown → **re-scoped to a decision-platform contract**: six invariants adopted now
  (no gradients upstream; artifact-mediated; agents pin the encoder version so a re-fit
  invalidates every policy; guardrails outside the learner; simulator-first; one
  incremental-margin objective) and four open items parked as their own arc — decision/transition
  log schema, actuation feedback, online incrementality (gated on randomization), and policy
  invalidation workflow **(DEC-409)**.
- 2026-10-27 s34 — operator chose **C5 (the unknown-consumer guarantee)**. Presented with the
  honest premise that **no mechanism proves** "covered by construction, not by enumeration":
  enumeration is a benchmark, the algebra guarantees *form and consistency* but not *content*,
  the `U` metric measures only over its declared family, and an adversary inherits its own blind
  spot. Recommendation **E — the coverage lattice**: (1) form = proven by construction;
  (2) content = measured over a **published** family; (3) adversary = hunted, class disclosed;
  (4) **boundary falsification** = every consumer gets a free machine verdict on "is my target
  served?" with NOT-SERVED attributed to the encoder. Plus a **by-construction subset** we may
  claim truthfully (additive/aggregate targets via the exact-additive mass pathway + the linear
  shipping head; chunking/time-translation consistency; certified geometry for distances;
  refusal over clamping) and an explicit non-promise (coverage is over **functions of the
  observed stream**; unlogged events carried as uncertainty). Rule pending on E and on the
  **v1 stub hunter** (random projections + programmatic future functionals) that keeps the row
  non-empty — a receipt addition touching DEC-390's deferral, so flagged for explicit approval.
- 2026-10-27 s35 — operator: **C5 = E ratified, stub hunter approved for v1 (DEC-410)**, and
  adopted the metaphor as the standard: *"I want ANY plugin to plug in to my wall outlet and be
  powered and work."* Presented **The Outlet Standard (O1–O8)** as the C-series capstone:
  O1 shape (schema, not signature) · O2 power (content recoverable by a simple plug) · O3 a
  **shipped conformance kit** giving SERVED / GAP-accessibility / GAP-sufficiency / NOT-SERVED in
  one second (new) · O4 faceplate metadata · O5 breakers (refuse, never burn the house) · O6 the
  wall is **live** — published cadence + last-good-publish stamp (open, serving arc) · O7 adapters
  not rewiring (`custom`, receipted mechanisms) · O8 **read access unconditional, certification
  opt-in** (new). "Works" defined mechanically: conforming plug + target measurable on the
  observed stream ⇒ a verdict, and if served a bounded gap to the full-history oracle at declared
  capacity, and if not served an attributed cause. Plus the two honest limits: the wall cannot fix
  a broken appliance or power a device that needs gas. Pending: ratify O1–O8 and close C3/C4/C6/C7.
- 2026-10-27 s36 — operator: **O1–O8 RATIFIED (DEC-412)**, which also closed **C3** (firewall +
  the new data-mediated leakage law: no derived/consumed table may be an event source), **C4**
  (gates computed by the card, never self-declared; frozen thresholds; uncertified advertised),
  **C6** (behavior-versioned, additive-only, deprecation ledger, geometry epoch) and **C7**
  (personalization = adapter heads at derived capacity, derived support floor, fallback with
  widened uncertainty). **The C-series is CLOSED** — one open arc (C10's four parked items) and
  one undesigned clause (**O6 — the wall must be *live*** → serving & refresh arc).
  Operator also stated their working mode: principles and context are theirs, technical rulings
  are mine, and they will usually take my recommendation. Named the governance risk out loud (this
  project was twice overturned on a false "SOLVED") and fixed it by changing **what gets asked**
  rather than asking more unevaluable questions: **decision rights logged as DEC-413** —
  [PRINCIPLE] / [AUTHORIZATION] / gate-touching items stay the operator's and are *always*
  surfaced; [TECHNICAL] items I decide, cite a principle, log with rejected alternatives, and are
  adopted unless vetoed on principle-fit.
- 2026-10-27 s37 — operator answered the three **[PRINCIPLE]** questions on O6: stale = *when the
  embedding fails to do its job anymore*; 30–90 days between re-fits tolerable but cadence must be
  **measured from performance degradation, not static** (their day-job's static 30-day cycle was
  explicitly rejected as a model); and train/retrain/infer **as cheap as possible**, shortcuts
  welcome if they change the answer. **O6 ratified as DEC-414** on the reconciling insight that
  **embedding freshness and encoder re-fit are different problems**: Clock A = continuous, exact,
  nearly free delta scoring of changed customers on the frozen encoder (O(1) + merge law);
  Clock B = rare, event-triggered re-fit fired by a **two-tier degradation detector** (label-free
  SSL-battery + drift signals first, matured-label backtest ledger confirming), with staleness
  per-leg, UNKNOWN-never-PASS, a **derived** backstop from the fitted utility-decay curve (30–90d
  recorded only as the outer envelope), and the cost law **"exactness is free; approximation must
  be bought with a receipt"** (warm starts, quantization, sampled cohorts and partial-battery
  retraining each need a certification receipt against a cold/full control). Design phase is now
  complete: T · A · C series closed, O1–O8 law, O6 law; remaining = GPU work-map (fused into the
  implementation plan) and the parked C10 items. Next: **implementation go/no-go**
  ([AUTHORIZATION] — step 0 rebuilds the CUDA env on the GB10).
- 2026-10-28 s38 — operator: **GO**, on a new branch: "what you are doing now will replace the core
  of the project when it is merged back in." Branch **`foundation-rebuild`** created (DEC-415);
  "replaces the core" interpreted as a **deletion obligation** — superseded paths get deleted when
  the new judge is green, not kept as a parallel implementation.
  **Step 0 done (DEC-416):** `.venv` on the GB10 with torch **2.14.1+cu130** (CUDA 13.0, compute cap
  12.1, warm bf16 matmul ≈15.2 TFLOPS — the first unwarmed read said 0.03, a cold-start artifact, so
  the smoke now warms before timing), cupy 14.2, numpy 2.5.4, polars 1.44.2;
  `scripts/setup_env.sh` + `scripts/env_receipt.py` →
  `docs/report_cards/experiments/env_receipt.json`. **Inherited-suite baseline recorded: 512
  collected · 504 passed · 8 failed** — 19 of the original 24 failures were **duckdb 2.0 removing
  `Relation.pl()`** (~30 sites), pinned to `duckdb>=1.5,<2` with a fix-forward item for one
  DuckDB→polars boundary; 7 failures are `torch.compile` failing to build without `python3-dev`
  (**operator action**: `sudo apt-get install -y python3-dev build-essential`) and stay failing
  rather than skipped, because a masked accelerator is the false green DEC-389 forbids; 1 failure is
  a typed `0.01` tolerance drifting, untouched because threshold edits are gate-touching.
  **Step 1 done (DEC-416):** the judge exists before any case is argued — `scripts/judge_spec.py`
  (frozen gates) + `scripts/judge_card.py` → **`JUDGE_CARD.md`**, currently **0/8 legs · 0/6 stages
  · NOT DONE**, with `tests/test_judge_card.py` (18 tests) proving no route to a false PASS (missing
  receipt, stale behavior hash, self-defined criterion, `n/a` comparator, missing N, missing controls
  → UNKNOWN; CI including zero or a control beating the brain → FAIL; degenerate → EXCLUDED).
  Freshness is mechanical now via `behavior_hash()` over the encoder + data-path source trees.
  Next per §8: **step 2 — S1/S2 receipts** (renamed-schema + new-field pass, resolution receipt,
  unseen-value path, transform receipts).
