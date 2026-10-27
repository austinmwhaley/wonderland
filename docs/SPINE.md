# SPINE — the single first-principles structure (living document)

> The one steering document for the foundation-model re-think (started 2026-10-27; merges
> and supersedes `docs/FIRST_PRINCIPLES.md`, whose argument log is folded into §9).
> Nothing here is implemented; this is the structure we are building by conversation.
> The operator reads the chat, not this file — this file is the record the chat produces.

## 0. The goal, one sentence

Input any event stream → output a **frozen** foundation model (one per stream) whose state
supports **supervised**, **unsupervised**, and **RL** heads using simple consumers.

## 1. The three consumers come first (tracing rule — operator ruling 2026-10-27)

Everything in the system must trace to the consumer set as a whole ("no feature without a
trace"): a component exists only if it can name a consumer need it serves or a gate it
satisfies. The full trace table lives in §3.

| Consumer | The question it asks | What it needs from the frozen state |
|---|---|---|
| **Supervised** (predict churn, value, orders…) | "What will happen to this customer?" | The state must **contain** everything about the customer that predicts their future, and a plain line must be able to **read** it |
| **Unsupervised** (segments, lookalikes, anomalies) | "Who is like this customer?" | **Distance = behavioral similarity**: neighbors must genuinely share futures; the *arrangement* of points matters, not just the contents |
| **RL** (what to do: email frequency, offers…) | "What should we do to this customer?" | The customer state **just before** an action + how they respond to actions; only where the logged data actually tested actions |

## 2. The constitution (ACCEPTED 2026-10-27)

**Axioms are gates** — violate one and the run is void (not "low-scoring").
**Maxims are ranked** — between two designs that pass the gates, the higher maxim wins.

### Axioms (gates)
| # | Axiom | The test |
|---|---|---|
| A1 | **Robust** | deterministic; divergence = hard failure (never rescued); reject or widen uncertainty, never clamp; every estimate carries calibrated uncertainty |
| A2 | **Honest** | no downstream labels in encoder training; never trained on the metric it is graded by; fair comparators with paired CIs; reproducible receipts at a recorded commit |
| A3 | **Strict-frozen** *(LOCKED by operator)* | after freeze the encoder is immutable; consumers ≤ {linear, kNN, small MLP}. Consequence: the encoder must pre-adapt to consumers it will never meet — *accessibility is a first-class encoder requirement*, not a consumer problem |

### Maxims (ranked core)
| Rank | Maxim | The test |
|---|---|---|
| M1 | **Universal / generic** | zero domain knowledge in code; any schema resolves from the input; test = renamed-schema + new-field training with zero code edits |
| M2 | **Adaptive / dynamic** | every resolvable quantity measured or learned at fit time; surviving literals are receipted fallbacks only |
| M3 | **Lawful** | correct algebra per quantity; exact symmetries (field order, time shift, sequence-merge); laws rank above efficiency because in this system the laws are the *mechanism* of speed |
| M4 | **Efficient / fast** | linear time, constant state, O(1) incremental update, vectorized; runtime receipts |

### The parameter trichotomy (the daily use of the constitution)
Every parameter lives in exactly one column; a parameter caught in the wrong column is a bug.
| Column | Meaning | Examples |
|---|---|---|
| **Contract** (fixed on purpose) | the rules of the game | the laws, evaluation oracles, label firewall, strict-frozen, per-stream rule, negative controls |
| **Derived** (measured at fit time) | read from the stream | field types, vocab caps, transforms, timescales τ, horizons, mask rates, capacity, loss weights |
| **Learned** (fitted) | the model's own knowledge | embeddings, gates, decays, prediction heads |

### Operator rulings log
- **Q3 LOCKED:** frozen is absolute (§A3). All decisions align to it.
- **Q5:** RL bar for v1 is **decision-useful** — receipted progress (beats table & random
  under known-effect/IPS with CI), not Bellman-completeness (stays a Layer-1 goal).
- **Constitution accepted** (axioms as gates, maxims as ranked core).
- **Consumer-backward tracing** is the organizing direction; single .md ruling.

## 3. The consumer-backward trace (the spine, inverted)

| Layer | Supervised | Unsupervised | RL |
|---|---|---|---|
| Needed property | **Predictive sufficiency** — state determines the customer's future behavior distribution | **Behavioral metric** — embedding distance ≈ similarity of futures | **Decision-usefulness (v1)** — pre-action state supports response models that beat table/random |
| Which view carries it | `shape`, `mass`, `forecast` | `shape` geometry (+ `support` for anomalies) | `pre-action` (+ `shape`) |
| Which stage creates it | S3 state + S5 pressure + S4 exposure | S3 state + S5 **distance pressure (currently missing)** + S4 | S3 state + S5 response pressure + S4 pre-action cut + randomized data (Layer A) |
| Progress gate (receipt) | beats raw & fair table, paired CI; ≥50% of ceiling gap | distance↔truth-similarity correlation beats table/random; kNN label transfer > table | known-effect recovery + IPS beats table & random with CI |
| What breaks if only this leg is served | table-like encoder with no usable neighbors | pretty clusters, blind predictions | confident policies with unverified effects |

Single-object argument (from the argument layer, kept as record): the three properties are
projections of ONE object — the stream's **future-predictive law** (action-conditioned where
the data identifies it). Supervised wants the state to *determine* it; unsupervised wants
*distance* to equal discrepancy between laws; RL wants that law usable under actions.
Hence one frozen state can serve all three — and a failure in one leg is a stage-localized
diagnosis, not a mystery.

## 4. The derivation spine (S1 → S6), each stage forced by a gap

```
 RAW events (time, customer, typed attributes)          — the only given; the system knows nothing
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

| Stage | Gap that forces it (why it cannot be skipped) | Failure if faked |
|---|---|---|
| **S1 Interpret** | No downstream operation is type-agnostic: before anything accumulates or predicts, the system must know whether a field is a count, a rate, a label, or an id — and the *data* must say so (M1 forbids being told) | new schema breaks it, or ids get treated as numbers; "works only on our dataset" |
| **S2 Symbolize** | S3 needs one continuous object per event, but events are typed field-maps; the mapping must preserve each field's algebra or totals become arithmetic on lies. Time is not a feature: the timestamp belongs to S3; binning time is banned (destroys the irregular timing this project is about) | clamped/crashed unseen values; hand-written templates; cadence literals in code |
| **S3 Accumulate** | Fixed-size state that matches full history + constant memory + O(1) updates ⇒ only a state recurrence can do it. *Which* recurrence is derived: if event influence fades over the past (standard event-process family), the sufficient state decays exponentially between events and jumps at events — the continuous-time gated recurrence IS that equation. Architecture = an ansatz about the process class; S5 tests the ansatz | hand-set timescales/width; divergence rescued by "keeping the best earlier state"; mass decayed (level information lost) |
| **S4 Expose** | A3 (locked): sufficiency survives scrambles that kill simple heads, and no consumer may adapt to the encoder ⇒ the encoder must pre-adapt to consumers it never meets. Named views = the equivalence classes the three consumer classes provably need; direction and magnitude readable separately; consumers can abstain | views picked on holdout data; one blended vector (measurably dilutes distance probes); "present but unreadable" sold as sufficiency |
| **S5 Pressure** | Architecture is a capacity claim, not a content claim — nothing enters the state unless a gradient demands it. Only self-supervised objectives are admissible (A2). Accessibility/distance failures feed back here as objective-mix changes, never as clamps | hand-weighted losses; no objective pressures distances (our geometry leg today); training on the graded metric |
| **S6 Freeze** | Without a versioned hash + provenance + per-view negative controls + a regenerable card, every earlier claim is narrative | card text newer than receipts; stale claims |

### What the encoder is NOT
- Not a consumer of labels/downstream tasks (firewall; the wide table is a
  **teacher/baseline**, never resident inside the learned state).
- Not cross-stream: one stream in → one model out; applying to an unseen stream = refuse.
- Not batch-coupled at serving (per-sample contract; no batch-statistics capacity fabrication).
- Not allowed to clamp (out-of-support → abstain/widen).

## 5. The views, and the OPEN measure question

Views = named buckets the frozen artifact exposes (so simple heads can select what they
need instead of drinking from one blended vector).

| View | Plain meaning | Who reads it |
|---|---|---|
| `shape` | the learned "profile direction" — who this customer behaves like | supervised + all distance/neighbor tasks |
| `mass` | learned total/level signals (how much, how big) | value/orders consumers |
| `forecast` | learned future-summary signals | prediction consumers |
| `support` | what the encoder actually saw (gaps, missingness) — lets heads abstain | anomaly; any careful consumer |
| `pre-action` | state cut strictly BEFORE each logged action | RL/response models |
| `measure` | **the ledger**: exact counts/totals computed faithfully from the visible history (N purchases, $ totals, % by channel, first/last activity…) — not learned, just computed | category identity, anomaly (per receipts, ONLY these work here) |

**The `measure` tension (OPEN — under conversation):** the old honesty rule says "no
table/counter inside the model" to prevent two cheats — (1) pre-computing the answer key
into what's being graded, (2) not being able to say whether a win came from the learned
brain or the carried ledger. But receipts show some consumer needs (category identity,
anomaly) work *only* in the ledger view. Proposed resolution (pending operator opinion):
the ledger is legitimate content when it passes three guardrails — **(a) history-only**
(computed from events visible before the as-of cut), **(b) target-blind** (never contains
the predicted quantity's definition), **(c) attribution always reported** (every claim
states which buckets it came from; learned-only + ledger-only + destroyed controls).
Alternative: remove the ledger from the artifact — but then each consumer must compute its
own counts, weakening "the foundation supports everything."

## 6. Progress mechanics (anti-narration ladder)

Stage states move only on receipts. "Mastered" = all six working on target streams.

| Stage | Working-state proof | Not-working looks like |
|---|---|---|
| S1 | renamed-schema + new-field pass with full resolution receipt, zero code edits | any field typed by hand; any cadence literal |
| S2 | unseen-value path exercised; transform receipts; field-permutation invariance ≤ tol | clamped/crashed unseen values; templates in code |
| S3 | merge/equivariance/mass-additivity exact on data; τ receipted from spectra; stable from step 0 | divergence rescued; τ/d hand-set |
| S4 | per-view accessibility parity + per-view negative controls | views selected on holdout; no decodability contract |
| S5 | objective battery beats naive (or excluded+reported); distance-pressure row exists | hand-weighted losses; no geometry pressure |
| S6 | one command regenerates the card at a recorded commit + hashes | card text older than last behavior change |

## 7. Decision protocol (how any change is processed)

1. Which stage (S1–S6) does it touch? **Which consumer(s) does it trace to?** (no trace = no)
2. Which maxim does it serve? Does it violate an axiom? (axiom violation = stop)
3. Which column is its parameter in (contract / derived / learned)?
4. What receipt moves that stage's state? What negative control catches the failure?
5. Record in `DECISIONS.md`; regenerate card; accept or revert.

## 8. Literature anchors (compressed; full citations in git history of FIRST_PRINCIPLES.md)

- Event-stream SSL & foundation models: JAIR 2025 unified-SSL survey (no adopted universal
  ES foundation model exists); MLEM 2024 (no single objective dominates; generative wins
  timing tasks; aligned mixtures most universal); ESGPT; PyTorch-Lifestream IJCAI 2025;
  HT-Transformer; RecSys Challenge 2025 (fixed-vector event autoencoder + normalized view
  concatenation); LLM4ES CIKM 2025; PinnerFormer KDD 2022 (frozen user embeddings at scale).
- Sufficiency & prediction: informative statistics (NeurIPS 1999); predictive information
  (Bialek 2001; Ver Steeg & Galstyan 2015); DAPC; CPIC; slow-feature principle
  (Wiskott & Sejnowski 2002; Creutzig & Sprekeler 2008).
- Control-relevance: state abstraction (Li–Walsh–Littman 2006; Du et al. 2019 Q*/π*-irrelevance;
  NeurIPS 2024 probing-through-abstraction); bisimulation metrics (Ferns–Precup–Pan; Castro
  et al. 2020; robust BSM NeurIPS 2021; RAP NeurIPS 2022; DBC 2019); ATC decoupled
  representations (Stooke 2020); action-sufficient reps (ICML 2022); predictive ≠ control
  sufficiency (Latent State Design under Sufficiency Constraints); frozen-encoder limits
  (response-aware planning, Shang 2026).
- Accessibility/probing: Hewitt & Manning 2019; Vo & Rogers 2020; Probing with Noise 2022
  (norm = separate information container); Present but Not Remembered 2026 (decodable ≠ used).

## 9. Argument log (from the merged FIRST_PRINCIPLES discussion)

1. **Freezing is the defining commitment** — it moves all learning from consumer to encoder;
   presence ≠ decodability (any invertible scramble of a sufficient state is sufficient but
   probe-dead). The object is therefore a pair: content + exposure contract.
2. **Three consumers, three sufficiencies, one generator** (the future-predictive law;
   §3). Decision sufficiency for a rich reward family subsumes prediction; geometry =
   isometry to the metric the law induces.
3. **No free lunch** — "universal" is relative to (a) the stream's real predictive content
   (Bayes floor), (b) capacity (right size ≈ the stream's predictive-information rate —
   measured by a retention-vs-d curve, not tuned), (c) a minimax over an adversarial,
   held-out task family. Task count is decoupled from information content; that is why
   universality is credible at all. "1 stream → 1 model" is an easier, more defensible
   contract than the cross-domain transfer the outside literature keeps failing at.
4. **Freezing strictly costs something** — frozen encoders can provably miss what fine-tuning
   recovers; hence accessibility must be designed + certified (S4/S5 feedback), not hoped for.

## 10. Conversation log

- 2026-10-27 s1 — framing (three sufficiencies; impossibility; U metric continuity).
- 2026-10-27 s2 — operator: Q3 LOCKED (strict frozen), Q5 decision-useful, wants structured
  raw→frozen derivation + consumer-backward tracing + simple language; SPINE v1 written.
- 2026-10-27 s3 — operator: single .md (this file), constitution ACCEPTED, feature-by-feature
  walkthrough begins; first feature = `measure` view (§5), under discussion.
