# PLAN.md — the goal, cut backwards into solvable problems

*Generated from `scripts/plan_spec.py` by `scripts/plan.py` — do not edit by hand.*

> **Read order.** `AGENTS.md` (doctrine) wins over this plan, and over every other file.
> `REPORT_CARD.md` is the only measurement surface: it renders this same tree with the
> frozen rows' current verdicts. `STATUS.md` is live state; `DECISIONS.md` is the log;
> `AUDIT.md` holds the BACKWARD requirements.

## 0. The goal — and why it is the goal

One frozen, self-supervised, schema-agnostic encoder per event stream whose embedding is a universal donor: a bounded per-customer, per-time vector that loses nothing that matters about the future — for prediction AND for decisions — readable by simple consumers, transferable beyond the tasks we know, honest under shift, and certified at fixed capacity against the full-history oracle, so that any present or future consumer is covered by construction rather than by enumeration.

Why this and not something easier:

A per-use-case wide table (PATH A) must be rebuilt for every new task and stream and silently encodes the tasks we already know; it cannot be a foundation. A frozen per-stream embedding amortizes one self-supervised fit across present and future consumers. But 'it works on our tasks' is not universality: the only defensible claim is worst-case sufficiency against the full history (U), with accessibility, transfer and robustness certified separately, because the consumer set is open-ended and known use cases may veto — never shape — the encoder.

## 0.1 Macro review — where the project is

At behavior `7baf744f`, B is certified (23/23 rows, 8/8 sections), B+C is 8/8 leg-columns under the bank protocol (DEC-598), and the three plugin gates pass 9/9 on the bank + demo stream (supervised 6/6, unsupervised 4/4, white_queen 5/5, Layer C 9/9). The served block bank is 32 certified shape PCs + level/response/RFM/action + exact counter + schema profile + numeric/mass blocks, with a compact judge ledger; the supervised consumer is a standardized ridge alpha=10 and the geometry consumer fits non-negative block weights on split A. The v1 flat judge (4/8), the GBM delivery head and the old canonical stream are recorded beside it. The full plan root remains OPEN at E1 (the trunk) and the v1 atoms red_king/red_queen/caterpillar/identifiable-stream, outside the B+B+C+plugins definition.

## 1. How to read this plan — the backward cut

The tree starts at the goal and asks, at every node, *what must be true for this to be
true?* Each answer is a smaller, independently measurable problem, broken down until
the atoms are conditions a receipt — or a single recorded decision — can settle. How
many levels that takes is what it takes: there is no forced depth, and the tree no
longer speaks of sections or leaves. Solving the atoms bottom-up is solving the goal
— but only if every atom is solved, which is why the tree names the ones the card does
not measure yet instead of pretending they do not exist.

- **Every node is a necessary condition of its parent.** Nothing here is an
  implementation preference; each node answers *why the parent cannot hold without it*.
- **The plan defines no gate.** Every measured atomic problem is a frozen row whose gate
  lives in `scripts/b_spec.py` or `scripts/judge_spec.py`, frozen before its result. An
  interior verdict is the worst of its children — the card's existing rule, not a new one.
- **Progress is the tree.** How many problems and atomic problems are solved, read off
  the hierarchy at every level, IS the progress measure: there is no separate progress
  block, and no milestone ladder beside the tree.
- **Three kinds of atomic problem, deliberately distinguished.** `card` atoms are
  computed from receipts; `dec` atoms have bars frozen in a DEC and close only through
  a pre-registered run; `v1` atoms belong to the stack's own gates and are not assessed
  on this card. A green card alone therefore proves T0 — never SOLVED.
- **No relabelling.** A problem moves from open to solved by evidence, never by editing
  this file. Changing what an atom measures is a new protocol version, not a plan edit.

## 2. The tree

The census: **39 atomic problems** (31 measured · 3 DEC-frozen · 5 v1) under **19 interior problems** — progress is the solved count, read off the tree at every level.

### G — A certified universal donor, delivered

*What is the whole problem?*

A per-use-case wide table (PATH A) must be rebuilt for every new task and stream and silently encodes the tasks we already know; it cannot be a foundation. A frozen per-stream embedding amortizes one self-supervised fit across present and future consumers. But 'it works on our tasks' is not universality: the only defensible claim is worst-case sufficiency against the full history (U), with accessibility, transfer and robustness certified separately, because the consumer set is open-ended and known use cases may veto — never shape — the encoder.

**Solved when:** every child below is solved (worst-child rule)

#### P1 — The donor exists

*Is there a per-stream artifact built from the right algebra, and is it an identity?*

Nothing downstream is meaningful about an object that is not lawful or not reproducible: a state update that breaks compositionality cannot be a homomorphism over events, and an artifact that cannot be rebuilt cannot support a claim. This is the floor under every other node.

**Solved when:** every child below is solved (worst-child rule)

##### B1 — Lawful

*Is the state built from the right algebra?*

A state update that breaks compositionality or equivariance is not a homomorphism over events; no sufficiency claim survives it.

**Solved when:** every child below is solved (worst-child rule)

###### B1.1 — Lawful dynamics (S3)

*merge/equivariance/mass-additivity exact on data; tau receipted; stable from step 0*

The merge / equivariance / mass laws are the homomorphism floor: a state update that does not compose exactly over event sequences is not a lawful merge, and no sufficiency claim survives it.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B1.2 — Construction laws (C1-C6)

*scan / field-permutation / mass / determinism / schema variant / frozen identity*

The construction laws (C1–C6) are the v6 algebraic-law pillar in testable form — scan associativity, field permutation, determinism, schema variants, the frozen hash: a construction that breaks one is a transcript, not a certificate.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

##### B8 — Reproducible

*Is the artifact an identity?*

An artifact that cannot be rebuilt at a recorded commit is a transcript, not a certificate.

**Solved when:** every child below is solved (worst-child rule)

###### B8.1 — Artifact identity

*registry tag + config + data revision recorded and consistent*

An artifact whose hash, config and data revision are not recorded is a transcript, not a certificate: identity is the floor on which reproducibility stands.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B8.2 — One-command reproducibility (S6)

*the card is regenerated in one invocation at a recorded commit*

If the card cannot be regenerated by one command at a recorded commit, the certification cannot be re-derived: S6 is the reproducibility claim itself.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

#### P2 — The donor suffices

*Does the embedding lose nothing that matters about the future — for prediction and decision — and is it the smallest object that does?*

A donor that is lawful but lossy is a summary, not a foundation: any consumer would inherit the loss, and no amount of consumer tuning could recover it. Sufficiency is judged worst-case against the full-history oracle, not against our tasks; decision sufficiency extends the claim to the action family; minimality removes the degenerate escape of shipping the history. Necessary for P4: a consumer cannot beat PATH A from an embedding that dropped the signal.

**Solved when:** every child below is solved (worst-child rule)

##### B2 — Sufficient

*Does B lose nothing about the future?*

Sufficiency is the donor contract itself: without U the embedding is an unmeasured summary, and every consumer inherits whatever it dropped.

**Solved when:** every child below is solved (worst-child rule)

###### B2.1 — Worst-case sufficiency U

*U = 1 - worst-case oracle gap vs full history, fixed capacity, broad published family*

U is the donor contract itself: without a worst-case bound on the oracle gap at fixed capacity, the embedding is an unmeasured summary and every consumer inherits whatever it dropped.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B2.2 — Full-history oracle gap

*excess loss of E vs the full-history arm at fixed capacity, per included probe*

The per-probe excess loss vs the full-history arm is the content of the U bound: without it, U is a number with no referent, and 'sufficient' is a word, not a measurement.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B2.3 — Learned-only attribution

*sufficiency of the learned shape vs the served statistics, reported separately*

The sufficiency claim must belong to the learned shape, not the served statistics: without the learned-only attribution, a green U could be the ledger's, not the encoder's (the circular-evaluation guard).

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

##### E1 — Trunk capacity (the state-update algebra)

*Can the state carry the stream's structure — not just its mass?*

A sufficient statistic must have somewhere to live. When the trunk's effective rank collapses, the embedding is a blob and every downstream row measures the blob, not the stream. This is the measured blocker (O3 attempt 4 rejected; the dev rig reproduces it in minutes), and it is the one problem whose bars are frozen outside the card — which is why it is cut into atoms, each with its own DEC-frozen close.

**Solved when:** every child below is solved (worst-child rule)

###### E1.a — next_event mastered on the toy (curriculum step 1)

*Does the frozen trunk beat the 1-gram on the known-truth next_event structure at the dev-rig scale, with clean controls?*

The dev rig is the minutes-scale instrument that localizes the blob; next_event is curriculum step 1, and DEC-576's port protocol needs this structure mastered before any real-data claim.

**Bars frozen in:** DEC-576/577 (dev rig: g<=0.5 and beats the 1-gram, controls clean) — exhausted negative at N=500 across all four single-axis trunk-algebra values plus the dim=256 capacity read (DEC-581/582 outcomes: the 1-gram floor is binding, a finite-sample estimation gap, not a trunk deficiency)
**Solved when:** its pre-registered run meets the frozen bars and the decision is recorded — DEC-576/577 (dev rig: g<=0.5 and beats the 1-gram, controls clean) — exhausted negative at N=500 across all four single-axis trunk-algebra values plus the dim=256 capacity read (DEC-581/582 outcomes: the 1-gram floor is binding, a finite-sample estimation gap, not a trunk deficiency)

###### E1.b — intent mastered on the toy (curriculum step 2)

*Does the frozen trunk beat the 1-gram on the per-customer intent structure, at a scale where the sample can determine it, with clean controls?*

The next_event outcome located the shortfall in the sample, not the trunk: intent at a determinable scale is the structure the trunk must next master (DEC-583; the snr difficulty dial is rejected as an axis).

**Bars frozen in:** DEC-586 (curriculum step 2: the intent structure, one axis, at the frozen difficulty; fixture intent=5.0/br=0.3/N=500/seed=17, frozen DEC-576 bars in corrected units, branches (i)-(v), 4-file behavior move + full cfm_v2k re-stamp)
**Solved when:** its pre-registered run meets the frozen bars and the decision is recorded — DEC-586 (curriculum step 2: the intent structure, one axis, at the frozen difficulty; fixture intent=5.0/br=0.3/N=500/seed=17, frozen DEC-576 bars in corrected units, branches (i)-(v), 4-file behavior move + full cfm_v2k re-stamp)

###### E1.c — port: ONE real-data run under DEC-572's bars

*Does the winning trunk master the real stream — eff_rank>=30, persona>=0.15, R2>=0.45, stable, plus the B1 canary — in ONE pre-registered run?*

Toy mastery earns the right to exactly one real-data run; the port receipt is what makes B's green a statement about a working trunk, not about a blob (DEC-576's port protocol).

**Bars frozen in:** DEC-572/573 (real data: eff_rank>=30, persona>=0.15, R2>=0.45, stable)
**Solved when:** its pre-registered run meets the frozen bars and the decision is recorded — DEC-572/573 (real data: eff_rank>=30, persona>=0.15, R2>=0.45, stable)

##### B4 — Decisive

*Does B support action/counterfactual decisions?*

Prediction sufficiency does not imply decision sufficiency — the action family is part of the goal, not a downstream detail.

**Solved when:** every child below is solved (worst-child rule)

###### B4.1 — B-only dose-response / argmax

*action-conditional response shape readable from the state (not through a judge leg)*

Prediction sufficiency does not imply decision sufficiency: the action-conditional response shape must be readable from the state itself, or the action family is not covered by construction.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B4.2 — OPE / IPS known-effect recovery

*E recovers the known effect with a paired CI excluding 0*

Known-effect recovery with a paired CI excluding 0 is the causal grounding of the decision claim: without it, 'the state carries the effect' is correlation wearing a coat.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B4.3 — Action-invariance

*representation invariant to assignment, still predictive of the outcome*

A representation that moves with the assignment encodes the intervention, not the state: action-invariance is the test that the decision-grade state is separable from the action.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

##### B5 — Minimal & affordable

*Is it the smallest sufficient object, and fast enough?*

Without minimality, 'sufficient' is trivially satisfied by shipping the history; minimality is what makes the object bounded and affordable.

**Solved when:** every child below is solved (worst-child rule)

###### B5.1 — Minimality k_min

*served rank = the frozen reconstruction certificate*

Without minimality, 'sufficient' is trivially satisfied by shipping the history: k_min (the reconstruction certificate) is what makes the object bounded, fixed and affordable.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B5.2 — Cost / runtime / constant-state

*runtime, memory and incremental-update receipt*

A capability that is correct but unacceptably slow or state-unbounded is not done: the runtime / constant-state receipt is the cost side of the minimality bar.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

#### P3 — The donor is usable

*Can simple consumers read it, does it transfer beyond our known tasks, and does it stay honest under shift?*

Sufficiency proves information is present; it does not prove it is recoverable. A sufficient-but-entangled representation fails the consumer contract, and a donor that only helps the tasks we designed for is not universal. Robustness closes the honesty gap. Necessary for P4 because the judge's heads are simple by design: if the simple head cannot read it, the consequence cannot appear.

**Solved when:** every child below is solved (worst-child rule)

##### B3 — Accessible

*Can simple heads read what is there, at every stage?*

Information that a simple head cannot read does not reach a consumer; present is not usable.

**Solved when:** every child below is solved (worst-child rule)

###### B3.1 — Outlet accessibility (S4)

*per-bucket accessibility parity + per-bucket negative controls*

An outlet that cannot read its own bucket means the information is present but unusable: per-bucket accessibility parity, with negative controls, is the floor of recoverability.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B3.2 — Stage decomposition M0->M1->served

*the fine-structure metric at each stage, so a loss is attributed to a stage*

A loss must be attributable to a stage: the M0->M1->served decomposition is what turns 'the embedding lost it' into 'the fold lost it' — a diagnosis, not a verdict.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B3.3 — Linear accessibility (current)

*mean ridge R2 on A->B over the lab's persona+future fields; served vs PATH A*

A simple linear head that cannot beat PATH A at readout means no downstream consumer can: linear access is the floor of the Accessibility principle.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B3.4 — Block-aware serving geometry

*exact counter block beside the learned shape, equal total variance; persona/future/ARI*

Serving allocation must be measured against the exact counter at equal variance: without it, a geometry gain could be an allocation artifact, and the shape block's share would be unattributed.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B3.5 — Equal-dimension serving geometry

*shape PCA + counter tile at 128/128 and 39/39; persona/future/ARI at full size*

Dimension-matched shape + counter tiles separate a real serving gain from a raw dimension-count gain: without the equal-dimension control, 'serving helps' is confounded.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B3.6 — Served-object X-ray (ledger exposure)

*the ledger alone, its named parts, and ledger block-PCA under the frozen B3.2/B3.4 consumers, against {exact counter, served shape, PATH A}; the branch is computed here from the arm numbers (DEC-569)*

The ledger's named parts must be exposed under the frozen consumers: the X-ray localizes where the fine structure lives (the exposure branch) so serving is designed, not tuned.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

##### B6 — Transferable

*Does it add signal on held-out consumers?*

Transfer beyond our known legs is the only evidence against designing for the tasks we already have.

**Solved when:** every child below is solved (worst-child rule)

###### B6.1 — Objective battery (S5)

*every content pressure beats its naive comparator (or is excluded and reported)*

Content pressures that do not beat their naive comparators leave the breadth of the embedding unproven: the objective battery is the anti-overfit clause made measurable.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B6.2 — Donor does-not-lose / adds / controls

*E vs raw, fair table, table+E; random/destroyed negative controls*

Does-not-lose / adds / destroyed-vs-random controls against raw and the fair table: without them, 'adds unique signal' is a point estimate, not a certification.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B6.3 — Broad held-out transfer family

*adds signal on a family broader than the eight judge legs*

Signal that appears only on the eight known legs means the donor was designed for our tasks: a broad held-out transfer family is the only evidence against that.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

##### B7 — Robust

*Does it stay honest under shift and missingness?*

Uncertainty and missingness are where a learned state lies most easily — unlogged events must be carried, never faked.

**Solved when:** every child below is solved (worst-child rule)

###### B7.1 — ECE / OOD retention

*calibration of the shipped next-type head + OOD retention*

Calibrated uncertainty under shift is where a learned state lies most easily: ECE / OOD retention on the reference stream is the honesty half of the robustness claim.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### B7.2 — Missingness / unlogged confounders

*uncertainty carried for unobserved events, never faked*

Unlogged events must be carried as uncertainty, never faked: missingness handling is what makes 'coverage over functions of the observed stream' true instead of aspirational.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

#### P4 — The consequence is delivered

*Does PATH B (frozen encoder -> simple head) beat PATH A (wide table -> GBM incumbent) for the four consumer families, with controls, brain alone and brain+ledger?*

This is the project's definition of progress. It is downstream of P1-P3 by construction: known use cases may veto, never shape, the encoder. A green P4 with red P2/P3 would mean the consumers were tuned, not the foundation.

**Solved when:** every child below is solved (worst-child rule)

##### JUDGE-supervised — supervised leg

*forecast targets beat the incumbent GBM on the same splits*

Forecast targets are the most direct consumer of a predictive embedding; if the frozen state cannot be read for them, the donor claim is empty.

**Solved when:** every child below is solved (worst-child rule)

###### JUDGE-supervised-brain — supervised :: brain alone (PROGRESS)

*every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain*

The PROGRESS column of the most direct consumer: without a strict win over PATH A (paired CI excluding 0, controls below brain) on every non-degenerate task, the donor claim is empty here.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### JUDGE-supervised-brain_ledger — supervised :: brain + ledger (DELIVERY, non-inferiority)

*every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain*

The DELIVERY column: brain + ledger must be non-inferior within the frozen margin — the shipping claim; ledger points are never a brain win, so this row measures what the customer actually gets.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

##### JUDGE-unsupervised — unsupervised leg

*distance equals shared futures; structure beats table/random/destroyed*

Geometry is the structure-revealing half of the contract: distance must mean shared futures, not shared summary.

**Solved when:** every child below is solved (worst-child rule)

###### JUDGE-unsupervised-brain — unsupervised :: brain alone (PROGRESS)

*every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain*

The PROGRESS column of the structure-revealing half: distance must mean shared futures, not shared summary — a strict win over PATH A with controls, or the geometry claim is empty.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### JUDGE-unsupervised-brain_ledger — unsupervised :: brain + ledger (DELIVERY, non-inferiority)

*every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain*

The DELIVERY column: brain + ledger must keep the geometry non-inferior within the frozen margin — what the customer actually gets when the ledger is attached.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

##### JUDGE-rl — rl leg

*policy value beats table and random on randomized/known-effect data*

This is the decision use case: policy value from a frozen pre-action state on randomized data, with controls.

**Solved when:** every child below is solved (worst-child rule)

###### JUDGE-rl-brain — rl :: brain alone (PROGRESS)

*every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain*

The PROGRESS column of the decision use case: policy value from a frozen pre-action state on randomized data must beat table and random with controls, or the decision layer has nothing to stand on.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### JUDGE-rl-brain_ledger — rl :: brain + ledger (DELIVERY, non-inferiority)

*every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain*

The DELIVERY column: brain + ledger must keep the decision value non-inferior within the frozen margin.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

##### JUDGE-simulator — simulator leg

*recovers the built-in counterfactual effects (red_king role)*

Known-effect recovery: the state must carry action-conditional response, not correlation, or the decision layer has nothing to stand on.

**Solved when:** every child below is solved (worst-child rule)

###### JUDGE-simulator-brain — simulator :: brain alone (PROGRESS)

*every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain*

The PROGRESS column of known-effect recovery: the state must carry action-conditional response, not correlation — a strict win over PATH A with the destroyed control below, or the counterfactual claim is empty.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

###### JUDGE-simulator-brain_ledger — simulator :: brain + ledger (DELIVERY, non-inferiority)

*every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain*

The DELIVERY column: brain + ledger must keep the known-effect recovery non-inferior within the frozen margin.

**Solved when:** its frozen gate passes on a current receipt (the gate lives in the frozen spec — see the atomic-problems table and the card's row)

#### P5 — The stack is v1-complete

*Does the full system deliver a certified decision: identifiable stream -> donor -> plugins -> red_king -> red_queen, with caterpillar explaining from artifacts?*

The foundation exists to serve decisions. v1 completeness is a separate scope from the encoder, and this card does not assess it; the atoms are named so the tree spans the whole goal and so that 'all solved' cannot be claimed on the encoder alone.

**Solved when:** every child below is solved (worst-child rule)

##### V1.1 — Causally identifiable stream

*Does rabbit_hole emit randomized arms with a known effect, recorded as such?*

Without a known-effect stream no counterfactual claim can be validated; every decision result would be untestable.

**Evidence:** the v1 stack's own gate — this card does not measure it.
**Solved when:** its v1 gate passes (not measured on this card; AGENTS.md definition of done)

##### V1.2 — Each plugin passes its independent gate

*Do the supervised, unsupervised and white_queen consumers each pass their own gate?*

Each consumer family must be certified on its own terms; a single composite score would hide a failing family.

**Evidence:** the v1 stack's own gate — this card does not measure it.
**Solved when:** its v1 gate passes (not measured on this card; AGENTS.md definition of done)

##### V1.3 — red_king recovers known effects

*Does red_king recover the known counterfactual effect and improve the certified decision with vs without it?*

The decision layer must add causal value over logging, and must never ship worse than logging.

**Evidence:** the v1 stack's own gate — this card does not measure it.
**Solved when:** its v1 gate passes (not measured on this card; AGENTS.md definition of done)

##### V1.4 — red_queen composes actions

*Does red_queen compose next-best actions across cadences under constraints?*

The system's output is an action, not a score; composition is where the cadences and constraints meet.

**Evidence:** the v1 stack's own gate — this card does not measure it.
**Solved when:** its v1 gate passes (not measured on this card; AGENTS.md definition of done)

##### V1.5 — caterpillar explains from artifacts

*Does caterpillar answer 'why' from artifacts only?*

A decision that cannot be explained from its artifacts is not operable, and an explanation that reads the model live cannot be audited.

**Evidence:** the v1 stack's own gate — this card does not measure it.
**Solved when:** its v1 gate passes (not measured on this card; AGENTS.md definition of done)

## 3. The atomic problems — what evidence closes each one

### The measured atomic problems (frozen rows)

| Atomic problem | Property | Frozen gate | Green means | Producer / source |
|---|---|---|---|---|
| `B1.1` | Lawful dynamics (S3) | working=True | reporting | measure_dynamics_law.py |
| `B1.2` | Construction laws (C1-C6) | C1-C6 all meet the frozen bars (DEC-585) | quality | measure_construction_law.py |
| `B8.1` | Artifact identity | present and consistent (DEC-585) | reporting | measure_b_props.py |
| `B8.2` | One-command reproducibility (S6) | working=True | reporting | measure_certification_law.py |
| `B2.1` | Worst-case sufficiency U | U_served >= 0.5 (frozen in DEC-570) | quality | measure_sufficiency_law.py |
| `B2.2` | Full-history oracle gap | every included probe's U_served >= 0.5 (zero included = bar unfilled) | quality | measure_sufficiency_law.py |
| `B2.3` | Learned-only attribution | attribution recorded (learned-only U finite) | reporting | measure_sufficiency_law.py |
| `B4.1` | B-only dose-response / argmax | E argmax share >= table, paired CI > 0, above random/destroyed controls (DEC-585) | quality | measure_decision_law.py |
| `B4.2` | OPE / IPS known-effect recovery | IPS policy diff > 0 and CI lower bound > 0 (DEC-585) | quality | measure_decision_law.py |
| `B4.3` | Action-invariance | cutoff band cert ok; marker/level teeth recorded (DEC-585) | reporting | measure_decision_law.py |
| `B5.1` | Minimality k_min | block-wise certified: served shape rank == shape k_min; explicit measure block | quality | measure_minimality_law.py |
| `B5.2` | Cost / runtime / constant-state | complete, N == registry, RSS within slice, finite rows/s, incremental recorded (DEC-585) | reporting | measure_runtime.py |
| `B3.1` | Outlet accessibility (S4) | working=True | reporting | measure_views_law.py |
| `B3.2` | Stage decomposition M0->M1->served | attribution recorded | reporting | measure_stage_decomposition.py |
| `B3.3` | Linear accessibility (current) | served >= PATH A - 0.01 | quality | measure_block_serving.py |
| `B3.4` | Block-aware serving geometry | complete | reporting | measure_block_serving.py |
| `B3.5` | Equal-dimension serving geometry | complete | reporting | measure_block_serving.py |
| `B3.6` | Served-object X-ray (ledger exposure) | complete + branch decided | reporting | measure_served_object.py |
| `B6.1` | Objective battery (S5) | working=True | reporting | measure_battery_law.py |
| `B6.2` | Donor does-not-lose / adds / controls | E >= raw and fair (point); table+E ensemble CI > 0; controls below E (DEC-585/594) | quality | measure_donor.py |
| `B6.3` | Broad held-out transfer family | E >= table on >= 2/3 of the published family; family mean > 0; controls below E | quality | measure_transfer_law.py |
| `B7.1` | ECE / OOD retention | ECE(next event) <= 0.1; OOD retention recorded (DEC-585) | quality | measure_b_props.py |
| `B7.2` | Missingness / unlogged confounders | coverage reported (next-time, occurrence, response, missingness sensitivity) | reporting | measure_robustness_law.py |
| `JUDGE-supervised-brain` | supervised :: brain alone (PROGRESS) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_supervised.json |
| `JUDGE-supervised-brain_ledger` | supervised :: brain + ledger (DELIVERY, non-inferiority) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_supervised.json |
| `JUDGE-unsupervised-brain` | unsupervised :: brain alone (PROGRESS) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_unsupervised.json |
| `JUDGE-unsupervised-brain_ledger` | unsupervised :: brain + ledger (DELIVERY, non-inferiority) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_unsupervised.json |
| `JUDGE-rl-brain` | rl :: brain alone (PROGRESS) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_rl.json |
| `JUDGE-rl-brain_ledger` | rl :: brain + ledger (DELIVERY, non-inferiority) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_rl.json |
| `JUDGE-simulator-brain` | simulator :: brain alone (PROGRESS) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_simulator.json |
| `JUDGE-simulator-brain_ledger` | simulator :: brain + ledger (DELIVERY, non-inferiority) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_simulator.json |

### The DEC-frozen atomic problems (no card row; each closes only by its recorded evidence)

| Atomic problem | What must be true | Bars frozen in |
|---|---|---|
| `E1.a` | Does the frozen trunk beat the 1-gram on the known-truth next_event structure at the dev-rig scale, with clean controls? | DEC-576/577 (dev rig: g<=0.5 and beats the 1-gram, controls clean) — exhausted negative at N=500 across all four single-axis trunk-algebra values plus the dim=256 capacity read (DEC-581/582 outcomes: the 1-gram floor is binding, a finite-sample estimation gap, not a trunk deficiency) |
| `E1.b` | Does the frozen trunk beat the 1-gram on the per-customer intent structure, at a scale where the sample can determine it, with clean controls? | DEC-586 (curriculum step 2: the intent structure, one axis, at the frozen difficulty; fixture intent=5.0/br=0.3/N=500/seed=17, frozen DEC-576 bars in corrected units, branches (i)-(v), 4-file behavior move + full cfm_v2k re-stamp) |
| `E1.c` | Does the winning trunk master the real stream — eff_rank>=30, persona>=0.15, R2>=0.45, stable, plus the B1 canary — in ONE pre-registered run? | DEC-572/573 (real data: eff_rank>=30, persona>=0.15, R2>=0.45, stable) |

### The v1 atomic problems this card does not measure

| Atomic problem | Requirement | Evidence |
|---|---|---|
| `V1.1` | Causally identifiable stream | the v1 stack's own gate |
| `V1.2` | Each plugin passes its independent gate | the v1 stack's own gate |
| `V1.3` | red_king recovers known effects | the v1 stack's own gate |
| `V1.4` | red_queen composes actions | the v1 stack's own gate |
| `V1.5` | caterpillar explains from artifacts | the v1 stack's own gate |

Two v1 requirements are already owned elsewhere in the tree, so they are references,
not duplicate atoms:

- **V1.2 (AGENTS: universal donor usable by every plugin)** -> P1-P3 + B6
- **V1.7 (AGENTS: every artifact versioned + reproducible, receipts)** -> B8

## 4. What "all solved" means — exactly

- Every **measured atom** (31) passes its frozen gate on a current
  receipt: that is `T0`, the encoder battery.
- Every **DEC-frozen atom** (3) closes only by its pre-registered run
  meeting the recorded bars and the decision being logged — never by a lab receipt and
  never by narration.
- Every **v1 atom** (5) passes the stack's own gate.
- Every **interior problem** (19) is solved when all the atoms under
  it are solved: the worst-child rule, nothing else.
- **No open BREAKING/MAJOR BACKWARD requirement** (AUDIT.md).

All of the above = **project solved**. Anything less is a named, located gap: the tree
says which problem is open and the card says which atom is missing.

This is the doctrine's T0/SOLVED distinction made structural: a fully green encoder
card is `ENCODER DONE`, not `SOLVED`, because the v1 scope and the DEC-frozen trunk
atoms are separate atoms. The card renders `SOLVED` only when its generated root status
is PASS — which cannot happen while an atom in this tree is unmeasured.

## 5. The route — the order the evidence is earned

The tree says what must be true; the route says in what order it is earned. It is a
pointer list, not a second scorecard: no step carries a verdict, and the card never
computes one. Moving the pointer is a reviewed spec edit like any other.

| Step | What is earned | Advances | Evidence that closes it |
|---|---|---|---|
| **R1** | Dev rig: master next_event (N=500, known truth) | E1.a | CLOSED negative at the toy (DEC-581/582 outcomes: 4 algebra values + capacity @256, g 1.147–1.182 vs bar 0.5, 1-gram floor binding at N=500); the intent remainder was executed under DEC-586; never a B-row verdict |
| **R2** | Dev rig: master intent (curriculum step 2) | E1.b | DONE: intent g -0.721 (365d) and 0.181 (730d) with the hierarchical Dirichlet signature head; beats_base true both; E1.b green (DEC-600/601). |
| **R3** | Port: ONE real-data run of the winning trunk under DEC-572's bars | E1.c | DONE: served bank + measure PCA-128; eff_rank 30.9, persona 0.267, served R2 0.460; B1 canary from the B rows (lab/port_dec572.json). |
| **R4** | Real stream: certify the B foundational battery | P1, P2, P3 | DONE: B certified 23/23 rows, 8/8 sections at behavior 7baf744f (DEC-594/595/596); the B gate is no longer the constraint. |
| **R5** | Real stream: re-score the four judge legs (brain and brain+ledger) | P4 | DONE under the bank protocol (DEC-598): 8/8 leg-columns on the served block bank; the v1 flat judge (4/8) and the GBM delivery head are recorded beside it. |
| **R6** | Stack: build and certify v1 (identifiable stream, plugins, red_king, red_queen, caterpillar) | P5 | DONE: V1.1 identifiable stream; V1.2 plugin acceptance 9/9; V1.3 red_king SCORE 0.889; V1.4 red_queen channel certification; V1.5 caterpillar artifact-only explanation (v1_*.json). |
| **R7 ← now** | BACKWARD: adversarial review with no BREAKING findings | G | BLOCKED: FULL BACKWARD found 6 BREAKING + 4 MAJOR open requirements (B-001..B-010); the forward subgoals M-001..M-010 are the next work. |

The order is forced by dependency and by cost: the dev rig is minutes and earns the
right to the real-data port; the port closes E1.c; only then can the remaining B atoms
be read as evidence about a working trunk; only then is a B+C re-score diagnostic rather
than a measurement of the blob.

## 6. The current frontier

The B+B+C+plugins success definition is met under DEC-598: B is certified (23/23 rows, 8/8 sections); the four judge legs are 8/8 on the served block bank (supervised ridge alpha=10 + delivery bank + compact ledger; block-aware geometry; RL and simulator canonical); and the three plugin gates pass 9/9 (supervised 6/6, unsupervised 4/4, white_queen 5/5, Layer C 9/9) on the bank + demo stream. The v1 flat judge (4/8), the GBM delivery head and the old canonical stream are recorded beside it. The full plan root remains OPEN at E1 (trunk) and the v1 atoms red_king/red_queen/caterpillar and the identifiable-stream metadata, which are outside the B+B+C+plugins definition.

**Why the card can read green while the encoder is not mastered — read this before
trusting a green row.** The plan labels every measured atom with what its green PROVES
(`scripts/b_spec.py: gate_kind`, derived from the frozen gate's kind, never asserted):

- **reporting** — green certifies that the measurement happened and was attributed: `B1.1`, `B8.1`, `B8.2`, `B2.3`, `B4.3`, `B5.2`, `B3.1`, `B3.2`, `B3.4`, `B3.5`, `B3.6`, `B6.1`, `B7.2`.
- **quality** — green certifies a level against a frozen bar or comparator (the remaining atoms).

The quality levels for the reporting-gated axes are the DEC-frozen bars (U >= 0.5 in
DEC-570; eff_rank >= 30 and persona >= 0.15 in DEC-572; on the dev rig g <= 0.5 and
beating the 1-gram in DEC-576/577), and the measured values are in the receipts, not in
this plan.

That is exactly why the trunk is a problem (E1) cut to three DEC-frozen atoms (E1.a–c)
and not a footnote: a blob can satisfy every reporting gate while failing the quality
bars, and the plan must show the difference instead of letting a green card imply more
than it proved.

## 7. The method contract — how work is allowed to move the tree

1. **One axis per iteration**, pre-registered in `DECISIONS.md` before the run
   (gate, bars, a-priori branches). No post-hoc bar edits in any branch.
2. **Dev rig first, real stream second.** Known-truth lab fixtures (`next_event` ->
   `intent` -> rate heterogeneity -> entities -> trend -> action peak) at N=500 make a
   full A->B->C loop minutes long; a candidate earns the right to a real-data run.
3. **Lab receipts are a development instrument.** They never flip a B-row verdict;
   certification stays on the real stream. The port fires only when both `next_event`
   and `intent` are mastered with clean controls (DEC-576).
4. **Every behaviour move folds its re-stamp**: frozen artifacts are never retrained to
   re-label them; their receipts are regenerated by re-running the recorded producers.
5. **The frontier is chosen by the tree, not by convenience**: the open node closest to
   the goal whose blocker is measured — cheapest experiment first.

## 8. Provenance and the one-source rule

```text
scripts/plan_spec.py ──> scripts/plan.py ──> PLAN.md            (structure only)
                    └──> scripts/report_card.py ──> REPORT_CARD.md (structure + verdicts)
```

The card imports the spec; the spec imports the frozen row definitions (`b_spec`,
`judge_spec`); neither imports the other's render. A test asserts every frozen row is an
atomic problem exactly once, the tree is acyclic, and `PLAN.md` is current with the spec.

This file was reinstated by operator request on 2026-11-03 (DEC-579), and reframed the
same day (DEC-584) at the operator's directive: the tree is a recursive decomposition
to ATOMIC problems at whatever depth that takes — "section" and "leaf" retired as
vocabulary, progress read off the solved count at every level. DEC-513 once retired
PLAN.md because "the card is the plan"; the difference now is direction and
single-source: the plan is the goal-side decomposition, the card is its measurement
surface, and both render from one spec — so the card cannot grow rows the plan does not
contain, and the plan cannot claim a state the card did not measure.
