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

## 1. How to read this plan — the backward cut

The tree starts at the goal and asks, at every node, *what must be true for this to be
true?* Each answer is a smaller, independently measurable problem, until the leaves are
conditions a receipt can settle. Solving the leaves bottom-up is solving the goal — but
only if every leaf is solved, which is why the tree names the ones the card does not
measure yet instead of pretending they do not exist.

- **Every node is a necessary condition of its parent.** Nothing here is an
  implementation preference; each node answers *why the parent cannot hold without it*.
- **The plan defines no gate.** Every measured leaf is a frozen row whose gate lives in
  `scripts/b_spec.py` or `scripts/judge_spec.py`, frozen before its result. An interior
  verdict is the worst of its children — the card's existing rule, not a new one.
- **Three leaf kinds, deliberately distinguished.** `card` leaves are computed from
  receipts; `dec` leaves have bars frozen in a DEC and close only through a
  pre-registered run; `v1` leaves belong to the stack's own gates and are not assessed
  on this card. A green card alone therefore proves T0 — never SOLVED.
- **No relabelling.** A node moves from open to solved by evidence, never by editing this
  file. Changing what a leaf measures is a new protocol version, not a plan edit.

## 2. The tree

The census: **31 card leaves**, **1 DEC-frozen
leaf**, **5 v1 leaves** — 37 conditions in all.

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

**Leaves:** `B1.1`, `B1.2`
**Solved when:** every leaf below passes its frozen gate (2 rows)

##### B8 — Reproducible

*Is the artifact an identity?*

An artifact that cannot be rebuilt at a recorded commit is a transcript, not a certificate.

**Leaves:** `B8.1`, `B8.2`
**Solved when:** every leaf below passes its frozen gate (2 rows)

#### P2 — The donor suffices

*Does the embedding lose nothing that matters about the future — for prediction and decision — and is it the smallest object that does?*

A donor that is lawful but lossy is a summary, not a foundation: any consumer would inherit the loss, and no amount of consumer tuning could recover it. Sufficiency is judged worst-case against the full-history oracle, not against our tasks; decision sufficiency extends the claim to the action family; minimality removes the degenerate escape of shipping the history. Necessary for P4: a consumer cannot beat PATH A from an embedding that dropped the signal.

**Solved when:** every child below is solved (worst-child rule)

##### B2 — Sufficient

*Does B lose nothing about the future?*

Sufficiency is the donor contract itself: without U the embedding is an unmeasured summary, and every consumer inherits whatever it dropped.

**Leaves:** `B2.1`, `B2.2`, `B2.3`
**Solved when:** every leaf below passes its frozen gate (3 rows)

##### E1 — Trunk capacity (the state-update algebra)

*Can the state carry the stream's structure — not just its mass?*

A sufficient statistic must have somewhere to live. When the trunk's effective rank collapses, the embedding is a blob and every downstream row measures the blob, not the stream. This is the measured blocker (O3 attempt 4 rejected; the dev rig reproduces it in minutes), and it is the one condition whose bars are frozen outside the card.

**Bars frozen in:** DEC-572/573 (real data: eff_rank>=30, persona>=0.15, R2>=0.45, stable) · DEC-575/576/577 (dev rig: g<=0.5 and beats the 1-gram, controls clean)
**Solved when:** its pre-registered run meets the frozen bars and the decision is recorded — DEC-572/573 (real data: eff_rank>=30, persona>=0.15, R2>=0.45, stable) · DEC-575/576/577 (dev rig: g<=0.5 and beats the 1-gram, controls clean)

##### B4 — Decisive

*Does B support action/counterfactual decisions?*

Prediction sufficiency does not imply decision sufficiency — the action family is part of the goal, not a downstream detail.

**Leaves:** `B4.1`, `B4.2`, `B4.3`
**Solved when:** every leaf below passes its frozen gate (3 rows)

##### B5 — Minimal & affordable

*Is it the smallest sufficient object, and fast enough?*

Without minimality, 'sufficient' is trivially satisfied by shipping the history; minimality is what makes the object bounded and affordable.

**Leaves:** `B5.1`, `B5.2`
**Solved when:** every leaf below passes its frozen gate (2 rows)

#### P3 — The donor is usable

*Can simple consumers read it, does it transfer beyond our known tasks, and does it stay honest under shift?*

Sufficiency proves information is present; it does not prove it is recoverable. A sufficient-but-entangled representation fails the consumer contract, and a donor that only helps the tasks we designed for is not universal. Robustness closes the honesty gap. Necessary for P4 because the judge's heads are simple by design: if the simple head cannot read it, the consequence cannot appear.

**Solved when:** every child below is solved (worst-child rule)

##### B3 — Accessible

*Can simple heads read what is there, at every stage?*

Information that a simple head cannot read does not reach a consumer; present is not usable.

**Leaves:** `B3.1`, `B3.2`, `B3.3`, `B3.4`, `B3.5`, `B3.6`
**Solved when:** every leaf below passes its frozen gate (6 rows)

##### B6 — Transferable

*Does it add signal on held-out consumers?*

Transfer beyond our known legs is the only evidence against designing for the tasks we already have.

**Leaves:** `B6.1`, `B6.2`, `B6.3`
**Solved when:** every leaf below passes its frozen gate (3 rows)

##### B7 — Robust

*Does it stay honest under shift and missingness?*

Uncertainty and missingness are where a learned state lies most easily — unlogged events must be carried, never faked.

**Leaves:** `B7.1`, `B7.2`
**Solved when:** every leaf below passes its frozen gate (2 rows)

#### P4 — The consequence is delivered

*Does PATH B (frozen encoder -> simple head) beat PATH A (wide table -> GBM incumbent) for the four consumer families, with controls, brain alone and brain+ledger?*

This is the project's definition of progress. It is downstream of P1-P3 by construction: known use cases may veto, never shape, the encoder. A green P4 with red P2/P3 would mean the consumers were tuned, not the foundation.

**Solved when:** every child below is solved (worst-child rule)

##### JUDGE-supervised — supervised leg

*forecast targets beat the incumbent GBM on the same splits*

Forecast targets are the most direct consumer of a predictive embedding; if the frozen state cannot be read for them, the donor claim is empty.

**Leaves:** `JUDGE-supervised-brain`, `JUDGE-supervised-brain_ledger`
**Solved when:** every leaf below passes its frozen gate (2 rows)

##### JUDGE-unsupervised — unsupervised leg

*distance equals shared futures; structure beats table/random/destroyed*

Geometry is the structure-revealing half of the contract: distance must mean shared futures, not shared summary.

**Leaves:** `JUDGE-unsupervised-brain`, `JUDGE-unsupervised-brain_ledger`
**Solved when:** every leaf below passes its frozen gate (2 rows)

##### JUDGE-rl — rl leg

*policy value beats table and random on randomized/known-effect data*

This is the decision use case: policy value from a frozen pre-action state on randomized data, with controls.

**Leaves:** `JUDGE-rl-brain`, `JUDGE-rl-brain_ledger`
**Solved when:** every leaf below passes its frozen gate (2 rows)

##### JUDGE-simulator — simulator leg

*recovers the built-in counterfactual effects (red_king role)*

Known-effect recovery: the state must carry action-conditional response, not correlation, or the decision layer has nothing to stand on.

**Leaves:** `JUDGE-simulator-brain`, `JUDGE-simulator-brain_ledger`
**Solved when:** every leaf below passes its frozen gate (2 rows)

#### P5 — The stack is v1-complete

*Does the full system deliver a certified decision: identifiable stream -> donor -> plugins -> red_king -> red_queen, with caterpillar explaining from artifacts?*

The foundation exists to serve decisions. v1 completeness is a separate scope from the encoder, and this card does not assess it; the leaves are named so the tree spans the whole goal and so that 'all solved' cannot be claimed on the encoder alone.

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

## 3. The measured leaves — what evidence closes each one

| Leaf | Property | Frozen gate | Green means | Producer / source |
|---|---|---|---|---|
| `B1.1` | Lawful dynamics (S3) | working=True | reporting | measure_dynamics_law.py |
| `B1.2` | Construction laws (C1-C6) | all construction laws met | unmeasured | the legacy encoder-construction producer |
| `B8.1` | Artifact identity (legacy) | present | unmeasured | the legacy artifact producer |
| `B8.2` | One-command reproducibility (S6) | working=True | reporting | measure_certification_law.py |
| `B2.1` | Worst-case sufficiency U | U_served >= 0.5 (frozen in DEC-570) | quality | measure_sufficiency_law.py |
| `B2.2` | Full-history oracle gap | every included probe's U_served >= 0.5 (zero included = bar unfilled) | quality | measure_sufficiency_law.py |
| `B2.3` | Learned-only attribution | attribution recorded (learned-only U finite) | reporting | measure_sufficiency_law.py |
| `B4.1` | B-only dose-response / argmax | gate met | unmeasured | measure_decision_law.py |
| `B4.2` | OPE / IPS known-effect recovery (legacy) | CI excludes 0 on the declared arms | unmeasured | the legacy causal producer |
| `B4.3` | Action-invariance | gate met | unmeasured | measure_decision_law.py |
| `B5.1` | Minimality k_min | k_min | unmeasured | measure_minimality_law.py |
| `B5.2` | Cost / runtime / constant-state (legacy) | present and within budget | unmeasured | the legacy runtime producer |
| `B3.1` | Outlet accessibility (S4) | working=True | reporting | measure_views_law.py |
| `B3.2` | Stage decomposition M0->M1->served | attribution recorded | reporting | measure_stage_decomposition.py |
| `B3.3` | Linear accessibility (current) | served >= PATH A - 0.01 | quality | measure_block_serving.py |
| `B3.4` | Block-aware serving geometry | complete | reporting | measure_block_serving.py |
| `B3.5` | Equal-dimension serving geometry | complete | reporting | measure_block_serving.py |
| `B3.6` | Served-object X-ray (ledger exposure) | complete + branch decided | reporting | measure_served_object.py |
| `B6.1` | Objective battery (S5) | working=True | reporting | measure_battery_law.py |
| `B6.2` | Donor does-not-lose / adds / controls (legacy) | all donor gates met | unmeasured | the legacy donor producer |
| `B6.3` | Broad held-out transfer family | family published and passed | unmeasured | measure_transfer_law.py |
| `B7.1` | ECE / OOD retention (legacy) | ECE <= 0.1 and OOD retained | unmeasured | the legacy calibration producer |
| `B7.2` | Missingness / unlogged confounders | coverage reported | unmeasured | measure_robustness_law.py |
| `JUDGE-supervised-brain` | supervised :: brain alone (PROGRESS) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_supervised.json |
| `JUDGE-supervised-brain_ledger` | supervised :: brain + ledger (DELIVERY, non-inferiority) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_supervised.json |
| `JUDGE-unsupervised-brain` | unsupervised :: brain alone (PROGRESS) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_unsupervised.json |
| `JUDGE-unsupervised-brain_ledger` | unsupervised :: brain + ledger (DELIVERY, non-inferiority) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_unsupervised.json |
| `JUDGE-rl-brain` | rl :: brain alone (PROGRESS) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_rl.json |
| `JUDGE-rl-brain_ledger` | rl :: brain + ledger (DELIVERY, non-inferiority) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_rl.json |
| `JUDGE-simulator-brain` | simulator :: brain alone (PROGRESS) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_simulator.json |
| `JUDGE-simulator-brain_ledger` | simulator :: brain + ledger (DELIVERY, non-inferiority) | every non-degenerate task: paired CI excludes 0 vs PATH A; controls below brain | quality | judge_leg_simulator.json |

### The leaves this card does not measure (v1)

| Leaf | Requirement | Evidence |
|---|---|---|
| `V1.1` | Causally identifiable stream | the v1 stack's own gate |
| `V1.2` | Each plugin passes its independent gate | the v1 stack's own gate |
| `V1.3` | red_king recovers known effects | the v1 stack's own gate |
| `V1.4` | red_queen composes actions | the v1 stack's own gate |
| `V1.5` | caterpillar explains from artifacts | the v1 stack's own gate |

Two v1 requirements are already owned elsewhere in the tree, so they are references,
not duplicate nodes:

- **V1.2 (AGENTS: universal donor usable by every plugin)** -> P1-P3 + B6
- **V1.7 (AGENTS: every artifact versioned + reproducible, receipts)** -> B8

## 4. What "all solved" means — exactly

- Every **card leaf** (31) passes its frozen gate on a current
  receipt: that is `T0`, the encoder battery.
- The **DEC-frozen leaf** (1) closes only by its pre-registered run
  meeting the recorded bars and the decision being logged — never by a lab receipt and
  never by narration.
- Every **v1 leaf** (5) passes the stack's own gate.
- **No open BREAKING/MAJOR BACKWARD requirement** (AUDIT.md).

All of the above = **project solved**. Anything less is a named, located gap: the tree
says which node is open and the card says which leaf is missing.

This is the doctrine's T0/SOLVED distinction made structural: a fully green encoder
card is `ENCODER DONE`, not `SOLVED`, because the v1 scope and the DEC-frozen port run
are separate leaves. The card renders `SOLVED` only when its generated root status is
PASS — which cannot happen while a leaf in this tree is unmeasured.

## 5. The route — the order the evidence is earned

The tree says what must be true; the route says in what order it is earned. It is a
pointer list, not a second scorecard: no step carries a verdict, and the card never
computes one. Moving the pointer is a reviewed spec edit like any other.

| Step | What is earned | Advances | Evidence that closes it |
|---|---|---|---|
| **R1** | Dev rig: master next_event (N=500, known truth) | E1 | CLOSED negative at the toy (DEC-581/582 outcomes: 4 algebra values + capacity @256, g 1.147–1.182 vs bar 0.5, 1-gram floor binding at N=500); remainder = intent under DEC-583 (on hold); never a B-row verdict |
| **R2** | Dev rig: master intent (curriculum step 2) | E1 | lab receipts + the intent iteration's pre-registration; both structures clean |
| **R3** | Port: ONE real-data run of the winning trunk under DEC-572's bars | E1 | a pre-registered port receipt: eff_rank>=30, persona>=0.15, R2>=0.45, stable + B1 canary |
| **R4 ← now** | Real stream: fill the remaining B rows with their producers | P1, P2, P3 | B1.2, B4.1-B4.3, B5.1-B5.2, B6.2-B6.3, B7.1-B7.2, B8.1 at the current behavior |
| **R5** | Real stream: re-score the four judge legs (brain and brain+ledger) | P4 | judge_leg_*.json at the current behavior, both columns, controls readable |
| **R6** | Stack: build and certify v1 (identifiable stream, plugins, red_king, red_queen, caterpillar) | P5 | each v1 gate on the stack's own card rows / receipts |
| **R7** | BACKWARD: adversarial review with no BREAKING findings | G | AUDIT.md requirements closed |

The order is forced by dependency and by cost: the dev rig is minutes and earns the
right to the real-data port; the port closes E1; only then can the remaining B rows be
read as evidence about a working trunk; only then is a B+C re-score diagnostic rather
than a measurement of the blob.

## 6. The current frontier

The live frontier is P2's enabler E1 (the trunk's state-update algebra). O3 attempt 4 was rejected on its frozen bars and the dev rig reproduced the blob in minutes; all three single-axis trunk-algebra candidates are now negative on the toy (DEC-577 outcome: candidate 1; DEC-578 outcome: candidate 2; DEC-581 outcome: candidate 3), and the capacity axis is negative too (DEC-582 outcome: baseline algebra at dim=256, g 1.149121 ≈ iteration 1's 1.14727, no usable rank gained) — the 1-gram floor is binding on next_event at N=500, a finite-sample estimation gap, not a trunk deficiency. The next dev-rig structure is `intent` (DEC-583, on hold) and then ONE pre-registered port run under DEC-572's bars; no lab receipt flips a B-row verdict. The PRIMARY progress track until then is R4: the 11 remaining B sub-problems (6 legacy re-stamps at the current behavior + 5 new O-series producers), certification-first per the operator's 2026-11-03 directive.

**Why the card can read green while the encoder is not mastered — read this before
trusting a green row.** The plan labels every leaf with what its green PROVES
(`scripts/b_spec.py: gate_kind`, derived from the frozen gate's kind, never asserted):

- **reporting** — green certifies that the measurement happened and was attributed: `B1.1`, `B8.2`, `B2.3`, `B3.1`, `B3.2`, `B3.4`, `B3.5`, `B3.6`, `B6.1`.
- **quality** — green certifies a level against a frozen bar or comparator (the remaining leaves).

The quality levels for the reporting-gated axes are the DEC-frozen bars (U >= 0.5 in
DEC-570; eff_rank >= 30 and persona >= 0.15 in DEC-572; on the dev rig g <= 0.5 and
beating the 1-gram in DEC-576/577), and the measured values are in the receipts, not in
this plan.

That is exactly why the trunk is a node (E1) and not a footnote: a blob can satisfy
every reporting gate while failing the quality bars, and the plan must show the
difference instead of letting a green card imply more than it proved.

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
`judge_spec`); neither imports the other's render. A test asserts every frozen row is a
leaf exactly once, the tree is acyclic, and `PLAN.md` is current with the spec.

This file was reinstated by operator request on 2026-11-03 (DEC-579). DEC-513 once
retired PLAN.md because "the card is the plan"; the difference now is direction and
single-source: the plan is the goal-side decomposition, the card is its measurement
surface, and both render from one spec — so the card cannot grow rows the plan does not
contain, and the plan cannot claim a state the card did not measure.
