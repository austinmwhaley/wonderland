# PROMPT_BACKWARD — the two-scale adversarial review prompt

> **State — two modes (operator directive 2026-11-03, DEC-580):**
>
> - **FULL** — run **only** when `PROMPT_FORWARD.md` has produced an ENCODER DONE v1 card
>   (the definition in `PROMPT_FORWARD.md` §4). Tries to falsify the completion claim at both
>   scales and is the only mode that can yield a completion verdict.
> - **NODE** — run after any plan node changes status (closes/opens) or the route pointer
>   moves. Read-only and scoped to that node + its leaves. It may file BREAKING/MAJOR
>   requirements (a fake node closure is breaking) but it can **never** produce a completion
>   verdict or satisfy the tick-tock's stop condition.
>
> **Job:** try to falsify the claim at **both scales**. MACRO: is the tree the right
> decomposition, are its statuses honestly derived, and is the route pointer the true frontier?
> MICRO: is every green leaf green under its frozen gate, and does a green mean what the plan
> says it means? Explain **why the project may not achieve its goal**, even if every row is
> green. Document every finding for the next forward run.
> **Read-only.** Do not fix, edit, regenerate, or weaken anything. You are the adversary,
> not the mechanic.

---

## 0. Read order

1. `PLAN.md` — the MACRO map: the tree (§2), the route (§5), the frontier and the
   quality/reporting distinction (§6). Generated from `scripts/plan_spec.py`; check
   `python3 scripts/plan.py --check`.
2. `REPORT_CARD.md` — the plan section (node verdicts, root, frontier readings) first, then
   the B scorecard, the gate-of-record section, and **every open row** (it is the only card;
   there is no separate audit card).
3. `DECISIONS.md` — the pre-registrations and measured outcomes (`NODE`/`ROUTE`/`MOVE` and
   `MOVEMENT` lines).
4. `STATUS.md` — the claimed state and blockers.
5. `docs/report_cards/experiments/` — every receipt the card cites.
6. `AUDIT.md` — the single BACKWARD file: principle audit, macro verdict, findings, and the
   `BACKWARD_REQUIREMENTS_JSON` block already on the card.
7. `docs/SPINE.md`, `AGENTS.md` — the goal, laws, and certification integrity rules.
8. The code and tests at the recorded commit.
9. The findings already in `AUDIT.md` — their disposition.

Record the exact **commit, behavior hash, artifact tag/hash, and card hash** at the top
of the review. A review of an unrecorded tree is not evidence.

---

## 1. The claim under attack

> "Under the frozen contract, the card is ENCODER DONE v1 and the evidence supports the
> project's completion claim; the plan tree is the right decomposition and the route is the
> true frontier."

Attack three things separately:

- **Tree validity (MACRO)** — is the decomposition the right one, is each node necessary, is
  every leaf under the node where its failure actually lands, and are the statuses derived
  honestly?
- **Measurement validity (MICRO)** — is each green row actually green under the frozen rules?
- **Construct validity** — if every row is green, does that mean the project's real goal
  is achieved? Or is the card measuring a proxy that can be green while the goal is not?

A finding that a green row is a tie, a leak, a stale receipt, a self-defined criterion,
a selected hyperparameter, or an excluded failure is **BREAKING**: it invalidates the
all-green claim. A finding that a node's status is misderived, that a reporting green is
being used as quality evidence, that the route skips the real frontier, or that the tree is
missing a necessary condition is **BREAKING** when the completion claim depends on it.

---

## 2. Rules of engagement

- **Read-only.** Do not edit receipts, code, gates, cards, or docs (except writing this
  review).
- **Attack the strongest version** of the claim, not a straw man.
- **Every finding needs evidence:** a file:line, a receipt key, a number, a command, or
  a concrete missing receipt. "I am not convinced" is not a finding.
- **No re-litigating settled decisions** without new evidence. Cite the decision and the
  new measurement.
- **No gate changes as a fix.** If the only fix is a gate, threshold, criterion, or
  exclusion change, mark it **operator-only** and stop.
- **Do not fix.** Report the finding and the exact forward subgoal that would close it.
- **Separate severity** (see §3). A MAJOR finding does not block completion unless it is
  upgraded with evidence.
- **State what would change your mind.** Every finding gets the measurement that would
  falsify it.
- **Prefer falsification to confirmation.** If a row can be broken by a cheap check,
  run the check (read-only) and report the number.

---

## 3. Severity taxonomy

| Severity | Meaning | Effect |
|---|---|---|
| **BREAKING** | Invalidates the green claim, the tree/macro verdict, or the completion condition (tie, leak, stale hash, self-defined criterion, post-hoc gate, selected-on-test parameter, hidden row, unadopted receipt, wrong comparator, non-reproducible claim, construct-validity failure the claim depends on). | The tick-tock returns to FORWARD. Completion is blocked. |
| **MAJOR** | Material weakness or missing evidence that does not, by itself, invalidate the green claim. | Convert to a FORWARD subgoal or get explicit operator acceptance. |
| **MINOR** | Documentation, clarity, hygiene, or a small robustness gap. | Fix opportunistically; does not block. |
| **INFO** | Disclosure or a question for the operator. | No action required. |

---

## 4. Attack dimensions (go through all of them)

### 0. The tree (MACRO) — attack the map before the territory

This dimension exists because a green card can be built on the wrong decomposition. Run it
first, and do not let row-level detail substitute for it.

- **Necessity.** For each level-1 problem (`P1`–`P5`), remove it in thought: does the goal
  still follow without it? If yes, the tree has a redundant node (finding); if the goal
  obviously needs something not present, the tree has a **gap** — name it as a proposed node
  (operator-level plan change, never an edit here).
- **Placement.** Is every leaf under the node where its failure actually lands? A trunk-capacity
  condition filed under "reproducible", or a decision failure filed under "accessible", would
  hide the bottleneck the frontier should point at.
- **Gate kind.** For every green leaf, read its `Green means` on the card. Is any `reporting`
  green used as *quality* evidence anywhere — in the DEC outcome, STATUS, the plan, or the
  card's prose? (E.g. "B3.2 PASS" cited as proof the shape is good when its gate is
  `attribution recorded`.) The classification is derived in `scripts/b_spec.py`; the **misuse**
  is the finding.
- **Closure.** Is `E1` (the DEC-frozen leaf) closed only by its pre-registered port receipt and
  a recorded decision — not by a lab receipt, a local eff_rank, or narration? Is any v1 leaf
  claimed without its stack gate?
- **Route honesty.** Is `CURRENT_ROUTE` the cheapest open blocker on the dependency path? Is a
  step skipped (e.g. the port run before both dev-rig structures are mastered)? Is the pointer
  moved without the evidence its own row names?
- **Derivation.** Recompute `scripts/plan_spec.statuses` from the receipts/card verdicts and
  compare with the card's plan table and root. Any disagreement — a node PASS the rows do not
  support, a root stronger than its leaves — is **BREAKING**.
- **Coverage and freshness.** Every frozen B/judge row is a leaf exactly once; `PLAN.md` is
  current with the spec (`plan.py --check`); the card is current with the tree; `PLAN.md`
  contains no measurement.
- **Sufficiency of the tree.** Name any necessary condition of the goal that has no node at all
  (today: anything the goal needs that is neither a B row, `E1`, nor a v1 gate). That is the
  deepest macro finding and it is operator-level.

### A. Goal and construct validity

- Is **PATH B > PATH A** the real goal, or a proxy? The business objective is
  discounted **incremental gross margin**. Does a green trait card imply better
  decisions? Is the RL leg's policy value incremental, or correlated?
- Is the **demo stream** representative? Is the synthetic generator's structure
  degenerate (uniform `segment`, constant `argmax`, hidden `d0`/`touch_pref`, action
  randomization)? Does a green card on this stream transfer to a real one?
- Are the **four consumers** the right set? Is there an unknown consumer the coverage
  lattice misses? Is the `U` metric minimax over a declared family, and is the family
  published?
- Is the **frozen encoder** actually better than fine-tuning or a strong table? Is the
  per-stream model contract viable at production scale? Is the cost per capability
  justified?
- Is the **ledger companion** doing the work? Would the brain alone pass? Does the
  delivery column hide a brain failure?
- Is the **simulator** causal only where the stream randomized? Is the dose-response
  curve identified? Is `argmax_optimum` unpassable by construction?
- Is "universal" used anywhere the coverage rows do not derive? The word must not appear
  in a generated artifact unless the card says so.

### B. Measurement validity (attack every green row)

For **every** leg-column, stage, and hygiene row:

- Is the comparator the **same head, same rows, same splits**? Is it finite and fair?
- Is the **paired CI** computed on the same customers/pairs as the point estimate? Does
  it exclude zero? Is it a point estimate dressed as a win?
- Is **N** recorded? Was N reduced by OOM/early stop? Are two Ns compared without saying
  so?
- Are **controls** below the brain and *readable*? Is a control at R² < −1 counted as
  evidence?
- Is the evaluation **circular**? Does the arm contain the oracle's features? Is the
  ledger/table duplicated inside `E`? Is the learned-only attribution present?
- Is there **label leakage**? Did a target key, future event, or downstream table enter
  the encoder or the feature path? Is the `as_of` cut strict?
- Is the **target transform** the right algebra (raw additive mass vs log1p)? Does the
  transform favor one arm?
- Was a **hyperparameter selected on the scored split** (e.g., a whitening rank swept on
  B)? Was a threshold, exclusion, or k chosen after seeing the result?
- Is an **exclusion** honest? A degenerate target may be excluded-and-reported only
  under the frozen rule. Is the exclusion hiding a real failure?
- Are **multiple comparisons** controlled? Is the best of many arms reported as if
  pre-registered?
- Is the receipt's **behavior hash** current? Is the block stamp current? Is the card
  newer than the receipt? Is an unadopted protocol receipt on the live card?
- Does the receipt **define its own criterion**? Is the criterion in the frozen code?
- Can the number be **regenerated by one repo command** at the recorded commit? Are the
  inputs hashed? Is the receipt under `/tmp`?
- Does the card **contradict** its receipts, its own prior text, or the audit? Is a row
  hidden or renamed?

### C. Data and generator

- Are the latent traits independent of behavior (segment), constant (argmax), or
  unobservable (d0/touch_pref)? Is a green distance/geometry claim measuring noise?
- Is the randomized action design real? Are the known effects identified and strong
  enough? Is the holdout randomized and untouched?
- Is the stream nonstationary, survivorship-biased, or label-immature? Does the demo's
  as-of cut leave enough history/future?
- Is the sample size enough for the claimed CIs? Are the splits stable across seeds?

### D. Model, serving, and artifacts

- Is the **train/serve tensor** the same? Does the served `E` read the tensor the heads
  trained on? Is the boundary/isometry what the card claims? Is the projection rank
  sufficient?
- Is the **geometry metric** lawful (fit on A, frozen, per-sample at serve, no per-batch
  ZCA, no rank manufacture)? Is the distance claim measured, not asserted?
- Is the **bucket dilution** hiding content (e.g., a 4-d signal in 600 dims)? Is the
  ledger a table in disguise? Do G1–G5 hold mechanically?
- Are **capacity, rank, isotropy, and mass exactness** measured on the artifact, not
  inferred?
- Is **determinism** real (same seed → identical)? Are there silent clamps, NaNs, or
  rescued divergences?
- Are the **caps** enforced (vLLM 11/15 GiB, GMU ≤ 0.70, trainer 20 GB, 8 GB floor)?
  Was a second LLM container run? Was the trainer capped?
- Is the **artifact contract** self-describing (version, hash, as_of, freshness,
  buckets, geometry stats)? Does a consumer self-validate and refuse a stale artifact?
- Is the **retrain cadence** measured from degradation, not a static interval? Is delta
  scoring exact? Is a warm start certified against a cold control?

### E. Process and integrity

- Was the protocol **pre-registered before the run**? Was the acceptance frozen? Was the
  **named axis** the one that moved? Were the rejected alternatives recorded?
- Was a FAIL reported honestly, or relabeled? Was a gate weakened after seeing the
  result? Was an old decision edited?
- Is the branch **clean and recorded**? Are the receipts committed? Is the card
  regenerated at the recorded commit? Is the tree dirty?
- Do the **tests** pass? Are the known failures disclosed? Are tests stale relative to
  adopted decisions? Is `principle_audit.py --strict` clean? Is `ruff` clean?
- Is the code aligned to SPINE (T/A/C-series laws, per-sample contract, no cross-layer
  imports, no per-row loops, vectorized hot paths, derived literals)?
- Are there **open operator rulings** that block a row? Is a [TECHNICAL] decision
  masquerading as settled? Is silence treated as assent?
- Is the **card current**? Is `STATUS.md` current? Is `DECISIONS.md` append-only? Does
  `REPORT_CARD.md` agree with its receipts and its generated assessment?

### F. Economics and operations

- What is the **cost per unit capability** (train, inference, retrain, storage)? Is the
  capability worth it?
- Is the **serving path** O(1) and incremental? Is the artifact small enough? Is the
  ledger accumulation cheap?
- Is the **deployment** path real (schema, freshness, refusal, monitoring, rollback)?
- Is the **data** available, legal, and stable? Is the label maturity a bottleneck?
- Is there a **moat**? Why would this beat a strong incumbent? What is the opportunity
  cost?

---

## 4b. Principle-alignment audit (binding)

A green card can still be built on code that violates the principles. Audit the work against
AGENTS.md principles 1–16 and the top-level charter, not just the metrics. For every principle,
record **aligned / gap / violated / not assessed** in `AUDIT.md` with evidence.

### How to test (specific, not vibes)

- **Universal (P1):** renamed-schema + new-field end-to-end with zero code edits; an unseen
  stream refuses (no cross-application); no business field names or domain literals in code;
  per-stream model.
- **Adaptive (P2):** no fixed `dim`/`layers`/`epochs`/`seq_len`/thresholds; values derived at
  fit; `--set` overrides receipted; runtime toggles in config, not code.
- **Lawful (P3):** merge/equivariance/mass-additivity receipts; per-sample at serve; no
  batch-coupled normalization; additive mass scored in raw units; no un-faded second copy of
  history; the correct algebra per quantity.
- **Efficient / SPEED (P4, E15, E16):** search the hot paths for per-row Python loops
  (`for ... in range(len(`, `.iterrows()`, `.itertuples()`, Python loops over customers/time in
  `looking_glass/` or `scripts/measure_*`); require scans/gathers/einsum/Polars; run
  `python3 scripts/principle_audit.py --strict`; time the hot path and compare to a vectorized
  alternative; check O(1) serve, constant state, and caps.
- **Robust (P5, E7):** determinism (same seed → identical); divergence = hard failure; no
  silent clamps/NaNs/swallowed exceptions; OOD → refuse or widen; memory caps enforced.
- **Honest (P6):** pre-registration before the run; frozen acceptance; paired CIs; controls;
  no training on the graded metric; no post-hoc gate edits; no unadopted receipts.
- **Accessible (P7):** linear shipping head; kNN for unsupervised; MLP diagnostic only;
  decodability and per-bucket parity; no hidden bucket dilution.
- **Diagnosable (P8):** each red row names its axis; the protocol changes one axis; rejected
  alternatives recorded; coupling and trade-offs disclosed.
- **Engineering (E1–E16):** magic numbers; convergence vs counts; learned vs taught;
  schema adaptation; modularity and cross-layer imports; single source of truth; uncertainty;
  measurement; versioning; observability; intent; deletion; best tool; cost; speed.

### What counts as a finding

A principle finding needs:

- the **principle** and the **mechanism** (file:line, code pattern, or design choice);
- the **consequence** (e.g., "4.2 s per 1k rows, O(N) Python overhead"; "batch-coupled ZCA at
  serve"; "fixed `dim=256`"; "comparator trained on the scored split");
- a **reproduction** (command, benchmark, grep, test);
- a **severity**: **BREAKING** if it invalidates a green claim or the project's ability to reach
  the goal; **MAJOR** if it is a material principle violation that must become a forward
  subgoal; **MINOR/INFO** otherwise;
- an entry in `AUDIT.md` **and** a matching `B-<id>` requirement with `principle` and
  `audit_ref`.

**The canonical example:** a per-row Python loop in a hot path that should be vectorized
violates SPEED/E16. In a committed evidence script it is at least **MAJOR** (the standing rule
requires evidence scripts to be vectorized and pass `principle_audit.py --strict`); in a one-off
diagnostic that is not evidence it may be **MINOR** — but it still goes in `AUDIT.md`.

### Do not

- Do not call a principle "aligned" because the card is green.
- Do not file a style nit as a principle violation.
- Do not propose a gate/threshold change as the fix.
- Do not accept "it runs" as evidence of speed or correctness; time and measure it.

---

## 5. Required output: one file — `AUDIT.md`

Write exactly this shape:

```text
# AUDIT — adversarial review <date>
## Header
  commit, behavior hash, artifact tag/hash, card hash, reviewer, date, mode (FULL | NODE:<node>)
## Verdict
  BREAKING / NO BREAKING FINDINGS (FULL only; NODE mode has no completion verdict)
## Macro verdict (the tree)
  | Node | Claimed (card) | Recomputed (plan_spec.statuses) | Honest? | Finding |
  every node status recomputed; route pointer checked; `Green means` misuse checked;
  tree sufficiency (any necessary condition with no node)
## Findings
  | ID | Node | Severity | Claim attacked | Why it may be false | Evidence (file:line / receipt key / number) |
  | Reproduction command | Proposed FORWARD subgoal | Gate touched? | What would change my mind |
## Per-row attack
  | Row (leg-column / stage / hygiene) | Attacked? | Finding ID(s) | Notes |
  Every green row must be attacked or explicitly marked "not attacked because ...".
## Goal-level findings
  The top reasons the project may not achieve the goal even if the card is green.
## Process audit
  Pre-registration, frozen acceptance, named axis, honest FAIL, freshness, commit,
  receipts, tests, audit, branch alignment, open rulings.
## Handoff to FORWARD
  For every BREAKING/MAJOR: a proposed subgoal with a named axis, a frozen-acceptance
  skeleton, the receipt that would close it, and the cost.
## Stop condition
  What must be true for the next BACKWARD to return NO BREAKING FINDINGS.
```

The review is committed with the next forward chunk. It is evidence, not narration:
every finding cites a receipt/line/number or a concrete missing receipt.

### Structured handoff (required)

For **every** finding, append a requirement to the `BACKWARD_REQUIREMENTS_JSON` block at the end of `AUDIT.md`:

```json
{
  "id": "B-<n>",
  "severity": "BREAKING|MAJOR|MINOR|INFO",
  "node": "P2/E1 or P4/JUDGE-supervised or G — the plan node the finding attacks",
  "principle": "P4/E16 (SPEED) or 'none' for a measurement finding",
  "status": "open",
  "finding": "one sentence, with the receipt/number",
  "forward_subgoal": "the named-axis subgoal (with NODE/ROUTE/MOVE) that would close it",
  "gate_touched": false,
  "evidence": "file:line or receipt key",
  "audit_ref": "A-<n>",
  "date": "YYYY-MM-DD"
}
```

Append the matching entry to the **Open findings** table in `AUDIT.md` with the same ID, and
update the principle's scorecard row. The JSON block is the machine gate; the narrative is
the record. They may not disagree. No other file is written.

The renderer displays this file in `REPORT_CARD.md`; an open `BREAKING` or `MAJOR`
requirement blocks the completion condition. FORWARD closes a requirement only by
appending the measured receipt/decision that resolves it and setting `status` to
`closed` (or, for `MAJOR` only, the operator sets `accepted` with a reason). `BREAKING`
is never accepted, only closed.

---

## 6. The tick-tock protocol

```text
state = FORWARD
reviewed = set()                             # scopes already FULL-reviewed at this tree
loop:
    run PROMPT_FORWARD                       # MACRO statement + one MICRO run -> card
    if a plan node changed status or CURRENT_ROUTE moved:
        run PROMPT_BACKWARD (NODE:<node>)    # scoped, read-only, requirements tagged

    scope = SOLVED        if card.plan.root == SOLVED
            ENCODER_DONE  if encoder_done_v1(card)
            None          otherwise
    if scope is None:
        continue                             # not at a reviewable milestone

    if scope not in reviewed:
        run PROMPT_BACKWARD (FULL) -> AUDIT.md
        if verdict == BREAKING:              # includes a dishonest macro verdict
            append a DECISIONS entry (review + findings)
            update STATUS.md (blockers, frontier)
            convert each BREAKING finding to a FORWARD subgoal with NODE/ROUTE/MOVE
            reviewed.clear()
            continue
        reviewed.add(scope)

    if scope == SOLVED:
        print REPORT_CARD.md
        stop                                 # project SOLVED
    # ENCODER DONE v1 certified: keep going at the next route step (port, B rows, B+C, v1)
```

Rules for the transition:

- **Two review modes, one standard.** NODE is scoped: it attacks the changed node and its
  leaves, may file BREAKING/MAJOR requirements, and never produces a completion verdict.
  FULL attacks the tree, every row, and the goal, and is the only mode that can stop the
  tick-tock.
- **FULL runs at milestones, not every loop.** It runs once per scope (`ENCODER DONE`, then
  `SOLVED`) at a given tree; a NODE review covers the iterations in between. A BREAKING
  finding voids the scope, so the next milestone re-runs FULL.
- **BACKWARD never fixes.** It reports; FORWARD implements. A BREAKING finding becomes a
  pre-registered FORWARD subgoal with its own `NODE`/`ROUTE`/`MOVE` and frozen acceptance.
- Do not fold several breaking findings into one uncontrolled run.
- If a finding's only fix is a gate/threshold/criterion/exclusion change, or a tree change,
  it is **operator-only**; completion is blocked until ruled.
- Do not re-run FULL on a red card to shop for a better verdict.
- **No completion claim** without all three: (a) the card's generated root says `SOLVED`,
  (b) the FULL review at that scope returns NO BREAKING FINDINGS, and (c) its macro verdict is
  honest. A hand-written "SOLVED" is a bug even if all three hold.

---

## 7. Specific attack checklist (run it)

### The tree (MACRO) — do this before the rows

- [ ] Every `P1`–`P5` is necessary: removing it breaks the goal's derivation.
- [ ] Every leaf hangs under the node where its failure actually lands.
- [ ] Every green leaf's `Green means` is respected: no `reporting` green promoted to quality
      evidence in the DEC/STATUS/plan/card prose.
- [ ] `E1` is open unless the port receipt meets DEC-572's bars; v1 leaves are open unless the
      stack gate passed.
- [ ] `CURRENT_ROUTE` is the cheapest open blocker on the dependency path; no step skipped.
- [ ] `plan_spec.statuses` recomputed from receipts equals the card's node verdicts and root.
- [ ] Every frozen row is a leaf exactly once; `plan.py --check` passes; the card matches the
      tree; `PLAN.md` carries no measurements.
- [ ] No necessary condition of the goal is missing from the tree (name it if so — operator).

### Card rows

- [ ] Every leg-column: same head, same rows, same splits, finite comparator, paired CI
      excluding zero, N recorded, controls below the brain and readable.
- [ ] Every stage: the working-state proof is a receipt, not a narrative; the failing
      sub-check is named; no stale block stamp.
- [ ] Hygiene 5/5: environment, behavior hash, N, controls, finite/fair comparator.
- [ ] S6 6/6: one invocation, deterministic render, clean recorded commit, hashes, every
      row reports N.
- [ ] No UNKNOWN rendered as PASS; no FAIL hidden as EXCLUDED; no excluded row without
      the frozen rule.
- [ ] No unadopted protocol receipt on the live card; the recorded artifact is the one
      the card names.

### Claims that have failed before (check them first)

- [ ] A point estimate above the comparator with a CI that includes zero (x0 0.909;
      action profile 0.9082).
- [ ] A target transform that favors the comparator (log1p mass; raw mass passes).
- [ ] A hyperparameter selected on the scored split (whitening k swept on B; 512-d
      ladder).
- [ ] A degenerate target (uniform segment; constant argmax; independent d0/touch_pref).
- [ ] A ledger/table result booked as a brain win.
- [ ] A stale behavior hash or a receipt from an older tree.
- [ ] A control that is unreadable (R² < −1) but counted.
- [ ] A receipt that defines its own criterion.
- [ ] A card that contradicts its receipts or the audit.
- [ ] A test suite red on stale expectations, or skipped tests counted as green.
- [ ] A "trained" objective that fit its target but did not move the judged axis
      (profile, survival, rate, capacity).
- [ ] A distance/geometry claim whose metric was never measured (anisotropy).
- [ ] A simulator/RL claim that is correlational, not incremental.
- [ ] A "universal" claim over a declared family only.

### Goal-level

- [ ] Does the card's green imply the **business** goal (discounted incremental margin)?
- [ ] Does the demo stream's generator make any row trivially green or unpassable?
- [ ] Is the frozen/universal claim tested beyond our known tasks?
- [ ] Is the cost per capability defensible?
- [ ] Is the deployment/refresh contract real?
- [ ] Is there a stronger incumbent that makes the whole approach unnecessary?

---

## 8. What a clean BACKWARD looks like

A clean review is not "everything is fine." It is:

- **MACRO clean:** every node recomputed and honest; the route pointer is the true frontier;
  no reporting green used as quality; `E1`/v1 closures backed by their receipts; the tree is
  sufficient for the goal (no missing necessary condition);
- every green row attacked with a concrete check and a stated result;
- every attack either falsified (with a receipt/number) or recorded as a finding;
- no BREAKING finding;
- every MAJOR finding either converted to a forward subgoal or explicitly accepted by
  the operator;
- the review committed at the recorded commit/hash;
- the generated card's plan root is **honest for the scope claimed**: ENCODER DONE may sit
  under an OPEN root (v1 and/or `E1` unmeasured) — the review must say so; `SOLVED` is
  claimed only when the root itself says `SOLVED`.

Only then does the encoder tick-tock stop. The project is not done until the route reaches
the v1 step and a FULL review at a `SOLVED` root returns no BREAKING findings — no matter how
green the card looks.
