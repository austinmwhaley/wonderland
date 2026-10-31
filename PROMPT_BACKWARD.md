# PROMPT_BACKWARD — the adversarial review prompt

> **State:** run this prompt **only** when `PROMPT_FORWARD.md` has produced an
> all-green `REPORT_CARD.md` gate-of-record section (the definition in
> `PROMPT_FORWARD.md` §4).
> **Job:** try to falsify the completion claim. Explain **why the project may not
> achieve its goal**, even if every row is green. Document every finding for the next
> forward run.
> **Read-only.** Do not fix, edit, regenerate, or weaken anything. You are the adversary,
> not the mechanic.

---

## 0. Read order

1. `REPORT_CARD.md` — the run report and the all-green gate-of-record section.
2. `docs/AUDIT_CARD.md` — the full research audit and its open frontier rows.
3. `DECISIONS.md` — the pre-registrations and measured outcomes.
4. `STATUS.md` — the claimed state and blockers.
5. `docs/report_cards/experiments/` — every receipt the card cites.
6. `docs/backward_requirements.json` — the requirements already on the card.
7. `AUDIT.md` — the principle-alignment audit and its open findings.
8. `docs/SPINE.md`, `AGENTS.md` — the goal, laws, and certification integrity rules.
9. The code and tests at the recorded commit.
10. The last `docs/BACKWARD_CARD.md` — findings already raised, and their disposition.

Record the exact **commit, behavior hash, artifact tag/hash, and card hash** at the top
of the review. A review of an unrecorded tree is not evidence.

---

## 1. The claim under attack

> "Under the frozen contract, the card is all green and the evidence supports the
> project's completion claim."

Attack two things separately:

- **Measurement validity** — is each green row actually green under the frozen rules?
- **Construct validity** — if every row is green, does that mean the project's real goal
  is achieved? Or is the card measuring a proxy that can be green while the goal is not?

A finding that a green row is a tie, a leak, a stale receipt, a self-defined criterion,
a selected hyperparameter, or an excluded failure is **BREAKING**: it invalidates the
all-green claim. A finding about the goal's construct validity is **BREAKING** if the
completion claim depends on it.

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
| **BREAKING** | Invalidates the all-green claim or the completion condition (tie, leak, stale hash, self-defined criterion, post-hoc gate, selected-on-test parameter, hidden row, unadopted receipt, wrong comparator, non-reproducible claim, construct-validity failure the claim depends on). | The tick-tock returns to FORWARD. Completion is blocked. |
| **MAJOR** | Material weakness or missing evidence that does not, by itself, invalidate the green claim. | Convert to a FORWARD subgoal or get explicit operator acceptance. |
| **MINOR** | Documentation, clarity, hygiene, or a small robustness gap. | Fix opportunistically; does not block. |
| **INFO** | Disclosure or a question for the operator. | No action required. |

---

## 4. Attack dimensions (go through all of them)

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

## 5. Required output: `docs/BACKWARD_CARD.md` + `docs/backward_requirements.json` + `AUDIT.md`

Write exactly this shape:

```text
# BACKWARD_CARD — adversarial review <date>
## Header
  commit, behavior hash, artifact tag/hash, card hash, reviewer, date
## Verdict
  BREAKING / NO BREAKING FINDINGS
## Findings
  | ID | Severity | Claim attacked | Why it may be false | Evidence (file:line / receipt key / number) |
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

For **every** finding, append a requirement to `docs/backward_requirements.json`:

```json
{
  "id": "B-<n>",
  "severity": "BREAKING|MAJOR|MINOR|INFO",
  "principle": "P4/E16 (SPEED) or 'none' for a measurement finding",
  "status": "open",
  "finding": "one sentence, with the receipt/number",
  "forward_subgoal": "the named-axis subgoal that would close it",
  "gate_touched": false,
  "evidence": "file:line or receipt key",
  "audit_ref": "A-<n>",
  "date": "YYYY-MM-DD"
}
```

Append the matching entry to `AUDIT.md` under **Open findings** with the same IDs, and
update the principle's scorecard row. The machine-readable file is the gate; `AUDIT.md` is
the detailed record. They may not disagree.

The renderer displays this file in `REPORT_CARD.md`; an open `BREAKING` or `MAJOR`
requirement blocks the completion condition. FORWARD closes a requirement only by
appending the measured receipt/decision that resolves it and setting `status` to
`closed` (or, for `MAJOR` only, the operator sets `accepted` with a reason). `BREAKING`
is never accepted, only closed.

---

## 6. The tick-tock protocol

```text
state = FORWARD
loop:
    run PROMPT_FORWARD  ->  REPORT_CARD.md (gate of record)
    if not all_green:                 # PROMPT_FORWARD §4
        state = FORWARD
        continue
    run PROMPT_BACKWARD ->  docs/BACKWARD_CARD.md
    if verdict == BREAKING:
        append a DECISIONS entry (review + findings)
        update STATUS.md (blockers)
        convert each BREAKING finding to a FORWARD subgoal
        state = FORWARD
        continue
    else:                             # NO BREAKING FINDINGS
        record completion in STATUS.md/DECISIONS.md
        print REPORT_CARD.md
        stop
```

Rules for the transition:

- **BACKWARD never fixes.** It reports; FORWARD implements.
- A BREAKING finding becomes a **pre-registered FORWARD subgoal** with its own named
  axis and frozen acceptance. Do not fold several breaking findings into one
  uncontrolled run.
- If a finding's only fix is a gate/threshold/criterion/exclusion change, it is
  **operator-only**; completion is blocked until ruled.
- The next BACKWARD runs only after the next **all-green** card. Do not re-run BACKWARD
  on a red card to shop for a better verdict.
- **No completion claim** without both: (a) the card's generated assessment says the
  completion condition, and (b) BACKWARD returns NO BREAKING FINDINGS. A hand-written
  "SOLVED" is a bug even if both hold; the card's generated assessment is the only
  completion statement.

---

## 7. Specific attack checklist (run it)

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

- every green row attacked with a concrete check and a stated result;
- every attack either falsified (with a receipt/number) or recorded as a finding;
- no BREAKING finding;
- every MAJOR finding either converted to a forward subgoal or explicitly accepted by
  the operator;
- the review committed at the recorded commit/hash;
- the generated card's assessment still says the completion condition.

Only then does the tick-tock stop. Until then, the project is not done — no matter how
green the card looks.
