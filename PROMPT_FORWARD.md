# PROMPT_FORWARD — the forward operating prompt

> **State:** run this prompt while the project is **not** all-green, or after
> `PROMPT_BACKWARD.md` returned at least one **BREAKING** finding.
> **Job:** move the project toward its goal by one measured axis per run, obeying the
> principles. This is not a coding prompt; code is only the instrument.
> **Every run ends with results written to `REPORT_CARD.md`.** When its gate-of-record
> section is all green, stop forward work and run `PROMPT_BACKWARD.md`.

---

## 0. Read order (do not skip)

1. `AGENTS.md` — doctrine, principles, certification integrity, workflow.
2. `docs/SPINE.md` — the constitution: goal, four consumers, judge, S1–S6, T/A/C-series
   laws, bucket table, ledger guardrails.
3. `STATUS.md` — live state (macro + micro). Read first to restore context.
4. `DECISIONS.md` — append-only decision log. Read the tail; never re-litigate a
   settled decision without new evidence.
5. `REPORT_CARD.md` — the run report and gate-of-record section (see §4).
6. `docs/JUDGE_CARD.md` — the rebuild's generated gate of record.
7. `docs/AUDIT_CARD.md` — the full research audit; never hide an open row.
8. `docs/report_cards/experiments/` — the receipts every number must trace to.
9. `PLAN.md`, `loop/UNIVERSALITY.md` — forward plan and Layer-1 contract.

If a file contradicts another, the order is: **AGENTS.md doctrine → SPINE laws →
frozen gate definitions → receipts → STATUS/DECISIONS narrative**. Narrative never
overrides a receipt.

---

## 1. The goal (one sentence, from SPINE §0)

> Input any event stream → output a **frozen** foundation model (one per stream) whose
> state supports **supervised**, **unsupervised**, **RL**, and **simulator** heads using
> simple consumers.

Layer 0 is the frozen universal donor. Layer 1 is certification: sufficiency,
minimality, decision-sufficiency, measured by `U`. Do not conflate them.

The judge is **PATH B (event stream) > PATH A (wide table)** on the consumer battery,
with paired CIs and controls. The two-number rule is binding: **brain alone = progress;
brain + ledger = delivery.** Ledger points never count as a brain win.

---

## 2. Principles that bind every run

- **Conflict order:** Robust → Universal → Adaptive → Lawful → Efficient → Honest.
  Lawful and Robust outrank convenience.
- **The one rule:** never encode an answer the system could learn or derive.
- **Anti-overfit clause:** Layer B is self-supervised only; known use cases may VETO
  (gates), never SHAPE the encoder; never train on the graded metric.
- **Accessibility:** information must be recoverable by simple consumers (linear for
  supervised shipping; kNN for unsupervised). MLP is diagnostic only.
- **Honesty:** pre-register before the run; grade against frozen acceptance; report
  FAIL; never weaken a gate to pass.
- **Efficiency:** one axis per run, cheapest diagnostic first, linear-time and
  vectorized, no per-row Python loops in hot paths.
- **Robustness:** divergence is a hard failure; reject or widen uncertainty, never
  clamp; deterministic and seeded.

---

## 3. The forward loop (binding)

For every run:

1. **Generate the card** — run the one command that regenerates it (see §6). Read the
   gate-of-record section and the blocking rows.
2. **Pick the highest-leverage unmet subgoal** that is achievable now; cheapest first;
   exactly **one axis**.
3. **Diagnose before protocol** — if a cheap read-only test can falsify the premise,
   run it first and record it as a diagnostic (DEC-462/464 rule). A diagnostic may not
   adopt anything.
4. **Pre-register the decision in `DECISIONS.md` before the run.** It must name:
   context; the principle(s) served; ≥2 rejected alternatives; the single axis changed;
   the named metric; the frozen acceptance threshold; guards; the falsifier; the cost;
   and what is **not** changed (gates, comparator, probes, E's content, ledger).
5. **Implement the one change** and run the one command that emits the receipt. The run
   is the measurement.
6. **Grade against the frozen acceptance before any verdict.** Adopt only if the named
   axis moved. Otherwise revert, archive the receipts under
   `docs/report_cards/experiments/protocols/<name>/`, and restore the live receipts of
   the recorded artifact.
7. **Regenerate the card at a clean, recorded commit**; write the run result to
   `REPORT_CARD.md`; update `STATUS.md`; append the measured outcome to `DECISIONS.md`;
   commit code + docs (never data/artifacts).
8. **If the gate-of-record is not all green, repeat.** If it is all green, run
   `PROMPT_BACKWARD.md`.

A run that cannot name its axis, its frozen acceptance, or its receipt is not a run.

---

## 4. The cards and the completion contract

This branch has two cards. They are **not interchangeable** (DEC-415):

| Card | What it is | Role |
|---|---|---|
| `docs/JUDGE_CARD.md` | the rebuild's generated judge (8 leg-columns × 6 stages, freshness hash, anti-false-PASS tests) | **gate of record** |
| `REPORT_CARD.md` | the run report; its mandatory top section is the gate of record; the inherited B+B+C card is **disclosure only** | **the artifact every run writes** |

**Hard rule:** `REPORT_CARD.md` may never be green while `docs/JUDGE_CARD.md` is not.
The inherited B+B+C section (28/28 + 6/6 on the old pipeline) is status-quo disclosure,
not evidence for the rebuild. If the operator ever rules that `REPORT_CARD.md` itself
is the literal gate of record, that is a **gate-touching** change and must be recorded
before use.

### Required `REPORT_CARD.md` shape

Every run writes/regenerates:

```text
# REPORT_CARD — run <id> · <date>
## Gate of record (rebuild)
  - generated from docs/JUDGE_CARD.md; behavior hash; commit; artifact tag/hash
  - legs green x/8; stages green x/6; assessment (NOT DONE / ...)
  - every blocking row with its exact missing receipt
## Run decision
  - DEC id, named axis, frozen acceptance, measured value, verdict
  - adopted / reverted; receipts (paths); N; controls
## Card delta
  - rows that moved; rows that went UNKNOWN/stale; rows excluded-and-reported
## Backward status
  - last BACKWARD verdict; open BREAKING findings; date
## Provenance
  - exact commands; hashes; clean-tree commit; N; data/as_of
## Inherited status-quo card (disclosure only — not the rebuild gate)
  - the existing B+B+C card content, clearly labelled
```

No number may appear in the run sections that is not copied from a generated
receipt/card. If the renderer for this shape does not exist yet, **building it is the
first FORWARD subgoal**; do not hand-edit the card.

### "All green" — the only definition that counts

All of the following, simultaneously:

1. `docs/JUDGE_CARD.md`: **0 FAIL, 0 UNKNOWN**; every one of the 8 leg-columns is PASS
   or **EXCLUDED-and-reported** under a frozen, operator-approved exclusion rule; all
   **6/6 stages** PASS.
2. Hygiene 5/5 (environment, behavior hash, N, controls, finite/fair comparator);
   S6 6/6 (one command, deterministic render, clean recorded commit, artifact hashes,
   every row reports N).
3. Freshness: the receipts' `behavior_hash` equals the current tree; no stale block;
   no unadopted receipt on the card.
4. `REPORT_CARD.md`'s gate section agrees with the generated judge; no contradiction.
5. No open operator ruling blocks a row; every stop-and-ask item is resolved or
   explicitly recorded as a non-blocking disclosure.
6. The generated assessment says the completion condition. A hand-written "SOLVED",
   "universal", or "PASS" is a bug.

EXCLUDED-and-reported is not a free pass: it applies only to a target with no variance
or a baseline that already saturates, and only under the frozen exclusion rule. Proposing
a new exclusion is a **gate change** → operator.

---

## 5. Evidence and certification integrity (binding)

Follow AGENTS.md's certification integrity rules 1–8. In practice:

- **Freeze the gate before the result.** Definitions, thresholds, criteria, and
  selection rules are in `DECISIONS.md` and the receipt **before** the run.
- **A receipt may not define its own criterion.** New criterion strings are UNKNOWN
  unless pre-registered.
- **No PASS from a point estimate.** A PASS needs a finite, fair comparator, a paired
  CI excluding zero, a recorded N, and controls below the brain.
- **No circular evaluation.** The object under test must not contain the oracle's
  features. Report learned-only, ledger-only, shape-only, random, destroyed controls.
- **No post-hoc gate edits.** A criterion change after seeing a result is a new
  protocol version: stop, get operator approval, re-run from scratch, report both.
- **Reproducibility is part of the claim.** One repo command at a clean, recorded
  commit; artifact hashes and N; receipts in the repo, not `/tmp`.
- **Freshness is mechanical.** A behavior change marks dependent receipts stale →
  UNKNOWN until regenerated.
- **Never train on the graded metric.** Known tasks may veto, never shape the encoder.

---

## 6. Documentation and artifact upkeep (mandatory)

| Artifact | What to keep current |
|---|---|
| `STATUS.md` | macro (goal, stage, progress, blockers) + micro (current task, last completed, next 3, environment). Update **before yielding**. |
| `DECISIONS.md` | append-only. Pre-registration before every run; measured outcome after; context / alternatives / decision+principle / trade-offs. Never edit old entries. |
| `REPORT_CARD.md` | the run report + gate-of-record section (§4). Regenerated each run; no hand-written numbers. |
| `docs/JUDGE_CARD.md` | the generated gate of record. Never hand-edit. |
| `docs/AUDIT_CARD.md` | the full research audit; never hide an open row. |
| `docs/report_cards/experiments/` | receipts; `protocols/<name>/` archives for non-adopted runs. |
| `docs/BACKWARD_CARD.md` | the last adversarial review and its open findings. |
| `PLAN.md` | the forward plan; reconcile at every phase transition. |

**Commit discipline:** every coherent verified chunk is committed; code + docs only;
data, checkpoints, logs, and virtualenvs are never committed. Commits stay **local**
until the operator explicitly asks to push. The card is regenerated at a clean recorded
commit; a receipt that no repo command regenerates is not evidence.

### Commands (read the current paths from `STATUS.md`; never guess an artifact)

- One-command chain + card:
  `python3 scripts/measure_certification_law.py --products <recorded artifact> --demo <demo> --train-log <artifact>/train.log`
- Judge render: `python3 scripts/judge_card.py`
- Inherited status-quo card (disclosure only): `python3 scripts/report_card.py`
- Principle audit: `python3 scripts/principle_audit.py --strict <changed evidence scripts>`

The chain command is the only S6-certifying path. A hand-rendered card is not evidence.

**Evidence scripts:** any script committed as evidence must pass
`python3 scripts/principle_audit.py --strict` with **0 findings**, be vectorized (no
per-row Python loops in hot paths), and emit a runtime receipt.

---

## 7. Resources and operating constraints

- **One GPU job at a time.** Derive budgets from VRAM/hardware; keep the CPU fallback.
- **Caps are the operating model:** vLLM 11 GiB RAM / 15 GiB total, GMU ≤ 0.70
  (refuse-above); training container 20 GB; system floor 8 GB. The watchdog is opt-in
  (`--watchdog`), not the default.
- **Co-tenancy:** validate the trainer's budget against the resident LLM before a run;
  never run two LLM containers at once; record the low-water/peak memory receipt.
- **No code edits to `looking_glass/` while a run's lazy imports are pending.**
- **Never run destructive deletes, material spend, customer-facing actions, or live
  exploration without explicit operator authorization.**
- **Never push** without explicit permission.

---

## 8. End-of-run checklist (print it, then commit)

1. The one command ran; every producer exited 0; the receipt exists.
2. Frozen acceptance was graded **before** the verdict; the named axis is stated.
3. Adopted → keep; not adopted → revert + archive receipt + restore live receipts.
4. `STATUS.md` updated; `DECISIONS.md` outcome appended.
5. `REPORT_CARD.md` run sections written from generated cards; gate-of-record section
   agrees with `docs/JUDGE_CARD.md`.
6. Clean-tree commit recorded; hashes/N/controls present; no `/tmp`-only evidence.
7. `python3 scripts/principle_audit.py --strict` clean on changed evidence scripts;
   ruff/tests per the repo's gates (or the known-failure baseline disclosed).
8. Print the rendered `REPORT_CARD.md` (never fenced) as the run's source of truth.
9. If all green → stop and run `PROMPT_BACKWARD.md`. Else → next FORWARD run.

---

## 9. Stop-and-ask triggers (operator only)

Stop **before** running if a proposed change:

- changes a gate, threshold, criterion, frontier definition, or exclusion rule;
- changes what "shape" contains, what "full-E" means, or what the comparator is;
- would move a verdict from FAIL to PASS by anything other than a real capability
  improvement;
- introduces a new consumer, artifact contract, bucket, or architecture;
- cannot be reproduced from the repo at a recorded commit;
- requires material spend, destructive deletes, customer-facing exposure, or a push.

Ask once, with the measured evidence and a recommendation. Silence is never assent for
a gate-touching or authorization item.

---

## 10. Forbidden shortcuts (the failure modes this project has already paid for)

- Declaring a point estimate a win when the paired CI includes zero.
- Selecting a hyperparameter on the scored split (e.g., a whitening rank swept on B).
- Training on the target the card grades (circularity), or letting labels reach Layer B.
- Adding an unadopted receipt to the card; leaving a stale behavior hash; hiding a row.
- Weakening a gate, redefining a target, or excluding a row to turn a verdict green.
- Narrating a number instead of generating it; editing the card to improve a verdict.
- Hope-driven reruns of a falsified ladder rung ("512-d next") without a new mechanism.
- Per-row Python loops in hot paths; batch-coupled serving statistics; silent clamps.
- Running a second LLM container, exceeding the caps, or training while the co-tenant
  is unbounded.

---

*The forward run ends when the card is all green. The project is not done until
`PROMPT_BACKWARD.md` returns no BREAKING findings against that green card.*
