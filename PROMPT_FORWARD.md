# PROMPT_FORWARD — the two-scale forward operating prompt (PLAN.md -> REPORT_CARD.md)

> **State:** run this prompt while the card is **not** ENCODER DONE v1, or after
> `PROMPT_BACKWARD.md` returned at least one **BREAKING** finding.
> **Job:** move the project toward its goal by one measured axis per run, obeying the
> principles. This is not a coding prompt; code is only the instrument.
> **Two scales, every run.**
> **MACRO** — `PLAN.md` is the map: the goal cut backwards into necessary problems, the
> route that earns the evidence, and the frontier. A run that cannot name the plan node it
> serves is maintenance, not progress.
> **MICRO** — `REPORT_CARD.md` is the instrument: one leaf, one axis, one frozen gate, one
> receipt. A run that moves no node is a diagnosis, not progress.
> **Every run ends with both artifacts current:** the card re-rendered, and — if the
> frontier moved — `PLAN.md` regenerated. When the card is ENCODER DONE v1, stop forward
> work and run `PROMPT_BACKWARD.md` (FULL).

---

## 0. Read order (do not skip)

1. `AGENTS.md` — doctrine, principles, certification integrity, workflow.
2. `PLAN.md` — the MACRO map: the goal cut backwards into necessary problems (`§2`), the
   route that earns the evidence (`§5`), the frontier and what a green proves (`§6`). Generated
   from `scripts/plan_spec.py`; never hand-edited. `python3 scripts/plan.py --check` must pass.
3. `docs/SPINE.md` — the constitution: goal, four consumers, judge, S1–S6, T/A/C-series
   laws, bucket table, ledger guardrails.
4. `STATUS.md` — live state (macro + micro). Read after the map to restore state.
5. `DECISIONS.md` — append-only decision log. Read the tail; never re-litigate a
   settled decision without new evidence.
6. `REPORT_CARD.md` — the ONE card: the plan section (node verdicts) first, then the
   B scorecard, the run report, and every open row (see §4).
7. `AUDIT.md` — the principle-alignment audit written by BACKWARD.
8. `docs/report_cards/experiments/` — the receipts every number must trace to.
9. `docs/SPINE.md` Appendix A — the Layer-1 `U` contract.
10. §11 below — the A/B/C pipeline contract (layers, schema, plugin surface, gates).

If a file contradicts another, the order is: **AGENTS.md doctrine → SPINE laws →
frozen gate definitions → receipts → PLAN.md's tree → STATUS/DECISIONS narrative**. Narrative
never overrides a receipt, and the map never overrides a gate.

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

- **Conflict order:** Robust → Adaptive → Universal → Lawful → Efficient → Honest.
  Lawful and Robust outrank convenience.
- **The one rule:** never encode an answer the system could learn or derive.
- **Generalized application (DEC-592):** a mechanism added for one block/statistic must be
  applied to every block/statistic it can serve, through the one composition; no
  per-block or per-path special case. A block in one path and not another is a bug.
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

## 3. The loop: one MACRO statement, one MICRO experiment (binding)

The plan is the map; the card is the instrument. Every run moves **one leaf** and must say
**which node it is trying to move**. If you cannot say the macro sentence, do not run:

> *To make `G` true, `<node>` must be true because `<why>`; this run moves `<leaf>` from
> `<current>` toward `<target>` by changing `<one axis>`.*

### 3.0 MACRO REFLECTION — "why isn't the project successful yet?" (binding, every run)

Before choosing a node, answer this from the card and the receipts — never from memory or
from the last run's narrative. Write the block into the DEC **before** `NODE/ROUTE/MOVE`:

```text
WHY-NOT:  <the ONE largest reason the goal is false today, one sentence>
EVIDENCE: <the rows/receipts that prove it — numbers, not narrative>
GAP:      <what must become true that is not true: a capability, not a missing receipt>
LAYER:    <capability | objective | instrument | identity> — where the gap actually lives
THIS RUN: <which gap this run attacks, and why it is the cheapest next probe>
```

Rules:

1. **The block precedes node selection.** A run that picks the node first and writes the
   reflection afterwards has not reflected; it has rationalised.
2. **Name the layer.** The four layers are the recurring failure modes: *capability* (the
   state cannot carry the statistic), *objective* (nothing asks it to carry it), *instrument*
   (the measurement cannot see it), *identity* (the claim/definition is wrong). Past cycles
   failed by fixing one layer while a different one was binding — the `next_event` ladder
   was a capability search judged by an instrument whose headroom was below its own noise.
3. **If `THIS RUN` does not attack `GAP`, say so.** The run is then maintenance, and the DEC
   must record that word; maintenance is allowed, but never counted as progress.
4. **Re-answer at the outcome.** Append `WHY-NOT-CHANGED: yes|no — <why>` beside `MOVEMENT`.
   A run that flips a row while the WHY-NOT sentence is unchanged is a row flip, not a step
   toward the goal.
5. **A gap may be closed only by a measured improvement on its own layer** — never by a green
   row in another layer, and never by a lab receipt (DEC-576).

### 3.1 MACRO — read the map before choosing anything

Open, in this order: `PLAN.md` §2 (the tree), §5 (the route), §6 (frontier + what a green
proves); then `REPORT_CARD.md`'s assessment (the plan root) and plan table (node verdicts).
Write three lines into the DEC draft **before** choosing the change:

```text
NODE:  <P#/node> — why it is the bottleneck (which leaf/status keeps it open)
ROUTE: <R#> — the step this run is on, from PLAN.md §5
MOVE:  <leaf id> <current verdict/reading> -> <target>, because <one axis>
```

Selection rules, in order:

1. **The route wins.** The node is the open one the route says is next — not the cheapest
   row that happens to be failing.
2. **The measured blocker wins inside a node.** If one leaf is the measured reason the node
   is open (a red quality row, a missing producer the node names), that leaf is the run.
3. **A reporting green cannot answer a quality question.** Every leaf carries
   `Green means: quality | reporting | unmeasured` (derived in `scripts/b_spec.py`,
   displayed on the card). `attribution recorded` / `complete` / `working=True` prove the
   measurement happened, not that the embedding is good. If the node's gap is quality (e.g.
   `eff_rank`, persona, `g`), the run targets the **DEC-frozen bar and its producer**, not a
   reporting row that is already green.
4. **Unmeasured beats re-measured.** A missing producer is a closer gap than a green row
   worth re-running.
5. **One node at a time.** If the run touches two nodes, split it.

### 3.2 MICRO — the experiment (unchanged rigor)

One axis; pre-register before the run; frozen acceptance; launch receipts verified before
the result is read; one command emits the receipt. See §5 for the integrity rules.

### 3.3 READ AT BOTH SCALES — the step that has been missing

After the run, regenerate the card (`python3 scripts/report_card.py`) and read:

- **MICRO:** the leaf's verdict and numbers against its frozen gate.
- **MACRO:** the named node's status before/after, and every other node that moved.
  Record the sentence explicitly in the DEC outcome and STATUS:
  `MOVEMENT: <node> <before> -> <after>; root <before> -> <after>.`
- **No movement = negative result.** If the named node did not move, the run is a diagnosis,
  not progress — even if a local number improved. Record it as such and take the
  pre-registered branch (DEC-577's bilinear run is the worked example: eff_rank 1.153 ->
  1.473, node unchanged, branch (iii), candidate-1-negative).
- **Unexpected movement = coupling.** If another node moved, record it (principle 8: axes
  interact; a fix on one may cost another).
- **A failed run still updates the map** when it falsifies the premise: the next route step
  or the node's blocker changes, and that is a plan-relevant fact.

### 3.4 UPDATE THE MAP — only when the map is wrong

- **Frontier advances** -> update `plan_spec.CURRENT_ROUTE` (and `ROUTE` if the order
  changed), regenerate `PLAN.md` (`python3 scripts/plan.py`), then the card. Commit both.
- **The tree itself is wrong** (a missing necessary condition, a leaf under the wrong node,
  a gate kind misclassified) -> that is a **plan change**: DEC first; operator approval if it
  adds/removes a gate, threshold, or leaf. Never edit `plan_spec.py` to make the frontier look
  nearer.
- **`plan.py --check` must pass**; a stale `PLAN.md` is a card-integrity failure.

### 3.5 The two failure modes this section exists to prevent

- **MICRO drift** — a run that flips a row, adds tests, or regenerates receipts without
  naming the node it serves. It looks like work; the root does not move.
- **MACRO drift** — editing the tree, the route pointer, or a leaf's `Green means` so the
  frontier looks closer: a reporting gate promoted to quality evidence, `R` moved before the
  evidence, `E1` marked closed without its port receipt. Forbidden; the tree is tested against
  the frozen rows precisely so this is visible.
- **WHY-NOT drift** — re-running the same axis without re-answering §3.0, or recording a row
  flip while the WHY-NOT sentence is unchanged. The reflection is the point; the row is the
  instrument.

### 3.6 Worked example (DEC-577 — a macro-honest run that ended in "no")

```text
NODE:  P2/E1 — the trunk blob keeps P2 open; B2's green is a sufficiency claim, not capacity.
ROUTE: R1 — dev rig, next_event N=500 (minutes), before any real-data run.
MOVE:  E1's toy reading, because one axis: bilinear_recurrence ON.
```

MICRO: one axis, frozen verdict (g <= 0.5 AND beats the 1-gram), controls as designed.
READ: `MOVEMENT: E1 FRONTIER -> FRONTIER; root FAILED -> FAILED.` g 1.147 -> 1.154,
eff_rank 1.153 -> 1.473. The local geometry moved; the node did not. Verdict: branch (iii),
candidate-1-negative — a negative result, not progress. Next run: DEC-578 (candidate 2).

---

## 4. The single card and the completion contract

**Operator ruling (DEC-497, extended): `REPORT_CARD.md` is the one and only card.** There is
no second card file: the old judge card was retired and its file deleted, and the separate
`docs/AUDIT_CARD.md` research audit was removed — the single renderer
`scripts/report_card.py` now carries the 8 leg-columns, the 6-stage ladder, certification
hygiene, freshness, the anti-false-PASS rules, S6, **and** every open row. If a row is open it
appears on the card; nothing may be parked in a side document.

### Required `REPORT_CARD.md` shape (generated, never hand-edited)

```text
# REPORT_CARD — B foundational, then PATH B > PATH A
  header: generated_at · commit · branch · clean/dirty · encoder behavior hash
## Assessment
  plan root first (SOLVED / FAILED / OPEN + green, FAIL, DEC-frozen, v1 counts, reds by problem)
  then B foundational · B+C consequence · Project v1
## The plan — the tree this card measures (PLAN.md)
  24 nodes with DERIVED verdicts (worst-child), the frontier readings, and the
  reporting-gate caveat; FRONTIER = DEC-frozen leaf, NOT ASSESSED = v1 leaf
## B — foundational scorecard
  B1..B8; every row carries Green means (quality | reporting | unmeasured) + verdict + evidence
## The judge — four legs × two numbers
  8 rows: supervised/unsupervised/rl/simulator × brain alone / brain + ledger
## Measured values · The stage ladder (S1..S6) · Certification hygiene (H1..H5)
## What is blocking
  the plan's open leaves: B rows tagged [P#·B#], judge blockers tagged [P4·JUDGE-<leg>]
## Backward requirements
  the structured findings from `AUDIT.md`, each tagged with its plan node
```

The renderer reads `scripts/plan_spec.py` (the tree), the frozen row specs
(`scripts/b_spec.py`, `scripts/judge_spec.py`) and the receipts under
`docs/report_cards/experiments/`, plus `AUDIT.md`; it never invents a number. A run that edits
the card by hand is not evidence. A card without the plan section is stale: regenerate it.

`PLAN.md` is the readable render of the same tree (structure + route only, no measurements);
`python3 scripts/plan.py --check` proves it is current with the spec.

### Backward requirements are card requirements

`PROMPT_BACKWARD.md` writes **one file, `AUDIT.md`**: the narrative, the principle-alignment
scorecard, and the `BACKWARD_REQUIREMENTS_JSON` block at the end. BACKWARD writes no other file.
Each requirement has an `id`, `severity`
(`BREAKING`/`MAJOR`/`MINOR`/`INFO`), `principle`, `status` (`open`/`closed`/`accepted`),
`finding`, `forward_subgoal`, `gate_touched`, `evidence`, `audit_ref`, and `date`. The
renderer reads that block and displays the requirements in `REPORT_CARD.md`; an open
`BREAKING` or `MAJOR` requirement blocks the completion condition. `MAJOR` may be closed by a forwarding protocol or
explicitly `accepted` by the operator; `BREAKING` may never be accepted, only closed by
evidence.

### "All green" — the three claims, never conflated

The plan makes the old single word precise:

| Claim | Meaning | Where it is computed |
|---|---|---|
| **T0** | every measured B leaf PASSes its frozen gate on a current receipt | the card's B scorecard / plan table |
| **ENCODER DONE v1** | T0 + the four judge legs' gate-of-record + stages + hygiene (the list below) | the card's generated assessment |
| **SOLVED** | ENCODER DONE + `E1` closed by its pre-registered port receipt + every v1 leaf PASS + no open BREAKING/MAJOR backward requirement | the card's plan root — it prints `SOLVED` only then |

A green T0 is not ENCODER DONE; ENCODER DONE is not SOLVED. Never use a stronger word than
the generated artifact says.

**ENCODER DONE v1** — all of the following, simultaneously:

1. `REPORT_CARD.md`: **0 FAIL, 0 UNKNOWN**; every one of the 8 leg-columns is PASS or
   **EXCLUDED-and-reported** under a frozen, operator-approved exclusion rule; all
   **6/6 stages** PASS.
2. Hygiene 5/5 (environment, behavior hash, N, controls, finite/fair comparator);
   S6 6/6 (one command, deterministic render, clean recorded commit, artifact hashes,
   every row reports N).
3. Freshness: the receipts' `behavior_hash` equals the current tree; no stale block;
   no unadopted receipt on the card.
4. **No open BREAKING/MAJOR backward or principle requirement** in `AUDIT.md`'s
   `BACKWARD_REQUIREMENTS_JSON` block; every `MAJOR` is closed or operator-accepted. This
   includes every open finding in the same file.
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
- **Audit what you RUN.** Any executing code is in scope, not just what you wrote or changed;
  a measurement from un-audited code is not evidence. `--all --strict` is the gate.

---

## 6. Documentation and artifact upkeep (mandatory)

| Artifact | What to keep current |
|---|---|
| `PLAN.md` | the MACRO map (tree + route), generated from `scripts/plan_spec.py`; `python3 scripts/plan.py --check` must pass. Update `CURRENT_ROUTE`/`ROUTE` when the frontier moves — never to make progress look nearer. |
| `STATUS.md` | macro (goal, stage, progress, blockers, **which plan node is the frontier**) + micro (current task, last completed, next 3, environment). Update **before yielding**. |
| `DECISIONS.md` | append-only. Pre-registration before every run (**with `NODE` / `ROUTE` / `MOVE`**); measured outcome after (**with `MOVEMENT`**); context / alternatives / decision+principle / trade-offs. Never edit old entries. |
| `REPORT_CARD.md` | the run report + plan section + gate-of-record + **every open row** — it is the only card; no row may be hidden. Regenerated each run; no hand-written numbers. |
| `AUDIT.md` | the single BACKWARD file: principle audit + macro verdict + findings + `BACKWARD_REQUIREMENTS_JSON` block (each requirement tagged `node`) rendered into `REPORT_CARD.md` (§4). |
| `docs/report_cards/experiments/` | receipts; `protocols/<name>/` archives for non-adopted runs. |

**Commit discipline:** every coherent verified chunk is committed; code + docs only;
data, checkpoints, logs, and virtualenvs are never committed. Commits stay **local**
until the operator explicitly asks to push. The card is regenerated at a clean recorded
commit; a receipt that no repo command regenerates is not evidence.

### Commands (read the current paths from `STATUS.md`; never guess an artifact)

- One-command chain + card:
  `python3 scripts/measure_certification_law.py --products <recorded artifact> --demo <demo> --train-log <artifact>/train.log`
- Plan render / freshness check (MACRO): `python3 scripts/plan.py` ·
  `python3 scripts/plan.py --check`
- Report card render (the single card, plan section included): `python3 scripts/report_card.py`
- Principle audit (ALL code we run, not just changed files):
  `python3 scripts/principle_audit.py --all --strict` · summary: `--all --summary`

The chain command is the only S6-certifying path. A hand-rendered card is not evidence. The
order is always **plan -> card**: if a run changed the tree or the route, regenerate `PLAN.md`
first, then the card, and commit both.

**Evidence scripts:** any script committed as evidence must pass
`python3 scripts/principle_audit.py --strict` with **0 findings**, be vectorized (no
per-row Python loops in hot paths), and emit a runtime receipt.

---

## 7. Resources and operating constraints

- **Estimate before every run.** State the cost from MEASURED numbers — `s_per_step` in the last
  registry x expected steps + producer timings — and say why the smallest rung that can answer the
  question is not sufficient. A run whose estimate was not stated is a run that will overrun its
  timeout. Calibration (v2, seq_len 996, batch 16, counter on): 1.72 s/step; a 5,000-customer
  convergence is ~2,300 steps (~66 min); the full chain is ~8-12 min.
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

**MACRO — did the map move?**

0. The §3.0 block was written **before** the node was chosen, names its layer, and is
   re-answered in the outcome as `WHY-NOT-CHANGED: yes|no`.
1. The DEC names `NODE`, `ROUTE`, `MOVE` **before** the run and `MOVEMENT` after it.
2. The named node's status was read before and after; if it did not move, the run is recorded
   as a negative result/diagnosis, not progress.
3. Any other node that moved unexpectedly is recorded as a coupling finding.
4. If the frontier advanced, `plan_spec.CURRENT_ROUTE` (and `ROUTE` if needed) is updated and
   `PLAN.md` regenerated; `python3 scripts/plan.py --check` passes. If the tree changed, a DEC
   records why (and operator approval if a gate/leaf changed).
5. No reporting-gated leaf is presented as quality evidence anywhere in the DEC/STATUS/prose.

**MICRO — is the evidence valid?**

6. The one command ran; every producer exited 0; the receipt exists.
7. Frozen acceptance was graded **before** the verdict; the named axis is stated.
8. Adopted → keep; not adopted → revert + archive receipt + restore live receipts.
9. `STATUS.md` updated; `DECISIONS.md` outcome appended; open `AUDIT.md` findings are
   either closed with their receipt or explicitly carried as the next subgoal.
10. `REPORT_CARD.md` is the generated render; the plan table and the gate-of-record agree
    with the renderer's own bytes; no contradiction.
11. Clean-tree commit recorded; hashes/N/controls present; no `/tmp`-only evidence.
12. `python3 scripts/principle_audit.py --all --strict` clean: **0 new blocking findings on the
    whole surface**, not merely on the files touched. The baseline may only shrink. ruff/tests per
    the repo's gates (or the known-failure baseline disclosed).
13. Print the rendered `REPORT_CARD.md` (never fenced) as the run's source of truth.
14. If ENCODER DONE v1 (the §4 list) → stop and run `PROMPT_BACKWARD.md` (FULL). Else → the
    next FORWARD run, starting again at §3.1 (MACRO), not at the row that looks easiest.

---

## 9. Stop-and-ask triggers (operator only)

Stop **before** running if a proposed change:

- changes a gate, threshold, criterion, frontier definition, or exclusion rule;
- changes what "shape" contains, what "full-E" means, or what the comparator is;
- **changes the plan tree itself** — adds/removes/re-parents a node, adds/removes a leaf,
  changes a leaf's `Green means`, or moves a DEC-frozen node to closed without its receipt
  (operator-only when it touches a gate; a documented DEC otherwise);
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
- **Micro drift:** a run with no `NODE` line, or one that counts tests/receipts/row flips as
  progress while the named node and the root do not move.
- **Macro drift:** editing `plan_spec`/`ROUTE`/`CURRENT_ROUTE`, reclassifying a leaf's
  `Green means`, or closing `E1`/a v1 leaf to make the frontier or the root look better.
- **Promoting a reporting green to quality evidence** (e.g. "B3.2 is PASS, so the shape is
  good" when its gate is `attribution recorded`).
- Hope-driven reruns of a falsified ladder rung ("512-d next") without a new mechanism.
- Per-row Python loops in hot paths; batch-coupled serving statistics; silent clamps.
- Running a second LLM container, exceeding the caps, or training while the co-tenant
  is unbounded.

---

## 11. The A/B/C pipeline contract

*(merged from the retired `docs/ABC_CONTRACT.md` — this is the only copy)*

One canonical event stream in, one frozen foundation model out, plugins on top. Everything
flows one way:

```text
Layer A: rabbit_hole (canonical events)
    -> Layer B: looking_glass (one frozen CFM per stream, E(customer,t))
        -> Layer C: plugins / gates / white_queen (read E at train AND serve)
```

Authoritative code: `rabbit_hole/schema.py`, `looking_glass/customer_foundation_model.py`,
`looking_glass/cfm_training.py`, `looking_glass/cfm_state.py`, `looking_glass/layer_b.py`,
`looking_glass/plugin_contract.py`, and the gates named below.

### Layer A — canonical event stream

`rabbit_hole` emits one append-only stream. Onboarding writes, and the CFM reader accepts, the
nine-column `looking_glass.onboard.CANONICAL_COLUMNS`:

```text
customer_key, event_ts, event_type, brand, entity_type, entity_id,
source_table, value, event_attributes
```

`rabbit_hole/schema.py` draws a stricter, deliberately minimal guarantee: every stream has
`customer_key, event_ts, event_type, source_table, event_attributes` (`CANONICAL_FIELDS`).
`brand`, `entity_type`, `entity_id`, `value` are *not* assumed universal there;
`canonicalize_row()` folds any non-canonical column into the `event_attributes` JSON map and
`canonical_view_sql()` exposes a wide table as the five-field canonical view
(`customer_events_canonical`). `cfm_data._read_stream` accepts both forms — top-level content
columns when present, otherwise extracted from `event_attributes` — and discovers every JSON
attribute key so arbitrary schemas are consumed without code edits.

- `customer_key` — customer identifier (string in canonical output).
- `event_ts` — event time; canonical output is UTC ISO text. `cfm_config.as_of_epoch()` is the
  one day-boundary rule: naive ISO dates mean UTC midnight.
- `event_type` — event name.
- `brand`, `entity_type`, `entity_id` — optional content dimensions; in the minimal
  `rabbit_hole.schema` contract these live inside `event_attributes`, and onboarding promotes
  them to top-level when detectable.
- `source_table` — stream/table origin; defaults to the source name during onboarding.
- `value` — monetary value (float; 0.0 when absent).
- `event_attributes` — JSON object holding every field not promoted above. Action keys
  (`freq`, `hour`, `channel`, `offer`, ...) are JSON keys on send-like events.

`rabbit_hole.schema.COLUMN_ALIASES` defines accepted aliases (`customer_id`, `timestamp`,
`event_name`, `amount`, ...). `canonicalize_row()` folds remaining columns into
`event_attributes`; no information is dropped.

### Layer B — one frozen CFM per stream

```text
state_c(t)   = CFM_version(events of customer c with timestamp <= t)
E(customer,t)= learned shape || learned compression of the full-history measure
```

Self-supervised (no labels), frozen before any plugin sees it. The full-history measure is
schema-derived from the whole stream (`layer_b.generic_measure`, DEC-197), not from the queried
customer subset; its compression is fit on training customers and frozen for serving. `E` is
produced by `cfm_training.forward_embedding()` and loaded at serve by
`cfm_state.load_frozen_encoder()`.

```bash
python3 -m looking_glass.customer_foundation_model train \
  --db <stream.parquet> --as-of <date> \
  [--set use_measure=True \
   --set effect_actions=('freq','hour','channel','offer') \
   --set action_fields=('freq','hour','channel','offer','discount_pct')]
```

Flags: `--customers`, `--sample-a`, `--anchors`, `--out-dir`, `--set`. `--as-of` cuts the raw
stream to events `<= as_of` before keys, split, anchors, and training
(`cfm_data._cut_as_of`) — point-in-time and leak-free.

**A/B split.** `cfm_data.assign_split()` hashes each customer into population A (default 70%,
`split_a_frac`) or B. The encoder trains only on sample A; A and B never overlap. `--as-of`
also rotates the split seed monthly (`cfm_config.monthly_split_seed`), so membership and anchor
dates refresh on each monthly rebuild. Everything downstream — plugin fitting, anchor
embeddings, gates, the card — uses held-out population B.

**Anchors.** For sample-B customers, `build_sequences(..., with_anchors=True)` draws `n_anchors`
(default 6) uniform-random anchor days over the customer's history. At each anchor the model
encodes history up to that day, then fades the state to the anchor boundary (the same rule used
at serve). These point-in-time embeddings are written to `anchor_embeddings` by
`cfm_state.build_products()`; plugins train on them.

**`E` composition (DEC-198).** At serve, `load_frozen_encoder()` sets
`model.accumulator_on = False`. The consumed embedding is

```text
E = shape (donor readout; 256 dims in the certified runs) || learned measure (82 dims)   ~338
```

The **accumulator is the hand-RFM baseline** (token-drive sum/mean/last at multiple timescales),
not part of the production pipeline; it is switched on only for the card's "hand features"
comparison row. `measure_dim=0` (default) keeps the measure **full/lossless** (DEC-195); a
positive value is explicit compression. `scripts/reference_pipeline.py` requests
`measure_dim=128`; with 82 available measure columns `LayerBEmbedder.fit()` caps components at
82, so the served `E` remains ~338. `use_measure=True` activates the learned measure;
`effect_actions` activates the action-conditioned response objective; `action_fields` excludes
logged action columns from the generic measure so `E` does not carry the treatment itself.

### Layer C — the plugin contract

A plugin reads the frozen `E` for **both** training and inference: it fits on held-out
B-customer anchors, is then frozen, and serving never retrains it.

`looking_glass/plugin_contract.py` defines one surface, pickle-persisted, dependency-light
(numpy + scikit-learn):

| member | meaning |
|---|---|
| `Plugin.fit(E, targets)` | learn from frozen `E` and family-specific targets |
| `Plugin.infer(E)` | apply the fitted artifact (no retraining) |
| `Plugin.save(path)` / `Plugin.load(path)` | pickle persistence |
| `Plugin.freeze()` | make immutable; later `fit` raises |
| `fit_and_freeze(plugin, E, targets, out_dir)` | fit, freeze, save, return the frozen plugin |

Families:

- `SupervisedPlugin(estimator, target=...)` — any scikit-learn regressor/classifier on `E`;
  `infer` returns predictions, `proba`/`predict_proba` where available.
- `UnsupervisedPlugin(n_clusters=..., scaler=..., kmeans=...)` — KMeans on `E` (optional
  scaler); `infer` returns cluster labels, `transform` returns distances.
- `DecisionPlugin(actions=..., estimator=...)` — value model `V(E,a)`; `fit` takes
  `targets=(A, y)` with numeric actions, `values(E)` returns per-action values, `infer` returns
  `argmax_a V(E,a)`.
- `PluginRegistry` — name-keyed collection; `save_all(dir)`, `load_all(dir)`, `from_dir(dir)`.

A plugin is trained against the frozen `E` and is never given a path back into the encoder.

### Certification gates

**`U` — universality (`looking_glass/universality.py`).**

```text
U        = 1 - max_{informative tau} normalized oracle-gap_tau
gap_tau  = score([E, exhaustive full-history measure]) - score(E)
normalized = gap_tau / range_tau
```

The oracle is the schema-derived exhaustive full-history measure (not hand-RFM), cross-checked
by an independent sequence model. Evaluation is on held-out population B. `SOLVED` means `E` is
within the oracle's own noise on every informative member.

```bash
python3 -m looking_glass.universality --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> [--customers 25000] [--horizons 30,90] [--require-solved]
```

`--synthetic` runs the known-sufficient-statistic machinery check.

**Use-case gate (`looking_glass/use_case_gate.py`).** `discover_action_keys(cfg)` finds
randomized discrete action keys (JSON keys on send events, preferring a declared `action_keys`
receipt). `run_use_case(model, cfg, action_key)` reports the empirical dose-response and a
per-customer policy from `E`: a joint value model `V(E,a)` is cross-fitted by customer, the
candidate is chosen on one half of customers, and its value is estimated on the other half by
**g-computation** (no importance reweighting, no winner's curse). Verdict is `DEPLOY` iff the
paired bootstrap 95% CI lower bound on the policy gain is `> 0`, else `HOLD`.

```bash
python3 -m looking_glass.use_case_gate --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --actions freq,hour
```

**Null controls.** The card re-runs the same use-case and decision gates on a signal-free
stream (`snr=0`) trained into a separate null checkpoint. Every real-stream `DEPLOY` must be
paired with a null `HOLD`; a null that also deploys invalidates the claim.

The card's default B+C arm uses `looking_glass.decision_certify.certify()` (doubly-robust value
with the logging marginal propensity; `N/A` when there is no per-customer arm structure). The
real tribunal below is the optional, stronger witness.

**Real white_queen (`looking_glass/white_queen_bridge.py`).** Puts
`white_queen.tribunal.ope.api.evaluate` directly on the `E`-policy:

- `certify(model, cfg, horizon_days=90)` — candidate policy reads `E`; reward is the logged
  future monetary outcome (`sum(log1p|value|)` on purchases); no logged propensity is passed
  (white_queen estimates it).
- `certify_period(model, cfg, action_key="channel")` — per-period randomized action (channel,
  offer, freq, ...); context is the customer's frozen `E`.
- `oracle(...)` — a known-truth policy (e.g. `arm_pref`/`channel_pref`) as a positive control.

DEPLOY/HOLD is white_queen's conservative verdict: never ship worse than logging.

**Supervised gate (`looking_glass/supervised_gate.py`).** `run_supervised()` fits heads on
held-out `E` for churn, reactivation, next purchase, return, LTV, next value, next
channel/category/item. Per target it reports skill, a bootstrap 95% CI, and `PASS`/`FAIL`
(`PASS` iff the CI lower bound is above the model-free baseline: 0 for regression,
majority/0.5 for classification). With `null=(db, tag, ckpt_dir)`, the same targets run on the
null encoder and `null_ok` is true only if every null target `FAIL`s.

**Unsupervised gate (`looking_glass/unsupervised_gate.py`).** `run_unsupervised()` certifies
segmentation and lookalike geometry on held-out `E`: silhouette `> 0`, value-differentiation
ANOVA `p < 0.05`, and lookalike enrichment `> 1.0` are all required for `PASS`. With a null
encoder, `null_ok` is true only if the null verdict is `FAIL`.

**Reference card (`looking_glass/foundation_report.py`).** `build_card()` emits the one-pager:
U verdict, per-target rows, architecture, use cases, B+C decision certification, optional
white_queen/supervised/unsupervised sections, and null controls. This is the *old* reference
pipeline's card — the project's steering card is `REPORT_CARD.md` (§4).

```bash
python3 -m looking_glass.foundation_report --db <stream> --ckpt-dir <dir> --tag <tag> \
  --as-of <date> --out card.md [--fast] [--white-queen --supervised --unsupervised]
```

`--fast` skips the bare/accumulator comparison U runs (the dominant cost); `--full` is **not** a
`foundation_report` flag — it is the `scripts/reference_pipeline.py` flag that adds white_queen,
supervised, and unsupervised to the card. Without it the reference card runs the fast core
(B + use-cases + null controls).

### One command

```bash
python3 scripts/reference_pipeline.py --as-of <date> [--full] [--dry-run]
```

Stages in `scripts/reference_pipeline.py`:

1. generate the REFERENCE stream (real signal);
2. generate the REFERENCE null control (`--snr 0`);
3. train the foundation model on the real stream;
4. train the foundation model on the null stream;
5. emit `card_reference.md` from the B & B+C gates, resolving the data-derived tags from the
   newest registry in each checkpoint directory.

`--full` adds white_queen/supervised/unsupervised and is slow; `--dry-run` prints the commands
without running them. Outputs live under `rabbit_hole/data/reference_pipeline` by default.

---

*The forward run ends when the card is ENCODER DONE v1. The project is not done until the plan
root reads SOLVED — `E1` closed by its port receipt and every v1 leaf green — and
`PROMPT_BACKWARD.md` returns no BREAKING findings against that card. Start every run from the
map (`PLAN.md` §3.1), not from the row that happens to be red.*
