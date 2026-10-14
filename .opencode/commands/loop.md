---
description: Run the wonderland autonomous loop CONTINUOUSLY until the goal is SOLVED
---
PERSISTENT AUTONOMOUS LOOP — NOT a one-shot task. Work in
/home/austin-whaley/Documents/wonderland; pass workdir=/home/austin-whaley/Documents/wonderland
on every shell call.

## The one behavioral rule (read first)
You are a LOOP, not an assistant answering once. After each iteration you MUST immediately start
the next one by making another tool call. **Emitting a final answer, a "status report", a
summary, or a question after a single iteration is a FAILURE of this command.** Do not yield
between iterations. Between iterations output AT MOST ONE LINE (e.g. `iter 7: <one clause>`),
then continue calling tools. A run ends only for the three reasons below.

## Stop ONLY when
1. `bash loop/status.sh` prints **GOAL_MET** → then write the final receipts and stop.
2. A **genuine goal-level or principle-level conflict**, or a **no-legitimate-path dead end** →
   write it to `STATUS.md` under "Open decisions" with a one-line question, then stop.
3. **Hard cap: 25 iterations OR ~90 minutes OR context exhaustion** → stop and say exactly where
   you got to. (Hitting the cap is "gave up", not "done".)

## GOAL / STOP predicate
North star: `AGENTS.md` "The project", `STATUS.md` "Canonical decomposition", `loop/UNIVERSALITY.md`,
`loop/PLAYBOOK.md` (hypothesis queue + guardrails + axis contract). **SOLVED (DEC-126):**
`E` matches the exhaustive full-history oracle on EVERY member, on EVERY stream (E within the
oracle's own noise) AND all B/B+C cards GREEN. **Layer 0 (NOW):** input an event stream → encode →
a FOUNDATION MODEL emitting UNIVERSAL FOUNDATIONAL EMBEDDINGS (self-supervised, frozen/reusable,
schema- and domain-agnostic, transferable). **Layer 1:** certification via `SOLVED`.
Known use cases may VETO (gates), never SHAPE the encoder.

## One iteration (repeat back-to-back)
Maintain a todo list with the iteration counter and the current hypothesis.
1. Restore state (≤1 min): STATUS.md top + tail of DECISIONS.md + `git log --oneline -8` + `git status`.
2. Name the single highest-leverage blocker (one line).
3. Cheapest matched experiment FIRST — Rung 0 (measure on an existing checkpoint; no training)
   → Rung 1 (tiny train `--customers 2000 --sample-a 1500 --set max_steps=250`) → Rung 2 (full).
   ALWAYS a matched control (identical args; **one variable differs**). Long jobs: run in background
   and poll; never sit blocking.
4. Keep the fast gate green: `ruff check . && ruff format --check . && pytest -m "not slow"`.
5. Keep what wins; DELETE dead ends (revert code, record why).
6. Record: append a DEC entry (context / what changed / what was MEASURED / verdict / principles
   served) and update STATUS.md. Decide it yourself — do not defer to the operator.
7. Commit + **push** (code + docs only; NEVER data `*.pt/*.parquet/*.duckdb`); message = measured
   facts + receipts + verdict.
8. **Immediately begin iteration N+1** (back to step 1). Do not stop to restate progress.

## Guardrails (honest means only)
- PRINCIPLES (AGENTS.md charter; cite the one(s) each DEC serves; conflict order robust > universal
  > adaptive > lawful > efficient > honest): universal · adaptive&dynamic · lawful · efficient ·
  robust · honest · accessible · diagnosable. DIAGNOSABLE = attribute each shortfall to an axis,
  propose the resolving change, record the cross-axis trade.
- No magic numbers (derive/learn); no domain literals; per-sample at serve; contractive recurrence
  (ρ<1); divergence/non-finite = HARD FAILURE (never masked/rolled back); no loss hacks.
- NEVER weaken/re-interpret a gate, baseline, threshold, or the oracle to pass.
- NEVER select weights on the metric you optimize (held-out / ground truth only).
- Layer B (encoder) is self-supervised only — no downstream labels in its training.
- MAKE IT LEARN, don't teach it.
- Do NOT repeat recorded negatives (grep DECISIONS.md before trying a measure variant).
