---
description: Run the wonderland autonomous loop until the goal is GREEN
---
AUTONOMOUS LOOP MODE. Work in /home/austin-whaley/Documents/wonderland; pass
workdir=/home/austin-whaley/Documents/wonderland on every shell call.

GOAL (north star; AGENTS.md "The project", STATUS.md "Canonical decomposition",
loop/UNIVERSALITY.md, loop/PLAYBOOK.md — hypothesis queue + guardrails + axis contract).
**STOP = SOLVED (DEC-126):** the checking predicate `bash loop/status.sh` prints GOAL_MET iff
`E` matches the exhaustive full-history oracle on EVERY member (on every stream) AND all B/B+C
cards are GREEN. **Layer 0 (NOW):** input any event stream → encode → a FOUNDATION MODEL
emitting UNIVERSAL FOUNDATIONAL EMBEDDINGS (self-supervised, frozen/reusable, schema- and
domain-agnostic, transferable, shared geometry). **Layer 1:** certification (sufficiency via
`SOLVED`). Known use cases may VETO (gates), never SHAPE the encoder.

KEEP GOING. Do NOT stop, summarize, or ask between steps. Decide everything yourself inside the
guardrails below; when you hit a fork (spec/gate/stream), choose the option that best serves the
goal and the doctrine, record it in DECISIONS.md, and continue. Only stop when `bash loop/status.sh`
prints GOAL_MET (then report receipts) — or if you are truly blocked with NO legitimate path,
say so in one line and stop.

Each step:
1. Restore state: STATUS.md top + tail of DECISIONS.md + `git log --oneline -8` + `git status`.
2. Pick the single highest-leverage remaining blocker; name it in one line.
3. Cheapest matched experiment FIRST — Rung 0 (measure on an existing checkpoint; no training)
   → Rung 1 (tiny train `--customers 2000 --sample-a 1500 --set max_steps=250`) → Rung 2 (full).
   ALWAYS a matched control (identical args; one variable differs). Run jobs in the background and
   poll; never sit blocking.
4. Keep the fast gate green each step: `ruff check . && ruff format --check . && pytest -m "not slow"`.
5. Keep what wins; DELETE dead ends (revert code, record why).
6. Record: append a DEC entry (context / what changed / what was MEASURED / verdict) and update
   STATUS.md. No decision is "for the operator" — decide it and move on.
7. Commit (code + docs only; NEVER data `*.pt/*.parquet/*.duckdb`); message = what was measured +
   receipts + verdict (fact-only).
8. Immediately begin the next step. Loop until `bash loop/status.sh` prints GOAL_MET.

GUARDRAILS — achieving the goal by honest means (no manual hacks):
- TOP-LEVEL PRINCIPLES (AGENTS.md charter — broad strokes; align every decision, cite it in
  the DEC entry; conflict order robust > universal > adaptive > lawful > efficient > honest):
  universal · adaptive&dynamic · lawful (compositionality/equivariance/algebra) · efficient ·
  robust · honest · accessible · diagnosable. DIAGNOSABLE = attribute each shortfall to an axis,
  propose the resolving change, record the cross-axis trade.
- No magic numbers: every threshold/dim/window/rate is derived from the data, learned, or a
  documented fallback resolved at runtime and recorded in a receipt.
- No domain literals (no business event names / hand-picked feature partitions).
- Per-sample at serve: NO batch statistics at inference.
- Contractive recurrence (ρ<1); divergence / non-finite is a HARD FAILURE, never masked or
  rolled back to fake a green.
- No loss hacks (focal/class weights/clips to pass); no silent masking of non-finite terms.
- NEVER weaken or re-interpret a gate / baseline / threshold to make a card pass. A green must be
  earned against strong baselines, held-out, with CIs.
- NEVER select weights on the metric you optimize (use held-out / ground truth).
- NEVER encode an answer the system could learn or derive; MAKE IT LEARN, don't teach it.
- Layer B (the encoder) is self-supervised only — no downstream labels in its training.
- You may re-add soft, per-sample, scale-free objectives (SPEC §3.1 "SR proposes, FR disposes"),
  but the certification filter (per-sample, no batch-coupled capacity faking) still applies.
