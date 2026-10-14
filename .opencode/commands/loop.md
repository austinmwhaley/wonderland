---
description: Run the wonderland autonomous loop until the goal is GREEN (or a real decision)
---
AUTONOMOUS LOOP MODE. Work in /home/austin-whaley/Documents/wonderland; pass
workdir=/home/austin-whaley/Documents/wonderland on every shell call.

GOAL (the north star, see STATUS.md "Primary goal"): `embedding(customer, timestamp)` is a
SUFFICIENT STATISTIC of the history, and every B card AND B+C card is GREEN across all three
streams. The checkable predicate is `bash loop/status.sh` — it prints GOAL_MET when green.
Also track the sufficiency gate (`python3 -m looking_glass.sufficiency_gate ...`).

CHAIN CONTINUOUSLY — do NOT stop, summarize, or ask between steps. Each step:
1. Restore state: STATUS.md top + tail of DECISIONS.md + `git log --oneline -8` + `git status`.
2. Pick the single highest-leverage remaining blocker; name it in one line.
3. Cheapest matched experiment FIRST — Rung 0 (measure on an existing checkpoint; no training)
   → Rung 1 (tiny train `--customers 2000 --sample-a 1500 --set max_steps=250`) → Rung 2 (full).
   ALWAYS a matched control (identical args; one variable differs). Run jobs in the background and
   poll; never sit blocking.
4. Keep the fast gate green each step: `ruff check . && ruff format --check . && pytest -m "not slow"`.
5. Keep what wins; DELETE dead ends (revert code, record why).
6. Record: append a DEC entry (context / what changed / what was MEASURED / verdict) and update
   STATUS.md. If a genuine operator decision arises (spec/gate/stream change), write it to
   STATUS.md under "Open decisions" and surface it in ONE line — then stop.
7. Commit (code + docs only; NEVER data `*.pt/*.parquet/*.duckdb`); message = what was measured +
   receipts + verdict (fact-only).
8. Immediately begin the next step. Loop until `bash loop/status.sh` prints GOAL_MET, then report
   the receipts and stop.

DOCTRINE (SPEC §3.1 "SR proposes, FR disposes" + AGENTS.md): no magic numbers (derive/learn);
no domain literals; per-sample at serve (no batch statistics); contractive ρ<1; no loss hacks;
no silent non-finite masking; divergence is a hard failure; never weaken a gate to pass; never
select on the metric you optimize; never encode an answer the system could learn; Self-supervised
only for Layer B.
