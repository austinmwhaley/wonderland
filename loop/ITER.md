# ONE ITERATION of the autonomous loop.
#
# This is the message handed to `opencode run` on every pass. It is deliberately
# BOUNDED (one loop step) and STATELESS: all continuity lives in the repo
# (STATUS.md, DECISIONS.md, git), never in the agent session.

You are one iteration of an autonomous loop in /home/austin-whaley/Documents/wonderland.
Pass workdir=/home/austin-whaley/Documents/wonderland to every shell call.

GOAL (north star; `AGENTS.md` "Project goal", `STATUS.md` "Canonical decomposition",
`loop/UNIVERSALITY.md`): make `embedding(customer, t)` a MINIMAL, BOUNDED, **UNIVERSAL
SUFFICIENT STATISTIC of the stream** — sufficient for the full observable future so ANY present
or future consumer is covered by construction. Measured by `U = 1 − worst-case oracle-gap`
(broad, adversarial, held-out, self-supervised family; oracle = full history / Bayes floor;
fixed capacity). `U → 1` ⟺ done. The B/B+C cards green across all streams are a NECESSARY
condition, not the goal. Known use cases may VETO (gates), never SHAPE the encoder.
PRIORITY this phase: build the universality gate (`looking_glass/universality.py` per the spec),
then drive `U` up with the highest-leverage architectural lever (the shape×measure decomposition).

NEXT ACTION: run `bash loop/status.sh`. If it prints GOAL_MET, do nothing further and
exit. Otherwise:

1. RESTORE STATE (≤2 min): read STATUS.md (top) and the tail of DECISIONS.md; run
   `git log --oneline -8` and `git status --short`. Do not re-litigate recorded
   negatives.
2. PICK the single highest-leverage remaining blocker (the one that unblocks the most).
3. RUN the CHEAPEST falsifying test first (Rung 0: measure on an existing checkpoint;
   → Rung 1 tiny train; → Rung 2 full). ALWAYS a matched control (one variable).
4. Keep the fast gate green: `ruff check . && ruff format --check . && pytest -m "not slow"`.
5. If it works: keep it. If not: revert it (delete the dead end).
6. RECORD: append a DEC entry (context / what / measured / verdict) and update STATUS.md.
7. COMMIT (stage code+docs only; NEVER data/*.pt/*.parquet/*.duckdb), message =
   what was measured + receipts + verdict (fact-only).
8. STOP after this one step. Do not start a second experiment in the same run.

TOP-LEVEL PRINCIPLES (AGENTS.md charter — every decision MUST align; conflict order
robustness > universality > adaptability > dynamism > speed > generality): universality,
adaptability, dynamism, speed, robustness, generality. Cite the principle(s) a non-trivial
decision serves in its DEC entry.

CONSTRAINTS: no magic numbers (derive/learn); no domain literals; per-sample at serve;
contractive; no loss hacks; no silent non-finite masking; never weaken a gate to pass;
never select on the metric you optimize; self-supervised only for Layer B. If you hit a
GENUINE operator decision (spec/gate/stream change), write it to STATUS.md under
"Open decisions" and exit the run.
