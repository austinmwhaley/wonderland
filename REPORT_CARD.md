# REPORT_CARD — PATH B > PATH A

*Generated `2026-11-01T13:40:26+00:00` · commit `dcfbe93` (`foundation-rebuild` · clean) · encoder behavior `94857be6815a96f3`*

**Assessment:** NOT DONE. Legs green: 0/8 · stages green: 0/6 · backward open: 0. This is a statement about receipts, not about effort; nothing here is a claim of universality (DEC-410: that word appears only when the coverage rows are green).

The verdict is a pure function of the receipts under `docs/report_cards/experiments/`.
Ledger points never count as progress: `brain alone` is the judge, `brain + ledger` is delivery.

## The judge — four legs × two numbers

Progress (`brain alone`) is a strict win. Delivery (`brain + ledger`) is a **non-inferiority** claim vs PATH A (margin 0.01 R2); a delivery tie is not a brain win.

| Leg | Column | Verdict | Evidence |
|---|---|---|---|
| supervised | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| supervised | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| unsupervised | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| unsupervised | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| rl | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| rl | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| simulator | brain alone (PROGRESS) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| simulator | brain + ledger (DELIVERY, non-inferiority) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |

## The stage ladder — what is broken when a leg is red

| Stage | Working-state proof | Verdict | Evidence |
|---|---|---|---|
| **S1** Schema resolution (universal input) | renamed-schema + new-field pass, zero code edits | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| **S2** Vocabulary & numeric encoding | unseen-value path exercised; transform receipts; permutation invariance <= tol | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| **S3** Lawful dynamics (the engine) | merge/equivariance/mass-additivity exact on data; tau receipted from spectra; stable from step 0 | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| **S4** Buckets (the outlets) | per-bucket accessibility parity + per-bucket negative controls | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| **S5** The objective battery | every content pressure beats its naive comparator (or is excluded and reported) | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |
| **S6** Certification (the card) | one command regenerates the card at a recorded commit + hashes | ❔ UNKNOWN | stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3) |

## Certification hygiene

| # | Check | Verdict | Evidence |
|---|---|---|---|
| H1 | Environment receipted | ✅ PASS | env_receipt.json device matmul ok |
| H2 | Behavior hash recorded | ✅ PASS | behavior_hash 94857be6815a96f3 |
| H3 | N recorded | ✅ PASS | enforced per row (no PASS without N) |
| H4 | Controls present | ✅ PASS | enforced per row (no PASS without controls) |
| H5 | Comparator finite and fair | ✅ PASS | enforced per row (comparator must be finite and fair) |

## What is blocking (exact missing receipts)

- `judge_leg_supervised.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_leg_supervised.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_leg_unsupervised.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_leg_unsupervised.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_leg_rl.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_leg_simulator.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_stages.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_stages.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_stages.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_stages.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_stages.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`
- `judge_stages.json — stale: encoder behavior changed since receipt (1b238954570cdc76 != 94857be6815a96f3)`

## Backward requirements (from PROMPT_BACKWARD)

Principle-alignment findings live in `AUDIT.md`; the machine-readable gate is this table.

| ID | Severity | Principle | Status | Finding | Forward subgoal | AUDIT |
|---|---|---|---|---|---|---|
| — | — | — | — | no backward review recorded | run `PROMPT_BACKWARD.md` after the card is green | — |

---

*The inherited status-quo card is archived at `docs/archive/REPORT_CARD_INHERITED.md`; it measures the pipeline as it was and is disclosure only (DEC-497).*
