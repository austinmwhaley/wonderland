# ROADMAP — wonderland

Updated after the scale/maturity/observational-first pass.
Suite: 239 tests (fast tier `pytest -m "not slow"` ≈ 53s, full ≈ 14min).

---

## Hard constraint (product)

**Production logs have no holdout and no A/B — nothing we do may require one.**
This is now enforced in code, not just documented:

| | On observational logs (production) | On lab/rabbit_hole data (has control) |
|---|---|---|
| OPE DEPLOY/HOLD (white_queen) | ✅ works — `estimate_propensity=True` (provenance receipt) + sensitivity analysis for hidden confounding | ✅ works (logged propensity) |
| Prediction heads (CLV, churn, segmentation) | ✅ pure prediction, no causal claim | ✅ |
| Certification-gated controllers | ✅ reads frozen artifacts | ✅ |
| Incrementality / uplift targeting / response model | ❌ **rejects**: `NotIdentifiableError` (clear message, never NaN, never fake lift) | ✅ identified via randomized control |
| IPW arm effects (engine / nba / evaluate_plan) | ❌ rejects without logged propensity | ✅ |
| nba WHO step | falls back to empty targeting **with a printed receipt** | ✅ uplift responders |

What this means in practice: on your logs the system earns DEPLOY/HOLD through
OPE certificates + sensitivity bounds, and it will *tell you* when you ask it
for lift numbers it cannot honestly produce. The holdout/switchback machinery
stays as a **lab tool** for validating estimators against known truth — never a
production dependency.

## Where we are

| DoD | State |
|---|---|
| 1 stream + identifiable design | ✅ (identification available **in lab data**; production = observational by constraint) |
| 2 universal donor | ✅ |
| 3 plugin gates | ✅ |
| 4 red_king improves white_queen with-vs-without | ✅ **resolved by decisive A/B**: 0/25 decision changes -> red_king REMOVED from decision path (analyst tool; locked by tests) |
| 5 red_queen multi-cadence NBA | ✅ + observational guards |
| 6 caterpillar "why" | 🟡 v1 (schema fixed; no NL surface) |
| 7 versioned artifacts + receipts | ✅ |

**Fixed this pass:** generator vectorized (5k: 4m06→1m11; 50k = 14m54,
83M events, acceptance 25/25), zero SQLite, smoke family deleted, `uv.lock`,
test tiers, observational-first identifiability guards (13 new tests).

## Remaining gaps

1. **Push + first real CI run** (blocking — everything below assumes green CI).
2. **50k generation tail** — Python is ~4% of wall now; the rest is DuckDB
   index maintenance (~450s) + 23.6GB Arrow tail (~170s). Optional: lighten
   DDL / post-load indexing to get 50k under 10min.
3. ~~Coverage floor~~ ✅ measured **85%** on `white_queen/tribunal`; CI enforces
   `--cov-fail-under=80` (5pt headroom).
4. **caterpillar NL Q&A** — the only untouched product surface.
5. **white_queen hardening** — 7/25 errors on the known-truth battery
   (see STATUS "DECISIVE A/B"); red_king proved it cannot help there.
5. Optional lab science: per-segment management report (lab data only);
   red_king with-vs-without A/B with a pre-committed ship-or-delete rule;
   donor 50k ladder re-run (generator can now produce it).

## Phase status

- **Phase 0 — land it**: ✅ pushed; CI green (lint + full suite + coverage 85%/80)
- **Phase 1 — science (reframed)**: ✅ observational-first implemented; ✅ red_king A/B run -> REMOVE verdict; remaining items are *optional lab validations*, not production blockers
- **Phase 2 — scale**: ✅ vectorized; DDL-tail optimization = optional follow-up
- **Phase 3 — product**: smoke family resolved by deletion; caterpillar NL Q&A remains
- **Phase 4 — maturity**: ✅ uv.lock, test tiers, coverage floor (85% measured / 80 enforced); mypy = **decided against** (ruff + 239 tests + receipts are the enforcement; revisit only if type-level bugs actually appear)

## Open decisions

1. ~~red_king DoD#4 A/B~~ **DONE — REMOVE**: ran the pre-committed battery
   (25 candidates, known truth): red_king changed 0 certified decisions ->
   analyst-tool only (`red_king/ab_witness.py` + receipt).
2. **DDL/index tail** — invest to get 50k under 10min, or accept 14m54s?
3. **white_queen hardening** — the battery found 7/25 errors of white_queen's
   own (3 missed deploys, 4 false deploys under low overlap). New top gap.
4. **caterpillar NL Q&A** — build it next, or leave v1?
