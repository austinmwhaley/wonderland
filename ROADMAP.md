# ROADMAP — wonderland (DRAFT for discussion)

Status: proposal — talk to me before treating phases as commitments.
Last updated: after the infrastructure/SQLite/refactor pass (suite: 226 green).

---

## Where we are (v1 definition of done)

| # | DoD item | State |
|---|---|---|
| 1 | rabbit_hole causally-identifiable stream | ✅ randomized arms + known effect + persistent holdout |
| 2 | looking_glass universal donor | ✅ battery PASS (unique 75%, beats raw 6/8) |
| 3 | each plugin passes its own gate | ✅ 6/6, 4/4, 5/5 |
| 4 | red_king recovers known effects **and improves white_queen with vs without** | ⚠️ half — population counterfactual validated (ordering 1.0, calib 0.985 under confounding); the **with-vs-without A/B has never run** |
| 5 | red_queen composes NBA across cadences under constraints | ✅ multi-cadence/multi-action, reject-not-clamp, certification-gated |
| 6 | caterpillar answers "why" from artifacts | 🟡 v1 (schema bug fixed this pass; NL Q&A outstanding) |
| 7 | artifacts versioned + receipts | ✅ vNrN identity, receipts throughout |

Also proven: incrementality (ATE +11.62, CI [11.19, 12.08], monotone uplift
quintiles, top-20% gain +18.95), robustness under realistic confounding for
white_queen / red_queen(validated) / donor / red_king(population), hardened OPE
(corroboration required). Engineering: CI + ruff + 226 tests + zero SQLite.

## Gaps

**Science / identification**
- Per-customer personalization is only identifiable for switchback customers
  (observed under all arms). Everything else falls back to population — by
  honest measurement, not by choice.
- Sparse incremental-GP rewards + weak arm effects cap HTE resolution
  (measured: rank acc 0.506 on identified vs 0.533 majority elsewhere).
- v1 DoD #4 (does red_king earn its keep in the decision loop?) is unmeasured.
- Management report lacks per-segment breakdown; segmentation's eta² k-search
  drifts to kmax on noisy labels (silhouette path is reliable).

**Scale / speed**
- Generator is still per-event Python — blocks the 50k+ ladder that the
  battery protocol says is "standard future pipeline".
- Donor training ~29 min at 30k; no receipts above 30k.
- Full test suite ≈ 14 min — fine for CI, slow as an inner loop.

**Engineering maturity**
- The new CI workflow has never actually run (3 local commits not yet pushed).
- No root lockfile (`uv sync` resolves fresh each time; eighth_square has the
  only `uv.lock`).
- No coverage measurement; no type checking (mypy unconfigured).
- looking_glass reference scripts (smoke_test family) read a dataset with no
  generator — they fail fast with a clear error, but the "full pipeline smoke"
  story is currently unwritten against the canonical stream.
- Root is import-by-cwd (`python -m` from repo root), not installable — chosen
  deliberately, worth re-affirming.

**Product**
- caterpillar is v1: no NL surface.
- Actions: email frequency + channel/discount certification exist; timing and
  promotions per AGENTS scope are young.
- Reference/smoke story for looking_glass as a standalone library is thin.

## Proposed roadmap

### Phase 0 — Land what exists (mostly done)
- [x] infra, tests, lint, splits, SQLite→DuckDB, run_full removal
- [ ] **push the 3 local commits; first real CI run green** ← blocker for everything else

### Phase 1 — Prove the differentiator (science first)
1. **Same-estimand A/B (v1 DoD #4).** white_queen action = randomized arm,
   reward = incremental GP; red_king as MB witness with vs without.
   Gate: HOLD→DEPLOY flips correctly on known-truth cases, never ships worse
   than logging. **Decision rule: if it doesn't move the verdict, delete
   red_king from the decision path (principle 13) and keep it as an analyst
   tool only.**
2. **Personalization for the identified.** Expand the switchback experiment
   (more customers/periods); wire `hte_model` outputs into red_queen:
   identified customers get per-customer arms, everyone else gets population +
   hierarchical shrinkage. Gate: identified-cohort targeting gain beats
   population targeting on held-out identified customers (CI, not point).
3. **Management incrementality report v2**: per-segment CIs + receipts on top
   of the existing ATE/quintile proof. Gate: every claim carries a CI; report
   regenerable from artifacts alone.

### Phase 2 — Scale (speed-first, doctrine #16)
1. **Vectorize the generator** — Arrow/Polars bulk construction, no per-item
   Python over rows/time. Gate: runtime receipt scales linearly; 50k+ stream
   generated in minutes, not hours.
2. **Ladder + battery rerun** at 50k: sample-A rungs, battery at each rung,
   donor speed receipts. Gate: donor PASS retained or improved at 50k.

### Phase 3 — Product surfaces
1. **caterpillar NL Q&A** (read-only over artifacts; refuses to speculate
   beyond receipts — fail-safe doctrine).
2. **Reference-data story**: either generate the looking_glass smoke dataset
   from the canonical rabbit_hole stream (one generator, doctrine #6) or retire
   the smoke family to documentation. Decision needed (see below).

### Phase 4 — Engineering maturity
1. Root `uv.lock` pinned; CI installs from it.
2. Test tiering: fast unit tier (<60s) for inner loop + full gate for CI;
   coverage floor on `white_queen/tribunal` (the money code).
3. Gradual mypy on `white_queen/tribunal/ope` only (highest-stakes logic).
4. Tag releases; receipts attached to each tagged artifact set.

## Open decisions (need your call)

1. **red_king in the decision loop**: fund the Phase-1 A/B, or pre-commit now
   to dropping it from red_queen if the A/B is null? (Everything else about
   red_king — population estimator, analyst tooling — survives either way.)
2. **looking_glass smoke family**: canonical-stream-backed generator, or retire?
3. **Push now?** Phase 0 is blocked on the first real CI run.
4. **Scope of Phase 2 generator work**: full vectorization in one pass, or
   profile first and fix the hot loop only?
5. **mypy**: yes at Phase 4, or skip entirely (ruff + tests may be enough)?
