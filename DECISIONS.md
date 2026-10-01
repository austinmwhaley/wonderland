# DECISIONS — wonderland (append-only)

Architectural decision record. Read before non-trivial changes; append after
every non-trivial decision so choices are never re-litigated. Format:
Context & problem → Alternatives considered → Decision & justification
(principle) → Trade-offs accepted.

---

## DEC-001 — One production stack: the CFM path only

- **Date:** 2026-09-28 (commit `009edef`)
- **Context:** looking_glass carried two stacks — the production CFM path
  (used by every downstream consumer) and an orphaned `create_*` factory
  library (EntityCore + Mamba-2 SequenceEngine). Import-graph proof: zero
  external consumers, zero CFM-core edges into the island.
- **Alternatives:** (a) maintain both; (b) finish the factory stack as the
  public API; (c) delete the island.
- **Decision:** delete it (23 modules, demo, 60 tests, `lancedb`/`mamba-ssm`/
  `bitsandbytes` deps). Principle: *simple over complex* + doctrine
  "prefer deletion over accretion".
- **Trade-offs accepted:** lost a generic library surface nobody used; specs
  in `looking_glass/specs/` are now historical records.

## DEC-002 — v2.1.0 encoder semantics (correctness over continuity)

- **Date:** 2026-09-30 (commit `22a689b`)
- **Context:** vendored-fork audit found: half-life derived from a merged
  timeline pinned it at the 1h floor on every dataset; training-anchor
  readouts were unfaded while serving fades; governor returned a non-minimal
  state; `min_events=3` hid sparse customers from the state store.
- **Alternatives:** (a) ignore (numbers were "stable"); (b) patch silently;
  (c) fix and bump the behavior version.
- **Decision:** fix all four, bump encoder `v2.0.0 → v2.1.0`, re-cert later.
  Principle: *correctness & robustness is tier 1*; stable-but-wrong is the
  worst outcome.
- **Trade-offs accepted:** certified v2.0.0 receipts are now known to measure
  an un-faded readout — re-cert is pending and blocks nothing until run.

## DEC-003 — Evidence identity (receipts survive their writers)

- **Date:** 2026-10-01 (commit `ad6c861`)
- **Context:** ladder rung dirs were not as_of-scoped (Nov's sizing receipt
  destroyed by Dec's run); monthly head/artifact overwrites under a constant
  tag; null encoder pins could pass the gate; no seed/git stamps.
- **Alternatives:** (a) accept overwrites as "regenerable"; (b) archive
  everything (costly); (c) scope by as_of + archive the run's artifacts.
- **Decision:** per-as_of rung dirs, as-of-stamped heads, `archives/` copies,
  reject null pins, record seed + git SHA. Principle: *correctness* +
  doctrine #10/#11 (receipts are identity).
- **Trade-offs accepted:** artifact dirs grow (archived copies retained on
  purpose); some legacy paths became orphans.

## DEC-004 — white_queen hardening: pre-committed rules, no iterating on the
same battery

- **Date:** 2026-10-01 (commit `0cf06a4`)
- **Context:** known-truth battery v1 = 7/25 errors (1 missed, 6 false), all
  false deploys ≤2 witnesses, several behavior-clones with truth == bar.
- **Alternatives:** (a) witnesses ≥2 alone; (b) certificate-vs-bar alone;
  (c) both; (d) keep tuning until 0 errors on the same battery.
- **Decision:** (c), measured once: 7→4 errors, 0 broken, missed unchanged;
  the 3 survivors (advantage-certificate route) are scoped as round 2 with
  NEW cells rather than tuned against this battery. Principle: *robust &
  antifragile* — a gate tuned on its own test is not a gate.
- **Trade-offs accepted:** simulator recall .39 → .213 on random cards (real
  battery shows no recall loss); errors are 4, not 0.

## DEC-005 — Ship/no-ship gates use measured noise, not literals

- **Date:** 2026-10-01 (commit `85019f0`)
- **Context:** plugin gates were literals (`lift>1.0`, `gap≤0.05`, zero-
  margin AUC-vs-baseline, winner = single-split argmax) — doctrine #1/#8
  violated at the exact point decisions are made.
- **Alternatives:** (a) keep literals (stable); (b) statistical gates
  everywhere (over-engineered); (c) paired-SE where rows are shared, fold-SE
  for singles, derived tolerances, roster-order tie-breaks.
- **Decision:** (c). Principle: *correctness* (a tie must not ship) with
  *simple over complex* (reuse existing `_paired_auc_se`/`_fold_se`; no new
  framework). Verified 5/5 on certified products with headroom.
- **Trade-offs accepted:** gates can now FAIL on statistically-tied data
  that the old rules passed — intended; thresholds are data-dependent so
  receipts must travel with every gate report.

## DEC-006 — Primary goal & the encoder-augmentation path (teach, don't feed)

- **Date:** 2026-10-01 (this entry)
- **Context & problem:** the E-vs-R ablation (P1-4) proved raw RFM beats the
  frozen donor on the shipped target — rabbit_hole 0.765 vs 0.734 and
  Instacart 0.894 vs 0.796 (both beyond paired noise) — while donor+raw beats
  either alone. The user's goal: rabbit_hole → looking_glass → propensity
  must work at its best, by making the encoder LEARN what raw has, without
  bolting hand-built features onto it (D14's sole-input lock stands for now).
- **Alternatives considered:**
  1. Concat raw features into the head input (fastest AUC; violates D14 and
     the "give it things that allow it to learn" directive).
  2. Teach-the-state, in three steps: (S0) decompose raw into components to
     measure WHICH statistic wins; (S1/B) query-time readout — train the
     fade-to-anchor path the serving actually uses (it has never seen a
     gradient); (S2/A) multi-horizon future-aggregate objectives (counts /
     value / days-to-next over gap-quantile horizons) so history-integration
     becomes a learning requirement rather than an input.
  3. Widen the dynamics family (unit-root/integrator expert + timescale span
     from gap quantiles) — capacity, deferred until S0/S1 say it's needed
     (the old "multi-scale gave no gain" verdict was measured under the 1h
     half-life bug and does not transfer).
- **Decision:** option 2, staged per the lifecycle — spike first (S0),
  then correctness (S1/B unconditionally: the serving path must be trained),
  then learning pressure (S2/A, with S3/C only if S0 points at counting and
  S1+S2 miss). **Pre-committed acceptance:** every step must flip
  `plugins.ablation` to `donor_beats_raw=True` on BOTH streams with the
  sufficiency battery (unique ≥75%) and plugin gate (4/4) unchanged.
  **Pre-committed fallback:** if S1+S2 fail the acceptance on both streams,
  the D14 question (heads may take `[E|R]`) is reopened with these receipts —
  no re-litigation. Principles: *correctness first* (S1), *simple over
  complex* (smallest change that could flip the measurement; no new
  framework), *earn the right to scale* (spike before architecture).
- **Trade-offs accepted:** slower than concatenating RFM; encoder version
  bumps again (re-cert); S2 adds loss terms (new hyperparameters must be
  derived + receipted, not tuned by hand).

## DEC-007 — The ablation is the encoder's regression test

- **Date:** 2026-10-01
- **Context:** donor quality was previously "proven" by Spearman-of-Ridge on
  `anchor_embeddings` — a metric/table the product never ships.
- **Decision:** `plugins/ablation` (paired ΔAUC on `donor_embeddings` under
  the production bake-off head) is the standing acceptance instrument for
  encoder changes (see DEC-006), with the sufficiency battery and plugin gate
  as unchanged guards. Principle: *measure everything; gate on ground truth*.
- **Trade-offs accepted:** every encoder change now costs 2×~2min ablation
  runs — cheap insurance.
## DEC-006a — Stage-1 spike results (amends/validates DEC-006 sequencing)

- **Date:** 2026-10-01 (receipts: `plugins/artifacts/ablation_..._20251201.json`,
  `/tmp/opencode/insta_ablation/ablation_..._20251101.json`)
- **Context:** DEC-006 ordered S0 (decompose raw) before aiming S1/S2. Ran
  `plugins/ablation --components` on both streams (component vs donor, paired
  2*SE):
  - **rabbit_hole:** frequency +0.0227 BEATS donor, event_mix +0.0195 BEATS;
    recency −0.1237 (worst single statistic). Winners = history-integration
    statistics (counts / mean gap / per-type histograms over history).
  - **Instacart:** recency +0.0652 BEATS donor (frequency/event_mix/monetary
    all lose). Its negatives are inactivity-defined (30d-censored gaps), so
    time-since-last-event is the label.
- **Decision (validates the DEC-006 order, gives each stage a proven target):**
  1. **S1 (query-time readout)** = the Instacart lever + correctness — the
     serving fade path has never seen a gradient, and recency beats the donor
     exactly where recency defines the label.
  2. **S2 (multi-horizon aggregate targets)** = the rabbit_hole lever —
     counting/histogram prediction is precisely what frequency+event_mix
     require.
  3. **S3 (integrator dynamics / timescale span)** stays conditional on S2.
  No single statistic wins both streams — the state must be able to represent
  BOTH gap-since-last-event and long-window integration; that is the encoder
  work, not a feature list.
- **Trade-offs accepted:** two-stage encoder work instead of one; each stage
  pays a version bump and re-cert.
