# THE SHAPE PROBLEM — the central problem of Layer B

> **Status:** problem statement + program. **Not a measurement surface.** Every number here
> points at a receipt or a DEC; verdicts live only on `REPORT_CARD.md`. Recorded as DEC-588.
>
> **One sentence:** the encoder's recurrent state is a **first moment** (a decayed convex
> combination of token embeddings), so it cannot carry the customer's *distribution* — and
> every red row that looks like a different failure is the same missing statistic measured
> through a different instrument.

---

## 1. What "shape" means in this project

The project's own decomposition is **shape vs mass** (DEC-385): shape is what the customer
*does* (the profile/distribution/order structure); mass is *how much* (the extensive level).
A histogram is shape; a count is mass. A transition matrix is shape; a rate is mass. The
exact counter beats the learned state on persona/future because the counter is a histogram
and the learned state is a mean.

Shape is not one object. Four kinds are measured by different targets:

| Kind | Statistic | What carries it | Where it is measured |
|---|---|---|---|
| **Distributional** | type / value histogram, quantiles | counts over learned buckets, kernel mean embedding | B3.2 persona, B3.6 named parts |
| **Sequential / order** | transition matrix, lead–lag | co-occurrence counts, level-2 path signature (Lévy area) | next-event probes, B4.1 dose |
| **Temporal profile** | activity over time, inter-arrival | multi-timescale / HiPPO projection | B3.2 future fields, gap probes |
| **Identity / segment** | latent group / persona | normalized shape channels + cross-customer separation | B3.2 persona, B3.6 ARI |

## 2. What was tried — and why it did not solve the problem

The mass/shape split **was** tried. The user's memory is correct. The crucial distinction is
*where* it was applied:

| Attempt | Level | What it established | What it did **not** establish |
|---|---|---|---|
| **DEC-385** — shape/mass split ratified as **law** ("no single blended recurrence"; mass purely additive, shape bounded/contractive; "exposure-only variants fail the data-processing inequality") | doctrine | The correct algebra per quantity is a law, not a tunable | The production trunk does not implement it (see below) |
| Early `GatedSSM`/`Enc` prototype (design sketch at the head of `DECISIONS.md`) — additive mass state `M` beside the shape state `H`, both concatenated into the readout | state | The split was conceived at the **state algebra** level | The production `SelectiveSSM` never shipped it |
| **E = shape (donor readout, 256) ‖ learned measure (82)** (DEC-198; §11 of `PROMPT_FORWARD.md`) | served object | Shape and mass are **published** separately; the measure is the ledger | The *learned* shape channel is still a mean; the mass channel is hand-built (PATH A lineage) |
| **B3.4/B3.5** block-aware serving (counter block beside shape; equal 128/128 and 39/39) | serving | Allocation of a mean-based shape cannot exceed the exact counter; the shape block alone is weak (0.081/0.057) | Reallocation cannot create what the accumulation forgot |
| **B3.6** ledger X-ray — named parts (`numeric_stats` 0.3196 persona) vs `ledger_full` (0.0172) | serving/ledger | The distributional *parts* carry the signal; blending dilutes it | The parts are hand-built; the learned shape is not competitive |
| **DEC-465/466/467** action-field profile head — per-field histogram as a linear readout of the state, CE loss | objective | A histogram objective works for the declared **action** fields (`freq hist → x0` R² 0.909) | Only the decision fields; not the general stream; B4.1/B4.2 still failed |
| **DEC-016** effective-rank pressure + variance floor | objective | Rank can be pressured **within the mean's subspace** | A convex combination cannot be spread beyond its input hull; the pressure hits a ceiling (eff_rank 1.2–4) |
| **DEC-577/578/581/582** trunk algebras — bilinear `h⊗x`, looped block, drive-looped block, capacity ×8 | state | Four ways of computing the mean did not move the toy (g 1.147–1.182 vs bar 0.5) | They were judged on an instrument whose **true headroom was 0.048 nats vs ~0.1 noise** (DEC-583/586 unit fix); the verdicts are not evidence about shape |
| **DEC-552/554/555/556** contrastive / DINO / SwAV / VICReg / whitening | objective/geometry | Falsified at the frozen bar; batch-coupled whitening is banned | Did not change the statistic type |
| **Readout skip / event residual** (DEC-048/055) | readout | Restores the current/previous event to a linear head | Relieves pressure on the state to carry order — a workaround, not a fix |

**The production trunk today** (`looking_glass/cfm_model.py`, `SelectiveSSM.forward`):

```text
h_t = a_t · h_{t-1} + (1 − a_t) · B x_t          a_t = exp(−softplus(W_d x_t + b))
```

with `a_t ∈ (0,1)`, so `h_t ∈ conv{B x_1, …, B x_t}` — a decayed **mean**. The additive
`stats_pathway` is another EMA; the in-net `counter` (S1) is an opt-in exact counter that
lives beside the trunk, not in its recurrence. The served object is `E = normalize(proj(h))`;
a linear `proj` cannot raise rank above `rank(h)`. So **DEC-385's law is implemented at the
interface, not in the state algebra** — exactly the "exposure-only" variant DEC-385 says
fails the data-processing inequality.

## 3. The three binding layers

| Layer | Statement | Evidence |
|---|---|---|
| **Capability (algebra)** | The state is a first moment; a mean cannot carry a distribution. | `h_t ∈ conv{B x_s}`; eff_rank 1.2–4; exact counter 0.189 vs learned shape 0.066 |
| **Objective (pressure)** | Nothing asks the state to retain shape; the objectives are satisfiable by a mean. | next-event/mask/value losses; readout skip relieves the pressure; 4/5 `U` probes excluded for zero headroom |
| **Instrument (visibility)** | The toy could not see shape, so four candidate verdicts are uninformative. | true `next_event` headroom 0.0483 nats (DEC-583/586), B-pair noise ≈ 0.1; `g ≤ 0.5` bar sat below the Bayes floor before the unit fix |

All three are binding. Fixing one alone has failed and will fail again.

## 3a. Phase-0 result (measured 2026-11-04) — the strong diagnosis is FALSIFIED

`scripts/measure_shape_battery.py` (DEC-588 Phase 0; receipt
`docs/report_cards/experiments/shape_battery.json`) fitted a standardized ridge on split A and
scored split B (n_B = 1485; paired bootstrap 2000):

**Does E encode the HISTORY's own shape?** (the counter arm is the target itself, so omitted)

| History statistic | R² from E | 95% CI | shuffle control |
|---|---|---|---|
| type histogram (6) | **0.9943** | [0.9935, 0.9949] | −0.0074 |
| gap quantiles (4) | **0.9766** | [0.9748, 0.9782] | −0.0183 |
| value quantiles (4) | **0.9844** | [0.9829, 0.9858] | −0.0115 |
| transition matrix (36) | **0.5020** | [0.4874, 0.5164] | +0.0043 |

**Can E predict the FUTURE shape as well as the exact counter?** (90-day window)

| Future statistic | E | counter | Δ(E−counter) | 95% CI |
|---|---|---|---|---|
| type histogram | 0.3581 | 0.3684 | −0.0103 | [−0.0187, −0.0022] |
| transition matrix | 0.1961 | 0.2037 | −0.0077 | [−0.0118, −0.0037] |
| gap quantiles | 0.5075 | 0.5159 | −0.0084 | [−0.0163, −0.0010] |
| value quantiles | 0.7712 | 0.7712 | +0.0001 | [−0.0041, +0.0040] |

**Read.** The strong form of the diagnosis — "the state is a mean, so it cannot carry a
distribution" — is **false on cfm_v2k**. A mean of one-hot/embedding features *is* a
histogram, and E decodes the history's type histogram, gap quantiles and value quantiles at
0.98–0.99; on future shape it is within ~0.01 R² of the exact counter (value: equal). What E
**does not** carry is the **second-order/order** statistic: the transition matrix decodes at
**0.502**, i.e. half of the order variance is missing from the state. The persona rows are
**contaminated** (segment/tier/region are in every event's `event_attributes`, so E reads
them directly — identity leakage, cf. B4.3 check-2; the counter has no persona fields). The
future targets are aligned with the training heads (rate/value/survival), so their absolute
R² is inflated; the comparator, not the absolute, is the claim.

**Revised statement of the central problem.** First-order shape is carried; **order /
co-occurrence (the second moment) is the missing statistic**, and it is exactly what the mean
cannot carry. The mass/shape split was tried and largely worked for first-order shape; the
missing piece is the order channel. Geometry (eff_rank 2.4) is a separate axis: the battery
shows low rank ≠ low information, so the geometry gate must not be conflated with shape
sufficiency.

## 4. The combination that should work

A **combination** is required because each layer blocks the others. The Phase-0 result
reorders the carriers: **order first**, first-order shape already largely present.

1. **Order channel (the binding gap) — level-2 co-occurrence in the state.**
   - Level-2 path signature in a projected space `r ≈ 32–64`: the symmetric part
     (covariance / co-occurrence) + the antisymmetric part (Lévy area / lead–lag). Chen's
     identity makes it an exact monoid; it is the algebraic carrier of "what follows what"
     and the only candidate that addresses the measured 0.502 transition decode.
   - If the battery shows associative recall (not just co-occurrence) is binding, escalate to
     a low-rank **delta-rule** associative memory (`S ← S(I−βkkᵀ)+βvkᵀ`), which is the
     error-correcting version of the same idea.
2. **Mass/shape contract (already largely carried) — keep it, don't expect it to close order.**
   - `m_t` mass: purely additive, no `(1−a)` normalization, monotone compressed, learned
     projection; mergeable by addition.
   - `s_t` shape: bounded/contractive, per-customer normalized; published separately (polar
     interface law). Formalizing the two channels is still correct law (DEC-385), but the
     battery shows the mean already carries the first-order histogram/quantiles; this is not
     where the order gap lives.
3. **Objective — pressure for order.**
   - Predict the next window's **transition matrix** (row distributions) and its
     histogram/quantiles from the served state only; distributional scoring rule
     (CE/CRPS/energy), not histogram-MSE. This generalizes DEC-465 beyond declared action
     fields and gives the order channel a gradient.
4. **Instrument — the shape battery (landed) + a sharper order target.**
   - The battery is live; the order target needs to be less noisy than a 90-day transition
     matrix (e.g. a longer future window, or a per-customer next-type log-loss against the
     1-gram), so a 0.50 → higher movement is measurable.
   - The mixture-transition toy where pair statistics are necessary and a mean provably
     fails remains the lab instrument.
5. **Geometry — a separate axis, not shape sufficiency.**
   - Per-customer intensiveness (no batch statistics); centering/high-pass as a diagnostic
     for the DC confound. The battery shows eff_rank 2.4 with 0.99 histogram decode: low rank
     is not low information, so the geometry gate and the shape battery must be reported
     separately.
6. **Escalation only by evidence.**
   - Delta-rule memory if the battery shows **associative recall** is binding; bounded exact
     window if the **bit-capacity theorem** is demonstrated.

Why the combination and not one piece:

- Algebra alone: the new channels can be ignored by the loss (the readout skip already lets
  the model bypass the state).
- Objective alone: the state cannot represent the target (data-processing inequality).
- Instrument alone: measures the gap and changes nothing.
- Geometry alone: reparameterizes a mean; adds no information.
- Capacity/nonlinearity alone: falsified at 8× and on four algebra candidates.

## 5. Solved when (the shape problem's own acceptance)

At equal width, on cfm_v2k, out-of-sample, paired CIs:

1. **Order:** E's decode of the history transition matrix rises materially above the measured
   **0.502** (the 1.0 ceiling is the counter itself), and the future transition shape beats
   the exact counter. This is the binding row after Phase 0.
2. **First-order shape (already met, keep it met):** E stays within noise of the exact
   counter on future type histogram, gap quantiles and value quantiles (measured deltas
   −0.010 / −0.008 / +0.000; do not regress them while adding order).
3. **Decision:** the B4.1 dose / B4.2 OPE reds move on the real stream (if the order channel
   is what the level-conditional response needs) — or the decision diagnosis is separated
   from the shape problem and recorded as its own axis.
4. **Geometry:** the geometry reading is reported alongside, not conflated: the battery
   measured eff_rank 2.4 with 0.99 histogram decode, so low rank is not automatically
   insufficient; any geometry bar change is operator-only (gate).
5. The real-stream rows that depend on shape move (B3.2 lineage, B6.2's 90-day value/order
   targets); lab receipts never flip a B-row verdict (DEC-576), and no gate, bar, criterion
   or exclusion is edited to make any of this true.

## 6. The loop (binding)

One axis per run; pre-register (DEC) before the run; freeze the falsifier; revert on no
movement; re-stamp on every behavior move (S6 chain + B-row producers, no-op proof);
record the three §3.0 layers (`WHY-NOT / EVIDENCE / GAP / LAYER / THIS RUN`).

**Phase gates.** Phase 0 (battery) must land before any architecture change; Phase 1
(mass/shape state + objective) must show battery movement before Phase 2 (signature/KME);
Phase 3 (delta-rule/hybrid) only on the evidence named above.

## 7. Pointers

- Law: `DECISIONS.md` **DEC-385** (shape/mass split), DEC-386 (engine), DEC-388
  (accessibility parity).
- Evidence: B3.2/B3.4/B3.5/B3.6 in `scripts/measure_block_serving.py` and the card; B6.2
  (`encoder_donor.json`); B4.1/B4.2 (`encoder_decision_law.json`); the DEC-583/586 unit
  correction.
- Program: **DEC-588** (this document + the shape battery + the phase gates).
- Measurements: `REPORT_CARD.md` only.
