# Structural problems — wonderland / looking_glass

Status: strategic review (DEC-235). This document names the **structural** problems, not
the measurement bugs. It separates what is fixed, what is a product-scope decision, and
what requires a decisive experiment. It is binding on the roadmap: no “SOLVED” claim is
allowed while any item below is open.

## 0. The project bundles three different bets

| Bet | What it is | Evidence today | Verdict |
|---|---|---|---|
| **A. Automated, schema-agnostic measure + point-in-time decision pipeline** | Generic feature construction, leakage-safe as-of features, response blocks, null controls, certification harness | Works: 8/8 schemas, point-in-time gate, known-effect 5/5, deterministic, 456 tests | **Strong — the reliable asset** |
| **B. Learned self-supervised encoder that beats the measure** | SSM donor/shape, distillation, counter | Donor-only U **0.358**; donor+measure 1.0; transfer **0 add / 5 NO_SIGNAL**; unsupervised FAIL; shape adds only on channel/category | **Unproven, currently losing to the measure** |
| **C. “Universal” across streams** | One architecture/pipeline; per-stream training | One predictive stream + schema-ingestion suite; no cross-stream predictive test; no shared weights | **Weakest — “universal” is the pipeline, not the model** |

The past confusion came from treating A’s success as evidence for B and C.

---

## P1 — Oracle circularity: the measure is inside E and defines the oracle

**Statement.** `E = shape ‖ measure`; the future-family oracle is the full-history measure.
U=1 is nearly guaranteed by construction, and the learned shape is never forced to beat the
measure — only not to lose.

**Evidence.** `looking_glass/cfm_training.py` appends the non-response `generic_measure`
columns to the shape; `u_controls` measures shape-U against the same measure
(`seqs=None`); `scripts/shape_attribution.py`: learned donor only **U 0.3576**, donor+measure
**1.0**, donor+measure full-E **0.7208×**.

**Why it matters.** The central research claim (“learned shape is sufficient”) is not what
the certificate measures. A measure-carried certificate is fine as a product claim, but it
must not be sold as a learned-representation claim.

**Resolution (claim fix + attribution).**
- Report **learned-only**, measure-only, donor+measure, and served E separately on the card.
- Label the claim “served-statistic sufficiency (measure-carried)” unless learned-only clears
  the bar.
- Do not put the oracle’s features into the shape for a “learned-shape” claim.

**Status:** attribution receipt implemented; card labels pending final regeneration. **Not a
model fix** — a claim/measurement fix.

---

## P2 — Testbed insufficiency: the reference stream cannot test the learned bet

**Statement.** The synthetic reference is additive/RFM-like, one fixed schema, small
vocabulary, and many self-supervised objectives are saturated (`NO_SIGNAL`). A sufficient
measure is easy by construction.

**Evidence.** `rabbit_hole/generators/reference.py` (one-factor daily, additive soft bowls);
§3.1 `NO_SIGNAL` rows; transfer/decisive results are negative/neutral.

**Why it matters.** We cannot tell whether the learned encoder is a good idea from this
stream. Optimizing the frontier here is optimizing the harness.

**Resolution (decisive experiment B).** Build one **measure-insufficient** stream:
variable-order/long-memory dependence, nonlinear interactions, exogenous shocks,
heterogeneous cadence, confounders. Baselines: measure+GBM, measure response block+policy,
learned E+same head, oracle. Pre-register the bar and a kill criterion.

**Status:** open; this is the experiment that decides whether B is worth continuing.

---

## P3 — Per-stream economics and the “universal” claim

**Statement.** A model is trained per stream (no shared weights). “Universal” is the
pipeline, not transfer learning. Each stream costs a training run and ~2000 customers.

**Evidence.** AGENTS.md (“one stream in → one model out”); no cross-stream predictive test;
one predictive stream.

**Why it matters.** Foundation-model economics assume amortization. Per-stream training is
closer to AutoML; the moat is process, not a reusable model. If there are many streams, the
cost may not pay; if few, a measure+GBM may be enough.

**Resolution (product-scope decision).**
- Measure cost per stream and value per stream (customers, decisions, margin).
- Decide: few high-value streams (learned R&D can pay) vs many cheap streams
  (measure-first pipeline).
- Do not use the word “universal” without a cross-stream predictive test.

**Status:** open; requires a business decision and a cost measurement.

---

## P4 — Objective mismatch: sufficiency is not usefulness

**Statement.** The project optimized “nothing is lost.” That does not imply “better
decisions.” The reference’s decision lift is conversion lift, not the stated objective
(incremental gross margin).

**Evidence.** T3 supervised 2 WIN / 6 tie vs fair generic; transfer 0 add/5 NO_SIGNAL;
decision gate is measure-driven; AGENTS objective = discounted incremental gross margin.

**Why it matters.** A sufficient statistic can be commercially worthless. The demo must show
a business metric, not AUC/U alone.

**Resolution.** Make **usefulness** a first-class gate: incremental margin on a randomized
holdout, per-consumer cost, and a fair baseline. Pre-register the pass bar.

**Status:** open; needs a business metric and a holdout.

---

## P5 — Consumer evidence is weak or tautological

**Statement.** Several consumer gates are weaker than they look:
- transfer: supervised **0 add / 5 NO_SIGNAL**; the old unsupervised PASS was a
  `raw=nan` row (now **FAIL** vs a fair raw/RFM comparator: eta² 0.2525 vs 0.2837,
  CI [−0.0606, −0.0051]);
- unsupervised “E vs generic” compares the measure block to the same generic measure
  (tautology at the block scope);
- decision gate is model-independent (real lifts bit-identical to 0.6.4);
- stream OPE **0/1**; the broad acceptance 9/9 is a different contract.

**Why it matters.** The “plugins work” story currently rests on the measure and on
scope-limited gates, not on the learned donor.

**Resolution.** Keep the measure-first product claim; report both scopes; require the
random-donor control; do not count tautological gates as transfer evidence.

**Status:** transfer comparator fixed; card disclosure pending; donor increment is still
negative.

---

## P6 — Gate/claim drift (governance)

**Statement.** Twice the project produced a “SOLVED” label that a later review overturned
(DEC-228, DEC-234). The causes: post-hoc minimality criterion, frontier relabeling, stale
card text.

**Resolution (implemented).**
- `AGENTS.md` → **Certification integrity** (freeze the gate before the result; T0 ≠ SOLVED;
  no receipt-defined criteria; no circular evaluation; finite comparators; single-command
  reproducibility; card consistency; stop-and-ask).
- `looking_glass/objectives.py` now rejects `block_wise_certified=true` with a non-frozen
  criterion (regression test added).
- `scripts/reference_certify.py` is the single-command certifier.

**Status:** fixed at the process level; the card must be regenerated under it.

---

## P7 — Reproducibility and provenance

**Statement.** Historically the card/receipts were in `/tmp`, the card’s git did not match
the solving commit, and there was no single command.

**Resolution.** `scripts/reference_certify.py` (refit → unserved rank → attribution →
frozen serve → certificates → receipts → objectives → cockpit), run at a recorded commit;
N and hashes on the card/manifest.

**Status:** implemented; final run in progress.

---

## P8 — Reference table is not derivable from the stream

**Statement.** PATH A contains columns that PATH B cannot reveal: `segment` and the initial
`loyalty_tier` are random (`rng.integers`) and absent from the stream; `signup_ts` is random
1–399 days before the window while `customer_signup` is emitted at simulation day 0, so
`tenure_days` is hidden (the stream's first-event tenure is understated by 1–399 days).

**Impact.** The table is a valid predictive baseline but not a derived view of the stream,
violating the "stream contains all information / table is derived" principle. Measured impact
is small on the 2k reference (tenure ablation value 0.349→0.344, orders 0.352→0.355), but the
fairness claim is unproven until fixed. The random static columns also hurt the table
(value/orders improve when they are dropped: 0.349/0.352 → 0.364/0.377).

**Fix options.** (a) Emit `customer_signup` at the true `signup_ts` with `segment` /
`loyalty_tier` attributes → rebuild PATH A/B and re-certify (new reference version).
(b) Keep the data; drop the non-derivable random columns from the fair baseline and document
them as unused CRM noise. The E-only claim is currently reported against the fair table.

**Status:** open; operator decision required before changing the certified reference.

---

## Two-track plan

**Track A — measure-first product/demo (the reliable asset).**
- Ship the schema-agnostic measure + point-in-time features + response blocks + fair
  baselines + the certification harness.
- Demo on the reference (labeled harness) and one real public event stream (predictive/
  segmentation); show incremental margin only where a randomized holdout exists.

**Track B — learned-encoder research (one decisive bet).**
- Run the measure-insufficient experiment (P2) with a pre-registered bar and a kill criterion.
- If learned E does not beat the measure on held-out value/decisions, stop B; keep A.

**Kill criterion for B (pre-registered).**
On the measure-insufficient stream, learned E must beat measure+GBM on (a) held-out value and
(b) incremental margin on a randomized holdout, with a CI excluding 0. If it does not, the
learned-foundation track is closed and the project ships as measure-first.

---

## Demo definition (cheapest honest demo)

1. One command ingests a stream (reference or a real public event stream).
2. Produces point-in-time features, a response block, and a policy.
3. Runs a randomized-holdout lift (reference) or a held-out prediction/segmentation
   (real data), with a fair baseline and CI.
4. Prints the certification card with scope and caveats.
5. Labels clearly: **harness** vs **real-data**, **predictive** vs **causal**.

The cheapest version is Track A on the existing reference plus **one** real public stream,
with the caveats on the first page. Do not wait for Track B to demo A.

---

## Status matrix

| ID | Problem | Fixed? | Next |
|---|---|---|---|
| P1 | Oracle circularity / measure-carried U | Attribution + labels | Card regeneration |
| P2 | Testbed cannot test the learned bet | **No** | Build measure-insufficient stream |
| P3 | Per-stream economics / “universal” | **No** | Cost/value decision |
| P4 | Sufficiency ≠ usefulness | **No** | Business metric + holdout |
| P5 | Weak/tautological consumer evidence | Comparator fixed; donor negative | Card disclosure |
| P6 | Gate/claim drift | **Yes** (AGENTS + reader + test) | Enforce every run |
| P7 | Reproducibility/provenance | Certifier implemented | Final run + commit |
| P8 | Table not derivable from stream (random `segment`/`loyalty`, hidden `signup_ts`) | **No** | Operator: fix generator/re-certify or fair-table baseline |
