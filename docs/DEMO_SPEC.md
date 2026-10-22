# Demo contract — Layer B (encoder) + Layer C (plugins), Layer A last

Status: **frozen for the demo build**. Changing a target, metric, baseline, split, seed,
or CI rule is a new contract version.

## Scope

- Data: the synthetic relational reference only (`relational_v3`, 2k and 6k). No business data.
- Label: synthetic **harness**, not business value; no causal claim beyond the randomized holdout.
- Layers: B encoder first, C plugins second, A data documentation last.
- Deliverable: one **B+C report card** and a fast live run.

## Encoder (Layer B)

- Per-stream, self-supervised, frozen after training. No labels/table/counter/summary inside `E`.
- Tokenizer: additive; auto-discovers attribute fields (categorical / numeric / ISO-time) with
  learned per-field gates; the reference's behavioural fields are the base set.
- Dynamics: continuous-time gated SSM, learned `τ` initialized from purchase-gap quantiles;
  data-derived Fourier periods; `qgap`; `_scan`.
- Readout: last / mean / max / attention + learned positive weights; `mass=explicit` branch.
- Objectives: next-event + sampled horizons + all-schema past-summary; learned loss weights.
- Fast: 2k, 1 seed, 5 epochs. Full card: 6k, 3 seeds, 20 epochs.
- Freeze `E` before any C plugin; record the E hash and training runtime.

## Layer C plugin menu

### Supervised (6)

value30, value90, orders30, orders90, churn30, churn90. Metric: R² / AUC. Arms: table, E only,
hybrid. 5-fold GBM; paired bootstrap CI on the holdout (fast: 1 seed; full: 3 seeds).

### Unsupervised (4)

- segmentation: KMeans on E; NMI/ARI vs truth `regime0` and `category_pref`; table baseline.
- anomaly: truth `return_proneness` top decile; IsolationForest/kNN score; AUC; table baseline.
- retrieval: kNN outcome (value/churn) consistency and latent purity; table baseline.
- representation: effective rank / isotropy; E vs table.

### RL (3)

Logged daily actions from the reference; reward = future gross margin; randomized holdout.

- channel bandit: arm = contact channel; E/table state; policy value via IPS/DR; truth `channel_pref`.
- offer bandit: arm = offer; E/table state; policy value via IPS/DR; truth `offer_opt`.
- OPE / known-effect: randomized-holdout IPS/DR estimate; random and table comparators; CI.

Fast: 2k holdout, one action per customer (last logged send before as-of). Full: 6k,
time-resolved actions. No policy is shipped; this is an offline evaluation.

## Card

Single generated Markdown card:

1. Header: data/version, N, seeds, runtime, E hash, synthetic-harness label.
2. B: E-only vs raw and fair table; E add-on to `table+seq` with paired CI; feature ablations;
   schema-adaptability probe; geometry/transfer; speed.
3. C: supervised table; unsupervised table with truth reference; RL table with IPS/DR CI and
   known-effect recovery; comparators.
4. Caveats: fair-table comparator, 2k vs 6k, churn trade, low-rank geometry, hand-listed core +
   auto extensions, randomized-only causal scope.

## Acceptance

- Fast path end-to-end < 10 min on the 2k reference.
- Every PASS has a finite comparator and a paired/randomized CI; no PASS from a point estimate.
- E is frozen before C; no label leakage into B.
- One command reproduces the fast card at a recorded commit; the 6k full card comes from receipts.
- Layer A documentation/certificate is regenerated last and matches the v3 reference.
