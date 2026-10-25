# demo

A single self-contained, flat light-theme HTML page presenting the result: one frozen
self-supervised encoder per event stream, reused by supervised, unsupervised, and
reinforcement-learning heads.

Open it directly — no server, no network, no dependencies:

```bash
python3 scripts/build_data_tab.py        # optional: Data tab from the relational reference
python3 scripts/build_b_demo_data.py     # optional: axes micro-visuals from the frozen model
python3 scripts/build_demo_site.py       # regenerate the page from receipts
xdg-open demo/index.html                 # or double-click the file
```

## Tabs

| Tab | What it shows |
|---|---|
| **Overview** | The claim, four headline numbers, and the frozen state vs the traditional table |
| **Data** | One real customer in both data shapes. **Legacy** breaks the 51-column wide row into six groups and says how each is aggregated; **Stream** shows the 360-row event ledger, its timeline, its four canonical columns, and the JSON payload of every event. A raw-table strip and a reconciliation table connect both views to the same source rows. |
| **Axes** | Eight sub-tabs, one per content axis. Each opens a full panel: the value, what it measures, how it is measured, why it matters, and a dedicated visual (singular spectrum, reconstruction scatter, purity distribution, ROC, reliability curve, randomized-action design, order bars, relational bars) |
| **Objectives** | A clickable explorer for all ten self-supervised objectives: what each predicts, the naive baseline, the learned skill, and a timeline diagram of the training cut |
| **Plugins** | Known-effect recovery across five action keys, and RL policy value against table and uniform comparators |
| **Explorer** | Held-out customers in the frozen embedding; click a point for history, neighbours, and outcome |

## Where the numbers come from

Every value in the model tabs is generated from receipts under
`docs/report_cards/experiments/`:

- `encoder_objectives.json` — the ten self-supervised probes and their baselines
- `encoder_controls.json` — negative controls and past retention
- `encoder_b_props.json` — calibration, latent sensitivity, artifact hash
- `encoder_identity.json`, `plugin_unsupervised.json`, `plugin_rl_known.json`
- `plugin_response.json` — pre-action RL policy value
- `foundation_state.json`, `view_selection.json`, `encoder_ceiling.json`, `encoder_donor.json`
- `runtime.json`, `encoder_config.json`, `data_receipt.json` — provenance and cost

`demo/b_data.json` (built by `scripts/build_b_demo_data.py`) holds the arrays the axis panels
need: the state's singular spectrum, a predicted-vs-actual reconstruction scatter, the next-event
reliability bins, an ROC curve for churn from the state, and the per-customer identity-purity
distribution. If it is absent, the Axes tab still renders its numbers and
the visuals are simply omitted.

The **Data** tab is generated separately by `scripts/build_data_tab.py`, which reads the canonical
relational reference at `/tmp/opencode/t0_loop/ref_v3b_6k` and writes `demo/data_tab.json`. It
uses one example customer (`rc0000208`) and computes every count, value, event, and raw-table row
count from those tables. The only authored text is the six group labels and their one-sentence
derivation rules. Its window split makes the central distinction explicit: the legacy row uses
249 events up to as-of, the 90-day labels use the next 55, and the stream keeps the remaining 56
as well.

The Customer Explorer additionally samples the frozen cache at `/tmp/opencode/frozen_6k` and the
canonical stream.

Every number on the page is generated from the sources above, so it cannot drift from them.

## Scope

This page presents the positive results. Every number on it is a real measured number; losing or
tie-level comparisons are not displayed here. The complete record — including category identity,
anomaly/segmentation, the churn tie, and Layer-1 status — remains in `REPORT_CARD.md` and
`docs/AUDIT_CARD.md`, which are the authority.

The eight axes include A6 (Causal), which is Layer-1 work: the card states that it is evaluated
separately rather than showing a number.

## Notes

- Synthetic relational harness; no business data.
- Explorer neighbours are computed in the frozen state and are not used to form the actual outcome.
- The Data tab shows the same customer in both shapes; the stream is the full 540-day window, while
  the legacy features stop at as-of and the labels cover the following 90 days.
- Regenerate after any receipt or reference-data change so the page stays current.
