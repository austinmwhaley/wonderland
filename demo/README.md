# wonderland — interactive demo

A single self-contained, flat light-theme HTML page presenting the result: one frozen
self-supervised encoder per event stream, reused by supervised, unsupervised, and
reinforcement-learning heads.

Open it directly — no server, no network, no dependencies:

```bash
python3 scripts/build_b_demo_data.py     # optional: axes micro-visuals from the frozen model
python3 scripts/build_demo_site.py       # regenerate the page from receipts
xdg-open demo/index.html                 # or double-click the file
```

## Tabs

| Tab | What it shows |
|---|---|
| **Overview** | The claim, four headline numbers, and the frozen state vs the traditional table |
| **Axes** | The eight content axes, one card each, with a micro-visual (spectrum, reconstruction scatter, purity, reliability, order, relational) |
| **Objectives** | A clickable explorer for all ten self-supervised objectives: what each predicts, the naive baseline, the learned skill, and a timeline diagram of the training cut |
| **Plugins** | Known-effect recovery across five action keys, and RL policy value against table and uniform comparators |
| **Explorer** | Held-out customers in the frozen embedding; click a point for history, neighbours, and outcome |

## Where the numbers come from

Every value is generated from receipts under `docs/report_cards/experiments/`:

- `encoder_objectives.json` — the ten self-supervised probes and their baselines
- `encoder_controls.json` — negative controls and past retention
- `encoder_b_props.json` — calibration, latent sensitivity, artifact hash
- `encoder_identity.json`, `plugin_unsupervised.json`, `plugin_rl_known.json`
- `plugin_response.json` — pre-action RL policy value
- `foundation_state.json`, `view_selection.json`, `encoder_ceiling.json`, `encoder_donor.json`
- `runtime.json`, `encoder_config.json`, `data_receipt.json` — provenance and cost

`demo/b_data.json` (built by `scripts/build_b_demo_data.py`) holds the small arrays the axes
micro-visuals need: the state's singular spectrum, a predicted-vs-actual reconstruction scatter,
and the next-event reliability bins. If it is absent, the Axes tab still renders its numbers and
the visuals are simply omitted.

The Customer Explorer additionally samples the frozen cache at `/tmp/opencode/frozen_6k` and the
canonical stream.

Nothing on the page is hand-typed, so it cannot drift from the reported evidence.

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
- Regenerate after any receipt change so the page stays current.
