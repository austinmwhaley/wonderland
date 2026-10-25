# wonderland — interactive demo

A single self-contained, light-theme HTML page that presents the result: one frozen
self-supervised encoder per event stream, reused by supervised, unsupervised, and
reinforcement-learning heads. It shows the winning comparisons — the full evidence
audit, including every open item, lives in `REPORT_CARD.md` and `docs/AUDIT_CARD.md`.

Open it directly — no server, no network, no dependencies:

```bash
python3 scripts/build_demo_site.py     # regenerate from receipts
xdg-open demo/index.html               # or just double-click the file
```

## Tabs

| Tab | What it shows |
|---|---|
| **Overview** | The claim, headline numbers, selected-view vs table, and what each state view carries |
| **Encoder vs Traditional** | Same head and holdout: table vs shape vs measure vs forecast vs full, deltas with 95% CI, and the floor → encoder → ceiling ladder |
| **State Views** | Per-view results, identity purity, unsupervised probes, and view dimensions |
| **Plugins** | Known-effect recovery by action key, and RL policy value against uniform and table comparators |
| **Customer Explorer** | Holdout customers in the frozen embedding; click a point for its history, lookalikes, and held-out outcome |

## Where the numbers come from

Every value is generated from receipts under `docs/report_cards/experiments/`:

- `foundation_state.json` — shape / learned state / forecast / measure / full attribution
- `view_selection.json` — train-only view selection and holdout evaluation
- `encoder_ceiling.json` — floor / encoder / full-history ceiling
- `encoder_donor.json`, `encoder_identity.json`, `plugin_unsupervised.json`, `plugin_rl_known.json`
- `plugin_response.json` — pre-action RL policy value
- `runtime.json`, `encoder_config.json`, `data_receipt.json` — provenance and cost

The customer explorer additionally samples the frozen cache at `/tmp/opencode/frozen_6k` and the
canonical stream. If those are absent the page still builds; the explorer tab degrades with a
message.

Nothing on the page is hand-typed, so it cannot drift from the reported evidence.

## Honesty

This page is a presentation of the positive results. Every number on it is a real measured
number; losing or tie-level comparisons are simply not displayed here. The complete record —
including category identity, anomaly/segmentation, the churn tie, and Layer-1 status — remains
in `REPORT_CARD.md` and `docs/AUDIT_CARD.md`, which are the authority.

Two protocols appear on different tabs and are labelled at their charts: the train-only view
selection holdout (table 0.424 → encoder 0.457) and the fixed-head attribution protocol
(table 0.357 → frozen state 0.433).

## Notes

- Synthetic relational harness; no business data.
- The explorer's "lookalike" panel is computed in the frozen embedding and is **not** used to form
  the actual outcome.
- Regenerate after any receipt changes so the page stays current.
