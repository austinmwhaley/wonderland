# wonderland — interactive demo

A single self-contained HTML page that presents the measured evidence: one frozen
self-supervised encoder per event stream, reused by supervised, unsupervised, and
reinforcement-learning heads.

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
| **Customer Explorer** | Sampled holdout customers in the frozen embedding; click a point for its history, lookalikes, and held-out outcome |

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

The page states plainly that the compact 128-dimension `shape` does **not** beat the table on
value/orders, that category identity and anomaly/segmentation still trail the table, that the RL
policy value is non-dominant, and that Layer-1 certification is not started. Those are the same
disclosures as `REPORT_CARD.md` and `docs/AUDIT_CARD.md`. The footer carries a short pointer to
them; the full disclosure list lives in the card rather than on a separate tab.

## Notes

- Synthetic relational harness; no business data.
- The explorer's "lookalike" panel is computed in the frozen embedding and is **not** used to form
  the actual outcome.
- Regenerate after any receipt changes so the page stays current.
