# Demo — 100% completion contract and proof

## Definition of 100% done (frozen)

The demo is 100% done only when **all** of the following are true:

1. **One repo command** reproduces the demo end-to-end at a recorded commit: build/validate v3,
   train the frozen auto E on GPU, run the eight-axis B probes, run the B+C heads vs the raw
   table + XGBoost GPU, and write both cards. It records N, hashes, runtime.
2. **B card fully measured**: all eight content axes and all foundation properties have a metric,
   a baseline, uncertainty/CI, and a status. No "pending"/"not measured" rows.
3. **B+C card**: two supervised, two unsupervised, two RL heads on the frozen E vs the traditional
   table + XGBoost GPU, with paired CIs; claims only where the CI excludes zero; every table win
   disclosed. No hybrid.
4. **Layer A**: current v3 data card/certificate, linked from the demo card.
5. **Alignment**: no hand summary/counter/table inside E; self-supervised; frozen; the hand-summary
   experiment is not used in the claimed E.
6. **Runtime**: fast path within budget; the loaded GPU actually utilised; 6k full run reproducible.
7. **Reproducibility**: single command, git commit, data build command, artifact hashes.

## Proof matrix (current, honest)

| # | Requirement | Evidence | Verdict |
|---|---|---|---|
| 1 | One command | `scripts/demo/fast_demo.py` is 2k-only; B probes and B+C are separate `/tmp` scripts; final card is static | **FAIL** |
| 2 | B fully measured | 8 axes present; uncertainty OOD, causal CI, lawfulness, minimality, accessibility, artifact still pending | **FAIL** |
| 3 | B+C 2/2/2 vs XGBoost GPU with paired CIs | 6k quick 1 seed/5; 2k 1 seed/5; value +4.4%; orders/churn ties; unsup/RL point estimates; RL CI crosses | **PARTIAL** |
| 4 | Layer A current | `LAYER_A_DATA.md`; v3 certificate not regenerated | **PARTIAL** |
| 5 | No hand summary in E | learned additive mass; hand summary removed | **PASS** |
| 6 | Runtime/GPU | GPU train/eval; `load_data` 58s + `collate` CPU-bound (GPU idles) | **PARTIAL** |
| 7 | Reproducibility | commits exist; static card; no single command; data build manual | **PARTIAL** |

## Verdict

**NOT 100% DONE.** The encoder and both cards are current and honest, but five of seven
completion requirements are not yet satisfied.

## Remaining critical path

1. Single `scripts/demo/demo.py` command (data → E → B probes → B+C → both cards; hashes/N/runtime).
2. Vectorize/cache `collate` (GPU utilisation); remove the 6k CPU bottleneck.
3. Fill the B card: uncertainty OOD, causal IPS CI, lawfulness/merge/permutation tests,
   minimality/retention, accessibility linear/kNN, artifact.
4. B+C: 3-seed/20 paired CIs on the selected heads at 6k; two per family.
5. Regenerate the v3 Layer A certificate and link it from the demo card.

Only after all five are green can the demo be called 100% done.
