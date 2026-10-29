# DEC-422 trial receipts — DEC-421 matched control (bilinear content×state)

**Verdict: NOT adopted.** `bilinear_recurrence` stays `False`. See DEC-421 (frozen gate) and
DEC-422 (measured outcome) in `DECISIONS.md`. These are *trial* receipts: they are deliberately
kept out of `docs/report_cards/experiments/` so the judge card cannot read them as canonical.

What each file is:

| File | Arm | Source script |
|---|---|---|
| `s5_battery_baseline.json` | control | `scripts/measure_battery_law.py` |
| `s5_battery_bilinear.json` | treatment | `scripts/measure_battery_law.py` |
| `s3_dynamics_baseline.json` | control | `scripts/measure_dynamics_law.py` (7/7 PASS) |
| `s3_dynamics_bilinear.json` | treatment | `scripts/measure_dynamics_law.py` (`ssm_merge_exact` FAIL: 7.904e-03 vs 1e-03) |
| `registry_baseline.json` | control | trainer registry (`bank_rank` 0.0321, `lambda_final` 1e6) |
| `registry_bilinear.json` | treatment | trainer registry (`bank_rank` 0.0196, `lambda_final` 1e6) |

## Reproduce (all commands from the repo root)

The demo stream is deterministic (seed 17) and rebuilds in ~11 s:

```sh
.venv/bin/python scripts/make_demo_data.py --out ~/wonderland_work/demo
```

Train the matched pair — ONE arm at a time (two concurrent trainers buy no throughput and OOM the
unified-memory GPU when another GPU tenant is resident; see DEC-422 cost account):

```sh
.venv/bin/python -m looking_glass.customer_foundation_model train \
  --customers 2500 --anchors 4 --as-of 2025-01-01 \
  --db ~/wonderland_work/demo/customer_event_stream.parquet \
  --out-dir ~/wonderland_work/cfm_s3b

.venv/bin/python -m looking_glass.customer_foundation_model train \
  --customers 2500 --anchors 4 --as-of 2025-01-01 \
  --db ~/wonderland_work/demo/customer_event_stream.parquet \
  --out-dir ~/wonderland_work/cfm_s4_bilinear --set bilinear_recurrence=true
```

Score both arms on the identical frozen probes (write OUTSIDE the canonical experiments dir):

```sh
for arm in cfm_s3b cfm_s4_bilinear; do
  .venv/bin/python scripts/measure_battery_law.py \
    --products ~/wonderland_work/$arm/cfm_products.duckdb \
    --demo ~/wonderland_work/demo \
    --out  ~/wonderland_work/battery_$arm.json

  .venv/bin/python scripts/measure_dynamics_law.py \
    --ckpt-dir ~/wonderland_work/$arm \
    --train-log ~/wonderland_work/cfm_train_$arm.log \
    --db ~/wonderland_work/demo/customer_event_stream.parquet \
    --out ~/wonderland_work/dyn_$arm.json
done
```

Evaluation contract (frozen, not tuned): seed 65537, N_BOOT=2000, 20k pairs, ≥10 arm-days/side
uplift identifiability — constants live in `scripts/measure_battery_law.py` (DEC-410).
