#!/usr/bin/env bash
# loop/status.sh — THE GOAL PREDICATE.
# Exit 0 (and print GOAL_MET) iff the goal holds; else print the blockers and exit 1.
# The driver (loop/run.sh) calls this before each iteration to decide stop/continue.
#
# EDIT the STREAMS table below to point at the current checkpoint dirs the loop trains
# into. Each entry: name|db|ckpt_dir|as_of
set -u
cd /home/austin-whaley/Documents/wonderland

# name|db|ckpt_dir|as_of   (point ckpt_dir at wherever the loop trains for that stream)
STREAMS=(
"rabbit_hole|rabbit_hole/data/duckdb/customer_event_stream.parquet|/tmp/opencode/rh_jun|2025-06-01"
"instacart|rabbit_hole/data/instacart/customer_event_stream.parquet|/tmp/opencode/insta_noinpm|2025-10-01"
"ecommerce|rabbit_hole/data/ecommerce_2019/customer_event_stream.parquet|/tmp/opencode/v64|2020-03-01"
)

fail=0
for entry in "${STREAMS[@]}"; do
  IFS='|' read -r name db dir asof <<< "$entry"
  reg=$(ls -t "$dir"/registry_*.json 2>/dev/null | head -1)
  if [ -z "${reg:-}" ]; then echo "MISSING ckpt: $name ($dir)"; fail=1; continue; fi
  tag=$(python3 -c "import json,sys;print(json.load(open('$reg'))['tag'])")
  b=/tmp/opencode/loop_${name}_b.md; bc=/tmp/opencode/loop_${name}_bc.md
  python3 -m looking_glass.report_card --db "$db" --as-of "$asof" --ckpt-dir "$dir" --tag "$tag" \
    --stream "$name" --out "$b" >/dev/null 2>&1
  python3 -m looking_glass.head_battery --db "$db" --ckpt-dir "$dir" --tag "$tag" --as-of "$asof" \
    --horizon-days 90 --customers 25000 --seeds 0,1,2 --out "$bc" >/dev/null 2>&1
  vb=$(grep -m1 "VERDICT" "$b" 2>/dev/null); vbc=$(grep -m1 "VERDICT" "$bc" 2>/dev/null)
  echo "[$name] B: ${vb:-<none>} | B+C: ${vbc:-<none>}"
  if ! grep -q "VERDICT: GREEN" "$b" 2>/dev/null || ! grep -q "VERDICT: GREEN" "$bc" 2>/dev/null; then
    fail=1
  fi
done

if [ "$fail" = "0" ]; then echo "GOAL_MET"; exit 0; fi
echo "NOT_MET"
exit 1
