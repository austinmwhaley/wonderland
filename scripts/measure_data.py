"""Data-layer receipt for the demo card.

Reads the generated reference and writes docs/report_cards/experiments/data_receipt.json with the
two checks the demo claims:
  1. PATH A passed its integrity gate (join/recompute/labels/action levels).
  2. PATH B is the canonical 4-column stream, sorted, with all 5 randomized
     action keys and >=2 observed levels per key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--out", default=str(ROOT / "docs/report_cards/experiments/data_receipt.json"))
    args = ap.parse_args()
    d = Path(args.data)
    pa = json.loads((d / "path_a_validation.json").read_text())
    man = json.loads((d / "manifest.json").read_text())
    truth = json.loads((d / "reference_truth.json").read_text())
    meta = truth.get("meta") or {}
    B = pl.read_parquet(d / "customer_event_stream.parquet")
    cols = list(B.columns)
    sorted_key = bool(B["customer_key"].is_sorted())
    per_cust_sorted = bool(
        B.group_by("customer_key", maintain_order=True)
        .agg(pl.col("event_ts").is_sorted().all())
        .select(pl.col("event_ts").all())
        .item()
    )
    levels = pa.get("action_levels", {})
    rec = {
        "data": str(d),
        "as_of": man["as_of"],
        "seed": (meta.get("spec") or {}).get("seed"),
        "labels": pa.get("labels", []),
        "action_design": meta.get("action", {}),
        "path_a": {
            "ok": bool(pa.get("ok")),
            "failures": pa.get("failures", []),
            "join_ok": bool(pa.get("join_ok")),
            "recompute_ok": pa.get("recompute_ok", {}),
            "label_balance": pa.get("label_balance", {}),
            "temporal_split": pa.get("temporal_split", {}),
        },
        "path_b": {
            "columns": cols,
            "canonical_4": cols == ["customer_key", "event_ts", "event_type", "event_attributes"],
            "rows": int(B.height),
            "customers": int(B["customer_key"].n_unique()),
            "sorted_key": sorted_key,
            "per_customer_sorted": per_cust_sorted,
            "event_types": {
                r["event_type"]: int(r["len"])
                for r in B.group_by("event_type").len().iter_rows(named=True)
            },
        },
        "actions": {
            "keys": sorted(levels),
            "all_five": sorted(levels) == ["channel", "dow", "freq", "hour", "offer"],
            "levels": levels,
            "all_keys_have_2plus": all(len(v) >= 2 for v in levels.values()),
        },
    }
    rec["path_a"]["pass"] = (
        rec["path_a"]["ok"] and not rec["path_a"]["failures"] and rec["path_a"]["join_ok"]
    )
    rec["path_b"]["pass"] = (
        rec["path_b"]["canonical_4"]
        and rec["path_b"]["sorted_key"]
        and rec["path_b"]["per_customer_sorted"]
        and rec["path_b"]["rows"] > 0
        and rec["path_b"]["customers"] > 0
    )
    rec["actions"]["pass"] = rec["actions"]["all_five"] and rec["actions"]["all_keys_have_2plus"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k]["pass"] for k in ("path_a", "path_b", "actions")}, indent=1))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
