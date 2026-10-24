"""Iteration-2 measurement: construction receipts (determinism + schema-agnosticism).

Writes docs/report_cards/experiments/encoder_construction.json
  determinism.identical  : two seed-0 runs produce bit-identical E + best-val
  schema_variant.pass    : E trains on the renamed schema + new field, zero code change
"""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/austin-whaley/Documents/wonderland")
sys.path.insert(0, "/home/austin-whaley/Documents/wonderland/scripts/demo")
import auto_encoder as AE  # noqa: E402

DATA = "/tmp/opencode/t0_loop/relational_v3"
SCHEMA = "/tmp/opencode/t0_loop/relational_v3_schema"
OUT = Path("/home/austin-whaley/Documents/wonderland/docs/report_cards/experiments/encoder_construction.json")

E1, _, _, _, _, _, _, _, best1, _ = AE.train_encoder(DATA, seed=0, epochs=12, d=128, layers=2)
print("[det] run1 done", flush=True)
E2, _, _, _, _, _, _, _, best2, _ = AE.train_encoder(DATA, seed=0, epochs=12, d=128, layers=2)
print("[det] run2 done", flush=True)
identical = bool(np.array_equal(np.nan_to_num(E1), np.nan_to_num(E2)) and float(best1) == float(best2))

schema_ok = False
try:
    Es, _, _, _, _, _, vss, _, bests, _ = AE.train_encoder(SCHEMA, seed=0, epochs=12, d=128, layers=2)
    schema_ok = Es.shape[0] > 0 and np.isfinite(bests)
    print("[schema] trained", Es.shape, "best", float(bests), flush=True)
except Exception as e:  # noqa: BLE001
    print("[schema] FAILED", e, flush=True)

OUT.write_text(json.dumps({
    "determinism": {"identical": identical},
    "schema_variant": {"pass": bool(schema_ok)},
}, indent=1))
print("WROTE", OUT)
