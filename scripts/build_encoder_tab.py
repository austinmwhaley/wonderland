"""Export the small payload for the Encoder tab.

Reads the frozen cache and receipts to produce:

  * one customer's 128-d frozen shape vector (for the feature map)
  * the published state's feature-family dimensions
  * encoder dimensions for the architecture flow

Writes `demo/encoder_tab.json`:

    python3 scripts/build_encoder_tab.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs/report_cards/experiments"
DEFAULT_CACHE = Path("/tmp/opencode/frozen_6k")
OUT = ROOT / "demo/encoder_tab.json"
DEFAULT_CUSTOMER = "rc0000208"

FAMILY_NOTES = {
    "shape": "learned sequence representation",
    "learned_state": "per-timescale recurrent state + learned mass",
    "forecast": "self-supervised past / future summaries",
    "uncertainty": "missingness, support and gap",
    "heads": "identity, next event and relational predictions",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--customer", default=DEFAULT_CUSTOMER)
    args = ap.parse_args()

    cache = Path(args.cache)
    npz = cache.with_suffix(".npz")
    if not npz.exists():
        print(f"missing frozen cache: {npz}", file=sys.stderr)
        raise SystemExit(1)

    z = np.load(npz, allow_pickle=True)
    E = np.asarray(z["E"], dtype=float)
    hold = list(z["hold"])
    if args.customer not in hold:
        print(f"customer {args.customer} not in frozen holdout", file=sys.stderr)
        raise SystemExit(1)
    shape = E[hold.index(args.customer)]

    runtime = json.loads((EXP / "runtime.json").read_text())
    cfg = json.loads((EXP / "encoder_config.json").read_text())
    data_receipt = json.loads((EXP / "data_receipt.json").read_text())

    state_dims = runtime.get("state_dims") or {}
    order = ["shape", "learned_state", "forecast", "uncertainty", "heads"]
    families = [
        {
            "id": name,
            "label": name.replace("_", " ").capitalize(),
            "dim": int(state_dims[name]),
            "note": FAMILY_NOTES[name],
        }
        for name in order
        if int(state_dims.get(name, 0)) > 0
    ]
    total = sum(f["dim"] for f in families)

    out = {
        "customer": args.customer,
        "shape": [round(float(v), 4) for v in shape],
        "shape_norm": round(float(np.linalg.norm(shape)), 3),
        "population_mean_norm": round(float(np.linalg.norm(E, axis=1).mean()), 3),
        "families": families,
        "state_dim": total,
        "trunk": {
            "width": int(cfg["d"]),
            "layers": int(cfg["layers"]),
            "experts": int(cfg["e"]),
            "read_heads": int(cfg["n_read"]),
            "measure_dim": int(cfg["measure_dim"]),
        },
        "event_types": len((data_receipt.get("path_b") or {}).get("event_types") or {}),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(
        f"WROTE {args.out} | {args.customer} shape {len(out['shape'])} | "
        f"state {total} | families {len(families)}"
    )


if __name__ == "__main__":
    main()
