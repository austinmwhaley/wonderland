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
    manifest = json.loads(Path(str(cache) + ".manifest.json").read_text())
    layout = manifest.get("layout") or {}
    blocks = {b["name"]: b for b in layout.get("blocks", [])}
    blk = np.asarray(z["BLK"], dtype=float)

    state_dims = runtime.get("state_dims") or {}
    order = ["shape", "learned_state", "forecast", "uncertainty", "heads"]
    families = []
    for name in order:
        dim = int(state_dims.get(name, 0))
        if dim <= 0:
            continue
        if name == "shape":
            values = shape
        else:
            view = (layout.get("views") or {}).get(name) or {}
            parts = [
                blk[hold.index(args.customer), blocks[b]["start"] : blocks[b]["stop"]]
                for b in view.get("blocks", [])
                if b in blocks
            ]
            if not parts:
                continue
            values = np.concatenate(parts)
        if len(values) != dim:
            print(f"family {name}: value count {len(values)} != state dim {dim}", file=sys.stderr)
            raise SystemExit(1)
        families.append(
            {
                "id": name,
                "label": name.replace("_", " ").capitalize(),
                "dim": dim,
                "values": [round(float(v), 4) for v in values],
            }
        )
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
