"""Export the small payload for the Encoder tab.

Reads the frozen cache and receipts to produce:

  * a shortlist of customers' frozen shape and full state-family vectors
  * the published state's feature-family dimensions
  * encoder dimensions for the feature explorer

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

    runtime = json.loads((EXP / "runtime.json").read_text())
    cfg = json.loads((EXP / "encoder_config.json").read_text())
    data_receipt = json.loads((EXP / "data_receipt.json").read_text())
    manifest = json.loads(Path(str(cache) + ".manifest.json").read_text())
    layout = manifest.get("layout") or {}
    blocks = {b["name"]: b for b in layout.get("blocks", [])}
    blk = np.asarray(z["BLK"], dtype=float)
    state_dims = runtime.get("state_dims") or {}
    order = ["shape", "learned_state", "forecast", "uncertainty", "heads"]

    truth = {}
    truth_path = Path(cfg.get("data", "")) / "reference_truth.json"
    if truth_path.exists():
        truth = {c["customer"]: c for c in json.loads(truth_path.read_text()).get("customers", [])}

    def families_for(key):
        if key not in hold:
            print(f"customer {key} not in frozen holdout", file=sys.stderr)
            raise SystemExit(1)
        row = hold.index(key)
        shape = E[row]
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
                    blk[row, blocks[b]["start"] : blocks[b]["stop"]]
                    for b in view.get("blocks", [])
                    if b in blocks
                ]
                if not parts:
                    continue
                values = np.concatenate(parts)
            if len(values) != dim:
                print(
                    f"family {name}: value count {len(values)} != state dim {dim}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            families.append(
                {
                    "id": name,
                    "label": name.replace("_", " ").capitalize(),
                    "dim": dim,
                    "values": [round(float(v), 4) for v in values],
                }
            )
        return shape, families

    shortlist = [
        "rc0000208",
        "rc0004902",
        "rc0000003",
        "rc0000863",
        "rc0004162",
    ]
    if args.customer not in shortlist:
        shortlist.insert(0, args.customer)

    customers = []
    for key in shortlist:
        shape, families = families_for(key)
        t = truth.get(key, {})
        customers.append(
            {
                "key": key,
                "channel": t.get("channel_pref"),
                "category": t.get("category_pref"),
                "regime": t.get("regime0"),
                "shape": [round(float(v), 4) for v in shape],
                "families": families,
            }
        )

    first = customers[0]
    total = sum(f["dim"] for f in first["families"])
    out = {
        "customer": first["key"],
        "customers": customers,
        "shape": first["shape"],
        "shape_norm": round(float(np.linalg.norm(E[hold.index(first["key"])])), 3),
        "population_mean_norm": round(float(np.linalg.norm(E, axis=1).mean()), 3),
        "families": first["families"],
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
        f"WROTE {args.out} | {len(customers)} customers | "
        f"{first['key']} shape {len(first['shape'])} | state {total}"
    )


if __name__ == "__main__":
    main()
