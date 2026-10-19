"""Assemble the BLOCK-WISE minimality receipt from the two real certificates.

The served E is deliberately block-wise (DEC-215): a learned shape block whose
rank is the smallest that preserves the future family, followed by the explicit
axis-aligned measure block that tree/k-means consumers need. The joint
unconstrained PCA minimum is recorded (usually smaller) but is NOT the served
form.

Inputs are produced by `scripts/minimality_certificate.py`:
  --shape  : `--targets future --block shape` on the SERVED E (shape rank minimal)
  --joint  : `--targets joint --block full` on the served E (joint minimum)
  --layer-b-json : the refit receipt (measure_cols, action_period_seconds)

This script is the single reproducible producer of the block-wise receipt; do
not hand-assemble it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--layer-b-json", required=True, dest="layer_b_json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    shape = json.loads(Path(a.shape).read_text())
    joint = json.loads(Path(a.joint).read_text())
    lb = json.loads(Path(a.layer_b_json).read_text())
    measure_cols = int(lb.get("measure_cols") or 0)
    shape_ok = bool(shape.get("joint_certified")) or (
        shape.get("measure_dim") is not None
        and shape.get("full_dim") is not None
        and int(shape["measure_dim"]) == int(shape["full_dim"])
    )
    served_dim = int(shape["measure_dim"]) + measure_cols
    rec = {
        "block_wise_certified": bool(shape_ok and measure_cols > 0),
        "joint_certified": bool(joint.get("joint_certified")),
        "measure_dim": served_dim,
        "full_dim": int(joint.get("full_dim") or served_dim),
        "served_dim": served_dim,
        "joint_min": int(joint.get("measure_dim") or served_dim),
        "shape_rank": int(shape["measure_dim"]),
        "measure_cols": measure_cols,
        "action_period_seconds": lb.get("action_period_seconds"),
        "rule": (
            "block-wise minimal for the SERVED accessible form: shape rank "
            f"{int(shape['measure_dim'])} (future family within noise) + measure "
            f"{measure_cols} (exact, axis-aligned). The unconstrained joint PCA "
            f"minimum is {int(joint.get('measure_dim') or served_dim)}; the "
            f"+{int(joint.get('full_dim') or served_dim) - int(joint.get('measure_dim') or served_dim)} "
            "is the explicit measure block required for tree/k-means accessibility (DEC-215)."
        ),
        "shape_certificate": {
            k: shape.get(k) for k in ("full_dim", "measure_dim", "joint_certified")
        },
        "joint_certificate": {
            k: joint.get(k) for k in ("full_dim", "measure_dim", "joint_certified")
        },
        "note": (
            f"Served block-wise form: shape rank {int(shape['measure_dim'])} + measure "
            f"{measure_cols} = {served_dim}. Joint unconstrained PCA minimum = "
            f"{int(joint.get('measure_dim') or served_dim)}."
        ),
    }
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if not k.endswith("certificate")}, indent=1))
    return rec


if __name__ == "__main__":
    main()
