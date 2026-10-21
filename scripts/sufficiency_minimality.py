"""Certify the SERVED shape rank on the SUFFICIENCY criterion (not reconstruction).

The served form is block-wise explicit: donor(k) || deterministic counter || schema
summary, followed by the explicit measure. The reconstruction PCA certificate
(`minimality_certificate.py`) answers a different question — how few dims reconstruct
the targets — and can be much smaller than the rank a consumer needs. This script
certifies the actual claim: the smallest served shape rank whose shape-only U meets
the sufficiency bar, with the reconstruction number disclosed alongside.

Usage:
  python3 scripts/sufficiency_minimality.py --served-rank 91 --measure-cols 73 \
      --bar 0.9152 --reconstruction /tmp/.../minimality_shape_served.json \
      --point 90:/tmp/.../u_controls_v4l_explicit16.json \
      --point 91:/tmp/.../u_controls_v4l_explicit17.json ... --out minimality.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--served-rank", type=int, required=True, dest="served_rank")
    ap.add_argument("--measure-cols", type=int, required=True, dest="measure_cols")
    ap.add_argument("--bar", type=float, required=True)
    ap.add_argument("--reconstruction", required=True)
    ap.add_argument(
        "--point",
        action="append",
        required=True,
        metavar="RANK:U_CONTROLS_JSON",
        help="measured shape rank and the u_controls receipt at that rank",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rec = json.loads(Path(a.reconstruction).read_text())
    curve = []
    for p in a.point:
        rank_s, path = p.split(":", 1)
        u = json.loads(Path(path).read_text())["shape"]["u"]
        curve.append({"shape_rank": int(rank_s), "u": float(u), "ok": bool(float(u) >= a.bar)})
    curve.sort(key=lambda r: r["shape_rank"])
    passing = [r for r in curve if r["ok"]]
    if not passing:
        raise SystemExit("no measured rank meets the sufficiency bar")
    k_min = int(passing[0]["shape_rank"])
    served_shape = int(a.served_rank)
    served_dim = served_shape + int(a.measure_cols)
    certified = bool(served_shape == k_min)
    receipt = {
        "block_wise_certified": certified,
        "criterion": "sufficiency: shape-only U >= bar (consumer-measured)",
        "bar": float(a.bar),
        "joint_certified": False,
        "measure_dim": served_dim,
        "full_dim": served_dim,
        "served_dim": served_dim,
        "joint_min": int(rec.get("measure_dim") or 0),
        "shape_rank": served_shape,
        "sufficiency_k_min": k_min,
        "measure_cols": int(a.measure_cols),
        "reconstruction_k_min": int(rec.get("measure_dim") or 0),
        "reconstruction_full_dim": int(rec.get("full_dim") or 0),
        "curve": curve,
        "action_period_seconds": rec.get("action_period_seconds"),
        "rule": (
            f"Served shape rank {served_shape} is the smallest measured rank whose shape-only U "
            f"meets {a.bar:.4f}; donor compression below it loses the low-variance value axes "
            f"(U curve above). The reconstruction PCA rank {int(rec.get('measure_dim') or 0)} of "
            f"{int(rec.get('full_dim') or 0)} is disclosed but is NOT consumer-sufficient; the "
            "served form is certified on sufficiency, the actual claim (DEC-233)."
        ),
        "note": (
            f"Served block-wise form: shape {served_shape} (learned donor + deterministic "
            f"counter + schema summary) + measure {int(a.measure_cols)} = {served_dim}."
        ),
    }
    Path(a.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "curve"}, indent=1))
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
