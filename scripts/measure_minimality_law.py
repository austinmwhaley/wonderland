"""B5.1 minimality law: block-wise certificate on the SERVED compact embedding.

The frozen definition (DEC-215, `scripts/minimality_blocks.py`) is block-wise, not one
unconstrained PCA:

    shape rank  = the smallest rank that preserves the self-supervised future family
                  within the held-out noise floor (`--targets future --block shape`)
    measure     = the explicit 16-dim level block (kept for simple consumers)

The 0.11.1 checkpoint serves 256 shape dims; the certificate says the shape is
compressible (k_min=32). `scripts/build_compact_e.py` materialises the served object as
32 shape PCs + the 16-dim block. B5.1 then certifies THE SERVED OBJECT ITSELF:

    shape certificate on the stored compact shape (32 dims):
        full_dim == 32 and k_min == 32  -> the served shape is minimal at its full rank
    joint certificate on the full stored E (48 dims):
        joint_min recorded (the unconstrained PCA floor; the explicit measure block is
        the accessibility form, so the served form is shape_k_min + block_dim)

    block_wise_certified = (shape k_min == shape full_dim) and block_dim > 0

The gate is structural: served shape rank == k_min, with the measure block explicit.
Split A fits the compact transform; split B fits/scores the certificate — no B leakage
into the served embedding.

    python scripts/measure_minimality_law.py --products .../cfm_v2k_vf2_c32

Writes docs/report_cards/experiments/encoder_minimality.json (stamped); the two
certificate JSONs are written next to the products for provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cfm_products import _db_path, resolve_served_tag  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_minimality.json"
DEFAULT_PRODUCTS = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_c32"
DEFAULT_DEMO = ROOT / "looking_glass/artifacts/demo_v2"


def served_dims(products: Path) -> tuple[int, int]:
    """(served_dim, block_dim) actually stored in the products database.

    `block_dim` is the explicit non-shape part of the served E: the 16-dim level block plus
    any appended published view (e.g. the 4-dim response). It comes from the compact
    builder's receipt when present, so the certificate never guesses the composition.
    """
    import duckdb

    from looking_glass.stat_bank import LEVEL_COLS

    con = duckdb.connect(str(_db_path(str(products))), read_only=True)
    try:
        dims = [
            int(r[0]) for r in con.execute("select distinct dim from state_embeddings").fetchall()
        ]
    finally:
        con.close()
    if len(dims) != 1:
        raise SystemExit(f"state_embeddings carries {len(dims)} dims {dims}; expected exactly one")
    block_dim = len(LEVEL_COLS)
    receipt_path = products / "compact_e_receipt.json"
    if receipt_path.exists():
        tr = json.loads(receipt_path.read_text()).get("transform") or {}
        if tr.get("block_dim") is not None:
            block_dim = int(tr["block_dim"])
    if dims[0] <= block_dim:
        raise SystemExit(
            f"served dim {dims[0]} <= block dim {block_dim}; cannot isolate the shape block"
        )
    return dims[0], block_dim


def _certificate(
    certificate_main,
    *,
    products: Path,
    tag: str,
    as_of: str,
    demo: Path,
    customers: int,
    horizons: str,
    targets: str,
    block: str,
    shape_dim: int | None,
    out: Path,
):
    argv = [
        "--db",
        str(demo / "customer_event_stream.parquet"),
        "--ckpt-dir",
        str(products),
        "--tag",
        tag,
        "--as-of",
        str(as_of),
        "--customers",
        str(customers),
        "--horizons",
        str(horizons),
        "--targets",
        targets,
        "--block",
        block,
        "--stored-products",
        str(products),
        "--out",
        str(out),
    ]
    if shape_dim is not None:
        argv += ["--shape-dim", str(shape_dim)]
    return certificate_main(argv)


def _summary(cert: dict, path: Path) -> dict:
    return {
        "path": str(path),
        "rule": cert.get("rule"),
        "noise": cert.get("noise"),
        "targets": cert.get("targets"),
        "block": cert.get("block"),
        "n_targets": cert.get("n_targets"),
        "n_targets_dropped_uninformative": cert.get("n_targets_dropped_uninformative"),
        "n_train": cert.get("n_train"),
        "n_test": cert.get("n_test"),
        "worst_target": cert.get("worst_target"),
        "full_mean_r2": cert.get("full_mean_r2"),
        "full_worst_r2": cert.get("full_worst_r2"),
        "retained_mean_r2": cert.get("retained_mean_r2"),
        "retained_worst_r2": cert.get("retained_worst_r2"),
        "joint_certified": cert.get("joint_certified"),
        "curve": cert.get("curve"),
    }


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    ap.add_argument("--demo", default=str(DEFAULT_DEMO))
    ap.add_argument("--as-of", default=None, dest="as_of")
    ap.add_argument("--customers", type=int, default=3000)
    ap.add_argument("--horizons", default="30,90")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--shape-certificate-out", default=None, dest="shape_certificate_out")
    ap.add_argument("--joint-certificate-out", default=None, dest="joint_certificate_out")
    a = ap.parse_args()

    from minimality_certificate import main as certificate_main

    t0 = time.perf_counter()
    products = Path(a.products)
    demo = Path(a.demo)
    tag, reg_path = resolve_served_tag(str(products))
    meta = json.loads(reg_path.read_text())
    as_of = a.as_of or meta.get("as_of")
    if not as_of:
        raise SystemExit("no --as-of and the registry does not record one")
    served_dim, block_dim = served_dims(products)
    shape_dim = served_dim - block_dim
    compact_path = products / "compact_e_receipt.json"
    compact = json.loads(compact_path.read_text()) if compact_path.exists() else None
    if compact:
        tr = compact.get("transform") or {}
        if int(tr.get("served_dim", -1)) != served_dim:
            raise SystemExit(
                f"compact receipt served_dim {tr.get('served_dim')} != stored {served_dim}"
            )
        if int(tr.get("rank", -1)) != shape_dim:
            raise SystemExit(
                f"compact receipt rank {tr.get('rank')} != served shape dim {shape_dim}"
            )

    shape_out = (
        Path(a.shape_certificate_out)
        if a.shape_certificate_out
        else products / "minimality_shape.json"
    )
    joint_out = (
        Path(a.joint_certificate_out)
        if a.joint_certificate_out
        else products / "minimality_joint.json"
    )
    common = dict(
        products=products,
        tag=tag,
        as_of=str(as_of),
        demo=demo,
        customers=a.customers,
        horizons=a.horizons,
    )
    shape_cert = _certificate(
        certificate_main,
        **common,
        targets="future",
        block="shape",
        shape_dim=shape_dim,
        out=shape_out,
    )
    joint_cert = _certificate(
        certificate_main,
        **common,
        targets="joint",
        block="full",
        shape_dim=None,
        out=joint_out,
    )

    k_min = int(shape_cert["measure_dim"])
    full_dim = int(shape_cert["full_dim"])
    joint_min = int(joint_cert["measure_dim"])
    block_wise = bool(k_min == full_dim and block_dim > 0)
    certified = bool(block_wise and shape_dim == k_min)

    body = {
        "complete": True,
        "k_min": k_min,
        "full_dim": full_dim,
        "served_dim": int(served_dim),
        "served_shape_dim": int(shape_dim),
        "block_dim": int(block_dim),
        "certified": certified,
        "block_wise_certified": block_wise,
        "joint_certified": bool(joint_cert.get("joint_certified")),
        "joint_min": joint_min,
        "shape_certificate": _summary(shape_cert, shape_out),
        "joint_certificate": _summary(joint_cert, joint_out),
        "products": products_identity(str(products)),
        "source_products": (compact or {}).get("source"),
        "compact_receipt": compact,
        "why": (
            "Block-wise minimality on the SERVED object (DEC-215): the shape certificate "
            f"sweeps the stored {shape_dim}-dim shape block of the compact E and returns "
            f"k_min={k_min} with full_dim={full_dim}; the joint certificate on the full "
            f"{served_dim}-dim stored E returns the unconstrained joint floor "
            f"joint_min={joint_min}. The served form keeps the shape at its certified full "
            f"rank ({shape_dim} == {full_dim}) plus the explicit {block_dim}-dim level block, "
            f"so block_wise_certified={block_wise}. The compact transform is fit on split A; "
            f"the certificates are fit/held out inside split B (n_train "
            f"{shape_cert.get('n_train')} / n_test {shape_cert.get('n_test')}). The joint "
            f"floor being below the served dim is the accessibility cost of the explicit "
            f"measure block, not a B5.1 failure."
        ),
        "device": "cpu",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stamp_merge_write(Path(a.out), body, " ".join(sys.argv), stamp_blocks=False)
    print(
        json.dumps(
            {k: v for k, v in body.items() if k not in ("compact_receipt", "joint_certificate")},
            indent=1,
            default=str,
        )
    )
    print("WROTE", a.out)
    return 0 if certified else 1


if __name__ == "__main__":
    sys.exit(main())
