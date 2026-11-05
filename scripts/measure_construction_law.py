"""B1.2 construction laws C1-C6, stamped on the cfm_v2k products (DEC-585).

C1  trunk h under reversed field order (bar CONSTRUCTION_FIELD_PERM_BAR; the counter raw
    layout is recorded, not compared — h is SSM-only)
C2  scan associativity  (bar CONSTRUCTION_SCAN_BAR; reuses the S3 dynamics law)
C3  mass additivity     (bar CONSTRUCTION_MASS_BAR;  reuses the S3 dynamics law)
C4  two forwards bit-identical (sha256)
C5  renamed-schema end-to-end (reuses the input law; trains its own d=64 1-epoch demo)
C6  products identity + frozen checkpoint present

CPU producer: C1-C4 compare two forwards on the SAME device, so the device cancels; C2/C3
run float64 with a six-order margin over the S3-committed GPU evidence (1.128e-15 /
5.015e-16). Re-stamps docs/report_cards/experiments/encoder_lawfulness.json; the legacy
numbers are quarantined in DEC-585.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cfm_products import resolve_served_tag, stream_sequences  # noqa: E402
from measure_dynamics_law import check_mass_additivity, check_scan_associativity  # noqa: E402
from measure_input_law import check_renamed_end_to_end, products_identity, stamp_merge_write  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_lawfulness.json"


def _rel_l2(h1, h2) -> float:
    denom = max(float(np.linalg.norm(h1)), 1e-12)
    return float(np.linalg.norm(np.asarray(h2) - np.asarray(h1))) / denom


def c1_field_permutation(model, seqs) -> dict:
    """Reversed field order must leave the trunk h unchanged (commutative per-field sum)."""
    m2 = copy.deepcopy(model)
    m2.vocab.attrs = dict(reversed(list(model.vocab.attrs.items())))
    m2.attr_fields = tuple(reversed(model.attr_fields))
    m2.numeric_fields = tuple(reversed(model.numeric_fields))
    m2._attr_index = {f: i for i, f in enumerate(m2.attr_fields)}
    m2._num_index = {f: i for i, f in enumerate(m2.numeric_fields)}
    worst = 0.0
    with torch.no_grad():
        for seq in seqs:
            h1 = model(seq)[1].cpu().numpy()
            h2 = m2(seq)[1].cpu().numpy()
            worst = max(worst, _rel_l2(h1, h2))
    # the counter raw layout is recorded (one sequence sizes it), not compared
    from looking_glass.cfm_training import _collate

    t = _collate([seqs[0]], model.vocab, model._dev())
    raw1 = model.raw_fields(t)
    raw2 = m2.raw_fields(_collate([seqs[0]], m2.vocab, m2._dev()))
    return {
        "max_rel_err": worst,
        "raw_layout_m1": int(raw1.shape[-1]),
        "raw_layout_m2": int(raw2.shape[-1]),
        "evidence": (
            f"deepcopy trunk with vocab.attrs + attr_fields + numeric_fields reversed "
            f"(index caches remapped): worst per-customer relative L2 of h over {len(seqs)} "
            f"sequences {worst:.3e}; counter raw width {raw1.shape[-1]} (recorded, not "
            f"compared — h is SSM-only)"
        ),
    }


def c4_determinism(model, seqs) -> dict:
    def digest():
        h = hashlib.sha256()
        with torch.no_grad():
            for seq in seqs:
                h.update(model(seq)[1].cpu().numpy().astype(np.float32).tobytes())
        return h.hexdigest()

    d1, d2 = digest(), digest()
    return {
        "bit_identical": d1 == d2,
        "sha256": d1,
        "evidence": f"two full forwards over {len(seqs)} sequences: sha256 {d1[:12]}… "
        f"{'bit-identical' if d1 == d2 else 'DIFFER'}",
    }


def c6_frozen_identity(products, tag: str) -> dict:
    ident = products_identity(products)
    ckpt = Path(products) / f"cfm_{tag.replace('.', '_')}.pt"
    present = bool(ident.get("tag")) and ident.get("registry_sha256") and ckpt.exists()
    return {
        "present": present,
        "tag": ident.get("tag"),
        "registry_sha256": ident.get("registry_sha256"),
        "checkpoint_sha256": (
            hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16] if ckpt.exists() else None
        ),
        "evidence": f"products identity {ident} + checkpoint {ckpt.name} "
        f"{'present' if ckpt.exists() else 'MISSING'}",
    }


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--products",
        default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/cfm_v2k",
    )
    ap.add_argument(
        "--demo",
        default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/demo_v2",
    )
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-sequences", type=int, default=20)
    a = ap.parse_args()

    from cfm_products import load_products

    t0 = time.perf_counter()
    tag, _reg = resolve_served_tag(a.products)
    model, cfg, _meta = load_products(tag, a.products, cpu=True)
    _df, _keys, _split, seqs = stream_sequences(cfg, min_events=1)
    pick = seqs[: a.n_sequences]
    if len(pick) < a.n_sequences:
        print(f"[construction] only {len(pick)} sequences (wanted {a.n_sequences})", flush=True)

    c1 = c1_field_permutation(model, pick)
    c2 = check_scan_associativity(model, pick)
    c3 = check_mass_additivity(model, pick)
    c4 = c4_determinism(model, pick)
    print(
        f"[construction] C1 field-perm {c1['max_rel_err']:.3e} · C2 scan {c2['worst_rel_err']:.3e} · "
        f"C3 mass {c3['worst_rel_err']:.3e} · C4 {c4['bit_identical']}",
        flush=True,
    )
    c5 = check_renamed_end_to_end(Path(a.demo))
    c6 = c6_frozen_identity(a.products, tag)

    body = {
        # complete = the measurement RAN (every check was produced); pass/fail is the
        # card's job, from the values against the frozen bars (certification rule 3).
        "complete": bool(pick),
        "device": "cpu",
        "tag": tag,
        "n_sequences": len(pick),
        "customer_keys": [s["customer"] for s in pick],
        "products": products_identity(a.products),
        "checks": {
            "field_permutation": c1,
            "scan_associativity": c2,
            "mass_additivity": c3,
            "determinism": c4,
            "schema_variant": c5,
            "frozen_identity": c6,
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stamp_merge_write(Path(a.out), body, " ".join(sys.argv), stamp_blocks=False)
    print(json.dumps({k: v for k, v in body.items() if k != "checks"}, indent=1))
    print("WROTE", a.out)
    return 0 if body["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
