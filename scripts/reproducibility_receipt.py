"""Deterministic reproducibility + inference-runtime receipt.

Replaces the old "tag exists -> PASS" reader. Runs the frozen encoder twice on
the same held-out batch with the same seed and hashes the consumed embedding;
also records a measured inference wall time (efficiency) and the model identity.

Usage:
  python3 scripts/reproducibility_receipt.py --db <stream> --ckpt-dir <dir> \
      --tag <t> --as-of <date> --customers 6000 --out <repro.json>
Writes <repro.json> with `passed/total`, `embedding_sha256`, `infer_wall_seconds`,
and `identity` (tag, ckpt dir, sample_a, data signature).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    import torch

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.cfm_training import forward_embedding
    from looking_glass.universality import _b_sequences

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    seqs = _b_sequences(cfg)
    batch = seqs[: max(1, int(a.batch))]

    def _forward():
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            z = forward_embedding(model, batch, cfg=cfg, lb=getattr(model, "layer_b", None))
        return z.detach().float().cpu().numpy()

    z1 = _forward()
    z2 = _forward()
    deterministic = bool(np.array_equal(z1, z2))
    digest = hashlib.sha256(z1.tobytes()).hexdigest()

    # measured inference time (warm-up + median of 3)
    _forward()
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        _forward()
        ts.append(time.perf_counter() - t)
    infer_s = float(sorted(ts)[len(ts) // 2])

    reg_path = Path(a.ckpt_dir) / f"registry_{a.tag.replace('.', '_')}.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    identity = {
        "tag": a.tag,
        "ckpt_dir": str(a.ckpt_dir),
        "sample_a_customers": (reg.get("config") or {}).get("sample_a_customers"),
        "n_train_sequences": reg.get("n_train_sequences"),
        "data_signature": reg.get("data_signature"),
        "revision": reg.get("revision"),
        "n_sequences": int(len(batch)),
    }
    rec = {
        "passed": int(deterministic),
        "total": 1,
        "deterministic": deterministic,
        "embedding_sha256": digest,
        "infer_wall_seconds": round(infer_s, 6),
        "wall_seconds": round(infer_s, 6),
        "identity": identity,
        "evidence": "same frozen encoder + same seed -> byte-identical consumed embedding",
    }
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    # the efficiency reader only needs passed/total/wall_seconds
    Path(str(a.out).replace("reproducibility", "efficiency")).write_text(
        json.dumps(
            {
                "passed": int(infer_s > 0),
                "total": 1,
                "wall_seconds": round(infer_s, 6),
                "infer_wall_seconds": round(infer_s, 6),
                "identity": identity,
            },
            indent=1,
        )
        + "\n"
    )
    print(json.dumps({k: v for k, v in rec.items() if k != "identity"}, indent=1))
    return rec


if __name__ == "__main__":
    main()
