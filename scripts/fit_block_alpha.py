"""Fit the frozen block-energy-parity scalar for the measure block (DEC-213).

alpha = sqrt(total variance of the shape block / total variance of the measure
block), fitted on split-A embeddings. A single scalar per block — NOT per-direction
whitening. Saved into the LayerBEmbedder, so `forward_embedding` applies it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit block-energy parity for the measure")
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch

    from looking_glass.cfm_data import _customer_keys, _read_stream, assign_split, build_sequences
    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.cfm_training import forward_embedding

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    lb = getattr(model, "layer_b", None)
    if lb is None:
        raise SystemExit("no LayerBEmbedder on this checkpoint")
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    a_keys = [k for k in keys if split.get(k) == "A"]
    seqs = build_sequences(df, a_keys, cfg, split, with_anchors=False)
    zs = []
    for i in range(0, len(seqs), 256):
        with torch.no_grad():
            z = forward_embedding(
                model, seqs[i : i + 256], cfg=cfg, lb=getattr(model, "layer_b", None)
            )
        zs.append(z.detach().float().cpu().numpy())
    Z = np.concatenate(zs, 0) if zs else np.zeros((0, 1))
    d_shape = int(model.dim)
    if Z.shape[1] <= d_shape:
        raise SystemExit("no measure block in E")
    v_shape = float(np.var(Z[:, :d_shape], axis=0).sum())
    v_meas = float(np.var(Z[:, d_shape:], axis=0).sum())
    alpha = float(np.sqrt(max(v_shape, 1e-12) / max(v_meas, 1e-12)))
    lb.block_alpha = alpha
    lb.save(a.out)
    rec = {
        "tag": a.tag,
        "shape_dim": d_shape,
        "measure_dim": int(Z.shape[1] - d_shape),
        "var_shape": round(v_shape, 6),
        "var_measure": round(v_meas, 6),
        "alpha": round(alpha, 6),
        "out": str(a.out),
    }
    Path(str(a.out) + ".json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    main()
