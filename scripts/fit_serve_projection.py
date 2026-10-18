"""Fit and freeze the certified minimal serve projection (DEC-213).

Fits StandardScaler + PCA(k) on split-A embeddings ONLY (the rank k was chosen on
held-out split B by `scripts/minimality_certificate.py`), and writes
`serve_proj_<tag>.npz` next to the checkpoint. `load_frozen_encoder` then applies
E -> minimal rank at serve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit the minimal serve projection")
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--receipt", default=None)
    ap.add_argument("--mode", choices=("pca", "blocks"), default="pca")
    a = ap.parse_args(argv)

    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from looking_glass.cfm_data import _customer_keys, _read_stream, assign_split, build_sequences
    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.cfm_training import forward_embedding

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    a_keys = [k for k in keys if split.get(k) == "A"]
    seqs = build_sequences(df, a_keys, cfg, split, with_anchors=False)
    # Fit on the RAW (pre-serve) embedding: disable any existing serve projection.
    _was = getattr(model, "serve_on", False)
    model.serve_on = False
    zs = []
    for i in range(0, len(seqs), 256):
        with torch.no_grad():
            z = forward_embedding(
                model, seqs[i : i + 256], cfg=cfg, lb=getattr(model, "layer_b", None)
            )
        zs.append(z.detach().float().cpu().numpy())
    model.serve_on = _was
    Z = np.concatenate(zs, 0) if zs else np.zeros((0, 1))
    if Z.shape[0] < 10:
        raise SystemExit("too few split-A embeddings to fit the projection")
    if a.mode == "blocks":
        d = int(model.dim)
        Zb = Z[:, :d]
        scl = StandardScaler().fit(Zb)
        k = int(min(a.k, Zb.shape[1]))
        pca = PCA(n_components=k, random_state=0).fit(scl.transform(Zb))
        np.savez(
            a.out,
            mean=scl.mean_,
            scale=scl.scale_,
            W=pca.components_[:k],
            mode="blocks",
            shape_dim=d,
            total_dim=int(Z.shape[1]),
        )
        served = k + (Z.shape[1] - d)
        rec = {
            "tag": a.tag,
            "mode": "blocks",
            "source_dim": int(Z.shape[1]),
            "shape_dim": d,
            "measure_dim": int(Z.shape[1] - d),
            "shape_rank": k,
            "served_dim": int(served),
            "n_train": int(Z.shape[0]),
            "explained_variance_ratio": round(float(pca.explained_variance_ratio_.sum()), 6),
            "out": str(a.out),
        }
    else:
        scl = StandardScaler().fit(Z)
        k = int(min(a.k, Z.shape[1]))
        pca = PCA(n_components=k, random_state=0).fit(scl.transform(Z))
        np.savez(a.out, mean=scl.mean_, scale=scl.scale_, W=pca.components_[:k], mode="pca")
        rec = {
            "tag": a.tag,
            "mode": "pca",
            "source_dim": int(Z.shape[1]),
            "k": k,
            "n_train": int(Z.shape[0]),
            "explained_variance_ratio": round(float(pca.explained_variance_ratio_.sum()), 6),
            "out": str(a.out),
        }
    if a.receipt:
        Path(a.receipt).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    main()
