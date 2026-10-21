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
    ap.add_argument("--period-seconds", type=float, default=None, dest="period_seconds")
    ap.add_argument("--receipt", default=None)
    ap.add_argument("--mode", choices=("pca", "blocks", "split", "explicit"), default="pca")
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
    if getattr(a, "period_seconds", None) is not None:
        cfg.action_period_seconds = float(a.period_seconds)
    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    a_keys = [k for k in keys if split.get(k) == "A"]
    seqs = build_sequences(df, a_keys, cfg, split, with_anchors=False)
    # Fit on the RAW (pre-serve) embedding: disable any existing serve projection.
    _was = getattr(model, "serve_on", False)
    model.serve_on = False
    zs = []
    from looking_glass.cfm_training import embed_batch_size

    _bs = embed_batch_size(cfg)
    for i in range(0, len(seqs), _bs):
        with torch.no_grad():
            z = forward_embedding(
                model, seqs[i : i + _bs], cfg=cfg, lb=getattr(model, "layer_b", None)
            )
        zs.append(z.detach().float().cpu().numpy())
    model.serve_on = _was
    Z = np.concatenate(zs, 0) if zs else np.zeros((0, 1))
    if Z.shape[0] < 10:
        raise SystemExit("too few split-A embeddings to fit the projection")
    if a.mode == "explicit":
        # EXPLICIT-SUMMARY serve: compress only the learned donor to k PCs; keep the
        # deterministic counter + schema summary (and the measure) axis-aligned.
        _names = list(getattr(getattr(model, "layer_b", None), "names", ()) or ())
        d = int(Z.shape[1] - len(_names)) if _names else int(model.dim)
        dd = int(model.dim)
        k = int(min(a.k, dd))
        scl = StandardScaler().fit(Z[:, :dd])
        pca = PCA(n_components=k, random_state=0).fit(scl.transform(Z[:, :dd])) if k >= 1 else None
        W = pca.components_[:k] if pca is not None else np.zeros((0, dd), dtype=np.float32)
        np.savez(
            a.out,
            mode="explicit",
            shape_dim=d,
            donor_dim=dd,
            total_dim=int(Z.shape[1]),
            mean_d=scl.mean_,
            scale_d=scl.scale_,
            W_d=W,
        )
        served = k + (Z.shape[1] - dd)
        rec = {
            "tag": a.tag,
            "mode": "explicit",
            "source_dim": int(Z.shape[1]),
            "shape_dim": d,
            "donor_dim": dd,
            "donor_rank": int(k),
            "explicit_shape_dims": int(d - dd),
            "measure_dim": int(Z.shape[1] - d),
            "served_dim": int(served),
            "n_train": int(Z.shape[0]),
            "out": str(a.out),
        }
    elif a.mode == "split":
        # MASS-AWARE serve: the counter's many correlated mass axes can crowd the
        # timing/dynamics axes out of a single shape PCA. Compress the mass block to
        # the smallest number of PCs explaining 95% of its variance (data-derived),
        # and give the remaining rank to the donor state.
        _names = list(getattr(getattr(model, "layer_b", None), "names", ()) or ())
        d = int(Z.shape[1] - len(_names)) if _names else int(model.dim)
        _c = int(getattr(model, "counter_raw_dim", 0) or 0)
        if _c <= 0 or _c >= d:
            raise SystemExit(
                "split serve requires an enabled counter with 0 < counter_dim < shape_dim"
            )
        k = int(min(a.k, d))
        Zd, Zm = Z[:, : d - _c], Z[:, d - _c : d]
        scl_m = StandardScaler().fit(Zm)
        _mm = min(_c, k - 1, Zm.shape[0] - 1)
        _pm = PCA(n_components=_mm, random_state=0).fit(scl_m.transform(Zm))
        _cum = np.cumsum(_pm.explained_variance_ratio_)
        n_m = int(np.searchsorted(_cum, 0.95) + 1)
        n_m = max(1, min(n_m, k - 1))
        n_d = k - n_m
        pca_m = PCA(n_components=n_m, random_state=0).fit(scl_m.transform(Zm))
        scl_d = StandardScaler().fit(Zd)
        pca_d = PCA(n_components=n_d, random_state=0).fit(scl_d.transform(Zd))
        np.savez(
            a.out,
            mode="split",
            shape_dim=d,
            counter_dim=_c,
            total_dim=int(Z.shape[1]),
            mean_d=scl_d.mean_,
            scale_d=scl_d.scale_,
            W_d=pca_d.components_[:n_d],
            mean_m=scl_m.mean_,
            scale_m=scl_m.scale_,
            W_m=pca_m.components_[:n_m],
        )
        served = k + (Z.shape[1] - d)
        rec = {
            "tag": a.tag,
            "mode": "split",
            "source_dim": int(Z.shape[1]),
            "shape_dim": d,
            "counter_dim": _c,
            "donor_rank": int(n_d),
            "mass_rank": int(n_m),
            "mass_var_explained": float(_cum[n_m - 1]),
            "shape_rank": int(k),
            "measure_dim": int(Z.shape[1] - d),
            "served_dim": int(served),
            "n_train": int(Z.shape[0]),
            "out": str(a.out),
        }
    elif a.mode == "blocks":
        # The SHAPE block is everything before the Layer-B measure (donor + counter
        # + accumulator when enabled), not just the donor dim.
        _names = list(getattr(getattr(model, "layer_b", None), "names", ()) or ())
        d = int(Z.shape[1] - len(_names)) if _names else int(model.dim)
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
