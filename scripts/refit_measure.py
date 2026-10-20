"""Refit the Layer-B measure block to be AXIS-ALIGNED (no PCA rotation) and
re-fit the block-energy scalar. DEC-215 accessibility fix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--period-seconds", type=float, default=None, dest="period_seconds")
    a = ap.parse_args(argv)

    import torch

    from looking_glass.cfm_data import _customer_keys, _read_stream, assign_split, build_sequences
    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.cfm_training import embed_batch_size, forward_embedding
    from looking_glass.layer_b import generic_measure

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    lb = getattr(model, "layer_b", None)
    if lb is None:
        raise SystemExit("no LayerBEmbedder")
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    if getattr(a, "period_seconds", None) is not None:
        cfg.action_period_seconds = float(a.period_seconds)
    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    a_keys = [k for k in keys if split.get(k) == "A"]
    m, names = generic_measure(cfg, a_keys)
    lb.fit(m, names)
    # reset the block scalar to 1.0 BEFORE measuring the measure block, otherwise
    # the old scalar inflates v_meas and the new alpha is wrong.
    lb.block_alpha = 1.0
    # re-fit block energy: shape total variance / measure total variance
    seqs = build_sequences(df, a_keys, cfg, split, with_anchors=False)
    was = getattr(model, "serve_on", False)
    model.serve_on = False
    zs = []
    _bs = embed_batch_size(cfg)
    for i in range(0, len(seqs), _bs):
        with torch.no_grad():
            z = forward_embedding(model, seqs[i : i + _bs], cfg=cfg, lb=model.layer_b)
        zs.append(z.detach().float().cpu().numpy())
    model.serve_on = was
    Z = np.concatenate(zs, 0) if zs else np.zeros((0, 1))
    # shape block = everything before the Layer-B measure (donor + counter + ...)
    _names = list(getattr(lb, "names", ()) or ())
    d = int(Z.shape[1] - len(_names)) if _names else int(model.dim)
    v_shape = float(np.var(Z[:, :d], axis=0).sum())
    v_meas = float(np.var(Z[:, d:], axis=0).sum())
    lb.block_alpha = float(np.sqrt(max(v_shape, 1e-12) / max(v_meas, 1e-12)))
    # persist the response cadence so serve rebuilds the same response block
    lb.action_period_seconds = (
        None if getattr(a, "period_seconds", None) is None else float(a.period_seconds)
    )
    lb.save(a.out)
    rec = {
        "tag": a.tag,
        "measure_cols": int(m.shape[1]),
        "pca": "none (axis-aligned)" if lb.pca is None else f"k={lb.pca.n_components_}",
        "block_alpha": round(lb.block_alpha, 6),
        "action_period_seconds": lb.action_period_seconds,
        "var_shape": round(v_shape, 4),
        "var_measure": round(v_meas, 4),
        "out": str(a.out),
    }
    Path(str(a.out) + ".json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    main()
