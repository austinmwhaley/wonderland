"""Attribute shape-U and full-E geometry to the blocks (F2/F3 of the review).

The served shape may contain deterministic schema-summary axes. This receipt reports:
  * learned donor only        (the first model.dim columns of the unserved E)
  * learned donor + measure
  * deterministic summary + measure
  * served full E
against the same full-history oracle. It also reports the full-E lookalike without
the duplicated measure in the shape. No labels; same held-out split as the card.
"""

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
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--horizons", default="30,90")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import polars as pl
    import torch

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.cfm_training import forward_embedding
    from looking_glass.unsupervised_gate import _run_unsupervised
    from looking_glass.universality import (
        _b_sequences,
        _full_history_features,
        _future_family,
        _oracle_gap,
        _rfm_targets,
    )

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    was = getattr(model, "serve_on", False)
    model.serve_on = False  # unserved source: donor | counter | summary | measure
    seqs = _b_sequences(cfg)
    cust = np.array([s["customer"] for s in seqs])
    tmax = max((len(s["event_type"]) for s in seqs), default=1)
    bs = max(16, min(512, 131072 // max(tmax, 1)))
    zs = []
    for i in range(0, len(seqs), bs):
        with torch.no_grad():
            z = forward_embedding(
                model, seqs[i : i + bs], cfg=cfg, lb=getattr(model, "layer_b", None)
            )
        zs.append(z.detach().float().cpu().numpy())
    Z = np.concatenate(zs, 0)
    model.serve_on = was
    names = list(getattr(getattr(model, "layer_b", None), "names", ()) or ())
    d = Z.shape[1] - len(names)
    donor_dim = int(model.dim)
    variants = {
        "learned_donor_only": Z[:, :donor_dim],
        "learned_donor_plus_measure": np.hstack([Z[:, :donor_dim], Z[:, d:]]),
        "deterministic_shape_plus_measure": np.hstack([Z[:, donor_dim:d], Z[:, d:]]),
        "served_full_E": Z,
    }
    Xh, _ = _full_history_features(cfg, cust)
    Xh = np.asarray(Xh, float)
    out = {}
    for h in (int(x) for x in a.horizons.split(",")):
        fut, _past, _tc = _rfm_targets(cfg, h)
        j = (
            pl.DataFrame({"customer": cust})
            .join(fut, left_on="customer", right_on="customer_key", how="left")
            .fill_null(0.0)
        )
        for name, (yy, kind) in _future_family(j).items():
            for vname, V in variants.items():
                r = _oracle_gap(V, Xh, yy, kind, cust, seeds=(a.seed,), seqs=None, dev=None)
                if r:
                    out.setdefault(vname, []).append(
                        {
                            "target": f"{name} @ {h}d",
                            "e_alone": round(float(r[0]), 4),
                            "oracle": round(float(r[2]), 4),
                            "norm_gap": round(float(r[4]), 5),
                        }
                    )
    summary = {}
    for v, rows in out.items():
        u = 1.0 - max(r["norm_gap"] for r in rows)
        summary[v] = {"u": round(float(u), 4), "worst": max(rows, key=lambda r: r["norm_gap"])}
    # full-E geometry WITHOUT the summary in the shape (donor + measure only).
    from looking_glass.layer_b import generic_measure

    G, _gn = generic_measure(cfg, cust)
    G = np.asarray(G, float)
    full_nodup = np.hstack([Z[:, :donor_dim], Z[:, d:]])
    geo = {}
    for label, states in (("donor_plus_measure", full_nodup), ("generic", G)):
        r = _run_unsupervised(model, cfg, states=(cust, states), block="full", seed=a.seed)
        geo[label] = {
            k: r.get(k) for k in ("silhouette", "eta_squared", "lookalike_enrichment", "anova_f")
        }
    geo["ratio_donor_plus_measure"] = round(
        float(
            geo["donor_plus_measure"]["lookalike_enrichment"]
            / geo["generic"]["lookalike_enrichment"]
        ),
        4,
    )
    rec = {"tag": a.tag, "customers": a.customers, "shape_u": summary, "geometry": geo}
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    raise SystemExit(main())
