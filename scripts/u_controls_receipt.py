"""U power controls: attribute the sufficiency index instead of trusting one number.

The main U is measured against the independent sequence oracle. This receipt runs
the SAME oracle-gap machinery on four variants of the consumed embedding:

    E          = shape ‖ measure (the served object)
    measure    = the explicit, axis-aligned measure block only
    shape      = the learned shape block only
    destroyed  = E with customer rows shuffled (a no-information control)

If the destroyed variant has U≈0 and the measure-only variant matches E, then a
U≈1 is a real statement about the MEASURE being sufficient for the declared
future family — not a fabricated universal claim about the shape.

Usage:
  python3 scripts/u_controls_receipt.py --db <stream> --ckpt-dir <dir> \
      --tag <t> --as-of <date> --customers 6000 --horizons 30,90 --out <json>
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
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--horizons", default="30,90")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import polars as pl
    import torch

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.cfm_training import forward_embedding
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
    Z = np.concatenate(zs, 0) if zs else np.zeros((0, 1))
    names = list(getattr(getattr(model, "layer_b", None), "names", ()) or ())
    d = Z.shape[1] - len(names) if names and Z.shape[1] > len(names) else Z.shape[1]
    Xh, _ = _full_history_features(cfg, cust)
    Xh = np.asarray(Xh, float)
    rng = np.random.default_rng(a.seed)
    Zsh = Z.copy()
    rng.shuffle(Zsh, axis=0)
    variants = {"E": Z, "shape": Z[:, :d], "measure": Z[:, d:], "destroyed": Zsh}

    out: dict = {}
    for h in (int(x) for x in a.horizons.split(",")):
        fut, _past, _tc = _rfm_targets(cfg, h)
        j = (
            pl.DataFrame({"customer": cust})
            .join(fut, left_on="customer", right_on="customer_key", how="left")
            .fill_null(0.0)
        )
        for name, (yy, kind) in _future_family(j).items():
            key = f"{name} @ {h}d"
            for vname, V in variants.items():
                r = _oracle_gap(V, Xh, yy, kind, cust, seeds=(a.seed,), seqs=None, dev=None)
                if r:
                    out.setdefault(vname, []).append(
                        {
                            "target": key,
                            "e_alone": round(float(r[0]), 4),
                            "oracle": round(float(r[2]), 4),
                            "norm_gap": round(float(r[4]), 5),
                            "solved": bool(r[6]),
                            "informative": bool(r[7]),
                        }
                    )
    summary = {}
    for vname, rows in out.items():
        inf = [r for r in rows if r["informative"]]
        u = 1.0 - max((r["norm_gap"] for r in inf), default=0.0) if inf else 0.0
        summary[vname] = {
            "u": round(float(u), 4),
            "worst": max(inf, key=lambda r: r["norm_gap"]) if inf else None,
            "rows": rows,
        }
    Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: {"u": v["u"]} for k, v in summary.items()}, indent=1))
    return summary


if __name__ == "__main__":
    main()
