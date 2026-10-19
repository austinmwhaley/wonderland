"""Ablation ladder: which component of the served E actually carries the signal?

Same held-out customers, same GBM head, same folds for every variant:

    E        = full served embedding (shape ‖ measure)
    shape    = the learned shape block only
    measure  = the explicit axis-aligned measure block only
    raw      = model-free RFM/recency features (production baseline)
    generic  = the exhaustive generic measure (same information as `measure`)

The ladder proves which block contributes (and which targets are measure-carried
ties rather than shape wins). It is a DIAGNOSTIC; it does not change the T3 gate.

Usage:
  python3 scripts/ablation_ladder.py --db <stream> --ckpt-dir <dir> --tag <t> \
      --as-of <date> --customers 6000 --horizon-days 90 --seed 0 --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _group_folds(groups, nf, seed):
    cs = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(cs)
    fo = {c: i % nf for i, c in enumerate(cs)}
    return np.array([fo[c] for c in groups])


def _oof_metric(X, y, cu, kind, metric, nf, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    from looking_glass.supervised_gate import _metric_value

    Cls = HistGradientBoostingClassifier if kind == "clf" else HistGradientBoostingRegressor
    fold = _group_folds(cu, nf, seed)
    pred, yy = [], []
    for fi in range(nf):
        tr, va = fold != fi, fold == fi
        if not tr.any() or not va.any():
            continue
        m = Cls(max_iter=200).fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1] if kind == "clf" and metric == "auc" else m.predict(X[va])
        pred.append(np.asarray(p, float))
        yy.append(np.asarray(y[va]))
    if not pred:
        return None
    return float(_metric_value(np.concatenate(pred), np.concatenate(yy), kind, metric))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from sklearn.preprocessing import StandardScaler

    from looking_glass.cfm_data import _customer_keys, _read_stream
    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.supervised_gate import TARGETS, _labels, _raw_features, _states
    from looking_glass.universality import _full_history_features

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    d = _read_stream(cfg)
    keys = _customer_keys(d, cfg)
    Z, C = _states(model, cfg, keys)
    names = list(getattr(getattr(model, "layer_b", None), "names", ()) or ())
    sd = Z.shape[1] - len(names) if names and Z.shape[1] > len(names) else Z.shape[1] // 2
    G, _ = _full_history_features(cfg, C)
    RAW, pos = _raw_features(d, C, cfg)
    raw = RAW[[pos[str(c)] for c in C]]
    variants = {
        "E": StandardScaler().fit_transform(Z),
        "shape": StandardScaler().fit_transform(Z[:, :sd]),
        "measure": StandardScaler().fit_transform(Z[:, sd:]),
        "raw": StandardScaler().fit_transform(np.asarray(raw, float)),
        "generic": StandardScaler().fit_transform(np.asarray(G, float)),
    }
    lab = _labels(cfg, C, a.horizon_days)
    rows = []
    for name, (kind, metric, _base) in TARGETS.items():
        src = "next_purchase" if name == "reactivation" else name
        yv = np.array([lab.get(c, {}).get(src, np.nan) for c in C], dtype=object)
        ok = np.array([c in lab and src in lab[c] for c in C])
        if not ok.any():
            continue
        cu = np.array(C)[ok]
        if kind == "clf":
            classes = sorted({str(v) for v in yv[ok]})
            if len(classes) < 2:
                continue
            cmap = {c: i for i, c in enumerate(classes)}
            y = np.array([cmap[str(v)] for v in yv[ok]], dtype=int)
        else:
            y = np.array([float(v) for v in yv[ok]], dtype=float)
        nf = max(2, min(5, len(np.unique(cu))))
        scores = {
            vn: _oof_metric(V[ok], y, cu, kind, metric, nf, a.seed) for vn, V in variants.items()
        }
        rows.append({"head": name, "metric": metric, "scores": scores})
    out = {
        "baseline": "same GBM head, same folds, held-out customers",
        "seed": a.seed,
        "rows": rows,
        "variant_means": {
            v: round(
                float(np.mean([r["scores"][v] for r in rows if r["scores"].get(v) is not None])), 4
            )
            for v in variants
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["variant_means"], indent=1))
    return out


if __name__ == "__main__":
    main()
