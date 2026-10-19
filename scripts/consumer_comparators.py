"""Fair consumer comparators: E vs a GENERIC hand-aggregated measure.

The curated RFM baseline is task-shaped. This script compares the frozen E
against the SAME generic full-history measure (every field aggregated) fed to
the SAME head, isolating representation quality from feature engineering.
Also compares E vs the generic measure for unsupervised geometry.
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
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument(
        "--supervised-from",
        default=None,
        dest="supervised_from",
        help="reuse a prior comparator receipt's supervised rows (recompute unsupervised only)",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.supervised_gate import (
        TARGETS,
        _bootstrap_delta,
        _labels,
        _metric_value,
        _states,
    )
    from looking_glass.universality import _full_history_features

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    d = None
    from looking_glass.cfm_data import _customer_keys, _read_stream

    d = _read_stream(cfg)
    keys = _customer_keys(d, cfg)
    Z, C = _states(model, cfg, keys)
    G, gnames = _full_history_features(cfg, C)
    G = np.asarray(G, dtype=float)
    lab = _labels(cfg, C, a.horizon_days)
    Z = StandardScaler().fit_transform(Z)
    G = StandardScaler().fit_transform(G)

    rows = []
    _loop = [] if a.supervised_from else list(TARGETS.items())
    for name, (kind, metric, _base) in _loop:
        src = "next_purchase" if name == "reactivation" else name
        yv = np.array([lab.get(c, {}).get(src, np.nan) for c in C], dtype=object)
        ok = np.array([c in lab and src in lab[c] for c in C])
        if not ok.any():
            continue
        cu = np.array(C)[ok]
        Xe, Xg = Z[ok], G[ok]
        if kind == "clf":
            classes = sorted({str(v) for v in yv[ok]})
            cmap = {c: i for i, c in enumerate(classes)}
            y = np.array([cmap[str(v)] for v in yv[ok]], dtype=int)
            if len(classes) < 2:
                continue
        else:
            y = np.array([float(v) for v in yv[ok]], dtype=float)
        nf = max(2, min(5, len(np.unique(cu))))
        oof_e, oof_g, oof_y = [], [], []
        for tr, va in GroupKFold(nf).split(Xe, groups=cu):
            if kind == "clf":
                me = HistGradientBoostingClassifier(max_iter=200).fit(Xe[tr], y[tr])
                mg = HistGradientBoostingClassifier(max_iter=200).fit(Xg[tr], y[tr])
                if metric == "auc":
                    pe = me.predict_proba(Xe[va])[:, 1]
                    pg = mg.predict_proba(Xg[va])[:, 1]
                else:
                    pe = me.predict(Xe[va])
                    pg = mg.predict(Xg[va])
            else:
                me = HistGradientBoostingRegressor(max_iter=200).fit(Xe[tr], y[tr])
                mg = HistGradientBoostingRegressor(max_iter=200).fit(Xg[tr], y[tr])
                pe = me.predict(Xe[va])
                pg = mg.predict(Xg[va])
            oof_e.append(np.asarray(pe, float))
            oof_g.append(np.asarray(pg, float))
            oof_y.append(np.asarray(y[va]))
        predE = np.concatenate(oof_e)
        predG = np.concatenate(oof_g)
        y_oof = np.concatenate(oof_y)
        mE = _metric_value(predE, y_oof, kind, metric)
        mG = _metric_value(predG, y_oof, kind, metric)
        dlt, lo, hi = _bootstrap_delta(predE, predG, y_oof, kind, metric)
        rows.append(
            {
                "head": name,
                "E": round(float(mE), 4),
                "generic": round(float(mG), 4),
                "delta": round(float(dlt), 4),
                "lo": round(float(lo), 4),
                "hi": round(float(hi), 4),
                "win": bool(dlt > 0 and lo >= 0),
                "non_inferior": bool(hi >= 0),
            }
        )

    if a.supervised_from:
        rows = json.loads(Path(a.supervised_from).read_text())["supervised"]

    # unsupervised: E vs generic measure (same pipeline). The value geometry is
    # certified on the ACCESSIBLE block-wise served view (the explicit measure
    # block; DEC-215); the full shape‖measure concatenation is reported because
    # the whitened shape is magnitude-blind and dilutes distance-based consumers.
    from looking_glass.unsupervised_gate import _run_unsupervised

    e_metrics = _run_unsupervised(model, cfg, states=(C, Z), block="measure")
    e_full = _run_unsupervised(model, cfg, states=(C, Z), block="full")
    g_metrics = _run_unsupervised(model, cfg, states=(C, G))
    _keys = ("silhouette", "eta_squared", "lookalike_enrichment", "anova_f")
    unsup = {
        "E": {k: e_metrics.get(k) for k in _keys},
        "E_full": {k: e_full.get(k) for k in _keys},
        "generic": {k: g_metrics.get(k) for k in _keys},
        "view": "block-wise served measure (accessible); full concat reported",
    }
    wins = sum(1 for r in rows if r["win"])
    rec = {
        "baseline": "generic full-history measure + same head",
        "supervised": rows,
        "supervised_wins": wins,
        "supervised_total": len(rows),
        "unsupervised": unsup,
    }
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(
        json.dumps(
            {"supervised_wins": wins, "supervised_total": len(rows), "unsupervised": unsup},
            indent=2,
        )
    )
    return rec


if __name__ == "__main__":
    main()
