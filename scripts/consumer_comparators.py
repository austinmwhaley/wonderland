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
        "--seeds",
        default="0",
        help="comma list of fold/bootstrap seeds (multi-seed consumer gate)",
    )
    ap.add_argument(
        "--supervised-from",
        default=None,
        dest="supervised_from",
        help="reuse a prior comparator receipt's supervised rows (recompute unsupervised only)",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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

    def _group_folds(groups, nf, seed):
        """Customer-disjoint folds with a SEEDED group order (multi-seed gate)."""
        cs = np.array(sorted(set(groups)))
        rng = np.random.default_rng(seed)
        rng.shuffle(cs)
        fo = {c: i % nf for i, c in enumerate(cs)}
        return np.array([fo[c] for c in groups])

    seeds = [int(x) for x in str(a.seeds).split(",") if str(x).strip()]
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
            _min_class = int(np.bincount(y, minlength=2).min())
            if _min_class < max(20, int(0.02 * len(y))):
                rows.append(
                    {
                        "head": name,
                        "E": None,
                        "generic": None,
                        "delta": None,
                        "delta_se": None,
                        "lo": None,
                        "hi": None,
                        "win": False,
                        "non_inferior": False,
                        "degenerate": True,
                        "minority": _min_class,
                        "minority_frac": round(_min_class / max(len(y), 1), 5),
                        "n": int(len(y)),
                    }
                )
                continue
        else:
            y = np.array([float(v) for v in yv[ok]], dtype=float)
        nf = max(2, min(5, len(np.unique(cu))))
        per_seed = []
        for seed in seeds:
            fold = _group_folds(cu, nf, seed)
            oof_e, oof_g, oof_y = [], [], []
            for fi in range(nf):
                tr, va = fold != fi, fold == fi
                if not tr.any() or not va.any():
                    continue
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
            if not oof_y:
                continue
            predE = np.concatenate(oof_e)
            predG = np.concatenate(oof_g)
            y_oof = np.concatenate(oof_y)
            mE = _metric_value(predE, y_oof, kind, metric)
            mG = _metric_value(predG, y_oof, kind, metric)
            dlt, lo, hi = _bootstrap_delta(predE, predG, y_oof, kind, metric)
            per_seed.append((float(mE), float(mG), float(dlt), float(lo), float(hi)))
        if not per_seed:
            continue
        arr = np.asarray(per_seed, float)
        dmean = float(arr[:, 2].mean())
        dse = float(arr[:, 2].std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        lo_worst = float(arr[:, 3].min())
        hi_worst = float(arr[:, 4].min())
        win_seeds = int(sum(1 for r in per_seed if r[2] > 0 and r[3] >= 0))
        ni_seeds = int(sum(1 for r in per_seed if r[4] >= 0))
        rows.append(
            {
                "head": name,
                "E": round(float(arr[:, 0].mean()), 4),
                "generic": round(float(arr[:, 1].mean()), 4),
                "delta": round(dmean, 4),
                "delta_se": round(dse, 4),
                "lo": round(lo_worst, 4),
                "hi": round(hi_worst, 4),
                "win": bool(dmean > 0 and lo_worst >= 0 and win_seeds == len(per_seed)),
                "non_inferior": bool(hi_worst >= 0 and ni_seeds == len(per_seed)),
                "seeds": len(per_seed),
                "win_seeds": win_seeds,
                "non_inferior_seeds": ni_seeds,
            }
        )

    if a.supervised_from:
        rows = json.loads(Path(a.supervised_from).read_text())["supervised"]

    # unsupervised: E vs generic measure (same pipeline). The value geometry is
    # certified on the ACCESSIBLE block-wise served view (the explicit measure
    # block; DEC-215); the full shape‖measure concatenation is reported because
    # the whitened shape is magnitude-blind and dilutes distance-based consumers.
    from looking_glass.unsupervised_gate import _run_unsupervised

    _keys = ("silhouette", "eta_squared", "lookalike_enrichment", "anova_f")

    def _avg_unsup(block, seed):
        r = _run_unsupervised(model, cfg, states=(C, Z), block=block, seed=seed)
        return {k: r.get(k) for k in _keys}

    def _avg_generic(seed):
        r = _run_unsupervised(model, cfg, states=(C, G), seed=seed)
        return {k: r.get(k) for k in _keys}

    def _mean_metrics(dicts):
        out = {}
        for k in _keys:
            vals = [d[k] for d in dicts if d.get(k) is not None]
            out[k] = round(float(np.mean(vals)), 4) if vals else None
        return out

    e_per_seed = [_avg_unsup("measure", s) for s in seeds]
    e_full_per_seed = [_avg_unsup("full", s) for s in seeds]
    g_per_seed = [_avg_generic(s) for s in seeds]
    unsup = {
        "E": _mean_metrics(e_per_seed),
        "E_full": _mean_metrics(e_full_per_seed),
        "generic": _mean_metrics(g_per_seed),
        "seeds": len(seeds),
        "view": "block-wise served measure (accessible); full concat reported",
    }
    _valid = [r for r in rows if not r.get("degenerate")]
    wins = sum(1 for r in _valid if r["win"])
    rec = {
        "baseline": "generic full-history measure + same head",
        "supervised": rows,
        "supervised_wins": wins,
        "supervised_total": len(_valid),
        "supervised_degenerate_excluded": len(rows) - len(_valid),
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
