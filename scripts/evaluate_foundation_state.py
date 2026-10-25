"""Attribution battery: which part of the frozen foundation state carries the signal?

Same holdout, same head, same folds. Reports table / shape / learned_state /
forecast / lawful measure / full state, so no row can conflate the compact learned
embedding with the larger published bundle.

Writes docs/report_cards/experiments/foundation_state.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
)
from sklearn.metrics import normalized_mutual_info_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))

import auto_encoder as AE  # noqa: E402
from looking_glass import measure_state as MS  # noqa: E402
from looking_glass import state_layout as SL  # noqa: E402

OUT = ROOT / "docs/report_cards/experiments/foundation_state.json"


def z(x):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def oof(x, y, kind, folds=3):
    out = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(x):
        m = (
            HistGradientBoostingRegressor(max_iter=80, random_state=0)
            if kind == "reg"
            else HistGradientBoostingClassifier(max_iter=80, random_state=0)
        )
        m.fit(x[tr], y[tr])
        out[te] = m.predict(x[te]) if kind == "reg" else m.predict_proba(x[te])[:, 1]
    return out


def metric(y, p, kind):
    return float(r2_score(y, p) if kind == "reg" else roc_auc_score(y, p))


def margin(y, a, b, kind, n=200):
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if kind == "clf" and len(np.unique(y[ix])) < 2:
            continue
        vals.append(metric(y[ix], a[ix], kind) - metric(y[ix], b[ix], kind))
    v = np.asarray(vals)
    return {
        "mean": round(float(v.mean()), 4),
        "lo": round(float(np.quantile(v, 0.025)), 4),
        "hi": round(float(np.quantile(v, 0.975)), 4),
    }


def purity(x, labels, k=10):
    dd = cdist(z(x), z(x))
    np.fill_diagonal(dd, np.inf)
    ix = np.argpartition(dd, k, axis=1)[:, :k]
    return round(float((labels[ix] == labels[:, None]).mean()), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--frozen", default="/tmp/opencode/frozen_6k")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    import frozen as F

    E, Xtab, labels, hold, _seg, model, vs, _B, _best, BLK = F.load(args.frozen)
    E = F.require_finite(E, "E")
    BLK = F.require_finite(BLK, "BLK")
    Xtab = np.nan_to_num(Xtab)
    layout = SL.describe(vs, model)

    _b, _a, keys, seqs, vs2, *_ = AE.load_data(args.data)
    hold_set = set(hold)
    train_keys = [k for k in keys if k not in hold_set]
    builder = MS.MeasureBuilder(seqs, train_keys, vs2)
    M = builder.transform(seqs, hold)

    learned = SL.slice_view(BLK, vs, model, "learned_state")
    forecast = SL.slice_view(BLK, vs, model, "forecast")
    arms = {
        "table": Xtab,
        "shape": E,
        "learned_state": learned,
        "forecast": forecast,
        "measure": M,
        "full_legacy": np.hstack([E, BLK]),
        "full_measure": np.hstack([E, BLK, M]),
    }

    truth = {
        c["customer"]: c
        for c in json.load(open(Path(args.data) / "reference_truth.json"))["customers"]
    }
    res = {
        "config": {
            "data": args.data,
            "frozen": args.frozen,
            "layout": layout,
            "measure_dim": builder.dim,
            "measure_coverage": builder.coverage,
        },
        "supervised": {},
        "identity": {},
        "geometry": {},
    }

    for target, kind in (("value90", "reg"), ("orders90", "reg"), ("churn90", "clf")):
        y = np.asarray(labels["y_" + target.replace("90", "_90")], dtype=float)
        row = {}
        preds = {}
        for name, x in arms.items():
            p = oof(x, y, kind)
            preds[name] = p
            row[name] = round(metric(y, p, kind), 4)
        for name in arms:
            if name != "table":
                row[name + "_vs_table"] = margin(y, preds[name], preds["table"], kind)
        res["supervised"][target] = row
        print(target, {k: v for k, v in row.items() if isinstance(v, float)}, flush=True)

    ch = np.array([str(truth[k]["channel_pref"]) for k in hold], dtype=object)
    cat = np.array([str(truth[k]["category_pref"]) for k in hold], dtype=object)
    for name, x in arms.items():
        res["identity"][name] = {"channel": purity(x, ch), "category": purity(x, cat)}

    yv = np.asarray(labels["y_value_90"], dtype=float)
    yc = np.asarray(labels["y_churn_90"], dtype=float)
    rp = np.array([float(truth[k]["return_proneness"]) for k in hold])
    anomaly = (rp >= np.quantile(rp, 0.9)).astype(int)
    regime = np.array([str(truth[k]["regime0"]) for k in hold], dtype=object)
    for name, x in arms.items():
        zz = z(x)
        dd = cdist(zz, zz)
        np.fill_diagonal(dd, np.inf)
        nn = dd.argmin(1)
        km = KMeans(4, n_init=5, random_state=0).fit_predict(zz)
        sc = -IsolationForest(random_state=0).fit(zz).score_samples(zz)
        res["geometry"][name] = {
            "retrieval": round(float(spearmanr(yv, yv[nn]).correlation), 4),
            "churn_nn": round(float(np.mean(yc == yc[nn])), 4),
            "anomaly_auc": round(float(roc_auc_score(anomaly, sc)), 4),
            "nmi_regime": round(float(normalized_mutual_info_score(regime, km)), 4),
        }

    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print("IDENTITY", json.dumps(res["identity"], indent=1), flush=True)
    print("GEOMETRY", json.dumps(res["geometry"], indent=1), flush=True)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
