"""Fast axis screen, one config: train the encoder and compute the discriminating
metrics IN-PROCESS (no cache save/reload).  Prints one JSON line.

    python3 scripts/screen_one.py --axes "measure" --epochs 4 --data /tmp/opencode/t0_loop/ref_v3b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--axes", default="")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.environ["WQ_AXES"] = a.axes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
)  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import auto_encoder as AE  # noqa: E402


def z(X):
    X = np.nan_to_num(np.asarray(X, float))
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def purity(Z, lab, k=10):
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return float((lab[idx] == lab[:, None]).mean())


def oof(X, y, kind, folds=3):
    o = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        m = (
            HistGradientBoostingRegressor(max_iter=80, random_state=0)
            if kind == "reg"
            else HistGradientBoostingClassifier(max_iter=80, random_state=0)
        )
        m.fit(X[tr], y[tr])
        o[te] = m.predict(X[te]) if kind == "reg" else m.predict_proba(X[te])[:, 1]
    return o


def main():
    E, Xtab, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(
        a.data, seed=a.seed, epochs=a.epochs
    )
    E = np.nan_to_num(E)
    BLK = np.nan_to_num(BLK)
    Xtab = np.nan_to_num(Xtab)
    STATE = np.hstack([E, BLK])
    data = Path(a.data)
    truth = {c["customer"]: c for c in json.load(open(data / "reference_truth.json"))["customers"]}
    r = {"axes": a.axes or "baseline", "E": int(E.shape[1]), "state": int(STATE.shape[1])}
    from sklearn.metrics import r2_score

    def sc(y, p, kind):
        return float(r2_score(y, p) if kind == "reg" else roc_auc_score(y, p))

    # supervised margin on the published state (the card's arm)
    rng = np.random.default_rng(0)
    for t, kind in (("value90", "reg"), ("orders90", "reg")):
        y = np.asarray(labels["y_" + t.replace("90", "_90")], float)
        pS, pT = oof(STATE, y, kind), oof(Xtab, y, kind)
        m = len(y)
        d = []
        for _ in range(100):
            ix = rng.integers(0, m, m)
            if kind == "clf" and len(np.unique(y[ix])) < 2:
                continue
            d.append(sc(y[ix], pS[ix], kind) - sc(y[ix], pT[ix], kind))
        d = np.array(d)
        r[t] = {
            "state": round(sc(y, pS, kind), 4),
            "margin": round(float(d.mean()), 4),
            "lo": round(float(np.quantile(d, 0.025)), 4),
        }

    # identity + geometry on the published state
    zS, zT = z(STATE), z(Xtab)
    ch = np.array([str(truth[k]["channel_pref"]) for k in hold], dtype=object)
    cat = np.array([str(truth[k]["category_pref"]) for k in hold], dtype=object)
    r["id_channel"] = round(purity(zS, ch), 4)
    r["id_category"] = round(purity(zS, cat), 4)
    r["id_category_table"] = round(purity(zT, cat), 4)
    yv = np.asarray(labels["y_value_90"], float)
    yc = np.asarray(labels["y_churn_90"], float)
    D = cdist(zS, zS)
    np.fill_diagonal(D, np.inf)
    i1 = D.argmin(1)
    r["retrieval"] = round(float(spearmanr(yv, yv[i1]).correlation), 4)
    r["churn_nn"] = round(float(np.mean(yc == yc[i1])), 4)
    rp = np.array([float(truth[k]["return_proneness"]) for k in hold])
    lab = (rp >= np.quantile(rp, 0.9)).astype(int)
    r["anomaly_auc"] = round(
        float(roc_auc_score(lab, -IsolationForest(random_state=0).fit(zS).score_samples(zS))), 4
    )

    # next-event probe (subset)
    _B, _A, _keys, seqs, vs2, *_ = AE.load_data(a.data)
    keys = hold[:80]
    Z, y, g = [], [], []
    with torch.no_grad():
        for i in range(0, len(keys), 48):
            bk = keys[i : i + 48]
            fc = [len(seqs[k][5]) for k in bk]
            fq = [
                (AE.AS_OF_TS - seqs[k][5][-1]) / 86400.0 for k in bk
            ]  # principle-ok: unit conversion
            t, _p, qg, lens, _f = AE.collate(seqs, bk, fc, fq, vs2, False, None, None)
            Hs, *_ = model(
                t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"]
            )
            Hs = Hs.cpu().numpy()
            et = t["et"].cpu().numpy()
            L = lens.cpu().numpy().astype(int)
            for b in range(len(bk)):
                if L[b] < 3:
                    continue
                Z.append(Hs[b, : L[b] - 1])
                y.append(et[b, 1 : L[b]])
                g.append(np.full(L[b] - 1, b + i))
    Z, y, g = np.concatenate(Z), np.concatenate(y), np.concatenate(g)
    counts = np.bincount(y, minlength=vs2["et"])
    naive = float((counts / counts.sum()).max())
    accs = []
    for tr, te in GroupKFold(2).split(Z, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
        m.fit(Z[tr], y[tr])
        accs.append(float(np.mean(m.predict(Z[te]) == y[te])))
    r["next_skill"] = round(float(np.mean(accs) - naive), 4)
    r["next_naive"] = round(naive, 4)
    print("SCREEN " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
