"""Ceiling / floor measurement: for the supervised targets, report
  floor   = table + XGBoost (legacy path)
  E       = frozen encoder (+blocks) + XGBoost
  ceiling = latent oracle: XGBoost on the generator's TRUE latent variables
and a paired bootstrap SE of (E - floor).

Writes docs/report_cards/experiments/encoder_ceiling.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "/home/austin-whaley/Documents/wonderland")
sys.path.insert(0, "/home/austin-whaley/Documents/wonderland/scripts/demo")
import torch  # noqa: E402
import auto_encoder as AE  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402
import xgboost as xgb  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/t0_loop/v7b"
OUT = Path("/home/austin-whaley/Documents/wonderland/docs/report_cards/experiments/encoder_ceiling.json")

import os as _os
_fz = _os.environ.get("FROZEN")
if _fz:
    from frozen import load as _fload
    E, Xtab, labels, hold, seg, model, vs, B, best, BLK = _fload(_fz)
else:
    E, Xtab, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(DATA, seed=0, epochs=12, d=128, layers=2)
torch.cuda.empty_cache()
E = np.nan_to_num(E)
BLK = np.nan_to_num(BLK)
Xe = np.hstack([E, BLK])
Xtab = np.nan_to_num(Xtab)

# full-history oracle: table features + the full past-window summary (sequence stats)
_, A2, keys2, seqs2, vs2, *_ = AE.load_data(DATA)
AE.AS_OF_TS = 0.0
fc = [len(seqs2[k][5]) for k in hold]
fq = [0.0] * len(hold)
_, P, _, _, _ = AE.collate(seqs2, hold, fc, fq, vs2, True, None, None)
Xseq = np.nan_to_num(P.cpu().numpy())
Xlat = np.nan_to_num(np.hstack([Xtab, Xseq]))  # full-history oracle feature set


def oof(X, y, kind):
    o = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = (xgb.XGBRegressor(device="cuda", tree_method="hist", n_estimators=300, max_depth=6, learning_rate=0.06)
             if kind == "reg" else xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=300, max_depth=6, learning_rate=0.06, eval_metric="logloss"))
        m.fit(X[tr], y[tr])
        o[te] = m.predict(X[te]) if kind == "reg" else m.predict_proba(X[te])[:, 1]
    return o


def metric(y, p, kind):
    return float(r2_score(y, p) if kind == "reg" else roc_auc_score(y, p))


def boot(y, pA, pB, kind, n=400, seed=0):
    rng = np.random.default_rng(seed)
    m = len(y)
    d = np.array([metric(y[i], pA[i], kind) - metric(y[i], pB[i], kind) for i in (rng.integers(0, m, m) for _ in range(n))])
    return float(d.mean()), float(d.std()), float(np.quantile(d, 0.025))


res = {}
for t, ykey, kind in (("value90", "y_value_90", "reg"), ("orders90", "y_orders_90", "reg"),
                      ("churn90", "y_churn_90", "clf"),
                      ("discount_sensitivity", "y_discount_sensitivity", "reg"),
                      ("ltv", "y_ltv", "reg")):
    if ykey not in labels:
        print(f"[ceil] {t}: label missing (skip)", flush=True)
        continue
    y = labels[ykey]
    p_f, p_e, p_c = oof(Xtab, y, kind), oof(Xe, y, kind), oof(Xlat, y, kind)
    floor, ev, ceil = metric(y, p_f, kind), metric(y, p_e, kind), metric(y, p_c, kind)
    mean, se, lo = boot(y, p_e, p_f, kind)
    gap = ceil - floor
    closed = (ev - floor) / gap if abs(gap) > 1e-9 else 0.0
    res[t] = {"floor": round(floor, 4), "E": round(ev, 4), "ceiling": round(ceil, 4),
              "E_minus_floor": round(mean, 4), "se": round(se, 4), "lo": round(lo, 4),
              "gap": round(gap, 4), "closed_frac": round(closed, 3),
              "beats_floor_2se": bool(mean > 2 * se),
              "reaches_ceiling": bool(ev >= floor + 0.5 * gap)}
    print(f"[ceil] {t:9s} floor {floor:.3f} E {ev:.3f} ceiling {ceil:.3f} | E-floor {mean:+.4f}±{se:.4f} "
          f"gap {gap:+.3f} closed {closed:.2f} beats2se={res[t]['beats_floor_2se']} reaches={res[t]['reaches_ceiling']}", flush=True)

OUT.write_text(json.dumps({"config": {"data": DATA, "seed": 0}, "results": res}, indent=1))
print("WROTE", OUT)
