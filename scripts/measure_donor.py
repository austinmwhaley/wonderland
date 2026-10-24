"""Iteration-5 measurement: donor battery (raw / fair table / E / table+seq / table+seq+E).

One learner (HistGB, 5-fold OOF) for all arms so comparisons are self-consistent.
Writes docs/report_cards/experiments/encoder_donor.json
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
import xgboost as xgb  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/t0_loop/relational_6k_v3"
OUT = Path("/home/austin-whaley/Documents/wonderland/docs/report_cards/experiments/encoder_donor.json")
FIXED = {"value_30d", "value_90d", "recency_days", "days_since_last_purchase"}

E, _, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(DATA, seed=0, epochs=12, d=128, layers=2)
torch.cuda.empty_cache()
E = np.nan_to_num(E)
BLK = np.nan_to_num(BLK)
Xe = np.hstack([E, BLK])  # the demo's frozen-E arm (E + blocks)

_, A, keys, seqs, vs2, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons = AE.load_data(DATA)
AE.AS_OF_TS = as_of_ts
feat = [c for c in A.columns if not c.startswith("y_") and c not in ("customer_key", "signup_ts")]
amap = {r["customer_key"]: r for r in A.to_dicts()}
rows = [amap[k] for k in hold]
dummy = [c for c in ("segment", "region", "acquisition_channel", "loyalty_tier") if c in feat]


def build(cols):
    X = pl.DataFrame([{c: r[c] for c in cols} for r in rows]).to_dummies(columns=dummy, drop_first=True)
    return np.nan_to_num(X.to_numpy().astype(float))


Xraw = build(feat)
Xfair = build([c for c in feat if c not in FIXED])
fc = [len(seqs[k][5]) for k in hold]
fq = [(as_of_ts - seqs[k][5][-1]) / 86400.0 for k in hold]
_, P, _, _, _ = AE.collate(seqs, hold, fc, fq, vs2, True, None, None)
Xseq = np.nan_to_num(P.cpu().numpy())
Xts = np.hstack([Xraw, Xseq])
Xtse = np.hstack([Xraw, Xseq, Xe])


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


def sc(X, y, kind):
    return metric(y, oof(X, y, kind), kind)


def boot_se(y, pA, pB, kind, n=400, seed=0):
    rng = np.random.default_rng(seed)
    m = len(y)
    d = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        d.append(metric(y[ix], pA[ix], kind) - metric(y[ix], pB[ix], kind))
    d = np.array(d)
    return float(d.mean()), float(d.std()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


res = {}
for t, kind in (("value90", "reg"), ("orders90", "reg"), ("churn90", "clf")):
    y = labels["y_" + t.replace("90", "_90")]
    p_raw, p_fair, p_e, p_ts, p_tse = (oof(Xraw, y, kind), oof(Xfair, y, kind), oof(Xe, y, kind),
                                      oof(Xts, y, kind), oof(Xtse, y, kind))
    m_raw = boot_se(y, p_e, p_raw, kind)
    m_fair = boot_se(y, p_e, p_fair, kind)
    m_add = boot_se(y, p_tse, p_ts, kind)
    res[t] = {
        "raw": sc(y, p_raw, kind), "fair": sc(y, p_fair, kind), "E": sc(y, p_e, kind),
        "table_seq": sc(y, p_ts, kind), "table_seq_E": sc(y, p_tse, kind),
        "E_minus_raw": {"mean": m_raw[0], "se": m_raw[1], "lo": m_raw[2], "hi": m_raw[3]},
        "E_minus_fair": {"mean": m_fair[0], "se": m_fair[1], "lo": m_fair[2], "hi": m_fair[3]},
        "add_over_tseq": {"mean": m_add[0], "se": m_add[1], "lo": m_add[2], "hi": m_add[3]},
    }
    r = res[t]
    print(f"[donor] {t:9s} raw {r['raw']:.4f} fair {r['fair']:.4f} E {r['E']:.4f} "
          f"tseq {r['table_seq']:.4f} tseqE {r['table_seq_E']:.4f} | "
          f"E-raw {r['E_minus_raw']['mean']:+.4f}±{r['E_minus_raw']['se']:.4f} "
          f"E-fair {r['E_minus_fair']['mean']:+.4f} add {r['add_over_tseq']['mean']:+.4f}±{r['add_over_tseq']['se']:.4f}", flush=True)

OUT.write_text(json.dumps({"config": {"data": DATA, "seed": 0, "epochs": 12}, "results": res}, indent=1))
print("WROTE", OUT)
