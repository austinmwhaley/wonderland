"""Iteration-1 measurement: donor negative controls + refreshed mass retention.

Trains the current-config encoder once, then emits ONE receipt:
  - supervised battery with four arms on the SAME learner:
      E (learned) / table / random E / destroyed E (per-dim shuffled)
  - past & future retention (linear probe R2) for the mass axis.

Writes docs/report_cards/experiments/encoder_controls.json
"""

import sys
import json
import numpy as np

sys.path.insert(0, "/home/austin-whaley/Documents/wonderland")
sys.path.insert(0, "/home/austin-whaley/Documents/wonderland/scripts/demo")
import torch  # noqa: E402
import auto_encoder as AE  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier  # noqa: E402

import frozen as _fzmod  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/t0_loop/relational_v3"
OUT = "/home/austin-whaley/Documents/wonderland/docs/report_cards/experiments/encoder_controls.json"

import os as _os

_fz = _os.environ.get("FROZEN")
if _fz:
    from frozen import load as _fload

    E, Xtab, labels, hold, seg, model, vs, B, best, BLK = _fload(_fz)
else:
    E, Xtab, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(
        DATA, seed=0, epochs=12, d=128, layers=2
    )
torch.cuda.empty_cache()
E = _fzmod.require_finite(E, "E")
Xtab = np.nan_to_num(Xtab)

rng = np.random.default_rng(0)
Er = rng.standard_normal(E.shape).astype(np.float32)
Ed = E.copy()
for j in range(Ed.shape[1]):
    rng.shuffle(Ed[:, j])


def oof(X, y, kind):
    o = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = (
            HistGradientBoostingRegressor(max_iter=200)
            if kind == "reg"
            else HistGradientBoostingClassifier(max_iter=200)
        )
        m.fit(X[tr], y[tr])
        o[te] = m.predict(X[te]) if kind == "reg" else m.predict_proba(X[te])[:, 1]
    return o


def score(y, p, kind):
    return float(r2_score(y, p) if kind == "reg" else roc_auc_score(y, p))


sup = {}
for t, kind in (("value90", "reg"), ("orders90", "reg"), ("churn90", "clf")):
    y = labels["y_" + t.replace("90", "_90")]
    sup[t] = {
        "table": score(y, oof(Xtab, y, kind), kind),
        "E": score(y, oof(E, y, kind), kind),
        "random": score(y, oof(Er, y, kind), kind),
        "destroyed": score(y, oof(Ed, y, kind), kind),
    }
    print(
        f"[ctrl] {t:9s} table {sup[t]['table']:.4f} E {sup[t]['E']:.4f} "
        f"random {sup[t]['random']:.4f} destroyed {sup[t]['destroyed']:.4f}",
        flush=True,
    )

# ---- retention (mass axis): is the past/future magnitude reconstructable from E? ----
_, A, keys, seqs, vs2, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons = AE.load_data(
    DATA
)
AE.AS_OF_TS = as_of_ts
h = float(np.median(taus))

ret_past, ret_fut = [], []
with torch.no_grad():
    for i in range(0, len(hold), 64):
        bk = hold[i : i + 64]
        bc = [max(8, int(0.7 * len(seqs[k][5]))) for k in bk]
        t, past_t, qg, lens, fut_t = AE.collate(
            seqs, bk, bc, [0.0] * len(bk), vs2, True, None, None, True, h, None, None
        )
        Hs, Eh, w, *_r = model(
            t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"]
        )
        Eh = Eh.cpu().numpy()
        pp = model.past(torch.tensor(Eh, device=AE.DEV)).cpu().numpy()
        _hi = int(np.argmin(np.abs(np.array(vs2["horizons"]) - h)))
        pf = model.head_fut[_hi](torch.tensor(Eh, device=AE.DEV)).cpu().numpy()
        P, F = past_t.cpu().numpy(), fut_t.cpu().numpy()
        for j in range(P.shape[1]):
            if np.std(P[:, j]) > 1e-9:
                ret_past.append((P[:, j], pp[:, j]))
        for j in range(F.shape[1]):
            if np.std(F[:, j]) > 1e-9:
                ret_fut.append((F[:, j], pf[:, j]))


def recon(pairs):
    vals = []
    for a, b in pairs:
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            r = np.corrcoef(a, b)[0, 1]
            vals.append(r * r)
    return float(np.mean(vals)) if vals else 0.0


retention = {
    "past_r2": recon(ret_past),
    "future_r2": recon(ret_fut),
    "past_dims": len(ret_past),
    "future_dims": len(ret_fut),
}

payload = {
    "config": {"data": DATA, "seed": 0, "epochs": 12, "d": 128, "best_val": float(best)},
    "supervised": sup,
    "retention": retention,
}
from pathlib import Path  # noqa: E402

Path(OUT).write_text(json.dumps(payload, indent=1))
print("retention", retention, flush=True)
print("WROTE", OUT)
