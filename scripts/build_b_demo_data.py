"""Export the small datasets the B-only demo visuals need.

Runs the frozen model once and writes `demo/b_data.json`:

  * mass      — predicted vs actual past total for one numeric field (scatter)
  * calibration — next-event confidence vs accuracy (10 bins)
  * spectrum  — singular values of the standardized state (log plot)

All three are self-supervised / label-free: the past-summary target, next-event
prediction, and the state's own geometry.

    python3 scripts/build_b_demo_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))

import auto_encoder as AE  # noqa: E402
import frozen as F  # noqa: E402

OUT = ROOT / "demo/b_data.json"


def _standardize(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--frozen", default="/tmp/opencode/frozen_6k")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sample", type=int, default=700)
    args = ap.parse_args()

    E, _Xtab, y_labels, hold, _seg, model, vs, _B, _best, _BLK = F.load(args.frozen)
    E = np.asarray(E, dtype=float)
    _b, _a, keys, seqs, vs2, _etv, catspec, _num, taus, _pm, as_of_ts, _per, _hor = AE.load_data(
        args.data
    )
    AE.AS_OF_TS = float(as_of_ts)
    vs2 = vs2 if vs2 else vs

    out = {}

    # ---- 1. spectrum of the frozen shape ---------------------------------
    z = _standardize(E)
    sv = np.linalg.svd(z, compute_uv=False)
    sv = sv / sv.max()
    out["spectrum"] = [round(float(v), 5) for v in sv[: min(96, len(sv))]]
    s2 = sv**2
    out["eff_rank"] = round(float(s2.sum() ** 2 / (s2**2).sum()), 2)

    # ---- 2. mass: predicted vs actual past summary -----------------------
    # The receipt defines past retention as the mean SQUARED CORRELATION over
    # dimensions (measure_controls.recon), not prediction R². Match that.
    rawspec = vs2.get("rawspec") or []
    et_n = int(vs2["et"])
    labels = [f"event type {j}" for j in range(et_n)]
    labels += [f"distinct {g['field']}" for g in catspec]
    for f, _kind in rawspec:
        labels += [f"{f} count", f"{f} sum", f"{f} mean", f"{f} sd", f"{f} min", f"{f} max"]
    labels += [
        "tenure",
        "time spread",
        "query gap",
        "query gap",
        "tenure",
        "events",
        "distinct events",
    ]

    per_dim: dict[int, list] = {}
    for i in range(0, len(hold), 32):
        bk = hold[i : i + 32]
        bc = [max(8, int(0.7 * len(seqs[k][5]))) for k in bk]
        t, past_t, qg, lens, _fut = AE.collate(seqs, bk, bc, [0.0] * len(bk), vs2, True, None, None)
        with torch.no_grad():
            _, Ecut, _, *_rest = model(
                t["et"],
                t["gcat"],
                t["gnum"],
                t["time"],
                t["gaps"],
                t["mask"],
                lens,
                qg,
                t["graw"],
            )
            pred = model.past(Ecut).detach().cpu().numpy()
        actual = past_t.cpu().numpy()
        for j in range(actual.shape[1]):
            a, pv = actual[:, j], pred[:, j]
            if np.std(a) > 1e-9 and np.std(pv) > 1e-9:
                per_dim.setdefault(j, [[], []])
                per_dim[j][0].append(a)
                per_dim[j][1].append(pv)

    scored = []
    for j, (aa, pp) in per_dim.items():
        a = np.concatenate(aa)
        pv = np.concatenate(pp)
        r = float(np.corrcoef(a, pv)[0, 1])
        if np.isfinite(r):
            scored.append((j, r * r))
    mean_r2 = float(np.mean([v for _j, v in scored])) if scored else 0.0
    scored.sort(key=lambda x: -x[1])
    best_j, best_r2 = scored[0] if scored else (0, 0.0)
    a = np.concatenate(per_dim[best_j][0])
    pv = np.concatenate(per_dim[best_j][1])
    az = (a - a.mean()) / (a.std() + 1e-12)
    pz = (pv - pv.mean()) / (pv.std() + 1e-12)
    out["mass"] = {
        "r2": round(mean_r2, 4),
        "best_dim": labels[best_j] if best_j < len(labels) else str(best_j),
        "best_r2": round(float(best_r2), 4),
        "n_dims": len(scored),
        "points": [[round(float(x), 3), round(float(y), 3)] for x, y in zip(az[:700], pz[:700])],
    }

    # ---- 3. calibration of the next-event head ---------------------------
    conf, corr = [], []
    bs = 32
    for i in range(0, len(hold), bs):
        bk = hold[i : i + bs]
        fc = [len(seqs[k][5]) for k in bk]
        fq2 = [(AE.AS_OF_TS - seqs[k][5][-1]) / 86400.0 for k in bk]
        t2, _p, qg2, lens2, _f = AE.collate(seqs, bk, fc, fq2, vs2, False, None, None)
        Hs, _E, _, *_r = model(
            t2["et"],
            t2["gcat"],
            t2["gnum"],
            t2["time"],
            t2["gaps"],
            t2["mask"],
            lens2,
            qg2,
            t2["graw"],
        )
        logits = model.head_et(Hs[:, :-1]).reshape(-1, model.head_et.out_features)
        y = t2["et"][:, 1:].reshape(-1)
        m = t2["mask"][:, 1:].reshape(-1) > 0
        p = logits[m].softmax(-1).detach().cpu().numpy()
        yy = y[m].cpu().numpy()
        conf.append(p.max(1))
        corr.append((p.argmax(1) == yy).astype(float))
    conf = np.concatenate(conf)
    corr = np.concatenate(corr)
    edges = np.linspace(0.0, 1.0, 11)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi)
        if sel.sum() > 0:
            bins.append(
                {
                    "conf": round(float(conf[sel].mean()), 4),
                    "acc": round(float(corr[sel].mean()), 4),
                    "n": int(sel.sum()),
                }
            )
    ece = float(np.mean([abs(b["acc"] - b["conf"]) for b in bins])) if bins else float("nan")
    out["calibration"] = {"bins": bins, "ece": round(ece, 4), "n": int(len(conf))}

    # ---- 4. ROC for the rate axis (churn ranking from the state) ---------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y_labels.get("y_churn_90", []), dtype=float)
    if len(y) == len(E) and len(np.unique(y)) > 1:
        oof = np.zeros(len(y))
        X = np.nan_to_num(E)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400))
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        fpr, tpr, _ = roc_curve(y, oof)
        idx = np.linspace(0, len(fpr) - 1, min(48, len(fpr))).astype(int)
        out["roc"] = {
            "auc": round(float(roc_auc_score(y, oof)), 4),
            "points": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx],
        }

    # ---- 5. identity: per-customer kNN purity distribution ----------------
    from scipy.spatial.distance import cdist

    truth = {
        c["customer"]: c
        for c in json.loads((Path(args.data) / "reference_truth.json").read_text())["customers"]
    }
    ch = np.array([str(truth[k]["channel_pref"]) for k in hold], dtype=object)
    Zs = _standardize(E)
    D = cdist(Zs, Zs)
    np.fill_diagonal(D, np.inf)
    k = 10
    nn = np.argpartition(D, k, axis=1)[:, :k]
    pur = (ch[nn] == ch[:, None]).mean(1)
    hist, edges = np.histogram(pur, bins=np.linspace(0, 1, 11))
    out["identity"] = {
        "mean": round(float(pur.mean()), 4),
        "chance": round(float(1.0 / max(len(set(ch)), 1)), 4),
        "bins": [
            {
                "lo": round(float(edges[i]), 2),
                "hi": round(float(edges[i + 1]), 2),
                "n": int(hist[i]),
            }
            for i in range(len(hist))
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(
        f"WROTE {args.out} | spectrum {len(out['spectrum'])} | mass R2 {out['mass']['r2']} "
        f"({len(out['mass']['points'])} pts) | ECE {out['calibration']['ece']} "
        f"({len(bins)} bins)"
    )


if __name__ == "__main__":
    main()
