"""Fast encoder battery: one frozen cache -> every axis metric in one pass.

Replaces running 5 separate scripts between axis runs.  Reports:
  - supervised: E vs table (HistGB 5-fold OOF) + paired bootstrap margin
  - identity:    channel / category kNN purity (E vs table) + paired CI
  - geometry:    retrieval rank, churn-neighbour, anomaly AUC, regime NMI, eff-rank
  - next-event:  linear probe skill vs the marginal baseline (holdout subset)

Usage:
  FROZEN=/tmp/opencode/frozen_6k python3 scripts/measure_battery.py --tag baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))

import auto_encoder as AE  # noqa: E402

OUT = ROOT / "docs/report_cards/experiments/battery.json"


def z(X):
    X = np.nan_to_num(np.asarray(X, float))
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def nn_idx(Z, k=1):
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    return np.argpartition(D, k, axis=1)[:, :k]


def purity(Z, lab, k=10):
    idx = nn_idx(Z, k)
    return (lab[idx] == lab[:, None]).mean(1)


def oof(X, y, kind, folds=5, seed=0):
    o = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m = (
            HistGradientBoostingRegressor(max_iter=200, random_state=0)
            if kind == "reg"
            else HistGradientBoostingClassifier(max_iter=200, random_state=0)
        )
        m.fit(X[tr], y[tr])
        o[te] = m.predict(X[te]) if kind == "reg" else m.predict_proba(X[te])[:, 1]
    return o


def score(y, p, kind):
    return float(r2_score(y, p) if kind == "reg" else roc_auc_score(y, p))


def boot(y, pA, pB, kind, n=400, seed=0):
    rng = np.random.default_rng(seed)
    m = len(y)
    d = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        if kind == "clf" and len(np.unique(y[ix])) < 2:
            continue
        d.append(score(y[ix], pA[ix], kind) - score(y[ix], pB[ix], kind))
    d = np.array(d)
    return float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def boot_pair(eo, to, n=400, seed=0):
    rng = np.random.default_rng(seed)
    m = len(eo)
    d = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        d.append(eo[ix].mean() - to[ix].mean())
    d = np.array(d)
    return float(eo.mean()), float(to.mean()), float(np.quantile(d, 0.025))


def next_probe(model, seqs, hold, vs, as_of_ts, n_cust=400):
    """Linear probe on per-step states: Hs[t] -> next event type. Skill vs marginal."""
    keys = hold[:n_cust]
    Z, y, g = [], [], []
    with torch.no_grad():
        for i in range(0, len(keys), 48):
            bk = keys[i : i + 48]
            fc = [len(seqs[k][5]) for k in bk]
            _day = 86400.0  # principle-ok: seconds per day (unit conversion, not a cadence)
            fq = [(as_of_ts - seqs[k][5][-1]) / _day for k in bk]
            t, _p, qg, lens, _f = AE.collate(seqs, bk, fc, fq, vs, False, None, None)
            Hs, *_ = model(
                t["et"], t["gcat"], t["gnum"], t["time"], t["gaps"], t["mask"], lens, qg, t["graw"]
            )
            Hs = Hs.cpu().numpy()
            et = t["et"].cpu().numpy()
            L = lens.cpu().numpy().astype(int)
            for b, k in enumerate(bk):
                n = L[b]
                if n < 3:
                    continue
                Z.append(Hs[b, : n - 1])
                y.append(et[b, 1:n])
                g.append(np.full(n - 1, b + i))
    if not Z:
        return {"skill": float("nan"), "naive": float("nan"), "n": 0}
    Z = np.concatenate(Z)
    y = np.concatenate(y)
    g = np.concatenate(g)
    counts = np.bincount(y, minlength=vs["et"])
    naive = float((counts / counts.sum()).max())
    accs = []
    gkf = GroupKFold(4)
    for tr, te in gkf.split(Z, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
        m.fit(Z[tr], y[tr])
        accs.append(float(np.mean(m.predict(Z[te]) == y[te])))
    return {
        "skill": float(np.mean(accs) - naive),
        "naive": naive,
        "raw_acc": float(np.mean(accs)),
        "n": int(len(y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="/tmp/opencode/frozen_6k")
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", default="/tmp/opencode/battery")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    import frozen as F

    E, Xtab, labels, hold, seg, model, vs, B, best, BLK = F.load(args.frozen)
    E = np.nan_to_num(E)
    Xtab = np.nan_to_num(Xtab)
    BLK = np.nan_to_num(BLK)
    STATE = np.hstack([E, BLK])  # the published state consumers receive
    data = Path(args.data)
    truth = {c["customer"]: c for c in json.load(open(data / "reference_truth.json"))["customers"]}
    res = {"tag": args.tag, "E_dim": int(E.shape[1]), "BLK_dim": int(BLK.shape[1])}

    # ---- supervised (E-only and published state) ----
    sup = {}
    for t, kind in (("value90", "reg"), ("orders90", "reg"), ("churn90", "clf")):
        y = np.asarray(labels["y_" + t.replace("90", "_90")], float)
        pE, pS, pT = oof(E, y, kind), oof(STATE, y, kind), oof(Xtab, y, kind)
        m, lo, hi = boot(y, pS, pT, kind)
        sup[t] = {
            "E": score(y, pE, kind),
            "state": score(y, pS, kind),
            "table": score(y, pT, kind),
            "margin": m,
            "lo": lo,
            "hi": hi,
        }
        print(
            f"[sup] {t:9s} E {sup[t]['E']:.4f} state {sup[t]['state']:.4f} "
            f"table {sup[t]['table']:.4f} margin {m:+.4f} [{lo:+.4f},{hi:+.4f}]",
            flush=True,
        )
    res["supervised"] = sup

    # ---- identity (E-only and published state) ----
    ch = np.array([str(truth[k]["channel_pref"]) for k in hold], dtype=object)
    cat = np.array([str(truth[k]["category_pref"]) for k in hold], dtype=object)
    ident = {}
    for name, lab in (("channel", ch), ("category", cat)):
        row = {}
        for arm, X in (("E", E), ("state", STATE)):
            eo, to = purity(z(X), lab), purity(z(Xtab), lab)
            em, tm, lo = boot_pair(eo, to)
            row[arm] = {"E": em, "table": tm, "lo": lo, "win": bool(lo > 0)}
        ident[name] = row
        print(
            f"[id] {name:9s} E {row['E']['E']:.3f} state {row['state']['E']:.3f} "
            f"table {row['E']['table']:.3f} state-lo {row['state']['lo']:+.4f}",
            flush=True,
        )
    res["identity"] = ident

    # ---- geometry (published state) ----
    yv = np.asarray(labels["y_value_90"], float)
    yc = np.asarray(labels["y_churn_90"], float)
    rp = np.array([float(truth[k]["return_proneness"]) for k in hold])
    lab_anom = (rp >= np.quantile(rp, 0.9)).astype(int)
    regime = np.array([str(truth[k]["regime0"]) for k in hold], dtype=object)
    geo = {}
    for arm, X in (("table", Xtab), ("E", E), ("state", STATE)):
        Z = z(X)
        i1 = nn_idx(Z, 1)[:, 0]
        km = KMeans(4, n_init=10, random_state=0).fit_predict(Z)
        sc = -IsolationForest(random_state=0).fit(Z).score_samples(Z)
        s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
        geo[arm] = {
            "retrieval": float(spearmanr(yv, yv[i1]).correlation),
            "churn_nn": float(np.mean(yc == yc[i1])),
            "anomaly_auc": float(roc_auc_score(lab_anom, sc)),
            "nmi_regime": float(normalized_mutual_info_score(regime, km)),
            "ari_regime": float(adjusted_rand_score(regime, km)),
            "eff_rank": float((s**2).sum() ** 2 / ((s**4).sum() + 1e-12)),
        }
        print(f"[geo] {arm:5s} { {k: round(v, 3) for k, v in geo[arm].items()} }", flush=True)
    res["geometry"] = geo

    # ---- next-event probe ----
    _B, _A, _keys, seqs, vs2, *_ = AE.load_data(str(data))
    res["next"] = next_probe(model, seqs, hold, vs2, AE.AS_OF_TS)
    print(f"[next] {res['next']}", flush=True)

    (outdir / f"battery_{args.tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    if args.tag == "baseline":
        OUT.write_text(json.dumps(res, indent=1) + "\n")
    # Config comes from the CACHE's run manifest, not from this process. That is
    # what prevents a differently-configured battery from relabelling the card.
    man = F.assert_axes(args.frozen, AE.AXES)
    cfg = {
        k: man.get(k)
        for k in (
            "axes",
            "seed",
            "data",
            "d",
            "layers",
            "e",
            "n_read",
            "E_dim",
            "BLK_dim",
            "state_dim",
            "git_sha",
            "built_at",
            "cache_sha256",
        )
    }
    cfg["hist_dim"] = int(vs.get("hist_dim", 0))
    cfg["manifest"] = str(F.manifest_path(args.frozen))
    (ROOT / "docs/report_cards/experiments/encoder_config.json").write_text(
        json.dumps(cfg, indent=1) + "\n"
    )
    print("WROTE", outdir / f"battery_{args.tag}.json", "and encoder_config.json (from manifest)")


if __name__ == "__main__":
    main()
