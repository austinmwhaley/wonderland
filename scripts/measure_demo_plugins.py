"""Demo plugin measurements: 2 unsupervised probes + known-effect recovery.

Uses the frozen encoder cache.  Writes:
  docs/report_cards/experiments/plugin_unsupervised.json   retrieval (value-neighbour) + anomaly (return-proneness)
  docs/report_cards/experiments/plugin_rl_known.json       known-effect kNN purity for all five randomized action keys

The RL rows are *known-effect recovery* on the synthetic harness (the generator's
randomized optima), not an IPS policy-value claim.  The latter is Layer 1 and is
reported separately by `measure_rl_v2.py` / the research card.
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))


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


def paired(Eok, Tok, n=400, seed=0):
    rng = np.random.default_rng(seed)
    m = len(Eok)
    d = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        d.append(Eok[ix].mean() - Tok[ix].mean())
    d = np.array(d)
    return {
        "E": round(float(Eok.mean()), 4),
        "table": round(float(Tok.mean()), 4),
        "se": round(float(d.std()), 4),
        "lo": round(float(np.quantile(d, 0.025)), 4),
        "hi": round(float(np.quantile(d, 0.975)), 4),
        "n": int(m),
    }


def boot_pair(fnE, fnT, m, n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        d.append(fnE(ix) - fnT(ix))
    d = np.array(d, dtype=float)
    return {
        "se": round(float(np.nanstd(d)), 4),
        "lo": round(float(np.nanquantile(d, 0.025)), 4),
        "hi": round(float(np.nanquantile(d, 0.975)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--frozen", default="/tmp/opencode/frozen_6k")
    args = ap.parse_args()
    data = Path(args.data)
    import frozen

    E, Xtab, labels, hold, seg, _model, _vs, _B, _best, _BLK = frozen.load(args.frozen)
    E = np.nan_to_num(E)
    Xtab = np.nan_to_num(Xtab)
    truth = {c["customer"]: c for c in json.load(open(data / "reference_truth.json"))["customers"]}
    zE, zT = z(E), z(Xtab)
    yv, yc = np.asarray(labels["y_value_90"], float), np.asarray(labels["y_churn_90"], float)
    rp = np.array([float(truth[k]["return_proneness"]) for k in hold])
    lab_anom = (rp >= np.quantile(rp, 0.9)).astype(int)

    regime = np.array([str(truth[k]["regime0"]) for k in hold], dtype=object)
    cat = np.array([str(truth[k]["category_pref"]) for k in hold], dtype=object)
    uns = {}
    fns = {}
    for arm, Z in (("table", zT), ("E", zE)):
        nn1 = nn_idx(Z, 1)[:, 0]
        sc = -IsolationForest(random_state=0).fit(Z).score_samples(Z)
        Dk = cdist(Z, Z)
        np.fill_diagonal(Dk, np.inf)
        knn_sc = np.sort(Dk, axis=1)[:, :10].mean(1)
        km = KMeans(4, n_init=10, random_state=0).fit_predict(Z)
        kmc = KMeans(5, n_init=10, random_state=0).fit_predict(Z)
        s_ = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
        uns[arm] = {
            "retrieval": round(float(spearmanr(yv, yv[nn1]).correlation), 4),
            "anomaly_auc": round(float(roc_auc_score(lab_anom, sc)), 4),
            "anomaly_knn_auc": round(float(roc_auc_score(lab_anom, knn_sc)), 4),
            "churn_nn": round(float(np.mean(yc == yc[nn1])), 4),
            "nmi_regime": round(float(normalized_mutual_info_score(regime, km)), 4),
            "ari_regime": round(float(adjusted_rand_score(regime, km)), 4),
            "nmi_category": round(float(normalized_mutual_info_score(cat, kmc)), 4),
            "eff_rank": round(float((s_**2).sum() ** 2 / ((s_**4).sum() + 1e-12)), 2),
        }

        def make_fns(nn1, sc, knn_sc, km, kmc):
            def _auc(ix):
                if len(np.unique(lab_anom[ix])) < 2:
                    return np.nan
                return float(roc_auc_score(lab_anom[ix], sc[ix]))

            def _knn_auc(ix):
                if len(np.unique(lab_anom[ix])) < 2:
                    return np.nan
                return float(roc_auc_score(lab_anom[ix], knn_sc[ix]))

            def _eff(ix):
                Zs = Z[ix]
                sv = np.linalg.svd(Zs - Zs.mean(0), compute_uv=False)
                return float((sv**2).sum() ** 2 / ((sv**4).sum() + 1e-12))

            return {
                "retrieval": lambda ix: float(spearmanr(yv[ix], yv[nn1[ix]]).correlation),
                "anomaly_auc": _auc,
                "anomaly_knn_auc": _knn_auc,
                "eff_rank": _eff,
                "churn_nn": lambda ix: float(np.mean(yc[ix] == yc[nn1[ix]])),
                "nmi_regime": lambda ix: float(normalized_mutual_info_score(regime[ix], km[ix])),
                "ari_regime": lambda ix: float(adjusted_rand_score(regime[ix], km[ix])),
                "nmi_category": lambda ix: float(normalized_mutual_info_score(cat[ix], kmc[ix])),
            }

        fns[arm] = make_fns(nn1, sc, knn_sc, km, kmc)
        print(f"[uns] {arm:5s} {uns[arm]}", flush=True)

    for metric in fns["E"]:
        if metric == "eff_rank":
            continue
        ci = boot_pair(fns["E"][metric], fns["table"][metric], len(yv))
        uns["E"][metric + "_ci"] = ci
        uns["E"][metric + "_gate"] = bool(ci["lo"] > 0)
        print(
            f"[uns-ci] {metric:12s} E {uns['E'][metric]:.4f} "
            f"table {uns['table'][metric]:.4f} lo {ci['lo']:+.4f} gate {ci['lo'] > 0}",
            flush=True,
        )

    from rabbit_hole.generators.reference import HOUR_BUCKETS

    hb = list(HOUR_BUCKETS)
    keys = {
        "channel": np.array([str(truth[k]["channel_pref"]) for k in hold], dtype=object),
        "offer": np.array([int(truth[k]["offer_opt"]) for k in hold], dtype=object),
        "freq": np.array([int(np.clip(round(float(truth[k]["x0"])), 1, 3)) for k in hold]),
        "hour": np.array(
            [int(np.argmin([abs(float(truth[k]["h0"]) - b) for b in hb])) for k in hold]
        ),
        "dow": np.array([int(np.clip(round(float(truth[k]["d0"])), 0, 6)) for k in hold]),
    }
    rl = {}
    for name, lab in keys.items():
        eo, to = purity(zE, lab, 10), purity(zT, lab, 10)
        r = paired(eo, to)
        r["gate_2se"] = bool(r["lo"] > 0)
        rl[name] = r
        print(
            f"[rl-known] {name:7s} E {r['E']:.3f} table {r['table']:.3f} "
            f"diff {r['E'] - r['table']:+.3f} lo {r['lo']:+.4f} gate {r['gate_2se']}",
            flush=True,
        )

    out = Path(ROOT / "docs/report_cards/experiments")
    out.mkdir(parents=True, exist_ok=True)
    (out / "plugin_unsupervised.json").write_text(json.dumps({"arms": uns}, indent=1) + "\n")
    (out / "plugin_rl_known.json").write_text(json.dumps(rl, indent=1) + "\n")
    print("WROTE", out / "plugin_unsupervised.json")
    print("WROTE", out / "plugin_rl_known.json")


if __name__ == "__main__":
    main()
