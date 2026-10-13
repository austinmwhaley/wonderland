"""Sufficiency gate (v8): is `embedding(customer, t)` a sufficient statistic?

Measures, per target, the RAW-features score vs the EMBEDDING-ALONE score. The
embedding is "sufficient" when E-alone matches (>= raw - tol) the best model-free
feature set — i.e. nothing that matters is lost. NOT sufficient = the embedding is
losing signal (the gap), which downstream would otherwise have to recover from raw.

Run:
  python3 -m looking_glass.sufficiency_gate --db <stream> --ckpt-dir <dir> \
      --tag <t> --as-of <date> [--customers 25000] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from looking_glass.cfm_state import load_frozen_encoder
from looking_glass.head_battery import GBM_CLF, GBM_REG, _b_states, _folds, _rfm_targets


def run_sufficiency(model, cfg, seeds=(0, 1, 2)):
    cust, Z = _b_states(model, cfg)
    fut, past, tc = _rfm_targets(cfg, 90)
    import polars as pl

    j = (
        pl.DataFrame({"customer": cust})
        .join(fut, left_on="customer", right_on="customer_key", how="left")
        .join(past, left_on="customer", right_on="customer_key", how="left")
        .fill_null(0.0)
    )
    rec = (tc - j["last_t"].to_numpy().astype(float)) / 86400.0
    Xraw = np.column_stack(
        [
            rec,
            j["freq"].to_numpy().astype(float),
            j["monetary"].to_numpy().astype(float),
            j["recent30"].to_numpy().astype(float),
            j["recent90"].to_numpy().astype(float),
            j["past_vmean"].to_numpy().astype(float),
            j["past_vlast"].to_numpy().astype(float),
        ]
    ).astype(float)
    yall = {
        "LTV": (j["ltv"].to_numpy().astype(float), "reg"),
        "churn": ((j["orders"].to_numpy().astype(float) == 0).astype(float), "clf"),
        "propensity": ((j["orders"].to_numpy().astype(float) > 0).astype(float), "clf"),
        "order_count": (j["orders"].to_numpy().astype(float), "reg"),
        "next_value": (j["next_v"].to_numpy().astype(float), "reg"),
    }
    rows = []
    for name, (yy, kind) in yall.items():
        raw_s, e_s = [], []
        for seed in seeds:
            fold = np.zeros(len(cust), dtype=int)
            for fi, f in enumerate(_folds(cust, seed=seed)):
                fold[np.isin(cust, f)] = fi
            for fi in range(5):
                tr, va = fold != fi, fold == fi
                if not tr.any() or not va.any():
                    continue
                if kind == "clf" and (len(set(yy[tr])) < 2 or len(set(yy[va])) < 2):
                    continue
                scR = StandardScaler().fit(Xraw[tr])
                scZ = StandardScaler().fit(Z[tr])
                Rtr, Rva = scR.transform(Xraw[tr]), scR.transform(Xraw[va])
                Ztr, Zva = scZ.transform(Z[tr]), scZ.transform(Z[va])
                if kind == "clf":
                    hr = HistGradientBoostingClassifier(**GBM_CLF).fit(Rtr, yy[tr])
                    he = HistGradientBoostingClassifier(**GBM_CLF).fit(Ztr, yy[tr])
                    sr = hr.predict_proba(Rva)[:, 1]
                    se = he.predict_proba(Zva)[:, 1]
                else:
                    hr = HistGradientBoostingRegressor(**GBM_REG).fit(Rtr, yy[tr])
                    he = HistGradientBoostingRegressor(**GBM_REG).fit(Ztr, yy[tr])
                    sr, se = hr.predict(Rva), he.predict(Zva)
                ya = yy[va]
                from sklearn.metrics import r2_score, roc_auc_score

                if kind == "clf":
                    if len(set(ya)) < 2:
                        continue
                    rs, es = roc_auc_score(ya, sr), roc_auc_score(ya, se)
                else:
                    rs, es = r2_score(ya, sr), r2_score(ya, se)
                raw_s.append(rs)
                e_s.append(es)
        rs, es = float(np.nanmean(raw_s)), float(np.nanmean(e_s))
        rows.append(
            {
                "target": name,
                "raw": round(rs, 4),
                "E_alone": round(es, 4),
                "gap": round(rs - es, 4),
                "sufficient": es >= rs - 0.02,
            }
        )
    return rows, len(cust)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    seeds = tuple(int(s) for s in a.seeds.split(","))
    rows, n = run_sufficiency(model, cfg, seeds)
    print(f"# SUFFICIENCY GATE — {a.tag} · {n} held-out · raw=RFM+clock+past-value")
    print("| target | raw | E alone | gap | sufficient |")
    print("|---|---|---|---|---|")
    nok = 0
    for r in rows:
        nok += int(r["sufficient"])
        print(
            f"| {r['target']} | {r['raw']:.4f} | {r['E_alone']:.4f} | {r['gap']:+.4f} | "
            f"{'✅' if r['sufficient'] else '❌'} |"
        )
    print(f"\nSUFFICIENT: {nok}/{len(rows)} targets")


if __name__ == "__main__":
    main()
