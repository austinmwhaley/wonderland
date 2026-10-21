"""PATH A dataset gate: fail the build if the traditional supervised view is not usable.

Checks (critical unless marked informational):
  * schema/keys, label nulls, join integrity A<->B
  * no constant features; no near-duplicate features (|corr| > 0.999)
  * point-in-time leakage: independently recompute features/labels from PATH B
  * label balance (minority >= 20 and 2%..50%)
  * feature sanity (counts >= 0, rates in [0,1], net-value identity; return_value is signed)
  * signal: GBM CV beats the mean/prior, and the shuffled-label null does not
  * temporal stability: train on older signup cohorts, test on newer (informational)
  * PATH B action balance: every action key has >= 2 observed levels

Writes <dir>/path_a_validation.json. Exit 1 on any critical failure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold


def _json_attrs(s):
    return json.loads(s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    man = json.loads((d / "manifest.json").read_text())
    as_of = datetime.fromisoformat(man["as_of"])
    A = pl.read_parquet(d / "path_a/path_a_wide_table.parquet")
    B = pl.read_parquet(d / "path_b/path_b_customer_event_stream.parquet")
    assert A.columns[-10:] == [
        "y_orders_30",
        "y_value_30",
        "y_margin_30",
        "y_next_days_30",
        "y_churn_30",
        "y_orders_90",
        "y_value_90",
        "y_margin_90",
        "y_next_days_90",
        "y_churn_90",
    ], A.columns
    feat = [
        c for c in A.columns if not c.startswith("y_") and c not in ("customer_key", "signup_ts")
    ]
    labels = [c for c in A.columns if c.startswith("y_")]
    cats = [c for c in ("segment", "region", "acquisition_channel", "loyalty_tier") if c in feat]
    X = A.select(feat).to_dummies(columns=cats, drop_first=True).to_numpy().astype(float)
    yc = np.asarray(A["y_churn_90"], float)
    yv = np.asarray(A["y_value_90"], float)
    failures: list[str] = []
    report: dict = {"shape": list(A.shape), "features": feat, "labels": labels}

    # 1. schema / keys / labels
    if A["customer_key"].n_unique() != A.height:
        failures.append("customer_key is not unique")
    report["label_nulls"] = {c: int(A[c].null_count()) for c in labels}
    if any(v for v in report["label_nulls"].values()):
        failures.append("label nulls")
    if set(A["customer_key"]) != set(B["customer_key"]):
        failures.append("A/B customer_key sets differ")
    report["join_ok"] = set(A["customer_key"]) == set(B["customer_key"])

    # 2. constants / near-duplicates
    report["constant_features"] = [c for c in feat if A[c].n_unique() <= 1]
    if report["constant_features"]:
        failures.append(f"constant features: {report['constant_features']}")
    num_feat = [c for c in feat if A[c].dtype.is_numeric()]
    corr = A.select([pl.col(c).cast(pl.Float64) for c in num_feat]).corr().to_numpy()
    hi = []
    for i in range(len(num_feat)):
        for j in range(i + 1, len(num_feat)):
            if abs(corr[i, j]) > 0.999:
                hi.append((num_feat[i], num_feat[j], round(float(corr[i, j]), 4)))
    report["near_duplicate_pairs"] = hi
    if hi:
        failures.append(f"near-duplicate features: {hi}")

    # 3. point-in-time recompute from PATH B
    sample = A["customer_key"].to_list()[:25]
    rows = B.filter(pl.col("customer_key").is_in(sample)).to_dicts()
    by_cust: dict[str, list[dict]] = {}
    for r in rows:
        by_cust.setdefault(r["customer_key"], []).append(r)
    rec = []
    for k in sample:
        hist = [r for r in by_cust[k] if datetime.fromisoformat(r["event_ts"]) <= as_of]
        fut = [
            r
            for r in by_cust[k]
            if as_of < datetime.fromisoformat(r["event_ts"]) <= as_of + timedelta(days=30)
        ]
        n_orders = sum(1 for r in hist if r["event_type"] == "purchase")
        pv = sum(
            float(_json_attrs(r["event_attributes"]).get("value", 0) or 0)
            for r in hist
            if r["event_type"] == "purchase"
        )
        f_orders = sum(1 for r in fut if r["event_type"] == "purchase")
        fv = sum(
            float(_json_attrs(r["event_attributes"]).get("value", 0) or 0)
            for r in fut
            if r["event_type"] == "purchase"
        )
        ar = A.filter(pl.col("customer_key") == k).row(0, named=True)
        rec.append(
            {
                "n_orders": (n_orders, ar["n_orders"]),
                "purchase_value": (round(pv, 2), round(ar["purchase_value"], 2)),
                "y_orders_30": (f_orders, ar["y_orders_30"]),
                "y_value_30": (round(fv, 2), round(ar["y_value_30"], 2)),
            }
        )
    report["recompute_ok"] = {
        "n_orders": all(r["n_orders"][0] == r["n_orders"][1] for r in rec),
        "purchase_value": all(
            abs(r["purchase_value"][0] - r["purchase_value"][1]) < 0.02 for r in rec
        ),
        "y_orders_30": all(r["y_orders_30"][0] == r["y_orders_30"][1] for r in rec),
        "y_value_30": all(abs(r["y_value_30"][0] - r["y_value_30"][1]) < 0.02 for r in rec),
    }
    if not all(report["recompute_ok"].values()):
        failures.append(f"point-in-time recompute mismatch: {report['recompute_ok']}")

    # 4. label balance
    pos = int((yc == 1).sum())
    rate = float(yc.mean())
    report["label_balance"] = {"churn90_positive": pos, "churn90_rate": round(rate, 4)}
    if not (pos >= 20 and 0.02 <= rate <= 0.50):
        failures.append(f"churn90 label degenerate: pos={pos} rate={rate:.3f}")

    # 5. feature sanity
    bad = {}
    for c in (
        "recency_days",
        "tenure_days",
        "days_since_last_purchase",
        "n_orders",
        "purchase_value",
    ):
        if c in A and float(A[c].min()) < 0:
            bad[c] = float(A[c].min())
    for c in ("return_rate", "discount_mean"):
        if c in A and not (0.0 <= float(A[c].min()) and float(A[c].max()) <= 1.0):
            bad[c] = [float(A[c].min()), float(A[c].max())]
    if "return_value" in A and float(A["return_value"].max()) > 1e-9:
        bad["return_value"] = "expected signed (<= 0)"
    if "net_value" in A:
        err = float((A["net_value"] - (A["purchase_value"] + A["return_value"])).abs().max())
        report["net_value_identity_maxerr"] = err
        if err > 0.01:
            bad["net_value_identity"] = err
    report["bad_ranges"] = bad
    if bad:
        failures.append(f"feature sanity: {bad}")

    # 6. signal + null
    def cv(y, kind):
        y = np.asarray(y, float)
        oof = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=a.seed).split(X):
            if kind == "clf":
                m = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[tr], y[tr])
                oof[te] = m.predict_proba(X[te])[:, 1]
            else:
                m = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(X[tr], y[tr])
                oof[te] = m.predict(X[te])
        return oof

    pc = cv(yc, "clf")
    pv = cv(yv, "reg")
    rng = np.random.default_rng(a.seed)
    pc_null = cv(rng.permutation(yc), "clf")
    pv_null = cv(rng.permutation(yv), "reg")
    report["baseline"] = {
        "churn90_auc": round(roc_auc_score(yc, pc), 4),
        "churn90_ap": round(average_precision_score(yc, pc), 4),
        "churn90_brier": round(brier_score_loss(yc, pc), 4),
        "churn90_null_auc": round(roc_auc_score(yc, pc_null), 4),
        "value90_r2": round(r2_score(yv, pv), 4),
        "value90_mae": round(mean_absolute_error(yv, pv), 2),
        "value90_null_r2": round(r2_score(yv, pv_null), 4),
    }
    if report["baseline"]["churn90_auc"] < 0.60:
        failures.append(f"churn signal too weak: AUC {report['baseline']['churn90_auc']}")
    if report["baseline"]["value90_r2"] < 0.05:
        failures.append(f"value signal too weak: R2 {report['baseline']['value90_r2']}")
    if report["baseline"]["churn90_null_auc"] > 0.60:
        failures.append(
            f"shuffled churn null not near chance: {report['baseline']['churn90_null_auc']}"
        )
    if report["baseline"]["value90_null_r2"] > 0.10:
        failures.append(
            f"shuffled value null not near chance: {report['baseline']['value90_null_r2']}"
        )

    # 7. temporal stability (informational)
    order = np.argsort(A["signup_ts"].to_list())
    cut = int(0.7 * len(order))
    tr, te = order[:cut], order[cut:]
    mc = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(X[tr], yc[tr])
    mr = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(X[tr], yv[tr])
    report["temporal_split"] = {
        "churn90_auc": round(roc_auc_score(yc[te], mc.predict_proba(X[te])[:, 1]), 4),
        "value90_r2": round(r2_score(yv[te], mr.predict(X[te])), 4),
    }

    # 8. PATH B action balance (sends)
    sends = B.filter(pl.col("event_type") == "email_send").sample(
        n=min(20000, B.filter(pl.col("event_type") == "email_send").height), seed=a.seed
    )
    levels: dict[str, set] = {}
    for r in sends.to_dicts():
        at = _json_attrs(r["event_attributes"])
        k = at.get("target_key")
        if k:
            levels.setdefault(str(k), set()).add(at.get(k))
    report["action_levels"] = {k: sorted(map(str, v)) for k, v in levels.items()}
    bad_actions = {k: sorted(v) for k, v in levels.items() if len(v) < 2}
    report["action_levels_missing"] = bad_actions
    if bad_actions:
        failures.append(f"action keys with <2 observed levels: {bad_actions}")

    report["failures"] = failures
    report["ok"] = not failures
    (d / "path_a_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"ok": report["ok"], "failures": failures, "baseline": report["baseline"]}, indent=1
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
