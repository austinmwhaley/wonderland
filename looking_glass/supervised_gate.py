"""Supervised plugin gate (Layer C, supervised family).

For a stream, each supervised plugin is a **target** derived from the OBSERVED future (never a
latent): churn / reactivation / LTV / next-purchase / next-order-value / return propensity /
channel-of-next / category affinity / item affinity. The gate fits a head on the frozen `E`
(train customers) and grades it on held-out customers against a STRONG model-free baseline
(majority for classification, mean for regression) and, where available, a known-truth oracle.

Targets are computed from the stream in (as_of, as_of+horizon]; no leakage into `E`.
"""

from __future__ import annotations

import argparse

import numpy as np

from looking_glass.cfm_config import as_of_epoch
from looking_glass.cfm_state import load_frozen_encoder


def _labels(cfg, cust, horizon_days):
    """Per-customer future outcomes computed ONLY from events in (as_of, as_of+horizon]."""
    import duckdb
    import polars as pl

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    p = str(cfg.db)
    src = f"read_parquet('{p}')" if p.endswith((".parquet", ".pq")) else cfg.table
    tc = float(as_of_epoch(cfg.as_of))
    hi = tc + horizon_days * 86400.0
    con.register("_cust", pl.DataFrame({"customer_key": [str(c) for c in cust]}))
    fut = con.execute(
        f"SELECT customer_key, epoch(CAST(event_ts AS TIMESTAMPTZ)) AS t, event_type, "
        f"TRY_CAST(json_extract_string(event_attributes,'$.value') AS DOUBLE) AS v, "
        f"json_extract_string(event_attributes,'$.touchpoint') AS touch, "
        f"json_extract_string(event_attributes,'$.category_code') AS cat, "
        f"entity_id AS eid "
        f"FROM {src} WHERE epoch(CAST(event_ts AS TIMESTAMPTZ)) > {tc} "
        f"AND epoch(CAST(event_ts AS TIMESTAMPTZ)) <= {hi} "
        f"AND customer_key IN (SELECT customer_key FROM _cust)"
    ).pl()
    con.close()
    fut = fut.sort(["customer_key", "t"])
    out = {}
    for g in fut.partition_by("customer_key", maintain_order=True):
        c = g["customer_key"][0]
        pur = g.filter(g["event_type"] == "purchase")
        rec = {
            "next_purchase": 1.0 if pur.height else 0.0,
            "churn": 0.0 if pur.height else 1.0,
            "ltv": float(pur["v"].fill_null(0.0).sum()) if pur.height else 0.0,
            "return": 1.0 if g.filter(g["event_type"] == "return").height else 0.0,
        }
        if pur.height:
            first = pur.sort("t").row(0, named=True)
            rec["next_value"] = float(first["v"] or 0.0)
            rec["channel_next"] = first["touch"] or "none"
            rec["category_next"] = first["cat"] or "none"
            rec["item_next"] = first["eid"] or "none"
        out[c] = rec
    return out


# name -> (kind, metric, baseline)
TARGETS = {
    "churn": ("clf", "auc", "majority"),
    "reactivation": ("clf", "auc", "majority"),  # purchase in horizon
    "next_purchase": ("clf", "auc", "majority"),
    "return": ("clf", "auc", "majority"),
    "ltv": ("reg", "r2", "mean"),
    "next_value": ("reg", "r2", "mean"),
    "channel_next": ("clf", "acc", "majority"),
    "category_next": ("clf", "acc", "majority"),
    "item_next": ("clf", "acc", "majority"),
}


def _states(model, cfg, keys):
    import torch

    from looking_glass.cfm_data import _read_stream, build_sequences
    from looking_glass.cfm_training import forward_embedding

    df = _read_stream(cfg)
    split = {k: "A" for k in keys}
    seqs = build_sequences(df, list(keys), cfg, split, with_anchors=False)
    Z, C = [], []
    for i in range(0, len(seqs), 512):
        with torch.no_grad():
            z = forward_embedding(
                model, seqs[i : i + 512], cfg=cfg, lb=getattr(model, "layer_b", None)
            )
        Z.append(z.detach().float().cpu().numpy())
        C += [s["customer"] for s in seqs[i : i + 512]]
    return np.concatenate(Z, 0), C


def run_supervised(model, cfg, horizon_days=90, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    import looking_glass.cfm_data as _d

    d = _d._read_stream(cfg)
    keys = _d._customer_keys(d, cfg)
    split = _d.assign_split(keys, cfg)
    need = [k for k in keys if split.get(k) in ("A", "B")]
    Z, C = _states(model, cfg, need)
    lab = _labels(cfg, C, horizon_days)
    # reduce the ~2141-d E to a compact subspace so the head fits are fast and stable
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = PCA(n_components=128, random_state=seed).fit_transform(StandardScaler().fit_transform(Z))
    rows = []
    for name, (kind, metric, _base) in TARGETS.items():
        _src = "next_purchase" if name == "reactivation" else name
        yv = np.array([lab.get(c, {}).get(_src, np.nan) for c in C], dtype=object)
        ok = np.array([c in lab and _src in lab[c] for c in C])
        cu = np.array(C)[ok]
        X = Z[ok]
        if kind == "clf":
            classes = sorted({str(v) for v in yv[ok]})
            cmap = {c: i for i, c in enumerate(classes)}
            y = np.array([cmap[str(v)] for v in yv[ok]], dtype=int)
            if len(classes) < 2:
                rows.append((name, "clf", float("nan"), float("nan"), len(classes)))
                continue
        else:
            y = np.array([float(v) for v in yv[ok]], dtype=float)
        nf = max(2, min(5, len(np.unique(cu))))
        aucs, accs, r2s, base = [], [], [], []
        for tr, va in GroupKFold(nf).split(X, groups=cu):
            if kind == "clf":
                m = HistGradientBoostingClassifier(max_iter=200).fit(X[tr], y[tr])
                if metric == "auc":
                    if len(set(y[va])) < 2:
                        continue
                    aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
                else:
                    accs.append(accuracy_score(y[va], m.predict(X[va])))
                    base.append(float(np.bincount(y[tr]).max()) / len(tr))
            else:
                m = HistGradientBoostingRegressor(max_iter=200).fit(X[tr], y[tr])
                r2s.append(r2_score(y[va], m.predict(X[va])))
                base.append(0.0)
        if kind == "clf":
            skill = float(np.mean(aucs)) if metric == "auc" else float(np.mean(accs))
            b = float(np.mean(base)) if base else 0.5
            rows.append((name, metric, round(skill, 4), round(b, 4), len(set(y))))
        else:
            rows.append((name, metric, round(float(np.mean(r2s)), 4), 0.0, -1))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument("--customers", type=int, default=12000)
    a = ap.parse_args(argv)
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    print("# SUPERVISED PLUGIN GATE (held-out, head on E)")
    for name, metric, skill, base, ncl in run_supervised(model, cfg, a.horizon_days):
        print(f"  {name:16s} {metric:4s} skill={skill:+.4f}  baseline={base:.4f}  classes={ncl}")


if __name__ == "__main__":
    main()
