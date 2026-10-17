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
import hashlib

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


def _stable_bucket(key) -> int:
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 1


def _conformal_split(pred, y, groups, alpha=0.1):
    """Split by stable customer hash; fit conformal on calib, score on test."""
    from looking_glass.conformal import ConformalRegressor

    pred = np.asarray(pred, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if not (pred.shape[0] == y.shape[0] == groups.shape[0]):
        raise ValueError("pred, y and groups must have the same length")
    calib = np.array([_stable_bucket(g) == 0 for g in groups], dtype=bool)
    test = ~calib
    if not calib.any() or not test.any():
        return float("nan"), float("nan")
    cr = ConformalRegressor().fit(pred[calib], y[calib], alpha)
    lo, hi = cr.interval(pred[test])
    coverage = cr.coverage(pred[test], y[test])
    width = float(np.mean(hi - lo))
    return coverage, width


def _metric_value(pred, y, kind, metric):
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

    if kind == "clf":
        if metric == "auc":
            if len(np.unique(y)) < 2:
                return float("nan")
            return float(roc_auc_score(y, pred))
        return float(accuracy_score(y, pred))
    return float(r2_score(y, pred))


def _bootstrap_skill(pred, y, kind, metric, baseline, n_boot=500, seed=0):
    """Bootstrap 95% CI of held-out skill by resampling customers (NO refit).

    Returns ``(skill, lo, hi, verdict)``; PASS iff the lower bound is strictly
    above the model-free baseline.
    """
    pred = np.asarray(pred, dtype=float).ravel()
    y = np.asarray(y).ravel()
    if pred.shape[0] != y.shape[0]:
        raise ValueError("pred and y must have the same length")
    if pred.size == 0:
        return float("nan"), float("nan"), float("nan"), "FAIL"
    skill = _metric_value(pred, y, kind, metric)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        value = _metric_value(pred[idx], y[idx], kind, metric)
        if np.isfinite(value):
            boot.append(value)
    if boot:
        lo, hi = np.percentile(boot, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    verdict = "PASS" if np.isfinite(lo) and lo > baseline else "FAIL"
    return float(skill), float(lo), float(hi), verdict


def _load_null_model(cfg, null):
    """Load the null-control encoder on the same data window/split as ``cfg``."""
    from looking_glass.cfm_state import load_frozen_encoder

    db, tag, ckpt_dir = null
    model, null_cfg = load_frozen_encoder(tag, ckpt_dir)
    null_cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    null_cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    null_cfg.db = db
    for key in ("as_of", "sample_customers", "split_seed", "split_a_frac"):
        if hasattr(cfg, key):
            setattr(null_cfg, key, getattr(cfg, key))
    return model, null_cfg


class SupervisedGateResult(list):
    """Per-target gate rows plus optional null-control metadata."""

    def __init__(self, rows=(), null_ok=None, null_rows=None):
        super().__init__(rows)
        self.null_ok = null_ok
        self.null_rows = null_rows


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


def _run_supervised(model, cfg, horizon_days=90, seed=0, fast=False):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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
                rows.append(
                    {
                        "name": name,
                        "metric": "clf",
                        "skill": float("nan"),
                        "baseline": float("nan"),
                        "n_classes": len(classes),
                        "coverage90": None,
                        "width90": None,
                        "skill_lo": float("nan"),
                        "skill_hi": float("nan"),
                        "verdict": "FAIL",
                    }
                )
                continue
        else:
            y = np.array([float(v) for v in yv[ok]], dtype=float)
        nf = max(2, min(3 if fast else 5, len(np.unique(cu))))
        oof_pred, oof_y, base = [], [], []
        covs, widths = [], []
        for tr, va in GroupKFold(nf).split(X, groups=cu):
            if kind == "clf":
                m = HistGradientBoostingClassifier(max_iter=(80 if fast else 200)).fit(X[tr], y[tr])
                if metric == "auc":
                    if len(getattr(m, "classes_", ())) < 2:
                        pv = np.zeros(len(va), dtype=float)
                    else:
                        pv = m.predict_proba(X[va])[:, 1]
                else:
                    pv = m.predict(X[va])
                    base.append(float(np.bincount(y[tr]).max()) / len(tr))
            else:
                m = HistGradientBoostingRegressor(max_iter=(80 if fast else 200)).fit(X[tr], y[tr])
                pv = m.predict(X[va])
                base.append(0.0)
                cov, width = _conformal_split(pv, y[va], cu[va], alpha=0.1)
                covs.append(cov)
                widths.append(width)
            oof_pred.append(np.asarray(pv, dtype=float))
            oof_y.append(np.asarray(y[va]))
        if kind == "clf":
            b = float(np.mean(base)) if base else (0.5 if metric == "auc" else float("nan"))
        else:
            b = 0.0
        pred = np.concatenate(oof_pred) if oof_pred else np.array([])
        y_oof = np.concatenate(oof_y) if oof_y else np.array([])
        skill, lo, hi, verdict = _bootstrap_skill(pred, y_oof, kind, metric, b, seed=seed)
        if kind == "clf":
            rows.append(
                {
                    "name": name,
                    "metric": metric,
                    "skill": round(skill, 4),
                    "baseline": round(b, 4),
                    "n_classes": len(set(y)),
                    "coverage90": None,
                    "width90": None,
                    "skill_lo": round(lo, 4),
                    "skill_hi": round(hi, 4),
                    "verdict": verdict,
                }
            )
        else:
            covs = [c for c in covs if np.isfinite(c)]
            widths = [w for w in widths if np.isfinite(w)]
            rows.append(
                {
                    "name": name,
                    "metric": metric,
                    "skill": round(skill, 4),
                    "baseline": 0.0,
                    "n_classes": -1,
                    "coverage90": round(float(np.mean(covs)), 4) if covs else float("nan"),
                    "width90": round(float(np.mean(widths)), 4) if widths else float("nan"),
                    "skill_lo": round(lo, 4),
                    "skill_hi": round(hi, 4),
                    "verdict": verdict,
                }
            )
    return rows


def run_supervised(model, cfg, horizon_days=90, seed=0, null=None, fast=False):
    """Grade supervised targets on held-out customers, with a bootstrap verdict.

    With ``null=(db, tag, ckpt_dir)``, the same targets are also run on a null
    encoder; ``null_ok`` is True iff every null target FAILs (no skill).
    """
    rows = _run_supervised(model, cfg, horizon_days=horizon_days, seed=seed, fast=fast)
    if null is None:
        return SupervisedGateResult(rows)
    null_model, null_cfg = _load_null_model(cfg, null)
    null_rows = _run_supervised(
        null_model, null_cfg, horizon_days=horizon_days, seed=seed, fast=fast
    )
    by_name = {r["name"]: r for r in null_rows}
    null_ok = all(r["verdict"] == "FAIL" for r in null_rows)
    for row in rows:
        nr = by_name.get(row["name"])
        row["null_skill"] = nr["skill"] if nr is not None else None
        row["null_verdict"] = nr["verdict"] if nr is not None else "FAIL"
        row["null_ok"] = null_ok
    return SupervisedGateResult(rows, null_ok=null_ok, null_rows=null_rows)


def _print_rows(rows, title):
    print(title)
    for row in rows:
        extra = ""
        if row["coverage90"] is not None:
            extra = f"  coverage90={row['coverage90']:.4f}  width90={row['width90']:.4f}"
        print(
            f"  {row['name']:16s} {row['metric']:4s} skill={row['skill']:+.4f}  "
            f"[{row['skill_lo']:+.4f}, {row['skill_hi']:+.4f}]  "
            f"baseline={row['baseline']:.4f}  {row['verdict']}  classes={row['n_classes']}{extra}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument("--customers", type=int, default=12000)
    ap.add_argument("--null-db", dest="null_db")
    ap.add_argument("--null-tag", dest="null_tag")
    ap.add_argument("--null-ckpt-dir", dest="null_ckpt_dir")
    a = ap.parse_args(argv)
    null = None
    if any((a.null_db, a.null_tag, a.null_ckpt_dir)):
        if not all((a.null_db, a.null_tag, a.null_ckpt_dir)):
            ap.error("--null-db, --null-tag and --null-ckpt-dir must be given together")
        null = (a.null_db, a.null_tag, a.null_ckpt_dir)
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    result = run_supervised(model, cfg, a.horizon_days, null=null)
    _print_rows(result, "# SUPERVISED PLUGIN GATE (held-out, head on E)")
    if result.null_rows is not None:
        _print_rows(result.null_rows, "# NULL CONTROL (same targets, null encoder)")
        print(f"  null_ok={result.null_ok}")


if __name__ == "__main__":
    main()
