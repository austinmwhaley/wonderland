"""Train-only state-view selection, then one holdout evaluation.

Protocol (pre-registered):
  1. Candidate views are declared in CANDIDATES before any scoring.
  2. Selection uses K-fold CV on TRAINING customers only.
  3. The chosen view per target is frozen.
  4. The holdout is scored ONCE, with a paired bootstrap CI against the table.

Labels for training customers come from the dataset's wide table (they exist for all
customers); the encoder never sees them. No view is chosen using holdout performance.

Writes docs/report_cards/experiments/view_selection.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))

from looking_glass.foundation import FrozenFoundation  # noqa: E402

CANDIDATES = ("shape", "learned_state", "forecast", "measure", "full")
TABLE_BLOCK = "table"
OUT = ROOT / "docs/report_cards/experiments/view_selection.json"


def _labels_all(data: Path, keys):
    """Per-customer labels for ALL customers, from the wide table (evaluation only)."""
    a = pl.read_parquet(data / "path_a/path_a_wide_table.parquet")
    amap = {r["customer_key"]: r for r in a.to_dicts()}
    rows = [amap[k] for k in keys]
    return {c: np.array([r[c] for r in rows], float) for c in a.columns if c.startswith("y_")}


def _table_features(data: Path, keys):
    a = pl.read_parquet(data / "path_a/path_a_wide_table.parquet")
    amap = {r["customer_key"]: r for r in a.to_dicts()}
    rows = [amap[k] for k in keys]
    feat = [
        c for c in a.columns if not c.startswith("y_") and c not in ("customer_key", "signup_ts")
    ]
    x = (
        pl.DataFrame([{c: r[c] for c in feat} for r in rows])
        .to_dummies(
            columns=[
                c for c in ("segment", "region", "acquisition_channel", "loyalty_tier") if c in feat
            ],
            drop_first=True,
        )
        .to_numpy()
        .astype(float)
    )
    return np.nan_to_num(x)


def _score(y, p, kind):
    return float(r2_score(y, p) if kind == "reg" else roc_auc_score(y, p))


def _fit_predict(x, y, kind, tr, te):
    m = (
        HistGradientBoostingRegressor(max_iter=80, random_state=0)
        if kind == "reg"
        else HistGradientBoostingClassifier(max_iter=80, random_state=0)
    )
    m.fit(x[tr], y[tr])
    return m.predict(x[te]) if kind == "reg" else m.predict_proba(x[te])[:, 1]


def _cv(x, y, kind, folds=3):
    out = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(x):
        out[te] = _fit_predict(x, y, kind, tr, te)
    return _score(y, out, kind)


def _boot(y, pa, pb, kind, n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if kind == "clf" and len(np.unique(y[ix])) < 2:
            continue
        d.append(_score(y[ix], pa[ix], kind) - _score(y[ix], pb[ix], kind))
    d = np.asarray(d)
    return {
        "mean": round(float(d.mean()), 4),
        "lo": round(float(np.quantile(d, 0.025)), 4),
        "hi": round(float(np.quantile(d, 0.975)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="/tmp/opencode/artifact_6k")
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    data = Path(args.data)

    art = FrozenFoundation.load(args.artifact)
    raw = pl.read_parquet(data / "customer_event_stream.parquet")
    keys, state, rcpt = art.transform_events(raw)
    pos = {k: i for i, k in enumerate(keys)}

    labels = _labels_all(data, keys)
    table = _table_features(data, keys)
    hold_path = Path("/tmp/opencode/frozen_6k.npz")
    hold = []
    if hold_path.exists():
        z = np.load(hold_path, allow_pickle=True)
        hold = [k for k in z["hold"].tolist() if k in pos]
    if not hold:
        raise SystemExit("no holdout keys available")
    hold_idx = np.array([pos[k] for k in hold])
    train_idx = np.array([i for k, i in pos.items() if k not in set(hold)])

    views = {name: state[name] for name in CANDIDATES if name in state}
    views[TABLE_BLOCK] = table
    result = {
        "protocol": {
            "candidates": list(CANDIDATES),
            "selection": f"{args.folds}-fold CV on TRAINING customers only",
            "evaluation": "holdout scored once after selection",
            "n_train": int(len(train_idx)),
            "n_hold": int(len(hold_idx)),
            "transform": rcpt,
        },
        "targets": {},
    }

    for target, kind in (("value90", "reg"), ("orders90", "reg"), ("churn90", "clf")):
        y = labels["y_" + target.replace("90", "_90")]
        train_scores = {
            v: _cv(x[train_idx], y[train_idx], kind, args.folds) for v, x in views.items()
        }
        best = max((v for v in CANDIDATES if v in views), key=lambda v: train_scores[v])
        x = views[best]
        p_sel = _fit_predict(x, y, kind, train_idx, hold_idx)
        p_tab = _fit_predict(table, y, kind, train_idx, hold_idx)
        entry = {
            "selected_view": best,
            "train_cv": {k: round(v, 4) for k, v in train_scores.items()},
            "holdout": {
                "selected": round(_score(y[hold_idx], p_sel, kind), 4),
                "table": round(_score(y[hold_idx], p_tab, kind), 4),
            },
            "selected_minus_table": _boot(y[hold_idx], p_sel, p_tab, kind),
        }
        result["targets"][target] = entry
        print(target, json.dumps(entry["holdout"]), "selected", best, flush=True)

    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
