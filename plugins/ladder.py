"""Training-size ladder for a supervised target.

"How much should this plugin train?" — fit the target's primary head on
increasing amounts of *grouped* training data against a FIXED held-out split and
find where performance plateaus (and whether it falls off). The noise floor is
MEASURED (fold-to-fold SE of the held-out metric), so the plateau rule is
data-derived: improvements smaller than 2 SE are noise, a drop larger than 2 SE
is a falloff.

Receipt -> plugins/artifacts/ladder_<target_tag>[_<as_of>].json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .head_template import HeadTemplate
from .targets import REGISTRY, Target

WORK = Path(__file__).resolve().parents[1]
OUT = WORK / "plugins" / "artifacts"
RUNGS = (250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000)
FOLDS = 5


def _metric(kind: str):
    if kind == "binary":
        from sklearn.metrics import roc_auc_score

        def f(pred, y):
            return float(roc_auc_score(y, pred)) if len(np.unique(y)) > 1 else float("nan")

        return f
    from scipy.stats import spearmanr

    return lambda pred, y: float(spearmanr(pred, y).statistic)


def _fold_se(pred, y, keys_te, metric, folds=FOLDS):
    """Measured noise: SE of the held-out metric across grouped folds."""
    uniq = np.unique(keys_te)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    groups = np.array_split(uniq, folds)
    vals = []
    for g in groups:
        m = np.isin(keys_te, g)
        if m.sum() == 0:
            continue
        v = metric(pred[m], y[m])
        if np.isfinite(v):
            vals.append(v)
    return float(np.std(vals) / np.sqrt(max(len(vals), 1))) if vals else float("nan")


def _load(target: Target, as_of):
    from .base import load_dataset

    if as_of is None and target.kind == "continuous":
        return load_dataset(target.window_days)
    return load_dataset(target.window_days, target=target, as_of=as_of)


def run(target: Target, as_of: str | None = None, seed: int = 0, rungs=RUNGS):
    ds = _load(target, as_of)
    tpl = HeadTemplate(target)
    metric = _metric(target.kind)
    rungs = [r for r in (*rungs, None) if r is None or r < len(ds.y)]

    print(f"== TRAINING-SIZE LADDER {target.name} ({target.kind}) ==")
    print(f"rows available {len(ds.y)}  customers {ds.meta['n_customers']}  as_of {as_of}")
    print(f"{'rung':>8s} {'n_train':>8s} {'metric':>9s} {' ±2se':>8s}  status")

    rows = []
    for r in rungs:
        # probe the reference family only — the full bake-off runs in
        # run_target; the ladder answers "how much data", not "which model"
        heads, _ = tpl.fit(ds, seed=seed, max_train=r, families=("logistic",))
        h = next(x for x in heads if x["name"] == tpl.primary_head)
        keys_te = np.asarray(ds.keys)[h["idx"]]
        val = metric(h["pred"], ds.y[h["idx"]])
        se = _fold_se(h["pred"], ds.y[h["idx"]], keys_te, metric)
        rows.append(
            {
                "max_train": r,
                "n_train": int(h["n_train"]),
                "metric": val,
                "se": se,
            }
        )
        print(
            f"{str(r):>8s} {h['n_train']:>8d} {val:>9.4f} {2 * se if np.isfinite(se) else float('nan'):>8.4f}"
        )

    finite = [x for x in rows if np.isfinite(x["metric"])]
    if not finite:
        raise ValueError("ladder produced no finite metric")
    tol = 2.0 * float(np.nanmax([x["se"] for x in finite]))
    best = max(x["metric"] for x in finite)
    chosen = next(x for x in finite if x["metric"] >= best - tol)
    # falloff: a clear drop (> tol) below the running best at a LARGER rung
    run_best, falloff = -np.inf, None
    for x in finite:
        if x["metric"] < run_best - tol:
            falloff = x
            break
        run_best = max(run_best, x["metric"])

    print(f"measured noise 2*SE = {tol:.4f}")
    print(f"best metric {best:.4f}")
    if falloff is not None:
        print(
            f"FALLOFF at n_train={falloff['n_train']} (metric {falloff['metric']:.4f} "
            f"< running best - tol) -> more data is hurting"
        )
    else:
        print("no falloff: metric is flat or improving to the largest rung")
    print(
        f"CHOSEN rung: {chosen['max_train']} (n_train={chosen['n_train']}, metric {chosen['metric']:.4f})"
    )

    receipt = {
        "target": target.name,
        "kind": target.kind,
        "as_of": as_of,
        "rows_available": int(len(ds.y)),
        "customers": int(ds.meta["n_customers"]),
        "encoder_version": ds.meta.get("encoder_version"),
        "rungs": rows,
        "tol": tol,
        "best": best,
        "chosen_max_train": chosen["max_train"],
        "chosen_n_train": chosen["n_train"],
        "falloff_n_train": None if falloff is None else falloff["n_train"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{as_of}" if as_of else "_full"
    path = OUT / f"ladder_{target.tag}{suffix}.json"
    path.write_text(json.dumps(receipt, indent=1, default=float))
    print(f"receipt -> {path}")
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Training-size ladder for a supervised target")
    ap.add_argument("target", choices=sorted(REGISTRY))
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(REGISTRY[a.target], as_of=a.as_of, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
