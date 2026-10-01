"""Supervised head TEMPLATE — one code path, parameterised by Target.

A supervised plugin owns only its `Target` (what to predict: label kind, window,
metric/gate contract). Everything else — dataset assembly, grouped train/held-out
split, fitting, metrics, gate rows, artifact, and the persisted head used for
inference — comes from this template. Adding a supervised plugin = adding a
Target in `plugins/targets.py`.

Kinds:
  * continuous -> heads point / two_part / quantile / trailing baseline
                  (rank agreement with realized target; the CLV contract)
  * binary     -> head logistic / trailing baseline
                  (AUC, PR-AUC, top-decile lift, calibration)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .base import Dataset, gate, save_artifact
from .targets import Target


# ---------------------------------------------------------------------------
# split / helpers
# ---------------------------------------------------------------------------
def _split(keys, seed=0, test=0.3, max_train=None):
    gss = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(keys)), groups=keys))
    if max_train is not None and len(tr) > max_train:
        tr = _subsample_grouped(np.asarray(tr), keys, int(max_train), seed)
    return tr, te


def _subsample_grouped(tr, keys, max_rows, seed):
    """Deterministic grouped subsample of TRAIN rows to >= max_rows.

    Customers are shuffled (seeded) and taken whole until the row budget is
    met — a customer is never split, so there is no within-customer leakage
    between the subsample and the held-out set.
    """
    rng = np.random.default_rng(seed)
    cust = np.asarray(keys)[tr]
    uniq = np.unique(cust)
    rng.shuffle(uniq)
    keep, n = [], 0
    for c in uniq:
        rows = tr[cust == c]
        keep.append(rows)
        n += len(rows)
        if n >= max_rows:
            break
    return np.sort(np.concatenate(keep))


def _metrics(pred, y):
    """Continuous contract (unchanged): rank agreement / MAE / decile capture."""
    from sklearn.metrics import mean_absolute_error

    sp = float(spearmanr(pred, y).statistic)
    k = max(1, int(0.1 * len(pred)))
    top = np.argsort(pred)[-k:]
    capture = float(y[top].mean() / max(y.mean(), 1e-9))
    return {
        "spearman": sp,
        "mae": float(mean_absolute_error(y, pred)),
        "top_decile_capture": capture,
    }


def _metrics_binary(pred, y):
    """Binary contract: discrimination, ranking capture, calibration."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(y)
    pred = np.asarray(pred, dtype=np.float64)
    base = float(y.mean())
    both = len(np.unique(y)) > 1
    auc = float(roc_auc_score(y, pred)) if both else float("nan")
    ap = float(average_precision_score(y, pred)) if both else float("nan")
    brier = float(brier_score_loss(y, pred)) if both else float("nan")
    k = max(1, int(0.1 * len(pred)))
    top = np.argsort(pred)[-k:]
    lift = float(y[top].mean() / max(base, 1e-9))
    return {
        "auc": auc,
        "pr_auc": ap,
        "brier": brier,
        "base_rate": base,
        "top_decile_lift": lift,
        "calibration_gap": float(abs(pred.mean() - base)),
    }


def _ridge_alpha(X):
    """Derived regularisation: scaled to the feature Gram scale (no magic)."""
    g = float(np.mean(np.sum(X * X, axis=0)))
    return max(1e-6, 1e-3 * g)


# ---------------------------------------------------------------------------
# continuous heads (CLV contract — behavior preserved exactly)
# ---------------------------------------------------------------------------
def head_point(ds: Dataset, seed=0, max_train=None):
    tr, te = _split(ds.keys, seed, max_train=max_train)
    sc = StandardScaler().fit(ds.X[tr])
    m = Ridge(alpha=_ridge_alpha(sc.transform(ds.X[tr]))).fit(sc.transform(ds.X[tr]), ds.y[tr])
    pred = m.predict(sc.transform(ds.X[te]))
    return {
        "name": "point",
        "pred": pred,
        "idx": te,
        "n_train": int(len(tr)),
        "metrics": _metrics(pred, ds.y[te]),
        "fitted": {"kind": "continuous_point", "scaler": sc, "model": m},
    }


def head_two_part(ds: Dataset, seed=0, max_train=None):
    tr, te = _split(ds.keys, seed, max_train=max_train)
    sc = StandardScaler().fit(ds.X[tr])
    Xtr, Xte = sc.transform(ds.X[tr]), sc.transform(ds.X[te])
    active = ds.y[tr] > 0
    clf = LogisticRegression(max_iter=2000).fit(Xtr, active.astype(int))
    p = clf.predict_proba(Xte)[:, 1]
    reg = Ridge(alpha=_ridge_alpha(Xtr[active])).fit(Xtr[active], ds.y[tr][active])
    pred = p * np.maximum(reg.predict(Xte), 0.0)
    return {
        "name": "two_part",
        "pred": pred,
        "idx": te,
        "n_train": int(len(tr)),
        "metrics": _metrics(pred, ds.y[te]),
        "fitted": {"kind": "continuous_two_part", "scaler": sc, "clf": clf, "reg": reg},
    }


def head_quantile(ds: Dataset, seed=0, max_train=None):
    tr, te = _split(ds.keys, seed, max_train=max_train)
    qs = (0.1, 0.5, 0.9)
    out = {}
    for q in qs:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, random_state=seed
        )
        m.fit(ds.X[tr], ds.y[tr])
        out[q] = m.predict(ds.X[te])
    y = ds.y[te]
    stack = np.sort(np.stack([out[0.1], out[0.5], out[0.9]], axis=0), axis=0)
    lo, med, hi = stack[0], stack[1], stack[2]
    cover = float(np.mean((y >= lo) & (y <= hi)))
    mono = bool(np.all(lo <= med + 1e-6) and np.all(med <= hi + 1e-6))
    return {
        "name": "quantile",
        "idx": te,
        "n_train": int(len(tr)),
        "coverage_80": cover,
        "monotone": mono,
        "pred": med,
        "lo": lo,
        "hi": hi,
        "metrics": _metrics(med, y),
    }


def head_baseline(ds: Dataset, seed=0, max_train=None):
    tr, te = _split(ds.keys, seed, max_train=max_train)
    return {
        "name": "trailing_baseline",
        "pred": ds.x_base[te],
        "idx": te,
        "n_train": int(len(tr)),
        "metrics": _metrics(ds.x_base[te], ds.y[te]),
    }


# ---------------------------------------------------------------------------
# binary heads (purchase propensity contract)
# ---------------------------------------------------------------------------
# Binary model roster: the template fits ALL families on the same split and
# lets held-out evidence pick the winner (no a-priori "NNs are best" claim).
BINARY_FAMILIES = ("logistic", "mlp", "hgb")


def _binary_family(name: str, seed: int, dim: int):
    """Estimator family for a binary target.

    The MLP trains to CONVERGENCE (early stop on validation loss); its
    ``max_iter`` is only a budget cap (AGENTS #2). Width is derived from the
    input dim (no hand-tuned architecture).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    if name == "logistic":
        return LogisticRegression(max_iter=2000)
    if name == "mlp":
        h1 = int(np.clip(dim, 64, 256))
        return MLPClassifier(
            hidden_layer_sizes=(h1, h1 // 2),
            activation="relu",
            early_stopping=True,
            n_iter_no_change=15,
            max_iter=400,  # budget cap; early stopping decides convergence
            random_state=seed,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, early_stopping=True, validation_fraction=0.1, random_state=seed
        )
    raise ValueError(f"unknown binary family: {name!r}")


def head_binary(ds: Dataset, seed=0, max_train=None, families=BINARY_FAMILIES):
    """Fit every family in the roster on one shared split (+ trailing baseline).

    The caller (HeadTemplate.fit) selects the winner by held-out AUC.
    """
    tr, te = _split(ds.keys, seed, max_train=max_train)
    sc = StandardScaler().fit(ds.X[tr])
    Xtr, Xte = sc.transform(ds.X[tr]), sc.transform(ds.X[te])
    heads = []
    for name in families:
        m = _binary_family(name, seed, ds.X.shape[1])
        m.fit(Xtr, ds.y[tr].astype(int))
        pred = m.predict_proba(Xte)[:, 1]
        heads.append(
            {
                "name": name,
                "pred": pred,
                "idx": te,
                "n_train": int(len(tr)),
                "metrics": _metrics_binary(pred, ds.y[te]),
                "fitted": {"kind": "binary", "family": name, "scaler": sc, "model": m},
            }
        )
    heads.append(
        {
            "name": "trailing_baseline",
            "pred": ds.x_base[te],
            "idx": te,
            "n_train": int(len(tr)),
            "metrics": _metrics_binary(ds.x_base[te], ds.y[te]),
        }
    )
    return heads


def head_logistic(ds: Dataset, seed=0, max_train=None):
    """Reference family only (stable import surface; the ladder's probe)."""
    return head_binary(ds, seed, max_train, families=("logistic",))[0]


def head_baseline_binary(ds: Dataset, seed=0, max_train=None):
    tr, te = _split(ds.keys, seed, max_train=max_train)
    return {
        "name": "trailing_baseline",
        "pred": ds.x_base[te],
        "idx": te,
        "n_train": int(len(tr)),
        "metrics": _metrics_binary(ds.x_base[te], ds.y[te]),
    }


HEADS = {
    "continuous": (head_point, head_two_part, head_quantile, head_baseline),
    "binary": (),  # goes through head_binary (roster bake-off) — see fit()
}
DEFAULT_PRIMARY = {"continuous": "two_part"}


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------
class HeadTemplate:
    """Fit → measure → gate → persist, driven entirely by the Target's kind."""

    def __init__(self, target: Target):
        self.target = target
        self._winner: str | None = None  # set by fit() for binary (bake-off)

    @property
    def primary_head(self) -> str | None:
        if self.target.primary_head:
            return self.target.primary_head
        # binary: the bake-off winner (set by fit); continuous: fixed default
        return self._winner or DEFAULT_PRIMARY.get(self.target.kind)

    def fit(self, ds: Dataset, seed: int = 0, max_train: int | None = None, families=None):
        """Fit the target's heads. Binary targets run the MODEL BAKE-OFF: every
        family in the roster on the same split, winner = best held-out AUC
        (evidence, not opinion). `families` narrows the roster (the ladder
        probes a single family)."""
        if self.target.kind == "binary":
            heads = head_binary(
                ds, seed, max_train, families=tuple(families) if families else BINARY_FAMILIES
            )
            cands = [h for h in heads if "fitted" in h]

            def _auc(h):
                v = h["metrics"]["auc"]
                return v if np.isfinite(v) else -np.inf

            win = max(cands, key=_auc)
            self._winner = win["name"]
            return heads, win["fitted"]
        heads = [fn(ds, seed, max_train) for fn in HEADS[self.target.kind]]
        self._winner = None
        fitted = next((h["fitted"] for h in heads if h["name"] == self.primary_head), None)
        return heads, fitted

    def gate_rows(self, heads):
        if self.target.kind == "binary":
            return self._binary_rows(heads)
        return self._continuous_rows(heads)

    @staticmethod
    def predict(fitted, X):
        """Score embeddings with a persisted head (the inference contract)."""
        Xs = fitted["scaler"].transform(np.asarray(X, dtype=np.float32))
        if fitted["kind"] == "binary":
            return fitted["model"].predict_proba(Xs)[:, 1]
        if fitted["kind"] == "continuous_two_part":
            p = fitted["clf"].predict_proba(Xs)[:, 1]
            return p * np.maximum(fitted["reg"].predict(Xs), 0.0)
        if fitted["kind"] == "continuous_point":
            return fitted["model"].predict(Xs)
        raise ValueError(f"unknown fitted head kind: {fitted['kind']}")

    def persist(self, fitted, path):
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, path)
        return path

    @staticmethod
    def load(path):
        import joblib

        return joblib.load(path)

    # -- gate contracts ----------------------------------------------------
    def _continuous_rows(self, heads):
        rows = []
        for h in heads:
            m = h["metrics"]
            if h["name"] == "quantile":
                rows.append(
                    {
                        "check": "quantile: 80% coverage in [0.6,1.0]",
                        "achieved": round(h["coverage_80"], 3),
                        "ok": 0.6 <= h["coverage_80"] <= 1.0,
                    }
                )
                rows.append(
                    {
                        "check": "quantile: monotone q10<=q50<=q90",
                        "achieved": h["monotone"],
                        "ok": h["monotone"],
                    }
                )
            elif h["name"] == "trailing_baseline":
                rows.append(
                    {
                        "check": "trailing_baseline: finite spearman (sanity)",
                        "achieved": round(m["spearman"], 3),
                        "ok": np.isfinite(m["spearman"]),
                    }
                )
            else:
                rows.append(
                    {
                        "check": f"{h['name']}: spearman > 0.1",
                        "achieved": round(m["spearman"], 3),
                        "ok": np.isfinite(m["spearman"]) and m["spearman"] > 0.1,
                    }
                )
        basesp = [h for h in heads if h["name"] == "trailing_baseline"][0]["metrics"]["spearman"]
        finite = [
            h["metrics"]["spearman"]
            for h in heads
            if h["name"] not in ("trailing_baseline", "quantile")
            and np.isfinite(h["metrics"]["spearman"])
        ]
        bestsp = max(finite) if finite else 0.0
        rows.append(
            {
                "check": "embedding head >= trailing baseline (spearman)",
                "achieved": f"{bestsp:.3f} vs {basesp:.3f}",
                "ok": bestsp >= (basesp if np.isfinite(basesp) else 0.0) - 0.05,
            }
        )
        return rows

    def _binary_rows(self, heads):
        model = next(
            (h for h in heads if h["name"] == self._winner)
            or (h for h in heads if h["name"] != "trailing_baseline"),
            None,
        )
        base = next((h for h in heads if h["name"] == "trailing_baseline"), None)
        if model is None or base is None:
            raise ValueError("binary contract requires a model head and a trailing baseline")
        m, b = model["metrics"], base["metrics"]
        rows = [
            {
                "check": "binary: finite metrics (both classes present)",
                "achieved": round(m["auc"], 3) if np.isfinite(m["auc"]) else "nan",
                "ok": np.isfinite(m["auc"])
                and np.isfinite(m["pr_auc"])
                and 0.0 < m["base_rate"] < 1.0,
            },
            {
                "check": "binary: top-decile lift > 1.0 (better than random)",
                "achieved": round(m["top_decile_lift"], 3),
                "ok": m["top_decile_lift"] > 1.0,
            },
            {
                "check": f"binary: {model['name']} AUC >= trailing baseline AUC",
                "achieved": f"{m['auc']:.3f} vs {b['auc']:.3f}",
                "ok": np.isfinite(m["auc"])
                and m["auc"] >= (b["auc"] if np.isfinite(b["auc"]) else 0.0),
            },
            {
                # documented conservative tolerance: mean predicted probability
                # must track the realized positive rate within 5 points
                "check": "binary: calibration gap <= 0.05",
                "achieved": round(m["calibration_gap"], 3),
                "ok": m["calibration_gap"] <= 0.05,
            },
        ]
        return rows


def run_target(
    target: Target,
    seed: int = 0,
    as_of: str | None = None,
    max_train: int | None = None,
    cfm_products=None,
    stream_db=None,
    out_dir=None,
):
    """Train/gate/persist one supervised Target. Plugins differ only by Target.

    ``stream_db`` / ``out_dir`` default to the canonical rabbit_hole stream and
    ``plugins/artifacts``; pass them to run a fixture (e.g. the Instacart
    stream) without touching certified receipts.
    """
    from . import base
    from .base import load_dataset

    kw = {}
    if cfm_products is not None:
        kw["cfm_products"] = cfm_products
    if stream_db is not None:
        kw["stream_db"] = stream_db
    if as_of is None and target.kind == "continuous":
        # legacy shape kept for the classic loader path (identical behavior)
        ds = load_dataset(target.window_days, **kw)
    else:
        ds = load_dataset(target.window_days, target=target, as_of=as_of, **kw)
    if ds.meta.get("encoder_version") is None:
        # a mixed-version products table would train a null-pinned head that
        # PASSES the gate and only fails later at inference — fail here instead
        raise ValueError(
            "products table holds multiple encoder versions (encoder_version=None); "
            "rebuild products so the head pins exactly one encoder"
        )

    tpl = HeadTemplate(target)
    families = (target.family,) if target.family else None
    heads, fitted = tpl.fit(ds, seed=seed, max_train=max_train, families=families)
    if target.kind == "binary":
        fam = {h["name"]: h["metrics"]["auc"] for h in heads if "fitted" in h}
        print(
            "  model bake-off (held-out AUC): "
            + ", ".join(f"{k}={v:.4f}" for k, v in fam.items())
            + f"  -> winner: {tpl.primary_head}"
        )
    rows = tpl.gate_rows(heads)
    ok = gate(rows, f"SUPERVISED {target.kind.upper()} PLUGIN {target.name}")

    out = Path(out_dir) if out_dir is not None else base.OUT
    # as-of-stamped head file: the manifest still points at the current head,
    # but each cycle's trained head survives its successor (evidence retention)
    head_stem = f"{target.tag}_{as_of}" if as_of else target.tag
    head_path = out / "heads" / f"{head_stem}.joblib"
    tpl.persist(fitted, head_path)

    payload = {
        "dataset": ds.meta,
        "seed": seed,
        "heads": [
            {
                "name": h["name"],
                "metrics": h["metrics"],
                **(
                    {"coverage_80": h.get("coverage_80"), "monotone": h.get("monotone")}
                    if h["name"] == "quantile"
                    else {}
                ),
            }
            for h in heads
        ],
        "target": {
            "name": target.name,
            "kind": target.kind,
            "window_days": target.window_days,
            "spec_target": target.spec_target,
            "primary_head": tpl.primary_head,
            "family_pin": target.family or "bake-off",
        },
        "encoder_version": ds.meta.get("encoder_version"),
        "feature_table": ds.meta.get("feature_table", target.feature_table),
        "as_of": as_of,
        "max_train": max_train,
        "head_path": str(head_path),
        "head_name": tpl.primary_head,
        "verdict": bool(ok),
    }
    save_artifact(target.spec(), payload, out=out)
    return ok, payload


def main(argv=None):
    import argparse

    from .targets import REGISTRY

    ap = argparse.ArgumentParser(description="Run one supervised target")
    ap.add_argument("target", choices=sorted(REGISTRY))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--as-of", default=None, help="ISO date: labels closed at this date")
    ap.add_argument("--products", default=None, help="cfm_products.duckdb path (fixture override)")
    ap.add_argument(
        "--stream", default=None, help="canonical stream duckdb path (fixture override)"
    )
    ap.add_argument("--out-dir", default=None, help="artifact dir (default: plugins/artifacts)")
    a = ap.parse_args(argv)
    ok, payload = run_target(
        REGISTRY[a.target],
        seed=a.seed,
        as_of=a.as_of,
        cfm_products=a.products,
        stream_db=a.stream,
        out_dir=a.out_dir,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
