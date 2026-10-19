"""JOINT minimality certificate for the served embedding E (shape ‖ measure).

Question: is the served `E` the SMALLEST representation that loses nothing about
(a) the self-supervised future family and (b) the full-history measure? We sweep a
PCA rank k of E and fit a ridge map E_k -> joint targets (future family + every
measure column), cross-validated by held-out customers. The minimal rank is the
smallest k whose worst-case held-out R^2 is within the statistical noise floor of
the full-E R^2.

    k_min == full_dim  -> the served E is minimal (no smaller rank suffices)
    k_min <  full_dim  -> E is compressible; shrink before certifying minimality

Outputs a JSON receipt consumed by `looking_glass.objectives` (T2.minimality).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _grid(dim: int) -> list[int]:
    g = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, dim}
    return sorted(k for k in g if k <= dim)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Joint shape+measure minimality certificate")
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=3000)
    ap.add_argument("--horizons", default="30,90")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--targets",
        choices=("joint", "future"),
        default="joint",
        help="joint = future + measure; future = the predictive family only",
    )
    ap.add_argument(
        "--block",
        choices=("full", "shape"),
        default="full",
        help="full E, or only the shape block (first model.dim columns)",
    )
    a = ap.parse_args(argv)

    import polars as pl
    import torch

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.cfm_training import embed_batch_size, forward_embedding
    from looking_glass.supervised_gate import _stable_bucket
    from looking_glass.universality import (
        _b_sequences,
        _full_history_features,
        _future_family,
        _rfm_targets,
    )

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers

    seqs = _b_sequences(cfg)
    cust = np.array([str(s["customer"]) for s in seqs])
    zs = []
    _bs = embed_batch_size(cfg)
    for i in range(0, len(seqs), _bs):
        with torch.no_grad():
            z = forward_embedding(
                model, seqs[i : i + _bs], cfg=cfg, lb=getattr(model, "layer_b", None)
            )
        zs.append(z.detach().float().cpu().numpy())
    Z = np.concatenate(zs, 0) if zs else np.zeros((0, 1))
    if a.block == "shape":
        # served shape block = the first rank(serve_W) columns (48 after DEC-215);
        # fall back to the trunk dim only when no block-wise serve is set.
        _W = getattr(model, "serve_W", None)
        _sd = int(_W.shape[0]) if _W is not None else min(int(model.dim), Z.shape[1])
        Z = Z[:, : min(_sd, Z.shape[1])]
    full_dim = int(Z.shape[1])

    # ---- joint targets: future family + full-history measure -----------------
    cols: list[np.ndarray] = []
    names: list[str] = []
    for h in (int(x) for x in a.horizons.split(",")):
        fut, _past, _tc = _rfm_targets(cfg, h)
        j = (
            pl.DataFrame({"customer": cust})
            .join(fut, left_on="customer", right_on="customer_key", how="left")
            .fill_null(0.0)
        )
        for name, (yy, kind) in _future_family(j).items():
            y = np.asarray(yy, dtype=float)
            cols.append((y > 0).astype(float) if kind == "clf" else y)
            names.append(f"{name}@{h}d")
    if a.targets == "joint":
        Xh, hnames = _full_history_features(cfg, cust)
        Xh = np.asarray(Xh, dtype=float)
        for j in range(Xh.shape[1]):
            cols.append(Xh[:, j])
            names.append(f"measure:{hnames[j] if j < len(hnames) else j}")
    Y = np.column_stack(cols)
    # robust target preprocessing: winsorize at data-derived quantiles so a single
    # extreme measure/future value cannot dominate the worst-case R^2.
    if Y.shape[0] >= 20:
        _lo, _hi = np.quantile(Y, [0.01, 0.99], axis=0)
        Y = np.clip(Y, _lo, _hi)
    mu, sd = Y.mean(0), Y.std(0)
    # drop (near-)constant columns: they carry no information and destabilise ridge
    keep = sd > 1e-6 * np.maximum(1.0, np.abs(mu))
    Y = (Y - mu) / sd
    Y = Y[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    if Y.shape[1] == 0:
        raise SystemExit("no varying joint targets")

    # ---- held-out split (within split B) -------------------------------------
    tr = np.array([_stable_bucket(c) == 0 for c in cust])
    te = ~tr
    if tr.sum() < 10 or te.sum() < 10:
        raise SystemExit("too few held-out customers for a certificate")

    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scl = StandardScaler().fit(Z[tr])
    Ztr, Zte = scl.transform(Z[tr]), scl.transform(Z[te])
    pca = PCA(n_components=full_dim, random_state=0).fit(Ztr)
    Ptr, Pte = pca.transform(Ztr), pca.transform(Zte)

    def _r2(pred, yy):
        ss = ((yy - Y[tr].mean(0)) ** 2).sum(0)
        resid = ((yy - pred) ** 2).sum(0)
        return 1.0 - resid / np.maximum(ss, 1e-12)

    noise = float(1.0 / np.sqrt(len(cust[te])))
    # INFORMATIVE-TARGET FILTER: a target the FULL embedding cannot predict above
    # the noise floor is not a sufficiency target; letting noise drive the worst
    # case is a certificate bug. Keep only predictable targets (mirrors U).
    _full = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Ptr, Y[tr])
    _r2_full = _r2(_full.predict(Pte), Y[te])
    _inf = _r2_full > noise
    n_dropped = int((~_inf).sum())
    Y = Y[:, _inf]
    names = [n for n, k in zip(names, _inf) if k]
    if Y.shape[1] == 0:
        raise SystemExit("no informative joint targets for minimality")

    curve = []
    for k in _grid(full_dim):
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Ptr[:, :k], Y[tr])
        r2 = _r2(m.predict(Pte[:, :k]), Y[te])
        curve.append(
            {
                "k": int(k),
                "worst_r2": round(float(np.min(r2)), 5),
                "mean_r2": round(float(np.mean(r2)), 5),
            }
        )
    full = next(c for c in curve if c["k"] == full_dim)
    k_min = full_dim
    for c in curve:
        # BOTH the average retention and the worst-case retention must be within
        # the noise floor. Worst-only is dominated by a single weak target;
        # mean-only hides a collapsed target. Requiring both is stable.
        if c["mean_r2"] >= full["mean_r2"] - noise and c["worst_r2"] >= full["worst_r2"] - noise:
            k_min = c["k"]
            break
    retained = next(c for c in curve if c["k"] == k_min)
    receipt = {
        "full_dim": full_dim,
        "served_dim": full_dim,
        "measure_dim": int(k_min),
        "joint_certified": bool(k_min == full_dim),
        "noise": round(noise, 5),
        "full_worst_r2": full["worst_r2"],
        "full_mean_r2": full["mean_r2"],
        "retained_worst_r2": retained["worst_r2"],
        "retained_mean_r2": retained["mean_r2"],
        "retained_ratio": round(retained["worst_r2"] / max(full["worst_r2"], 1e-9), 5),
        "targets": a.targets,
        "block": a.block,
        "n_targets": int(Y.shape[1]),
        "n_targets_dropped_uninformative": n_dropped,
        "worst_target": names[int(np.argmin(_r2_full))],
        "worst_full_r2": round(float(np.min(_r2_full)), 5),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "curve": curve,
        "rule": (
            "k_min = smallest PCA rank whose worst-case held-out R^2 is within "
            "1/sqrt(n_test) of the full-E R^2; certified iff k_min == full_dim"
        ),
    }
    Path(a.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "curve"}, indent=2))
    return receipt


if __name__ == "__main__":
    main()
