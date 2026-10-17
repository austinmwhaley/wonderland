"""Unsupervised plugin gate (Layer C, unsupervised family).

Certifies the label-free geometry of the frozen, held-out embedding `E`:

* **Segmentation** -- KMeans over a standardized PCA subspace of `E`, with k chosen by
  silhouette. When a truth receipt (`<stream_dir>/apex_truth.json` or `*_truth.json`)
  exists, the discovered clusters are checked against the latent `segment` (ARI). The
  clusters are also checked for VALUE differentiation (ANOVA on realized 90d margin).
* **Lookalike** -- the top-decile-LTV seed class is enriched among each seed member's
  k nearest neighbours in `E` relative to the random (prevalence) baseline.

Pure helpers (`_ari`, `_enrichment`, `_choose_k`, `_top_decile_mask`) are unit-tested on
synthetic data; the full run needs the frozen model (via `head_battery._b_states`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

OUTCOME_HORIZON_DAYS = 90


def _ari(a, b):
    """Adjusted Rand Index between two label vectors (identity is permutation-invariant)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(a) != len(b):
        return float("nan")
    return float(adjusted_rand_score(a, b))


def _choose_k(X, max_k=8, seed=0):
    """Choose k in 2..max_k by silhouette; returns (k, silhouette, labels)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    best_k, best_score, best_lab = 1, float("-inf"), np.zeros(n, dtype=int)
    for k in range(2, min(int(max_k), n - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(np.unique(lab)) < 2:
            continue
        score = float(silhouette_score(X, lab))
        if score > best_score:
            best_k, best_score, best_lab = k, score, lab
    return best_k, float(best_score), best_lab


def _enrichment(X, seed_mask, k=10):
    """Seed-class enrichment among each seed member's k nearest neighbours.

    Returns (observed seed fraction) / (seed prevalence). 1.0 is random; >1 means the
    geometry places lookalikes next to the seed class.
    """
    X = np.asarray(X, dtype=float)
    seed = np.asarray(seed_mask, dtype=bool)
    n, nseed = len(X), int(seed.sum())
    if n < 2 or nseed < 2 or nseed == n:
        return float("nan")
    k = int(max(1, min(int(k), n - 1)))
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[seed], return_distance=False)
    neighbours = idx[:, 1:]  # drop self
    fraction = float(seed[neighbours].mean())
    return float(fraction / (nseed / n))


def _top_decile_mask(y):
    """Boolean mask of the top 10% (at least 2) finite values of ``y``."""
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    mask = np.zeros(len(y), dtype=bool)
    m = int(finite.sum())
    if m == 0:
        return mask
    ntop = min(max(2, int(round(0.1 * m))), m)
    order = np.flatnonzero(finite)[np.argsort(-y[finite], kind="stable")]
    mask[order[:ntop]] = True
    return mask


def _truth_segments(cfg):
    """Load {customer: latent segment} from the stream's truth receipt, if present."""
    p = Path(str(cfg.db))
    stream_dir = p if p.is_dir() else p.parent
    candidates = [stream_dir / "apex_truth.json", *sorted(stream_dir.glob("*_truth.json"))]
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        try:
            blob = json.loads(path.read_text())
        except Exception:
            continue
        seg = {
            str(c["customer"]): str(c["segment"])
            for c in blob.get("customers", [])
            if c.get("customer") is not None and c.get("segment") is not None
        }
        if seg:
            return seg, str(path)
    return {}, None


def _anova_by_segment(labels, y):
    """One-way ANOVA of a realized outcome across discovered segments."""
    from scipy.stats import f_oneway

    labels, y = np.asarray(labels), np.asarray(y, dtype=float)
    groups = [y[labels == c] for c in np.unique(labels)]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2 or np.allclose(np.concatenate(groups).var(), 0.0):
        return float("nan"), float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        res = f_oneway(*groups)
    return float(res.statistic), float(res.pvalue)


def _eta_squared(labels, y):
    """Fraction of outcome variance explained by the segment labels."""
    labels, y = np.asarray(labels), np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    labels, y = labels[ok], y[ok]
    overall = y.mean()
    ssb = sum(
        len(y[labels == c]) * (y[labels == c].mean() - overall) ** 2 for c in np.unique(labels)
    )
    sst = ((y - overall) ** 2).sum()
    return float(ssb / max(sst, 1e-12))


def run_unsupervised(model, cfg, max_k=8, seed=0):
    """Certify segmentation + lookalike geometry on the held-out (split-B) embedding."""
    from looking_glass.head_battery import _b_states
    from looking_glass.supervised_gate import _labels

    cust, Z = _b_states(model, cfg)
    cust = np.asarray([str(c) for c in cust], dtype=object)
    Z = np.asarray(Z, dtype=float)
    n = len(cust)

    Zs = StandardScaler().fit_transform(Z)
    ncomp = int(min(64, Zs.shape[1], max(2, n - 1)))
    X = PCA(n_components=ncomp, random_state=seed).fit_transform(Zs) if ncomp >= 2 else Zs

    k, silhouette, lab = _choose_k(X, max_k=max_k, seed=seed)
    if k < 2:
        silhouette = float("nan")

    labmap = _labels(cfg, cust, OUTCOME_HORIZON_DAYS)
    ltv = np.array([float(labmap.get(c, {}).get("ltv", 0.0)) for c in cust], dtype=float)
    anova_f, anova_p = _anova_by_segment(lab, ltv)
    eta2 = _eta_squared(lab, ltv)

    truth, truth_path = _truth_segments(cfg)
    has_truth = np.array([c in truth for c in cust])
    if k >= 2 and has_truth.sum() >= 2:
        ari = _ari(lab[has_truth], np.array([truth[c] for c in cust[has_truth]]))
    else:
        ari = float("nan")

    seed_mask = _top_decile_mask(ltv)
    lookalike_k = int(max(1, min(10, n - 1)))
    enrichment = _enrichment(X, seed_mask, k=lookalike_k)

    sizes = [int((lab == c).sum()) for c in range(k)]
    means = [float(ltv[lab == c].mean()) for c in range(k)]
    return {
        "n_customers": int(n),
        "pca_components": int(X.shape[1]),
        "k": int(k),
        "silhouette": None if np.isnan(silhouette) else round(silhouette, 4),
        "segment_sizes": sizes,
        "segment_ltv_means": [round(m, 4) for m in means],
        "ari": None if np.isnan(ari) else round(ari, 4),
        "truth_customers": int(has_truth.sum()),
        "truth_file": truth_path,
        "anova_f": None if np.isnan(anova_f) else round(anova_f, 4),
        "anova_p": None if np.isnan(anova_p) else round(anova_p, 6),
        "eta_squared": None if np.isnan(eta2) else round(eta2, 4),
        "lookalike_k": lookalike_k,
        "lookalike_seeds": int(seed_mask.sum()),
        "lookalike_base_rate": round(float(seed_mask.mean()), 4),
        "lookalike_enrichment": None if np.isnan(enrichment) else round(enrichment, 4),
    }


def main(argv=None):
    from looking_glass.cfm_state import load_frozen_encoder

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=12000)
    ap.add_argument("--max-k", type=int, default=8, dest="max_k")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    metrics = run_unsupervised(model, cfg, max_k=a.max_k, seed=a.seed)
    print("# UNSUPERVISED PLUGIN GATE (held-out E)")
    for key, value in metrics.items():
        print(f"  {key:26s} {value}")


if __name__ == "__main__":
    main()
