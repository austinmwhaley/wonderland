"""Unsupervised plugin gate (Layer C, unsupervised family).

Certifies the label-free geometry of the frozen, held-out embedding `E`:

* **Segmentation** -- KMeans over a standardized PCA subspace of `E`, with k chosen by
  silhouette. When a truth receipt (`<stream_dir>/reference_truth.json` or `*_truth.json`)
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


def _verdict(metrics):
    """PASS iff the geometry shows separation, value differentiation, and lookalike structure."""
    silhouette = metrics.get("silhouette")
    anova_p = metrics.get("anova_p")
    enrichment = metrics.get("lookalike_enrichment")
    ok = (
        silhouette is not None
        and silhouette > 0
        and anova_p is not None
        and anova_p < 0.05
        and enrichment is not None
        and enrichment > 1.0
    )
    return "PASS" if ok else "FAIL"


def _truth_segments(cfg):
    """Load {customer: latent segment} from the stream's truth receipt, if present."""
    p = Path(str(cfg.db))
    stream_dir = p if p.is_dir() else p.parent
    candidates = [stream_dir / "reference_truth.json", *sorted(stream_dir.glob("*_truth.json"))]
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


def _run_unsupervised(model, cfg, max_k=8, seed=0, states=None, block="full"):
    """Certify segmentation + lookalike geometry on the held-out (split-B) embedding.

    ``states=(cust, Z)`` may be supplied to run the SAME pipeline on any feature
    matrix (the traditional raw-feature comparator).

    ``block`` selects the CONSUMED served view: ``full`` (shape ‖ measure), or the
    explicit ``measure`` block of a block-wise served E. The shape is a whitened
    contractive state (magnitude-blind by design); distance-based value consumers
    use the accessible measure block that DEC-215 exposes for exactly this.
    """
    from looking_glass.head_battery import _b_states
    from looking_glass.supervised_gate import _labels

    if states is None:
        cust, Z = _b_states(model, cfg)
    else:
        cust, Z = states
    cust = np.asarray([str(c) for c in cust], dtype=object)
    Z = np.asarray(Z, dtype=float)
    if block != "full":
        _names = list(getattr(getattr(model, "layer_b", None), "names", ()) or ())
        if _names and Z.shape[1] > len(_names):
            d = Z.shape[1] - len(_names)
            Z = Z[:, d:] if block == "measure" else Z[:, :d]
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
    metrics = {
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
    metrics["verdict"] = _verdict(metrics)
    return metrics


def run_unsupervised(model, cfg, max_k=8, seed=0, null=None, block="full"):
    """Certify geometry; with ``null=(db, tag, ckpt_dir)`` also run a null control.

    ``null_ok`` is True iff the null encoder's verdict is FAIL (no skill).
    ``block`` = the consumed served view (``full`` or the accessible ``measure``
    block). The full-concat metrics are reported alongside when they differ.
    """
    from looking_glass.head_battery import _b_states

    cust, Z = _b_states(model, cfg)
    metrics = _run_unsupervised(model, cfg, max_k=max_k, seed=seed, states=(cust, Z), block=block)
    if block != "full":
        metrics["full_metrics"] = _run_unsupervised(
            model, cfg, max_k=max_k, seed=seed, states=(cust, Z), block="full"
        )
    # TRADITIONAL COMPARATOR (DEC-212): the same segmentation/lookalike pipeline on
    # raw/RFM aggregate features. "Exceeds traditional" = E >= raw on silhouette,
    # value differentiation (eta^2), and lookalike enrichment, with E's own gate PASS.
    try:
        from looking_glass.cfm_data import _read_stream
        from looking_glass.supervised_gate import _raw_features

        d = _read_stream(cfg)
        RAW, pos = _raw_features(d, [str(c) for c in cust], cfg)
        Xr = RAW[[pos[str(c)] for c in cust]]
        raw_metrics = _run_unsupervised(model, cfg, max_k=max_k, seed=seed, states=(cust, Xr))

        def _ge(a, b):
            return a is not None and b is not None and float(a) >= float(b)

        comp = {
            "silhouette": _ge(metrics.get("silhouette"), raw_metrics.get("silhouette")),
            "eta_squared": _ge(metrics.get("eta_squared"), raw_metrics.get("eta_squared")),
            "lookalike": _ge(
                metrics.get("lookalike_enrichment"), raw_metrics.get("lookalike_enrichment")
            ),
        }
        beats = bool(metrics.get("verdict") == "PASS" and all(comp.values()))
        metrics["baseline"] = "raw-feature k-means + lookalike"
        metrics["raw_metrics"] = raw_metrics
        metrics["comparative"] = comp
        metrics["comparative_passed"] = int(sum(comp.values()))
        metrics["comparative_total"] = len(comp)
        metrics["beats_traditional"] = beats
        # scorecard contract: all comparative sub-gates pass, else 0/total
        metrics["passed"] = int(metrics["comparative_passed"] if beats else 0)
        metrics["total"] = len(comp)
    except Exception as exc:  # noqa: BLE001
        metrics["comparative_error"] = str(exc)
    if null is None:
        return metrics
    from looking_glass.supervised_gate import _load_null_model

    null_model, null_cfg = _load_null_model(cfg, null)
    null_metrics = _run_unsupervised(null_model, null_cfg, max_k=max_k, seed=seed)
    metrics["null_metrics"] = null_metrics
    metrics["null_ok"] = null_metrics["verdict"] == "FAIL"
    return metrics


def _print_metrics(metrics, title):
    print(title)
    for key, value in metrics.items():
        if key in ("null_metrics", "null_ok"):
            continue
        print(f"  {key:26s} {value}")


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
    metrics = run_unsupervised(model, cfg, max_k=a.max_k, seed=a.seed, null=null)
    _print_metrics(metrics, "# UNSUPERVISED PLUGIN GATE (held-out E)")
    if "null_metrics" in metrics:
        _print_metrics(metrics["null_metrics"], "# NULL CONTROL (same targets, null encoder)")
        print(f"  null_ok={metrics['null_ok']}")


if __name__ == "__main__":
    main()
