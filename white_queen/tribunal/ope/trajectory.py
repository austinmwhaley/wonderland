"""Trajectory-weight family extensions: WDR + MAGIC-lite.

WDR: self-normalized doubly-robust (Thomas & Brunskill 2016) — DR's calmer
sibling; normalization is data-adaptive by construction.
MAGIC-lite (Thomas et al. 2015, abridged): partial-DR estimates at several
truncation horizons + pure DM, combined by bootstrap-covariance min-variance
weights (diagonal-shrinkage for stability). Horizons autotuned from episode-
length quantiles; B/alpha/shrinkage derived from n. Nothing hardcoded.
"""

import numpy as np

from .autotune import VAR_FLOOR, resolve_bootstrap
from .estimators import dr_terms, episodes


def wdr_values(diet, cand, gamma, qnet, temperature, cap, alive_frac=None):
    """Per-episode WDR contributions: DR correction with weights normalized
    per timestep across episodes (self-normalization kills scale blowups).

    alive_frac=None -> max(1/n_episodes, 1e-3): a horizon survives only if its
    mean weight carries at least one episode's share (relative, not absolute).
    """
    eps = episodes(diet)
    n = len(eps)
    if alive_frac is None:
        alive_frac = max(1.0 / max(n, 1), 1e-3)
    T_max = max(len(e["act"]) for e in eps)
    cum_mat, terms = [], []
    for ep in eps:
        t = dr_terms(ep, cand, gamma, qnet, temperature, cap)
        terms.append((t["v0"], t["corr"], t["cum"]))
        cum_mat.append(np.pad(t["cum"], (0, T_max - t["T"])))
    cum_mat = np.asarray(cum_mat)
    denom = cum_mat.sum(0, keepdims=True) / len(eps)
    # Degenerate-horizon fallback: where the mean weight has collapsed,
    # no episode carries information at that horizon — normalizing would
    # divide ~0/~0 and launder noise as signal. Zero the correction there
    # (DM only), and report how far out the weights stayed alive.
    alive = denom[0] >= alive_frac * max(denom[0].max(), VAR_FLOOR)
    horizon_kept = int(alive.sum())
    vals = []
    for (v0, corr, cum), row in zip(terms, cum_mat):
        T = len(corr)
        norm = np.zeros(T)
        ok = alive[:T]
        norm[ok] = row[:T][ok] / np.maximum(denom[0, :T][ok], VAR_FLOOR)
        vals.append(float(v0 + np.sum(norm * corr)))
    wdr_values.horizon_kept = horizon_kept
    return vals


def magic_lite(
    diet,
    cand,
    gamma,
    qnet,
    temperature,
    cap,
    horizons=None,
    B=None,
    shrink=None,
    seed=0,
    ci_alpha=None,
    spread_factor=None,
):
    """Partial-horizon DR estimates + DM, min-variance combined via bootstrap
    covariance. Returns (est, weights, horizon_labels, ci, dropped).

    horizons=None -> quantiles of episode lengths (autotune).
    B/ci_alpha=None -> autotuned from n. shrink=None -> 1/sqrt(n_members).
    spread_factor=None -> max(10, sqrt(n)) x healthiest spread.
    """
    from .autotune import resolve_magic_horizons

    eps = episodes(diet)
    n = len(eps)
    if horizons is None:
        ep_lens = [len(e["act"]) for e in eps]
        horizons = resolve_magic_horizons(ep_lens)
    B_a, alpha_a = resolve_bootstrap(n, B, ci_alpha)
    B = B_a
    if shrink is None:
        shrink = 1.0 / max(np.sqrt(len(horizons) + 1), 1.0)
    if spread_factor is None:
        spread_factor = max(10.0, float(np.sqrt(n)))
    Hs = [h if h is not None else max(len(e["act"]) for e in eps) for h in horizons]
    labels = ["DM" if h is None else f"DR{h}" for h in horizons]
    # per-episode contribution matrix: rows episodes, cols horizons(+DM col)
    G = np.zeros((len(eps), len(Hs) + 1))
    for i, ep in enumerate(eps):
        t = dr_terms(ep, cand, gamma, qnet, temperature, cap)
        T = t["T"]
        G[i, -1] = t["v0"]
        for j, h in enumerate(Hs):
            G[i, j] = t["v0"] + float(np.sum(t["corr"][: min(h, T)]))
    rng = np.random.default_rng(seed)
    n = len(eps)
    boots = np.array([G[rng.integers(0, n, n)].mean(0) for _ in range(B)])
    # Variance-based member selection: horizons whose bootstrap spread
    # dwarfs the healthiest member are detonated, not informative — drop
    # them (DM always survives). MAGIC combines survivors only.
    spreads = boots.std(0)
    keep = np.ones(G.shape[1], bool)
    keep[:-1] = spreads[:-1] <= spread_factor * max(spreads.min(), VAR_FLOOR)
    keep[-1] = True
    dropped = [labels[i] for i in range(len(labels)) if not keep[i]]
    Gs, keep_idx = G[:, keep], np.flatnonzero(keep)
    if keep.sum() == 1:
        # Sole survivor (usually DM): no combination to solve.
        w = np.ones(1)
        draws = boots[:, keep_idx[0]]
    else:
        cov1d = np.cov(boots[:, keep].T)
        cov = cov1d + shrink * np.eye(keep.sum()) * (np.trace(cov1d) / max(keep.sum(), 1))
        try:
            w = np.linalg.solve(cov, np.ones(keep.sum()))
            w = w / w.sum()
        except np.linalg.LinAlgError:
            w = np.full(keep.sum(), 1.0 / max(keep.sum(), 1))
        draws = boots[:, keep] @ w
    lo, hi = np.quantile(draws, [alpha_a / 2, 1 - alpha_a / 2])
    full_w = [
        round(float(w[list(keep_idx).index(i)]), 3) if keep[i] else 0.0 for i in range(G.shape[1])
    ]
    return (
        float(w @ Gs.mean(0)),
        full_w,
        labels + ["DM"],
        (round(float(lo), 1), round(float(hi), 1)),
        dropped,
    )
