"""Coverage receipts: the honesty columns beside every OPE number. ESS below
floor = automatic no-go no matter what the lift says (the retail doctrine,
in miniature). All sample-size dials autotune from n; pass explicit B/alpha
to override."""
import numpy as np

from .autotune import resolve_bootstrap


def ess(episode_weights):
    w = np.asarray(episode_weights, dtype=np.float64)
    return float((w.sum() ** 2) / max((w ** 2).sum(), 1e-12))


def ess_frac(episode_weights, n_episodes=None):
    n = n_episodes or len(episode_weights)
    return ess(episode_weights) / max(n, 1)


def _segment_disc_returns(diet, gamma):
    """Per-episode discounted returns. Loops over episodes (not rows), using
    numpy inner sums — correct and fast. (A fully segment-vectorized rescale
    by gamma^-start overflows for long t; discarded.)"""
    rew = np.asarray(diet["rew"], dtype=np.float64)
    ep = np.asarray(diet["episode"])
    if len(rew) == 0:
        return np.zeros(0)
    change = np.flatnonzero(np.diff(ep) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(rew)]])
    out = np.empty(len(starts), dtype=np.float64)
    for i in range(len(starts)):
        r = rew[starts[i]:ends[i]]
        out[i] = float(np.sum(r * gamma ** np.arange(len(r))))
    return out


def behavior_stats(diet, gamma=0.99):
    """Per-episode DISCOUNTED returns — same units as every estimator.
    Comparing discounted estimates against undiscounted means is a scale
    bug; this function makes it unrepresentable."""
    vals = _segment_disc_returns(diet, gamma)
    return {"mean": float(vals.mean()) if len(vals) else 0.0,
            "std": float(vals.std()) if len(vals) else 0.0,
            "n_episodes": len(vals)}


def bootstrap_ci(values, B=None, alpha=None, seed=0):
    """Percentile CI over per-episode contributions. Uncertainty quantified
    from the data — the gate's strictness is computed, not chosen.
    B/alpha=None -> autotuned from n (alpha 0.10 if n<30 else 0.05).
    Vectorized: one (B, n) index draw instead of B Python-loop iterations —
    identical statistics, ~10x faster."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    B_auto, alpha_auto = resolve_bootstrap(n, B, alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B_auto, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha_auto / 2, 1 - alpha_auto / 2])
    return float(lo), float(hi)


def bootstrap_p(values, threshold, B=None, seed=0):
    """One-sided bootstrap p-value: P(mean* <= threshold). Same resampling as
    bootstrap_ci (same seed convention) so p and CI never contradict:
    threshold below the lower α/2 bound implies p <= α/2 (up to Monte Carlo
    noise). Vectorized; returns float in [0, 1]."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    B_auto, _ = resolve_bootstrap(n, B, None)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, size=(B_auto, n))].mean(axis=1)
    return float((means <= threshold).mean())


def holm_reject(pvals, alpha=0.05):
    """Holm-Bonferroni step-down. pvals: {name: p}. Returns {name: bool}
    (True = reject H0). Family-wise error <= alpha without independence
    assumptions. Deterministic; ties broken by name for reproducibility."""
    m = len(pvals)
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    rejected, stop = {}, False
    for k, name in enumerate(order):
        if stop:
            rejected[name] = False
            continue
        if pvals[name] <= alpha / (m - k):
            rejected[name] = True
        else:
            rejected[name] = False
            stop = True
    return rejected


def spearman(x, y):
    """Rank correlation without scipy: OPE scores vs ground truth."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / max(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()), 1e-12))


def bootstrap_p_ratio(num, den, threshold, B=None, seed=0):
    """One-sided bootstrap p-value for a self-normalized ratio statistic
    sum(num)/sum(den): P(ratio* <= threshold) over episode resamples.
    Used for the horizon-free step-DR, where the estimator is a ratio."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    n = len(num)
    if n == 0:
        return 1.0
    B_auto, _ = resolve_bootstrap(n, B, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B_auto, n))
    stats = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1e-12)
    return float((stats <= threshold).mean())


def bootstrap_p_advantage(c_num, c_den, b_num, b_den, threshold=0.0,
                          B=None, seed=0):
    """One-sided bootstrap p-value for the matched-estimand step-DR advantage

        Delta = sum(c_num)/sum(c_den) - sum(b_num)/sum(b_den)

    i.e. candidate minus behavior on the SAME behavior-state distribution,
    resampled by episode. Tests H0: Delta <= threshold."""
    c_num = np.asarray(c_num, dtype=np.float64)
    c_den = np.asarray(c_den, dtype=np.float64)
    b_num = np.asarray(b_num, dtype=np.float64)
    b_den = np.asarray(b_den, dtype=np.float64)
    n = len(c_num)
    if n == 0 or len(b_num) != n:
        return 1.0
    B_auto, _ = resolve_bootstrap(n, B, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B_auto, n))
    c = c_num[idx].sum(axis=1) / np.maximum(c_den[idx].sum(axis=1), 1e-12)
    b = b_num[idx].sum(axis=1) / np.maximum(b_den[idx].sum(axis=1), 1e-12)
    return float(((c - b) <= threshold).mean())

