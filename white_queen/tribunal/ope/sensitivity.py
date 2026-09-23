"""Hidden-confounding sensitivity (Rosenbaum-style, MSM odds model).

Human-log behavior breaks the core OPE assumption: logged actions may depend
on things not in the observation (a glance, a radio call). Then estimated AND
exact propensities are both wrong in the same direction, and no receipt
computed so far detects it — CIs quantify sampling noise, not hidden bias.

Marginal sensitivity model (Tan/Zhao-style): an unobserved confounder can
distort each episode's weight by a bounded odds factor Gamma >= 1
(w_i' in [w_i/Gamma, w_i*Gamma]). Question answered: smallest Gamma that
could overturn a DEPLOY (drag the worst-case mean to the bar). Large
Gamma* = robust conclusion; Gamma* near 1 = fragile, human-log caveat applies.

Worst case over distortions is EXACT, not sampled: to minimize a weighted
mean under box constraints, put max weight on the smallest values —
threshold structure, try all n+1 splits after sorting (O(n log n)).
Sampling uncertainty is layered by evaluating the frontier at the DR
lower-CI endpoint too (reported separately); the headline curve uses the
point contributions with the sampling bar fixed.
"""

import numpy as np

GAMMA_GRID = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)


def worst_case_mean(values, weights, gamma):
    """Minimum achievable weighted mean under Gamma-bounded distortion
    (w_i' in [w_i/Gamma, w_i*Gamma]). Threshold structure: some k minimizes
    with max weight on the k smallest values, min weight elsewhere (exchange
    argument). Tries all n+1 splits — exact in O(n log n)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0 or w.sum() <= 0 or not np.isfinite(v).all():
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    n = len(v)
    cum_num_hi = np.concatenate([[0.0], np.cumsum(v * w * gamma)])
    cum_num_lo = np.concatenate([[0.0], np.cumsum(v * w / gamma)])
    cum_den_hi = np.concatenate([[0.0], np.cumsum(w * gamma)])
    cum_den_lo = np.concatenate([[0.0], np.cumsum(w / gamma)])
    tot_num_lo, tot_den_lo = cum_num_lo[-1], cum_den_lo[-1]
    best = float("inf")
    for k in range(n + 1):
        # First k (smallest values) get max weight, rest get min weight.
        num = cum_num_hi[k] + (tot_num_lo - cum_num_lo[k])
        den = cum_den_hi[k] + (tot_den_lo - cum_den_lo[k])
        if den > 0:
            m = num / den
            if m < best:
                best = m
    return float(best)


def breakdown_frontier(values, weights, bar, grid=GAMMA_GRID):
    """Worst-case mean at each Gamma. Returns list of (gamma, worst_mean)."""
    return [(float(g), worst_case_mean(values, weights, float(g))) for g in grid]


def gamma_star(values, weights, bar, grid=GAMMA_GRID):
    """Smallest Gamma whose worst case reaches the bar (decision flips).
    Returns (gamma_star or None if never flips on grid, frontier). None with
    worst case already below bar at Gamma=1 means the point itself fails."""
    frontier = breakdown_frontier(values, weights, bar, grid)
    for g, m in frontier:
        if m <= bar:
            return (None if g == 1.0 else float(g)), frontier
    return float("inf"), frontier
