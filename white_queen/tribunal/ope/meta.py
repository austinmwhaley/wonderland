"""Meta layer: behavior anchor + SLOPE-lite estimator selection.

Anchor: the diet's discounted mean as an explicit panelist. Any estimator
losing to 'predict average' is distrusted (the panelist is bad, not the
candidate) — embarrassing when it wins, load-bearing when it does.
SLOPE-lite (Lepski-style, abridged): order panelists weak-to-strong by
assumption strength; walk up while bootstrap CIs overlap the running
choice; stop at the sharpest estimate consistent with everything weaker.
No selection knobs beyond the shared CI alpha. Reported, not yet gate power.
"""
import numpy as np

ORDER = ("anchor", "fqe_dm", "mis", "wis", "wdr", "dr", "magic", "is")


def slope_lite(panelists, ci_map, alpha=None):
    """panelists: {name: point}; ci_map: {name: (lo, hi)}.
    Returns (selected_name, selected_value).

    alpha is informational only (the CIs already encode it; autotuned
    upstream via resolve_bootstrap). Kept as an explicit arg so callers must
    acknowledge which level the intervals were built at.
    """
    avail = [n for n in ORDER if n in panelists and n in ci_map]
    chosen, (clo, chi) = avail[0], ci_map[avail[0]]
    for n in avail[1:]:
        lo, hi = ci_map[n]
        if lo <= chi and clo <= hi:
            chosen, (clo, chi) = n, (min(clo, lo), max(chi, hi))
        else:
            break
    return chosen, panelists[chosen]
