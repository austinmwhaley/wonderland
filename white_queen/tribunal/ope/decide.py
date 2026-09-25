"""Decision as inference over the estimator contract.

H0: the candidate is NOT better than behaviour by the required edge.
DEPLOY is the default when H0 can be rejected using INDEPENDENT evidence;
HOLD only when the evidence is insufficient or a needed estimate is unusable.

The rule falls out of the evidence model (contracts.py) instead of a
hand-tuned witness count:

  * A deploy needs two usable estimates that (a) share an estimand,
    (b) have disjoint sources, and (c) both clear the bar.
  * If both are on the DEPLOYABLE policy, no action-support coverage is
    required (direct methods extrapolate off-policy).
  * Otherwise the trajectory weighting must be covered (ESS floor); the pair
    may then include proxy estimates corroborated by an independent source.
"""

from __future__ import annotations

import math

from .contracts import estimates_from_row, independent_pairs


def _finite(v, default=None):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _pair_str(a, b):
    return (
        f"{a.name}({a.value:.1f}, lo "
        f"{(a.lo if a.lo is not None else a.value):.1f}) + "
        f"{b.name}({b.value:.1f})"
    )


def coverage(row, min_ess):
    """Trajectory weighting coverage: the logged weights support the estimand."""
    e = _finite(row.get("ess_frac"))
    return bool(e is not None and e >= min_ess), e


def decide(row, bar, min_ess, behavior_mean=None):
    """Return {deploy, rule, evidence, covered, estimates}."""
    est = estimates_from_row(row, bar=bar, min_ess=min_ess)
    covered, ess = coverage(row, min_ess)
    dep_pairs = independent_pairs(est, bar, require_deployable=True)
    cov_pairs = independent_pairs(est, bar) if covered else []
    evidence = [f"{a.name}∥{b.name}" for a, b in (dep_pairs or cov_pairs)]

    if dep_pairs:
        a, b = dep_pairs[0]
        rule = f"superiority: independent deployable witnesses agree ({a.name} ∥ {b.name})"
        return {
            "deploy": True,
            "rule": rule,
            "evidence": evidence,
            "covered": True,
            "estimates": est,
        }
    if cov_pairs:
        a, b = cov_pairs[0]
        rule = f"superiority: independent sources agree on covered weights ({a.name} ∥ {b.name})"
        return {
            "deploy": True,
            "rule": rule,
            "evidence": evidence,
            "covered": True,
            "estimates": est,
        }

    # ---- HOLD, with a reason that says what was missing -------------------
    unreliable = [e for e in est if not e.reliable]
    if unreliable:
        why = "; ".join(f"{e.name}: {e.reason}" for e in unreliable[:3])
        rule = f"HOLD: estimate not trustworthy ({why})"
    elif not covered:
        rule = (
            "HOLD: insufficient coverage (trajectory ESS "
            f"{ess if ess is not None else float('nan'):.3f} < {min_ess}) "
            "and the deployable witnesses disagree"
        )
    else:
        rule = "HOLD: cannot reject 'not better than behaviour'"
    return {"deploy": False, "rule": rule, "evidence": [], "covered": covered, "estimates": est}
