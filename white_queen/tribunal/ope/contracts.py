"""Estimator contract.

The decision layer must never see a bare number. Every estimate is an
`Estimate`: (value, interval, support, estimand, sources, reliability). This
module is the single place that knows

  * WHAT each estimator estimates (estimand), and
  * WHERE its information comes from (source).

Two estimates may corroborate each other ONLY if their source sets are
disjoint. That makes the sigma principle structural instead of hand-counted:
soft-FQE and FQE-argmax share one Q net, so they are one source and cannot
corroborate; DR shares the Q net with FQE, so its only independent partner is
the dynamics model.

Nothing else in the codebase should compare estimates across estimands or
count witnesses by name.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ---- estimands (WHAT is estimated) ----------------------------------------
V_START = "V(s0)"            # value at the episode's initial state(s)
V_MARGINAL = "V(d_mu)"       # behavior-state-weighted (marginalized) value
V_ADVANTAGE = "E[(rho-1)A]"  # candidate-minus-behavior marginalized advantage

# ---- sources (WHERE the information comes from) ---------------------------
S_QNET = "qnet"          # fitted action-value function (FQE / DR control variate)
S_DYNAMICS = "dynamics"  # learned world model, rolled out
S_WEIGHTING = "weighting"  # importance weights applied to logged returns
S_TD = "td"              # linear TD / LSTDQ
S_EMPIRICAL = "empirical"  # a logged-return statistic (behavior anchor)

# ---- policies (on WHAT policy the estimand is evaluated) ------------------
POL_DEPLOY = "deployable"  # the policy we would actually ship (argmax/greedy)
POL_PROXY = "proxy"        # a softer proxy policy (soft/weighted)


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    estimand: str
    sources: frozenset
    policy: str
    value_key: str
    lo_key: Optional[str] = None      # explicit interval-lower key in the row
    hi_key: Optional[str] = None
    support_key: Optional[str] = None  # ESS-like support key (weights-based)
    advisory: bool = False             # may corroborate, never gate alone

    @property
    def deployable(self) -> bool:
        return self.policy == POL_DEPLOY


# The registry. Adding an estimator = add a row here (and a panel key); the
# decision layer picks it up automatically. Values are read from the gate row.
ESTIMATOR_CONTRACT = {
    # FQE family (one Q net -> one source). Soft and argmax are the SAME source.
    "fqe_soft": EstimatorSpec("fqe_soft", V_START, frozenset({S_QNET}),
                              POL_PROXY, "fqe_dm", support_key="ess_frac"),
    "fqe_argmax": EstimatorSpec("fqe_argmax", V_START, frozenset({S_QNET}),
                                POL_DEPLOY, "sharp_dm"),
    # Model-based family (one dynamics model -> one source).
    "mb_soft": EstimatorSpec("mb_soft", V_START, frozenset({S_DYNAMICS}),
                             POL_PROXY, "mb", support_key="ess_frac"),
    "mb_argmax": EstimatorSpec("mb_argmax", V_START, frozenset({S_DYNAMICS}),
                               POL_DEPLOY, "mb_sharp"),
    # Weighting family. DR reuses the Q net, so its sources are {weights, qnet}.
    "dr": EstimatorSpec("dr", V_START, frozenset({S_WEIGHTING, S_QNET}),
                        POL_PROXY, "dr", lo_key="dr_ci"),
    # Diagnostics: displayed and recorded, but NOT vetted to gate a deploy
    # (advisory). WIS/IS are high-variance point estimates; LSTDQ is a linear
    # approximation; level is a median composite of several sources it cannot
    # claim independence from.
    "wis": EstimatorSpec("wis", V_START, frozenset({S_WEIGHTING}),
                         POL_PROXY, "wis", advisory=True),
    "is": EstimatorSpec("is", V_START, frozenset({S_WEIGHTING}),
                        POL_PROXY, "is", advisory=True),
    "lstdq": EstimatorSpec("lstdq", V_START, frozenset({S_TD}),
                           POL_PROXY, "lstdq", advisory=True),
    "level": EstimatorSpec("level", V_START,
                           frozenset({S_QNET, S_TD, S_DYNAMICS}),
                           POL_PROXY, "level_est", advisory=True),
    # Horizon-free step-DR: a DIFFERENT estimand (marginalized advantage).
    "step_dr": EstimatorSpec("step_dr", V_ADVANTAGE,
                             frozenset({S_WEIGHTING, S_QNET}),
                             POL_PROXY, "dr_step_adv", advisory=True),
}


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _row_value(row, key):
    """Read a scalar from a row that may hold dicts (mb) or None."""
    if key not in row or row[key] is None:
        return None
    v = row[key]
    if isinstance(v, dict):
        v = v.get("mb") if "mb" in v else v.get("dm")
    return _finite(v)


@dataclass
class Estimate:
    name: str
    estimand: str
    sources: frozenset
    policy: str
    value: Optional[float]
    lo: Optional[float] = None
    hi: Optional[float] = None
    support: Optional[float] = None
    reliable: bool = True
    reason: str = ""
    advisory: bool = False

    @property
    def deployable(self) -> bool:
        return self.policy == POL_DEPLOY

    def clears(self, bar) -> bool:
        """True if the usable lower bound of this estimate exceeds the bar."""
        if not self.reliable:
            return False
        bound = self.lo if self.lo is not None else self.value
        return bool(bound is not None and bound > bar)

    def with_sources(self, sources):
        return Estimate(self.name, self.estimand, frozenset(sources),
                        self.policy, self.value, self.lo, self.hi, self.support,
                        self.reliable, self.reason, self.advisory)


def estimates_from_row(row, bar=None, min_ess=None, z=1.96):
    """Build the tagged, reliability-checked Estimate list from a gate row.

    Reliability is structural, not a magic threshold: an estimate is unusable
    when its value is absent/out-of-plausible-range, when the panel flagged it
    diverged or under-budget, or (for weighting-based estimates) when its
    support is below the ESS floor. Out-of-range values are REJECTED here
    rather than silently clamped in the panel.
    """
    out = []
    clamp = row.get("value_clamp") or {}
    diverged = bool((row.get("sharp_info") or {}).get("diverged"))
    under_budget = bool((row.get("fqe_info") or {}).get("under_budget"))
    efqe_dis = _finite((row.get("efqe") or {}).get("disagreement"))
    sharp_info = row.get("sharp_info") or {}
    for name, spec in ESTIMATOR_CONTRACT.items():
        if name in ("step_dr",):
            # advisory, different estimand; only included when present.
            v = _row_value(row, spec.value_key)
            if v is None:
                continue
            out.append(Estimate(name, spec.estimand, spec.sources, spec.policy,
                                v, reliable=True, advisory=True,
                                reason="advisory (different estimand)"))
            continue
        v = _row_value(row, spec.value_key)
        if v is None:
            continue
        reliable, reason = True, ""
        # explicit interval
        lo = hi = None
        if spec.lo_key and spec.lo_key in row:
            ci = row[spec.lo_key]
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                lo, hi = _finite(ci[0]), _finite(ci[1])
        elif "sharp_info" in row and name == "fqe_argmax" and \
                sharp_info.get("lower") is not None:
            lo = _finite(sharp_info.get("lower"))
            hi = v
        elif name in ("fqe_soft",) and efqe_dis is not None:
            lo, hi = v - z * efqe_dis, v + z * efqe_dis
        elif name in ("mb_soft", "mb_argmax"):
            se = _finite(row.get("mb_sharp_se") if name == "mb_argmax"
                         else row.get("mb_se"))
            if se is not None:
                lo, hi = v - z * se, v + z * se
        # support
        support = _finite(row.get(spec.support_key)) if spec.support_key else None
        # reliability rules
        _clamped = (("fqe" in clamp and name in ("fqe_soft", "fqe_argmax"))
                    or ("mb" in clamp and name in ("mb_soft", "mb_argmax")))
        if _clamped:
            reliable, reason = False, "out of plausible range"
        elif diverged and name in ("fqe_soft", "fqe_argmax"):
            reliable, reason = False, "FQE diverged/uncalibrated"
        elif under_budget and name in ("fqe_soft", "fqe_argmax"):
            reliable, reason = False, "FQE under-budget"
        elif spec.sources == frozenset({S_WEIGHTING}) and support is not None \
                and min_ess is not None and support < min_ess:
            reliable, reason = False, f"low weighting support {support:.3f}"
        out.append(Estimate(name, spec.estimand, spec.sources, spec.policy,
                            v, lo, hi, support, reliable, reason, spec.advisory))
    return out


def independent_pairs(estimates, bar, require_deployable=False):
    """All ordered pairs (a, b), a != b, of usable estimates that share an
    estimand AND have disjoint sources AND both clear the bar. `require_
    deployable` restricts a to the deployable policy (b corroborates it)."""
    usable = [e for e in estimates if e.reliable and not e.advisory]
    pairs = []
    for i, a in enumerate(usable):
        for b in usable:
            if b is a:
                continue
            if a.estimand != b.estimand:
                continue
            if not a.sources.isdisjoint(b.sources):
                continue
            if require_deployable and not a.deployable:
                continue
            if a.clears(bar) and b.clears(bar):
                pairs.append((a, b))
    return pairs
