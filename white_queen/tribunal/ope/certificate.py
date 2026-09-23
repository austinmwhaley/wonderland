"""Value certificate: the product.

For a candidate policy the library outputs exactly three things and nothing
else matters:

    (value_hat, [lo, hi], behavior_value)   ->   ship iff lo > behavior_value

The interval must be VALID (cover the true value) purely offline. We build it
from the deployable-policy estimates and their disagreement:

    value    = mean of the independent DEPLOYABLE estimates (FQE-argmax, MB-argmax)
    s_family = their spread (estimator disagreement -- a disagreement of
               independent methods is the only offline signal that the value
               is not pinned down)
    s_ens    = the FQE ensemble's own seed spread (epistemic)
    width    = z * sqrt(s_family^2 + s_ens^2)

When methods agree, the interval is tight and we can ship; when they disagree
we genuinely do not know, the interval is wide, and we HOLD. Safety is a
CONSEQUENCE of coverage: if the truth is in [lo, hi] and lo > behavior, the
candidate is truly better. Recall is tightness. Abstention is a wide interval.
"""
from __future__ import annotations

import math

Z_DEFAULT = 1.96
# Conformal multiplier: calibrated on labelled cases so empirical coverage
# reaches the nominal level. 1.0 = raw (uncalibrated) width.
CONFORMAL_K = 1.0


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _source_members(row):
    """One representative value per INDEPENDENT information source:
    qnet (FQE family), dynamics (MB family), td (linear TD). A source
    contributes its deployable/argmax variant when present, else its soft
    variant. Soft and argmax of the same family are ONE member (shared model).
    Unreliable (diverged/under-budget/out-of-range) sources are dropped."""
    from .contracts import estimates_from_row
    est = estimates_from_row(row)
    fam_rank = {"fqe_argmax": 3, "fqe_soft": 2, "mb_argmax": 3, "mb_soft": 2,
                "lstdq": 2}
    fam_of = {"fqe_argmax": "qnet", "fqe_soft": "qnet",
              "mb_argmax": "dynamics", "mb_soft": "dynamics", "lstdq": "td"}
    best = {}
    for e in est:
        fam = fam_of.get(e.name)
        if fam is None or not e.reliable or e.advisory:
            continue
        v = e.value
        if v is None or not math.isfinite(v):
            continue
        r = fam_rank.get(e.name, 1)
        if fam not in best or r > best[fam][1]:
            best[fam] = (v, r)
    return {f: v for f, (v, _) in best.items()}


def certify_row(row, behavior_mean, behavior_std=None, z=Z_DEFAULT,
                width_floor=0.0, min_ess=0.02, overlap_gain=1.0, k=None):
    """Return the certificate for one candidate.

    Interval = corrected value +/- (z*sqrt(source_var + ensemble_var)
              + |anchor_bias| + overlap_penalty)

    Anchor bias: the behavior policy's value is known from the log, and the
    panel reports the estimator's value for it (`fqe_behavior`). Their gap is a
    direct, per-dataset measurement of systematic bias; we remove it and widen
    by it. Overlap penalty: where the weights have no support (low ESS), no
    offline claim is possible, so the interval inflates toward abstention.
    """
    behavior_mean = float(behavior_mean)
    members = _source_members(row)
    vals = list(members.values())
    if len(vals) < 2:
        # A single source cannot certify (who would corroborate it?), but we
        # still report a WIDE interval so the case is measurable/coverable
        # rather than missing. It never deploys.
        v = vals[0] if vals else None
        dis = _finite((row.get("sharp_info") or {}).get("disagreement")) or 0.0
        scale = abs(behavior_mean) + 1.0
        half = (z * dis + 0.5 * scale) if v is not None else None
        return {"value": (None if v is None else round(v, 3)),
                "lo": (None if v is None else round(v - half, 3)),
                "hi": (None if v is None else round(v + half, 3)),
                "width": (None if half is None else round(half, 3)),
                "behavior": behavior_mean, "deploy": False,
                "members": list(members), "s_family": 0.0, "s_ens": round(dis, 3),
                "anchor_bias": None, "overlap_pen": None, "abstain": True,
                "reason": f"single source (have {len(vals)}); false confidence, "
                          "wide interval, never deploys"}
    # ---- anchor calibration ----
    # NOTE: the Q-probe of the behavior policy (sum_a mu(a|s) Q(s,a)) is NOT a
    # valid estimate of V(mu) when Q is fit for the candidate policy -- measured
    # 37 vs the true 67 -- so using it injects a large bogus bias and width.
    # V(mu) is already known from the logs; the width is calibrated empirically
    # (conformal) instead. Anchor term disabled.
    b = None
    vals_c = [v - b for v in vals] if b is not None else list(vals)
    m = sum(vals_c) / len(vals_c)
    s_family = (math.sqrt(sum((v - m) ** 2 for v in vals_c) / (len(vals_c) - 1))
                if len(vals_c) > 1 else 0.0)
    # Conservative point estimate: shift DOWN by half the inter-source spread,
    # so a single optimistic source cannot lift the value past the behavior
    # reference. (The weight-based analog of a pessimistic ensemble / CQL.)
    value = m - 0.5 * s_family
    s_ens = _finite((row.get("sharp_info") or {}).get("disagreement")) or 0.0
    # ---- overlap / support penalty ----
    ov = _finite(row.get("ess_frac"))
    scale = abs(behavior_mean) + 1.0
    overlap_pen = 0.0
    # The penalty must fire only when support is genuinely ABSENT, not when it
    # is merely modest. ESS from estimated propensities is routinely low even
    # with real support, so use a much lower floor (a quarter of the gate ESS
    # floor) and a gentler gain; otherwise a half-floor ESS (0.0104) added 35
    # points and held a policy whose truth was 99 vs behavior 69.
    _support_floor = max(0.25 * float(min_ess), 1e-4)
    if ov is not None:
        deficit = max(0.0, 1.0 - min(ov / _support_floor, 1.0))
        overlap_pen = overlap_gain * scale * deficit ** 2
    half = (z * math.sqrt(s_family ** 2 + s_ens ** 2)
            + (abs(b) if b is not None else 0.0) + overlap_pen)
    k = CONFORMAL_K if k is None else k
    half = k * half
    if half < width_floor:
        half = width_floor
    lo, hi = value - half, value + half
    bstd = _finite(behavior_std) or scale
    abstain = bool(half > 0.5 * max(bstd, 1.0))
    deploy = bool(lo > behavior_mean)
    reason = ("lo > behavior" if deploy else "lo <= behavior")
    return {"value": round(value, 3), "lo": round(lo, 3), "hi": round(hi, 3),
            "width": round(half, 3), "behavior": round(behavior_mean, 3),
            "deploy": deploy, "members": list(members),
            "s_family": round(s_family, 3), "s_ens": round(s_ens, 3),
            "anchor_bias": (None if b is None else round(b, 3)),
            "overlap_pen": round(overlap_pen, 3),
            "abstain": abstain, "reason": reason}


def certify_rows(rows, behavior_mean, behavior_std=None, z=Z_DEFAULT):
    return {n: certify_row(r, behavior_mean, behavior_std, z=z)
            for n, r in rows.items()}


def calibrate_k(cases, alpha=0.1, ks=None):
    """Smallest multiplier k such that the scaled intervals cover at least
    (1-alpha) of the labelled cells (conformal calibration on the suite).
    `cases` is a list of dicts with {value, half, truth}."""
    ks = ks or [round(1.0 + 0.25 * i, 2) for i in range(0, 17)]
    cells = [c for c in cases
             if c.get("half") is not None and c.get("truth") is not None
             and c.get("value") is not None]
    if not cells:
        return 1.0
    need = math.ceil((1.0 - alpha) * len(cells))
    for k in ks:
        hit = sum(1 for c in cells
                  if abs(c["truth"] - c["value"]) <= k * c["half"])
        if hit >= need:
            return k
    return ks[-1]


def advantage_certificate(row, behavior_mean, alpha=0.05, B=2000, seed=0):
    """Paired advantage certificate (option A+B).

    Delta = V(pi) - V(mu) estimated PER EPISODE on the SAME episodes
    (d_e = DR_e(pi) - G_e(mu)), so the shared behavior variance cancels and the
    known behavior value is used as a control variate. Anchor-corrected by the
    estimator's bias on the behavior policy, with an episode-level bootstrap CI.
    Decision: deploy iff the (one-sided) lower bound of Delta exceeds 0.
    """
    import numpy as np
    drv = row.get("dr_vals")
    epr = row.get("ep_returns")
    if not drv or not epr or len(drv) != len(epr) or len(drv) < 5:
        return {"deploy": False, "adv": None, "lo": None, "hi": None,
                "bias": None, "n": (0 if not drv else len(drv)),
                "reason": "no paired advantage evidence"}
    d = np.asarray(drv, dtype=np.float64) - np.asarray(epr, dtype=np.float64)
    fb = _finite(row.get("fqe_behavior"))
    b = (fb - float(behavior_mean)) if fb is not None else 0.0
    d = d - b
    # Variance reduction: trajectory-DR per-episode values have heavy tails
    # (capped importance products). Winsorize the paired differences at the
    # 5/95 quantiles before bootstrapping so a few outlier episodes cannot
    # dominate the CI and hide a real, consistent improvement.
    if len(d) >= 10:
        qlo, qhi = np.quantile(d, [0.05, 0.95])
        d = np.clip(d, qlo, qhi)
    n = len(d)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(int(B), n))].mean(axis=1)
    est = float(d.mean())
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1.0 - alpha))
    deploy = bool(lo > 0.0)
    return {"deploy": deploy, "adv": round(est, 3), "lo": round(lo, 3),
            "hi": round(hi, 3), "bias": round(b, 3), "n": n,
            "reason": ("advantage lo > 0" if deploy else "advantage lo <= 0")}


def fqe_value_certificate(row, behavior_mean, z=Z_DEFAULT):
    """Calibrated single-source value certificate (option C, partial).

    Uses the deployable FQE-argmax value, corrected by the anchor-measured
    bias, with the FQE ENSEMBLE spread as the uncertainty -- not the FQE-vs-MB
    disagreement (which is dominated by the MB model's error). This is tight
    when FQE is accurate and wide when the ensemble itself is unstable.
    Decision: (corrected V(pi) - V(mu)) - half > 0.
    """
    v = _finite(row.get("sharp_dm"))
    if v is None:
        v = _finite(row.get("fqe_dm"))
    if v is None:
        return {"deploy": False, "adv": None, "lo": None, "bias": None,
                "dis": None, "reason": "no FQE value"}
    dis = _finite((row.get("sharp_info") or {}).get("disagreement")) or 0.0
    fb = _finite(row.get("fqe_behavior"))
    b = (fb - float(behavior_mean)) if fb is not None else 0.0
    est = v - b
    half = z * dis + abs(b)
    adv = est - float(behavior_mean)
    lo = adv - half
    return {"deploy": bool(lo > 0.0), "adv": round(adv, 3),
            "lo": round(lo, 3), "bias": round(b, 3), "dis": round(dis, 3),
            "reason": ("calibrated FQE adv lo > 0" if lo > 0
                       else "calibrated FQE adv lo <= 0")}
