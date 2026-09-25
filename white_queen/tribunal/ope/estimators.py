"""Full OPE validation suite, v3 (adaptive): every knob derived from the diet.

- Temperature: ESS-targeted per candidate/diet (sharp when coverage allows).
  Grid itself autotuned from max episode length (autotune.resolve_temps).
- Clip cap: data quantile 1-1/sqrt(N) (autotune.resolve_clip_quantile).
- FQE: validation early-stop; budget/capacity/batch/LR from N/obs_dim
  (autotune.resolve_fqe_cfg). The curve decides, via held-out Bellman error.
- Gate inputs: bootstrap CIs (B/alpha from n) + ESS-continuous DR<->DM blend
  (range 1/sqrt(n)..5/sqrt(n)). Strictness computed.
- Candidate-agnostic: anything implementing protocols.CandidateProtocol
  plugs in; discrete-action general via nA; continuous path = optional
  log_prob_fn hook.

Pass meta=None / fqe_cfg=None for full autotune (recommended). Explicit
values override per-key. META below is the legacy fallback (frozen v2
literals, kept for reproducibility of old verdicts only).
"""

import numpy as np

from .autotune import (
    PROB_FLOOR,
    VAR_FLOOR,
    WEIGHT_CEIL,
    RHO_CAP_FLOOR,
    diet_fingerprint,
    resolve_blend_range,
    resolve_bootstrap,
    resolve_clip_quantile,
    resolve_fqe_cfg,
    resolve_meta,
    resolve_target_ess,
    resolve_temps,
)
from .protocols import check_candidate

META = {
    "target_ess_frac": 0.2,
    "clip_quantile": 0.99,
    "ci_alpha": 0.05,
    "bootstrap_B": 200,
    "blend_lo": 0.05,
    "blend_hi": 0.5,
    "rel_edge_std": 0.5,
    "temps": (0.05, 0.1, 0.25, 0.5, 1.0),
}
# DEPRECATED (v12+): frozen v2 literals, kept so old verdicts reproduce
# bit-identically. New code passes meta=None (full autotune). Passing this
# exact object emits FutureWarning in _resolve_meta.


_EP_CACHE = {}


def episodes(diet):
    """Split diet into per-episode dicts. Cached by diet identity (panel calls
    this dozens of times on the same arrays — same math, no recompute)."""
    key = (id(diet), len(diet["rew"]))
    hit = _EP_CACHE.get(key)
    if hit is not None and hit[0] is diet:
        return hit[1]
    # Evict stale ids (diet gc'd and id reused by a new object).
    _EP_CACHE.clear()
    ep = diet.get("episode")
    if ep is None:
        ep = np.arange(len(diet["rew"]))
    out, idx = [], np.argsort(ep, kind="stable")
    cur, start = ep[idx[0]], 0
    ids = []
    for k, i in enumerate(idx):
        if ep[i] != cur:
            ids.append(idx[start:k])
            cur, start = ep[i], k
    ids.append(idx[start:])
    _mt = diet.get("mu_take")
    _done = diet.get("done")
    if _done is None:
        _done = np.zeros(len(diet["rew"]), dtype=np.float32)
    for ii in ids:
        out.append(
            {
                "obs": diet["obs"][ii],
                "act": diet["act"][ii],
                "rew": diet["rew"][ii],
                "done": _done[ii],
            }
        )
        if _mt is not None:
            out[-1]["mu_take"] = _mt[ii]
        else:
            out[-1]["mu_take"] = diet["mu"][ii, diet["act"][ii]]
    _EP_CACHE[key] = (diet, out)
    return out


def _taken_probs(cand, obs, act, temperature):
    from .protocols import safe_probs

    probs = safe_probs(cand.action_probs(obs, temperature=temperature))
    return probs[np.arange(len(act)), act]


def _uncapped_rhos(diet, cand, temperature):
    """Per-episode uncapped per-step ratios. One pass; caps applied after."""
    rhos = []
    for ep in episodes(diet):
        pi = _taken_probs(cand, ep["obs"], ep["act"], temperature)
        rhos.append(pi / np.maximum(ep["mu_take"], PROB_FLOOR))
    return rhos


def _resolve_meta(diet, gamma, meta):
    """None -> full autotune. Dict -> fill missing keys from data.

    resolve_meta already merges overrides; this wrapper only guarantees a
    temps grid for hand-made legacy dicts that predate autotune.
    """
    import warnings

    if meta is META:
        warnings.warn(
            "estimators.META is deprecated (frozen v2 literals); pass meta=None for autotune.",
            FutureWarning,
            stacklevel=3,
        )
    m, _ = resolve_meta(diet, gamma, meta)
    if m.get("temps") is None:
        fp = diet_fingerprint(diet, gamma)
        m["temps"] = resolve_temps(fp["max_len"])
    return m


def select_temperature(diet, cand, gamma, meta=None):
    """Sharpest proxy whose episode-weight ESS clears the target fraction.

    Target and grid both autotuned from the diet unless overridden in meta.
    """
    check_candidate(cand)
    m = _resolve_meta(diet, gamma, meta)
    target = m.get("target_ess_frac")
    if target is None:
        fp = diet_fingerprint(diet, gamma)
        target = resolve_target_ess(fp["n_episodes"])
    temps = m.get("temps")
    if temps is None:
        fp = diet_fingerprint(diet, gamma)
        temps = resolve_temps(fp["max_len"])
    eps = episodes(diet)
    n = len(eps)
    best = {"temperature": temps[-1], "ess_frac": 0.0}
    for t in temps:
        ws = []
        for ep in eps:
            pi = _taken_probs(cand, ep["obs"], ep["act"], t)
            cum = np.cumprod(np.clip(pi / np.maximum(ep["mu_take"], PROB_FLOOR), 0, WEIGHT_CEIL))
            ws.append(float(cum[-1] * (gamma ** (len(cum) - 1))))
        ws = np.asarray(ws)
        ef = float((ws.sum() ** 2) / max((ws**2).sum(), VAR_FLOOR)) / n
        if ef >= target:
            return {"temperature": t, "ess_frac": ef}
        best = {"temperature": t, "ess_frac": ef}
    return best  # coverage too thin for any sharpness: softest + honest receipt


def _ep_returns(ep, gamma):
    T = len(ep["act"])
    r = ep["rew"]
    return np.array([np.sum(r[t:] * (gamma ** np.arange(T - t))) for t in range(T)])


def _ep_disc_returns(diet, gamma):
    from .receipts import _segment_disc_returns

    return _segment_disc_returns(diet, gamma).tolist()


def plausible_bounds(diet, gamma):
    """Physically possible discounted-return range [lo, hi] for this reward
    scale and horizon. A policy may terminate at any step in [1, L], so the
    bound admits early termination: lo = min(0, rmin*G), hi = max(0, rmax*G).
    Using observed per-reward min/max without the early-termination term
    collapses on all-positive-reward envs (CartPole, r=+1 every step) to a
    single point and clamps every value estimate to that point."""
    rmax = float(np.max(diet["rew"])) if len(diet["rew"]) else 1.0
    rmin = float(np.min(diet["rew"])) if len(diet["rew"]) else 0.0
    L = 1
    try:
        L = int(np.max(np.unique(diet["episode"], return_counts=True)[1]))
    except Exception:
        L = 1
    G = (1.0 - gamma**L) / max(1.0 - gamma, 1e-9)
    return min(0.0, rmin * G), max(0.0, rmax * G)


def panel(
    diet,
    cand,
    gamma,
    meta=None,
    fqe_cfg=None,
    cand_id=None,
    cache_dir=None,
    weights_hash=None,
    ensemble_K=None,
    dice_steps=None,
    magic_B=None,
    fast=False,
):
    """Full panel for one candidate/diet. Returns estimates + per-episode
    contributions (for bootstrapping) + the adaptive settings actually used.

    meta=None and fqe_cfg=None both mean full autotune from the diet.
    Pass partial dicts to override individual keys. cand_id+cache_dir enable
    the weights cache (exact on hit; needs weights_hash in production).
    ensemble_K/dice_steps/magic_B pin screen-tier budgets (None = autotune).
    fast=True skips the non-gating saddle loops (MIS, GradientDICE) and the
    other non-gating diagnostics (WDR, MAGIC, FVE, support) — a ~3-4x panel
    speedup with zero change to the HOLD/DEPLOY decision (which reads only
    DR, FQE soft+argmax, MB, LSTDQ). Skipped keys are None with receipts.
    Result includes "timing" (per-stage wall seconds).
    """
    import time as _time
    from .protocols import validate_diet

    check_candidate(cand)
    validate_diet(diet)
    _t0 = _time.perf_counter()
    _last = [_t0]
    timing = {}

    def _tick(stage):
        now = _time.perf_counter()
        timing[stage] = round(now - _last[0], 3)
        _last[0] = now

    m = _resolve_meta(diet, gamma, meta)
    fqe_cfg = dict(fqe_cfg) if fqe_cfg else {}
    sel = select_temperature(diet, cand, gamma, m)
    temp = sel["temperature"]
    _tick("temp_select")
    eps = episodes(diet)
    rhos_all = _uncapped_rhos(diet, cand, temp)
    q = m.get("clip_quantile")
    if q is None:
        fp = diet_fingerprint(diet, gamma)
        q = resolve_clip_quantile(fp["N"])
    cap = float(np.quantile(np.concatenate(rhos_all), q))
    cap = max(cap, RHO_CAP_FLOOR)
    T_ep = len(eps)
    from .autotune import capped_cumprod

    is_vals, wis_num, wsum, ws = [], 0.0, 0.0, []
    cum_hits = []
    for ep, rho_u in zip(eps, rhos_all):
        rho = np.clip(rho_u, 0.0, cap)
        T = len(rho)
        disc = gamma ** np.arange(T)
        cum, hit = capped_cumprod(rho)
        cum_hits.append(hit)
        is_vals.append(float(np.sum(cum * disc * ep["rew"])))
        w_ep = float(cum[-1])
        ws.append(w_ep)
        wis_num += w_ep * float(np.sum(disc * ep["rew"]))
        wsum += w_ep
    ws = np.asarray(ws)
    is_est = float(np.mean(is_vals))
    wis_est = float(wis_num / max(wsum, PROB_FLOOR))
    _tick("weights")
    qnet, _, fqe_info = fit_fqe(
        diet,
        cand,
        gamma,
        fqe_cfg,
        temp,
        cand_id=cand_id,
        cache_dir=cache_dir,
        weights_hash=weights_hash,
    )
    _tick("fqe_single")
    # DM headline = VAL-WEIGHTED ensemble mean (single-seed DM is lottery
    # debt; plain mean lets one collapsed seed drag it (v15 expert: 2.2 among
    # ~20s). Members earn weight by inverse held-out Bellman error; ties
    # recover the plain mean. qnet (single fit) stays for DR/WDR/MAGIC.
    from .direct import ensemble_fqe as _efqe_fn

    efqe = _efqe_fn(
        diet,
        cand,
        gamma,
        fqe_cfg,
        temp,
        K=ensemble_K,
        cand_id=cand_id,
        cache_dir=cache_dir,
        weights_hash=weights_hash,
    )
    dm_est = float(efqe["val_weighted_mean"])
    _tick("ensemble")
    dr_vals = doubly_robust_values(diet, cand, gamma, qnet, temp, cap)
    dr_est = float(np.mean(dr_vals))
    # Horizon-free step-level DR (see step_dr_values): survives long horizons
    # where trajectory DR's ESS collapses. Kept as a corroborating witness.
    _step = step_dr_values(diet, cand, gamma, qnet, temp, cap)
    dr_step_est = _step["est"]
    step_ess_frac = _step["ess_frac"]
    step_clip = _step["clipped_frac"]
    _tick("dr")
    ess = float((ws.sum() ** 2) / max((ws**2).sum(), VAR_FLOOR))
    ess_frac = ess / T_ep
    lo, hi = m.get("blend_lo"), m.get("blend_hi")
    if lo is None or hi is None:
        fp = diet_fingerprint(diet, gamma)
        lo_a, hi_a = resolve_blend_range(fp["n_episodes"])
        lo = lo_a if lo is None else lo
        hi = hi_a if hi is None else hi
    lam = float(np.clip((ess_frac - lo) / max(hi - lo, VAR_FLOOR), 0.0, 1.0))
    # Robust blend (v4+): never average in a DR the gate would reject.
    # If DR claims below-behavior while DM says above (or DR CI is wider
    # than the DM scale), the weights collapsed — trust DM (lam=0).
    # Receipts record the override; blend can no longer be worse than both.
    from .receipts import behavior_stats as _bstat

    _b = _bstat(diet, gamma)
    _dr_std = float(np.std(dr_vals)) if len(dr_vals) > 1 else 0.0
    blend_guard = None
    if dr_est < _b["mean"] <= dm_est:
        lam, blend_guard = 0.0, "dr_below_anchor_dm_above"
    elif _dr_std > 5 * max(abs(dm_est - _b["mean"]), abs(_b["std"]), 1.0):
        lam, blend_guard = 0.0, "dr_ci_dwarfs_signal"
    blended = lam * dr_est + (1 - lam) * dm_est
    # ------------------------------------------------------------------
    # Fast mode (v20): the decision uses only DR, FQE (soft + argmax), MB,
    # LSTDQ. MIS + GradientDICE (two saddle loops) are ~70% of panel wall
    # time and nothing gates on them; WDR/MAGIC/FVE/support are likewise
    # non-gating diagnostics. fast=True skips them and reports the skip.
    # Full mode (fast=False) is unchanged for archival/registry runs.
    # ------------------------------------------------------------------
    from .receipts import behavior_stats, bootstrap_ci
    from .meta import slope_lite
    from .direct import lstdq
    from .model_based import learn_dynamics_ensemble, rollout_estimate

    fp = diet_fingerprint(diet, gamma)
    if fast:
        mis_est, mis_diag, support = None, {"skipped": "fast"}, {}
        wdr_vals, wdr_est = [], None
        magic_est, magic_w, magic_ci, magic_dropped = None, [], None, []
        fve_dm, gd_est, gd_info = None, None, {"skipped": "fast"}
        _tick("skipped_non_gating")
    else:
        from .marginalized import learn_ratio, mis_diagnostics, support_stats
        from .trajectory import wdr_values, magic_lite
        from .direct import fve
        from .variants_dice import learn_ratio_gd

        mis_steps = fqe_cfg.get("mis_steps")
        if mis_steps is None:
            from .autotune import resolve_mis_cfg

            mis_steps = resolve_mis_cfg(diet, fqe_cfg)["steps_max"]
        w_fn, _ = learn_ratio(
            diet,
            cand,
            gamma,
            temperature=temp,
            steps=mis_steps,
            cand_id=cand_id,
            cache_dir=cache_dir,
            weights_hash=weights_hash,
        )
        mis_diag = mis_diagnostics(diet, w_fn, gamma)
        mis_est = mis_diag["mis"]
        _tick("mis")
        support = support_stats(diet, w_fn)
        _tick("support")
        wdr_vals = wdr_values(diet, cand, gamma, qnet, temp, cap)
        wdr_est = float(np.mean(wdr_vals))
        _tick("wdr")
        magic_est, magic_w, magic_labels, magic_ci, magic_dropped = magic_lite(
            diet, cand, gamma, qnet, temp, cap, B=magic_B
        )
        _tick("magic")
        fve_r = fve(
            diet,
            cand,
            gamma,
            fqe_cfg,
            temp,
            cand_id=cand_id,
            cache_dir=cache_dir,
            weights_hash=weights_hash,
        )
        fve_dm = fve_r["dm"]
        _tick("fve")
        w_fn_g, gd_info = learn_ratio_gd(
            diet,
            cand,
            gamma,
            temperature=temp,
            steps=dice_steps,
            cand_id=cand_id,
            cache_dir=cache_dir,
            weights_hash=weights_hash,
        )
        gd_est = mis_diagnostics(diet, w_fn_g, gamma)["mis"]
        _tick("gdice")
    lstd = lstdq(diet, cand, gamma, temp)
    _tick("lstdq")
    # ---- anchor probe (pure-offline calibration) --------------------------
    # The behavior policy's value is KNOWN from the log (its empirical return).
    # Evaluate the behavior policy with the SAME fitted estimator and record the
    # gap to that known value: it is a direct, per-dataset measurement of the
    # estimator's systematic bias, which no amount of seed-ensembling can see.
    fqe_behavior = None
    try:
        import torch as _torch

        _st = np.unique(np.asarray(diet["episode"]), return_index=True)[1]
        with _torch.no_grad():
            _q = qnet(_torch.as_tensor(diet["obs"][_st]).float()).cpu().numpy()
        _mu = diet.get("mu")
        if _mu is not None and np.asarray(_mu).shape[0] == len(diet["obs"]):
            _m = np.asarray(_mu, dtype=np.float64)[_st]
            fqe_behavior = float(np.mean(np.sum(_m * _q, axis=1)))
        else:
            _am = np.asarray(diet["act"])[_st].astype(int)
            fqe_behavior = float(np.mean(_q[np.arange(len(_st)), _am]))
    except Exception:
        fqe_behavior = None
    # Sharp-policy estimate (v20): evaluate the DEPLOYABLE policy — the
    # candidate's argmax — with FQE. This is the estimand that matters, and
    # unlike trajectory reweighting FQE needs no action support, so it works
    # where sharp ESS is 0 (v20 validations: argmax DM 80/87/101 vs truths
    # 81/94/99; soft-proxy FQE read 49/55/30). Divergence guard: a value above
    # the maximum possible discounted return is a numerical blow-up (expert
    # uniform read 88428), not an estimate -> invalidate, fail safe.
    from .protocols import ArgmaxPolicy as _Argmax

    sharp_cand = _Argmax(cand)
    # Ensemble the deployable-policy FQE so we have an UNCERTAINTY on the
    # value that actually ships. Without it, a systematically-optimistic FQE
    # on out-of-distribution argmax actions ships worse policies (Acrobot: FQE
    # read -83 while truth was -99). The contract then compares the LOWER
    # confidence bound, not the point.
    _Ksh = int(ensemble_K or (fqe_cfg or {}).get("ensemble_K") or 2)
    _sh_dms = []
    sharp_info = {}
    for _k in range(max(1, _Ksh)):
        _cfgk = dict(fqe_cfg) if fqe_cfg else {}
        _cfgk["seed"] = int((fqe_cfg or {}).get("seed", 0)) + 1000 * _k
        _cidk = (
            None
            if cand_id is None
            else (f"{cand_id}__sharp" if _k == 0 else f"{cand_id}__sharp{_k}")
        )
        _, _dmk, _ik = fit_fqe(
            diet,
            sharp_cand,
            gamma,
            _cfgk,
            1.0,
            cand_id=_cidk,
            cache_dir=cache_dir,
            weights_hash=weights_hash,
        )
        _sh_dms.append(float(_dmk))
        sharp_info = _ik
    sharp_dm = float(np.mean(_sh_dms))
    sharp_dis = float(np.std(_sh_dms))
    sharp_info = dict(sharp_info, disagreement=round(sharp_dis, 4), K=len(_sh_dms))
    # Divergence bound from the magnitude of the POSSIBLE discounted return.
    # Must not use max(reward): envs whose peak reward is 0 (MountainCar /
    # Acrobot step reward is -1 or 0) gave bound = max(0,1)*1.5 = 1.5, which
    # invalidated every legitimate negative value and forced a blind HOLD.
    _pb_lo, _pb_hi = plausible_bounds(diet, gamma)
    _bound = max(abs(_pb_lo), abs(_pb_hi), 1.0) * 1.5
    if (not np.isfinite(sharp_dm)) or abs(sharp_dm) > _bound:
        sharp_info = dict(sharp_info, diverged=True, bound=round(_bound, 1), raw=sharp_dm)
        sharp_dm = None
    else:
        # Lower confidence bound the contract compares against the bar.
        sharp_info = dict(sharp_info, lower=round(float(sharp_dm - 1.96 * sharp_dis), 3))
    _tick("sharp")
    # Dynamics shares the panel budget source (was bare autotune: same rule,
    # now one knob). Cached per diet (see model_based).
    step_fn, dyn_info = learn_dynamics_ensemble(
        diet, K=int((fqe_cfg or {}).get("dyn_ensemble", 3)), cfg=fqe_cfg, cache_dir=cache_dir
    )
    _tick("dynamics")
    mb_r = rollout_estimate(diet, cand, gamma, step_fn, temp)
    _tick("rollout")
    # Second DEPLOYABLE-policy witness: roll out the argmax policy in the world
    # model. Soft-proxy witnesses (DR, FQE-soft, MB-soft) judge a different,
    # weaker policy and can be below bar while the shippable policy clears it
    # (mixed: soft 50-55 vs argmax 87-91). Two argmax witnesses corroborate
    # the same estimand: FQE-argmax + MB-argmax.
    mb_sharp = rollout_estimate(diet, sharp_cand, gamma, step_fn, 1.0)
    _tick("rollout_sharp")
    bstat = behavior_stats(diet, gamma)
    # SLOPE-lite over members with CIs (ensemble interval approx for FQE).
    # FQE half-width scales with ensemble disagreement: 2.0 is the Gaussian
    # ~95% multiplier (z=1.96 rounded), not a tuning dial.
    FQE_Z = 2.0
    B, al = resolve_bootstrap(len(eps), m.get("bootstrap_B"), m.get("ci_alpha"))
    ci = {
        "dr": (bootstrap_ci(dr_vals, B, al)[0], bootstrap_ci(dr_vals, B, al)[1]),
        "is": (bootstrap_ci(is_vals, B, al)[0], bootstrap_ci(is_vals, B, al)[1]),
        "wis": (bstat["mean"], bstat["mean"]),
        "fqe_dm": (dm_est - FQE_Z * efqe["disagreement"], dm_est + FQE_Z * efqe["disagreement"]),
        "anchor": (
            bootstrap_ci(_ep_disc_returns(diet, gamma), B, al)[0],
            bootstrap_ci(_ep_disc_returns(diet, gamma), B, al)[1],
        ),
    }
    if wdr_vals:
        ci["wdr"] = (bootstrap_ci(wdr_vals, B, al)[0], bootstrap_ci(wdr_vals, B, al)[1])
    if magic_ci is not None:
        ci["magic"] = magic_ci
    pts = {"anchor": bstat["mean"], "fqe_dm": dm_est, "wis": wis_est, "dr": dr_est, "is": is_est}
    for _k, _v in (("mis", mis_est), ("wdr", wdr_est), ("magic", magic_est)):
        if _v is not None:
            pts[_k] = _v
    slope_pick, slope_val = slope_lite(pts, ci)
    below_anchor = sorted([k for k, v in pts.items() if k != "anchor" and v < bstat["mean"]])
    _tick("cis_slope")
    timing["total"] = round(_time.perf_counter() - _t0, 3)
    # Plausibility bound: any value estimate must lie in the physically
    # possible discounted-return range for this reward scale and horizon. An
    # estimate outside it is a numerical blow-up, not a value (Acrobot: soft
    # FQE read +196 when the whole range is negative). We do NOT silently
    # substitute a bound — the raw value is reported and flagged, and the
    # estimator contract (contracts.py) REJECTS it from the decision. Silent
    # clamping is how the CartPole 99.3 degeneracy hid for so long.
    _lo, _hi = plausible_bounds(diet, gamma)
    clamp = {}
    for _n, _v in (("fqe", dm_est), ("mb", float(mb_r["mb"]))):
        if not (_lo - 1e-6 <= _v <= _hi + 1e-6):
            clamp[_n] = {"raw": round(float(_v), 2), "lo": round(_lo, 2), "hi": round(_hi, 2)}
    # Level headline: MEDIAN of {FQE-vwmean, LSTDQ, MB}.
    import statistics as _st

    level_parts = {"fqe": dm_est, "lstdq": float(lstd["dm"]), "mb": float(mb_r["mb"])}
    level_est = float(_st.median(level_parts.values()))
    return {
        "is": is_est,
        "wis": wis_est,
        "dr": dr_est,
        "fqe_dm": dm_est,
        "efqe": efqe,
        "lstdq": lstd,
        "fve_dm": fve_dm,
        "mb": mb_r,
        "mb_sharp": mb_sharp,
        "gdice_mis": gd_est,
        "gd_info": gd_info,
        "dyn_info": dyn_info,
        "anchor": round(bstat["mean"], 1),
        "magic": magic_est,
        "magic_w": magic_w,
        "wdr": wdr_est,
        "magic_dropped": magic_dropped,
        "slope_pick": slope_pick,
        "slope_val": round(slope_val, 1),
        "below_anchor": below_anchor,
        "mis": mis_est,
        "mis_info": mis_diag,
        "support": support,
        "blended": blended,
        "lambda_dr": lam,
        "blend_guard": blend_guard,
        "cum_cap_hit": float(np.mean(cum_hits)) if cum_hits else 0.0,
        "is_vals": is_vals,
        "dr_vals": dr_vals,
        "ep_weights": ws,
        "ep_returns": [float(x) for x in _ep_disc_returns(diet, gamma)],
        "ess_frac": ess_frac,
        "temperature": temp,
        "rho_cap": cap,
        "fqe_behavior": fqe_behavior,
        "fqe_info": fqe_info,
        "timing": timing,
        "level_est": level_est,
        "level_parts": level_parts,
        "value_clamp": clamp,
        "dr_step": dr_step_est,
        "step_ess_frac": step_ess_frac,
        "step_clip_frac": step_clip,
        "dr_step_adv": _step["adv"],
        "dr_step_num": _step["num"],
        "dr_step_den": _step["den"],
        "dr_step_adv_num": _step["adv_num"],
        "dr_step_adv_den": _step["adv_den"],
        "sharp_dm": sharp_dm,
        "sharp_info": sharp_info,
        "mu_source": diet.get("_mu_source", "logged"),
        "mu_info": diet.get("_mu_info", {}),
    }


def fit_fqe(
    diet,
    cand,
    gamma,
    fqe_cfg=None,
    temperature=None,
    cand_id=None,
    cache_dir=None,
    weights_hash=None,
):
    """FQE with validation early-stop. Budget follows difficulty; the curve
    decides, via held-out Bellman error with patience.

    fqe_cfg=None -> autotuned from diet (resolve_fqe_cfg). temperature=None
    -> resolved via select_temperature (ESS-targeted). cand_id+cache_dir
    enable the weights cache (exact on hit; needs weights_hash in production
    so stale weights under a reused id can't hit).
    """
    import os
    import sys
    import torch
    import torch.nn.functional as F

    sys.path.insert(
        0,
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
    )
    from algorithms.deep.networks import QNetwork

    check_candidate(cand)
    if temperature is None:
        temperature = select_temperature(diet, cand, gamma)["temperature"]
    cfg = resolve_fqe_cfg(diet, fqe_cfg, gamma)
    steps_max = cfg["steps_max"]
    batch = cfg["batch"]
    hidden = cfg["hidden"]
    seed = cfg["seed"]
    lr = cfg["lr"]
    holdout = cfg["holdout"]
    from .autotune import resolve_device

    device = resolve_device(cfg.get("device"))
    N, in_dim = len(diet["obs"]), diet["obs"].shape[1]
    nA = diet["nA"]
    from .training import seed_all as _seed_all

    _seed_all(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    cut = max(N - holdout, 1)
    tr, va = perm[:cut], perm[cut:]
    q = QNetwork(in_dim, hidden, nA).to(device)
    # Target network with POLYAK soft updates (deadly-triad fix, v20).
    # Prior scheme hard-synced the target once per eval round (~steps_max/40,
    # e.g. ~2242 steps) — far too slow: value information propagated at ~40%
    # of truth (toy with exact answer 100 read 39.5), which is why FQE read
    # 18-29 on policies whose true value is 94-99. Polyak tracks fast enough
    # to propagate within budget (toy tau in [0.01,0.1] all read 99-100%).
    # tau derived from budget (no magic constant): smaller step budgets get a
    # faster-tracking target; large budgets a slower one, both in the safe band.
    tau = float(cfg.get("target_tau", min(0.02, max(0.005, 1000.0 / max(steps_max, 1)))))
    qt = QNetwork(in_dim, hidden, nA).to(device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr, weight_decay=1e-4)
    # Weights cache: exact on hit (seeded training is bit-identical).
    _ckey = None
    if cand_id is not None and cache_dir:
        from .cache import diet_hash, make_key, load as _cload

        _ckey = make_key(
            "fqe", diet_hash(diet), cand_id, cfg, temperature, weights_hash or "noweights"
        )
        _hit = _cload(cache_dir, _ckey, map_location=device)
    else:
        _hit = None
    # Force float32 + device: numpy float64 mocks / python-float promotion
    # otherwise yields Double loss vs Float params -> backward RuntimeError.
    obs = torch.as_tensor(diet["obs"]).float().to(device)
    A = torch.as_tensor(diet["act"], dtype=torch.long).to(device)
    R = torch.as_tensor(diet["rew"]).float().unsqueeze(1).to(device)
    O2 = torch.as_tensor(diet["obs2"]).float().to(device)
    D = torch.as_tensor(diet["done"]).float().unsqueeze(1).to(device)
    # Target clamp to the physically plausible discounted-return range:
    # without it, bootstrapping on sparse/large-magnitude rewards can blow the
    # Q estimate out of range (Acrobot uniform read +88428), which then fails
    # the plausibility guard and forces a blind HOLD. A value outside the
    # possible range is not information, so clamping the bootstrapped target
    # keeps FQE usable exactly where coverage is hardest.
    _clip_lo, _clip_hi = plausible_bounds(diet, gamma)

    # Hot-loop surgery: candidate probs are batch-queried ~steps_max times
    # (numpy->GPU->forward->CPU roundtrips dominating wall time on small nets).
    # The candidate is frozen during OPE: precompute full-diet probs ONCE and
    # index per batch. Bit-identical values (independent per-row softmax).
    from .protocols import safe_probs as _sp

    _P = _sp(cand.action_probs(np.asarray(diet["obs"]), temperature=temperature))
    _P2 = _sp(cand.action_probs(np.asarray(diet["obs2"]), temperature=temperature))
    _P_va = _P[va]

    def val_err():
        with torch.no_grad():
            pi2 = torch.as_tensor(_P_va).float().to(device)
            v2 = (pi2 * qt(O2[va])).sum(1, keepdim=True)
            tgt = R[va] + gamma * (1 - D[va]) * v2
            tgt = tgt.clamp(_clip_lo, _clip_hi)
            return float(F.mse_loss(q(obs[va]).gather(1, A[va].unsqueeze(1)), tgt))

    from .training import govern

    _syncs = [0]

    def _step(n, lr):
        last = None
        for _ in range(n):
            i = rng.integers(0, cut, batch)
            ii = tr[i]
            with torch.no_grad():
                pi2 = torch.as_tensor(_P2[ii]).float().to(device)
                v2 = (pi2 * qt(O2[ii])).sum(1, keepdim=True)
                tgt = R[ii] + gamma * (1 - D[ii]) * v2
                tgt = tgt.clamp(_clip_lo, _clip_hi)
            pred = q(obs[ii]).gather(1, A[ii].unsqueeze(1))
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            # Polyak soft target update every step (propagation within budget).
            # Batched foreach ops: one CUDA launch per op instead of one per
            # parameter tensor (the per-tensor zip loop was pure overhead).
            with torch.no_grad():
                _qp = list(q.parameters())
                _tq = list(qt.parameters())
                torch._foreach_mul_(_tq, 1.0 - tau)
                torch._foreach_add_(_tq, _qp, alpha=tau)
            _syncs[0] += 1
            last = float(loss.item())
        return last

    # Budget-mode govern (val_fn=None): cosine LR + NaN guard, NO early-stop /
    # best-restore. Rationale (v20): FQE's Bellman validation is self-
    # referential, so best-restore crowned a degenerate snapshot — governed
    # FQE read -0.0 at 80k where the same loop without restore read 48.2
    # (novice proxy truth ~52). Value-based selection needs a grounded metric;
    # until then, run the budget and keep the final weights. Bellman error is
    # still computed as a receipt. Patience is irrelevant in this mode.
    _gov = (
        govern({"q": q}, [opt], _step, None, cfg)
        if _hit is None
        else {"steps": 0, "n_evals": 0, "best_val": None, "stopped": "cache-hit", "lr_final": lr}
    )
    with torch.no_grad():
        _val_receipt = val_err()
    best = _val_receipt
    _cache_hit = _hit is not None
    if _cache_hit:
        # Cache hit: bit-identical weights, skip straight to eval.
        q.load_state_dict(_hit["q"])
        qt.load_state_dict(_hit["q"])
        best = _hit["info"]["val_bellman"]
    elif _ckey is not None:
        from .cache import save as _csave

        _csave(cache_dir, _ckey, {"q": q.state_dict(), "info": {"val_bellman": best}})
    # Eval on CPU: training was GPU-accelerated, downstream DR/WDR/MAGIC are
    # cheap forwards — keep them device-agnostic.
    q = q.cpu()
    dm_vals = []
    for ep in episodes(diet):
        pi0 = cand.action_probs(ep["obs"][:1], temperature=temperature)[0]
        with torch.no_grad():
            qv = q(torch.as_tensor(ep["obs"][:1]).float()).cpu().numpy()[0]
        dm_vals.append(float(np.dot(pi0, qv)))
    return (
        q,
        float(np.mean(dm_vals)),
        {
            "steps": _gov["steps"],
            "n_evals": _gov["n_evals"],
            "val_bellman": best,
            "stopped": _gov["stopped"],
            "lr_final": _gov["lr_final"],
            "cache_hit": _cache_hit,
            "target": "polyak-soft",
            "target_tau": tau,
            "target_syncs": _syncs[0],
            "device": device,
            "min_steps": int(cfg.get("min_steps", steps_max)),
            "under_budget": bool(
                not cfg.get("allow_under_budget")
                and steps_max < 0.5 * int(cfg.get("min_steps", steps_max))
            ),
            "cfg": {
                k: cfg[k]
                for k in (
                    "hidden",
                    "batch",
                    "steps_max",
                    "eval_every",
                    "patience",
                    "lr",
                    "holdout",
                    "seed",
                    "device",
                )
                if k in cfg
            },
        },
    )


# ---------------------------------------------------------------------------
# Estimator registry (contract enforcement, not just docs): panel must return
# exactly these top-level estimate keys; gate/judge/slope may only read keys
# listed here. Adding an estimator = implement it + add its row here + wire
# one panel call + one gate row + slope ORDER entry + judge rank note + tests.
# test_registry_keys_match_panel fails CI otherwise.
# ---------------------------------------------------------------------------
ESTIMATORS = {
    # name: (result keys produced, needs, torch?)
    "is": (("is", "is_vals"), "weights", False),
    "wis": (("wis",), "weights", False),
    "fqe": (("fqe_dm", "fqe_info"), "candidate", True),
    "dr": (("dr", "dr_vals"), "qnet+weights", True),
    "mis": (("mis", "mis_info", "support"), "candidate", True),
    "wdr": (("wdr",), "qnet+weights", True),
    "magic": (("magic", "magic_w", "magic_dropped"), "qnet+weights", True),
    "efqe": (("efqe",), "candidate", True),
    "lstdq": (("lstdq",), "candidate", False),
    "fve": (("fve_dm",), "candidate", True),
    "mb": (("mb", "dyn_info"), "candidate", True),
    "gdice": (("gdice_mis", "gd_info"), "candidate", True),
    "level": (("level_est", "level_parts"), "direct trio", False),
    "sharp": (("sharp_dm", "sharp_info"), "candidate", True),
}


def dr_terms(ep, cand, gamma, qnet, temperature, cap):
    """Shared per-episode DR ingredients. DR, WDR and MAGIC were three
    copy-pasted loops that drifted (DR capped cumprod, WDR/MAGIC raw cumprod
    -> 1e15 blowups; DR sanitized probs, WDR/MAGIC didn't). One function:
    capped cumulative weights + sanitized probs everywhere. Returns numpy
    dict {cum, disc, v0, corr, T}. corr = wcorr * control-variate vector, so
    DR = v0 + sum(corr); WDR normalizes corr by cross-episode weights;
    MAGIC truncates corr at horizons."""
    import torch
    from .autotune import capped_cumprod
    from .protocols import safe_probs

    try:
        dev = next(qnet.parameters()).device
    except Exception:
        dev = torch.device("cpu")
    pi = _taken_probs(cand, ep["obs"], ep["act"], temperature)
    rho = np.clip(pi / np.maximum(ep["mu_take"], PROB_FLOOR), 0.0, cap)
    cum, _ = capped_cumprod(rho)
    T = len(rho)
    disc = gamma ** np.arange(T)
    with torch.no_grad():
        x = torch.as_tensor(ep["obs"]).float().to(dev)
        x2 = torch.as_tensor(np.vstack([ep["obs"][1:], ep["obs"][-1:]])).float().to(dev)
        q = qnet(x).cpu().numpy()
        q2 = qnet(x2).cpu().numpy()
    pi_full = safe_probs(cand.action_probs(ep["obs"], temperature=temperature))
    pi2 = safe_probs(cand.action_probs(np.asarray(x2.cpu()), temperature=temperature))
    v, v2 = (pi_full * q).sum(1), (pi2 * q2).sum(1)
    qa = q[np.arange(T), ep["act"]]
    wcorr = np.ones(T)
    wcorr[1:] = cum[:-1]
    ctrl = np.zeros(T)
    for t in range(T):
        nxt = v2[t] if t < T - 1 else 0.0
        ctrl[t] = disc[t] * (ep["rew"][t] + gamma * nxt - qa[t])
    return {"cum": cum, "disc": disc, "v0": float(v[0]), "corr": wcorr * ctrl, "T": T}


def doubly_robust_values(diet, cand, gamma, qnet, temperature, cap):
    """Per-episode DR contributions (returned, not averaged — bootstrapping
    resamples these). Cumulative weights capped (overflow guard). Device-aware:
    follows qnet's device."""
    vals = []
    for ep in episodes(diet):
        t = dr_terms(ep, cand, gamma, qnet, temperature, cap)
        vals.append(float(t["v0"] + np.sum(t["corr"])))
    return vals


def step_dr_values(diet, cand, gamma, qnet, temperature, cap):
    """Horizon-free STEP-level DR (no cumulative products).

    Trajectory DR multiplies per-step ratios over the whole episode; any tiny
    mismatch compounds and ESS collapses (measured 0.000 on 200-500 step
    envs). Here each transition is weighted by its OWN ratio only, and the
    per-step DR terms are self-normalized:

        rho_t = clip(pi(a_t|s_t)/mu(a_t|s_t), 0, cap)
        d_t   = V(s_t) + rho_t * (r_t + gamma V(s_{t+1}) - Q(s_t,a_t))
        est   = sum_t rho_t d_t / sum_t rho_t          (self-normalized)
        ess_frac = (sum rho)^2 / (sum rho^2) / N_steps

    Estimand is the behavior-state-weighted candidate value (a marginalized
    analogue), not V(s0) — reported alongside trajectory DR, and its step ESS
    stays healthy on long horizons where the trajectory ESS does not. Returns
    {"est", "ess_frac", "clipped_frac", "num", "den", "adv", "adv_num",
    "adv_den"} where est is the marginalized candidate value and adv is the
    matched-estimand advantage (rho-1)*A vs the behavior baseline; num/den and
    adv_num/adv_den are the per-episode numerator/denominator pieces."""
    import torch
    from .protocols import safe_probs

    N = len(diet["obs"])
    obs = torch.as_tensor(diet["obs"]).float()
    O2 = torch.as_tensor(diet["obs2"]).float()
    mu_take = diet.get("mu_take")
    if mu_take is None:
        mu_take = diet["mu"][np.arange(N), diet["act"]]
    mu_take = np.asarray(mu_take, dtype=np.float64)
    p = safe_probs(cand.action_probs(diet["obs"], temperature=temperature))
    p2 = safe_probs(cand.action_probs(diet["obs2"], temperature=temperature))
    pi_take = p[np.arange(N), diet["act"]].astype(np.float64)
    rho_u = pi_take / np.maximum(mu_take, PROB_FLOOR)
    rho = np.clip(rho_u, 0.0, cap)
    clipped_frac = float((rho_u > cap).mean())
    try:
        dev = next(qnet.parameters()).device
    except Exception:
        dev = torch.device("cpu")
    with torch.no_grad():
        q = qnet(obs.to(dev)).cpu().numpy()
        q2 = qnet(O2.to(dev)).cpu().numpy()
    qa = q[np.arange(N), diet["act"]]
    v = (p * q).sum(1)
    v2 = (p2 * q2).sum(1)
    rew = np.asarray(diet["rew"], dtype=np.float64)
    done = np.asarray(
        diet.get("done") if diet.get("done") is not None else np.zeros(N), dtype=np.float64
    )
    # A_t = candidate Bellman residual at the LOGGED (behavior) action, i.e.
    # the candidate's advantage estimate at behavior actions. The matched
    # candidate-minus-behavior estimand on the same behavior states is then the
    # per-decision DR advantage (rho-1)*A (the shared V(s) baseline cancels),
    # which is horizon-free and has no self-normalization mismatch.
    A = rew + gamma * (1.0 - done) * v2 - qa
    d = v + rho * A
    wsum = max(rho.sum(), PROB_FLOOR)
    est = float(np.sum(rho * d) / wsum)
    ess_frac = float((rho.sum() ** 2) / max((rho**2).sum(), VAR_FLOOR) / max(N, 1))
    adv = (rho - 1.0) * A
    # Per-episode pieces so the judge can bootstrap a p-value over EPISODES
    # (correct for the ratio) for the matched advantage.
    ep = np.asarray(diet["episode"])
    order = np.argsort(ep, kind="stable")
    bounds = np.flatnonzero(np.diff(ep[order]) != 0) + 1
    num, den, adv_num, adv_den = [], [], [], []
    for seg in np.split(order, bounds):
        num.append(float(np.sum(rho[seg] * d[seg])))
        den.append(float(np.sum(rho[seg])))
        adv_num.append(float(np.sum(adv[seg])))
        adv_den.append(float(len(seg)))
    adv_est = float(np.sum(adv_num) / max(np.sum(adv_den), 1e-12))
    return {
        "est": est,
        "ess_frac": ess_frac,
        "clipped_frac": clipped_frac,
        "num": num,
        "den": den,
        "adv": adv_est,
        "adv_num": adv_num,
        "adv_den": adv_den,
    }
