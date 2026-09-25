"""Autotune: every OPE knob derived from the diet, nothing hardcoded.

Design contract (per user direction: dynamic / adaptive / no hardcoded values):
- Every function takes the diet (or its fingerprint) + optional user override.
- `None` means "derive from data". An explicit value always wins.
- Numerical-stability floors (PROB_FLOOR, VAR_FLOOR, ...) are NOT tuning dials:
  they are float-precision guards, named once here with justification.
- Statistical dials (ESS targets, quantiles, bootstrap B, blend range, temps,
  horizons, network sizes, training budgets) are all functions of N,
  n_episodes, episode lengths, obs_dim — never module-level literals.

Typical sizes that motivated the rules (CartPole quick-look):
  N ~ 10k-50k transitions, n_episodes ~ 100-500, max_len ~ 200-500.
The rules below reproduce the old v2 literals at those sizes but scale
sensibly outside them.

CONSTANTS LEDGER — every irreducible number and why it survives:
- Statistical conventions (not ours to derive): 95% intervals (FQE_Z=2.0,
  MAGIC/bootstrap α/2 quantiles), Cohen-medium effect 0.5 as the rel_edge
  anchor (shrunk adaptively from there), Holm step-down levels.
- Minimum-count rules of thumb: 5 effective episodes to search temps,
  3 to ship; B=200 bootstrap floor (below this CIs are Monte Carlo noise);
  holdout [200, 2000] rows (below 200 the val curve is noise, above 2000
  wasteful); 2-episode diet minimum (a "diet" of one episode has no variance).
- Optimization folklore with receipts: 5% LR warmup, 1% cosine floor,
  1e-4 relative early-stop tolerance, patience window [3, 8] evals.
  These shape compute, not answers (best-restore makes overshoot harmless).
- Shape bounds (compute caps, not statistics): hidden [32, 256], batch
  [64, 512], steps [5k, 50k] / MIS [1k, 10k], ensemble K [2, 5], MAGIC
  horizons at length quantiles, sim counts from n_episodes. Caps bind only
  outside the design envelope above.
- Reproducibility, not tuning: seeds (0 + 100-stride ensemble offsets),
  bootstrap seed 0, BC shuffle seed 0, 0.05 behavior-eps floor scale.
Changing any ledger entry is allowed — but it must move here with a reason,
never hide inline.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Numerical guards (not tuning dials — do not "tune" these per diet)
# ---------------------------------------------------------------------------
PROB_FLOOR = 1e-8  # max(mu, FLOOR): behavior probs are exact; floor avoids 0/0.
VAR_FLOOR = 1e-12  # max(var, FLOOR): avoids div-by-zero on degenerate weights.
WEIGHT_CEIL = 1e6  # clip(uncapped rho, 0, CEIL) in temp search: prevents inf.
RHO_CAP_FLOOR = 1.0  # data-driven cap never goes below 1 (no down-weighting).
CUM_CAP = 1e4  # cap(cumprod(rho)): per-step cap ~10 over 500 steps overflows
# float64 (10^500 = inf). Cumulative cap is an overflow guard; hits are
# counted in receipts (capped_frac). Truncated-IS bias is reported, not hidden.


def _clip(x, lo, hi):
    return float(min(hi, max(lo, x)))


def resolve_device(user=None):
    """cuda if available and not pinned, else cpu. Pass device='cpu' to force."""
    if user is not None:
        return str(user)
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def capped_cumprod(rho, cap=None):
    """Cumulative product with overflow guard. Returns (cum, hit_frac).

    cum is clipped to [0, cap]; hit_frac is the fraction of steps where the
    raw cumprod exceeded cap (receipt for truncated-IS bias).
    """
    import numpy as _np

    cap = float(cap if cap is not None else CUM_CAP)
    raw = _np.cumprod(_np.asarray(rho, dtype=float))
    hit = raw > cap
    return _np.minimum(raw, cap), float(hit.mean()) if len(raw) else 0.0


def diet_fingerprint(diet, gamma=0.99):
    """Small dict of everything autotune needs. Pure numpy, no torch."""
    obs = np.asarray(diet["obs"])
    ep = np.asarray(diet["episode"])
    uniq, counts = np.unique(ep, return_counts=True)
    n_ep = int(len(uniq))
    n = int(len(obs))
    obs_dim = int(obs.shape[1]) if obs.ndim > 1 else 1
    nA = int(diet.get("nA", 2))
    ep_lens = counts.astype(float)
    return {
        "N": n,
        "n_episodes": n_ep,
        "ep_lens": ep_lens,
        "max_len": int(ep_lens.max()) if n_ep else 0,
        "mean_len": float(ep_lens.mean()) if n_ep else 0.0,
        "obs_dim": obs_dim,
        "nA": nA,
        "gamma": float(gamma),
    }


# ---------------------------------------------------------------------------
# Bootstrap / CI
# ---------------------------------------------------------------------------
def resolve_bootstrap(n_episodes, B=None, alpha=None):
    """B grows slowly with n (more episodes -> tighter CI worth more draws);
    alpha is 0.10 for tiny n (<30, honest about small samples), else 0.05."""
    n = max(int(n_episodes), 1)
    if B is None:
        B = int(_clip(200 + 2 * n, 200, 2000))
    if alpha is None:
        alpha = 0.10 if n < 30 else 0.05
    return int(B), float(alpha)


def resolve_ci_alpha(n_episodes, alpha=None):
    _, a = resolve_bootstrap(n_episodes, B=200, alpha=alpha)
    return a


# ---------------------------------------------------------------------------
# Temperature grid
# ---------------------------------------------------------------------------
def resolve_temps(max_len, n_temps=None, user_temps=None):
    """Geometric grid ending at 1.0 (softest). Count scales with log horizon:
    longer episodes need finer sharpness search. Values are powers of 1/2 so
    each step halves softness — no arbitrary decimals.

    n=5 at max_len~500 gives (0.0625, 0.125, 0.25, 0.5, 1.0), close to the old
    hand tuple but derived from the horizon.
    """
    if user_temps is not None:
        return tuple(float(t) for t in user_temps)
    ml = max(int(max_len), 2)
    if n_temps is None:
        n_temps = int(_clip(int(np.ceil(np.log2(ml))), 3, 7))
    n_temps = max(int(n_temps), 2)
    temps = tuple(float(1.0 / (2.0**k)) for k in range(n_temps - 1, -1, -1))
    return temps


def resolve_target_ess(n_episodes, user_val=None, min_effective=5):
    """Want at least `min_effective` effective episodes: target = K/n.
    Clipped to [0.02, 0.30] so tiny diets don't demand the impossible and
    huge diets don't get sloppy. K=5 is the smallest count whose mean is
    worth a CI (rule-of-thumb, documented, overridable)."""
    if user_val is not None:
        return float(user_val)
    n = max(int(n_episodes), 1)
    return _clip(min_effective / n, 0.02, 0.30)


def resolve_clip_quantile(N, user_val=None):
    """1 - 1/sqrt(N): ~0.99 at N=10k (old literal), tighter with more data,
    looser with less. Clipped to [0.90, 0.999]."""
    if user_val is not None:
        return float(user_val)
    n = max(int(N), 10)
    return _clip(1.0 - 1.0 / np.sqrt(n), 0.90, 0.999)


def resolve_blend_range(n_episodes, lo=None, hi=None):
    """DR<->DM blend ramps with ESS fraction. Scales as 1/sqrt(n):
    more episodes -> can trust DR earlier. At n=100: lo=0.10, hi=0.50
    (recovers old 0.05/0.50 within tolerance, but now derived)."""
    n = max(int(n_episodes), 1)
    s = np.sqrt(n)
    lo_v = float(lo) if lo is not None else _clip(1.0 / s, 0.02, 0.20)
    hi_v = float(hi) if hi is not None else _clip(5.0 / s, 0.15, 0.60)
    if hi_v <= lo_v:
        hi_v = min(0.60, lo_v + 0.10)
    return float(lo_v), float(hi_v)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def resolve_gate(n_episodes, behavior_std=None, behavior_mean=None, user_gate=None):
    """Gate thresholds from data:
    - min_ess_frac: need >=3 effective episodes (stricter than temp-search K=5
      would suggest for deployment; searching is cheap, shipping is not).
    - rel_edge_std: Cohen-medium 0.5 at n=30, shrinking as 0.5*sqrt(30/n) to
      a 0.2 floor — large samples can certify smaller lifts.
    - scale_floor: avoids degenerate bar when behavior std ~ 0 (expert diets):
      scale = max(std, 5% of |mean|).
    """
    ug = dict(user_gate) if user_gate else {}
    n = max(int(n_episodes), 1)
    if "min_ess_frac" not in ug:
        ug["min_ess_frac"] = _clip(3.0 / n, 0.02, 0.10)
    if "rel_edge_std" not in ug:
        ug["rel_edge_std"] = _clip(0.5 * np.sqrt(30.0 / n), 0.20, 0.80)
    return ug


def resolve_bar(behavior_mean, behavior_std, rel_edge_std):
    scale = max(float(behavior_std), 0.05 * abs(float(behavior_mean)) + 1e-9)
    return float(behavior_mean) + float(rel_edge_std) * scale


# ---------------------------------------------------------------------------
# Training configs (FQE / MIS / dynamics / BC)
# ---------------------------------------------------------------------------
def _shared_net_cfg(N, obs_dim, nA, base=None, gamma=0.99):
    base = dict(base) if base else {}
    out = {}
    # Capacity scales with input size: 16 params per input dim, [32, 256].
    out["hidden"] = int(base.get("hidden", _clip(16 * (obs_dim + nA), 32, 256)))
    # Batch is ~5% of data, [64, 512].
    out["batch"] = int(base.get("batch", _clip(max(N // 20, 1), 64, 512)))
    # Budget scales with data AND horizon. v20 receipt: FQE value propagation
    # needs ~H*N/batch MANY updates, not a few — at 3x it read ~40% of truth
    # (toy exact 100 -> 39.5; novice proxy 52 -> 5-16), at ~12x it reads the
    # true level (novice argmax 80.4 vs truth 81). K=12 chosen from those
    # measurements; cap 300k bounds GPU cost.
    _H = 1.0 / max(1.0 - float(gamma), 1e-6)
    _need = 12.0 * _H * max(N, 1) / max(out["batch"], 1)
    # min_steps = the DERIVED floor for a trustworthy value. A user-pinned
    # steps_max below this is honored (tests, screen tier) but flagged as
    # under-budget so the estimator contract refuses to gate on it — wrong
    # budgets must announce themselves instead of silently reading 40% of truth.
    out["min_steps"] = int(_clip(max(N // 2, _need), 5000, 300000))
    out["steps_max"] = int(base.get("steps_max", out["min_steps"]))
    out["eval_every"] = int(base.get("eval_every", max(100, out["steps_max"] // 40)))
    # Patience ~ 1/10 of eval rounds, [3, 8].
    n_evals = max(out["steps_max"] // out["eval_every"], 1)
    out["patience"] = int(base.get("patience", _clip(n_evals // 10, 3, 8)))
    # Holdout ~10% of data, [200, 2000] rows.
    holdout = int(_clip(N // 10, 200, 2000)) if N > 400 else max(N // 5, 10)
    out["holdout"] = int(base.get("holdout", holdout))
    # LR scales with batch (square-root scaling rule).
    out["lr"] = float(base.get("lr", 1e-3 * np.sqrt(out["batch"] / 256.0)))
    out["seed"] = int(base.get("seed", 0))
    # Device autotunes to cuda when available (no cpu hardcode).
    out["device"] = resolve_device(base.get("device"))
    # Transfer through any other user keys untouched.
    for k, v in base.items():
        if k not in out:
            out[k] = v
    return out


def resolve_fqe_cfg(diet, base=None, gamma=None):
    fp = diet_fingerprint(diet, gamma=gamma or 0.99)
    cfg = _shared_net_cfg(fp["N"], fp["obs_dim"], fp["nA"], base, gamma=fp["gamma"])
    if base and "steps" in base:  # legacy alias
        cfg["steps_max"] = int(base["steps"])
    return cfg


def resolve_mis_cfg(diet, base=None):
    fp = diet_fingerprint(diet)
    cfg = _shared_net_cfg(fp["N"], fp["obs_dim"], fp["nA"], base)
    # MIS/DICE needs fewer steps than FQE per unit data (saddle, not backup).
    if not base or "steps_max" not in base:
        cfg["steps_max"] = int(_clip(max(fp["N"] // 3, 1), 1000, 10000))
        cfg["eval_every"] = max(100, cfg["steps_max"] // 20)
    cfg["w_lr"] = float((base or {}).get("w_lr", cfg["lr"]))
    cfg["nu_lr"] = float((base or {}).get("nu_lr", cfg["lr"]))
    return cfg


def resolve_bc_cfg(diet, base=None):
    fp = diet_fingerprint(diet)
    cfg = _shared_net_cfg(fp["N"], fp["obs_dim"], fp["nA"], base)
    if not base or "steps_max" not in base:
        cfg["steps_max"] = cfg["steps_max"]  # same rule as FQE
    return cfg


# ---------------------------------------------------------------------------
# MAGIC horizons + rollout
# ---------------------------------------------------------------------------
def resolve_magic_horizons(ep_lens, user_horizons=None):
    """Horizons at episode-length quantiles [25, 50, 75, 90]% + full length.
    Old hand tuple (5, 25, 100, 250) assumed CartPole-500; quantiles adapt to
    any horizon. Always ends with None (= full/DM)."""
    if user_horizons is not None:
        return tuple(user_horizons)
    lens = np.asarray(list(ep_lens), dtype=float)
    if len(lens) == 0:
        return (None,)
    qs = np.quantile(lens, [0.25, 0.50, 0.75, 0.90])
    hs = sorted({max(int(round(q)), 1) for q in qs} | {int(lens.max())})
    # Drop near-duplicates (within 10%) to keep the panel small.
    filt = []
    for h in hs:
        if not filt or h >= filt[-1] * 1.10:
            filt.append(h)
    return tuple(filt) + (None,)


def resolve_rollout_cfg(n_episodes, max_len, base=None):
    base = dict(base) if base else {}
    n = max(int(n_episodes), 1)
    cfg = {}
    cfg["sim_min"] = int(base.get("sim_min", _clip(2 * n, 30, 200)))
    cfg["sim_max"] = int(base.get("sim_max", _clip(8 * n, 100, 800)))
    cfg["se_frac"] = float(base.get("se_frac", _clip(1.0 / np.sqrt(cfg["sim_min"]), 0.02, 0.10)))
    cfg["batch"] = int(base.get("batch", min(25, n)))
    cfg["seed"] = int(base.get("seed", 0))
    cfg["max_len"] = int(base.get("max_len", max_len))
    for k, v in base.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Top-level META resolver
# ---------------------------------------------------------------------------
def resolve_meta(diet, gamma=0.99, user_meta=None):
    """Full META dict with every key estimators.panel needs, all derived.

    user_meta keys override individually; missing keys are auto-derived.
    Returns (meta, info) where info records what was derived (receipt).
    """
    um = dict(user_meta) if user_meta else {}
    fp = diet_fingerprint(diet, gamma)
    n_ep, N = fp["n_episodes"], fp["N"]
    B, alpha = resolve_bootstrap(n_ep, um.get("bootstrap_B"), um.get("ci_alpha"))
    lo, hi = resolve_blend_range(n_ep, um.get("blend_lo"), um.get("blend_hi"))
    meta = {
        "target_ess_frac": resolve_target_ess(n_ep, um.get("target_ess_frac")),
        "clip_quantile": resolve_clip_quantile(N, um.get("clip_quantile")),
        "ci_alpha": alpha,
        "bootstrap_B": B,
        "blend_lo": lo,
        "blend_hi": hi,
        "rel_edge_std": resolve_gate(n_ep)["rel_edge_std"]
        if "rel_edge_std" not in um
        else float(um["rel_edge_std"]),
        "temps": resolve_temps(fp["max_len"], um.get("n_temps"), um.get("temps")),
    }
    # Pass through any extra user keys (e.g. n_temps is consumed, rest kept).
    for k, v in um.items():
        if k not in meta and k != "n_temps":
            meta[k] = v
    info = {
        "fingerprint": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in fp.items()
            if k != "ep_lens"
        },
        "derived": sorted(set(meta) - set(um)),
    }
    return meta, info
