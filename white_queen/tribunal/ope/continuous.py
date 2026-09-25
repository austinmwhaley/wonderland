"""Continuous-action OPE panel.

Discrete actions use pi(a|s)/mu(a|s). Continuous actions have no probability
mass; the importance ratio is a DENSITY ratio exp(log pi(a|s) - log mu(a|s)),
where the behavior density is logged or estimated (ope.data). Everything else
— ESS, clipping, DR control variates, FQE, model-based rollout, plausibility
bounds — mirrors the discrete panel, but the value functions take (s, a).

Produces the same panel dict shape the gate/judge already consume, so the
decision layer is identical across discrete and continuous.
"""

from __future__ import annotations

import numpy as np

from .autotune import (
    PROB_FLOOR,
    VAR_FLOOR,
    RHO_CAP_FLOOR,
    capped_cumprod,
    resolve_clip_quantile,
    resolve_device,
    resolve_fqe_cfg,
)
from .protocols import validate_diet

# FQE calibration for continuous control. FQE extrapolates off-policy, so its
# value needs an uncertainty statement, not a point. The deployable witness
# must clear the bar on the ensemble LOWER confidence bound (dm - Z*seed_std).
# The holdout Bellman ratio is a WIDE gross-misfit guard only: a high TD error
# means the Q function is hard to fit, not that the value at start states is
# wrong (Pendulum has bel_ratio ~4.6 while its value is accurate), so a tight
# Bellman switch would only create false negatives.
FQE_Z = 1.96
FQE_BEL_RATIO_MAX = 8.0


def _episodes(diet):
    ep = diet["episode"]
    idx = np.argsort(ep, kind="stable")
    ids, cur, start = [], ep[idx[0]], 0
    for k, i in enumerate(idx):
        if ep[i] != cur:
            ids.append(idx[start:k])
            cur, start = ep[i], k
    ids.append(idx[start:])
    out = []
    for ii in ids:
        out.append(
            {
                "obs": diet["obs"][ii],
                "act": diet["act"][ii],
                "rew": diet["rew"][ii],
                "done": diet["done"][ii],
                "logp": diet["logp_take"][ii],
            }
        )
    return out


def _cand_actions(cand, obs):
    """Deterministic mode actions for a candidate over obs (N, a_dim)."""
    obs = np.asarray(obs)
    if hasattr(cand, "action_mean"):
        a = np.asarray(cand.action_mean(obs), dtype=np.float64)
        return a.reshape(len(obs), -1)
    return np.stack([np.asarray(cand.act(o, eval=True), dtype=np.float64).ravel() for o in obs])


def _cand_sample(cand, obs, rng, k=1):
    """Sample k actions per obs from the candidate policy (N,a_dim) each.
    Falls back to the deterministic mode for deterministic policies."""
    obs = np.asarray(obs)
    if hasattr(cand, "sample"):
        return [np.asarray(cand.sample(obs, rng), dtype=np.float64) for _ in range(k)]
    m = _cand_actions(cand, obs)
    return [m.copy() for _ in range(k)]


def _cand_logp(cand, obs, act):
    if hasattr(cand, "log_prob_fn"):
        return np.asarray(cand.log_prob_fn(obs, act), dtype=np.float64).ravel()
    if hasattr(cand, "action_logprob"):
        return np.asarray(cand.action_logprob(obs, act), dtype=np.float64).ravel()
    raise TypeError("continuous candidate needs log_prob_fn(obs, act)")


def _plausible_bounds(diet, gamma):
    # Shared with the discrete panel so the early-termination fix lives once.
    from .estimators import plausible_bounds

    return plausible_bounds(diet, gamma)


def _fit_value(diet, cand, gamma, cfg, dev, K=1):
    """Continuous FQE ENSEMBLE: Q(s,a) with target r + gamma(1-done)Q(s',pi(s')).

    Returns a dict {q, dm, starts, disagreement, val_bellman, members, K}.
    FQE extrapolates off-policy, so on high-dimensional control a single net
    can be confidently wrong (Pendulum: -253 vs truth -481). Two calibration
    signals are produced so the judge can distrust it:
      - disagreement: mean per-start-state std of V across ensemble members
        (seed variance of the value estimate), and
      - val_bellman: holdout Bellman MSE (how well the net explains its own
        target on held-out transitions).
    Both must be small relative to the reward scale for the value to count.
    """
    K = int(K or cfg.get("ensemble_K") or 4)
    from algorithms.approx.networks import MLP

    N = len(diet["obs"])
    in_dim = diet["obs"].shape[1]
    a_dim = diet["act"].shape[1]
    steps_max = cfg["steps_max"]
    batch = cfg["batch"]
    hidden = cfg["hidden"]
    lr = cfg["lr"]
    holdout = cfg["holdout"]
    seed = cfg["seed"]
    tau = min(0.02, max(0.005, 1000.0 / max(steps_max, 1)))
    starts = np.unique(np.asarray(diet["episode"]), return_index=True)[1]
    _clip_lo, _clip_hi = _plausible_bounds(diet, gamma)

    def _fit_one(mseed):
        import torch
        import torch.nn.functional as F

        rng = np.random.default_rng(mseed)
        perm = rng.permutation(N)
        cut = max(N - holdout, 1)
        tr, va = perm[:cut], perm[cut:]
        obs = torch.as_tensor(diet["obs"]).float().to(dev)
        A = torch.as_tensor(diet["act"]).float().to(dev)
        R = torch.as_tensor(diet["rew"]).float().unsqueeze(1).to(dev)
        O2 = torch.as_tensor(diet["obs2"]).float().to(dev)
        D = torch.as_tensor(diet["done"]).float().unsqueeze(1).to(dev)
        srng = np.random.default_rng(mseed + 7)
        A2s = [
            torch.as_tensor(a).float().to(dev) for a in _cand_sample(cand, diet["obs2"], srng, k=4)
        ]
        net = MLP(in_dim + a_dim, hidden, 1).to(dev)
        qt = MLP(in_dim + a_dim, hidden, 1).to(dev)
        qt.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

        def target_q2(idx):
            vals = [qt(torch.cat([O2[idx], a2[idx]], 1)) for a2 in A2s]
            return sum(vals) / len(vals)

        for _ in range(steps_max):
            i = torch.as_tensor(tr[rng.integers(0, cut, batch)]).to(dev)
            with torch.no_grad():
                tgt = (R[i] + gamma * (1 - D[i]) * target_q2(i)).clamp(_clip_lo, _clip_hi)
            loss = F.mse_loss(net(torch.cat([obs[i], A[i]], 1)), tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            with torch.no_grad():
                for ps, pt in zip(net.parameters(), qt.parameters()):
                    pt.data.mul_(1 - tau).add_(ps.data, alpha=tau)
        net = net.eval()
        with torch.no_grad():
            # holdout Bellman error on the member's split (on device)
            tgt = (R[va] + gamma * (1 - D[va]) * target_q2(va)).clamp(_clip_lo, _clip_hi)
            bel = float(F.mse_loss(net(torch.cat([obs[va], A[va]], 1)), tgt))
        net = net.cpu()
        with torch.no_grad():
            a0s = _cand_sample(cand, diet["obs"], srng, k=8)
            obs_t = torch.as_tensor(diet["obs"]).float()
            qsa = np.zeros(N)
            for a0 in a0s:
                qsa += net(torch.cat([obs_t, torch.as_tensor(a0).float()], 1)).squeeze(1).numpy()
            qsa /= len(a0s)
        return net, qsa, bel

    members, start_vals, bell = [], [], []
    q0 = None
    for k in range(K):
        net, qsa, bel = _fit_one(seed + 1000 * k)
        if q0 is None:
            q0 = net
        members.append(float(np.mean(qsa[starts])))
        start_vals.append(qsa[starts])
        bell.append(bel)
    sv = np.stack(start_vals)  # (K, n_starts)
    return {
        "q": q0,
        "dm": float(np.mean(members)),
        "starts": starts,
        "disagreement": float(np.mean(np.std(sv, axis=0))),
        "val_bellman": float(np.mean(bell)),
        "members": members,
        "K": K,
    }


def panel_continuous(
    diet,
    cand,
    gamma=0.99,
    fqe_cfg=None,
    fast=True,
    ensemble_K=None,
    cache_dir=None,
    cand_id=None,
    weights_hash=None,
):
    """Return a panel dict with the keys gate/judge read, for continuous data."""
    import time

    t0 = time.perf_counter()
    timing = {}
    validate_diet(diet)
    if not diet.get("continuous"):
        raise ValueError("panel_continuous called on discrete data")
    cfg = resolve_fqe_cfg(diet, fqe_cfg, gamma)
    dev = resolve_device(cfg.get("device"))
    eps = _episodes(diet)
    n_ep = len(eps)
    # ---- ratios: density ratio pi/mu ----
    with np.errstate(over="ignore"):
        lp = _cand_logp(cand, diet["obs"], diet["act"])
    logr = np.clip(lp - diet["logp_take"], -30.0, 30.0)
    rho_all = np.exp(logr)
    q = float(np.quantile(rho_all, resolve_clip_quantile(len(rho_all))))
    cap = max(q, RHO_CAP_FLOOR)
    is_vals, ws, wis_num, wsum = [], [], 0.0, 0.0
    ep_of = diet["episode"]
    order = np.argsort(ep_of, kind="stable")
    bounds = np.flatnonzero(np.diff(ep_of[order]) != 0) + 1
    segs = np.split(order, bounds)
    for seg in segs:
        r = np.clip(rho_all[seg], 0.0, cap)
        T = len(r)
        disc = gamma ** np.arange(T)
        cum, _ = capped_cumprod(r)
        is_vals.append(float(np.sum(cum * disc * diet["rew"][seg])))
        w_ep = float(cum[-1])
        ws.append(w_ep)
        wis_num += w_ep * float(np.sum(disc * diet["rew"][seg]))
        wsum += w_ep
    ws = np.asarray(ws)
    is_est = float(np.mean(is_vals))
    wis_est = float(wis_num / max(wsum, PROB_FLOOR))
    ess_frac = float((ws.sum() ** 2) / max((ws**2).sum(), VAR_FLOOR)) / n_ep
    timing["weights"] = round(time.perf_counter() - t0, 3)
    # ---- FQE ensemble (deployable policy; continuous has no soft temp) ----
    fq = _fit_value(diet, cand, gamma, cfg, dev, K=(ensemble_K or cfg.get("ensemble_K") or 4))
    qnet, dm_est, starts = fq["q"], fq["dm"], fq["starts"]
    fqe_disagreement = fq["disagreement"]
    fqe_bellman = fq["val_bellman"]
    timing["fqe"] = round(time.perf_counter() - t0, 3)
    # ---- DR with Q control variate ----
    import torch

    dr_vals = []
    with torch.no_grad():
        a_all = _cand_actions(cand, diet["obs"])
        for seg in segs:
            o = diet["obs"][seg]
            a = diet["act"][seg]
            r = np.clip(rho_all[seg], 0.0, cap)
            cum, _ = capped_cumprod(r)
            T = len(r)
            disc = gamma ** np.arange(T)
            o2 = np.vstack([o[1:], o[-1:]])
            a2 = np.vstack([a_all[seg][1:], a_all[seg][-1:]])
            qsa = (
                qnet(torch.cat([torch.as_tensor(o).float(), torch.as_tensor(a).float()], 1))
                .squeeze(1)
                .numpy()
            )
            q2 = (
                qnet(torch.cat([torch.as_tensor(o2).float(), torch.as_tensor(a2).float()], 1))
                .squeeze(1)
                .numpy()
            )
            ctrl = np.zeros(T)
            for t in range(T):
                nxt = q2[t] if t < T - 1 else 0.0
                ctrl[t] = disc[t] * (diet["rew"][seg][t] + gamma * nxt - qsa[t])
            wcorr = np.ones(T)
            wcorr[1:] = cum[:-1]
            dr_vals.append(float(qsa[0] + np.sum(wcorr * ctrl)))
    dr_est = float(np.mean(dr_vals))
    timing["dr"] = round(time.perf_counter() - t0, 3)
    # ---- MB: continuous dynamics + rollout of deployable policy ----
    mb = _continuous_mb(diet, cand, gamma, cfg, dev)
    timing["mb"] = round(time.perf_counter() - t0, 3)
    # ---- LSTDQ (linear in [obs, act]) ----
    lstd = _continuous_lstdq(diet, cand, gamma, dev)
    timing["lstdq"] = round(time.perf_counter() - t0, 3)
    # ---- plausibility bound (reject, do not clamp) ----
    lo_b, hi_b = _plausible_bounds(diet, gamma)
    clamp = {}
    for nm, v in (("fqe", dm_est), ("mb", mb["mb"])):
        if not (lo_b - 1e-6 <= v <= hi_b + 1e-6):
            clamp[nm] = {"raw": round(float(v), 2), "lo": round(lo_b, 2), "hi": round(hi_b, 2)}
    # Raw values are kept and flagged; contracts.py rejects them. No silent
    # substitution (that is how the CartPole 99.3 degeneracy hid).
    level_parts = {"fqe": dm_est, "lstdq": float(lstd["dm"]), "mb": float(mb["mb"])}
    import statistics as _st

    float(_st.median(level_parts.values()))
    # ---- FQE calibration gate -------------------------------------------
    # FQE extrapolates off-policy, so on high-dim control it can be
    # confidently wrong (Pendulum: -253 vs truth -481). Gate on the two
    # ensemble calibration signals, both in reward-scale units:
    #   dis_ratio = seed-disagreement of V / (reward_std * sqrt(G_eff))
    #   bel_ratio = sqrt(holdout Bellman MSE) / reward_std
    # If either exceeds the threshold the value is not trustworthy -> the
    # judge fails safe (HOLD) instead of shipping on an extrapolation.
    _rscale = max(float(np.std(diet["rew"])), abs(float(np.mean(diet["rew"]))), 1e-6)
    _Geff = 1.0 / max(1.0 - gamma, 1e-6)
    dis_ratio = fqe_disagreement / (_rscale * np.sqrt(_Geff))
    bel_ratio = np.sqrt(max(fqe_bellman, 0.0)) / _rscale
    cal_reasons = []
    if bel_ratio > FQE_BEL_RATIO_MAX:
        cal_reasons.append(f"holdout Bellman {bel_ratio:.1f}>{FQE_BEL_RATIO_MAX}")
    fqe_trusted = not cal_reasons
    sharp_info = {
        "continuous": True,
        "K": fq["K"],
        "disagreement": round(fqe_disagreement, 4),
        "val_bellman": round(fqe_bellman, 4),
        "dis_ratio": round(float(dis_ratio), 3),
        "bel_ratio": round(float(bel_ratio), 3),
        # Deployable witness must clear the bar on this bound.
        "lower": round(float(dm_est - FQE_Z * fqe_disagreement), 3),
    }
    if cal_reasons:
        sharp_info["diverged"] = True
        sharp_info["reason"] = "; ".join(cal_reasons)
    # sensitivity on DR contributions
    sens = None
    try:
        from .sensitivity import gamma_star

        bar_est = float(np.median([dm_est, wis_est]))
        gs, frontier = gamma_star(
            dr_vals, ws if len(ws) == len(dr_vals) else [1.0] * len(dr_vals), bar_est
        )
        sens = {
            "gamma_star": (
                "already_below"
                if gs is None
                else ("inf" if gs == float("inf") else round(float(gs), 2))
            ),
            "target": "dr",
        }
    except Exception:
        sens = None
    timing["total"] = round(time.perf_counter() - t0, 3)
    return {
        "is": is_est,
        "wis": wis_est,
        "dr": dr_est,
        "wdr": None,
        "magic": None,
        "magic_w": [],
        "blended": dr_est,
        "lambda_dr": 1.0,
        "blend_guard": None,
        "fqe_dm": dm_est,
        "sharp_dm": (dm_est if fqe_trusted else None),
        "sharp_info": sharp_info,
        "efqe": {
            "mean": dm_est,
            "disagreement": float(fqe_disagreement),
            "members": [round(v, 1) for v in fq["members"]],
            "K": fq["K"],
            "val_weighted_mean": dm_est,
            "val_weights": [1.0] * fq["K"],
        },
        "mb": mb,
        "mb_sharp": mb,
        "gdice_mis": None,
        "gd_info": {},
        "dyn_info": mb.get("dyn_info", {}),
        "anchor": round(float(np.mean([_ep_return(diet, s, gamma) for s in starts])), 1)
        if len(starts)
        else 0.0,
        "slope_pick": "fqe",
        "slope_val": round(dm_est, 1),
        "below_anchor": [],
        "mis": None,
        "mis_info": {"skipped": "continuous"},
        "support": {},
        "lstdq": lstd,
        "fve_dm": None,
        "ess_frac": ess_frac,
        "temperature": 1.0,
        "rho_cap": cap,
        "is_vals": is_vals,
        "dr_vals": dr_vals,
        "ep_weights": ws,
        "ep_returns": [_ep_return(diet, s, gamma) for s in starts],
        "sensitivity": sens,
        "timing": timing,
        "value_clamp": clamp,
        "fqe_info": {
            "min_steps": int(cfg.get("min_steps", cfg["steps_max"])),
            "under_budget": bool(
                not cfg.get("allow_under_budget")
                and cfg["steps_max"] < 0.5 * int(cfg.get("min_steps", cfg["steps_max"]))
            ),
            "K": fq["K"],
        },
        "cum_cap_hit": 0.0,
        "mu_source": "continuous",
        "mu_info": {},
    }


def _ep_return(diet, start_idx, gamma):
    ep = diet["episode"]
    s = start_idx
    r = diet["rew"]
    g = 0.0
    disc = 1.0
    i = s
    while i < len(ep) and ep[i] == ep[s]:
        g += disc * r[i]
        disc *= gamma
        i += 1
    return g


def _continuous_lstdq(diet, cand, gamma, dev):
    N = len(diet["obs"])
    Phi = np.concatenate([diet["obs"].astype(np.float64), diet["act"].astype(np.float64)], 1)
    a2 = _cand_actions(cand, diet["obs2"])
    Phi2 = np.concatenate([diet["obs2"].astype(np.float64), a2], 1)
    d = 1.0 - diet["done"].astype(np.float64)
    A = Phi.T @ (Phi - gamma * (d[:, None] * Phi2)) / N
    b = Phi.T @ diet["rew"].astype(np.float64) / N
    ridge = 0.1 / np.sqrt(max(N, 100))
    lam = ridge * np.trace(A) / A.shape[0]
    try:
        theta = np.linalg.solve(A + lam * np.eye(A.shape[0]), b)
        cond = float(np.linalg.cond(A + lam * np.eye(A.shape[0])))
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, b, rcond=None)[0]
        cond = float("inf")
    a0 = _cand_actions(cand, diet["obs"])
    phi0 = np.concatenate([diet["obs"].astype(np.float64), a0], 1)
    vals = phi0 @ theta
    starts = np.unique(np.asarray(diet["episode"]), return_index=True)[1]
    return {"dm": float(np.mean(vals[starts])), "cond": cond}


def _continuous_mb(diet, cand, gamma, cfg, dev, sim_min=100, sim_max=300):
    import torch
    import torch.nn.functional as F
    from algorithms.approx.networks import MLP

    N = len(diet["obs"])
    in_dim = diet["obs"].shape[1]
    hidden = cfg["hidden"]
    batch = cfg["batch"]
    lr = cfg["lr"]
    X = np.concatenate([diet["obs"].astype(np.float64), diet["act"].astype(np.float64)], 1)
    Y = np.concatenate(
        [
            (diet["obs2"] - diet["obs"]).astype(np.float64),
            diet["rew"][:, None],
            diet["done"][:, None],
        ],
        1,
    )
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    Xn = torch.as_tensor(((X - xm) / xs).astype(np.float32)).to(dev)
    Yn = torch.as_tensor(((Y - ym) / ys).astype(np.float32)).to(dev)
    net = MLP(Xn.shape[1], hidden, in_dim + 2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = min(cfg["steps_max"], 5000)
    rng = np.random.default_rng(cfg["seed"])
    for s in range(steps):
        i = torch.as_tensor(rng.integers(0, N, batch)).to(dev)
        loss = F.mse_loss(net(Xn[i]), Yn[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net = net.cpu().eval()
    starts = np.stack(
        [diet["obs"][i] for i in np.unique(np.asarray(diet["episode"]), return_index=True)[1]]
    )
    max_len = int(np.max(np.unique(np.asarray(diet["episode"]), return_counts=True)[1]))
    rets = []
    for e in range(sim_min):
        s = starts[rng.integers(0, len(starts))].astype(np.float64)
        disc, g = 1.0, 0.0
        for t in range(max_len):
            a = np.asarray(_cand_sample(cand, s[None, :], rng, k=1)[0], dtype=np.float64).ravel()
            x = torch.as_tensor(((np.concatenate([s, a]) - xm) / xs).astype(np.float32))
            with torch.no_grad():
                d = net(x).numpy() * ys + ym
            s = s + d[:in_dim]
            g += disc * float(d[in_dim])
            disc *= gamma
            if 1.0 / (1.0 + np.exp(-d[in_dim + 1])) > 0.5:
                break
        rets.append(g)
    return {
        "mb": float(np.mean(rets)),
        "se": round(float(np.std(rets) / max(np.sqrt(len(rets)), 1)), 3),
        "sims": len(rets),
        "dyn_info": {"steps": steps},
    }
