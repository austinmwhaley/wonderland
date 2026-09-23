"""Marginalized (stationary-ratio) OPE: horizon-free weights.

Trajectory cumprods detonate over 500-step episodes; stationary ratios don't
compound with horizon at all. Estimates w(s,a) = d^pi/d^mu via the DualDICE
saddle-point (Nachum et al. 2019), offline-only: no env, no behavior model
beyond logged rows, target actions sampled from the candidate proxy itself.
Value = self-normalized E_mu[w*r]. Validated directly against live truth —
the toy's privilege; in retail this panelist earns trust via the ensemble.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))


def _prep(diet, eps=1e-6):
    mu = diet["obs"].mean(0)
    sd = diet["obs"].std(0) + eps  # eps: z-score guard for constant obs dims.
    Z = lambda x: (np.asarray(x, dtype=np.float32) - mu) / sd
    return Z


def learn_ratio(diet, cand, gamma, temperature=1.0, hidden=None, steps=None,
                batch=None, seed=0, w_lr=None, nu_lr=None, cfg=None,
                cand_id=None, cache_dir=None, weights_hash=None):
    """Returns (w_fn, info) with w_fn(obs_batch, act_batch) -> ratios.

    hidden/steps/batch/lrs=None -> autotuned from diet (resolve_mis_cfg).
    cfg dict overrides per-key; explicit args beat cfg. cand_id+cache_dir
    enable the weights cache (exact on hit).
    """
    from algorithms.approx.networks import MLP
    from .autotune import resolve_mis_cfg
    from .protocols import check_candidate
    check_candidate(cand)
    base = dict(cfg) if cfg else {}
    if hidden is not None:
        base["hidden"] = hidden
    if steps is not None:
        base["steps_max"] = steps
    if batch is not None:
        base["batch"] = batch
    if w_lr is not None:
        base["w_lr"] = w_lr
    if nu_lr is not None:
        base["nu_lr"] = nu_lr
    if seed is not None:
        base["seed"] = seed
    ac = resolve_mis_cfg(diet, base)
    hidden, steps, batch = ac["hidden"], ac["steps_max"], ac["batch"]
    seed, w_lr, nu_lr = ac["seed"], ac["w_lr"], ac["nu_lr"]
    from .training import seed_all as _seed_all
    _seed_all(seed)
    rng = np.random.default_rng(seed)
    Z = _prep(diet)
    nA = diet["nA"]
    from .autotune import resolve_device
    dev = resolve_device(ac.get("device"))

    def onehot(a):
        oh = np.zeros((len(np.atleast_1d(a)), nA), dtype=np.float32)
        oh[np.arange(len(oh)), np.atleast_1d(a)] = 1.0
        return oh

    def feat(o, a):
        o = Z(o)
        return torch.as_tensor(np.concatenate([o, onehot(a)], 1), device=dev)

    w_net = MLP(Z(diet["obs"][:1]).shape[1] + nA, hidden, 1).to(dev)
    nu_net = MLP(Z(diet["obs"][:1]).shape[1] + nA, hidden, 1).to(dev)
    import torch.nn.functional as F
    opt_w = torch.optim.Adam(w_net.parameters(), lr=w_lr)
    opt_n = torch.optim.Adam(nu_net.parameters(), lr=nu_lr)
    N = len(diet["obs"])
    s0_pool = np.stack([ep["obs"][0] for ep in _episodes(diet)])
    from .protocols import sample_actions, safe_probs
    from .training import govern
    # Hot-loop surgery (see fit_fqe): candidate frozen during OPE; precompute
    # full-diet + pool probs once, index per step. Bit-identical rows.
    _P2full = safe_probs(cand.action_probs(np.asarray(diet["obs2"]),
                                           temperature=temperature))
    _P0pool = safe_probs(cand.action_probs(s0_pool, temperature=temperature))
    # Weights cache (exact on hit; saddle has no val to restore from, so the
    # receipt records budget completion either way).
    _ckey = None
    if cand_id is not None and cache_dir:
        from .cache import diet_hash, make_key, load as _cload
        _ckey = make_key("mis", diet_hash(diet), cand_id, ac, temperature,
                         weights_hash or "noweights")
        _hit = _cload(cache_dir, _ckey, map_location=dev)
    else:
        _hit = None
    _cache_hit = _hit is not None
    if _cache_hit:
        w_net.load_state_dict(_hit["w"])
        nu_net.load_state_dict(_hit["nu"])

    def _step(n, lr):
        last = None
        for _ in range(n):
            i = rng.integers(0, N, batch)
            s, a = diet["obs"][i], diet["act"][i]
            s2 = diet["obs2"][i]
            d = diet["done"][i].astype(np.float32)
            a2 = sample_actions(rng, _P2full[i])
            j = rng.integers(0, len(s0_pool), batch)
            a0 = sample_actions(rng, _P0pool[j])
            w = F.softplus(w_net(feat(s, a))).squeeze(1)
            with torch.no_grad():
                w2 = F.softplus(w_net(feat(s2, a2))).squeeze(1) * (1 - torch.as_tensor(d, device=dev))
            nu = nu_net(feat(s, a)).squeeze(1)
            nu0 = nu_net(feat(s0_pool[j], a0)).squeeze(1)
            # max_nu: (w - g w2) nu - (1-g) nu0 - .5 nu^2 ; min_w: (w - g w2) nu
            obj_nu = ((w.detach() - gamma * w2) * nu).mean() \
                - (1 - gamma) * nu0.mean() - 0.5 * (nu ** 2).mean()
            opt_n.zero_grad()
            (-obj_nu).backward()
            opt_n.step()
            w = F.softplus(w_net(feat(s, a))).squeeze(1)
            with torch.no_grad():
                w2 = F.softplus(w_net(feat(s2, a2))).squeeze(1) * (1 - torch.as_tensor(d, device=dev))
                nu_d = nu_net(feat(s, a)).squeeze(1)
            obj_w = ((w - gamma * w2) * nu_d).mean()
            opt_w.zero_grad()
            obj_w.backward()
            opt_w.step()
            last = float(obj_w.item())
        return last

    # Saddle mode: no val_fn (fake early-stop on minimax selects collapse).
    # Cosine schedule + budget + NaN guard; diagnostics reported, not optimized.
    if _cache_hit:
        _gov = {"steps": 0, "stopped": "cache-hit"}
    else:
        _gov = govern({"w": w_net, "nu": nu_net}, [opt_w, opt_n], _step, None,
                      {"steps_max": steps, "eval_every": max(steps // 20, 100),
                       "lr": w_lr, "patience": 0})
        if _ckey is not None:
            from .cache import save as _csave
            _csave(cache_dir, _ckey, {"w": w_net.state_dict(),
                                      "nu": nu_net.state_dict()})

    def w_fn(obs, act):
        with torch.no_grad():
            return F.softplus(w_net(feat(obs, act))).squeeze(1).cpu().numpy()

    with torch.no_grad():
        wall = F.softplus(w_net(feat(diet["obs"], diet["act"]))).squeeze(1).cpu().numpy()
    info = {"mean_w": round(float(wall.mean()), 3),
            "p99_w": round(float(np.quantile(wall, 0.99)), 2),
            "cache_hit": _cache_hit}
    return w_fn, info


def _episodes(diet):
    # Thin projection over the shared splitter (single ordering everywhere;
    # s0_pool sampling is order-sensitive, so two splitters was a real
    # divergence risk, not just duplication).
    from .estimators import episodes
    return [{"obs": ep["obs"]} for ep in episodes(diet)]


def mis_diagnostics(diet, w_fn, gamma):
    """Estimate + receipt in one pass over the ratios. Receipt = effective
    fraction of diet rows actually carrying the estimate (1.0 = flat,
    ~0 = degenerate emphasis). Reported per candidate/diet; gate wiring
    follows once its range is observed."""
    w = np.asarray(w_fn(diet["obs"], diet["act"]), dtype=np.float64)
    r = np.asarray(diet["rew"], dtype=np.float64)
    t = np.asarray(diet["t"], dtype=np.float64)
    n_ep = len(np.unique(diet["episode"]))
    w = w / max(w.mean(), 1e-12)  # scale-free: uniform weights recover
    gw = (gamma ** t) * w         # behavior exactly
    gw_sum = max(gw.sum(), 1e-12)
    ess_frac = float((gw_sum ** 2) / max((gw ** 2).sum(), 1e-12) / len(gw))
    return {"mis": float(np.sum(gw * r) / n_ep),
            "mis_ess_frac": ess_frac,
            "mean_w": round(float(w.mean()), 3),
            "p99_w": round(float(np.quantile(w, 0.99)), 2)}


def mis_estimate(diet, w_fn, gamma):
    return mis_diagnostics(diet, w_fn, gamma)["mis"]


def support_stats(diet, w_fn):
    """Where does the candidate live relative to the data? The MIS ratios
    w(s,a) ARE a support meter: high w = candidate-heavy, data-thin regions.
    Returns quantiles, top-decile share of total weight, and the standardized
    obs-center of the top decile (domain-free "where": per-dim sigma offsets
    vs the diet mean). Cheap: one w_fn forward over logged rows."""
    import numpy as _np
    w = _np.asarray(w_fn(diet["obs"], diet["act"]), dtype=np.float64)
    w = w / max(w.mean(), 1e-12)
    obs = _np.asarray(diet["obs"], dtype=np.float64)
    mu, sd = obs.mean(0), obs.std(0) + 1e-12
    cut = float(np.quantile(w, 0.90))
    top = w >= cut
    top_share = float(w[top].sum() / max(w.sum(), 1e-12))
    center = ((obs[top].mean(0) - mu) / sd).tolist() if top.sum() else [0.0] * obs.shape[1]
    return {"w_p50": round(float(np.median(w)), 3),
            "w_p90": round(float(np.quantile(w, 0.90)), 3),
            "w_p99": round(float(np.quantile(w, 0.99)), 3),
            "top_decile_share": round(top_share, 3),
            "top_decile_obs_center": [round(float(x), 2) for x in center],
            "n_top": int(top.sum())}
