"""Ratio-learner variants: GradientDICE beside the DualDICE-style default.
Same interface (learn_ratio -> w_fn), same episodic time-weighted aggregation
in marginalized.mis_diagnostics — ratio learners are interchangeable parts.
GradientDICE (Zhang et al. 2020): single-timescale saddle objective with an
f-divergence-flavored regularizer; in practice: different stability profile,
same contract. Earns standing the same way: diagnostics + truth proximity."""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)


def learn_ratio_gd(
    diet,
    cand,
    gamma,
    temperature=1.0,
    hidden=None,
    steps=None,
    batch=None,
    seed=1,
    lr=None,
    reg=None,
    cfg=None,
    cand_id=None,
    cache_dir=None,
    weights_hash=None,
):
    """GradientDICE-flavored: min_w max_f E[(w - g w' - (1-g)) f] - .5 f^2
    + reg * E[(w-1)^2 /2-ish anchor toward uniform]. The anchor is the
    honesty feature: ratios start at 'trust behavior' and move only on
    gradient evidence. All sizes autotuned from diet (resolve_mis_cfg);
    reg defaults to 1/sqrt(N) (0.01 at N=10k was 0.1 — now scaled)."""
    from algorithms.approx.networks import MLP
    from .autotune import resolve_mis_cfg
    from .protocols import check_candidate

    check_candidate(cand)
    base = dict(cfg) if cfg else {}
    for k, v in (("hidden", hidden), ("batch", batch), ("seed", seed)):
        if v is not None:
            base[k] = v
    if steps is not None:
        base["steps_max"] = steps
    if lr is not None:
        base["lr"] = lr
        base["w_lr"] = lr
    ac = resolve_mis_cfg(diet, base)
    hidden, batch, seed = ac["hidden"], ac["batch"], ac["seed"]
    steps, lr = ac["steps_max"], ac["lr"]
    if reg is None:
        import numpy as _np

        reg = 1.0 / _np.sqrt(max(len(diet["obs"]), 100))
    from .training import seed_all as _seed_all

    _seed_all(seed)
    rng = np.random.default_rng(seed)
    mu, sd = diet["obs"].mean(0), diet["obs"].std(0) + 1e-6
    nA = diet["nA"]

    def Z(x):
        return (np.asarray(x, dtype=np.float32) - mu) / sd

    from .autotune import resolve_device

    dev = resolve_device(ac.get("device"))

    def onehot(a):
        oh = np.zeros((len(np.atleast_1d(a)), nA), dtype=np.float32)
        oh[np.arange(len(oh)), np.atleast_1d(a)] = 1.0
        return oh

    def feat(o, a):
        return torch.as_tensor(np.concatenate([Z(o), onehot(a)], 1), device=dev)

    w_net = MLP(Z(diet["obs"][:1]).shape[1] + nA, hidden, 1).to(dev)
    f_net = MLP(Z(diet["obs"][:1]).shape[1] + nA, hidden, 1).to(dev)
    opt_w = torch.optim.Adam(w_net.parameters(), lr=lr)
    opt_f = torch.optim.Adam(f_net.parameters(), lr=lr)
    N = len(diet["obs"])
    s0_pool = np.stack([diet["obs"][i] for i in np.unique(diet["episode"], return_index=True)[1]])
    from .protocols import sample_actions, safe_probs
    from .training import govern

    _P2full = safe_probs(cand.action_probs(np.asarray(diet["obs2"]), temperature=temperature))
    _P0pool = safe_probs(cand.action_probs(s0_pool, temperature=temperature))
    _ckey = None
    if cand_id is not None and cache_dir:
        from .cache import diet_hash, make_key, load as _cload

        _ckey = make_key(
            "gdice", diet_hash(diet), cand_id, ac, temperature, weights_hash or "noweights"
        )
        _hit = _cload(cache_dir, _ckey, map_location=dev)
    else:
        _hit = None
    _cache_hit = _hit is not None
    if _cache_hit:
        w_net.load_state_dict(_hit["w"])
        f_net.load_state_dict(_hit["f"])

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
                w2 = F.softplus(w_net(feat(s2, a2))).squeeze(1) * (
                    1 - torch.as_tensor(d, device=dev)
                )
            f = f_net(feat(s, a)).squeeze(1)
            f0 = f_net(feat(s0_pool[j], a0)).squeeze(1)
            obj_f = (
                ((w.detach() - gamma * w2) * f).mean()
                - (1 - gamma) * f0.mean()
                - 0.5 * (f**2).mean()
            )
            opt_f.zero_grad()
            (-obj_f).backward()
            opt_f.step()
            w = F.softplus(w_net(feat(s, a))).squeeze(1)
            with torch.no_grad():
                w2 = F.softplus(w_net(feat(s2, a2))).squeeze(1) * (
                    1 - torch.as_tensor(d, device=dev)
                )
                f_d = f_net(feat(s, a)).squeeze(1)
            obj_w = ((w - gamma * w2) * f_d).mean() + reg * ((w - 1.0) ** 2).mean()
            opt_w.zero_grad()
            obj_w.backward()
            opt_w.step()
            last = float(obj_w.item())
        return last

    if _cache_hit:
        _gov = {"steps": 0, "stopped": "cache-hit"}
    else:
        _gov = govern(
            {"w": w_net, "f": f_net},
            [opt_w, opt_f],
            _step,
            None,
            {"steps_max": steps, "eval_every": max(steps // 20, 100), "lr": lr, "patience": 0},
        )
        if _ckey is not None:
            from .cache import save as _csave

            _csave(cache_dir, _ckey, {"w": w_net.state_dict(), "f": f_net.state_dict()})

    def w_fn(obs, act):
        with torch.no_grad():
            return F.softplus(w_net(feat(obs, act))).squeeze(1).cpu().numpy()

    with torch.no_grad():
        wall = F.softplus(w_net(feat(diet["obs"], diet["act"]))).squeeze(1).cpu().numpy()
    info = {
        "mean_w": round(float(wall.mean()), 3),
        "p99_w": round(float(np.quantile(wall, 0.99)), 2),
        "variant": "gradient",
        "cache_hit": _cache_hit,
    }
    return w_fn, info
