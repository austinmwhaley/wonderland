"""Model-based rollout: the world-model miniature. Learns CartPole dynamics
(s_next delta + reward) from the diet with validation early-stop, then
simulates the candidate proxy from real diet starts. Adaptive sim count:
episodes grow until the standard error clears a fraction of the estimate.
Same role the beside-B world model will play in retail: one more panelist,
strongest on sequential claims, never the decider alone."""
import numpy as np


_DYN_CACHE = {}


def learn_dynamics(diet, hidden=None, batch=None, seed=0, patience=None,
                   eval_every=None, steps_max=None, cfg=None, cache_dir=None):
    """Dynamics MLP autotuned from diet (same shared net rule as FQE).
    Pass cfg dict or explicit kwargs to override per-key.

    Cached per diet identity (in-memory) AND on disk (cache_dir): dynamics
    don't depend on the candidate, and the panel calls this once per candidate
    on the same diet — same math, train once (saves 2/3 of dynamics cost on
    a 3-candidate grid; disk cache saves it across runs too)."""
    import os
    import sys
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from algorithms.approx.networks import MLP
    from .autotune import resolve_fqe_cfg
    base = dict(cfg) if cfg else {}
    for k, v in (("hidden", hidden), ("batch", batch), ("seed", seed),
                 ("patience", patience), ("eval_every", eval_every),
                 ("steps_max", steps_max)):
        if v is not None:
            base[k] = v
    ac = resolve_fqe_cfg(diet, base)
    # Cache lookup AFTER resolving (key includes resolved budget so explicit
    # overrides still retrain).
    cache_key = (id(diet), len(diet["obs"]), ac["hidden"], ac["batch"],
                 ac["steps_max"], ac["seed"])
    hit = _DYN_CACHE.get(cache_key)
    if hit is not None and hit[0] is diet:
        return hit[1], hit[2]
    _dkey, _dh = None, None
    if cache_dir:
        from .cache import diet_hash, make_key, load as _cload
        _dkey = make_key("dyn", diet_hash(diet), "nodiet-cand", ac, None, "")
        _dh = _cload(cache_dir, _dkey, map_location="cpu")
    hidden, batch, seed = ac["hidden"], ac["batch"], ac["seed"]
    patience, eval_every, steps_max = ac["patience"], ac["eval_every"], ac["steps_max"]
    lr, holdout = ac["lr"], ac["holdout"]
    N, in_dim = len(diet["obs"]), diet["obs"].shape[1]
    nA = diet["nA"]
    from .training import seed_all as _seed_all
    _seed_all(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    cut = max(N - holdout, 1)
    tr, va = perm[:cut], perm[cut:]

    def onehot(a):
        oh = np.zeros((len(np.atleast_1d(a)), nA), dtype=np.float32)
        oh[np.arange(len(oh)), np.atleast_1d(a)] = 1.0
        return oh

    X = np.concatenate([diet["obs"].astype(np.float64), onehot(diet["act"])], 1)
    # Termination head: [delta_obs, reward, done]. Without done the world model
    # rolls every policy to max_len and accrues ~500 reward on CartPole — so
    # MB read ~99 for a random policy (truth ~9) and could not distinguish
    # good from bad. done is what makes survival, hence return, observable.
    Y = np.concatenate([(diet["obs2"] - diet["obs"]).astype(np.float64),
                        diet["rew"][:, None].astype(np.float64),
                        diet["done"][:, None].astype(np.float64)], 1)
    mu, sd = X.mean(0), X.std(0) + 1e-6  # z-score guard, not tuning.
    y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-6  # normalize targets: reward scale varies by diet.
    from .autotune import resolve_device
    dev = resolve_device(ac.get("device"))
    Xn = torch.as_tensor(((X - mu) / sd).astype(np.float32)).to(dev)
    Yn = torch.as_tensor(((Y - y_mu) / y_sd).astype(np.float32)).to(dev)
    net = MLP(Xn.shape[1], hidden, in_dim + 2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # Observed ranges for output clamping (prevents 1e31 compounding).
    rew_min, rew_max = float(diet["rew"].min()), float(diet["rew"].max())
    delta_max = float(np.abs(Y[:, :in_dim]).max()) * 3.0 + 1e-6

    def val_err():
        with torch.no_grad():
            return float(F.mse_loss(net(Xn[va]), Yn[va]))

    from .training import govern

    def _step(n, lr):
        last = None
        for _ in range(n):
            i = torch.as_tensor(tr[rng.integers(0, cut, batch)]).to(dev)
            loss = F.mse_loss(net(Xn[i]), Yn[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return last

    # Try cache first (shape-validated), else train. Loading before deciding
    # to train avoids the old ordering bug where a shape-mismatched artifact
    # skipped training and then failed to load.
    _cache_hit = False
    if _dh is not None:
        try:
            net.load_state_dict(_dh["net"])
            net.to(dev)
            best = _dh["info"]["val_mse"]
            _cache_hit = True
        except Exception:
            _dh, _cache_hit = None, False
    if _cache_hit:
        _gov = {"steps": 0, "n_evals": 0, "stopped": "cache-hit",
                "lr_final": ac["lr"]}
    else:
        _gov = govern({"net": net}, [opt], _step, val_err, ac)
        best = _gov["best_val"]

    def step_fn(obs_batch, act_batch):
        with torch.no_grad():
            o = np.asarray(obs_batch, dtype=np.float64)
            x = torch.as_tensor(np.concatenate([o, onehot(act_batch)], 1).astype(np.float32)).to(dev)
            mu_t = torch.as_tensor(mu.astype(np.float32)).to(dev)
            sd_t = torch.as_tensor(sd.astype(np.float32)).to(dev)
            x = (x - mu_t) / sd_t
            d_norm = net(x).cpu().numpy()
        d = d_norm * y_sd + y_mu  # denormalize
        # Clamp outputs to observed ranges (prevents compounding blowups).
        d[:, :in_dim] = np.clip(d[:, :in_dim], -delta_max, delta_max)
        d[:, in_dim] = np.clip(d[:, in_dim], rew_min, rew_max)
        # done head is MSE-regressed on {0,1}; use the value directly (clip to
        # [0,1]) as P(done). NOT sigmoid — that would fire at 0.5 for a
        # near-zero denormalized done (bias ~1/n_steps).
        done_prob = np.clip(d[:, in_dim + 1], 0.0, 1.0)
        return o + d[:, :in_dim], d[:, in_dim], done_prob
    info = {"val_mse": best, "steps": _gov.get("steps", 0),
            "n_evals": _gov.get("n_evals", 0),
            "stopped": _gov.get("stopped", "?"),
            "lr_final": _gov.get("lr_final", ac["lr"]),
            "cache_hit": _cache_hit, "termination_head": True,
            "y_sd": [round(float(v), 5) for v in y_sd],
            "rew_range": [rew_min, rew_max]}
    if _dkey is not None and not _cache_hit:
        from .cache import save as _csave
        _csave(cache_dir, _dkey, {"net": net.state_dict(), "info": info})
    _DYN_CACHE[cache_key] = (diet, step_fn, info)
    return step_fn, info


def rollout_estimate(diet, cand, gamma, step_fn, temperature=1.0, seed=0,
                     se_frac=None, sim_min=None, sim_max=None, cfg=None):
    """Simulate the proxy from diet starts; grow sims until SE/estimate is
    small (adaptive compute, capped). sim counts and se_frac autotuned from
    n_episodes unless pinned. Termination: learned done head (survival is the
    score on CartPole), plus out-of-bounds and max-length guards."""
    import numpy as _np
    from .autotune import diet_fingerprint, resolve_rollout_cfg
    from .protocols import check_candidate
    check_candidate(cand)
    fp = diet_fingerprint(diet, gamma)
    base = dict(cfg) if cfg else {}
    for k, v in (("se_frac", se_frac), ("sim_min", sim_min),
                 ("sim_max", sim_max), ("seed", seed)):
        if v is not None:
            base[k] = v
    rc = resolve_rollout_cfg(fp["n_episodes"], fp["max_len"], base)
    seed, se_frac = rc["seed"], rc["se_frac"]
    sim_min, sim_max = rc["sim_min"], rc["sim_max"]
    rng = _np.random.default_rng(seed)
    from .estimators import episodes
    from .protocols import sample_actions
    starts = _np.stack([ep["obs"][0] for ep in episodes(diet)])
    max_len = rc["max_len"]
    bounds = np.stack([diet["obs"].min(0), diet["obs"].max(0)])
    rets = []
    n = 0
    nan_rows = 0
    done_hits = 0
    # Warmup floor: at least 10 steps past the shortest episode before the
    # out-of-bounds guard can fire (scales with horizon, not a literal 10
    # for all envs: 5% of max_len, min 5).
    warmup = max(5, int(0.05 * max_len))
    while n < sim_min or (n < sim_max and _se(rets) > se_frac * abs(_np.mean(rets) or 1)):
        i = rng.integers(0, len(starts), min(rc["batch"], len(starts)))
        s = starts[i].astype(float)
        disc, g = 1.0, np.zeros(len(i))
        alive = np.ones(len(i), dtype=bool)
        for t in range(max_len):
            p = _np.asarray(cand.action_probs(s.astype(np.float32), temperature=temperature), dtype=float)
            if not _np.isfinite(p).all() or (p < 0).any():
                nan_rows += 1
            a = sample_actions(rng, p)
            s, r, dp = step_fn(s, a)
            g += disc * r * alive
            disc *= gamma
            # Learned termination: the episode ends where the model says the
            # environment ends (the thing that made MB blind before).
            ended = dp > 0.5
            done_hits += int((ended & alive).sum())
            alive = alive & (~ended)
            if not alive.any():
                break
            if ((s < bounds[0]) | (s > bounds[1])).any(axis=1).all() and t > warmup:
                break
        rets.extend(g.tolist())
        n += len(i)
    out = {"mb": float(_np.mean(rets)), "se": round(_se(rets), 2), "sims": n,
           "done_hits": int(done_hits),
           "cfg": {"se_frac": se_frac, "sim_min": sim_min, "sim_max": sim_max}}
    if nan_rows:
        out["nan_prob_rows"] = int(nan_rows)  # receipt: how often sanitize fired
    return out


def _se(x):
    import numpy as _np
    x = _np.asarray(x, float)
    return float(x.std() / max(len(x) ** 0.5, 1)) if len(x) > 1 else float("inf")


def learn_dynamics_ensemble(diet, K=3, cfg=None, cache_dir=None):
    """Ensemble of dynamics models: train K nets with different seeds and
    average their predicted next-state/reward/done. A single learned model is
    easily exploited by open-loop rollouts (it over-predicts the value of
    actions it never saw), which is why MB disagreed with FQE; averaging K
    independently-seeded models cancels much of that model-exploitation bias.
    Returns (step_fn, info); members are individually disk-cached.
    """
    import numpy as _np
    K = int(K or 1)
    if K <= 1:
        return learn_dynamics(diet, cfg=cfg, cache_dir=cache_dir)
    base = dict(cfg) if cfg else {}
    fns, infos = [], []
    for k in range(K):
        c = dict(base)
        c["seed"] = int(base.get("seed", 0)) + k
        fn, info = learn_dynamics(diet, cfg=c, cache_dir=cache_dir)
        fns.append(fn)
        infos.append(info)

    def step_fn(obs_batch, act_batch):
        outs = [f(obs_batch, act_batch) for f in fns]
        ns = _np.mean([o[0] for o in outs], 0)
        r = _np.mean([o[1] for o in outs], 0)
        dp = _np.mean([o[2] for o in outs], 0)
        # model disagreement (std of done and reward) is a free model-uncertainty
        # signal; report it so the certificate can widen when the model is unsure.
        dis = float(_np.mean(_np.std([o[2] for o in outs], 0))) if K > 1 else 0.0
        return ns, r, dp, dis

    # Normalize the 3-tuple return to the 4-tuple above (rollout_estimate reads
    # the first three; disagreement is carried separately).
    def compat(obs_batch, act_batch):
        ns, r, dp, _ = step_fn(obs_batch, act_batch)
        return ns, r, dp
    compat._disagreement = True
    info = {"ensemble_K": K,
            "member_val_mse": [i.get("val_mse") for i in infos],
            "termination_head": True}
    return compat, info
