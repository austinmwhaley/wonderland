"""Direct family extensions: ensemble-FQE, LSTDQ, FVE.

Ensemble-FQE: K independently-seeded FQE nets (K autotuned from n_episodes);
mean = estimate, std across members = disagreement receipt.
LSTDQ: least-squares TD evaluation — closed-form-ish, data-efficient where
gradient FQE is hungry; linear in [obs, one-hot act] with scale-set ridge +
condition-number diagnostics. Ridge = 0.1/sqrt(N) (matches 1e-3 at N=10k).
FVE: fitted VALUE evaluation — V directly instead of Q. Same autotuned
early-stop discipline as FQE; rho clip passed from panel (no 10.0 literal).
"""

import numpy as np

from .autotune import PROB_FLOOR, resolve_fqe_cfg


def _auto_K(n_episodes, K=None):
    if K is not None:
        return int(K)
    import numpy as _np

    return int(_np.clip(max(int(n_episodes) // 50, 2), 2, 5))


def ensemble_fqe(
    diet,
    cand,
    gamma,
    fqe_cfg=None,
    temperature=None,
    K=None,
    cand_id=None,
    cache_dir=None,
    weights_hash=None,
):
    """K seeded FQE fits; mean = estimate, std = disagreement receipt. PLUS
    val-weighted mean: members weight by inverse held-out Bellman error, so
    one collapsed seed (v15 expert: 2.2 among ~20s) can't drag the headline.
    Uniform weights recover the plain mean when vals tie. Receipted."""
    from .estimators import fit_fqe, select_temperature
    from .protocols import check_candidate

    check_candidate(cand)
    if temperature is None:
        temperature = select_temperature(diet, cand, gamma)["temperature"]
    n_ep = len(np.unique(np.asarray(diet["episode"])))
    K = _auto_K(n_ep, K)
    fqe_cfg = dict(fqe_cfg) if fqe_cfg else {}
    means, nets, vals = [], [], []
    base_seed = int(fqe_cfg.get("seed", 0))
    # NOTE: members hold out different splits (seed-strided), so vals compare
    # approximately — fine for downweighting collapsed members (val gaps are
    # orders of magnitude, split noise is percent-level on 200+ holdout rows).
    for k in range(K):
        cfg = dict(fqe_cfg)
        # Deterministic stride (100) separates ensemble streams; not tuning.
        cfg["seed"] = base_seed + 100 * k
        qnet, dm, info = fit_fqe(
            diet,
            cand,
            gamma,
            cfg,
            temperature,
            cand_id=cand_id,
            cache_dir=cache_dir,
            weights_hash=weights_hash,
        )
        means.append(dm)
        nets.append(qnet)
        v = info.get("val_bellman")
        vals.append(float(v) if v is not None and v == v and v > 0 else float("inf"))
    import numpy as _np

    w = 1.0 / _np.asarray(vals)
    w = w / w.sum() if _np.isfinite(w).all() and w.sum() > 0 else None
    vwmean = float(_np.average(means, weights=w)) if w is not None else float(_np.mean(means))
    return {
        "mean": float(np.mean(means)),
        "disagreement": float(np.std(means)),
        "members": [round(m, 1) for m in means],
        "nets": nets,
        "K": K,
        "val_weighted_mean": vwmean,
        "val_weights": [
            round(float(x), 3) for x in (w.tolist() if w is not None else [1.0 / K] * K)
        ],
    }


def lstdq(diet, cand, gamma, temperature=1.0, ridge=None, seed=0):
    """LSTD-Q with linear features phi(s,a)=[s, onehot(a)]. Closed form:
    theta = A^{-1} b. Ridge = 0.1/sqrt(N) scaled to feature-covariance trace
    (1e-3 at N=10k, the old literal, now derived). seed is reproducibility
    only (action sampling for expected next features), not a tuning dial."""
    from .protocols import check_candidate

    check_candidate(cand)
    nA = diet["nA"]
    N = len(diet["obs"])
    if ridge is None:
        import numpy as _np

        ridge = 0.1 / _np.sqrt(max(N, 100))
    oh = np.zeros((N, nA), dtype=np.float32)
    oh[np.arange(N), diet["act"]] = 1.0
    Phi = np.concatenate([diet["obs"].astype(np.float64), oh], 1)
    oh2 = np.zeros((N, nA), dtype=np.float32)
    pi2 = cand.action_probs(diet["obs2"], temperature=temperature)
    from .protocols import sample_actions

    a2 = sample_actions(np.random.default_rng(seed), pi2)
    oh2[np.arange(N), a2] = 1.0
    Phi2 = np.concatenate([diet["obs2"].astype(np.float64), oh2], 1)
    d = 1.0 - diet["done"].astype(np.float64)
    A = Phi.T @ (Phi - gamma * (d[:, None] * Phi2)) / N
    b = Phi.T @ diet["rew"].astype(np.float64) / N
    lam = ridge * np.trace(A) / A.shape[0]
    try:
        theta = np.linalg.solve(A + lam * np.eye(A.shape[0]), b)
        cond = float(np.linalg.cond(A + lam * np.eye(A.shape[0])))
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, b, rcond=None)[0]
        cond = float("inf")
    # V(s0) under pi: need per-episode starts
    from .estimators import episodes

    vals = []
    for ep in episodes(diet):
        pi0 = cand.action_probs(ep["obs"][:1], temperature=temperature)[0]
        np.concatenate([ep["obs"][:1].astype(np.float64), np.eye(nA)[[np.argmax(pi0)]]], 1)
        q_all = np.stack(
            [
                np.concatenate([ep["obs"][:1].astype(np.float64), np.eye(nA)[[a]]], 1) @ theta
                for a in range(nA)
            ]
        )
        vals.append(float((pi0 * q_all.squeeze(1)).sum()))
    return {
        "dm": float(np.mean(vals)),
        "cond": cond,
        "rank_note": "linear-phi; trust scales with cond",
    }


def _mc_returns(diet, gamma):
    """Per-row discounted Monte-Carlo returns (from that step to episode end).
    O(total steps), numpy. Unbiased V targets for validation — unlike backups,
    they never contain the net's own output."""
    import numpy as _np

    ep = _np.asarray(diet["episode"])
    rew = _np.asarray(diet["rew"], dtype=np.float64)
    out = _np.empty(len(rew))
    # Diet rows arrive tid-ordered with contiguous episode runs (db asserts).
    bounds = _np.flatnonzero(_np.diff(ep.astype(_np.int64)) != 0) + 1
    starts = _np.concatenate([[0], bounds])
    ends = _np.concatenate([bounds, [len(rew)]])
    for s, e in zip(starts, ends):
        T = e - s
        disc = gamma ** _np.arange(T)
        out[s:e] = _np.cumsum((rew[s:e] * disc)[::-1])[::-1] / np.maximum(disc, 1e-12)
    return out.astype(np.float32)


def fve(
    diet,
    cand,
    gamma,
    fqe_cfg=None,
    temperature=1.0,
    rho_cap=None,
    cand_id=None,
    cache_dir=None,
    weights_hash=None,
):
    """Fitted value evaluation: V directly (half the outputs of Q). Same
    autotuned early-stop discipline as FQE. rho_cap=None -> derived from the
    diet via the same clip-quantile rule as the main panel (no 10.0 literal).
    cand_id+cache_dir enable the weights cache (exact on hit)."""
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
    from algorithms.approx.networks import MLP
    from .protocols import check_candidate

    check_candidate(cand)
    cfg = resolve_fqe_cfg(diet, fqe_cfg, gamma)
    batch, hidden = cfg["batch"], cfg["hidden"]
    seed = cfg["seed"]
    lr = cfg["lr"]
    holdout = cfg["holdout"]
    N, in_dim = len(diet["obs"]), diet["obs"].shape[1]
    from .training import seed_all as _seed_all

    _seed_all(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    cut = max(N - holdout, 1)
    tr, va = perm[:cut], perm[cut:]
    from .autotune import resolve_device

    device = resolve_device(cfg.get("device"))
    vnet = MLP(in_dim, hidden, 1).to(device)
    opt = torch.optim.Adam(vnet.parameters(), lr=lr)
    obs = torch.as_tensor(diet["obs"]).float().to(device)
    R = torch.as_tensor(diet["rew"]).float().unsqueeze(1).to(device)
    O2 = torch.as_tensor(diet["obs2"]).float().to(device)
    D = torch.as_tensor(diet["done"]).float().unsqueeze(1).to(device)
    # Per-step clipped ratios make TD(V) target the CANDIDATE's value
    # (plain TD on behavior data would converge to the behavior's value —
    # correctly labeled or not at all). Tree-backup(0)-flavored, exact mu.
    # Cap derived from data unless caller pins rho_cap.
    from .estimators import _taken_probs

    rho_cap_source = "caller" if rho_cap is not None else "data-quantile"
    if rho_cap is None:
        from .autotune import resolve_clip_quantile
        from .estimators import _uncapped_rhos

        try:
            rhos = np.concatenate(_uncapped_rhos(diet, cand, temperature))
            rho_cap = float(np.quantile(rhos, resolve_clip_quantile(N)))
            rho_cap = max(rho_cap, 1.0)
        except Exception as e:
            # LOUD fallback (was silent 10.0): ratio estimation itself failed
            # (broken candidate probs). Receipted so it can never pass as data.
            import warnings

            warnings.warn(
                f"fve rho_cap derivation failed ({e!r}); using fallback 10.0",
                RuntimeWarning,
                stacklevel=2,
            )
            rho_cap, rho_cap_source = 10.0, "fallback-loud"
    _mu_take = diet.get("mu_take")
    if _mu_take is None:
        _mu_take = diet["mu"][np.arange(N), diet["act"]]
    rho_all = np.clip(
        _taken_probs(cand, diet["obs"], diet["act"], temperature)
        / np.maximum(_mu_take, PROB_FLOOR),
        0.0,
        rho_cap,
    )
    Rho = torch.as_tensor(rho_all.astype(np.float32)).unsqueeze(1).to(device)

    # Grounded validation (NOT tree-backup error): the backup target contains
    # (1-Rho)*V(obs) — the net's own output — so an untrained near-constant net
    # scores a beautifully low "Bellman error" and best-restore would crown
    # it (v13: FVE -0.0 everywhere). Validate against Monte-Carlo returns
    # instead: unbiased, net-independent, measures what V should actually be.
    MC = torch.as_tensor(_mc_returns(diet, gamma).astype(np.float32)).unsqueeze(1).to(device)

    def v_target(idx):
        with torch.no_grad():
            one_step = R[idx] + gamma * (1 - D[idx]) * vnet(O2[idx])
            return Rho[idx] * one_step + (1 - Rho[idx]) * vnet(obs[idx])

    def val_err():
        with torch.no_grad():
            return float(F.mse_loss(vnet(obs[va]), MC[va]))

    from .training import govern

    def _step(n, lr):
        last = None
        for _ in range(n):
            i = tr[rng.integers(0, cut, batch)]
            loss = F.mse_loss(vnet(obs[i]), v_target(i))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return last

    _ckey, _hit = None, None
    if cand_id is not None and cache_dir:
        from .cache import diet_hash, make_key, load as _cload, save as _csave

        _ckey = make_key(
            "fve", diet_hash(diet), cand_id, cfg, temperature, weights_hash or "noweights"
        )
        _hit = _cload(cache_dir, _ckey, map_location=device)
    if _hit is not None:
        vnet.load_state_dict(_hit["v"])
        _gov = {"steps": 0, "n_evals": 0, "best_val": None, "stopped": "cache-hit", "lr_final": lr}
        _cache_hit = True
    else:
        _gov = govern({"v": vnet}, [opt], _step, val_err, cfg)
        _cache_hit = False
        if _ckey is not None:
            from .cache import save as _csave

            _csave(cache_dir, _ckey, {"v": vnet.state_dict()})
    from .estimators import episodes

    vnet = vnet.cpu()
    vals = []
    for ep in episodes(diet):
        with torch.no_grad():
            vals.append(float(vnet(torch.as_tensor(ep["obs"][:1]).float()).item()))
    return {
        "dm": float(np.mean(vals)),
        "steps": _gov["steps"],
        "stopped": _gov["stopped"],
        "device": device,
        "cache_hit": _cache_hit,
        "rho_cap": round(float(rho_cap), 3),
        "rho_cap_source": rho_cap_source,
    }
