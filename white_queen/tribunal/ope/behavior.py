"""Industry track: behavior propensities you don't get for free.

Academic track (v11 and earlier): diet["mu"] holds EXACT behavior probs —
the lab logs its own epsilon mixture analytically. Real company logs don't
have that column: you know what action was taken, not with what probability.
Every IS/DR/WDR/MIS ratio in the panel divides by mu, so unknown propensities
must be ESTIMATED from (state -> action) pairs before any judging happens.

This module is the entire difference between the tracks:
- estimate_behavior(diet, cfg): supervised fit of mu-hat(s) on logged
  (obs, act), governed by training.govern (train/val NLL early-stop, cosine,
  best-restore) — same discipline as every other model, no new loop.
- with_estimated_propensities(diet, cfg): diet COPY with mu := mu-hat
  (floored, receipted). The panel consumes it unchanged — it never knew
  where mu came from, which is exactly the modularity story.

Floor principle: exact mu uses PROB_FLOOR=1e-8 (precision guard). Estimated
mu gets a bigger floor because estimation error near 0 explodes ratios:
floor = 0.05 * uniform = 0.05/nA (never trust an estimate below 5% of
chance), clipped fraction reported. Separate constants, separate reasons.

No new data, no agent retraining: mu-hat trains on the same logged rows.
Per-diet estimation (your logs are what they are — a novice-only slice gets
a novice-only behavior model, mirroring industry reality).
"""

import numpy as np


def estimate_behavior(diet, cfg=None, seed=0):
    """Fit mu-hat(s) on diet (obs -> act). Returns (probs, info) with
    probs (N, nA) row-stochastic on diet obs, info holding val_nll, accuracy,
    steps, stopped, device. cfg: hidden/batch/lr-device autotuned from diet
    (resolve_fqe_cfg reuse: capacity follows input size, not a literal)."""
    import os
    import sys
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from algorithms.approx.networks import DiscretePolicy
    from .autotune import resolve_device, resolve_fqe_cfg
    from .protocols import validate_diet
    from .training import govern
    validate_diet(diet)
    base = dict(cfg) if cfg else {}
    base.setdefault("seed", int(seed))
    ac = resolve_fqe_cfg(diet, base)
    device = resolve_device(ac.get("device"))
    hidden, batch, lr = ac["hidden"], ac["batch"], ac["lr"]
    holdout, seed = ac["holdout"], int(ac["seed"])
    N, in_dim = len(diet["obs"]), diet["obs"].shape[1]
    nA = diet["nA"]
    from .training import seed_all as _seed_all
    _seed_all(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    cut = max(N - holdout, 1)
    tr, va = perm[:cut], perm[cut:]
    net = DiscretePolicy(in_dim, hidden, nA).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    O = torch.as_tensor(np.asarray(diet["obs"], dtype=np.float32)).to(device)
    A = torch.as_tensor(np.asarray(diet["act"], dtype=np.int64)).to(device)

    def val_err():
        with torch.no_grad():
            return float(F.nll_loss(net(O[va]).clamp(min=1e-8).log(), A[va]))

    def _step(n, lr_):
        last = None
        for _ in range(n):
            idx = tr[rng.integers(0, cut, batch)]
            loss = F.nll_loss(net(O[idx]).clamp(min=1e-8).log(), A[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return last

    gov = govern({"net": net}, [opt], _step, val_err, ac)
    net = net.cpu().eval()
    # Temperature calibration (Guo et al. 2017): single scalar T fit on HOLDOUT
    # only (no new data), minimizing NLL. Fixes BC overconfidence — the v12
    # ESS collapse came from sharp-but-wrong μ-hat making extreme ratios.
    with torch.no_grad():
        # Raw pre-softmax scores (DiscretePolicy.forward already softmaxes).
        _all_logits = net.net(torch.as_tensor(
            np.asarray(diet["obs"], dtype=np.float32))).numpy().astype(np.float64)
        _va_logits = _all_logits[va]
        _tgt = np.asarray(diet["act"])[va]
    _best_T, _best_nll = 1.0, float("inf")
    for _T in np.linspace(0.2, 5.0, 49):
        _l = _va_logits / _T
        _l = _l - _l.max(1, keepdims=True)
        _p = np.exp(_l)
        _p = _p / _p.sum(1, keepdims=True)
        _nll = float(-np.log(np.maximum(_p[np.arange(len(_tgt)), _tgt], 1e-12)).mean())
        if _nll < _best_nll:
            _best_nll, _best_T = _nll, float(_T)
    _l = _all_logits / _best_T
    _l = _l - _l.max(1, keepdims=True)
    _e = np.exp(_l)
    probs = (_e / _e.sum(1, keepdims=True)).astype(np.float32)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0)
    s = probs.sum(1, keepdims=True)
    probs[s.squeeze(1) <= 0] = 1.0 / nA
    probs = probs / probs.sum(1, keepdims=True)
    with torch.no_grad():
        pred = probs.argmax(1)
    acc = float((pred == np.asarray(diet["act"])).mean())
    return probs.astype(np.float32), {
        "val_nll": gov["best_val"], "accuracy": round(acc, 4),
        "steps": gov["steps"], "stopped": gov["stopped"],
        "device": device, "hidden": hidden, "track": "industry",
        "cal_temperature": round(_best_T, 3), "cal_holdout_nll": round(_best_nll, 4)}


def with_estimated_propensities(diet, cfg=None, seed=0, floor_frac=0.05):
    """Diet copy with mu := floored mu-hat. floor = floor_frac * uniform
    (default 5% of chance: nA=2 -> 0.025). Returns (new_diet, info) with
    clipped_frac + behavior accuracy receipts. Original diet untouched."""
    from .protocols import validate_diet
    validate_diet(diet)
    probs, binfo = estimate_behavior(diet, cfg, seed)
    nA = int(diet["nA"])
    floor = float(floor_frac / nA)
    clipped = probs < floor
    mu = np.maximum(probs, floor).astype(np.float32)
    mu = mu / mu.sum(1, keepdims=True).astype(np.float32)
    new = dict(diet)
    new["mu"] = mu
    new["_mu_source"] = "estimated"
    new["_mu_info"] = dict(binfo, floor=round(floor, 5),
                           clipped_frac=round(float(clipped.mean()), 4))
    return new, {"mu_source": "estimated", "behavior": binfo,
                 "floor": round(floor, 5),
                 "clipped_frac": round(float(clipped.mean()), 4)}


def estimate_behavior_continuous(obs, act, cfg=None, seed=0):
    """Diagonal-Gaussian behavior density for continuous actions: log p(a|s).

    Returns (logp (N,), info). Real continuous logs usually don't record the
    logging density, so we fit it (mean and log-std heads on the lab MLP) and
    evaluate it at the taken actions. Governed with early stop; calibration
    temp omitted (density, not classification).
    """
    import os
    import sys
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from algorithms.approx.networks import MLP
    from .autotune import resolve_device, resolve_fqe_cfg
    obs = np.asarray(obs, dtype=np.float32)
    act = np.asarray(act, dtype=np.float32)
    N, d = obs.shape
    a_dim = act.shape[1] if act.ndim > 1 else 1
    act = act.reshape(N, a_dim)
    stub = {"obs": obs, "act": act[:, 0].astype(np.int64),
            "rew": np.zeros(N, dtype=np.float32),
            "done": np.zeros(N, dtype=np.float32),
            "mu": np.full((N, 2), 0.5, dtype=np.float32),
            "episode": np.arange(N), "t": np.zeros(N), "nA": 2, "N": N}
    base = dict(cfg) if cfg else {}
    base.setdefault("seed", int(seed))
    ac = resolve_fqe_cfg(stub, base)
    dev = resolve_device(ac.get("device"))
    hidden, batch, lr = ac["hidden"], ac["batch"], ac["lr"]
    holdout = min(max(N // 10, 50), 2000)
    rng = np.random.default_rng(ac["seed"])
    perm = rng.permutation(N)
    cut = max(N - holdout, 1)
    tr, va = perm[:cut], perm[cut:]
    net = MLP(d, hidden, 2 * a_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    O = torch.as_tensor(obs).to(dev)
    A = torch.as_tensor(act).to(dev)

    def nll(idx):
        out = net(O[idx])
        mean, logstd = out[:, :a_dim], out[:, a_dim:].clamp(-5, 2)
        z = (A[idx] - mean) / logstd.exp()
        return (0.5 * z ** 2 + logstd - 0.5 * np.log(2 * np.pi)).sum(1).mean()

    best, bad, steps, eval_every, patience = None, 0, 0, max(50, cut // 20), 5
    with torch.no_grad():
        best = float(nll(va))
    while steps < ac["steps_max"] and bad < patience:
        for _ in range(eval_every):
            i = torch.as_tensor(tr[rng.integers(0, cut, batch)]).to(dev)
            loss = nll(i)
            opt.zero_grad(); loss.backward(); opt.step(); steps += 1
        with torch.no_grad():
            ve = float(nll(va))
        bad = bad + 1 if ve >= best - 1e-6 else 0
        best = min(best, ve)
    net = net.cpu().eval()
    with torch.no_grad():
        out = net(torch.as_tensor(obs))
        mean, logstd = out[:, :a_dim], out[:, a_dim:].clamp(-5, 2)
        z = (torch.as_tensor(act) - mean) / logstd.exp()
        logp = (-0.5 * z ** 2 - logstd - 0.5 * float(np.log(2 * np.pi))).sum(1)
    return logp.numpy(), {"val_nll": round(float(best), 4), "steps": steps,
                          "a_dim": a_dim, "device": dev, "track": "continuous"}
