"""Offline candidates on SQLite diets. Reuses lab IQL/CQL untouched via the
buffer-injection seam (fill lab OfflineBuffer from the diet, assign
agent.buffer, drive agent._update()). BC is implemented here on the lab's
DiscretePolicy net — the supervised-template analog: same inputs, same pins,
only the loss differs. Candidate net sizes autotune from the diet; the
CandidateProtocol lives in tribunal.ope.protocols (imported here for compat).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from algorithms.offline.common import OfflineBuffer  # noqa: E402
from algorithms.offline.iql import IQL  # noqa: E402
from algorithms.offline.cql import CQL  # noqa: E402
from algorithms.approx.networks import DiscretePolicy  # noqa: E402
from white_queen.tribunal.ope.protocols import (  # noqa: E402,F401
    CandidateProtocol,
    check_candidate,
)

# Built-in OUTPUT trainers evaluated by the tribunal. `random` is deliberately
# NOT here: a uniform-random policy is an INPUT/logging agent (it generates the
# offline logs), not a candidate we would ever ship. It remains trainable by
# explicit name for use as a behavior/eval reference.
CANDIDATES = ("iql", "bc", "cql")


class _RandomCandidate:
    """Uniform-random policy. The floor baseline: OPE should always HOLD it
    against behavior (no skill to certify), which makes it a live sanity
    check on the whole panel. Needs no training/checkpoint."""

    def __init__(self, nA, seed=0):
        self.nA = int(nA)
        self._rng = np.random.default_rng(seed)

    def act(self, state, eval=True):
        return int(self._rng.integers(self.nA))

    def action_probs(self, obs, temperature=1.0):
        o = np.asarray(obs)
        n = len(o) if o.ndim > 1 else 1
        return np.full((n, self.nA), 1.0 / self.nA, dtype=np.float32)
# Pluggable candidate registry. The library evaluates ANY protocol-compliant
# policy; these trainers are conveniences. register_candidate("mypolicy", fn)
# where fn(name, env, diet, cfg, out_path) -> handle with act/action_probs.
CANDIDATE_TRAINERS = {}


def register_candidate(name, trainer, overwrite=False):
    """Register a custom offline trainer so the pipeline can use it by name."""
    if name in CANDIDATE_TRAINERS and not overwrite:
        raise ValueError(f"candidate {name!r} already registered")
    CANDIDATE_TRAINERS[name] = trainer
    return trainer


def _fill_buffer(diet, device, seed=0):
    n, d = len(diet["obs"]), diet["obs"].shape[1]
    buf = OfflineBuffer(n, d, device, rng=np.random.default_rng(seed))
    for i in range(n):
        buf.push(diet["obs"][i], int(diet["act"][i]), float(diet["rew"][i]),
                 diet["obs2"][i], float(diet["done"][i]))
    return buf


def _drive(agent, steps, batch_note=""):
    agent.t = 0
    for _ in range(steps):
        agent._update()
        agent.t += 1
    return agent


class _BCWrapper:
    """Behavior cloning on the lab DiscretePolicy net. Exposes the candidate
    protocol: act(state, eval) + action_probs(obs_batch). Sizes autotuned
    from diet unless pinned in config."""

    def __init__(self, env, config, diet=None):
        import numpy as _np  # noqa
        from white_queen.tribunal.ope.autotune import resolve_device
        self.device = resolve_device(config.get("device"))
        in_dim = int(_np.prod(env.observation_space.shape))
        self.nA = int(env.action_space.n)
        hidden = config.get("hidden")
        if hidden is None and diet is not None:
            from white_queen.tribunal.ope.autotune import resolve_fqe_cfg
            hidden = resolve_fqe_cfg(diet, config)["hidden"]
        hidden = int(hidden or 128)
        from white_queen.tribunal.ope.training import seed_all as _seed_all
        _seed_all(config.get("seed", 0))
        self.net = DiscretePolicy(in_dim, hidden, self.nA).to(self.device)
        lr = config.get("lr")
        if lr is None and diet is not None:
            from white_queen.tribunal.ope.autotune import resolve_fqe_cfg
            lr = resolve_fqe_cfg(diet, config)["lr"]
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(lr or 1e-3))
        self.batch = config.get("batch_size")
        if self.batch is None and diet is not None:
            from white_queen.tribunal.ope.autotune import resolve_fqe_cfg
            self.batch = resolve_fqe_cfg(diet, config)["batch"]
        self.batch = int(self.batch or 256)

    def fit(self, diet, steps=None):
        """Supervised fit governed by training.govern: holdout NLL early-stop,
        cosine schedule, best-restore. First BC version with validation —
        previously fixed steps on full data (overtrained by construction)."""
        import torch.nn.functional as F
        from white_queen.tribunal.ope.autotune import resolve_fqe_cfg
        from white_queen.tribunal.ope.training import govern
        ac = resolve_fqe_cfg(diet, {})
        if steps is not None:
            ac = dict(ac, steps_max=int(steps))
        N = len(diet["obs"])
        rng = np.random.default_rng(0)  # reproducible shuffling, not tuning.
        perm = rng.permutation(N)
        holdout = min(2000, max(200, N // 10))
        cut = max(N - holdout, 1)
        tr, va = perm[:cut], perm[cut:]
        obs = torch.as_tensor(np.asarray(diet["obs"], dtype=np.float32),
                              device=self.device)
        act = torch.as_tensor(np.asarray(diet["act"], dtype=np.int64),
                              device=self.device)

        def val_err():
            with torch.no_grad():
                logp = self.net(obs[va]).clamp(min=1e-8).log()
                return float(F.nll_loss(logp, act[va]))

        def _step(n, lr):
            last = None
            for _ in range(n):
                idx = tr[rng.integers(0, cut, self.batch)]
                logp = self.net(obs[idx]).clamp(min=1e-8).log()
                loss = F.nll_loss(logp, act[idx])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                last = float(loss.item())
            return last

        _gov = govern({"net": self.net}, [self.opt], _step, val_err, ac)
        self.val_info = {"val_nll": _gov["best_val"], "steps": _gov["steps"],
                         "stopped": _gov["stopped"]}
        return self

    def act(self, state, eval=True):
        import numpy as _np
        with torch.no_grad():
            p = self.net(torch.as_tensor(_np.asarray(state, dtype=_np.float32),
                                         device=self.device).unsqueeze(0))
        return int(p.argmax().item())

    def action_probs(self, obs, temperature=1.0):
        with torch.no_grad():
            out = self.net(torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                           device=self.device)).log().clamp(max=0)
            tempered = (out / max(float(temperature), 1e-3)).softmax(dim=1)
            probs = tempered.cpu().numpy()
        # Sanitize: dead nets emit 0/NaN -> uniform (panel degrades to veto).
        import numpy as _np
        probs = _np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = _np.maximum(probs, 0.0)
        s = probs.sum(1, keepdims=True)
        probs[s.squeeze(1) <= 0] = 1.0 / probs.shape[1]
        return probs / probs.sum(1, keepdims=True)


def train_candidate(name, env, diet, cfg, out_path):
    """Train a candidate by name. Checks the pluggable registry first, then
    falls back to the built-in iql/cql/bc trainers."""
    if name in CANDIDATE_TRAINERS:
        return CANDIDATE_TRAINERS[name](name, env, diet, cfg, out_path)
    from white_queen.tribunal.ope.autotune import resolve_device, resolve_fqe_cfg
    ac = resolve_fqe_cfg(diet, cfg.get("candidate_cfg"))
    device = resolve_device(cfg.get("device", ac.get("device")))
    config = {"seed": int(cfg.get("seed", 0)), "device": device,
              "gamma": cfg["gamma"],
              "hidden": int(cfg.get("hidden", ac["hidden"])),
              "batch_size": int(cfg.get("batch_size", ac["batch"]))}
    steps = cfg.get("offline_steps")
    if steps is None:
        steps = ac["steps_max"]
    if name == "random":
        return _RandomCandidate(int(diet["nA"]), seed=config["seed"])
    if name in ("iql_cont", "bc_cont"):
        from algorithms.offline.continuous_iql import ContinuousIQL
        low = high = None
        try:
            low = np.asarray(env.action_space.low, dtype=float).ravel()
            high = np.asarray(env.action_space.high, dtype=float).ravel()
        except Exception:
            pass
        if low is None:
            a_dim = int(diet["act"].shape[1])
            low, high = -np.ones(a_dim), np.ones(a_dim)
        cfgc = dict(config, action_low=low, action_high=high,
                    expectile=cfg.get("expectile", 0.7),
                    beta=cfg.get("beta", 3.0))
        cand = ContinuousIQL(diet, cfgc).fit(int(steps))
        cand.save(out_path)
        return cand
    if name == "bc":
        cand = _BCWrapper(env, config, diet).fit(diet, steps)
        torch.save(cand.net.state_dict(), out_path)
        return cand
    cls = {"iql": IQL, "cql": CQL}[name]
    agent = cls(env, config)
    agent.buffer = _fill_buffer(diet, config["device"])
    _drive(agent, int(steps))
    agent.save(out_path)
    return _handle(agent, name)


def _handle(agent, name):
    class Handle:
        def __init__(self, agent, name):
            self._a, self._n = agent, name

        def act(self, state, eval=True):
            return self._a.act(state, eval=eval)

        def action_probs(self, obs, temperature=1.0):
            import torch.nn.functional as F
            net = self._a.policy if self._n == "iql" else self._a.online
            with torch.no_grad():
                x = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                    device=self._a.device)
                logits = net(x)
                probs = F.softmax(logits / max(float(temperature), 1e-3),
                                  dim=1).cpu().numpy()
            import numpy as _np
            probs = _np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = _np.maximum(probs, 0.0)
            s = probs.sum(1, keepdims=True)
            probs[s.squeeze(1) <= 0] = 1.0 / probs.shape[1]
            return probs / probs.sum(1, keepdims=True)

    return Handle(agent, name)


def load_candidate(name, env, diet, cfg, ckpt_path):
    """Reload saved weights WITHOUT retraining (pure-OPE reruns).

    Rebuilds the same architecture train_candidate used (autotuned hidden)
    and loads weights with map_location. Raises FileNotFoundError if missing.
    """
    import os
    if name == "random":
        # Floor baseline: no training, no checkpoint to look for.
        return _RandomCandidate(int(diet["nA"]), seed=int(cfg.get("seed", 0)))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"candidate checkpoint missing: {ckpt_path}")
    from white_queen.tribunal.ope.autotune import resolve_device, resolve_fqe_cfg
    ac = resolve_fqe_cfg(diet, cfg.get("candidate_cfg"))
    device = resolve_device(cfg.get("device", ac.get("device")))
    config = {"seed": int(cfg.get("seed", 0)), "device": device,
              "gamma": cfg["gamma"],
              "hidden": int(cfg.get("hidden", ac["hidden"])),
              "batch_size": int(cfg.get("batch_size", ac["batch"]))}
    if name == "bc":
        cand = _BCWrapper(env, config, diet)
        cand.net.load_state_dict(torch.load(ckpt_path, map_location=device,
                                            weights_only=True))
        cand.net.to(device).eval()
        return cand
    cls = {"iql": IQL, "cql": CQL}[name]
    agent = cls(env, config)
    agent.load(ckpt_path)
    try:
        agent.policy.to(device).eval()
    except Exception:
        pass
    try:
        agent.online.to(device).eval()
    except Exception:
        pass
    return _handle(agent, name)
