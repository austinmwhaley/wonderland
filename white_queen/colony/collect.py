"""Behavior Colony collector. Trains each roster policy (scaled budgets), rolls
out episodes under a colony-level epsilon mixture with EXACT analytic behavior
probs, and writes everything to SQLite. Reuses lab agents untouched through
the BaseAgent duck API (act/train/save/load); collection epsilon lives here,
not in the agents, so mu is exact for every family uniformly."""

import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environments.registry import make_env  # noqa: E402
from white_queen import db  # noqa: E402
from white_queen.colony.roster import build_roster  # noqa: E402


class _NullTracker:
    def log(self, **kw):
        pass

    def save_config(self, *a, **k):
        pass

    def close(self):
        pass


class _RandomAgent:
    family, algo = "random", "random"

    def __init__(self, env, config):
        self.env = env
        self.nA = int(env.action_space.n)
        self.rng = np.random.default_rng(config.get("seed", 0))

    def act(self, state, eval=False):
        return int(self.rng.integers(self.nA))

    def train(self, env, config, tracker):
        pass

    def save(self, path):
        open(path, "w").write("random")

    def load(self, path):
        pass


def _make_agent(row, env, seed, steps):
    if row["module"] is None:
        return _RandomAgent(env, {"seed": seed})
    mod = importlib.import_module(row["module"])
    cls = getattr(mod, row["class"])
    cfg = {"seed": seed, "steps": steps, "device": "cpu", "buffer_size": max(steps * 2, 10_000)}
    cfg.update(row["extra"])
    return cls(env, cfg)


def _rollout(agent, env, rng, nA, n_episodes, eps):
    """Yields per-episode dicts. Greedy act + colony eps mixture; exact probs."""
    out = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        o, a, r, d, g, pr = [], [], [], [], [], []
        done = False
        while not done:
            greedy = (
                int(agent.act(state, eval=True))
                if not isinstance(agent, _RandomAgent)
                else int(rng.integers(nA))
            )
            if isinstance(agent, _RandomAgent) or rng.random() < eps:
                act = int(rng.integers(nA))
            else:
                act = greedy
            p_take = ((1.0 - eps) + eps / nA) if act == greedy else (eps / nA)
            if isinstance(agent, _RandomAgent):
                p_take = 1.0 / nA
            ns, rew, term, trunc, _ = env.step(act)
            done = bool(term or trunc)
            o.append(np.asarray(state, dtype=np.float32))
            a.append(act)
            r.append(float(rew))
            d.append(float(done))
            g.append(greedy)
            pr.append(float(p_take))
            state = ns
        out.append(
            {
                "obs": np.asarray(o),
                "act": np.asarray(a),
                "rew": np.asarray(r),
                "done": np.asarray(d),
                "greedy": np.asarray(g),
                "prob": np.asarray(pr),
            }
        )
    return out


def run(cfg, db_path, ckpt_dir="white_queen_checkpoints"):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    probe = make_env(cfg["env"], seed=0)
    obs_dim = int(np.prod(probe.observation_space.shape))
    nA = int(probe.action_space.n)
    con = db.init_db(db_path, obs_dim, nA, cfg["gamma"])
    for row in build_roster(cfg):
        for seed in cfg["seeds"]:
            frac = row["frac"]
            steps = 0 if frac is None else max(500, int(cfg[row["budget"]] * frac))
            env = make_env(cfg["env"], seed=seed)
            agent = _make_agent(row, env, seed, steps)
            if steps:
                agent.train(
                    env,
                    {
                        "seed": seed,
                        "steps": steps,
                        "device": "cpu",
                        "buffer_size": max(steps * 2, 10_000),
                        **row["extra"],
                    },
                    _NullTracker(),
                )
                agent.save(os.path.join(ckpt_dir, f"{row['policy_id']}_s{seed}.pt"))
            pid = f"{row['policy_id']}_s{seed}"
            db.register_policy(
                con, pid, row["family"], row["key"], frac, seed, steps, cfg["collect_eps"]
            )
            rng = np.random.default_rng(1000 + seed)
            rets = []
            for ep in _rollout(agent, env, rng, nA, cfg["collect_episodes"], cfg["collect_eps"]):
                db.insert_episode(
                    con,
                    pid,
                    seed,
                    ep["obs"],
                    ep["act"],
                    ep["rew"],
                    ep["done"],
                    ep["greedy"],
                    np.full_like(ep["prob"], cfg["collect_eps"]),
                    ep["prob"],
                )
                rets.append(float(np.sum(ep["rew"])))
            print(f"colony: {pid} steps={steps} mean_ret={np.mean(rets):.1f}", flush=True)
    con.close()
    return db_path
