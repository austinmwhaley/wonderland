import numpy as np
import torch


class NullTracker:
    def log(self, **kw):
        pass

    def save_config(self, *a, **k):
        pass

    def close(self):
        pass


class OfflineBuffer:
    def __init__(self, capacity, in_dim, device, rng=None):
        self.cap = capacity
        self.device = device
        self.rng = rng or np.random.default_rng(0)
        self.obs = np.zeros((capacity, in_dim), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.obs2 = np.zeros((capacity, in_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.n = 0

    def push(self, s, a, r, s2, done):
        self.obs[self.pos] = s
        self.act[self.pos] = a
        self.rew[self.pos] = r
        self.obs2[self.pos] = s2
        self.done[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, size=batch)

        def to(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        return (
            to(self.obs[idx]),
            torch.as_tensor(self.act[idx], dtype=torch.long, device=self.device),
            to(self.rew[idx]).unsqueeze(1),
            to(self.obs2[idx]),
            to(self.done[idx]).unsqueeze(1),
        )


def train_behavior(env, config, rng):
    from ..deep.dqn import DQN

    cfg = dict(config)
    cfg["steps"] = config.get("behavior_steps", 20_000)
    cfg["buffer_size"] = max(config.get("behavior_steps", 20_000) * 2, 10_000)
    cfg["eps_end"] = config.get("behavior_eps_end", 0.05)
    agent = DQN(env, cfg)
    agent.train(env, cfg, NullTracker())
    return agent


def collect_transitions(env, agent, n, eps=0.1, rng=None, device="cpu"):
    rng = rng or np.random.default_rng(0)
    in_dim = int(np.prod(env.observation_space.shape))
    buf = OfflineBuffer(n, in_dim, device, rng=rng)
    nA = int(env.action_space.n)
    state, _ = env.reset()
    while buf.n < n:
        if rng.random() < eps:
            a = int(rng.integers(nA))
        else:
            a = agent.act(state, eval=True)
        ns, r, term, trunc, _ = env.step(a)
        done = bool(term or trunc)
        buf.push(state, a, r, ns, done)
        state = ns if not done else env.reset()[0]
    return buf


def collect_episodes(env, agent, n_steps, eps=0.1, rng=None):
    rng = rng or np.random.default_rng(0)
    nA = int(env.action_space.n)
    episodes = []
    state, _ = env.reset()
    step = 0
    while step < n_steps:
        ep_obs, ep_act, ep_rew = [], [], []
        done = False
        while not done and step < n_steps:
            if rng.random() < eps:
                a = int(rng.integers(nA))
            else:
                a = agent.act(state, eval=True)
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            ep_obs.append(state)
            ep_act.append(a)
            ep_rew.append(r)
            state = ns if not done else env.reset()[0]
            step += 1
        if ep_obs:
            episodes.append(
                (
                    np.asarray(ep_obs, dtype=np.float32),
                    np.asarray(ep_act, dtype=np.int64),
                    np.asarray(ep_rew, dtype=np.float32),
                )
            )
    return episodes
