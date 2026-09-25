import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..tabular.common import episode_stats


class HERNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class HERAgent(BaseAgent):
    """Hindsight Experience Replay (Andrychowicz et al. 2017): off-policy
    value learning on goal-conditioned transitions, where failed episodes are
    relabeled with goals the agent actually reached so sparse rewards become
    dense. The goal is the second component of the observation."""

    family = "value-based"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = torch.device(config.get("device", "cpu"))
        obs_dim = getattr(env.observation_space, "shape", [2])[0]
        self.n_actions = getattr(env, "n_actions", getattr(env.action_space, "n", 3))
        hidden = config.get("hidden", 128)
        self.online = HERNet(obs_dim, self.n_actions, hidden).to(self.device)
        self.target = HERNet(obs_dim, self.n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=config.get("lr", 3e-4))
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.eps_start = config.get("eps_start", 1.0)
        self.eps_end = config.get("eps_end", 0.05)
        self.k_relabel = config.get("her_k", 4)
        self.buffer = []

    def act(self, state, eval=True, eps=None):
        eps = self.eps_end if eval else (eps if eps is not None else self.eps_start)
        with torch.no_grad():
            q = self.online(
                torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            )[0]
        if self.rng.random() > eps:
            return int(torch.argmax(q).item())
        return int(self.rng.integers(self.n_actions))

    def train(self, env, config, tracker):
        steps = config.get("steps", 50_000)
        batch_size = config.get("batch_size", 64)
        buf_size = config.get("buffer_size", 100_000)
        eps = self.eps_start
        eps_decay = (self.eps_start - self.eps_end) / max(steps, 1)
        ep = 0
        env_steps = 0
        ret = 0.0
        t = 0
        done = False
        s, _ = env.reset()
        ep_transitions = []
        while env_steps < steps:
            if done:
                ep += 1
                episode_stats(tracker, t, ep, ret)
                self._store_episode(ep_transitions)
                ep_transitions = []
                s, _ = env.reset()
                ret = 0.0
                t = 0
                done = False
            a = self.act(s, eval=False, eps=eps)
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            ep_transitions.append((s.copy(), a, float(r), done, ns.copy()))
            self.buffer.append((s.copy(), a, float(r), done, ns.copy()))
            if len(self.buffer) > buf_size:
                self.buffer = self.buffer[-buf_size:]
            if len(self.buffer) >= batch_size:
                self._learn(config, batch_size)
            s = ns
            ret += r
            t += 1
            env_steps += 1
            eps = max(self.eps_end, eps - eps_decay)
        self.episodes = ep

    def _store_episode(self, ep_transitions):
        T = len(ep_transitions)
        if T < 2:
            return
        for j, (s, a, r, done, ns) in enumerate(ep_transitions):
            if j >= T - 1:
                continue
            for _ in range(self.k_relabel):
                t_f = int(self.rng.integers(j + 1, T))
                g = ep_transitions[t_f][4][0]
                tol = 0.4
                rr = float(abs(ns[0] - g) < tol)
                nd = bool(rr)
                self.buffer.append(
                    (
                        np.array([s[0], g], dtype=np.float32),
                        a,
                        rr,
                        nd,
                        np.array([ns[0], g], dtype=np.float32),
                    )
                )

    def _learn(self, config, batch_size):
        idx = self.rng.integers(0, len(self.buffer), size=batch_size)
        b = [self.buffer[i] for i in idx]
        s = torch.as_tensor(np.stack([x[0] for x in b]), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.array([x[1] for x in b]), dtype=torch.long, device=self.device)
        r = torch.as_tensor(
            np.array([x[2] for x in b]), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        done = torch.as_tensor(
            np.array([x[3] for x in b]), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        ns = torch.as_tensor(np.stack([x[4] for x in b]), dtype=torch.float32, device=self.device)
        q = self.online(s).gather(1, a.unsqueeze(1))
        with torch.no_grad():
            q_next = self.target(ns).max(dim=1, keepdim=True).values
            target = r + self.gamma * (1 - done) * q_next
        self.opt.zero_grad()
        F.smooth_l1_loss(q, target).backward()
        self.opt.step()
        for p, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def save(self, path):
        torch.save({"online": self.online.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
