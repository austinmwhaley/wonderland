import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..tabular.common import episode_stats


class IQNet(nn.Module):
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


class IQLAgent(BaseAgent):
    """Independent Q-Learning (IQL) baseline: each agent runs its own DQN on the
    shared team reward with NO centralized mixing. The comparison against QMIX
    isolates the value of QMIX's state-dependent mixing network (CTDE)."""

    family = "multi-agent"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = torch.device(config.get("device", "cpu"))
        obs_dim = int(getattr(env, "obs_dim", 4))
        self.n_agents = getattr(env, "n_agents", 2)
        self.n_actions = getattr(env, "n_actions", 2)
        hidden = config.get("hidden", 128)
        self.qnets = [
            IQNet(obs_dim, self.n_actions, hidden).to(self.device) for _ in range(self.n_agents)
        ]
        self.qnets_t = [
            IQNet(obs_dim, self.n_actions, hidden).to(self.device) for _ in range(self.n_agents)
        ]
        for q, qt in zip(self.qnets, self.qnets_t):
            qt.load_state_dict(q.state_dict())
        self.opts = [
            torch.optim.Adam(q.parameters(), lr=config.get("lr", 1e-3)) for q in self.qnets
        ]
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.buffer = []
        self.eps_start = config.get("eps_start", 1.0)
        self.eps_end = config.get("eps_end", 0.01)

    def act(self, state, eval=True):
        eps = self.eps_end if eval else self.eps_start
        actions = []
        with torch.no_grad():
            for i, q in enumerate(self.qnets):
                oi = np.asarray(state[i], dtype=np.float32)
                qv = q(torch.as_tensor(oi, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
                if self.rng.random() > eps:
                    actions.append(int(torch.argmax(qv).item()))
                else:
                    actions.append(int(self.rng.integers(self.n_actions)))
        return tuple(actions)

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
        (s1, s2), _ = env.reset()
        while env_steps < steps:
            if done:
                ep += 1
                episode_stats(tracker, t, ep, ret)
                (s1, s2), _ = env.reset()
                ret = 0.0
                t = 0
                done = False
            with torch.no_grad():
                qv1 = self.qnets[0](
                    torch.as_tensor(s1, dtype=torch.float32, device=self.device).unsqueeze(0)
                )[0]
                qv2 = self.qnets[1](
                    torch.as_tensor(s2, dtype=torch.float32, device=self.device).unsqueeze(0)
                )[0]
            a1 = (
                int(torch.argmax(qv1).item())
                if self.rng.random() > eps
                else int(self.rng.integers(self.n_actions))
            )
            a2 = (
                int(torch.argmax(qv2).item())
                if self.rng.random() > eps
                else int(self.rng.integers(self.n_actions))
            )
            (ns1, ns2), r, term, trunc, _ = env.step((a1, a2))
            done = bool(term or trunc)
            self.buffer.append(
                (s1.copy(), s2.copy(), a1, a2, float(r), done, ns1.copy(), ns2.copy())
            )
            if len(self.buffer) > buf_size:
                self.buffer = self.buffer[-buf_size:]
            if len(self.buffer) >= batch_size:
                self._learn(config, batch_size)
            s1, s2 = ns1, ns2
            ret += r
            t += 1
            env_steps += 1
            eps = max(self.eps_end, eps - eps_decay)
        self.episodes = ep

    def _learn(self, config, batch_size):
        idx = self.rng.integers(0, len(self.buffer), size=batch_size)
        b = [self.buffer[i] for i in idx]
        s1 = torch.as_tensor(np.stack([x[0] for x in b]), dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.stack([x[1] for x in b]), dtype=torch.float32, device=self.device)
        a1 = torch.as_tensor(np.array([x[2] for x in b]), dtype=torch.long, device=self.device)
        a2 = torch.as_tensor(np.array([x[3] for x in b]), dtype=torch.long, device=self.device)
        r = torch.as_tensor(
            np.array([x[4] for x in b]), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        done = torch.as_tensor(
            np.array([x[5] for x in b]), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        ns1 = torch.as_tensor(np.stack([x[6] for x in b]), dtype=torch.float32, device=self.device)
        ns2 = torch.as_tensor(np.stack([x[7] for x in b]), dtype=torch.float32, device=self.device)
        # independent TD per agent (no centralized mixing)
        for s, a, ns, q, qt, opt, si in (
            (s1, a1, ns1, self.qnets[0], self.qnets_t[0], self.opts[0], 0),
            (s2, a2, ns2, self.qnets[1], self.qnets_t[1], self.opts[1], 1),
        ):
            qv = q(s).gather(1, a.unsqueeze(1))
            with torch.no_grad():
                q_next = qt(ns).max(1, keepdim=True).values
                target = r + self.gamma * (1 - done) * q_next
            opt.zero_grad()
            F.smooth_l1_loss(qv, target).backward()
            opt.step()
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def save(self, path):
        torch.save({"qnets": [q.state_dict() for q in self.qnets]}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for q, sd in zip(self.qnets, ckpt["qnets"]):
            q.load_state_dict(sd)
