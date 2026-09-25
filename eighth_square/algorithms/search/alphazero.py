import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..tabular.common import episode_stats
from .mcts import EnvModel


class AZNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.p_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.p_head(h), self.v_head(h)


class AlphaZero:
    """AlphaZero search: PUCT selection with per-node centered Q, leaf
    evaluation by the value net blended with a real model rollout, visit-count
    backup. Operates on an EnvModel (real environment)."""

    def __init__(
        self,
        model,
        net,
        device,
        cpuct=2.0,
        iterations=100,
        max_depth=60,
        tau_horizon=5,
        rollout_depth=15,
        rng=None,
    ):
        self.model = model
        self.net = net
        self.device = device
        self.cpuct = cpuct
        self.iterations = iterations
        self.max_depth = max_depth
        self.tau_horizon = tau_horizon
        self.rollout_depth = rollout_depth
        self.rng = rng or np.random.default_rng(0)
        self.root = {}

    def _node(self, s):
        if s not in self.root:
            with torch.no_grad():
                logits, v = self.net(
                    torch.as_tensor(np.asarray(s, dtype=np.float32), device=self.device).unsqueeze(
                        0
                    )
                )
            node = {
                "N": np.zeros(self.model.nA),
                "W": np.zeros(self.model.nA),
                "P": torch.softmax(logits, dim=1)[0].cpu().numpy(),
                "v": float(v[0, 0]),
                "children": [None] * self.model.nA,
            }
            self.root[s] = node
        return self.root[s]

    def search(self, s):
        self.model.save()
        s = tuple(np.asarray(s, dtype=np.float32))
        for _ in range(self.iterations):
            self._simulate(s, 0)
        self.model.restore()
        node = self._node(s)
        N = node["N"]
        tau = 1.0 if N.sum() <= self.tau_horizon else 0.0
        if tau > 0:
            pi = N ** (1.0 / tau)
            pi = pi / pi.sum()
            a = int(self.rng.choice(self.model.nA, p=pi))
        else:
            a = int(np.argmax(N))
            pi = np.zeros(self.model.nA)
            pi[a] = 1.0
        return a, pi

    def _leaf_value(self, s, depth):
        node = self._node(s)
        if self.rollout_depth <= 0:
            return node["v"]
        val = 0.0
        g = 1.0
        d = depth
        while d < self.max_depth:
            a = int(self.rng.integers(self.model.nA))
            s, r, done = self.model.step(a)
            val += g * r
            if done:
                break
            g *= self.gamma
            d += 1
            if d - depth >= self.rollout_depth:
                break
        return 0.5 * node["v"] + 0.5 * val

    def _simulate(self, s, depth):
        node = self._node(s)
        N, W = node["N"], node["W"]
        total = N.sum()
        if total == 0:
            a = int(np.argmax(node["P"]))
            self.model.save()
            s2, r, done = self.model.step(a)
            s2 = tuple(s2)
            if done or depth + 1 >= self.max_depth:
                val = r
            else:
                child = self._node(s2)
                node["children"][a] = child
                val = r + self.gamma * self._leaf_value(s2, depth + 1)
            self.model.restore()
            node["N"][a] += 1
            node["W"][a] += val
            return val
        q = np.where(N > 0, W / N, 0.0)
        q = q - q.mean()
        u = self.cpuct * node["P"] * np.sqrt(total + 1.0) / (1.0 + N)
        a = int(np.argmax(q + u))
        self.model.save()
        s2, r, done = self.model.step(a)
        s2 = tuple(s2)
        if done or depth + 1 >= self.max_depth:
            val = r
        else:
            child = node["children"][a]
            if child is None:
                child = self._node(s2)
                node["children"][a] = child
            val = self._simulate(s2, depth + 1)
        self.model.restore()
        val = r + self.gamma * val
        node["N"][a] += 1
        node["W"][a] += val
        return val


class AlphaZeroAgent(BaseAgent):
    """AlphaZero: MCTS guided by a policy/value network, trained on
    self-play data where the environment is the 'opponent'. Leaf values blend
    the network prediction with real model rollouts (needed on dense-reward
    tasks, where a naive value-only backup offers no gradient)."""

    family = "model-based"
    policy = "on-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = torch.device(config.get("device", "cpu"))
        obs_dim = env.observation_space.shape[0]
        self.net = AZNet(obs_dim, int(env.action_space.n)).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.get("lr", 1e-3))
        self.gamma = config.get("gamma", 0.99)
        self.iterations = config.get("az_iterations", 150)
        self.cpuct = config.get("az_cpuct", 2.0)
        self.max_depth = config.get("az_max_depth", 60)
        self.tau_horizon = config.get("az_tau_horizon", 5)
        self.rollout_depth = config.get("az_rollout_depth", 15)
        self.buffer = []

    def act(self, state, eval=True):
        az = AlphaZero(
            EnvModel(self.env),
            self.net,
            self.device,
            self.cpuct,
            self.iterations,
            self.max_depth,
            self.tau_horizon,
            self.rollout_depth,
            self.rng,
        )
        az.gamma = self.gamma
        a, _ = az.search(state)
        return a

    def train(self, env, config, tracker):
        model = EnvModel(env)
        steps = config.get("steps", 20_000)
        batch_size = config.get("batch_size", 256)
        ep = 0
        env_steps = 0
        state, _ = env.reset()
        traj = []
        ret = 0.0
        t = 0
        done = False
        while env_steps < steps:
            if done:
                z = 0.0
                for s, pi, r in reversed(traj):
                    z = r + self.gamma * z
                    self.buffer.append((s, pi, z))
                buf_size = config.get("buffer_size", 50_000)
                if len(self.buffer) > buf_size:
                    self.buffer = self.buffer[-buf_size:]
                if len(self.buffer) >= batch_size:
                    for _ in range(min(8, len(traj) // 16 + 1)):
                        self._learn(config, batch_size)
                ep += 1
                episode_stats(tracker, t, ep, ret)
                state, _ = env.reset()
                traj = []
                ret = 0.0
                t = 0
                done = False
            az = AlphaZero(
                model,
                self.net,
                self.device,
                self.cpuct,
                self.iterations,
                self.max_depth,
                self.tau_horizon,
                self.rollout_depth,
                self.rng,
            )
            az.gamma = self.gamma
            a, pi = az.search(state)
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            traj.append((np.asarray(state, dtype=np.float32), pi, float(r)))
            state = ns
            ret += r
            t += 1
            env_steps += 1
            if t >= config.get("max_episode_steps", 10_000):
                done = True
        self.episodes = ep

    def _learn(self, config, batch_size):
        self.net.train()
        idx = self.rng.integers(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        s = torch.as_tensor(
            np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device
        )
        pi = torch.as_tensor(
            np.stack([b[1] for b in batch]), dtype=torch.float32, device=self.device
        )
        z = torch.as_tensor(
            np.array([b[2] for b in batch]), dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        logits, v = self.net(s)
        self.optimizer.zero_grad()
        loss = -torch.sum(pi * F.log_softmax(logits, dim=1), dim=1).mean() + F.mse_loss(v, z)
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "opt": self.optimizer.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["opt"])
