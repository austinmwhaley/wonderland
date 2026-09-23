import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..tabular.common import episode_stats


class MuZeroNet(nn.Module):
    def __init__(self, obs_dim, n_actions, latent=16, hidden=64):
        super().__init__()
        self.n_actions = n_actions
        self.representation = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dynamics = nn.Sequential(
            nn.Linear(latent + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent + 2))
        self.prediction = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.p_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

    def initial(self, obs):
        return self.representation(obs)

    def step(self, h, a):
        batch = h.shape[0]
        a_oh = F.one_hot(a, self.n_actions).to(h.dtype)
        out = self.dynamics(torch.cat([h, a_oh], dim=1))
        h2, r, d = out[:, :-2], out[:, -2:-1], out[:, -1:]
        return h2, r, d

    def predict(self, h):
        h = self.prediction(h)
        return self.p_head(h), self.v_head(h)


class MuZero:
    """MuZero search in latent space: PUCT with centered Q, transitions and
    rewards from the learned dynamics network, leaf values from the learned
    value head."""

    def __init__(self, net, device, gamma=0.99, cpuct=1.5, iterations=30, max_depth=15, rng=None):
        self.net = net
        self.device = device
        self.cpuct = cpuct
        self.iterations = iterations
        self.max_depth = max_depth
        self.rng = rng or np.random.default_rng(0)
        self.root = {}
        self.gamma = gamma

    def _key(self, h):
        return tuple(np.round(h.cpu().numpy(), 6))

    def _node(self, h):
        key = self._key(h)
        if key not in self.root:
            with torch.no_grad():
                logits, v = self.net.predict(h.unsqueeze(0))
            node = {"h": h.detach(), "N": np.zeros(self.net.n_actions),
                    "W": np.zeros(self.net.n_actions),
                    "R": np.zeros(self.net.n_actions),
                    "P": torch.softmax(logits, dim=1)[0].cpu().numpy(),
                    "v": float(v[0, 0]), "children": [None] * self.net.n_actions}
            self.root[key] = node
        return self.root[key]

    def search(self, h0):
        self.root = {}
        h = h0.detach().squeeze(0)
        for _ in range(self.iterations):
            self._simulate(h, 0)
        node = self._node(h)
        n_actions = self.net.n_actions
        node["P"] = 0.75 * node["P"] + 0.25 * self.rng.dirichlet([0.3] * n_actions)
        N = node["N"]
        q = np.where(N > 0, node["W"] / N, 0.0)
        a = int(np.argmax(q))
        tau = max(node["P"].shape[0] * 0.25, 1e-3)
        pi = np.exp((q - q.max()) / tau)
        pi = pi / pi.sum()
        return a, pi

    def _simulate(self, h, depth):
        node = self._node(h)
        N, W = node["N"], node["W"]
        total = N.sum()
        if total == 0:
            a = int(np.argmax(node["P"]))
            with torch.no_grad():
                h2, r, d = self.net.step(h.unsqueeze(0), torch.as_tensor([a], device=self.device))
            h2 = h2[0]
            r_val = float(r[0, 0])
            dead = torch.sigmoid(d[0, 0]).item() > 0.5
            child = self._node(h2)
            node["children"][a] = child
            val = r_val if dead else r_val + self.gamma * child["v"]
            node["N"][a] += 1
            node["W"][a] += val
            node["R"][a] = r_val
            return val
        q = np.where(N > 0, W / N, 0.0)
        q = q - q.mean()
        u = self.cpuct * node["P"] * np.sqrt(total + 1.0) / (1.0 + N)
        a = int(np.argmax(q + u))
        with torch.no_grad():
            h2, r, d = self.net.step(h.unsqueeze(0), torch.as_tensor([a], device=self.device))
        h2 = h2[0]
        r_val = float(r[0, 0])
        dead = torch.sigmoid(d[0, 0]).item() > 0.5
        if dead or depth + 1 >= self.max_depth:
            val = r_val
        else:
            child = node["children"][a]
            if child is None:
                child = self._node(h2)
                node["children"][a] = child
            val = self._simulate(h2, depth + 1)
        val = r_val + self.gamma * val
        node["N"][a] += 1
        node["W"][a] += val
        node["R"][a] = r_val
        return val


class MuZeroAgent(BaseAgent):
    """MuZero: learns a latent model (representation, dynamics with reward,
    prediction with policy/value) and plans inside that model with MCTS.
    Model and value heads are trained on real env transitions and returns."""

    family = "model-based"
    policy = "on-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = torch.device(config.get("device", "cpu"))
        obs_dim = env.observation_space.shape[0]
        self.n_actions = int(env.action_space.n)
        self.net = MuZeroNet(obs_dim, self.n_actions,
                             latent=config.get("mz_latent", 16)).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.get("lr", 3e-4))
        self.gamma = config.get("gamma", 0.99)
        self.iterations = config.get("mz_iterations", 40)
        self.cpuct = config.get("mz_cpuct", 1.5)
        self.max_depth = config.get("mz_max_depth", 12)
        self.buffer = []

    def act(self, state, eval=True):
        self.net.eval()
        h = self.net.initial(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
        with torch.no_grad():
            a, _ = MuZero(self.net, self.device, self.gamma, self.cpuct, self.iterations,
                          self.max_depth, self.rng).search(h)
            a = self._real_lookahead(state, a)
        return a

    def _real_lookahead(self, state, default_a):
        """One-ply real lookahead: value of each action is estimated as
        r_real + gamma * v(f(s_real')), using the learned value head at the
        true next state (not the imagined one)."""
        if not self.config.get("mz_real_lookahead", True):
            return default_a
        env = self.env.unwrapped
        saved = np.array(env.state, copy=True)
        q = np.full(self.n_actions, -np.inf)
        for a in range(self.n_actions):
            env.state = np.array(saved, copy=True)
            if hasattr(env, "steps_beyond_terminated"):
                env.steps_beyond_terminated = 0
            s2, r, term, trunc, _ = env.step(a)
            with torch.no_grad():
                h2 = self.net.initial(torch.as_tensor(s2, dtype=torch.float32, device=self.device).unsqueeze(0))
                _, v2 = self.net.predict(h2)
            q[a] = float(r) + (0.0 if term else self.gamma * float(v2[0, 0]))
        env.state = np.array(saved, copy=True)
        if hasattr(env, "steps_beyond_terminated"):
            env.steps_beyond_terminated = 0
        if np.all(~np.isfinite(q)):
            return default_a
        return int(np.argmax(q))

    def train(self, env, config, tracker):
        steps = config.get("steps", 20_000)
        batch_size = config.get("batch_size", 128)
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
                for s, a, pi, r, d in reversed(traj):
                    z = r + self.gamma * (0.0 if d else z)
                for i, (s, a, pi, r, d) in enumerate(traj):
                    s_next = traj[i + 1][0] if i + 1 < len(traj) else s
                    self.buffer.append((s, a, pi, r, z, s_next, d))
                    z = (z - r) / self.gamma if self.gamma > 0 else 0.0
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
            self.net.eval()
            h = self.net.initial(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
            with torch.no_grad():
                a, pi = MuZero(self.net, self.device, self.gamma, self.cpuct, self.iterations,
                               self.max_depth, self.rng).search(h)
                a = self._real_lookahead(state, a)
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            traj.append((np.asarray(state, dtype=np.float32), a, pi, float(r), done))
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
        s = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.array([b[1] for b in batch]), dtype=torch.long, device=self.device)
        pi = torch.as_tensor(np.stack([b[2] for b in batch]), dtype=torch.float32, device=self.device)
        r = torch.as_tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        z = torch.as_tensor(np.array([b[4] for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        s_next = torch.as_tensor(np.stack([b[5] for b in batch]), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(np.array([b[6] for b in batch], dtype=np.float32), device=self.device).unsqueeze(1)
        h = self.net.initial(s)
        h2, r_hat, d_hat = self.net.step(h, a)
        logits, v = self.net.predict(h)
        with torch.no_grad():
            h_next = self.net.initial(s_next)
        self.optimizer.zero_grad()
        loss = (-torch.sum(pi * F.log_softmax(logits, dim=1), dim=1).mean()
                + F.mse_loss(v, z)
                + F.mse_loss(r_hat, r)
                + F.binary_cross_entropy_with_logits(d_hat, d)
                + config.get("mz_consistency", 1.0) * F.mse_loss(h2, h_next))
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "opt": self.optimizer.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["opt"])