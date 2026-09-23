import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent, EpsilonScheduler
from .networks import MLP


class SemiGradientSarsa(BaseAgent):
    family = "value-based"
    policy = "on-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        in_dim = int(np.prod(env.observation_space.shape))
        self.nA = int(env.action_space.n)
        self.Q = MLP(in_dim, config.get("hidden", 128), self.nA).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=config.get("lr", 1e-3))
        self.sched = EpsilonScheduler(
            config.get("eps_start", 0.3),
            config.get("eps_end", 0.01),
            config.get("eps_decay_steps", config.get("steps", 50_000)),
            self.rng,
        )
        self.t = 0

    def _t(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def act(self, state, eval=False):
        with torch.no_grad():
            q = self.Q(self._t(state))
            if eval or self.rng.random() >= self.sched.epsilon(self.t):
                return int(q.argmax().item())
            return int(self.rng.integers(self.nA))

    def _target(self, q_next, done, a_next):
        return torch.as_tensor(
            self._r + (0.0 if done else self.gamma * q_next[a_next]),
            dtype=self._q_dtype, device=self.device,
        )

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        losses = []
        while self.t < config["steps"]:
            state, _ = env.reset()
            done = False
            ret = 0.0
            a = self.act(state)
            while not done and self.t < config["steps"]:
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                a_next = self.act(ns)
                q = self.Q(self._t(state)).squeeze(0)
                with torch.no_grad():
                    q_next = self.Q(self._t(ns)).squeeze(0)
                    self._r, self._q_dtype = r, q.dtype
                    target = self._target(q_next, done, a_next)
                self.optimizer.zero_grad()
                loss = F.mse_loss(q[a], target)
                loss.backward()
                self.optimizer.step()
                losses.append(float(target.item() - q[a].item()))
                state, a = ns, a_next
                ret += r
                self.t += 1
            ep += 1
            tracker.log(timestep=self.t, episode=ep, ret=ret,
                        loss=float(np.mean(losses)) if losses else None)
            losses = []
        self.episodes = ep

    def save(self, path):
        torch.save(self.Q.state_dict(), path)

    def load(self, path):
        self.Q.load_state_dict(torch.load(path, map_location=self.device))


class NaiveQLearning(SemiGradientSarsa):
    family = "value-based"
    policy = "off-policy"

    def _target(self, q_next, done, a_next):
        return torch.as_tensor(
            self._r + (0.0 if done else self.gamma * q_next.max()),
            dtype=self._q_dtype, device=self.device,
        )