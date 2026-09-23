import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent, evaluate
from ..deep.dqn import DQN
from ..deep.networks import QNetwork, polyak_copy
from .common import collect_transitions, train_behavior


class CQL(DQN):
    """Conservative Q-Learning (Kumar et al. 2020, NeurIPS): offline Q-learning
    that penalizes out-of-distribution actions by pushing down Q-values of
    actions not present in the dataset. The penalty alpha*(log-sum-exp Q - Q_data)
    makes the learned Q lower-bound the true return of the behavior policy."""

    family = "value-based"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        config["prioritized"] = False
        super().__init__(env, config)
        self.cql_alpha = config.get("cql_alpha", 1.0)
        self.behavior_steps = config.get("behavior_steps", 20_000)
        self.dataset_size = config.get("dataset_size", 40_000)
        self.collect_eps = config.get("collect_eps", 0.1)

    def _loss(self, obs, act, rew, obs2, done, weights):
        q_all = self.online(obs)
        q = q_all.gather(1, act.unsqueeze(1))
        target = rew + self.gamma_n * (1 - done) * self._target_values(obs2, done)
        td = F.smooth_l1_loss(q, target, reduction="none")
        cql = torch.logsumexp(q_all, dim=1, keepdim=True) - q
        loss = ((td + self.cql_alpha * cql) * weights).mean()
        return loss, td.detach().abs().squeeze(1)

    def _update(self):
        obs, act, rew, obs2, done = self.buffer.sample(self.batch_size)
        weights = torch.ones(self.batch_size, 1, device=self.device)
        loss, td = self._loss(obs, act, rew, obs2, done, weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.tau > 0:
            polyak_copy(self.online, self.target, self.tau)
        elif self.t % self.target_freq == 0 and self.t > 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def train(self, env, config, tracker):
        behavior = train_behavior(env, config, self.rng)
        self.buffer = collect_transitions(env, behavior, self.dataset_size,
                                          self.collect_eps, self.rng, self.device)
        self.t = 0
        losses = []
        while self.t < config["steps"]:
            losses.append(self._update())
            self.t += 1
            if self.t % config.get("log_freq", 500) == 0:
                tracker.log(timestep=self.t, loss=float(np.mean(losses)))
                losses = []
            if self.t % self.eval_freq == 0:
                tracker.log(timestep=self.t, eval_return=float(evaluate(self, env, self.eval_episodes)))
        self.episodes = self.t

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])