import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent, evaluate
from ..approx.networks import Critic, DiscretePolicy, MLP
from ..deep.networks import QNetwork
from .common import collect_transitions, train_behavior


class IQL(BaseAgent):
    """Implicit Q-Learning (Kostrikov et al. 2022, NeurIPS): offline Q-learning
    whose value function is fit with an expectile regression instead of the
    max operator. Because the expectile ignores actions worse than the data
    policy, OOD actions never inflate the target; the policy is extracted by
    advantage-weighted behavioral cloning of the dataset actions."""

    family = "value-based"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 1e-3)
        self.hidden = config.get("hidden", 128)
        self.batch_size = config.get("batch_size", 64)
        self.tau = config.get("expectile", 0.7)
        self.beta = config.get("beta", 3.0)
        self.eval_freq = config.get("eval_freq", 5_000)
        self.eval_episodes = config.get("eval_episodes", 5)
        in_dim = int(np.prod(env.observation_space.shape))
        self.nA = int(env.action_space.n)
        self.Q = QNetwork(in_dim, self.hidden, self.nA).to(self.device)
        self.V = MLP(in_dim, self.hidden, 1).to(self.device)
        self.V_target = MLP(in_dim, self.hidden, 1).to(self.device)
        self.V_target.load_state_dict(self.V.state_dict())
        self.policy = DiscretePolicy(in_dim, self.hidden, self.nA).to(self.device)
        self.q_opt = torch.optim.Adam(list(self.Q.parameters()) + list(self.V.parameters()), lr=self.lr)
        self.p_opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.q_opt_clip = config.get("q_opt_clip", 5.0)
        self.v_clip = config.get("v_clip", 1_000.0)
        self.v_tau = config.get("v_tau", 0.005)

    def act(self, state, eval=True):
        with torch.no_grad():
            logits = self.policy(torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0))
            return int(logits.argmax().item())

    def _update(self):
        obs, act, rew, obs2, done = self.buffer.sample(self.batch_size)
        q = self.Q(obs).gather(1, act.unsqueeze(1))
        v = self.V(obs)
        with torch.no_grad():
            target = rew + self.gamma * (1 - done) * self.V_target(obs2).clamp(-self.v_clip, self.v_clip)
        diff = q.detach() - v
        w = torch.where(diff > 0, torch.full_like(diff, self.tau),
                        torch.full_like(diff, 1.0 - self.tau))
        l_q = F.smooth_l1_loss(q, target)
        l_v = (w * diff ** 2).mean()
        self.q_opt.zero_grad()
        (l_q + l_v).backward()
        torch.nn.utils.clip_grad_norm_(list(self.Q.parameters()) + list(self.V.parameters()), self.q_opt_clip)
        self.q_opt.step()
        for ps, pt in zip(self.V.parameters(), self.V_target.parameters()):
            pt.data.mul_(1 - self.v_tau).add_(ps.data, alpha=self.v_tau)
        with torch.no_grad():
            adv = q.detach() - self.V(obs).detach()
            scale = adv.abs().mean().clamp(min=1e-6)
            w_p = torch.exp(self.beta * (adv / scale).clamp(-1.0, 1.0))
            w_p = w_p / w_p.mean().clamp(min=1e-8)
        logp = self.policy(obs).clamp(min=1e-8).log().gather(1, act.unsqueeze(1))
        l_p = -(w_p * logp).mean()
        self.p_opt.zero_grad()
        l_p.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.p_opt.step()
        return float(l_q.item())

    def train(self, env, config, tracker):
        behavior = train_behavior(env, config, self.rng)
        self.buffer = collect_transitions(env, behavior, config.get("dataset_size", 40_000),
                                          config.get("collect_eps", 0.1), self.rng, self.device)
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
        torch.save({"Q": self.Q.state_dict(), "V": self.V.state_dict(),
                    "V_target": self.V_target.state_dict(),
                    "policy": self.policy.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.Q.load_state_dict(data["Q"])
        self.V.load_state_dict(data["V"])
        self.V_target.load_state_dict(data["V_target"])
        self.policy.load_state_dict(data["policy"])