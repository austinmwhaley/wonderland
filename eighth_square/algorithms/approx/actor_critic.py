import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent
from .networks import Critic, DiscretePolicy, GaussianPolicy


class ActorCritic(BaseAgent):
    family = "actor-critic"
    policy = "on-policy"
    action_space = "both"
    state_space = "both"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.discrete = hasattr(env.action_space, "n")
        in_dim = int(np.prod(env.observation_space.shape))
        hidden = config.get("hidden", 128)
        if self.discrete:
            self.policy = DiscretePolicy(in_dim, hidden, int(env.action_space.n)).to(self.device)
        else:
            self.policy = GaussianPolicy(in_dim, hidden, int(env.action_space.shape[0])).to(self.device)
        self.critic = Critic(in_dim, hidden).to(self.device)
        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 1e-3))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 1e-3))
        self.entropy_coef = config.get("entropy_coef", 0.01)

    def _to_tensor(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def act(self, state, eval=False):
        with torch.no_grad():
            x = self._to_tensor(state)
            if self.discrete:
                probs = self.policy(x)
                if eval:
                    return int(probs.argmax().item())
                return int(torch.distributions.Categorical(probs).sample().item())
            if eval:
                return self.policy.mean_action(x).cpu().numpy()[0]
            a, _ = self.policy.sample(x)
            return a.cpu().numpy()[0]

    def train(self, env, config, tracker):
        ep = 0
        t_global = 0
        losses = []
        while t_global < config["steps"]:
            state, _ = env.reset()
            done = False
            ret = 0.0
            while not done and t_global < config["steps"]:
                x = self._to_tensor(state)
                v = self.critic(x)
                if self.discrete:
                    probs = self.policy(x)
                    dist = torch.distributions.Categorical(probs)
                    a = dist.sample()
                    log_prob = dist.log_prob(a)
                    entropy = dist.entropy()
                    action = int(a.item())
                else:
                    a, log_prob = self.policy.sample(x)
                    action = a.cpu().numpy()[0]
                    entropy = None
                ns, r, term, trunc, _ = env.step(action)
                done = bool(term or trunc)
                with torch.no_grad():
                    v_next = self.critic(self._to_tensor(ns)) if not done else torch.zeros_like(v)
                delta = r + self.gamma * v_next - v
                actor_loss = -log_prob * delta.detach()
                if entropy is not None and self.entropy_coef:
                    actor_loss = actor_loss - self.entropy_coef * entropy
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
                critic_loss = F.mse_loss(v, (r + self.gamma * v_next).detach())
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()
                losses.append(float(delta.item()))
                state = ns
                ret += r
                t_global += 1
            ep += 1
            tracker.log(timestep=t_global, episode=ep, ret=ret,
                        loss=float(np.mean(losses)) if losses else None)
            losses = []
        self.episodes = ep

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.critic.load_state_dict(data["critic"])