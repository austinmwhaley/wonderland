import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent
from .networks import Critic, DiscretePolicy, GaussianPolicy


class Reinforce(BaseAgent):
    family = "policy-gradient"
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
            self.act_space_dim = None
        else:
            self.policy = GaussianPolicy(in_dim, hidden, int(env.action_space.shape[0])).to(self.device)
        self.use_baseline = config.get("baseline", True)
        if self.use_baseline:
            self.critic = Critic(in_dim, hidden).to(self.device)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 1e-3))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 1e-3))
        self.entropy_coef = config.get("entropy_coef", 0.0)

    def _to_tensor(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def act(self, state, eval=False):
        with torch.no_grad():
            x = self._to_tensor(state)
            if self.discrete:
                probs = self.policy(x)
                if eval:
                    return int(probs.argmax().item())
                dist = torch.distributions.Categorical(probs)
                return int(dist.sample().item())
            if eval:
                return self.policy.mean_action(x).cpu().numpy()[0]
            a, _ = self.policy.sample(x)
            return a.cpu().numpy()[0]

    def train(self, env, config, tracker):
        ep = 0
        t_global = 0
        while t_global < config["steps"]:
            state, _ = env.reset()
            done = False
            ret = 0.0
            log_probs, rewards, states, entropies = [], [], [], []
            while not done and t_global < config["steps"]:
                x = self._to_tensor(state)
                if self.discrete:
                    probs = self.policy(x)
                    dist = torch.distributions.Categorical(probs)
                    a = dist.sample()
                    log_probs.append(dist.log_prob(a))
                    entropies.append(dist.entropy())
                    action = int(a.item())
                else:
                    a, lp = self.policy.sample(x)
                    log_probs.append(lp)
                    action = a.cpu().numpy()[0]
                ns, r, term, trunc, _ = env.step(action)
                done = bool(term or trunc)
                rewards.append(r)
                states.append(x)
                state = ns
                ret += r
                t_global += 1
            G = 0.0
            returns = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
            if self.use_baseline:
                states_t = torch.cat(states)
                baseline = self.critic(states_t)
                adv = (returns - baseline).detach()
                critic_loss = F.mse_loss(baseline, returns)
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()
            else:
                adv = returns
            policy_loss = -torch.cat(log_probs).unsqueeze(-1) * adv
            if self.entropy_coef:
                policy_loss = policy_loss - self.entropy_coef * torch.cat(entropies).unsqueeze(-1)
            self.optimizer.zero_grad()
            policy_loss.mean().backward()
            self.optimizer.step()
            ep += 1
            tracker.log(timestep=t_global, episode=ep, ret=ret,
                        loss=float(policy_loss.mean().item()))
        self.episodes = ep

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(),
                    "critic": self.critic.state_dict() if self.use_baseline else None}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        if data.get("critic") is not None:
            self.critic.load_state_dict(data["critic"])