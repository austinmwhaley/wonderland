import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent
from .networks import Critic, DiscretePolicy, GaussianPolicy


class PPO(BaseAgent):
    family = "policy-gradient"
    policy = "on-policy"
    action_space = "both"
    state_space = "both"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("lambda", 0.95)
        self.clip = config.get("clip", 0.2)
        self.rollout = config.get("rollout", 2048)
        self.epochs = config.get("epochs", 10)
        self.minibatch = config.get("minibatch", 64)
        self.discrete = hasattr(env.action_space, "n")
        in_dim = int(np.prod(env.observation_space.shape))
        hidden = config.get("hidden", 128)
        if self.discrete:
            self.policy = DiscretePolicy(in_dim, hidden, int(env.action_space.n)).to(self.device)
        else:
            self.policy = GaussianPolicy(in_dim, hidden, int(env.action_space.shape[0])).to(
                self.device
            )
        self.critic = Critic(in_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=config.get("lr", 3e-4),
        )
        self.lr0 = config.get("lr", 3e-4)
        self.total_steps = config.get("steps", 1)
        self.entropy_coef = config.get("entropy_coef", 0.0)

    def _anneal_lr(self, t):
        frac = max(0.0, 1.0 - t / self.total_steps)
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr0 * frac

    def _batch(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)

    def act(self, state, eval=False):
        with torch.no_grad():
            x = self._batch([state])
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
        state, _ = env.reset()
        t_global = 0
        self.total_steps = config["steps"]
        ep = 0
        ep_ret = 0.0
        while t_global < config["steps"]:
            obs_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
            for _ in range(self.rollout):
                x = self._batch([state])
                with torch.no_grad():
                    v = self.critic(x)
                    if self.discrete:
                        probs = self.policy(x)
                        dist = torch.distributions.Categorical(probs)
                        a = dist.sample()
                        lp = dist.log_prob(a)
                        action = int(a.item())
                    else:
                        a, lp = self.policy.sample(x)
                        action = a.cpu().numpy()[0]
                obs_buf.append(x)
                act_buf.append(a)
                lp_buf.append(lp)
                val_buf.append(v)
                ns, r, term, trunc, _ = env.step(action)
                done = bool(term or trunc)
                rew_buf.append(r)
                done_buf.append(bool(term))
                ep_ret += r
                state = ns
                t_global += 1
                if done:
                    ep += 1
                    tracker.log(timestep=t_global, episode=ep, ret=float(ep_ret))
                    ep_ret = 0.0
                    state, _ = env.reset()
            with torch.no_grad():
                v_last = self.critic(self._batch([state]))
            returns, advantages = [], []
            G = v_last.item()
            adv = 0.0
            for t in reversed(range(self.rollout)):
                done = done_buf[t]
                delta = rew_buf[t] + self.gamma * (0.0 if done else G) - val_buf[t].item()
                adv = delta + self.gamma * self.lam * (0.0 if done else adv)
                G = rew_buf[t] + self.gamma * (0.0 if done else G)
                advantages.insert(0, adv)
                returns.insert(0, G)
            obs_t = torch.cat(obs_buf)
            act_t = torch.cat(act_buf)
            lp_old = torch.cat(lp_buf).detach()
            ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)
            adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(
                1
            )
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            for _ in range(self.epochs):
                idx = torch.randperm(len(obs_t), device=self.device)
                for start in range(0, len(obs_t), self.minibatch):
                    i = idx[start : start + self.minibatch]
                    if self.discrete:
                        probs = self.policy(obs_t[i])
                        dist = torch.distributions.Categorical(probs)
                        lp_new = dist.log_prob(act_t[i])
                        ent = dist.entropy().mean()
                    else:
                        lp_new = self.policy.log_prob(obs_t[i], act_t[i])
                        ent = None
                    ratio = (lp_new - lp_old[i]).exp().unsqueeze(-1)
                    surr1 = ratio * adv_t[i]
                    surr2 = ratio.clamp(1 - self.clip, 1 + self.clip) * adv_t[i]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    if ent is not None and self.entropy_coef:
                        policy_loss = policy_loss - self.entropy_coef * ent
                    v_new = self.critic(obs_t[i])
                    value_loss = F.mse_loss(v_new, ret_t[i])
                    self.optimizer.zero_grad()
                    (policy_loss + value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optimizer.step()
                    last_loss = float((policy_loss + value_loss).item())
            self._anneal_lr(t_global)
            tracker.log(timestep=t_global, episode=ep, loss=last_loss)
        self.episodes = ep

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.critic.load_state_dict(data["critic"])
