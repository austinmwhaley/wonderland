import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from ..base import BaseAgent
from .networks import Critic, DiscretePolicy, GaussianPolicy


class A2C(BaseAgent):
    family = "actor-critic"
    policy = "on-policy"
    action_space = "both"
    state_space = "both"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.n_envs = config.get("n_envs", 8)
        self.n_steps = config.get("n_steps", 5)
        self.discrete = hasattr(env.action_space, "n")
        in_dim = int(np.prod(env.observation_space.shape))
        hidden = config.get("hidden", 128)
        if self.discrete:
            self.policy = DiscretePolicy(in_dim, hidden, int(env.action_space.n)).to(self.device)
        else:
            self.policy = GaussianPolicy(in_dim, hidden, int(env.action_space.shape[0])).to(self.device)
        self.critic = Critic(in_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=config.get("lr", 3e-4))
        self.entropy_coef = config.get("entropy_coef", 0.01)
        try:
            gid = env.unwrapped.spec.id
            self.envs = gym.vector.SyncVectorEnv(
                [lambda: gym.make(gid) for _ in range(self.n_envs)])
        except AttributeError:
            self.envs = gym.vector.SyncVectorEnv([lambda: env] * self.n_envs)

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
        obs, _ = self.envs.reset()
        ep = 0
        t_global = 0
        ep_returns = np.zeros(self.n_envs)
        while t_global < config["steps"]:
            obs_buf, act_buf, lp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
            for _ in range(self.n_steps):
                x = self._batch(obs)
                with torch.no_grad():
                    v = self.critic(x)
                    if self.discrete:
                        probs = self.policy(x)
                        dist = torch.distributions.Categorical(probs)
                        a = dist.sample()
                        lp = dist.log_prob(a)
                        ent = dist.entropy()
                        actions = a.cpu().numpy()
                    else:
                        a, lp = self.policy.sample(x)
                        ent = None
                        actions = a.cpu().numpy()
                obs_buf.append(x)
                act_buf.append(a)
                lp_buf.append(lp)
                val_buf.append(v)
                nobs, r, term, trunc, _ = self.envs.step(actions)
                done = np.logical_or(term, trunc)
                rew_buf.append(r)
                done_buf.append(term)
                ep_returns += r
                if done.any():
                    for i in np.where(done)[0]:
                        ep += 1
                        tracker.log(timestep=t_global, episode=ep, ret=float(ep_returns[i]))
                    ep_returns[done] = 0.0
                obs = nobs
                t_global += self.n_envs
            with torch.no_grad():
                x_last = self._batch(obs)
                v_last = self.critic(x_last).cpu().numpy().squeeze(1)
            returns = []
            adv = []
            G = v_last.copy()
            for t in reversed(range(self.n_steps)):
                G = rew_buf[t] + self.gamma * G * (1 - done_buf[t])
                returns.insert(0, G.copy())
                adv.insert(0, G - val_buf[t].cpu().numpy().squeeze())
            obs_t = torch.cat(obs_buf)
            act_t = torch.cat(act_buf)
            lp_t = torch.cat(lp_buf)
            ret_t = torch.as_tensor(np.concatenate(returns), dtype=torch.float32, device=self.device).unsqueeze(1)
            adv_t = torch.as_tensor(np.concatenate(adv), dtype=torch.float32, device=self.device).unsqueeze(1)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            if self.discrete:
                probs_new = self.policy(obs_t)
                dist_new = torch.distributions.Categorical(probs_new)
                lp_new = dist_new.log_prob(act_t)
                ent_new = dist_new.entropy()
            else:
                lp_new = self.policy.log_prob(obs_t, act_t)
                ent_new = None
            v_new = self.critic(obs_t)
            policy_loss = -(lp_new.unsqueeze(-1) * adv_t).mean()
            if ent_new is not None:
                policy_loss = policy_loss - self.entropy_coef * ent_new.mean()
            value_loss = F.mse_loss(v_new, ret_t)
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()
            tracker.log(timestep=t_global, episode=ep, loss=float((policy_loss + value_loss).item()))
        self.episodes = ep

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.critic.load_state_dict(data["critic"])