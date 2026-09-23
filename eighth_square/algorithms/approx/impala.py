import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from ..base import BaseAgent
from .networks import Critic, DiscretePolicy


class IMPALA(BaseAgent):
    """Importance-weighted actor-learner: A2C-style parallel rollouts corrected
    by the V-trace operator, which bounds the importance ratios rho (for the
    value target) and c (for the trace) so the update stays stable even when
    the acting policy lags the learner (Espeholt et al. 2018).

    Single-learner port: the acting policy is only one update behind the
    learner, so the true ratio rho ~ 1 + O(lr).  The clipped ratios are used
    for the value target and the trace (paper defaults c1 = c2 = 1); the
    policy gradient is the on-policy one, since an IS-weighted gradient here
    only adds variance proportional to the last update's drift."""

    family = "actor-critic"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.n_envs = config.get("n_envs", 8)
        self.n_steps = config.get("n_steps", 20)
        self.c1 = config.get("c1", 1.0)
        self.c2 = config.get("c2", 1.0)
        in_dim = int(np.prod(env.observation_space.shape))
        hidden = config.get("hidden", 128)
        self.policy = DiscretePolicy(in_dim, hidden, int(env.action_space.n)).to(self.device)
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
            probs = self.policy(self._batch([state]))
            if eval:
                return int(probs.argmax().item())
            return int(torch.distributions.Categorical(probs).sample().item())

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
                    probs = self.policy(x)
                    dist = torch.distributions.Categorical(probs)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                obs_buf.append(x)
                act_buf.append(a)
                lp_buf.append(lp)
                val_buf.append(v)
                nobs, r, term, trunc, _ = self.envs.step(a.cpu().numpy())
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
                v_last = self.critic(self._batch(obs)).cpu().numpy().squeeze(1)
            obs_t = torch.cat(obs_buf)
            act_t = torch.cat(act_buf)
            lp_t = torch.cat(lp_buf)
            probs_new = self.policy(obs_t)
            dist_new = torch.distributions.Categorical(probs_new)
            lp_new = dist_new.log_prob(act_t)
            ent = dist_new.entropy()
            with torch.no_grad():
                rho = (lp_new - lp_t).exp().clamp(max=10.0)
            rho_hist = torch.minimum(torch.tensor(self.c1, device=self.device), rho)
            rho_trace = torch.minimum(torch.tensor(self.c2, device=self.device), rho)
            rew = np.stack(rew_buf)
            done = np.stack(done_buf).astype(np.float32)
            val = np.stack([v.cpu().numpy().squeeze(1) for v in val_buf])
            rho_hist_np = rho_hist.cpu().numpy().reshape(self.n_steps, self.n_envs)
            rho_trace_np = rho_trace.cpu().numpy().reshape(self.n_steps, self.n_envs)
            n_steps, n_envs = rew.shape
            delta = np.zeros((n_steps, n_envs))
            for t in reversed(range(n_steps)):
                v_next = v_last if t == n_steps - 1 else val[t + 1]
                delta[t] = rho_hist_np[t] * (rew[t] + self.gamma * v_next * (1 - done[t]) - val[t])
            v_trace = np.zeros((n_steps, n_envs))
            v_trace[n_steps - 1] = val[n_steps - 1] + delta[n_steps - 1]
            for t in reversed(range(n_steps - 1)):
                v_trace[t] = (val[t] + delta[t]
                              + self.gamma * rho_trace_np[t + 1] * (1 - done[t])
                              * (v_trace[t + 1] - val[t + 1]))
            adv = np.zeros((n_steps, n_envs))
            for t in range(n_steps):
                v_next = v_last if t == n_steps - 1 else v_trace[t + 1]
                adv[t] = rew[t] + self.gamma * v_next * (1 - done[t]) - val[t]
            vt = torch.as_tensor(v_trace.reshape(-1, 1), dtype=torch.float32, device=self.device)
            adv_t = torch.as_tensor(adv.reshape(-1, 1), dtype=torch.float32, device=self.device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            policy_loss = -(adv_t * lp_new.unsqueeze(1)).mean()
            policy_loss = policy_loss - self.entropy_coef * ent.mean()
            value_loss = F.mse_loss(self.critic(obs_t), vt)
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