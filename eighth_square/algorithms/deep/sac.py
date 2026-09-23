import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent, evaluate
from .networks import ContinuousCritic, GaussianPolicy, polyak_copy
from .replay import ContinuousReplayBuffer


class SAC(BaseAgent):
    family = "actor-critic"
    policy = "off-policy"
    action_space = "continuous"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 128)
        self.lr = config.get("lr", 3e-4)
        self.hidden = config.get("hidden", 256)
        self.warmup = config.get("warmup", 1000)
        self.update_freq = config.get("update_freq", 1)
        self.alpha = config.get("alpha", 0.2)
        self.auto_tune = config.get("auto_tune", True)
        self.eval_freq = config.get("eval_freq", 5_000)
        self.eval_episodes = config.get("eval_episodes", 5)
        in_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(env.action_space.shape[0])
        self.policy = GaussianPolicy(in_dim, self.hidden, self.action_dim).to(self.device)
        self.critic1 = ContinuousCritic(in_dim, self.action_dim, self.hidden).to(self.device)
        self.critic2 = ContinuousCritic(in_dim, self.action_dim, self.hidden).to(self.device)
        self.critic1_target = ContinuousCritic(in_dim, self.action_dim, self.hidden).to(self.device)
        self.critic2_target = ContinuousCritic(in_dim, self.action_dim, self.hidden).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.lr)
        if self.auto_tune:
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)
            self.target_entropy = -self.action_dim
        self.buffer = ContinuousReplayBuffer(config.get("buffer_size", 100_000), in_dim, self.action_dim, self.device)
        self.t = 0

    def _t(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    @property
    def alpha_val(self):
        return self.log_alpha.exp() if self.auto_tune else torch.tensor(self.alpha, device=self.device)

    def act(self, state, eval=False):
        with torch.no_grad():
            x = self._t(state)
            if eval:
                return self.policy.mean_action(x).cpu().numpy()[0]
            a, _ = self.policy.sample(x)
            return a.cpu().numpy()[0]

    def _update(self):
        obs, act, rew, obs2, done = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            a2, lp2 = self.policy.sample(obs2)
            q1 = self.critic1_target(obs2, a2)
            q2 = self.critic2_target(obs2, a2)
            target_q = rew + self.gamma * (1 - done) * (torch.min(q1, q2) - self.alpha_val * lp2)
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        a_new, lp_new = self.policy.sample(obs)
        q1_new = self.critic1(obs, a_new)
        q2_new = self.critic2(obs, a_new)
        policy_loss = (self.alpha_val * lp_new - torch.min(q1_new, q2_new)).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        if self.auto_tune:
            alpha_loss = -(self.log_alpha * (lp_new.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        polyak_copy(self.critic1, self.critic1_target, self.tau)
        polyak_copy(self.critic2, self.critic2_target, self.tau)
        return float(critic_loss.item())

    def train(self, env, config, tracker):
        state, _ = env.reset()
        self.t = 0
        ep = 0
        ep_ret = 0.0
        losses = []
        while self.t < config["steps"]:
            a = self.act(state)
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            self.buffer.push(state, a, r, ns, done)
            ep_ret += r
            state = ns
            self.t += 1
            if done:
                ep += 1
                tracker.log(timestep=self.t, episode=ep, ret=float(ep_ret),
                            loss=float(np.mean(losses)) if losses else None)
                ep_ret = 0.0
                losses = []
                state, _ = env.reset()
            if self.t >= self.warmup and self.t % self.update_freq == 0:
                losses.append(self._update())
            if self.t % self.eval_freq == 0 and self.t > 0:
                tracker.log(timestep=self.t, eval_return=evaluate(self, env, self.eval_episodes))
                state, _ = env.reset()
        self.episodes = ep

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(),
                    "critic1": self.critic1.state_dict(), "critic2": self.critic2.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.critic1.load_state_dict(data["critic1"])
        self.critic2.load_state_dict(data["critic2"])