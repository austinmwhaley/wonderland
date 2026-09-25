import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent, evaluate
from .networks import ContinuousCritic, DeterministicActor, polyak_copy
from .replay import ContinuousReplayBuffer


class TD3(BaseAgent):
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
        self.lr = config.get("lr", 1e-3)
        self.hidden = config.get("hidden", 256)
        self.warmup = config.get("warmup", 1000)
        self.policy_freq = config.get("policy_freq", 2)
        self.noise_std = config.get("noise_std", 0.1)
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.eval_freq = config.get("eval_freq", 5_000)
        self.eval_episodes = config.get("eval_episodes", 5)
        in_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(env.action_space.shape[0])
        bound = float(env.action_space.high[0])
        self.actor = DeterministicActor(in_dim, self.hidden, self.action_dim, bound).to(self.device)
        self.actor_target = DeterministicActor(in_dim, self.hidden, self.action_dim, bound).to(
            self.device
        )
        self.critic1 = ContinuousCritic(in_dim, self.action_dim, self.hidden).to(self.device)
        self.critic2 = ContinuousCritic(in_dim, self.action_dim, self.hidden).to(self.device)
        self.critic1_target = ContinuousCritic(in_dim, self.action_dim, self.hidden).to(self.device)
        self.critic2_target = ContinuousCritic(in_dim, self.action_dim, self.hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.lr
        )
        self.buffer = ContinuousReplayBuffer(
            config.get("buffer_size", 100_000), in_dim, self.action_dim, self.device
        )
        self.t = 0

    def _t(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def act(self, state, eval=False):
        with torch.no_grad():
            a = self.actor(self._t(state))
            if not eval and self.t >= self.warmup:
                a = a + torch.randn_like(a) * self.noise_std
            return a.cpu().numpy()[0]

    def _update(self):
        obs, act, rew, obs2, done = self.buffer.sample(self.batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            a2 = (self.actor_target(obs2) + noise).clamp(-1.0, 1.0)
            q1 = self.critic1_target(obs2, a2)
            q2 = self.critic2_target(obs2, a2)
            target_q = rew + self.gamma * (1 - done) * torch.min(q1, q2)
        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        loss = float(critic_loss.item())
        if self.t % self.policy_freq == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            polyak_copy(self.actor, self.actor_target, self.tau)
            polyak_copy(self.critic1, self.critic1_target, self.tau)
            polyak_copy(self.critic2, self.critic2_target, self.tau)
        return loss

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
                tracker.log(
                    timestep=self.t,
                    episode=ep,
                    ret=float(ep_ret),
                    loss=float(np.mean(losses)) if losses else None,
                )
                ep_ret = 0.0
                losses = []
                state, _ = env.reset()
            if self.t >= self.warmup:
                losses.append(self._update())
            if self.t % self.eval_freq == 0 and self.t > 0:
                tracker.log(timestep=self.t, eval_return=evaluate(self, env, self.eval_episodes))
                state, _ = env.reset()
        self.episodes = ep

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data["actor"])
        self.critic1.load_state_dict(data["critic1"])
        self.critic2.load_state_dict(data["critic2"])
