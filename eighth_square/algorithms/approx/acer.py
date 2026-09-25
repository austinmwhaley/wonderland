import numpy as np
import torch
import torch.nn.functional as F

from ..base import BaseAgent
from .networks import DiscretePolicy, MLP


class ACERBuffer:
    """Replay of complete episodes (state, action, reward, done, behavior probs)."""

    def __init__(self, capacity, obs_dim, n_actions):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.episodes = []

    def push_episode(self, ep):
        obs, act, rew, done, mu = ep
        self.episodes.append(
            (
                np.asarray(obs, dtype=np.float32),
                np.asarray(act, dtype=np.int64),
                np.asarray(rew, dtype=np.float32),
                np.asarray(done, dtype=np.float32),
                np.asarray(mu, dtype=np.float32),
            )
        )
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def sample(self, device, max_len=1000):
        obs, act, rew, done, mu = self.episodes[np.random.randint(len(self.episodes))]
        n = len(rew)
        if n > max_len:
            start = np.random.randint(0, n - max_len + 1)
            obs, act, rew, done, mu = (
                obs[start : start + max_len],
                act[start : start + max_len],
                rew[start : start + max_len],
                done[start : start + max_len],
                mu[start : start + max_len],
            )
        t = torch.from_numpy
        return (
            t(obs).to(device),
            t(act).to(device),
            t(rew).to(device).unsqueeze(1),
            t(done).to(device).unsqueeze(1),
            t(mu).to(device),
        )

    def __len__(self):
        return len(self.episodes)


class ACER(BaseAgent):
    """Sample-efficient actor-critic with experience replay of full episodes
    (Wang et al. 2017): retrace targets folded backwards through the episode
    (Q_ret = rho-bar * (r + gamma * Q_ret - Q) + V, rho-bar = min(1, rho)),
    truncated importance sampling with the bias-correction term
    relu(1 - c * mu / pi) * pi * (Q - V) * log pi, and an efficient-TRPO
    trust region on KL(avg_policy || policy) that bounds every update.

    Known weakness: the Q-based advantage cannot differentiate actions while
    the behavior policy is near-uniform on constant-reward tasks (e.g.
    acrobot), so the actor receives no learning signal there; the algorithm
    is reliable on tasks with rich reward structure (e.g. cartpole)."""

    family = "actor-critic"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = config.get("device", "cpu")
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 7e-4)
        self.hidden = config.get("hidden", 128)
        self.batch_size = config.get("batch_size", 64)
        self.n_times_replay = config.get("n_times_replay", 4)
        self.c = config.get("c", 10.0)
        self.beta_kl = config.get("beta_kl", 1.0)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.tau = config.get("tau", 0.005)
        in_dim = int(np.prod(env.observation_space.shape))
        self.nA = int(env.action_space.n)
        self.policy = DiscretePolicy(in_dim, self.hidden, self.nA).to(self.device)
        self.target_policy = DiscretePolicy(in_dim, self.hidden, self.nA).to(self.device)
        self.q = MLP(in_dim, self.hidden, self.nA).to(self.device)
        self.target_q = MLP(in_dim, self.hidden, self.nA).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.q.parameters()), lr=self.lr
        )
        self.avg_policy = DiscretePolicy(in_dim, self.hidden, self.nA).to(self.device)
        self.avg_policy.load_state_dict(self.policy.state_dict())
        self.use_trust_region = config.get("use_trust_region", True)
        self.trust_region_delta = config.get("trust_region_delta", 1.0)
        self.trust_region_alpha = config.get("trust_region_alpha", 0.99)
        self.buffer = ACERBuffer(config.get("buffer_size", 1000), in_dim, self.nA)
        self.t = 0

    def _batch(self, obs):
        return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)

    def act(self, state, eval=False):
        with torch.no_grad():
            probs = self.policy(self._batch([state]))
            if eval:
                return int(probs.argmax().item())
            return int(torch.distributions.Categorical(probs).sample().item())

    def _polyak(self):
        for ps, pt in zip(self.policy.parameters(), self.target_policy.parameters()):
            pt.data.mul_(1 - self.tau).add_(ps.data, alpha=self.tau)
        for ps, pt in zip(self.q.parameters(), self.target_q.parameters()):
            pt.data.mul_(1 - self.tau).add_(ps.data, alpha=self.tau)
        for ps, pt in zip(self.policy.parameters(), self.avg_policy.parameters()):
            pt.data.mul_(self.trust_region_alpha).add_(ps.data, alpha=1 - self.trust_region_alpha)

    def _trust_region_loss(self, policy_loss, probs, probs_avg, delta):
        params = [p for p in self.policy.parameters() if p.requires_grad]
        g = torch.autograd.grad(policy_loss, params, retain_graph=True, allow_unused=True)
        g = [gp if gp is not None else torch.zeros_like(p) for gp, p in zip(g, params)]
        kl = (
            (probs_avg * (probs_avg.clamp(min=1e-8).log() - probs.clamp(min=1e-8).log()))
            .sum(-1)
            .mean()
        )
        k = torch.autograd.grad(-kl, params, retain_graph=True, allow_unused=True)
        k = [kp if kp is not None else torch.zeros_like(p) for kp, p in zip(k, params)]
        kg_dot = sum((kp * gp).sum() for kp, gp in zip(k, g))
        kk_dot = sum((kp * kp).sum() for kp in k)
        k_factor = max(0.0, (float(kg_dot) - delta) / float(kk_dot)) if kk_dot > 0 else 0.0
        loss = 0.0
        for p, gp, kp in zip(params, g, k):
            loss = loss + (p * (gp - k_factor * kp)).sum()
        return loss, float(kl)

    def _kl(self, mu, probs):
        return (mu * (mu.clamp(min=1e-8).log() - probs.clamp(min=1e-8).log())).sum(-1).mean()

    def _update(self):
        obs, act, rew, done, mu = self.buffer.sample(self.device)
        probs = self.policy(obs)
        dist = torch.distributions.Categorical(probs)
        lp = dist.log_prob(act)
        lp_mu = torch.log(mu.gather(1, act.unsqueeze(1)).clamp(min=1e-8))
        rho = (lp.unsqueeze(1) - lp_mu).exp().clamp(max=self.c)
        rho_bar = torch.minimum(torch.tensor(1.0, device=self.device), rho)
        q_now = self.q(obs)
        v = (probs * q_now).sum(-1, keepdim=True)
        q_a = q_now.gather(1, act.unsqueeze(1))
        q_ret = torch.zeros_like(rew)
        acc = torch.zeros(rew.shape[0], 1, device=self.device)
        for t in reversed(range(len(rew))):
            acc = rew[t] + self.gamma * acc
            acc = rho_bar[t] * (acc - q_a[t].detach()) + v[t].detach()
            q_ret[t, 0] = acc[0, 0]
        adv = (q_ret - v).detach()
        g_loss = -(rho * adv * lp.unsqueeze(1)).mean()
        corr_w = F.relu(1 - self.c * mu / probs.clamp(min=1e-8)) * probs
        g_loss = g_loss - (corr_w * (q_now - v) * torch.log(probs.clamp(min=1e-8))).sum(-1).mean()
        if self.use_trust_region:
            with torch.no_grad():
                probs_avg = self.avg_policy(obs)
            policy_loss, _ = self._trust_region_loss(
                g_loss, probs, probs_avg, self.trust_region_delta
            )
        elif self.beta_kl > 0:
            policy_loss = g_loss + self.beta_kl * self._kl(mu.detach(), probs)
        else:
            policy_loss = g_loss
        policy_loss = policy_loss - self.entropy_coef * dist.entropy().mean()
        q_pred = self.q(obs).gather(1, act.unsqueeze(1))
        value_loss = F.mse_loss(q_pred, q_ret.detach())
        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 0.5)
        self.optimizer.step()
        self._polyak()
        return float((policy_loss + value_loss).item())

    def train(self, env, config, tracker):
        state, _ = env.reset()
        self.t = 0
        ep = 0
        losses = []
        while self.t < config["steps"]:
            ep_obs, ep_act, ep_rew, ep_done, ep_mu = [], [], [], [], []
            while True:
                x = self._batch([state])
                with torch.no_grad():
                    probs = self.policy(x)
                    a = int(torch.distributions.Categorical(probs).sample().item())
                    p = probs[0].cpu().numpy()
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                ep_obs.append(state)
                ep_act.append(a)
                ep_rew.append(r)
                ep_done.append(1.0 if done else 0.0)
                ep_mu.append(p)
                state = ns
                self.t += 1
                if done:
                    break
            self.buffer.push_episode((ep_obs, ep_act, ep_rew, ep_done, ep_mu))
            ep += 1
            tracker.log(
                timestep=self.t,
                episode=ep,
                ret=float(sum(ep_rew)),
                loss=float(np.mean(losses)) if losses else None,
            )
            losses = []
            for _ in range(self.n_times_replay):
                losses.append(self._update())
            state, _ = env.reset()
        self.episodes = ep

    def save(self, path):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "q": self.q.state_dict(),
                "target_policy": self.target_policy.state_dict(),
                "target_q": self.target_q.state_dict(),
            },
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.q.load_state_dict(data["q"])
        self.target_policy.load_state_dict(data["target_policy"])
        self.target_q.load_state_dict(data["target_q"])
