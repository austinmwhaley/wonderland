import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..tabular.common import episode_stats


class MPPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class MCritic(nn.Module):
    def __init__(self, obs_dim, n_agents, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim * n_agents, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


class MAPPOAgent(BaseAgent):
    """MAPPO (Yu et al. 2022): independent PPO policies for each agent,
    trained with the same shared advantage computed from a centralized
    critic that observes the joint state (CTDE)."""

    family = "multi-agent"
    policy = "on-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = torch.device(config.get("device", "cpu"))
        obs_dim = int(getattr(env, "obs_dim", 4))
        self.n_agents = getattr(env, "n_agents", 2)
        self.n_actions = getattr(env, "n_actions", 2)
        hidden = config.get("hidden", 128)
        self.policies = [MPPolicy(obs_dim, self.n_actions, hidden).to(self.device) for _ in range(self.n_agents)]
        self.critic = MCritic(obs_dim, self.n_agents, hidden).to(self.device)
        self.opt_ps = [torch.optim.Adam(p.parameters(), lr=config.get("lr", 3e-4)) for p in self.policies]
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=config.get("lr", 3e-4))
        self.gamma = config.get("gamma", 0.99)
        self.lamb = config.get("lambda", 0.95)
        self.clip = config.get("clip", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)

    def act(self, state, eval=True):
        actions = []
        with torch.no_grad():
            for i, p in enumerate(self.policies):
                oi = np.asarray(state[i], dtype=np.float32)
                probs = p(torch.as_tensor(oi, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
                if eval:
                    actions.append(int(torch.argmax(probs).item()))
                else:
                    actions.append(int(torch.multinomial(probs, 1).item()))
        return tuple(actions)

    def train(self, env, config, tracker):
        steps = config.get("steps", 50_000)
        batch_size = config.get("batch_size", 64)
        ep = 0
        env_steps = 0
        tbuf = []
        ret = 0.0
        t = 0
        done = False
        (s1, s2), _ = env.reset()
        while env_steps < steps:
            if done:
                ep += 1
                episode_stats(tracker, t, ep, ret)
                (s1, s2), _ = env.reset()
                ret = 0.0
                t = 0
                done = False
            with torch.no_grad():
                probs = [p(torch.as_tensor(np.stack([s1, s2]), dtype=torch.float32, device=self.device)) for p in self.policies]
            a1 = int(torch.multinomial(probs[0][0], 1).item())
            a2 = int(torch.multinomial(probs[1][1], 1).item())
            logp1 = float(torch.log(probs[0][0, a1] + 1e-8).item())
            logp2 = float(torch.log(probs[1][1, a2] + 1e-8).item())
            (ns1, ns2), r, term, trunc, _ = env.step((a1, a2))
            done = bool(term or trunc)
            tbuf.append((s1.copy(), s2.copy(), a1, a2, logp1, logp2, float(r), done, ns1.copy(), ns2.copy()))
            s1, s2 = ns1, ns2
            ret += r
            t += 1
            env_steps += 1
            if len(tbuf) >= config.get("rollout_len", 512) or env_steps >= steps:
                self._learn(config, batch_size, tbuf)
                tbuf = []
        self.episodes = ep

    def _learn(self, config, batch_size, tbuf):
        s1 = torch.as_tensor(np.stack([x[0] for x in tbuf]), dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.stack([x[1] for x in tbuf]), dtype=torch.float32, device=self.device)
        a1 = torch.as_tensor(np.array([x[2] for x in tbuf]), dtype=torch.long, device=self.device)
        a2 = torch.as_tensor(np.array([x[3] for x in tbuf]), dtype=torch.long, device=self.device)
        lp1 = torch.as_tensor(np.array([x[4] for x in tbuf]), dtype=torch.float32, device=self.device)
        lp2 = torch.as_tensor(np.array([x[5] for x in tbuf]), dtype=torch.float32, device=self.device)
        r = torch.as_tensor(np.array([x[6] for x in tbuf]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.array([x[7] for x in tbuf]), dtype=torch.float32, device=self.device)
        s_all = torch.cat([s1, s2], dim=1)
        with torch.no_grad():
            v = self.critic(s_all).squeeze(1)
            v_next = torch.cat([v[1:], torch.zeros(1, device=self.device)])
            deltas = r + self.gamma * v_next * (1 - done) - v
            adv = torch.zeros_like(deltas)
            gae = 0.0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.gamma * self.lamb * (1 - done[t]) * gae
                adv[t] = gae
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            v_target = adv + v
        for epoch in range(config.get("epochs", 4)):
            idx = self.rng.permutation(len(tbuf))
            for i in range(0, len(tbuf), batch_size):
                mb = idx[i:i + batch_size]
                s1_mb, s2_mb = s1[mb], s2[mb]
                a1_mb, a2_mb = a1[mb], a2[mb]
                adv_mb = adv[mb]
                v_mb = v_target[mb]
                probs1 = self.policies[0](s1_mb)
                probs2 = self.policies[1](s2_mb)
                logp1 = torch.log(probs1.gather(1, a1_mb.unsqueeze(1)).squeeze(1) + 1e-8)
                logp2 = torch.log(probs2.gather(1, a2_mb.unsqueeze(1)).squeeze(1) + 1e-8)
                ratio1 = torch.exp(logp1 - lp1[mb])
                ratio2 = torch.exp(logp2 - lp2[mb])
                ent1 = -(probs1 * torch.log(probs1 + 1e-8)).sum(1)
                ent2 = -(probs2 * torch.log(probs2 + 1e-8)).sum(1)
                for ratio, logp, ent, p, opt in ((ratio1, logp1, ent1, self.policies[0], self.opt_ps[0]),
                                                 (ratio2, logp2, ent2, self.policies[1], self.opt_ps[1])):
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_mb
                    loss = -(torch.min(surr1, surr2).mean() + self.entropy_coef * ent.mean())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                v_pred = self.critic(torch.cat([s1_mb, s2_mb], dim=1)).squeeze(1)
                self.opt_c.zero_grad()
                F.mse_loss(v_pred, v_mb).backward()
                self.opt_c.step()

    def save(self, path):
        torch.save({"policies": [p.state_dict() for p in self.policies],
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for p, sd in zip(self.policies, ckpt["policies"]):
            p.load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic"])