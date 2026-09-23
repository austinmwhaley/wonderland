import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..tabular.common import episode_stats


class AgentQ(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class MixingNet(nn.Module):
    """Nonnegative-weighted mixing of per-agent Q-values into a joint Q,
    with weights and bias produced by a hypernetwork of the joint state
    (abs() enforces the IGM condition)."""

    def __init__(self, obs_dim, n_agents, hidden=128):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Sequential(nn.Linear(obs_dim * n_agents, hidden), nn.ReLU(),
                                      nn.Linear(hidden, n_agents * hidden))
        self.hyper_w2 = nn.Sequential(nn.Linear(obs_dim * n_agents, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden))
        self.hyper_b1 = nn.Linear(obs_dim * n_agents, hidden)
        self.hyper_b2 = nn.Linear(obs_dim * n_agents, 1)
        self.hidden = hidden

    def forward(self, qs, s_all):
        batch = qs.shape[0]
        w1 = torch.abs(self.hyper_w1(s_all)).view(batch, self.n_agents, self.hidden)
        b1 = self.hyper_b1(s_all)
        h = torch.relu(torch.bmm(qs.unsqueeze(1), w1).squeeze(1) + b1)
        w2 = torch.abs(self.hyper_w2(s_all)).view(batch, self.hidden, 1)
        b2 = self.hyper_b2(s_all)
        return torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2


class QMIXAgent(BaseAgent):
    """QMIX (Rashid et al. 2018): per-agent Q-networks whose values are
    combined by a state-dependent nonnegative mixer into a joint Q; trained
    by joint TD (the argmax of the joint Q equals the per-agent argmaxes)."""

    family = "multi-agent"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = torch.device(config.get("device", "cpu"))
        obs_dim = int(getattr(env, "obs_dim", 4))
        self.n_agents = getattr(env, "n_agents", 2)
        self.n_actions = getattr(env, "n_actions", 2)
        hidden = config.get("hidden", 128)
        self.qnets = [AgentQ(obs_dim, self.n_actions, hidden).to(self.device) for _ in range(self.n_agents)]
        self.qnets_t = [AgentQ(obs_dim, self.n_actions, hidden).to(self.device) for _ in range(self.n_agents)]
        self.mixer = MixingNet(obs_dim, self.n_agents, hidden).to(self.device)
        self.mixer_t = MixingNet(obs_dim, self.n_agents, hidden).to(self.device)
        for q, qt in zip(self.qnets, self.qnets_t):
            qt.load_state_dict(q.state_dict())
        self.mixer_t.load_state_dict(self.mixer.state_dict())
        self.opt = torch.optim.Adam(
            [p for q in self.qnets for p in q.parameters()] + list(self.mixer.parameters()),
            lr=config.get("lr", 1e-3))
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.buffer = []
        self.eps_start = config.get("eps_start", 1.0)
        self.eps_end = config.get("eps_end", 0.01)

    def act(self, state, eval=True):
        eps = self.eps_end if eval else self.eps_start
        actions = []
        with torch.no_grad():
            for i, q in enumerate(self.qnets):
                oi = np.asarray(state[i], dtype=np.float32)
                qv = q(torch.as_tensor(oi, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
                if self.rng.random() > eps:
                    actions.append(int(torch.argmax(qv).item()))
                else:
                    actions.append(int(self.rng.integers(self.n_actions)))
        return tuple(actions)

    def train(self, env, config, tracker):
        steps = config.get("steps", 50_000)
        batch_size = config.get("batch_size", 64)
        buf_size = config.get("buffer_size", 100_000)
        eps = self.eps_start
        eps_decay = (self.eps_start - self.eps_end) / max(steps, 1)
        ep = 0
        env_steps = 0
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
                qv1 = self.qnets[0](torch.as_tensor(s1, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
                qv2 = self.qnets[1](torch.as_tensor(s2, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
            a1 = int(torch.argmax(qv1).item()) if self.rng.random() > eps else int(self.rng.integers(self.n_actions))
            a2 = int(torch.argmax(qv2).item()) if self.rng.random() > eps else int(self.rng.integers(self.n_actions))
            (ns1, ns2), r, term, trunc, _ = env.step((a1, a2))
            done = bool(term or trunc)
            self.buffer.append((s1.copy(), s2.copy(), a1, a2, float(r), done, ns1.copy(), ns2.copy()))
            if len(self.buffer) > buf_size:
                self.buffer = self.buffer[-buf_size:]
            if len(self.buffer) >= batch_size:
                self._learn(config, batch_size)
            s1, s2 = ns1, ns2
            ret += r
            t += 1
            env_steps += 1
            eps = max(self.eps_end, eps - eps_decay)
        self.episodes = ep

    def _learn(self, config, batch_size):
        idx = self.rng.integers(0, len(self.buffer), size=batch_size)
        b = [self.buffer[i] for i in idx]
        s1 = torch.as_tensor(np.stack([x[0] for x in b]), dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.stack([x[1] for x in b]), dtype=torch.float32, device=self.device)
        a1 = torch.as_tensor(np.array([x[2] for x in b]), dtype=torch.long, device=self.device)
        a2 = torch.as_tensor(np.array([x[3] for x in b]), dtype=torch.long, device=self.device)
        r = torch.as_tensor(np.array([x[4] for x in b]), dtype=torch.float32, device=self.device).unsqueeze(1)
        done = torch.as_tensor(np.array([x[5] for x in b]), dtype=torch.float32, device=self.device).unsqueeze(1)
        ns1 = torch.as_tensor(np.stack([x[6] for x in b]), dtype=torch.float32, device=self.device)
        ns2 = torch.as_tensor(np.stack([x[7] for x in b]), dtype=torch.float32, device=self.device)
        s_all = torch.cat([s1, s2], dim=1)
        ns_all = torch.cat([ns1, ns2], dim=1)
        qs = torch.stack([q(s1) if i == 0 else q(s2) for i, q in enumerate(self.qnets)], dim=1)
        q_sel = torch.stack([qs[:, i, :].gather(1, a.unsqueeze(1)).squeeze(1)
                             for i, a in enumerate([a1, a2])], dim=1)
        q_tot = self.mixer(q_sel, s_all)
        with torch.no_grad():
            qn = torch.stack([q(ns1) if i == 0 else q(ns2) for i, q in enumerate(self.qnets_t)], dim=1)
            a_max = qn.argmax(dim=2)
            qn_sel = torch.stack([qn[:, i, :].gather(1, a_max[:, i].unsqueeze(1)).squeeze(1)
                                  for i in range(self.n_agents)], dim=1)
            q_tot_t = self.mixer_t(qn_sel, ns_all)
            target = r + self.gamma * (1 - done) * q_tot_t
        self.opt.zero_grad()
        F.smooth_l1_loss(q_tot, target).backward()
        self.opt.step()
        for q, qt in zip(self.qnets, self.qnets_t):
            for p, pt in zip(q.parameters(), qt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.mixer.parameters(), self.mixer_t.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def save(self, path):
        torch.save({"qnets": [q.state_dict() for q in self.qnets],
                    "mixer": self.mixer.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for q, sd in zip(self.qnets, ckpt["qnets"]):
            q.load_state_dict(sd)
        self.mixer.load_state_dict(ckpt["mixer"])