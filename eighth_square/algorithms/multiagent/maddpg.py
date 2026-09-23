import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..tabular.common import episode_stats


class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class Critic(nn.Module):
    def __init__(self, obs_dim, n_actions, n_agents, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim * n_agents + n_actions * n_agents, hidden),
                                 nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, obs_all, act_all):
        return self.net(torch.cat([obs_all, act_all], dim=1))


class MADDPGAgent(BaseAgent):
    """MADDPG (Lowe et al. 2017): per-agent policy with a centralized critic
    that sees all agents' observations and actions. Discrete-action variant:
    the actor outputs a softmax policy and the gradient is the expectation of
    the centralized Q under the actor's distribution."""

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
        self.actors = [Actor(obs_dim, self.n_actions, hidden).to(self.device) for _ in range(self.n_agents)]
        self.actors_t = [Actor(obs_dim, self.n_actions, hidden).to(self.device) for _ in range(self.n_agents)]
        self.critics = [Critic(obs_dim, self.n_actions, self.n_agents, hidden).to(self.device) for _ in range(self.n_agents)]
        self.critics_t = [Critic(obs_dim, self.n_actions, self.n_agents, hidden).to(self.device) for _ in range(self.n_agents)]
        for a, at in zip(self.actors_t, self.actors):
            at.load_state_dict(a.state_dict())
        for c, ct in zip(self.critics_t, self.critics):
            ct.load_state_dict(c.state_dict())
        self.opt_actors = [torch.optim.Adam(a.parameters(), lr=config.get("actor_lr", 1e-4)) for a in self.actors]
        self.opt_critics = [torch.optim.Adam(c.parameters(), lr=config.get("lr", 1e-3)) for c in self.critics]
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.buffer = []
        self.eps = config.get("eps_start", 0.3)
        self.eps_end = config.get("eps_end", 0.01)

    def act(self, state, eval=True):
        actions = []
        for a in self.actors:
            with torch.no_grad():
                probs = a(torch.as_tensor(np.stack(state), dtype=torch.float32, device=self.device).unsqueeze(0))[0]
            if eval or self.rng.random() > self.eps:
                actions.append(int(torch.argmax(probs).item()))
            else:
                actions.append(int(torch.multinomial(probs, 1).item()))
        return tuple(actions)

    def train(self, env, config, tracker):
        steps = config.get("steps", 50_000)
        batch_size = config.get("batch_size", 64)
        buf_size = config.get("buffer_size", 100_000)
        ep = 0
        env_steps = 0
        ret = 0.0
        t = 0
        done = False
        (s1, s2), _ = env.reset()
        self.eps = max(self.eps_end, self.eps * 0.9999)
        while env_steps < steps:
            if done:
                ep += 1
                episode_stats(tracker, t, ep, ret)
                (s1, s2), _ = env.reset()
                ret = 0.0
                t = 0
                done = False
            with torch.no_grad():
                probs = [a(torch.as_tensor(np.stack([s1, s2]), dtype=torch.float32, device=self.device)) for a in self.actors]
            a1 = int(torch.argmax(probs[0][0]).item()) if self.rng.random() > self.eps else int(torch.multinomial(probs[0][0], 1).item())
            a2 = int(torch.argmax(probs[1][1]).item()) if self.rng.random() > self.eps else int(torch.multinomial(probs[1][1], 1).item())
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
        a1_oh = F.one_hot(a1, self.n_actions).to(s1.dtype)
        a2_oh = F.one_hot(a2, self.n_actions).to(s2.dtype)
        a_all = torch.cat([a1_oh, a2_oh], dim=1)
        with torch.no_grad():
            ns_all_rep = ns_all.repeat(self.n_actions * self.n_actions, 1)
            combos = []
            for a1 in range(self.n_actions):
                for a2 in range(self.n_actions):
                    oh = torch.zeros(1, self.n_actions * self.n_actions, device=self.device)
                    oh[0, a1] = 1.0
                    oh[0, self.n_actions + a2] = 1.0
                    combos.append(oh)
            combos = torch.cat(combos, dim=0).repeat(batch_size, 1)
            q_current = self.critics[0](ns_all_rep, combos).view(batch_size, self.n_actions * self.n_actions)
            best = q_current.argmax(dim=1, keepdim=True)
            q_best = self.critics_t[0](ns_all_rep, combos).view(batch_size, self.n_actions * self.n_actions)
            q_next_max = q_best.gather(1, best)
            target = r + self.gamma * (1 - done) * q_next_max
        q1 = self.critics[0](s_all, a_all)
        q2 = self.critics[1](s_all, a_all)
        self.opt_critics[0].zero_grad()
        F.smooth_l1_loss(q1, target).backward()
        self.opt_critics[0].step()
        self.opt_critics[1].zero_grad()
        F.smooth_l1_loss(q2, target).backward()
        self.opt_critics[1].step()
        for i in range(self.n_agents):
            other = 1 - i
            other_oh = a1_oh if other == 0 else a2_oh
            q_all = []
            for a_i in range(self.n_actions):
                a_i_oh = F.one_hot(torch.full((batch_size,), a_i, dtype=torch.long, device=self.device), self.n_actions).to(s1.dtype)
                if i == 0:
                    q_all.append(self.critics[i](s_all, torch.cat([a_i_oh, other_oh], dim=1)))
                else:
                    q_all.append(self.critics[i](s_all, torch.cat([other_oh, a_i_oh], dim=1)))
            q_all = torch.cat(q_all, dim=1)
            greedy = q_all.argmax(dim=1)
            pi = self.actors[i](s1 if i == 0 else s2)
            logp = torch.log(pi.gather(1, greedy.unsqueeze(1)).squeeze(1) + 1e-8)
            loss = -logp.mean()
            self.opt_actors[i].zero_grad()
            loss.backward()
            self.opt_actors[i].step()
        for i in range(self.n_agents):
            for p, pt in zip(self.actors[i].parameters(), self.actors_t[i].parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.critics[i].parameters(), self.critics_t[i].parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def save(self, path):
        torch.save({"actors": [a.state_dict() for a in self.actors],
                    "critics": [c.state_dict() for c in self.critics]}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for a, sd in zip(self.actors, ckpt["actors"]):
            a.load_state_dict(sd)
        for c, sd in zip(self.critics, ckpt["critics"]):
            c.load_state_dict(sd)