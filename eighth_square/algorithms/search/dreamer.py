import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..tabular.common import episode_stats


class RSSM(nn.Module):
    """Recurrent State-Space Model: deterministic GRU state h, stochastic
    latent z, with prior p(z|h), posterior q(z|h,obs), and heads that
    reconstruct obs, reward and the continuation flag."""

    def __init__(self, obs_dim, n_actions, z_dim=16, hidden=64, free_bits=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.n_actions = n_actions
        self.free_bits = free_bits
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.z_mu = nn.Linear(hidden, z_dim)
        self.z_logvar = nn.Linear(hidden, z_dim)
        self.in_proj = nn.Sequential(nn.Linear(z_dim + n_actions, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.prior_mu = nn.Linear(hidden, z_dim)
        self.prior_logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim + hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.obs_head = nn.Linear(hidden, obs_dim)
        self.rew_head = nn.Linear(hidden, 1)
        self.cont_head = nn.Linear(hidden, 1)

    def _sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        return mu + torch.randn_like(std) * std

    def encode(self, obs):
        h = self.enc(obs)
        return self.z_mu(h), self.z_logvar(h)

    def prior(self, h):
        return self.prior_mu(h), self.prior_logvar(h)

    def step(self, h, z, a):
        a_oh = F.one_hot(a, self.n_actions).to(z.dtype)
        h = self.gru(self.in_proj(torch.cat([z, a_oh], dim=1)), h)
        return h

    def heads(self, h, z):
        d = self.dec(torch.cat([z, h], dim=1))
        return self.obs_head(d), self.rew_head(d), self.cont_head(d)


class DreamerAgent(BaseAgent):
    """Dreamer (Hafner et al. 2020): trains an RSSM world model on collected
    episodes (reconstruction + reward + continuation + KL with free bits),
    then learns an actor and a critic by dreaming inside the model."""

    family = "model-based"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = torch.device(config.get("device", "cpu"))
        obs_dim = env.observation_space.shape[0]
        self.n_actions = int(env.action_space.n)
        self.z_dim = config.get("dr_z_dim", 16)
        hidden = config.get("dr_hidden", 64)
        self.rssm = RSSM(obs_dim, self.n_actions, self.z_dim, hidden).to(self.device)
        self.actor = nn.Sequential(nn.Linear(self.z_dim + hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, self.n_actions)).to(self.device)
        self.critic = nn.Sequential(nn.Linear(self.z_dim + hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1)).to(self.device)
        self.opt_rssm = torch.optim.Adam(self.rssm.parameters(), lr=config.get("lr", 3e-4))
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=config.get("lr", 3e-4))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=config.get("lr", 3e-4))
        self.gamma = config.get("gamma", 0.99)
        self.lamb = config.get("dr_lambda", 0.95)

    def _collect_data(self, env, n_episodes):
        episodes = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            ep = []
            while not done:
                a = int(self.rng.integers(self.n_actions))
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                ep.append((np.asarray(state, dtype=np.float32), a, float(r), done,
                           np.asarray(ns, dtype=np.float32)))
                state = ns
                if len(ep) >= env.spec.max_episode_steps:
                    break
            episodes.append(ep)
        return episodes

    def _train_rssm(self, episodes, steps):
        seqs = []
        for ep in episodes:
            if len(ep) < 3:
                continue
            obs = np.stack([e[0] for e in ep])
            obs_next = np.stack([e[4] for e in ep])
            acts = np.array([e[1] for e in ep], dtype=np.int64)
            rews = np.array([e[2] for e in ep], dtype=np.float32)
            conts = np.array([1.0 if not e[3] else 0.0 for e in ep], dtype=np.float32)
            seqs.append((obs, obs_next, acts, rews, conts))
        for _ in range(steps):
            seq = seqs[self.rng.integers(0, len(seqs))]
            obs, obs_next, acts, rews, conts = seq
            if len(obs) > 20:
                i = self.rng.integers(0, len(obs) - 19)
                obs = obs[i:i + 20]
                obs_next = obs_next[i:i + 20]
                acts = acts[i:i + 20]
                rews = rews[i:i + 20]
                conts = conts[i:i + 20]
            obs_t = torch.as_tensor(obs, device=self.device)
            obs_n = torch.as_tensor(obs_next, device=self.device)
            act_t = torch.as_tensor(acts, device=self.device)
            rew_t = torch.as_tensor(rews, device=self.device).unsqueeze(1)
            cont_t = torch.as_tensor(conts, device=self.device).unsqueeze(1)
            h = torch.zeros(1, self.rssm.gru.hidden_size, device=self.device)
            z = torch.zeros(1, self.z_dim, device=self.device)
            tot = torch.zeros((), device=self.device)
            for t in range(obs_t.shape[0]):
                h = self.rssm.step(h, z, act_t[t].unsqueeze(0))
                pmu, plogvar = self.rssm.prior(h)
                qmu, qlogvar = self.rssm.encode(obs_n[t].unsqueeze(0))
                z = self.rssm._sample(qmu, qlogvar)
                o_hat, r_hat, c_hat = self.rssm.heads(h, z)
                qvar = qlogvar.clamp(-10, 10).exp()
                pvar = plogvar.clamp(-10, 10).exp()
                kl = 0.5 * (plogvar.clamp(-10, 10) - qlogvar.clamp(-10, 10)
                            + (qvar + (qmu - pmu).pow(2)) / (pvar + 1e-6) - 1).sum(1)
                kl = kl.clamp(min=self.rssm.free_bits * self.z_dim)
                tot = (tot + F.mse_loss(o_hat, obs_n[t].unsqueeze(0))
                       + F.mse_loss(r_hat, rew_t[t].unsqueeze(0))
                       + F.binary_cross_entropy_with_logits(c_hat, cont_t[t].unsqueeze(0),
                                                            pos_weight=torch.tensor(self.config.get("dr_cont_weight", 5.0), device=self.device))
                       + 0.1 * kl.sum())
            self.opt_rssm.zero_grad()
            (tot / obs_t.shape[0]).backward()
            self.opt_rssm.step()

    def _imagination(self, episodes, n_steps, L=15):
        starts = []
        for ep in episodes:
            obs = np.stack([e[0] for e in ep])
            starts.append(obs)
        for _ in range(n_steps):
            obs = starts[self.rng.integers(0, len(starts))]
            i = self.rng.integers(0, len(obs))
            with torch.no_grad():
                qmu, qlogvar = self.rssm.encode(torch.as_tensor(obs[i], device=self.device).unsqueeze(0))
                z = self.rssm._sample(qmu, qlogvar)
            h = torch.zeros(1, self.rssm.gru.hidden_size, device=self.device)
            states = []
            actions = []
            rewards = []
            conts = []
            for _ in range(L):
                s = torch.cat([z, h], dim=1)
                logits = self.actor(s)
                probs = F.softmax(logits, dim=1)
                a = torch.multinomial(probs, 1)
                states.append(s)
                actions.append(a)
                h2 = self.rssm.step(h, z, a.squeeze(0))
                pmu, plogvar = self.rssm.prior(h2)
                z = self.rssm._sample(pmu, plogvar)
                with torch.no_grad():
                    _, r_hat, c_hat = self.rssm.heads(h2, z)
                rewards.append(torch.sigmoid(r_hat))
                conts.append(torch.sigmoid(c_hat))
                h = h2
            S = torch.cat(states, dim=0)
            A = torch.cat(actions, dim=0)
            R = torch.cat(rewards, dim=0)
            C = torch.cat(conts, dim=0)
            with torch.no_grad():
                V = self.critic(S)
            V = torch.cat([V, torch.zeros(1, 1, device=self.device)], dim=0)
            R = torch.cat([R, torch.zeros(1, 1, device=self.device)], dim=0)
            C = torch.cat([C, torch.zeros(1, 1, device=self.device)], dim=0)
            targets = torch.zeros_like(R)
            acc = V[-1]
            for t in reversed(range(L)):
                acc = R[t] + self.gamma * C[t] * ((1 - self.lamb) * V[t + 1] + self.lamb * acc)
                targets[t] = acc
            targets = targets[:L]
            probs = F.softmax(self.actor(S), dim=1)
            log_probs = torch.log(probs.gather(1, A) + 1e-8)
            adv = (targets - V[:L]).detach()
            self.opt_actor.zero_grad()
            actor_loss = -(log_probs * adv).mean() - 0.01 * torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            actor_loss.backward()
            self.opt_actor.step()
            self.opt_critic.zero_grad()
            critic_loss = F.mse_loss(self.critic(S.detach()), targets)
            critic_loss.backward()
            self.opt_critic.step()

    def train(self, env, config, tracker):
        episodes = self._collect_data(env, config.get("dr_data_episodes", 60))
        self._train_rssm(episodes, config.get("dr_rssm_steps", 3000))
        self._imagination(episodes, config.get("dr_dream_steps", 3000),
                          L=config.get("dr_horizon", 15))
        ep = 0
        for _ in range(config.get("eval_episodes", 10)):
            state, _ = env.reset()
            done = False
            ret = 0.0
            t = 0
            while not done:
                a = self.act(state)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                state = ns
                ret += r
                t += 1
                if t >= env.spec.max_episode_steps:
                    break
            ep += 1
            episode_stats(tracker, t, ep, ret)

    def act(self, state, eval=True):
        with torch.no_grad():
            qmu, qlogvar = self.rssm.encode(torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0))
            z = self.rssm._sample(qmu, qlogvar)
            h = torch.zeros(1, self.rssm.gru.hidden_size, device=self.device)
            logits = self.actor(torch.cat([z, h], dim=1))
            return int(torch.argmax(logits, dim=1).item())

    def save(self, path):
        torch.save({"rssm": self.rssm.state_dict(), "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.rssm.load_state_dict(ckpt["rssm"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])