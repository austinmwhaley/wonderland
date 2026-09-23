import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..tabular.common import episode_stats


class VAE(nn.Module):
    def __init__(self, obs_dim, z_dim=8, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, obs_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def loss(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.decode(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return F.mse_loss(recon, x) + 0.1 * kl


class MDNRNN(nn.Module):
    def __init__(self, z_dim, n_actions, n_mix=5, hidden=64):
        super().__init__()
        self.z_dim = z_dim
        self.n_actions = n_actions
        self.n_mix = n_mix
        self.rnn = nn.GRUCell(hidden, hidden)
        self.in_proj = nn.Linear(z_dim + n_actions, hidden)
        self.z_out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, n_mix * (2 * z_dim + 1)))

    def forward(self, z, a, h):
        a_oh = F.one_hot(a, self.n_actions).to(z.dtype)
        h = self.rnn(self.in_proj(torch.cat([z, a_oh], dim=1)), h)
        out = self.z_out(h)
        logits = out[:, :self.n_mix]
        mus = out[:, self.n_mix:self.n_mix * (self.z_dim + 1)].view(-1, self.n_mix, self.z_dim)
        logvars = out[:, self.n_mix * (self.z_dim + 1):].view(-1, self.n_mix, self.z_dim)
        return h, logits, mus, logvars

    def loss(self, z, a, z_next, h):
        h, logits, mus, logvars = self.forward(z, a, h)
        pi = F.softmax(logits, dim=1)
        logvar = logvars.clamp(-10, 10)
        diff = z_next.unsqueeze(1) - mus
        log_gauss = (-0.5 * ((diff.pow(2) / logvar.exp()).sum(2) + (logvar.sum(2) + self.z_dim * np.log(2 * np.pi))))
        log_mix = torch.logsumexp(torch.log(pi.clamp_min(1e-8)) + log_gauss, dim=1)
        return -log_mix.mean(), h

    def sample_next(self, z, a, h, rng, temperature=1.0):
        with torch.no_grad():
            h, logits, mus, logvars = self.forward(z.unsqueeze(0), a, h)
            logits = logits / temperature
            pi = F.softmax(logits, dim=1)
            comp = torch.multinomial(pi, 1).item()
            mu = mus[0, comp]
            std = torch.exp(0.5 * logvars[0, comp])
            return mu + torch.randn_like(std) * std, h


class WorldModelsAgent(BaseAgent):
    """World Models (Ha & Schmidhuber 2018): a VAE compresses observations to
    a latent code, an MDN-RNN learns the latent dynamics, and a linear
    controller is trained by CEM in the imagined environment."""

    family = "model-based"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.device = torch.device(config.get("device", "cpu"))
        obs_dim = env.observation_space.shape[0]
        self.n_actions = int(env.action_space.n)
        self.z_dim = config.get("wm_z_dim", 8)
        self.vae = VAE(obs_dim, self.z_dim).to(self.device)
        self.mdn = MDNRNN(self.z_dim, self.n_actions, config.get("wm_n_mix", 5)).to(self.device)
        self.opt_vae = torch.optim.Adam(self.vae.parameters(), lr=config.get("lr", 1e-3))
        self.opt_mdn = torch.optim.Adam(self.mdn.parameters(), lr=config.get("lr", 1e-3))
        self.dataset = None
        self.controller = None

    def _encode(self, obs):
        with torch.no_grad():
            mu, logvar = self.vae.encode(obs)
            return mu

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
                ep.append((np.asarray(state, dtype=np.float32), a, float(r), done))
                state = ns
                if len(ep) >= env.spec.max_episode_steps:
                    break
            episodes.append(ep)
        return episodes

    def _train_vae(self, obs_all):
        n = len(obs_all)
        for step in range(2000):
            idx = self.rng.integers(0, n, size=256)
            x = torch.as_tensor(obs_all[idx], device=self.device)
            self.opt_vae.zero_grad()
            loss = self.vae.loss(x)
            loss.backward()
            self.opt_vae.step()

    def _train_mdn(self, episodes):
        seqs = []
        for ep in episodes:
            if len(ep) < 2:
                continue
            obs = np.stack([e[0] for e in ep])
            z = self._encode(torch.as_tensor(obs, device=self.device)).cpu().numpy()
            seqs.append((z, [e[1] for e in ep[:-1]], z[1:]))
        all_loss = 0.0
        for step in range(4000):
            ep = seqs[self.rng.integers(0, len(seqs))]
            z, acts, z_next = ep
            if len(z) > 30:
                i = self.rng.integers(0, len(z) - 29)
                z, acts, z_next = z[i:i + 30], acts[i:i + 30], z_next[i:i + 30]
            z_t = torch.as_tensor(z[:-1], device=self.device)
            a_t = torch.as_tensor(np.array(acts), device=self.device)
            zn = torch.as_tensor(z_next, device=self.device)
            loss = torch.zeros((), device=self.device)
            h = torch.zeros(1, 64, device=self.device)
            for t in range(len(z_t)):
                l, h = self.mdn.loss(z_t[t].unsqueeze(0), a_t[t].unsqueeze(0),
                                     zn[t].unsqueeze(0), h)
                loss = loss + l
            self.opt_mdn.zero_grad()
            (loss / len(z_t)).backward()
            self.opt_mdn.step()
            all_loss += float(loss) / len(z_t)
        return all_loss / 4000

    def _cem(self, n_samples=64, n_iter=12, horizon=40, elite_frac=0.2):
        start_states = [ep[0][0] for ep in self.dataset]
        mean = np.zeros(self.z_dim * self.n_actions)
        cov = np.eye(self.z_dim * self.n_actions) * 1.0
        for it in range(n_iter):
            weights = self.rng.multivariate_normal(mean, cov, size=n_samples)
            scores = np.zeros(n_samples)
            for i in range(n_samples):
                W = weights[i].reshape(self.z_dim, self.n_actions)
                for _ in range(3):
                    s0 = start_states[self.rng.integers(0, len(start_states))]
                    z = self._encode(torch.as_tensor(s0, dtype=torch.float32, device=self.device))
                    h = torch.zeros(1, 64, device=self.device)
                    ret = 0.0
                    g = 1.0
                    for _ in range(horizon):
                        logits = torch.as_tensor(z @ torch.as_tensor(W, dtype=z.dtype), device=z.device)
                        a = int(torch.argmax(logits).item())
                        z, h = self.mdn.sample_next(z, torch.as_tensor([a], device=self.device),
                                                    h, self.rng)
                        ret += g * 1.0
                        g *= 0.99
                        done = float(torch.abs(z[2]).item()) > 0.5
                        if done:
                            break
                    scores[i] = ret
            elite = np.argsort(scores)[-int(n_samples * elite_frac):]
            mean = weights[elite].mean(axis=0)
            cov = np.cov(weights[elite], rowvar=False) + 1e-3 * np.eye(self.z_dim * self.n_actions)
        return mean.reshape(self.z_dim, self.n_actions)

    def train(self, env, config, tracker):
        n_episodes = config.get("wm_data_episodes", 100)
        episodes = self._collect_data(env, n_episodes)
        self.dataset = episodes
        obs_all = np.stack([e[0] for ep in episodes for e in ep])
        self._train_vae(obs_all)
        avg_loss = self._train_mdn(episodes)
        W = self._cem(n_samples=config.get("wm_cem_samples", 64),
                      n_iter=config.get("wm_cem_iter", 12),
                      horizon=config.get("wm_horizon", 40))
        self.controller = torch.as_tensor(W, dtype=torch.float32, device=self.device)
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
        z = self._encode(torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device))
        logits = z @ self.controller
        return int(torch.argmax(logits).item())

    def save(self, path):
        torch.save({"vae": self.vae.state_dict(), "mdn": self.mdn.state_dict(),
                    "controller": self.controller.cpu()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(ckpt["vae"])
        self.mdn.load_state_dict(ckpt["mdn"])
        self.controller = ckpt["controller"].to(self.device)