"""Continuous-action Implicit Q-Learning (IQL) for offline RL.

The discrete IQL in this package uses a categorical policy + per-action Q. For
continuous control we mirror it exactly with a tanh-Gaussian policy and a
Q(s,a) critic:

  - Q(s,a) and V(s) fit by expectile regression (never the max -> no OOD
    inflation), with a Polyak target V.
  - Policy extraction is advantage-weighted behavioral cloning (AWR-style)
    of the logged actions under the tanh-Gaussian.

Actions are handled in normalized [-1,1] space internally and mapped to the
env's [low, high] at the boundary, so density ratios in OPE are computed in a
single consistent space (log_prob_fn returns the ENV-space density).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.approx.networks import MLP
from algorithms.deep.networks import ContinuousCritic


class _GaussianMLP(nn.Module):
    """Diagonal Gaussian policy (no squashing): logged actions are arbitrary
    reals, so a plain Normal head is the correct density model for AWR.
    Bounded envs are handled by clipping at the boundary when acting."""

    def __init__(self, in_dim, hidden, a_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, a_dim)
        self.log_std = nn.Parameter(torch.zeros(a_dim))

    def forward(self, x):
        h = self.body(x)
        std = self.log_std.exp().clamp(min=1e-4, max=10.0)
        return self.mu(h), std

    def mean_action(self, x):
        return self.mu(self.body(x))

    def log_prob(self, x, a):
        mu, std = self.forward(x)
        return torch.distributions.Normal(mu, std).log_prob(a).sum(-1, keepdim=True)

    def sample(self, x):
        mu, std = self.forward(x)
        return mu + std * torch.randn_like(mu)


class ContinuousIQL:
    action_space = "continuous"
    state_space = "continuous"

    def __init__(self, diet, config=None):
        cfg = dict(config or {})
        self.device = cfg.get("device", "cpu")
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lr = float(cfg.get("lr", 3e-4))
        self.hidden = int(cfg.get("hidden", 128))
        self.batch = int(cfg.get("batch_size", 256))
        self.tau = float(cfg.get("expectile", 0.7))
        self.beta = float(cfg.get("beta", 3.0))
        self.v_tau = float(cfg.get("v_tau", 0.005))
        self.v_clip = float(cfg.get("v_clip", 1000.0))
        self.seed = int(cfg.get("seed", 0))
        self.a_dim = int(diet["act"].shape[1])
        low = cfg.get("action_low")
        high = cfg.get("action_high")
        if low is None or high is None:
            low, high = -np.ones(self.a_dim), np.ones(self.a_dim)
        self.low = np.asarray(low, dtype=np.float64).reshape(-1)
        self.high = np.asarray(high, dtype=np.float64).reshape(-1)
        self.span = np.maximum(self.high - self.low, 1e-8)
        d = int(diet["obs"].shape[1])
        self.Q = ContinuousCritic(d, self.a_dim, self.hidden).to(self.device)
        self.V = MLP(d, self.hidden, 1).to(self.device)
        self.Vt = MLP(d, self.hidden, 1).to(self.device)
        self.Vt.load_state_dict(self.V.state_dict())
        self.pi = _GaussianMLP(d, self.hidden, self.a_dim).to(self.device)
        self.q_opt = torch.optim.Adam(list(self.Q.parameters()) +
                                      list(self.V.parameters()), lr=self.lr)
        self.p_opt = torch.optim.Adam(self.pi.parameters(), lr=self.lr)
        self.O = torch.as_tensor(diet["obs"]).float().to(self.device)
        self.A = torch.as_tensor(self._norm(diet["act"])).float().to(self.device)
        self.R = torch.as_tensor(diet["rew"]).float().unsqueeze(1).to(self.device)
        self.O2 = torch.as_tensor(diet["obs2"]).float().to(self.device)
        self.D = torch.as_tensor(diet["done"]).float().unsqueeze(1).to(self.device)
        self.N = len(diet["obs"])

    # ---- action space mapping ----
    def _norm(self, a):
        a = np.asarray(a, dtype=np.float64)
        return 2.0 * (a - self.low) / self.span - 1.0

    def _denorm(self, a_norm):
        return self.low + (np.asarray(a_norm) + 1.0) / 2.0 * self.span

    def fit(self, steps, log_every=0, rng_seed=None):
        rng = np.random.default_rng(self.seed if rng_seed is None else rng_seed)
        losses = []
        for s in range(int(steps)):
            i = torch.as_tensor(rng.integers(0, self.N, self.batch)).to(self.device)
            q = self.Q(self.O[i], self.A[i])
            v = self.V(self.O[i])
            with torch.no_grad():
                tgt = self.R[i] + self.gamma * (1 - self.D[i]) * \
                    self.Vt(self.O2[i]).clamp(-self.v_clip, self.v_clip)
            diff = q.detach() - v
            w = torch.where(diff > 0, torch.full_like(diff, self.tau),
                            torch.full_like(diff, 1.0 - self.tau))
            l_q = F.smooth_l1_loss(q, tgt)
            l_v = (w * diff ** 2).mean()
            self.q_opt.zero_grad()
            (l_q + l_v).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.Q.parameters()) + list(self.V.parameters()), 5.0)
            self.q_opt.step()
            with torch.no_grad():
                for ps, pt in zip(self.V.parameters(), self.Vt.parameters()):
                    pt.data.mul_(1 - self.v_tau).add_(ps.data, alpha=self.v_tau)
            with torch.no_grad():
                adv = q.detach() - self.V(self.O[i]).detach()
                scale = adv.abs().mean().clamp(min=1e-6)
                wp = torch.exp(self.beta * (adv / scale).clamp(-1.0, 1.0))
                wp = wp / wp.mean().clamp(min=1e-8)
            logp = self.pi.log_prob(self.O[i], self.A[i])
            l_p = -(wp * logp).mean()
            self.p_opt.zero_grad()
            l_p.backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            self.p_opt.step()
            losses.append(float(l_q.item()))
            if log_every and (s + 1) % log_every == 0:
                print(f"  iql_cont step {s+1}/{steps} q={np.mean(losses[-log_every:]):.2f}",
                      flush=True)
        return self

    # ---- candidate protocol (continuous) ----
    def action_mean(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                device=self.device)
            a_norm = self.pi.mean_action(x).cpu().numpy()
        return np.clip(self._denorm(a_norm), self.low, self.high)

    def act(self, state, eval=True):
        obs = np.asarray(state, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        return self.action_mean(obs)[0]

    def sample(self, obs, rng):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                device=self.device)
            a_norm = self.pi.sample(x)
            a_norm = a_norm.cpu().numpy()
        a = self._denorm(a_norm)
        return np.clip(a, self.low, self.high)

    def log_prob_fn(self, obs, act):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                device=self.device)
            a_norm = torch.as_tensor(self._norm(act), dtype=torch.float32,
                                     device=self.device)
            lp = self.pi.log_prob(x, a_norm).cpu().numpy().ravel()
        # change of variables: env action = low + span/2 * (a_norm + 1)
        lp = lp - float(np.sum(np.log(self.span / 2.0)))
        return lp

    def save(self, path):
        torch.save({"Q": self.Q.state_dict(), "V": self.V.state_dict(),
                    "Vt": self.Vt.state_dict(), "pi": self.pi.state_dict(),
                    "low": self.low, "high": self.high}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.Q.load_state_dict(d["Q"]); self.V.load_state_dict(d["V"])
        self.Vt.load_state_dict(d["Vt"]); self.pi.load_state_dict(d["pi"])
        return self
