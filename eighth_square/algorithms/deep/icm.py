import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dqn import DQN


class ICMNet(nn.Module):
    """Intrinsic Curiosity Module: feature encoder, forward model
    (s, a) -> phi(s'), and inverse model (phi(s), phi(s')) -> a."""

    def __init__(self, in_dim, n_actions, hidden=128, feat=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feat + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat),
        )
        self.inverse = nn.Sequential(
            nn.Linear(2 * feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward_error(self, phi_s, a_onehot, phi_s2):
        pred = self.forward_model(torch.cat([phi_s, a_onehot], dim=-1))
        return (pred - phi_s2) ** 2


class ICM(DQN):
    """DQN with curiosity: intrinsic reward = prediction error of a learned
    forward model, trained alongside the inverse model (Pathak et al. 2017)."""

    def __init__(self, env, config):
        super().__init__(env, config)
        in_dim = int(
            getattr(env.observation_space, "n", None) or np.prod(env.observation_space.shape)
        )
        self.beta = config.get("beta", 0.2)
        feat = config.get("icm_feat", 64)
        self.icm = ICMNet(in_dim, self.nA, config.get("hidden", 128), feat).to(self.device)
        self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=config.get("icm_lr", 1e-3))
        self.r_mean, self.r_std = 0.0, 1.0
        self.icm_count = 0

    def _onehot(self, a):
        return F.one_hot(a, self.nA).float()

    def _intrinsic(self, s, a, s2):
        self.icm.eval()
        with torch.no_grad():
            phi_s = self.icm.encode(self._t(s))
            phi_s2 = self.icm.encode(self._t(s2))
            if phi_s.dim() == 1:
                phi_s = phi_s.unsqueeze(0)
                phi_s2 = phi_s2.unsqueeze(0)
            err = self.icm.forward_error(phi_s, self._onehot(torch.tensor([a])), phi_s2).mean()
        return float(err)

    def _normalize(self, ri):
        n = self.icm_count
        self.r_mean += (ri - self.r_mean) / (n + 1)
        self.r_std += ((ri - self.r_mean) ** 2 - self.r_std) / (n + 1)
        self.icm_count += 1
        return self.beta * ri / (np.sqrt(self.r_std) + 1e-6)

    def push(self, s, a, r, s2, done):
        ri = self._normalize(self._intrinsic(s, a, s2))
        super().push(s, a, r + ri, s2, done)

    def _update(self):
        loss = super()._update()
        sample = self.buffer.sample(self.batch_size)
        obs, act, obs2 = sample[0], sample[1], sample[3]
        self.icm.train()
        phi_s = self.icm.encode(obs)
        phi_s2 = self.icm.encode(obs2).detach()
        pred = self.icm.forward_model(torch.cat([phi_s, self._onehot(act)], dim=-1))
        fwd_loss = F.mse_loss(pred, phi_s2)
        inv_logits = self.icm.inverse(torch.cat([phi_s, self.icm.encode(obs2)], dim=-1))
        inv_loss = F.cross_entropy(inv_logits, act)
        total = fwd_loss + inv_loss
        self.icm_opt.zero_grad()
        total.backward()
        self.icm_opt.step()
        return loss

    def save(self, path):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "icm": self.icm.state_dict(),
            },
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])
        self.icm.load_state_dict(data["icm"])
