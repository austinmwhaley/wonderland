import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dqn import DQN


class RNDNet(nn.Module):
    """Two identical MLPs; the target is frozen at init, the predictor is
    trained to match it.  Prediction error = novelty = intrinsic reward."""

    def __init__(self, in_dim, hidden=128, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feat),
        )

    def forward(self, x):
        return self.net(x)


class RND(DQN):
    """DQN with Random Network Distillation: a frozen random target network
    and a trained predictor; intrinsic reward is their disagreement
    (Burda et al. 2018)."""

    def __init__(self, env, config):
        super().__init__(env, config)
        in_dim = int(getattr(env.observation_space, "n", None) or np.prod(env.observation_space.shape))
        self.beta = config.get("beta", 1.0)
        feat = config.get("rnd_feat", 64)
        self.rnd_target = RNDNet(in_dim, config.get("hidden", 128), feat).to(self.device)
        self.rnd_predictor = RNDNet(in_dim, config.get("hidden", 128), feat).to(self.device)
        for p in self.rnd_target.parameters():
            p.requires_grad_(False)
        self.rnd_opt = torch.optim.Adam(self.rnd_predictor.parameters(), lr=config.get("rnd_lr", 1e-3))
        self.r_mean, self.r_var = 0.0, 1.0
        self.rnd_count = 0

    def _intrinsic(self, s2):
        self.rnd_predictor.eval()
        with torch.no_grad():
            pred = self.rnd_predictor(self._t(s2))
            tgt = self.rnd_target(self._t(s2))
        return float(((pred - tgt) ** 2).mean())

    def _normalize(self, ri):
        n = self.rnd_count
        self.r_mean += (ri - self.r_mean) / (n + 1)
        self.r_var += ((ri - self.r_mean) ** 2 - self.r_var) / (n + 1)
        self.rnd_count += 1
        return self.beta * ri / (np.sqrt(self.r_var) + 1e-6)

    def push(self, s, a, r, s2, done):
        ri = self._normalize(self._intrinsic(s2))
        super().push(s, a, r + ri, s2, done)

    def _update(self):
        loss = super()._update()
        sample = self.buffer.sample(self.batch_size)
        obs2 = sample[3]
        self.rnd_predictor.train()
        loss_rnd = F.mse_loss(self.rnd_predictor(obs2), self.rnd_target(obs2).detach())
        self.rnd_opt.zero_grad()
        loss_rnd.backward()
        self.rnd_opt.step()
        return loss

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "rnd_target": self.rnd_target.state_dict(),
                    "rnd_predictor": self.rnd_predictor.state_dict()}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data["target"])
        self.rnd_target.load_state_dict(data["rnd_target"])
        self.rnd_predictor.load_state_dict(data["rnd_predictor"])