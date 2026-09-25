import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..approx.networks import GaussianPolicy


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(
            torch.full((out_features, in_features), sigma0 / math.sqrt(in_features))
        )
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.full((out_features,), sigma0 / math.sqrt(in_features)))
        self.reset()

    def reset(self):
        nn.init.uniform_(
            self.mu_w, -1 / math.sqrt(self.mu_w.size(1)), 1 / math.sqrt(self.mu_w.size(1))
        )
        nn.init.uniform_(
            self.mu_b, -1 / math.sqrt(self.mu_w.size(1)), 1 / math.sqrt(self.mu_w.size(1))
        )

    def forward(self, x):
        if self.training:
            eps_w = torch.randn_like(self.sigma_w)
            eps_b = torch.randn_like(self.sigma_b)
        else:
            eps_w, eps_b = 0.0, 0.0
        w = self.mu_w + self.sigma_w * eps_w
        b = self.mu_b + self.sigma_b * eps_b
        return F.linear(x, w, b)


class QNetwork(nn.Module):
    def __init__(self, in_dim, hidden, n_actions, dueling=False, noisy=False, out_heads=1):
        super().__init__()
        self.dueling = dueling
        self.n_actions = n_actions
        self.out_heads = out_heads
        Lin = NoisyLinear if noisy else nn.Linear
        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        if dueling:
            self.value = nn.Sequential(Lin(hidden, hidden), nn.ReLU(), Lin(hidden, out_heads))
            self.advantage = nn.Sequential(
                Lin(hidden, hidden), nn.ReLU(), Lin(hidden, out_heads * n_actions)
            )
        else:
            self.head = Lin(hidden, out_heads * n_actions)

    def forward(self, x):
        h = self.features(x)
        if self.dueling:
            v = self.value(h)
            a = self.advantage(h)
            if self.out_heads == 1:
                return v + a - a.mean(dim=-1, keepdim=True)
            a3 = a.view(x.size(0), self.out_heads, self.n_actions)
            return v.unsqueeze(2) + a3 - a3.mean(dim=-1, keepdim=True)
        return (
            self.head(h).view(x.size(0), self.out_heads, self.n_actions)
            if self.out_heads > 1
            else self.head(h)
        )


class DeterministicActor(nn.Module):
    def __init__(self, in_dim, hidden, action_dim, bound=1.0):
        super().__init__()
        self.bound = bound
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x) * self.bound


class ContinuousCritic(nn.Module):
    def __init__(self, in_dim, action_dim, hidden, layers=2):
        super().__init__()
        blocks = [nn.Linear(in_dim + action_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.ReLU()]
        blocks += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))


class SACPolicy(GaussianPolicy):
    pass


def polyak_copy(source, target, tau):
    for ps, pt in zip(source.parameters(), target.parameters()):
        pt.data.mul_(1 - tau).add_(ps.data, alpha=tau)
