import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers=2):
        super().__init__()
        blocks = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.ReLU()]
        blocks += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class DiscretePolicy(nn.Module):
    def __init__(self, in_dim, hidden, n_actions, layers=2):
        super().__init__()
        self.net = MLP(in_dim, hidden, n_actions, layers)

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class GaussianPolicy(nn.Module):
    def __init__(self, in_dim, hidden, action_dim, log_std=0.0, layers=2):
        super().__init__()
        self.net = MLP(in_dim, hidden, action_dim, layers)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std))

    def forward(self, x):
        mean = self.net(x)
        std = self.log_std.exp().clamp(min=1e-4, max=10.0)
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        u = dist.sample()
        a = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_prob.sum(-1, keepdim=True)

    def log_prob(self, x, a):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        a_raw = torch.clamp(0.5 * torch.log((1 + a) / (1 - a + 1e-6)), min=-20, max=20)
        log_prob = dist.log_prob(a_raw) - torch.log(1 - a.pow(2) + 1e-6)
        return log_prob.sum(-1, keepdim=True)

    def mean_action(self, x):
        mean, _ = self.forward(x)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden, layers=2):
        super().__init__()
        self.net = MLP(in_dim, hidden, 1, layers)

    def forward(self, x):
        return self.net(x)


def kl_discrete(p_old_probs, logits_new):
    log_p_new = F.log_softmax(logits_new, dim=-1)
    return (p_old_probs * (p_old_probs.clamp(min=1e-8).log() - log_p_new)).sum(-1).mean()


def kl_gaussian(mean_old, std_old, mean_new, std_new):
    return (
        (
            torch.log(std_new / std_old)
            + (std_old.pow(2) + (mean_old - mean_new).pow(2)) / (2 * std_new.pow(2))
            - 0.5
        )
        .sum(-1)
        .mean()
    )


def flat_params(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def set_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p))
        idx += n
