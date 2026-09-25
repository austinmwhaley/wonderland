import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        mu_range = 1.0 / np.sqrt(in_features)
        self.mu_w = nn.Parameter(
            torch.FloatTensor(out_features, in_features).uniform_(-mu_range, mu_range)
        )
        self.sigma_w = nn.Parameter(
            torch.full((out_features, in_features), sigma_init / np.sqrt(in_features))
        )
        self.mu_b = nn.Parameter(torch.FloatTensor(out_features).uniform_(-mu_range, mu_range))
        self.sigma_b = nn.Parameter(torch.full((out_features,), sigma_init / np.sqrt(out_features)))

        self.register_buffer("eps_w", torch.zeros(out_features, in_features))
        self.register_buffer("eps_b", torch.zeros(out_features))
        self.sample_noise()

    def sample_noise(self):
        eps_in = self._f(torch.randn(self.in_features))
        eps_out = self._f(torch.randn(self.out_features))
        self.eps_w.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.eps_b.copy_(eps_out)

    @staticmethod
    def _f(x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def forward(self, x):
        weight = self.mu_w + self.sigma_w * self.eps_w.to(x.device)
        bias = self.mu_b + self.sigma_b * self.eps_b.to(x.device)
        return F.linear(x, weight, bias)


class NoisyDuelingMLP(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dims: tuple = (128, 128)):
        super().__init__()
        dims = (input_dim,) + hidden_dims

        shared = []
        for i in range(len(dims) - 1):
            shared.append(layer_init(nn.Linear(dims[i], dims[i + 1])))
            shared.append(nn.ReLU())
        self.shared = nn.Sequential(*shared)
        self.shared_noisy = NoisyLinear(hidden_dims[-1], hidden_dims[-1])

        self.value_stream = nn.Sequential(
            layer_init(nn.Linear(hidden_dims[-1], hidden_dims[-1])),
            nn.ReLU(),
        )
        self.value_noisy = NoisyLinear(hidden_dims[-1], 1)

        self.advantage_stream = nn.Sequential(
            layer_init(nn.Linear(hidden_dims[-1], hidden_dims[-1])),
            nn.ReLU(),
        )
        self.advantage_noisy = NoisyLinear(hidden_dims[-1], n_actions)

    def forward(self, x):
        features = self.shared(x)
        features = F.relu(self.shared_noisy(features))
        value = self.value_noisy(self.value_stream(features))
        advantage = self.advantage_noisy(self.advantage_stream(features))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def sample_noise(self):
        self.shared_noisy.sample_noise()
        self.value_noisy.sample_noise()
        self.advantage_noisy.sample_noise()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple = (128, 128),
        activation=nn.ReLU,
        final_activation=None,
    ):
        super().__init__()
        layers = []
        dims = (input_dim,) + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(layer_init(nn.Linear(dims[i], dims[i + 1])))
            layers.append(activation())
        layers.append(layer_init(nn.Linear(dims[-1], output_dim), std=1.0))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DuelingMLP(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dims: tuple = (128, 128)):
        super().__init__()
        dims = (input_dim,) + hidden_dims

        shared = []
        for i in range(len(dims) - 1):
            shared.append(layer_init(nn.Linear(dims[i], dims[i + 1])))
            shared.append(nn.ReLU())
        self.shared = nn.Sequential(*shared)

        self.value_stream = nn.Sequential(
            layer_init(nn.Linear(hidden_dims[-1], hidden_dims[-1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dims[-1], 1), std=1.0),
        )
        self.advantage_stream = nn.Sequential(
            layer_init(nn.Linear(hidden_dims[-1], hidden_dims[-1])),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dims[-1], n_actions), std=1.0),
        )

    def forward(self, x):
        features = self.shared(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: tuple = (256, 256),
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        dims = (input_dim,) + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(layer_init(nn.Linear(dims[i], dims[i + 1])))
            layers.append(nn.ReLU())
        self.shared = nn.Sequential(*layers)

        self.mean = layer_init(nn.Linear(hidden_dims[-1], action_dim), std=0.01)
        self.log_std = layer_init(nn.Linear(hidden_dims[-1], action_dim), std=0.01)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        features = self.shared(x)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, mean


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: tuple = (256, 256),
        action_scale: float = 1.0,
    ):
        super().__init__()
        dims = (input_dim,) + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(layer_init(nn.Linear(dims[i], dims[i + 1])))
            layers.append(nn.ReLU())
        layers.append(layer_init(nn.Linear(hidden_dims[-1], action_dim), std=0.01))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.action_scale = action_scale

    def forward(self, x):
        return self.net(x) * self.action_scale


class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dims: tuple = (128, 128)):
        super().__init__()
        self.net = MLP(input_dim, n_actions, hidden_dims)

    def forward(self, x):
        return self.net(x)

    def sample(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(1)
        return action, log_prob, dist.entropy().unsqueeze(1)

    def evaluate(self, x, action):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action).unsqueeze(1)
        entropy = dist.entropy().unsqueeze(1)
        return log_prob, entropy


class QNetwork(MLP):
    def __init__(self, input_dim, n_actions, hidden_dims=(128, 128)):
        super().__init__(input_dim, n_actions, hidden_dims)


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple = (256, 256)):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dims)
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dims)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state, action):
        return self.q1(torch.cat([state, action], dim=1))
