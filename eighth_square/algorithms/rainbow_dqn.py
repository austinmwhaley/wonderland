import copy
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.replay_buffer import PrioritizedReplayBuffer
from core.trainer import PlateauTracker
from environments.base import EnvWrapper


def _ortho_init(layer, std=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        mu_range = 1.0 / np.sqrt(in_features)
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features).uniform_(-mu_range, mu_range))
        self.sigma_w = nn.Parameter(torch.full((out_features, in_features), sigma / np.sqrt(in_features)))
        self.mu_b = nn.Parameter(torch.empty(out_features).uniform_(-mu_range, mu_range))
        self.sigma_b = nn.Parameter(torch.full((out_features,), sigma / np.sqrt(out_features)))
        self.register_buffer("eps_w", torch.zeros(out_features, in_features))
        self.register_buffer("eps_b", torch.zeros(out_features))

    def sample_noise(self):
        eps_in = torch.sign(torch.randn(self.in_features)) * torch.sqrt(torch.abs(torch.randn(self.in_features)))
        eps_out = torch.sign(torch.randn(self.out_features)) * torch.sqrt(torch.abs(torch.randn(self.out_features)))
        self.eps_w.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.eps_b.copy_(eps_out)

    def forward(self, x):
        w = self.mu_w + self.sigma_w * self.eps_w.to(x.device)
        b = self.mu_b + self.sigma_b * self.eps_b.to(x.device)
        return F.linear(x, w, b)


class RainbowQNet(nn.Module):
    def __init__(self, input_dim, n_actions, n_atoms=51, v_min=-10, v_max=10, hidden_dims=(128, 128)):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.noisy1 = NoisyLinear(hidden_dims[0], hidden_dims[1])
        self.fc_v = nn.Linear(hidden_dims[1], hidden_dims[1] // 2)
        self.noisy_v = NoisyLinear(hidden_dims[1] // 2, n_atoms)
        self.fc_a = nn.Linear(hidden_dims[1], hidden_dims[1] // 2)
        self.noisy_a = NoisyLinear(hidden_dims[1] // 2, n_actions * n_atoms)

        _ortho_init(self.fc1)
        _ortho_init(self.fc_v)
        _ortho_init(self.fc_a)

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.register_buffer("delta_z", (v_max - v_min) / (n_atoms - 1))

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.noisy1(h))
        v = self.noisy_v(F.relu(self.fc_v(h))).view(-1, 1, self.n_atoms)
        a = self.noisy_a(F.relu(self.fc_a(h))).view(-1, self.n_actions, self.n_atoms)
        q = v + a - a.mean(dim=1, keepdim=True)
        return F.softmax(q, dim=2)

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()

    def get_q_values(self, x):
        return (self.forward(x) * self.support.to(x.device)).sum(dim=2)


def train_rainbow_dqn(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete, "Rainbow DQN requires discrete action space"

    device = torch.device(config.device)
    n_atoms, v_min, v_max = 51, -10.0, 10.0

    q_net = RainbowQNet(env.state_dim, env.action_dim, n_atoms, v_min, v_max, config.hidden_dims).to(device)
    target_net = copy.deepcopy(q_net)
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.lr)
    buffer = PrioritizedReplayBuffer(config.buffer_capacity, device=config.device)

    support = torch.linspace(v_min, v_max, n_atoms).to(device)
    delta = (v_max - v_min) / (n_atoms - 1)

    n_step = 3
    n_buf = deque(maxlen=n_step)

    rewards_history, losses = [], []
    t0 = time.time()
    total_steps, episode = 0, 0
    converged, episodes_to_solve = False, None
    tracker = PlateauTracker(config.early_stop_patience, config.early_stop_min_delta, config.solve_window)
    max_steps = config.max_steps_per_episode

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)

    while episode < config.max_episodes:
        state = env.reset()
        episode_reward, episode_loss = 0, []
        n_buf.clear()
        q_net.sample_noise()

        for step in range(max_steps):
            total_steps += 1

            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = q_net.get_q_values(s).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            n_buf.append((state, action, reward, next_state, float(done)))
            if len(n_buf) == n_step:
                s0, a0, _, _, _ = n_buf[0]
                cumulative, gamma_pow = 0.0, 1.0
                dn_final = 0.0
                for k in range(n_step):
                    _, _, rk, _, dk = n_buf[k]
                    cumulative += gamma_pow * rk
                    gamma_pow *= config.gamma
                    if dk:
                        dn_final = 1.0
                        break
                sn, _, _, _, _ = n_buf[-1]
                buffer.push(s0, a0, cumulative, sn, dn_final)

            state = next_state

            if len(buffer) >= config.min_buffer_size:
                states, actions, rewards, next_states, dones, weights, indices = buffer.sample(config.batch_size)
                B = config.batch_size

                with torch.no_grad():
                    next_probs = target_net(next_states)
                    next_q = (next_probs * support).sum(dim=2)
                    next_acts = next_q.argmax(dim=1)

                    target_probs = next_probs[range(B), next_acts]

                    Tz = rewards.unsqueeze(1) + (config.gamma ** n_step) * (1 - dones.unsqueeze(1)) * support.unsqueeze(0)
                    Tz = Tz.clamp(v_min, v_max)
                    b_idx = (Tz - v_min) / delta
                    l = b_idx.floor().long()
                    u = b_idx.ceil().long()

                    m = torch.zeros(B, n_atoms, device=device)
                    for i in range(n_atoms):
                        m.scatter_add_(1, l.clamp(0, n_atoms - 1),
                                        target_probs * (u.float() - b_idx.float()) * (l == i).float())
                        m.scatter_add_(1, u.clamp(0, n_atoms - 1),
                                        target_probs * (b_idx.float() - l.float()) * (u == i).float())

                log_p = torch.log(q_net(states)[range(B), actions.long()] + 1e-8)
                loss_td = -(m * log_p).sum(dim=1)
                loss = (weights.flatten() * loss_td).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

                td_errs = loss_td.detach().cpu().abs().numpy()
                buffer.update_priorities(indices, td_errs + 1e-6)
                episode_loss.append(loss.item())

            if total_steps % config.target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())
                target_net.sample_noise()

            if done:
                break

        episode += 1
        pbar.update(1)
        rewards_history.append(episode_reward)
        losses.append(np.mean(episode_loss) if episode_loss else 0.0)

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window:])
            pbar.set_postfix({"avg100": f"{avg:.1f}", "buf": len(buffer)})
            if config.is_solved(avg) and not converged:
                converged = True
                episodes_to_solve = episode

        if tracker.update(rewards_history):
            pbar.set_description(f"{config.algo_name} (plateau)")
            break

    pbar.close()
    result = Result(
        algo_name=config.algo_name, env_name=env.config.env_name,
        config=vars(config), episode_rewards=rewards_history, losses=losses,
        converged=converged, episodes_to_solve=episodes_to_solve,
        wall_time=time.time() - t0, total_steps=total_steps,
    )
    result.compute_running_avg()
    return result
