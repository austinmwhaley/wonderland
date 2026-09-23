import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.networks import QNetwork
from core.replay_buffer import ReplayBuffer
from core.trainer import PlateauTracker
from environments.base import EnvWrapper


class QuantileQNetwork(QNetwork):
    def __init__(self, input_dim, n_actions, n_quantiles=200, hidden_dims=(128, 128)):
        super().__init__(input_dim, n_actions * n_quantiles, hidden_dims)
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

    def forward(self, x):
        q = super().forward(x)
        return q.view(-1, self.n_actions, self.n_quantiles)


def _quantile_huber_loss(td_errors, kappa=1.0):
    return torch.where(td_errors.abs() <= kappa,
                       0.5 * td_errors.pow(2),
                       kappa * (td_errors.abs() - 0.5 * kappa))


def train_qr_dqn(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete, "QR-DQN requires discrete action space"

    device = torch.device(config.device)
    n_quantiles = 200
    q_net = QuantileQNetwork(env.state_dim, env.action_dim, n_quantiles, config.hidden_dims).to(device)
    target_net = copy.deepcopy(q_net)
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_capacity, config.device)

    tau = torch.FloatTensor((2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)).view(1, 1, -1).to(device)

    rewards_history, losses = [], []
    epsilon = config.epsilon_start
    t0 = time.time()
    total_steps, episode = 0, 0
    converged, episodes_to_solve = False, None
    tracker = PlateauTracker(config.early_stop_patience, config.early_stop_min_delta, config.solve_window)
    max_steps = config.max_steps_per_episode

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)

    while episode < config.max_episodes:
        state = env.reset()
        episode_reward, episode_loss = 0, []

        for step in range(max_steps):
            total_steps += 1
            epsilon_val = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * \
                          np.exp(-total_steps / (config.max_episodes * 25))

            if random.random() < epsilon_val:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = q_net(s).mean(dim=2).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state

            if len(buffer) >= config.min_buffer_size:
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

                with torch.no_grad():
                    next_q = target_net(next_states)
                    next_actions = next_q.mean(dim=2).argmax(dim=1, keepdim=True)
                    next_q = next_q.gather(1, next_actions.unsqueeze(2).expand(-1, -1, n_quantiles))

                target_q = rewards.unsqueeze(2) + (1 - dones.unsqueeze(2)) * config.gamma * next_q
                current_q = q_net(states).gather(1, actions.long().unsqueeze(1).unsqueeze(2).expand(-1, -1, n_quantiles))

                td_error = target_q - current_q
                huber = _quantile_huber_loss(td_error)
                loss = (tau * (td_error > 0).float() * huber + (1 - tau) * (td_error < 0).float() * huber).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                episode_loss.append(loss.item())

            if total_steps % config.target_update_freq == 0:
                for tp, p in zip(target_net.parameters(), q_net.parameters()):
                    tp.data.copy_(config.tau * p.data + (1 - config.tau) * tp.data)

            if done:
                break

        episode += 1
        pbar.update(1)
        rewards_history.append(episode_reward)
        losses.append(np.mean(episode_loss) if episode_loss else 0.0)

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window:])
            pbar.set_postfix({"avg100": f"{avg:.1f}", "eps": f"{epsilon_val:.3f}", "buf": len(buffer)})
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
