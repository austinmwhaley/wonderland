import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.networks import DuelingMLP, QNetwork
from core.replay_buffer import ReplayBuffer
from core.trainer import PlateauTracker
from environments.base import EnvWrapper


def _train_step(q_net, target_net, optimizer, buffer, config, device):
    if len(buffer) < config.min_buffer_size:
        return 0.0

    states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        target = rewards.flatten() + (1 - dones.flatten()) * config.gamma * next_q

    current_q = q_net(states).gather(1, actions.long().unsqueeze(1)).squeeze()
    loss = F.smooth_l1_loss(current_q, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def train_dqn(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete, "DQN requires discrete action space"

    device = torch.device(config.device)
    q_net = QNetwork(env.state_dim, env.action_dim, config.hidden_dims).to(device)
    target_net = copy.deepcopy(q_net)
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_capacity, config.device)

    rewards_history, losses = [], []
    t0 = time.time()
    total_steps, episode = 0, 0
    converged, episodes_to_solve = False, None
    tracker = PlateauTracker(
        config.early_stop_patience, config.early_stop_min_delta, config.solve_window
    )

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)

    while episode < config.max_episodes:
        state = env.reset()
        episode_reward, episode_loss = 0, []

        for step in range(config.max_steps_per_episode):
            total_steps += 1
            epsilon_val = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * np.exp(
                -total_steps / (config.max_episodes * 25)
            )

            if random.random() < epsilon_val:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = q_net(s).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            buffer.push(state, action, reward, next_state, float(done))
            state = next_state

            loss_v = _train_step(q_net, target_net, optimizer, buffer, config, device)
            if loss_v:
                episode_loss.append(loss_v)

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
            avg = np.mean(rewards_history[-config.solve_window :])
            pbar.set_postfix(
                {"avg100": f"{avg:.1f}", "eps": f"{epsilon_val:.3f}", "buf": len(buffer)}
            )
            if config.is_solved(avg) and not converged:
                converged = True
                episodes_to_solve = episode

        if tracker.update(rewards_history):
            pbar.set_description(f"{config.algo_name} (plateau)")
            break

    pbar.close()
    result = Result(
        algo_name=config.algo_name,
        env_name=env.config.env_name,
        config=vars(config),
        episode_rewards=rewards_history,
        losses=losses,
        converged=converged,
        episodes_to_solve=episodes_to_solve,
        wall_time=time.time() - t0,
        total_steps=total_steps,
    )
    result.compute_running_avg()
    return result


def train_double_dqn(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete, "Double DQN requires discrete action space"

    device = torch.device(config.device)
    q_net = QNetwork(env.state_dim, env.action_dim, config.hidden_dims).to(device)
    target_net = copy.deepcopy(q_net)
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_capacity, config.device)

    rewards_history, losses = [], []
    t0 = time.time()
    total_steps, episode = 0, 0
    converged, episodes_to_solve = False, None
    tracker = PlateauTracker(
        config.early_stop_patience, config.early_stop_min_delta, config.solve_window
    )

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)

    while episode < config.max_episodes:
        state = env.reset()
        episode_reward, episode_loss = 0, []

        for step in range(config.max_steps_per_episode):
            total_steps += 1
            epsilon_val = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * np.exp(
                -total_steps / (config.max_episodes * 25)
            )

            if random.random() < epsilon_val:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = q_net(s).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state

            if len(buffer) >= config.min_buffer_size:
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

                with torch.no_grad():
                    best_actions = q_net(next_states).argmax(dim=1, keepdim=True)
                    next_q = target_net(next_states).gather(1, best_actions).squeeze()
                    target = rewards.flatten() + (1 - dones.flatten()) * config.gamma * next_q

                current_q = q_net(states).gather(1, actions.long().unsqueeze(1)).squeeze()
                loss = F.smooth_l1_loss(current_q, target)

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
            avg = np.mean(rewards_history[-config.solve_window :])
            pbar.set_postfix(
                {"avg100": f"{avg:.1f}", "eps": f"{epsilon_val:.3f}", "buf": len(buffer)}
            )
            if config.is_solved(avg) and not converged:
                converged = True
                episodes_to_solve = episode

        if tracker.update(rewards_history):
            pbar.set_description(f"{config.algo_name} (plateau)")
            break

    pbar.close()
    result = Result(
        algo_name=config.algo_name,
        env_name=env.config.env_name,
        config=vars(config),
        episode_rewards=rewards_history,
        losses=losses,
        converged=converged,
        episodes_to_solve=episodes_to_solve,
        wall_time=time.time() - t0,
        total_steps=total_steps,
    )
    result.compute_running_avg()
    return result


def train_dueling_dqn(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete, "Dueling DQN requires discrete action space"

    device = torch.device(config.device)
    q_net = DuelingMLP(env.state_dim, env.action_dim, config.hidden_dims).to(device)
    target_net = copy.deepcopy(q_net)
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_capacity, config.device)

    rewards_history, losses = [], []
    t0 = time.time()
    total_steps, episode = 0, 0
    converged, episodes_to_solve = False, None
    tracker = PlateauTracker(
        config.early_stop_patience, config.early_stop_min_delta, config.solve_window
    )

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)

    while episode < config.max_episodes:
        state = env.reset()
        episode_reward, episode_loss = 0, []

        for step in range(config.max_steps_per_episode):
            total_steps += 1
            epsilon_val = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * np.exp(
                -total_steps / (config.max_episodes * 25)
            )

            if random.random() < epsilon_val:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = q_net(s).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state

            if len(buffer) >= config.min_buffer_size:
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

                with torch.no_grad():
                    best_actions = q_net(next_states).argmax(dim=1, keepdim=True)
                    next_q = target_net(next_states).gather(1, best_actions).squeeze()
                    target = rewards.flatten() + (1 - dones.flatten()) * config.gamma * next_q

                current_q = q_net(states).gather(1, actions.long().unsqueeze(1)).squeeze()
                loss = F.smooth_l1_loss(current_q, target)

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
            avg = np.mean(rewards_history[-config.solve_window :])
            pbar.set_postfix(
                {"avg100": f"{avg:.1f}", "eps": f"{epsilon_val:.3f}", "buf": len(buffer)}
            )
            if config.is_solved(avg) and not converged:
                converged = True
                episodes_to_solve = episode

        if tracker.update(rewards_history):
            pbar.set_description(f"{config.algo_name} (plateau)")
            break

    pbar.close()
    result = Result(
        algo_name=config.algo_name,
        env_name=env.config.env_name,
        config=vars(config),
        episode_rewards=rewards_history,
        losses=losses,
        converged=converged,
        episodes_to_solve=episodes_to_solve,
        wall_time=time.time() - t0,
        total_steps=total_steps,
    )
    result.compute_running_avg()
    return result
