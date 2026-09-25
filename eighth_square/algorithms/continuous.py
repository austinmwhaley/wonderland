import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.networks import DeterministicPolicy, GaussianPolicy, TwinQNetwork
from core.replay_buffer import ReplayBuffer
from core.trainer import PlateauTracker
from environments.base import EnvWrapper


def train_ddpg(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_continuous, "DDPG requires continuous action space"

    device = torch.device(config.device)
    action_dim = env.action_dim
    action_scale = float(env.action_space.high[0])

    actor = DeterministicPolicy(env.state_dim, action_dim, config.hidden_dims, action_scale).to(
        device
    )
    critic = TwinQNetwork(env.state_dim, action_dim, config.hidden_dims).to(device)

    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.lr)

    buffer = ReplayBuffer(config.buffer_capacity, config.device)

    rewards_history = []
    losses = []
    t0 = time.time()
    total_steps = 0
    converged = False
    episodes_to_solve = None
    tracker = PlateauTracker(
        config.early_stop_patience, config.early_stop_min_delta, config.solve_window
    )

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(config.max_steps_per_episode):
            total_steps += 1

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(state_t).cpu().numpy().flatten()
                noise = np.random.normal(0, 0.1, size=action_dim)
                action = np.clip(action + noise, -action_scale, action_scale)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(buffer) >= config.batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    target_q1, target_q2 = target_critic(next_states, next_actions)
                    target_q = torch.min(target_q1, target_q2)
                    target = rewards + config.gamma * target_q * (1 - dones)

                q1, q2 = critic(states, actions)
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -critic.q1_forward(states, actor(states)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(
                        config.tau * param.data + (1 - config.tau) * target_param.data
                    )
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(
                        config.tau * param.data + (1 - config.tau) * target_param.data
                    )

                episode_loss.append(critic_loss.item())

            if done:
                break

        rewards_history.append(episode_reward)
        losses.append(np.mean(episode_loss) if episode_loss else 0.0)

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window :])
            pbar.set_postfix({"avg100": f"{avg:.1f}", "buf": len(buffer)})
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


def train_td3(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_continuous, "TD3 requires continuous action space"

    device = torch.device(config.device)
    action_dim = env.action_dim
    action_scale = float(env.action_space.high[0])

    actor = DeterministicPolicy(env.state_dim, action_dim, config.hidden_dims, action_scale).to(
        device
    )
    critic = TwinQNetwork(env.state_dim, action_dim, config.hidden_dims).to(device)

    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.lr)

    buffer = ReplayBuffer(config.buffer_capacity, config.device)

    rewards_history = []
    losses = []
    policy_update_freq = 2
    t0 = time.time()
    total_steps = 0
    converged = False
    episodes_to_solve = None
    tracker = PlateauTracker(
        config.early_stop_patience, config.early_stop_min_delta, config.solve_window
    )

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(config.max_steps_per_episode):
            total_steps += 1

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(state_t).cpu().numpy().flatten()
                noise = np.random.normal(0, 0.1, size=action_dim)
                action = np.clip(action + noise, -action_scale, action_scale)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(buffer) >= config.batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

                with torch.no_grad():
                    noise_td = (torch.randn_like(actions) * 0.2).clamp(-0.5, 0.5)
                    next_actions = (target_actor(next_states) + noise_td).clamp(
                        -action_scale, action_scale
                    )
                    target_q1, target_q2 = target_critic(next_states, next_actions)
                    target_q = torch.min(target_q1, target_q2)
                    target = rewards + config.gamma * target_q * (1 - dones)

                q1, q2 = critic(states, actions)
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                episode_loss.append(critic_loss.item())

                if total_steps % policy_update_freq == 0:
                    actor_loss = -critic.q1_forward(states, actor(states)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                        target_param.data.copy_(
                            config.tau * param.data + (1 - config.tau) * target_param.data
                        )
                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(
                            config.tau * param.data + (1 - config.tau) * target_param.data
                        )

            if done:
                break

        rewards_history.append(episode_reward)
        losses.append(np.mean(episode_loss) if episode_loss else 0.0)

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window :])
            pbar.set_postfix({"avg100": f"{avg:.1f}", "buf": len(buffer)})
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


def train_sac(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_continuous, "SAC requires continuous action space"

    device = torch.device(config.device)
    action_dim = env.action_dim

    actor = GaussianPolicy(env.state_dim, action_dim, config.hidden_dims).to(device)
    critic = TwinQNetwork(env.state_dim, action_dim, config.hidden_dims).to(device)

    target_critic = copy.deepcopy(critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.lr)

    if config.sac_auto_alpha:
        target_entropy = -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.lr)
        alpha = log_alpha.exp()
    else:
        alpha = config.sac_alpha

    buffer = ReplayBuffer(config.buffer_capacity, config.device)

    rewards_history = []
    losses = []
    t0 = time.time()
    total_steps = 0
    converged = False
    episodes_to_solve = None
    tracker = PlateauTracker(
        config.early_stop_patience, config.early_stop_min_delta, config.solve_window
    )

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(config.max_steps_per_episode):
            total_steps += 1

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = actor.sample(state_t)
                action = action.cpu().numpy().flatten()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(buffer) >= config.batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(config.batch_size)

                with torch.no_grad():
                    next_actions, next_log_probs, _ = actor.sample(next_states)
                    target_q1, target_q2 = target_critic(next_states, next_actions)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
                    target = rewards + config.gamma * target_q * (1 - dones)

                q1, q2 = critic(states, actions)
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                new_actions, log_probs, _ = actor.sample(states)
                q1_new, q2_new = critic(states, new_actions)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * log_probs - q_new).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if config.sac_auto_alpha:
                    alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    alpha = log_alpha.exp()

                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(
                        config.tau * param.data + (1 - config.tau) * target_param.data
                    )

                episode_loss.append(critic_loss.item())

            if done:
                break

        rewards_history.append(episode_reward)
        losses.append(np.mean(episode_loss) if episode_loss else 0.0)

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window :])
            pbar.set_postfix({"avg100": f"{avg:.1f}", "buf": len(buffer)})
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
