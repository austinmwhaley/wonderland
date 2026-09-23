import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.networks import QNetwork
from core.trainer import PlateauTracker
from environments.base import EnvWrapper


def train_vanilla_dqn(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete, "Vanilla DQN requires discrete action space"

    device = torch.device(config.device)
    q_net = QNetwork(env.state_dim, env.action_dim, config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.lr)

    rewards_history = []
    losses = []
    epsilon = config.epsilon_start
    t0 = time.time()
    total_steps = 0
    converged = False
    episodes_to_solve = None
    tracker = PlateauTracker(config.early_stop_patience, config.early_stop_min_delta, config.solve_window)

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(config.max_steps_per_episode):
            total_steps += 1

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = q_net(state_t).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            with torch.no_grad():
                target = reward + (0 if done else config.gamma * q_net(next_state_t).max().item())

            current_q = q_net(state_t)[0, action]
            loss = F.mse_loss(current_q, torch.tensor(target, dtype=torch.float32, device=device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss.append(loss.item())
            episode_reward += reward
            state = next_state

            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        rewards_history.append(episode_reward)
        losses.append(np.mean(episode_loss) if episode_loss else 0.0)

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window:])
            pbar.set_postfix({"avg100": f"{avg:.1f}", "eps": f"{epsilon:.3f}"})
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
