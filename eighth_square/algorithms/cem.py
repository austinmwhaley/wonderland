import time

import numpy as np
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.trainer import PlateauTracker
from environments.base import EnvWrapper


def train_cem(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    state_dim = env.state_dim
    action_dim = env.action_dim
    is_continuous = env.is_continuous

    if is_continuous:
        action_scale = float(env.action_space.high[0])
    else:
        action_scale = None

    population_size = 60
    elite_frac = 0.2
    n_elite = max(1, int(population_size * elite_frac))

    if is_continuous:
        param_dim = state_dim * action_dim
        mean = np.zeros(param_dim)
        std = np.ones(param_dim) * action_scale * 0.5
    else:
        param_dim = action_dim
        theta = np.random.randn(param_dim)
        extra_std = 0.5

    rewards_history = []
    losses = []
    t0 = time.time()
    total_steps = 0
    converged = False
    episodes_to_solve = None
    tracker = PlateauTracker(config.early_stop_patience, config.early_stop_min_delta, config.solve_window)

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)
    for episode in pbar:
        episode_rewards_pop = []
        episode_samples = []

        for _ in range(population_size):
            if is_continuous:
                params = np.random.randn(param_dim) * std + mean
                W = params.reshape(state_dim, action_dim)
                total_reward = 0
                state = env.reset()
                for step in range(config.max_steps_per_episode):
                    total_steps += 1
                    action = np.clip(np.dot(state, W).flatten(), -action_scale, action_scale)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    state = next_state
                    if done:
                        break
            else:
                action_weights = theta + np.random.randn(param_dim) * extra_std
                total_reward = 0
                state = env.reset()
                for step in range(config.max_steps_per_episode):
                    total_steps += 1
                    state_flat = state.flatten()
                    scores = np.array([action_weights[a] * (state_flat[a % len(state_flat)] if a < len(state_flat) else 0)
                                       for a in range(param_dim)])
                    action = int(np.argmax(scores)) % min(action_dim, max(1, len(scores)))
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    state = next_state
                    if done:
                        break

            episode_rewards_pop.append(total_reward)
            episode_samples.append(params if is_continuous else action_weights)

        elite_indices = np.argsort(episode_rewards_pop)[-n_elite:]
        elite_samples = [episode_samples[i] for i in elite_indices]
        episode_reward = np.max(episode_rewards_pop)

        if is_continuous:
            mean = np.mean(elite_samples, axis=0)
            std = np.std(elite_samples, axis=0) + 1e-6
        else:
            theta = np.mean(elite_samples, axis=0)
            extra_std *= 0.99

        rewards_history.append(episode_reward)
        losses.append(np.mean(episode_rewards_pop))

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window:])
            pbar.set_postfix({"avg100": f"{avg:.1f}"})
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
