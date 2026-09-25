import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.trainer import PlateauTracker
from environments.base import DiscretizedEnvWrapper, EnvWrapper


def _choose_action_eps_greedy(q_table, state, n_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        q_vals = np.array([q_table[state][a] for a in range(n_actions)])
        return int(np.argmax(q_vals))


def train_q_learning(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete or isinstance(env, DiscretizedEnvWrapper), (
        "Q-Learning requires discrete actions and discretized states"
    )

    q_table = defaultdict(lambda: np.zeros(env.action_dim))
    rewards_history = []
    losses = []
    epsilon = config.epsilon_start
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
        if isinstance(env, DiscretizedEnvWrapper):
            state = env.discretize(state)
        episode_reward = 0

        for step in range(config.max_steps_per_episode):
            action_idx = _choose_action_eps_greedy(q_table, state, env.action_dim, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated

            if isinstance(env, DiscretizedEnvWrapper):
                next_state_disc = env.discretize(next_state)
            else:
                next_state_disc = next_state

            best_next = (
                np.max([q_table[next_state_disc][a] for a in range(env.action_dim)])
                if not done
                else 0.0
            )
            td_target = reward + config.gamma * best_next
            td_error = td_target - q_table[state][action_idx]
            q_table[state][action_idx] += config.lr * td_error

            state = next_state_disc
            episode_reward += reward
            total_steps += 1

            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        rewards_history.append(episode_reward)
        losses.append(float(abs(td_error)))

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window :])
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


def train_sarsa(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete or isinstance(env, DiscretizedEnvWrapper), (
        "SARSA requires discrete actions and discretized states"
    )

    q_table = defaultdict(lambda: np.zeros(env.action_dim))
    rewards_history = []
    losses = []
    epsilon = config.epsilon_start
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
        if isinstance(env, DiscretizedEnvWrapper):
            state = env.discretize(state)
        action = _choose_action_eps_greedy(q_table, state, env.action_dim, epsilon)
        episode_reward = 0

        for step in range(config.max_steps_per_episode):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if isinstance(env, DiscretizedEnvWrapper):
                next_state_disc = env.discretize(next_state)
            else:
                next_state_disc = next_state

            next_action = _choose_action_eps_greedy(
                q_table, next_state_disc, env.action_dim, epsilon
            )
            td_target = reward + (
                0.0 if done else config.gamma * q_table[next_state_disc][next_action]
            )
            td_error = td_target - q_table[state][action]
            q_table[state][action] += config.lr * td_error

            state = next_state_disc
            action = next_action
            episode_reward += reward
            total_steps += 1

            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        rewards_history.append(episode_reward)
        losses.append(float(abs(td_error)))

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window :])
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


def train_expected_sarsa(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete or isinstance(env, DiscretizedEnvWrapper), (
        "Expected SARSA requires discrete actions and discretized states"
    )

    q_table = defaultdict(lambda: np.zeros(env.action_dim))
    rewards_history = []
    losses = []
    epsilon = config.epsilon_start
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
        if isinstance(env, DiscretizedEnvWrapper):
            state = env.discretize(state)
        episode_reward = 0

        for step in range(config.max_steps_per_episode):
            action = _choose_action_eps_greedy(q_table, state, env.action_dim, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if isinstance(env, DiscretizedEnvWrapper):
                next_state_disc = env.discretize(next_state)
            else:
                next_state_disc = next_state

            if done:
                expected_value = 0.0
            else:
                q_vals = np.array([q_table[next_state_disc][a] for a in range(env.action_dim)])
                best_action = np.max(q_vals)
                prob_best = 1.0 - epsilon + epsilon / env.action_dim
                prob_other = epsilon / env.action_dim
                expected_value = prob_best * best_action
                for a in range(env.action_dim):
                    if q_vals[a] != best_action:
                        expected_value += prob_other * q_vals[a]

            td_target = reward + config.gamma * expected_value
            td_error = td_target - q_table[state][action]
            q_table[state][action] += config.lr * td_error

            state = next_state_disc
            episode_reward += reward
            total_steps += 1

            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        rewards_history.append(episode_reward)
        losses.append(float(abs(td_error)))

        if len(rewards_history) >= config.solve_window:
            avg = np.mean(rewards_history[-config.solve_window :])
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
