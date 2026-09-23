import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.trainer import PlateauTracker
from environments.base import DiscretizedEnvWrapper, EnvWrapper


def train_dyna_q(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete or isinstance(env, DiscretizedEnvWrapper), \
        "Dyna-Q requires discrete actions and discretized states"

    n_planning_steps = 50
    q_table = defaultdict(lambda: np.zeros(env.action_dim))
    model = {}

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
        if isinstance(env, DiscretizedEnvWrapper):
            state = env.discretize(state)
        episode_reward = 0

        for step in range(config.max_steps_per_episode):
            if np.random.random() < epsilon:
                action = np.random.randint(env.action_dim)
            else:
                q_vals = np.array([q_table[state][a] for a in range(env.action_dim)])
                action = int(np.argmax(q_vals))

            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if isinstance(env, DiscretizedEnvWrapper):
                next_state = env.discretize(next_state_raw)
            else:
                next_state = next_state_raw

            best_next = np.max([q_table[next_state][a] for a in range(env.action_dim)]) \
                if not done else 0.0
            td_error = reward + config.gamma * best_next - q_table[state][action]
            q_table[state][action] += config.lr * td_error

            model[(state, action)] = (reward, next_state, done)
            total_steps += 1
            episode_reward += reward
            state = next_state

            if done:
                break

        for _ in range(n_planning_steps):
            if len(model) == 0:
                break
            idx = np.random.randint(len(model))
            (s_plan, a_plan), (r_plan, s_next_plan, d_plan) = list(model.items())[idx]

            best = 0.0 if d_plan else np.max([q_table[s_next_plan][a] for a in range(env.action_dim)])
            td_error_plan = r_plan + config.gamma * best - q_table[s_plan][a_plan]
            q_table[s_plan][a_plan] += config.lr * td_error_plan

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        rewards_history.append(episode_reward)
        losses.append(float(abs(td_error)))

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
