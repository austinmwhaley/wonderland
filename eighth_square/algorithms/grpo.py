import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.networks import CategoricalPolicy
from core.trainer import PlateauTracker
from environments.base import EnvWrapper


def train_grpo(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete, "GRPO requires discrete action space"

    device = torch.device(config.device)
    policy = CategoricalPolicy(env.state_dim, env.action_dim, config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)

    group_size = 16
    min_transitions = 512
    max_steps = config.max_steps_per_episode

    rewards_history = []
    losses = []
    t0 = time.time()
    total_steps = 0
    converged = False
    episodes_to_solve = None
    tracker = PlateauTracker(config.early_stop_patience, config.early_stop_min_delta, config.solve_window)
    episode_count = 0

    pbar = tqdm(range(config.max_episodes), desc=config.algo_name, unit="ep", leave=False)

    while episode_count < config.max_episodes:
        group_trajectories = []
        total_transitions = 0

        while len(group_trajectories) < group_size or total_transitions < min_transitions:
            if episode_count >= config.max_episodes:
                break

            state = env.reset()
            states, actions, log_probs = [], [], []
            episode_reward = 0

            for step in range(max_steps):
                total_steps += 1
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, log_prob, _ = policy.sample(state_t)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                states.append(state)
                actions.append(action.item())
                log_probs.append(log_prob.item())
                episode_reward += reward
                state = next_state

                if done:
                    break

            group_trajectories.append({
                "states": states, "actions": actions,
                "log_probs": log_probs, "return_val": episode_reward,
            })
            total_transitions += len(states)
            episode_count += 1

            rewards_history.append(float(episode_reward))
            pbar.update(1)

            if len(rewards_history) >= config.solve_window:
                avg = np.mean(rewards_history[-config.solve_window:])
                pbar.set_postfix({"avg100": f"{avg:.1f}"})

            if config.is_solved(np.mean(rewards_history[-min(config.solve_window, len(rewards_history)):])) \
                    and not converged and len(rewards_history) >= config.solve_window:
                converged = True
                episodes_to_solve = episode_count

            if tracker.update(rewards_history):
                pbar.set_description(f"{config.algo_name} (plateau)")
                episode_count = config.max_episodes
                break

        if episode_count >= config.max_episodes:
            break

        returns = np.array([t["return_val"] for t in group_trajectories])
        mean_return = returns.mean()
        std_return = returns.std() + 1e-8

        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []

        for traj in group_trajectories:
            advantage = (traj["return_val"] - mean_return) / std_return
            all_states.extend(traj["states"])
            all_actions.extend(traj["actions"])
            all_old_log_probs.extend(traj["log_probs"])
            all_advantages.extend([advantage] * len(traj["states"]))

        T = len(all_states)
        if T < 2:
            continue

        states_t = torch.FloatTensor(np.array(all_states)).to(device)
        actions_t = torch.LongTensor(np.array(all_actions)).to(device)
        old_log_probs_t = torch.FloatTensor(np.array(all_old_log_probs)).to(device)
        advantages_t = torch.FloatTensor(np.array(all_advantages)).to(device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_batches = max(1, T // config.ppo_mini_batch_size)
        idx = np.arange(T)

        for _ in range(config.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, config.ppo_mini_batch_size):
                end = start + config.ppo_mini_batch_size
                mb_idx = idx[start:end]

                log_probs_new, entropy = policy.evaluate(states_t[mb_idx], actions_t[mb_idx])
                ratio = torch.exp(log_probs_new.squeeze() - old_log_probs_t[mb_idx])
                surr1 = ratio * advantages_t[mb_idx]
                surr2 = torch.clamp(ratio, 1 - config.ppo_clip, 1 + config.ppo_clip) * advantages_t[mb_idx]
                loss = -torch.min(surr1, surr2).mean() + config.ppo_entropy_coef * (-entropy.mean())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

    pbar.close()
    if not losses:
        losses.append(0.0)
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
