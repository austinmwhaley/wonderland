import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from algorithms.base import Result
from core.config import AlgorithmConfig
from core.networks import CategoricalPolicy, MLP
from core.trainer import PlateauTracker
from environments.base import EnvWrapper


def train_trpo(env: EnvWrapper, config: AlgorithmConfig) -> Result:
    assert env.is_discrete, "TRPO requires discrete action space"

    device = torch.device(config.device)
    policy = CategoricalPolicy(env.state_dim, env.action_dim, config.hidden_dims).to(device)
    value_net = MLP(env.state_dim, 1, config.hidden_dims).to(device)

    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=config.lr)
    rollout_size = config.rollout_steps
    max_steps = config.max_steps_per_episode
    kl_beta = config.kl_beta
    kl_target = config.kl_target

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
        rollout_states, rollout_actions, rollout_log_probs = [], [], []
        rollout_rewards, rollout_dones, rollout_values = [], [], []
        last_next_state = None

        while len(rollout_states) < rollout_size and episode_count < config.max_episodes:
            state = env.reset()
            episode_reward = 0
            ep_states, ep_actions, ep_log_probs = [], [], []
            ep_rewards, ep_dones, ep_values = [], [], []

            for step in range(max_steps):
                total_steps += 1
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, log_prob, _ = policy.sample(state_t)
                    value = value_net(state_t).item()

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                ep_states.append(state)
                ep_actions.append(action.item())
                ep_log_probs.append(log_prob.item())
                ep_rewards.append(reward)
                ep_dones.append(float(done))
                ep_values.append(value)
                episode_reward += reward
                state = next_state

                if done:
                    last_next_state = None
                    break

                if len(rollout_states) + len(ep_states) >= rollout_size:
                    last_next_state = state
                    break

            rollout_states.extend(ep_states)
            rollout_actions.extend(ep_actions)
            rollout_log_probs.extend(ep_log_probs)
            rollout_rewards.extend(ep_rewards)
            rollout_dones.extend(ep_dones)
            rollout_values.extend(ep_values)
            episode_count += 1

            rewards_history.append(episode_reward)
            pbar.update(1)

            if len(rewards_history) >= config.solve_window:
                avg = np.mean(rewards_history[-config.solve_window:])
                pbar.set_postfix({"avg100": f"{avg:.1f}", "kl_beta": f"{kl_beta:.3f}"})
                if config.is_solved(avg) and not converged:
                    converged = True
                    episodes_to_solve = episode_count

            if tracker.update(rewards_history):
                pbar.set_description(f"{config.algo_name} (plateau)")
                episode_count = config.max_episodes
                break

        if len(rollout_states) < 2:
            continue

        T = len(rollout_states)
        advantages = np.zeros(T, dtype=np.float32)
        rewards_arr = np.array(rollout_rewards, dtype=np.float32)
        dones_arr = np.array(rollout_dones, dtype=np.float32)
        values_arr = np.array(rollout_values, dtype=np.float32)

        if last_next_state is not None:
            with torch.no_grad():
                next_state_t = torch.FloatTensor(last_next_state).unsqueeze(0).to(device)
                bootstrap_value = value_net(next_state_t).item()
        else:
            bootstrap_value = 0.0

        last_gae = 0.0
        next_value = bootstrap_value
        for t in reversed(range(T)):
            delta = rewards_arr[t] + config.gamma * next_value * (1 - dones_arr[t]) - values_arr[t]
            last_gae = delta + config.gamma * config.gae_lambda * (1 - dones_arr[t]) * last_gae
            advantages[t] = last_gae
            next_value = values_arr[t]

        returns = advantages + values_arr

        states_t = torch.FloatTensor(np.array(rollout_states)).to(device)
        actions_t = torch.LongTensor(np.array(rollout_actions)).to(device)
        old_log_probs_t = torch.FloatTensor(np.array(rollout_log_probs)).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        returns_t = torch.FloatTensor(returns).to(device)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_batches = max(1, T // config.ppo_mini_batch_size)
        idx = np.arange(T)

        # TRPO: KL-penalty with adaptive beta
        for _ in range(config.ppo_epochs):
            np.random.shuffle(idx)
            total_kl = 0.0
            n_updates = 0

            for start in range(0, T, config.ppo_mini_batch_size):
                end = start + config.ppo_mini_batch_size
                mb_idx = idx[start:end]

                log_probs_new, entropy = policy.evaluate(states_t[mb_idx], actions_t[mb_idx])
                ratio = torch.exp(log_probs_new.squeeze() - old_log_probs_t[mb_idx])

                kl = (old_log_probs_t[mb_idx] - log_probs_new.squeeze()).mean()
                total_kl += kl.item()
                n_updates += 1

                policy_loss = -(ratio * advantages_t[mb_idx]).mean()
                value_loss = F.mse_loss(value_net(states_t[mb_idx]).squeeze(), returns_t[mb_idx])
                loss = policy_loss + kl_beta * kl + config.ppo_value_coef * value_loss + config.ppo_entropy_coef * (-entropy.mean())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # Adaptive KL beta
            avg_kl = total_kl / max(n_updates, 1)
            if avg_kl > kl_target * 1.5:
                kl_beta *= 2.0
            elif avg_kl < kl_target / 1.5:
                kl_beta /= 2.0
            kl_beta = max(0.01, min(10.0, kl_beta))

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
