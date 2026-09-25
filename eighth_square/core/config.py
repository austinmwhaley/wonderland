from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AlgorithmConfig:
    algo_name: str
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    max_episodes: int = 2000
    max_steps_per_episode: int = 500
    log_interval: int = 50
    seed: int = 42
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    )

    # Early stopping
    solve_threshold: Optional[float] = None
    solve_window: int = 100
    early_stop_patience: int = 200
    early_stop_min_delta: float = 0.01

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Replay buffer
    buffer_capacity: int = 100_000
    min_buffer_size: int = 1000
    target_update_freq: int = 100
    tau: float = 0.005

    # Networks
    hidden_dims: tuple = (128, 128)

    # PPO specific
    ppo_epochs: int = 10
    ppo_clip: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    gae_lambda: float = 0.95
    rollout_steps: int = 2048
    ppo_mini_batch_size: int = 64

    # TRPO / KL-penalty specific
    kl_target: float = 0.01
    kl_beta: float = 0.5

    # SAC specific
    sac_alpha: float = 0.2
    sac_auto_alpha: bool = True

    def is_solved(self, avg_reward: float) -> bool:
        return self.solve_threshold is not None and avg_reward >= self.solve_threshold


@dataclass
class EnvConfig:
    env_name: str
    n_discretization_bins: int = 20
    max_episodes: int = 2000
    max_steps_per_episode: int = 1000
    solve_threshold: float = 195.0
    solve_window: int = 100
    seed: int = 42


ALGO_DEFAULTS: dict[str, AlgorithmConfig] = {
    "q_learning": AlgorithmConfig(
        algo_name="q_learning",
        lr=0.1,
        epsilon_decay=0.999,
        max_episodes=5000,
        early_stop_patience=500,
    ),
    "sarsa": AlgorithmConfig(
        algo_name="sarsa", lr=0.1, epsilon_decay=0.999, max_episodes=5000, early_stop_patience=500
    ),
    "expected_sarsa": AlgorithmConfig(
        algo_name="expected_sarsa",
        lr=0.1,
        epsilon_decay=0.999,
        max_episodes=5000,
        early_stop_patience=500,
    ),
    "dyna_q": AlgorithmConfig(
        algo_name="dyna_q", lr=0.1, epsilon_decay=0.999, max_episodes=5000, early_stop_patience=500
    ),
    "vanilla_dqn": AlgorithmConfig(
        algo_name="vanilla_dqn",
        lr=1e-3,
        solve_threshold=195.0,
        max_episodes=5000,
        early_stop_patience=500,
    ),
    "dqn": AlgorithmConfig(
        algo_name="dqn",
        lr=1e-3,
        solve_threshold=195.0,
        target_update_freq=200,
        tau=0.005,
        max_episodes=5000,
        early_stop_patience=500,
        min_buffer_size=10000,
    ),
    "double_dqn": AlgorithmConfig(
        algo_name="double_dqn",
        lr=1e-3,
        solve_threshold=195.0,
        target_update_freq=200,
        tau=0.005,
        max_episodes=5000,
        early_stop_patience=500,
        min_buffer_size=10000,
    ),
    "dueling_dqn": AlgorithmConfig(
        algo_name="dueling_dqn",
        lr=1e-3,
        solve_threshold=195.0,
        target_update_freq=200,
        tau=0.005,
        max_episodes=5000,
        early_stop_patience=500,
        min_buffer_size=10000,
    ),
    "prioritized_dqn": AlgorithmConfig(
        algo_name="prioritized_dqn",
        lr=1e-3,
        solve_threshold=195.0,
        target_update_freq=200,
        tau=0.005,
        max_episodes=5000,
        early_stop_patience=500,
        min_buffer_size=10000,
    ),
    "qr_dqn": AlgorithmConfig(
        algo_name="qr_dqn",
        lr=1e-3,
        solve_threshold=195.0,
        target_update_freq=200,
        tau=0.005,
        max_episodes=5000,
        early_stop_patience=500,
        min_buffer_size=10000,
    ),
    "rainbow_dqn": AlgorithmConfig(
        algo_name="rainbow_dqn",
        lr=5e-4,
        solve_threshold=195.0,
        target_update_freq=2000,
        tau=1.0,
        max_episodes=5000,
        early_stop_patience=500,
        min_buffer_size=20000,
    ),
    "reinforce": AlgorithmConfig(
        algo_name="reinforce",
        lr=1e-3,
        solve_threshold=195.0,
        max_episodes=5000,
        early_stop_patience=500,
    ),
    "a2c": AlgorithmConfig(
        algo_name="a2c", lr=3e-4, solve_threshold=195.0, max_episodes=5000, early_stop_patience=500
    ),
    "ppo": AlgorithmConfig(
        algo_name="ppo",
        lr=3e-4,
        solve_threshold=195.0,
        max_episodes=5000,
        early_stop_patience=500,
        ppo_epochs=4,
    ),
    "grpo": AlgorithmConfig(
        algo_name="grpo", lr=3e-4, solve_threshold=195.0, max_episodes=5000, early_stop_patience=500
    ),
    "trpo": AlgorithmConfig(
        algo_name="trpo",
        lr=3e-4,
        solve_threshold=195.0,
        max_episodes=5000,
        early_stop_patience=500,
        ppo_epochs=4,
    ),
    "ddpg": AlgorithmConfig(
        algo_name="ddpg", lr=1e-3, solve_threshold=-100.0, max_episodes=500, tau=0.005
    ),
    "td3": AlgorithmConfig(
        algo_name="td3", lr=1e-3, solve_threshold=-100.0, max_episodes=500, tau=0.005
    ),
    "sac": AlgorithmConfig(
        algo_name="sac", lr=3e-4, solve_threshold=-100.0, max_episodes=500, tau=0.005
    ),
    "cem": AlgorithmConfig(algo_name="cem", lr=0.0, solve_threshold=195.0, max_episodes=500),
}

ENV_DEFAULTS: dict[str, EnvConfig] = {
    "CartPole-v1": EnvConfig(env_name="CartPole-v1", solve_threshold=195.0),
    "LunarLander-v3": EnvConfig(
        env_name="LunarLander-v3", solve_threshold=200.0, max_episodes=3000
    ),
    "MountainCar-v0": EnvConfig(
        env_name="MountainCar-v0", solve_threshold=-110.0, max_steps_per_episode=200
    ),
    "Acrobot-v1": EnvConfig(env_name="Acrobot-v1", solve_threshold=-100.0),
    "Pendulum-v1": EnvConfig(env_name="Pendulum-v1", solve_threshold=-150.0, max_episodes=500),
    "BipedalWalker-v3": EnvConfig(
        env_name="BipedalWalker-v3",
        solve_threshold=300.0,
        max_episodes=5000,
        max_steps_per_episode=1600,
    ),
    "FrozenLake-v1": EnvConfig(
        env_name="FrozenLake-v1",
        solve_threshold=0.7,
        max_episodes=5000,
        max_steps_per_episode=100,
        seed=42,
    ),
}
