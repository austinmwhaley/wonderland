from typing import Any

import gymnasium as gym
import numpy as np

from core.config import EnvConfig


class EnvWrapper:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.env = gym.make(config.env_name, max_episode_steps=config.max_steps_per_episode)
        self.env.reset(seed=config.seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._is_continuous = isinstance(self.action_space, gym.spaces.Box)
        self._is_discrete = isinstance(self.action_space, gym.spaces.Discrete)
        self._obs_is_discrete = isinstance(self.observation_space, gym.spaces.Discrete)

    @property
    def state_dim(self) -> int:
        if isinstance(self.observation_space, gym.spaces.Box):
            return int(np.prod(self.observation_space.shape))
        elif isinstance(self.observation_space, gym.spaces.Discrete):
            return self.observation_space.n
        return 0

    @property
    def action_dim(self) -> int:
        if self._is_discrete:
            return int(self.action_space.n)
        elif self._is_continuous:
            return int(np.prod(self.action_space.shape))
        return 0

    @property
    def is_continuous(self) -> bool:
        return self._is_continuous

    @property
    def is_discrete(self) -> bool:
        return self._is_discrete

    @property
    def obs_is_discrete(self) -> bool:
        return self._obs_is_discrete

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs

    def step(self, action: Any) -> tuple:
        return self.env.step(action)

    def seed(self, s: int):
        self.env.reset(seed=s)

    def close(self):
        self.env.close()


class DiscretizedEnvWrapper(EnvWrapper):
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.bins = config.n_discretization_bins

        if not isinstance(self.observation_space, gym.spaces.Box):
            raise ValueError("DiscretizedEnvWrapper only supports Box observation spaces")

        self.low = np.nan_to_num(self.observation_space.low, nan=-10.0, posinf=10.0, neginf=-10.0)
        self.high = np.nan_to_num(self.observation_space.high, nan=10.0, posinf=10.0, neginf=-10.0)
        self.bin_edges = [
            np.linspace(self.low[i], self.high[i], self.bins + 1)[1:-1]
            for i in range(self.state_dim)
        ]

    @staticmethod
    def _discretize_state(state: np.ndarray, bin_edges: list) -> int:
        indices = []
        for i, val in enumerate(state.flatten()):
            idx = int(np.digitize(val, bin_edges[i]))
            indices.append(idx)
        return tuple(indices)

    def discretize(self, state: np.ndarray):
        return self._discretize_state(state, self.bin_edges)


class OneHotEnvWrapper(EnvWrapper):
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        if not isinstance(self.observation_space, gym.spaces.Discrete):
            raise ValueError("OneHotEnvWrapper only supports Discrete observation spaces")

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        one_hot = np.zeros(self.observation_space.n, dtype=np.float32)
        one_hot[int(obs)] = 1.0
        return one_hot

    def step(self, action: Any) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        one_hot = np.zeros(self.observation_space.n, dtype=np.float32)
        one_hot[int(obs)] = 1.0
        return one_hot, reward, terminated, truncated, info


def _check_obs_type(env_name: str) -> str:
    tmp = gym.make(env_name)
    obs_space = tmp.observation_space
    tmp.close()
    if isinstance(obs_space, gym.spaces.Discrete):
        return "discrete"
    return "continuous"


def make_env(env_name: str, discretize: bool = False, one_hot: bool = False, **kwargs) -> Any:
    from core.config import ENV_DEFAULTS

    base_config = ENV_DEFAULTS.get(env_name, EnvConfig(env_name=env_name))
    for k, v in kwargs.items():
        if hasattr(base_config, k):
            setattr(base_config, k, v)

    if one_hot and _check_obs_type(env_name) == "discrete":
        return OneHotEnvWrapper(base_config)
    elif discretize and _check_obs_type(env_name) == "continuous":
        return DiscretizedEnvWrapper(base_config)
    return EnvWrapper(base_config)
