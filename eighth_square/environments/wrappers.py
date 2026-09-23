import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Tabularize(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Tuple)
        self.sizes = [s.n for s in env.observation_space.spaces]
        self.strides = np.cumprod([1] + self.sizes[:-1])
        self.observation_space = spaces.Discrete(int(np.prod(self.sizes)))

    def observation(self, obs):
        return int(np.dot(np.asarray(obs), self.strides))


def discretize_observation(env, n_bins):
    lo = env.observation_space.low
    hi = env.observation_space.high
    widths = (hi - lo) / n_bins

    def obs_to_state(obs):
        idx = np.clip((np.asarray(obs) - lo) // widths, 0, n_bins - 1).astype(int)
        return int(np.ravel_multi_index(idx, (n_bins,) * len(lo)))

    return obs_to_state