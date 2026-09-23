import numpy as np


class BanditEnv:
    def __init__(self, n_arms=10, seed=None):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.means = self.rng.normal(0.0, 1.0, n_arms)
        self.action_space = type("Space", (), {"n": n_arms})()
        self.observation_space = type("Space", (), {"n": 1})()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return 0, {}

    def step(self, action):
        r = float(self.rng.normal(self.means[int(action)], 1.0))
        return 0, r, False, False, {}

    def close(self):
        pass


class GaussianBanditEnv(BanditEnv):
    def __init__(self, n_arms=10, seed=None):
        super().__init__(n_arms, seed)
        self.means = self.rng.normal(0.0, 1.0, n_arms)
        self.variances = self.rng.uniform(0.5, 2.0, n_arms)

    def step(self, action):
        r = float(self.rng.normal(self.means[int(action)], np.sqrt(self.variances[int(action)])))
        return 0, r, False, False, {}