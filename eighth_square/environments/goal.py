import numpy as np


class PointReach:
    """1-D point mass that must reach a random goal within a horizon.
    Sparse reward: +1 on reaching the goal, 0 otherwise (classic HER setup).
    Observation is (x, goal); the goal changes every episode."""

    metadata = {"render_modes": []}
    spec = None

    def __init__(self, seed=None, max_episode_steps=50):
        self.action_space = type("AS", (), {"n": 3, "shape": (1,)})()
        self.observation_space = type("OS", (), {"shape": (2,), "low": -6, "high": 6})()
        self.max_episode_steps = max_episode_steps
        self.n_actions = 3
        self.force = 0.12
        self.noise = 0.08
        self.goal_tol = 0.4
        self.low, self.high = -6.0, 6.0
        self._rng = np.random.default_rng(seed)
        self._elapsed_steps = 0

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.x = 0.0
        side = 1.0 if self._rng.random() < 0.5 else -1.0
        self.goal = side * self._rng.uniform(2.0, 5.0)
        self._elapsed_steps = 0
        return np.array([self.x, self.goal], dtype=np.float32), {}

    def step(self, action):
        a = int(action) - 1
        self.x = np.clip(
            self.x + self.force * a + self._rng.normal(0.0, self.noise), self.low, self.high
        )
        self._elapsed_steps += 1
        reached = abs(self.x - self.goal) < self.goal_tol
        term = bool(reached)
        trunc = self._elapsed_steps >= self.max_episode_steps
        info = {"goal": self.goal, "x": self.x}
        return np.array([self.x, self.goal], dtype=np.float32), float(reached), term, trunc, info

    def close(self):
        pass
