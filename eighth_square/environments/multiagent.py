import numpy as np


class CooperativeCartPole:
    """Two agents control a shared cart: the applied force is the sum of both
    agents' forces. The state is fully shared (each agent observes the whole
    state vector) and the reward (+1 per step while balanced) is shared.
    Cooperative: agents must coordinate their force signs."""

    n_agents = 2
    n_actions = 2
    max_episode_steps = 500

    def __init__(self, seed=None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 5.0
        self.tau = 0.02
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.state = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        return (self.state.copy(), self.state.copy()), {}

    def step(self, actions):
        a1, a2 = int(actions[0]), int(actions[1])
        force = (self.force_mag if a1 == 1 else -self.force_mag) + \
                (self.force_mag if a2 == 1 else -self.force_mag)
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot
        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)
        self.steps += 1
        terminated = bool(x < -self.x_threshold or x > self.x_threshold
                          or theta < -self.theta_threshold_radians
                          or theta > self.theta_threshold_radians)
        truncated = self.steps >= self.max_episode_steps
        reward = 1.0
        return (self.state.copy(), self.state.copy()), reward, terminated, truncated, {}

    @property
    def unwrapped(self):
        return self

    @property
    def spec(self):
        return type("Spec", (), {"max_episode_steps": self.max_episode_steps})

    def close(self):
        pass