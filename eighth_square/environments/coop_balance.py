import numpy as np


class CooperativeBalanceDeliver:
    """OFFSET L3: two drones carry a shared payload (balanced on a cart) to a
    common goal.

    The payload cart moves along x by the SUM of both drones' forces (each +F or
    -F), exactly the CooperativeCartPole summed-control mechanism the ladder
    calls for. To move the cart toward the goal without the balanced payload
    tipping past a threshold, the two drones must apply coordinated (and at
    times anti-correlated) forces: keep the payload upright (each contributes
    against tilt) while accelerating/decelerating to reach the goal cleanly.
    A single greedy 'always push toward goal' policy tilts the payload and
    fails, so independent learners struggle; centralized value-decomposition
    (QMIX) can learn the joint force-balancing rhythm.

    obs per agent (shared): x, x_dot, theta, theta_dot, goal_delta (relative to
    world width) => obs_dim 5. Common goal + shared reward.
    """

    n_agents = 2
    n_actions = 2
    max_episode_steps = 400

    def __init__(self, seed=None, world_width=10.0):
        self.world_width = float(world_width)
        self.goal_x = 1.8  # cart-pole x is confined to ~[-2.4, 2.4]
        # cart-pole dynamics (same tuned params as CooperativeCartPole)
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
        self.init_scale = 0.05
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.state = np.zeros(4)
        self.obs_dim = 5

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(low=-self.init_scale, high=self.init_scale, size=(4,))
        self.steps = 0
        o = self._obs()
        return (o.copy(), o.copy()), {}

    def _obs(self):
        x, x_dot, theta, theta_dot = self.state
        goal_delta = (self.goal_x - x) / self.world_width
        return np.array([x / self.world_width, x_dot, theta, theta_dot, goal_delta],
                        dtype=np.float64)

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
        delivered = abs(x - self.goal_x) < 0.5 and abs(theta) < self.theta_threshold_radians
        terminated = bool(x < -self.x_threshold or x > self.x_threshold
                          or theta < -self.theta_threshold_radians
                          or theta > self.theta_threshold_radians)
        truncated = self.steps >= self.max_episode_steps
        # shared reward: encourage progress toward goal while balanced, speed reward on delivery
        goal_delta = abs(x - self.goal_x)
        r = float((self.theta_threshold_radians - abs(theta)) / self.theta_threshold_radians) * 0.5
        if not terminated:
            r += 0.1
        if delivered:
            r += 5.0
        o = self._obs()
        return (o.copy(), o.copy()), r, terminated, truncated, \
            {"goal_delta": goal_delta, "theta": theta, "delivered": delivered}

    @property
    def unwrapped(self):
        return self

    @property
    def spec(self):
        return type("Spec", (), {"max_episode_steps": self.max_episode_steps})

    def close(self):
        pass
