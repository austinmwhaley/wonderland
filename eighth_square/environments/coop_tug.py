import numpy as np


class CooperativeTug:
    """L3 redesign: two-dronne package tug (summed-control delivery).

    A shared package P must be moved to a common goal G. Each step each drone
    picks a 5-action move; the package's displacement is the SUM of both drones'
    unit velocity vectors, gated by friction: the package only moves when the net
    push has magnitude >= 1.0 (i.e. BOTH drones push in the same axis direction).
    A single drone pushing alone cannot move the package.

    => coordination is MANDATORY: the two drones must agree on direction and push
    together, so independent learners struggle while centralized value-decomposition
    (QMIX) can learn the joint push sequence. Same summed-control mechanism as
    CooperativeCartPole, extended to a 2D shared goal.
    """

    n_agents = 2
    n_actions = 5
    max_episode_steps = 200

    def __init__(self, seed=None, world_size=10.0, buildings=None, comm_radius=6.0,
                 max_steps=200):
        self.world_size = float(world_size)
        self.buildings = buildings if buildings is not None else []
        self.comm_radius = float(comm_radius)
        self.max_episode_steps = int(max_steps)
        self.move = 0.9
        self.reach_radius = 1.5
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.pos = np.zeros((2, 2), dtype=np.float64)
        self.pkg = np.zeros(2, dtype=np.float64)
        self.pkg_v = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        # lidar8 + pkg-delta + goal-delta + pkg-velocity + teammate-bearing
        self.obs_dim = 8 + 2 + 2 + 2 + 2

    # ---- helpers (mirror UrbanWorld) ----
    def _inside_building(self, p):
        x, y = float(p[0]), float(p[1])
        for b in self.buildings:
            if len(b) == 5:
                x1, y1, x2, y2, _h = b
            else:
                x1, y1, x2, y2 = b
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _free_point(self, avoid=None, min_d=0.0):
        avoid = avoid if avoid is not None else []
        for _ in range(200):
            x = self.rng.uniform(1.5, self.world_size - 1.5)
            y = self.rng.uniform(1.5, self.world_size - 1.5)
            p = np.array([x, y])
            if self._inside_building(p):
                continue
            if all(np.linalg.norm(p - a) >= min_d for a in avoid):
                return p
        return np.array([1.5, 1.5])

    def _ring_point(self, center, rmin, rmax):
        """Point in annulus [rmin, rmax] around center, inside world, off buildings."""
        for _ in range(200):
            ang = self.rng.uniform(0, 2 * np.pi)
            r = self.rng.uniform(rmin, rmax)
            p = center + np.array([np.cos(ang), np.sin(ang)]) * r
            p[0] = np.clip(p[0], 1.0, self.world_size - 1.0)
            p[1] = np.clip(p[1], 1.0, self.world_size - 1.0)
            if not self._inside_building(p):
                return p
        return center + np.array([rmin, 0])

    def _lidar(self, idx, origin):
        max_range = self.world_size * 1.4
        angles = np.deg2rad(np.arange(0, 360, 45))
        dxs = np.cos(angles)
        dys = np.sin(angles)
        W = self.world_size
        tx = np.where(dxs > 0, (W - origin[0]) / np.maximum(dxs, 1e-9),
                      np.where(dxs < 0, (0 - origin[0]) / np.minimum(dxs, -1e-9), np.inf))
        ty = np.where(dys > 0, (W - origin[1]) / np.maximum(dys, 1e-9),
                      np.where(dys < 0, (0 - origin[1]) / np.minimum(dys, -1e-9), np.inf))
        t_border = np.minimum(tx, ty)
        t = np.clip(t_border, 0.0, max_range)
        return (t / max_range).astype(np.float64)

    def _agent_obs(self, idx):
        pos = self.pos[idx]
        lidar = self._lidar(idx, pos)
        pkg_d = (self.pkg - pos) / self.world_size
        goal_d = (self.goal - self.pkg) / self.world_size
        vel = (self.pkg_v / (self.world_size)) * 10.0
        other = 1 - idx
        if np.linalg.norm(pos - self.pos[other]) <= self.comm_radius:
            bearing = (self.pos[other] - pos) / self.world_size
        else:
            bearing = np.zeros(2, dtype=np.float64)
        return np.concatenate([lidar, pkg_d, goal_d, vel, bearing]).astype(np.float64)

    # ---- interface ----
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # package near a "pickup" region, goal far away
        self.pkg = np.array([self.world_size * 0.25, self.world_size * 0.5])
        self.goal = np.array([self.world_size * 0.75, self.world_size * 0.5])
        # drones spawn opposite sides of the package, both within reach of it
        d0 = self._ring_point(self.pkg, 1.5, 3.0)
        d1 = self._ring_point(self.pkg, 1.5, 3.0)
        for _ in range(50):
            if np.linalg.norm(d1 - d0) >= 2.0:
                break
            d1 = self._ring_point(self.pkg, 1.5, 3.0)
        self.pos[0] = d0
        self.pos[1] = d1
        self.pkg_v = np.zeros(2, dtype=np.float64)
        self.steps = 0
        return (self._agent_obs(0).astype(np.float32),
                self._agent_obs(1).astype(np.float32)), {}

    def step(self, actions):
        a0, a1 = int(actions[0]), int(actions[1])
        dirs = {0: np.array([0, 0]), 1: np.array([0, 1]), 2: np.array([0, -1]),
                3: np.array([-1, 0]), 4: np.array([1, 0])}
        self._step_agent(0, dirs.get(a0, np.array([0, 0])) * self.move)
        self._step_agent(1, dirs.get(a1, np.array([0, 0])) * self.move)
        prev_pkg_dist = float(np.linalg.norm(self.goal - self.pkg))
        prev_speed = float(np.linalg.norm(self.pkg_v))
        # summed-control with inertia (CooperativeCartPole-style): the package's
        # acceleration is the SUM of both drones' thrust. One drone's thrust alone
        # is weak; and because of inertia, both drones thrusting toward the goal
        # makes the package OVERSHOOT past it. To deliver cleanly the drones must
        # coordinate a push-then-BRAKE (anti-correlated) rhythm that only a
        # centralized method learns reliably.
        net = dirs.get(a0, np.array([0, 0])) + dirs.get(a1, np.array([0, 0]))
        self.pkg_v = self.pkg_v + net * (self.move * 0.6)
        self.pkg_v = self.pkg_v * 0.85  # friction
        new_pkg = self.pkg + self.pkg_v
        # bounce/stop at world walls
        for i in range(2):
            if new_pkg[i] < 1.0:
                new_pkg[i] = 1.0
                self.pkg_v[i] = 0.0
            elif new_pkg[i] > self.world_size - 1.0:
                new_pkg[i] = self.world_size - 1.0
                self.pkg_v[i] = 0.0
        self.pkg = new_pkg
        self.steps += 1
        pkg_dist = float(np.linalg.norm(self.goal - self.pkg))
        speed = float(np.linalg.norm(self.pkg_v))
        delivered = pkg_dist < self.reach_radius
        # dense team reward: progress reward + penalty for uncontrolled speed
        # (overshooting is punished) => coordinated, controlled delivery.
        progress = prev_pkg_dist - pkg_dist
        r = 3.0 * progress - 0.05 * prev_speed
        if delivered:
            r += 25.0
        terminated = bool(delivered)
        truncated = bool(self.steps >= self.max_episode_steps)
        return ((self._agent_obs(0).astype(np.float32),
                 self._agent_obs(1).astype(np.float32)),
                float(r), terminated, truncated,
                {"pkg_dist": pkg_dist, "speed": speed, "moved": progress > 0})

    def _step_agent(self, idx, move):
        new_pos = self.pos[idx] + move
        new_pos[0] = np.clip(new_pos[0], 0.2, self.world_size - 0.2)
        new_pos[1] = np.clip(new_pos[1], 0.2, self.world_size - 0.2)
        if not self._inside_building(new_pos):
            self.pos[idx] = new_pos

    @property
    def unwrapped(self):
        return self

    @property
    def spec(self):
        return type("Spec", (), {"max_episode_steps": self.max_episode_steps})

    def close(self):
        pass
