import numpy as np


class CooperativeUrban:
    """Two-dronne cooperative urban search-and-reach (L3).

    Two UAVs share one goal; the team succeeds only when BOTH drones are within
    the reach radius of the goal. Each drone has its own 5-action discrete control
    and observes: 8-ray lidar, goal delta, and (if teammate is within comm_radius)
    the teammate's normalized relative position. Reward is shared (team-wide).

    This is a CTDE benchmark: per-agent obs is used by decentralized actors, and
    QMIX/MAPPO apply a centralized critic over the joint obs.
    """

    n_agents = 2
    n_actions = 5
    max_episode_steps = 150

    def __init__(self, seed=None, world_size=10.0, buildings=None, comm_radius=5.0, max_steps=150):
        self.world_size = float(world_size)
        self.buildings = buildings if buildings is not None else []
        self.comm_radius = float(comm_radius)
        self.max_episode_steps = int(max_steps)
        self.dt = 0.2
        self.max_vel = 2.0
        self.reach_radius = 1.5
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.pos = np.zeros((2, 2), dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.obs_dim = 8 + 2 + 2  # lidar8 + goal delta + teammate x,y

    # ---- shared helpers (mirror UrbanWorld) ----
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

    def _free_point(self):
        for _ in range(100):
            x = self.rng.uniform(1.5, self.world_size - 1.5)
            y = self.rng.uniform(1.5, self.world_size - 1.5)
            p = np.array([x, y])
            if not self._inside_building(p):
                return p
        return np.array([1.5, 1.5])

    def _lidar(self, agent_idx):
        angles = np.deg2rad(np.arange(0, 360, 45))
        max_range = self.world_size * 1.4
        pos = self.pos[agent_idx]
        dists = []
        for ang in angles:
            dx, dy = np.cos(ang), np.sin(ang)
            best = max_range
            for t in np.arange(0.2, max_range, 0.25):
                p = pos + np.array([dx, dy]) * t
                if p[0] < 0 or p[0] > self.world_size or p[1] < 0 or p[1] > self.world_size:
                    best = min(best, t)
                    break
                if self._inside_building(p):
                    best = min(best, t)
                    break
            dists.append(best / max_range)
        return np.array(dists, dtype=np.float64)

    def _agent_obs(self, agent_idx):
        pos = self.pos[agent_idx]
        lidar = self._lidar(agent_idx)
        nd = (self.goal - pos) / self.world_size
        other = 1 - agent_idx
        d_teammate = np.linalg.norm(pos - self.pos[other])
        if d_teammate <= self.comm_radius:
            bearing = (self.pos[other] - pos) / self.world_size
        else:
            bearing = np.zeros(2, dtype=np.float64)  # out of range -> no comms
        return np.concatenate([lidar, nd, bearing]).astype(np.float64)

    # ---- gym-style interface ----
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos[0] = self._free_point()
        self.pos[1] = self._free_point()
        while np.linalg.norm(self.pos[1] - self.pos[0]) < 3.0:
            self.pos[1] = self._free_point()
        goal = self._free_point()
        while np.linalg.norm(goal - self.pos[0]) < 3.0 or np.linalg.norm(goal - self.pos[1]) < 3.0:
            goal = self._free_point()
        self.goal = goal
        self.steps = 0
        return (self._agent_obs(0).astype(np.float32), self._agent_obs(1).astype(np.float32)), {}

    def step(self, actions):
        a0, a1 = int(actions[0]), int(actions[1])
        dirs = {
            0: np.array([0, 0]),
            1: np.array([0, 1]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
            4: np.array([1, 0]),
        }
        self._step_agent(0, dirs.get(a0, np.array([0, 0])) * 0.9)
        self._step_agent(1, dirs.get(a1, np.array([0, 0])) * 0.9)
        self.steps += 1
        d0 = float(np.linalg.norm(self.goal - self.pos[0]))
        d1 = float(np.linalg.norm(self.goal - self.pos[1]))
        both_reached = d0 < self.reach_radius and d1 < self.reach_radius
        # team reward: sparse success + mild combined progress shaping
        r = 1.0 if both_reached else 0.0
        terminated = bool(both_reached)
        truncated = bool(self.steps >= self.max_episode_steps)
        return (
            (self._agent_obs(0).astype(np.float32), self._agent_obs(1).astype(np.float32)),
            float(r),
            terminated,
            truncated,
            {"dist0": d0, "dist1": d1, "both": both_reached},
        )

    def _step_agent(self, idx, move):
        new_pos = self.pos[idx] + move
        new_pos[0] = np.clip(new_pos[0], 0.2, self.world_size - 0.2)
        new_pos[1] = np.clip(new_pos[1], 0.2, self.world_size - 0.2)
        if not self._inside_building(new_pos):
            self.pos[idx] = new_pos
        else:
            self.pos[idx] = self.pos[idx].copy()  # collide -> stay

    @property
    def unwrapped(self):
        return self

    @property
    def spec(self):
        return type("Spec", (), {"max_episode_steps": self.max_episode_steps})

    def close(self):
        pass
