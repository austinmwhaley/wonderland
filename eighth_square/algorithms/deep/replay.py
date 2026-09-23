import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, s, a, r, s2, done):
        i = self.pos
        self.obs[i] = s
        self.obs2[i] = s2
        self.act[i] = a
        self.rew[i] = r
        self.done[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._get(idx)

    def _get(self, idx):
        t = torch.from_numpy
        return (t(self.obs[idx]).to(self.device),
                t(self.act[idx]).to(self.device),
                t(self.rew[idx]).to(self.device).unsqueeze(1),
                t(self.obs2[idx]).to(self.device),
                t(self.done[idx]).to(self.device).unsqueeze(1))


class ContinuousReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, obs_dim, action_dim, device="cpu"):
        super().__init__(capacity, obs_dim, device)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.action_dim = action_dim


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.pos = 0
        self.size = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def add(self, priority, data):
        idx = self.pos + self.capacity - 1
        self.data[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.update(idx, priority)

    def update(self, idx, priority):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, value):
        idx = 0
        while True:
            left, right = 2 * idx + 1, 2 * idx + 2
            if left >= len(self.tree):
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, obs_dim, device="cpu", alpha=0.6, beta=0.4, beta_steps=100_000, eps=1e-6):
        super().__init__(capacity, obs_dim, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_steps = beta_steps
        self.eps = eps
        self.tree = SumTree(capacity)
        self._step = 0

    def push(self, s, a, r, s2, done, priority=1.0):
        self.tree.add(priority ** self.alpha, (np.asarray(s, dtype=np.float32),
                                               int(a), float(r),
                                               np.asarray(s2, dtype=np.float32), float(done)))
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        self._step += 1
        beta = min(1.0, self.beta + (1 - self.beta) * self._step / self.beta_steps)
        idxs, priorities, data = [], [], []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, p, d = self.tree.get(value)
            idxs.append(idx)
            priorities.append(p)
            data.append(d)
        weights = (np.array(priorities) * self.size) ** (-beta)
        weights /= weights.max()
        obs, act, rew, obs2, done = (np.stack([d[0] for d in data]),
                                     np.array([d[1] for d in data]),
                                     np.array([d[2] for d in data]),
                                     np.stack([d[3] for d in data]),
                                     np.array([d[4] for d in data]))
        t = torch.from_numpy
        return (t(obs).to(self.device), t(act).to(self.device),
                t(rew).to(self.device).unsqueeze(1), t(obs2).to(self.device),
                t(done).to(self.device).unsqueeze(1), idxs,
                t(weights).to(self.device).unsqueeze(1))

    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, (p + self.eps) ** self.alpha)