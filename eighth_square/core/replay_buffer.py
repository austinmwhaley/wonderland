import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        priority = self.priorities.max() if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            batch_size = self.size

        probs = self.priorities[: self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = abs(td_err) + 1e-6

    def __len__(self):
        return self.size
