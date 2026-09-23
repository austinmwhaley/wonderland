from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Result:
    algo_name: str
    env_name: str
    config: dict = field(default_factory=dict)
    episode_rewards: list = field(default_factory=list)
    avg_rewards: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    converged: bool = False
    episodes_to_solve: int | None = None
    wall_time: float = 0.0
    total_steps: int = 0

    def compute_running_avg(self, window: int = 100):
        rewards = np.array(self.episode_rewards)
        if len(rewards) < window:
            self.avg_rewards = [float(rewards[: i + 1].mean()) for i in range(len(rewards))]
        else:
            self.avg_rewards = [
                float(rewards[i - window + 1: i + 1].mean())
                for i in range(window - 1, len(rewards))
            ]
        return self.avg_rewards


# ==================== merged from reinforcement_learning ====================

import os
import time

import numpy as np


class BaseAgent:
    family = "base"
    policy = "none"
    action_space = "any"
    state_space = "any"
    status = "implemented"

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.get("seed", 0))

    def act(self, state, eval=False):
        raise NotImplementedError

    def train(self, env, config, tracker):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    @staticmethod
    def checkpoint(agent, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        agent.save(path)


class EpsilonScheduler:
    def __init__(self, start, end, decay_steps, rng=None):
        self.start = start
        self.end = end
        self.decay_steps = max(1, decay_steps)
        self.rng = rng if rng is not None else np.random.default_rng()

    def epsilon(self, t):
        frac = min(1.0, t / self.decay_steps)
        return self.end + (self.start - self.end) * (1.0 - frac)

    def should_explore(self, t):
        return self.rng.random() < self.epsilon(t)


def seed_env(env, seed):
    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    return env


def evaluate(agent, env, episodes=5):
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        if hasattr(agent, "reset_episode"):
            agent.reset_episode()
        done = False
        ret = 0.0
        steps = 0
        while not done:
            action = agent.act(state, eval=True)
            ns, r, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            state = ns
            ret += r
            steps += 1
            if steps > 10_000:
                break
        returns.append(ret)
    return float(np.mean(returns))


def timeit():
    return time.time()