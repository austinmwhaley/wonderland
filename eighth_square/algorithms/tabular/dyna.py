import heapq

import numpy as np

from .common import TabularAgent, episode_stats


class DynaQ(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.n_plan = config.get("n_plan", 10)
        self.model = {}
        self.t = 0

    def learn(self, s, a, r, s_next, done):
        target = r + (0.0 if done else self.gamma * self.Q[s_next].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def plan(self):
        if not self.model:
            return
        keys = list(self.model.keys())
        for _ in range(self.n_plan):
            s = keys[self.rng.integers(len(keys))]
            a = int(self.rng.integers(self.nA))
            if a not in self.model[s]:
                continue
            r, s_next, done = self.model[s][a]
            self.learn(s, a, r, s_next, done)

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        while self.t < config["steps"]:
            state, _ = env.reset()
            done = False
            ret = 0.0
            while not done and self.t < config["steps"]:
                a = self.select(state)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                self.model.setdefault(state, {})[a] = (r, ns, done)
                self.learn(state, a, r, ns, done)
                self.plan()
                state = ns
                ret += r
                self.t += 1
            ep += 1
            episode_stats(tracker, self.t, ep, ret)

    def save(self, path):
        np.savez(path, Q=self.Q)


class DynaQPlus(DynaQ):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.kappa = config.get("kappa", 0.1)
        self.last_seen = {}

    def plan(self):
        if not self.model:
            return
        keys = list(self.model.keys())
        for _ in range(self.n_plan):
            s = keys[self.rng.integers(len(keys))]
            a = int(self.rng.integers(self.nA))
            if a not in self.model[s]:
                continue
            r, s_next, done = self.model[s][a]
            tau = self.t - self.last_seen.get((s, a), 0)
            target = r + self.kappa * np.sqrt(tau) + (0.0 if done else self.gamma * self.Q[s_next].max())
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        while self.t < config["steps"]:
            state, _ = env.reset()
            done = False
            ret = 0.0
            while not done and self.t < config["steps"]:
                a = self.select(state)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                self.model.setdefault(state, {})[a] = (r, ns, done)
                self.last_seen[(state, a)] = self.t
                self.learn(state, a, r, ns, done)
                self.plan()
                state = ns
                ret += r
                self.t += 1
            ep += 1
            episode_stats(tracker, self.t, ep, ret)


class PrioritizedSweeping(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.n_plan = config.get("n_plan", 10)
        self.theta = config.get("theta", 0.5)
        self.model = {}
        self.predecessors = {}
        self.t = 0

    def learn(self, s, a, r, s_next, done):
        target = r + (0.0 if done else self.gamma * self.Q[s_next].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def priority(self, s, a, r, s_next, done):
        target = r + (0.0 if done else self.gamma * self.Q[s_next].max())
        return abs(target - self.Q[s, a])

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        queue = []
        while self.t < config["steps"]:
            state, _ = env.reset()
            done = False
            ret = 0.0
            while not done and self.t < config["steps"]:
                a = self.select(state)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                self.model.setdefault(state, {})[a] = (r, ns, done)
                self.predecessors.setdefault(ns, []).append((state, a, r))
                p = self.priority(state, a, r, ns, done)
                if p > self.theta:
                    heapq.heappush(queue, (-p, self.t, state, a))
                for _ in range(self.n_plan):
                    if not queue:
                        break
                    _, _, s, a0 = heapq.heappop(queue)
                    r0, s_next0, done0 = self.model[s][a0]
                    self.learn(s, a0, r0, s_next0, done0)
                    for s_pred, a_pred, r_pred in self.predecessors.get(s, []):
                        p = self.priority(s_pred, a_pred, r_pred, s, done0)
                        if p > self.theta:
                            heapq.heappush(queue, (-p, self.t, s_pred, a_pred))
                state = ns
                ret += r
                self.t += 1
            ep += 1
            episode_stats(tracker, self.t, ep, ret)

    def save(self, path):
        np.savez(path, Q=self.Q)