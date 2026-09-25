import numpy as np

from .common import (
    TabularAgent,
    episode_stats,
    greedy_action,
)


class TDControl(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.t = 0

    def update(self, s, a, r, s_next, done, a_next=None):
        raise NotImplementedError

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        losses = []
        while self.t < config["steps"]:
            state, _ = env.reset()
            done = False
            ret = 0.0
            a = self.select(state)
            while not done and self.t < config["steps"]:
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                a_next = self.select(ns)
                loss = self.update(state, a, r, ns, done, a_next)
                if loss is not None:
                    losses.append(loss)
                state, a = ns, a_next
                ret += r
                self.t += 1
            ep += 1
            episode_stats(tracker, self.t, ep, ret, np.mean(losses) if losses else None)
            losses = []


class Sarsa(TDControl):
    def update(self, s, a, r, s_next, done, a_next=None):
        target = r + (0.0 if done else self.gamma * self.Q[s_next, a_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        return abs(target - self.Q[s, a])


class ExpectedSarsa(TDControl):
    def update(self, s, a, r, s_next, done, a_next=None):
        if done:
            target = r
        else:
            eps = self.eps_sched.epsilon(self.t)
            target = r + self.gamma * (
                eps / self.nA * self.Q[s_next].sum() + (1 - eps) * self.Q[s_next].max()
            )
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        return abs(target - self.Q[s, a])


class QLearning(TDControl):
    def update(self, s, a, r, s_next, done, a_next=None):
        target = r + (0.0 if done else self.gamma * self.Q[s_next].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        return abs(target - self.Q[s, a])


class DoubleQLearning(TDControl):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.Q2 = np.zeros((self.nS, self.nA))

    def select(self, state):
        if self.rng.random() < self.eps_sched.epsilon(self.t):
            return int(self.rng.integers(self.nA))
        return int(np.argmax(self.Q[state] + self.Q2[state]))

    def update(self, s, a, r, s_next, done, a_next=None):
        if self.rng.random() < 0.5:
            qa, qb = self.Q, self.Q2
        else:
            qa, qb = self.Q2, self.Q
        if done:
            target = r
        else:
            a_star = greedy_action(qa, s_next)
            target = r + self.gamma * qb[s_next, a_star]
        qa[s, a] += self.alpha * (target - qa[s, a])
        return abs(target - qa[s, a])

    def act(self, state, eval=False):
        if eval:
            return int(np.argmax(self.Q[state] + self.Q2[state]))
        return self.select(state)

    def save(self, path):
        np.savez(path, Q=self.Q, Q2=self.Q2)

    def load(self, path):
        data = np.load(path)
        self.Q, self.Q2 = data["Q"], data["Q2"]
