import numpy as np

from .common import TabularAgent, episode_stats, greedy_action


class TraceControl(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("lambda", 0.9)
        self.trace = config.get("trace", "accumulating")
        self.E = np.zeros((self.nS, self.nA))
        self.t = 0

    def target(self, s, a, r, s_next, done, a_next):
        raise NotImplementedError

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        losses = []
        while self.t < config["steps"]:
            self.E[:] = 0.0
            state, _ = env.reset()
            done = False
            ret = 0.0
            a = self.select(state)
            while not done and self.t < config["steps"]:
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                a_next = self.select(ns)
                if self.trace == "replacing":
                    self.E[state, a] = 1.0
                else:
                    self.E[state, a] += 1.0
                target = self.target(state, a, r, ns, done, a_next)
                delta = target - self.Q[state, a]
                self.Q += self.alpha * delta * self.E
                self.E *= self.gamma * self.lam
                losses.append(abs(delta))
                state, a = ns, a_next
                ret += r
                self.t += 1
            ep += 1
            episode_stats(tracker, self.t, ep, ret, np.mean(losses) if losses else None)
            losses = []


class SarsaLambda(TraceControl):
    def target(self, s, a, r, s_next, done, a_next):
        return r + (0.0 if done else self.gamma * self.Q[s_next, a_next])


class QLambda(TraceControl):
    def target(self, s, a, r, s_next, done, a_next):
        if done:
            self.E[:] = 0.0
            return r
        if a_next != greedy_action(self.Q, s_next):
            self.E[s_next, :] = 0.0
        return r + self.gamma * self.Q[s_next].max()