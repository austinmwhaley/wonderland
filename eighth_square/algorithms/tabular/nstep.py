from collections import deque

import numpy as np

from .common import TabularAgent, episode_stats, greedy_action


class NStepControl(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.n = config.get("n", 3)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.t = 0

    def bootstrap(self, s, a_next):
        raise NotImplementedError

    def target(self, buf, s_next, done, a_next):
        G = 0.0
        for i in range(len(buf)):
            G += self.gamma**i * buf[i][2]
        if not done:
            G += self.gamma ** len(buf) * self.bootstrap(s_next, a_next)
        return G

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        losses = []
        while self.t < config["steps"]:
            state, _ = env.reset()
            buf = deque()
            done = False
            ret = 0.0
            a = self.select(state)
            while not done and self.t < config["steps"]:
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                a_next = self.select(ns)
                buf.append((state, a, r))
                if len(buf) >= self.n:
                    s0, a0, _ = buf[0]
                    G = self.target(list(buf), ns, done, a_next)
                    self.Q[s0, a0] += self.alpha * (G - self.Q[s0, a0])
                    losses.append(abs(G - self.Q[s0, a0]))
                    buf.popleft()
                state, a = ns, a_next
                ret += r
                self.t += 1
            while buf:
                s0, a0, _ = buf[0]
                G = self.target(list(buf), None, True, None)
                self.Q[s0, a0] += self.alpha * (G - self.Q[s0, a0])
                buf.popleft()
            ep += 1
            episode_stats(tracker, self.t, ep, ret, np.mean(losses) if losses else None)
            losses = []


class NStepSarsa(NStepControl):
    def bootstrap(self, s, a_next):
        return self.Q[s, a_next]


class NStepExpectedSarsa(NStepControl):
    def bootstrap(self, s, a_next):
        eps = self.eps_sched.epsilon(self.t)
        return eps / self.nA * self.Q[s].sum() + (1 - eps) * self.Q[s].max()


class TreeBackup(NStepControl):
    def bootstrap(self, s, a_next):
        eps = self.eps_sched.epsilon(self.t)
        return eps / self.nA * self.Q[s].sum() + (1 - eps) * self.Q[s].max()


class NStepOffPolicy(NStepControl):
    def bootstrap(self, s, a_next):
        return self.Q[s].max()

    def target(self, buf, s_next, done, a_next):
        eps = self.eps_sched.epsilon(self.t)
        consistent = True
        for s_prev, a_prev, _ in buf[1:]:
            if a_prev != greedy_action(self.Q, s_prev):
                consistent = False
                break
        if not consistent:
            return None
        G = 0.0
        for i in range(len(buf)):
            G += self.gamma**i * buf[i][2]
        if not done:
            G += self.gamma ** len(buf) * self.Q[s_next].max()
            for _, a, _ in buf[1:]:
                G *= 1.0 / (1.0 - eps + eps / self.nA)
        return G

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        losses = []
        while self.t < config["steps"]:
            state, _ = env.reset()
            buf = deque()
            done = False
            ret = 0.0
            a = self.select(state)
            while not done and self.t < config["steps"]:
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                a_next = self.select(ns)
                buf.append((state, a, r))
                if len(buf) >= self.n:
                    s0, a0, _ = buf[0]
                    G = self.target(list(buf), ns, done, a_next)
                    if G is not None:
                        self.Q[s0, a0] += self.alpha * (G - self.Q[s0, a0])
                        losses.append(abs(G - self.Q[s0, a0]))
                    buf.popleft()
                state, a = ns, a_next
                ret += r
                self.t += 1
            while buf:
                s0, a0, _ = buf[0]
                G = self.target(list(buf), None, True, None)
                if G is not None:
                    self.Q[s0, a0] += self.alpha * (G - self.Q[s0, a0])
                buf.popleft()
            ep += 1
            episode_stats(tracker, self.t, ep, ret, np.mean(losses) if losses else None)
            losses = []
