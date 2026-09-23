import numpy as np

from .common import TabularAgent, episode_stats, greedy_action, n_actions, n_states


class MCPrediction(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.V = np.zeros(self.nS)
        self.returns = [[] for _ in range(self.nS)]
        self.alpha = config.get("alpha", 0.05)
        self.gamma = config.get("gamma", 0.99)
        self.every_visit = config.get("every_visit", False)
        self.t = 0

    def policy(self, state):
        return int(self.rng.integers(self.nA))

    def act(self, state, eval=False):
        return self.policy(state)

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        while self.t < config["steps"]:
            state, _ = env.reset()
            traj = []
            done = False
            ret = 0.0
            while not done and self.t < config["steps"]:
                a = self.policy(state)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                traj.append((state, r))
                state = ns
                ret += r
                self.t += 1
            G = 0.0
            seen = set()
            for s, r in reversed(traj):
                G = r + self.gamma * G
                if self.every_visit or s not in seen:
                    self.returns[s].append(G)
                    self.V[s] += self.alpha * (G - self.V[s])
                    seen.add(s)
            ep += 1
            episode_stats(tracker, self.t, ep, ret)

    def save(self, path):
        np.savez(path, V=self.V)


class MCControlOnPolicy(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.t = 0

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        while self.t < config["steps"]:
            state, _ = env.reset()
            traj = []
            done = False
            ret = 0.0
            while not done and self.t < config["steps"]:
                a = self.act(state)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                traj.append((state, a, r))
                state = ns
                ret += r
                self.t += 1
            G = 0.0
            seen = set()
            for s, a, r in reversed(traj):
                G = r + self.gamma * G
                if (s, a) not in seen:
                    self.Q[s, a] += self.alpha * (G - self.Q[s, a])
                    seen.add((s, a))
            ep += 1
            episode_stats(tracker, self.t, ep, ret)

    def save(self, path):
        np.savez(path, Q=self.Q)


class MCControlOffPolicy(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.t = 0

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        n_actions = self.nA
        while self.t < config["steps"]:
            state, _ = env.reset()
            traj = []
            done = False
            ret = 0.0
            while not done and self.t < config["steps"]:
                a = self.act(state)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                traj.append((state, a, r))
                state = ns
                ret += r
                self.t += 1
            G = 0.0
            rho = 1.0
            seen = set()
            eps = self.eps_sched.epsilon(self.t)
            for s, a, r in reversed(traj):
                G = r + self.gamma * G
                if a != greedy_action(self.Q, s, self.rng):
                    rho = 0.0
                elif rho > 0:
                    rho *= 1.0 / (1.0 - eps + eps / n_actions)
                if (s, a) not in seen and rho > 0:
                    self.Q[s, a] += self.alpha * rho * (G - self.Q[s, a])
                    seen.add((s, a))
            ep += 1
            episode_stats(tracker, self.t, ep, ret)

    def save(self, path):
        np.savez(path, Q=self.Q)