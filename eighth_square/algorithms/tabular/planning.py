import numpy as np

from .common import TabularAgent, episode_stats


class ModelBasedAgent(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        if not hasattr(env.unwrapped, "P"):
            raise ValueError(
                f"{type(self).__name__} needs an environment with explicit transition model (.P); "
                f"use frozenlake or cliffwalking"
            )
        self.P = env.unwrapped.P
        self.gamma = self.config.get("gamma", 0.99)
        self.policy = np.full((self.nS, self.nA), 1.0 / self.nA)

    def plan(self):
        raise NotImplementedError

    def act(self, state, eval=False):
        return int(self.rng.choice(self.nA, p=self.policy[state]))

    def train(self, env, config, tracker):
        self.plan()
        ep = 0
        for _ in range(self.config.get("eval_episodes", 10)):
            state, _ = env.reset()
            done = False
            ret = 0.0
            t = 0
            while not done:
                a = self.act(state, eval=True)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                state = ns
                ret += r
                t += 1
                if t >= self.config.get("max_episode_steps", 10_000):
                    break
            ep += 1
            episode_stats(tracker, t, ep, ret)

    def save(self, path):
        np.savez(path, policy=self.policy)


class PolicyEvaluation(ModelBasedAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.theta = self.config.get("theta", 1e-4)

    def plan(self):
        V = np.zeros(self.nS)
        for _ in range(self.config.get("max_iter", 1000)):
            delta = 0.0
            for s in range(self.nS):
                v = 0.0
                for a in range(self.nA):
                    for prob, s_next, r, term in self.P[s][a]:
                        if term:
                            v += self.policy[s, a] * prob * r
                        else:
                            v += self.policy[s, a] * prob * (r + self.gamma * V[s_next])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < self.theta:
                break
        self.V = V

    def save(self, path):
        np.savez(path, V=self.V)


class PolicyIteration(ModelBasedAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.theta = self.config.get("theta", 1e-4)

    def plan(self):
        self.policy = np.full((self.nS, self.nA), 1.0 / self.nA)
        for _ in range(self.config.get("max_policy_iters", 1000)):
            P_pi = np.zeros((self.nS, self.nS))
            r_pi = np.zeros(self.nS)
            for s in range(self.nS):
                for a in range(self.nA):
                    for prob, s_next, r, term in self.P[s][a]:
                        if not term:
                            P_pi[s, s_next] += self.policy[s, a] * prob
                        r_pi[s] += self.policy[s, a] * prob * r
            V = np.linalg.solve(np.eye(self.nS) - self.gamma * P_pi, r_pi)
            stable = True
            for s in range(self.nS):
                old = self.policy[s].copy()
                q = np.zeros(self.nA)
                for a in range(self.nA):
                    for prob, s_next, r, term in self.P[s][a]:
                        if term:
                            q[a] += prob * r
                        else:
                            q[a] += prob * (r + self.gamma * V[s_next])
                best = np.argmax(q)
                self.policy[s] = np.eye(self.nA)[best]
                if not np.array_equal(old, self.policy[s]):
                    stable = False
            if stable:
                break
        self.V = V


class ValueIteration(ModelBasedAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.theta = self.config.get("theta", 1e-4)

    def plan(self):
        V = np.zeros(self.nS)
        for _ in range(self.config.get("max_iter", 1000)):
            delta = 0.0
            for s in range(self.nS):
                q = np.zeros(self.nA)
                for a in range(self.nA):
                    for prob, s_next, r, term in self.P[s][a]:
                        if term:
                            q[a] += prob * r
                        else:
                            q[a] += prob * (r + self.gamma * V[s_next])
                v = q.max()
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < self.theta:
                break
        self.V = V
        self.policy = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            q = np.zeros(self.nA)
            for a in range(self.nA):
                for prob, s_next, r, term in self.P[s][a]:
                    if term:
                        q[a] += prob * r
                    else:
                        q[a] += prob * (r + self.gamma * V[s_next])
            self.policy[s, np.argmax(q)] = 1.0

    def save(self, path):
        np.savez(path, V=self.V)
