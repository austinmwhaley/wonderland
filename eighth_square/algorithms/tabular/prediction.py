import numpy as np

from .common import TabularAgent, episode_stats, n_actions, n_states


class TDPrediction(TabularAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.V = np.zeros(self.nS)
        self.alpha = config.get("alpha", 0.05)
        self.gamma = config.get("gamma", 0.99)
        self.t = 0

    def policy(self, state):
        return int(self.rng.integers(self.nA))

    def act(self, state, eval=False):
        return self.policy(state)

    def update(self, s, r, s_next, done):
        raise NotImplementedError

    def train(self, env, config, tracker):
        self.t = 0
        ep = 0
        losses = []
        while self.t < config["steps"]:
            state, _ = env.reset()
            done = False
            ret = 0.0
            while not done and self.t < config["steps"]:
                a = self.policy(state)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                loss = self.update(state, r, ns, done)
                losses.append(loss)
                state = ns
                ret += r
                self.t += 1
            ep += 1
            episode_stats(tracker, self.t, ep, ret, np.mean(losses) if losses else None)
            losses = []

    def save(self, path):
        np.savez(path, V=self.V)


class TD0Prediction(TDPrediction):
    def update(self, s, r, s_next, done):
        target = r + (0.0 if done else self.gamma * self.V[s_next])
        self.V[s] += self.alpha * (target - self.V[s])
        return abs(target - self.V[s])


class TDLambdaPrediction(TDPrediction):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.lam = config.get("lambda", 0.9)
        self.trace = config.get("trace", "accumulating")
        self.E = np.zeros(self.nS)

    def update(self, s, r, s_next, done):
        self.E[s] += 1.0
        target = r + (0.0 if done else self.gamma * self.V[s_next])
        delta = target - self.V[s]
        self.V += self.alpha * delta * self.E
        self.E *= self.gamma * self.lam
        return abs(delta)

    def train(self, env, config, tracker):
        self.E[:] = 0.0
        super().train(env, config, tracker)

    def save(self, path):
        np.savez(path, V=self.V)