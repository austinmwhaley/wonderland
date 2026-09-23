import numpy as np

from .base import BaseAgent, EpsilonScheduler


class BanditAgent(BaseAgent):
    family = "bandit"
    action_space = "discrete"
    state_space = "bandit"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.nA = env.n_arms
        self.Q = np.zeros(self.nA)
        self.N = np.zeros(self.nA)
        self.t = 0

    def act(self, state, eval=False):
        raise NotImplementedError

    def train(self, env, config, tracker):
        self.t = 0
        optimal = float(np.max(env.means))
        while self.t < config["steps"]:
            a = self.act(0)
            _, r, _, _, _ = env.step(a)
            self.t += 1
            self.update(a, r)
            tracker.log(
                timestep=self.t,
                episode=self.t,
                ret=r,
                regret=optimal - r,
                loss=None,
            )

    def update(self, a, r):
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]

    def save(self, path):
        np.savez(path, Q=self.Q, N=self.N)


class EpsilonGreedyBandit(BanditAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.sched = EpsilonScheduler(
            config.get("eps_start", 0.1),
            config.get("eps_end", 0.01),
            config.get("eps_decay_steps", config.get("steps", 5_000)),
            self.rng,
        )

    def act(self, state, eval=False):
        if self.rng.random() < self.sched.epsilon(self.t):
            return int(self.rng.integers(self.nA))
        return int(np.argmax(self.Q))


class OptimisticBandit(BanditAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.Q[:] = config.get("optimistic_init", 5.0)

    def act(self, state, eval=False):
        return int(np.argmax(self.Q))


class UCB(BanditAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.c = config.get("c", 2.0)

    def act(self, state, eval=False):
        with np.errstate(divide="ignore"):
            bonus = self.c * np.sqrt(np.log(self.t + 1) / np.maximum(self.N, 1e-8))
        return int(np.argmax(self.Q + bonus))


class GradientBandit(BanditAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.alpha = config.get("alpha", 0.1)
        self.H = np.zeros(self.nA)
        self.mean_reward = 0.0

    def act(self, state, eval=False):
        pi = np.exp(self.H - self.H.max())
        pi /= pi.sum()
        return int(self.rng.choice(self.nA, p=pi))

    def update(self, a, r):
        self.N[a] += 1
        self.mean_reward += (r - self.mean_reward) / self.N.sum()
        pi = np.exp(self.H - self.H.max())
        pi /= pi.sum()
        baseline = self.mean_reward
        self.H[a] += self.alpha * (r - baseline) * (1 - pi[a])
        self.H -= self.alpha * (r - baseline) * pi

    def save(self, path):
        np.savez(path, H=self.H, mean_reward=self.mean_reward)


class ThompsonSampling(BanditAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.mu = np.zeros(self.nA)
        self.tau = np.full(self.nA, 1.0)

    def act(self, state, eval=False):
        samples = self.rng.normal(self.mu, 1.0 / np.sqrt(self.tau))
        return int(np.argmax(samples))

    def update(self, a, r):
        self.N[a] += 1
        self.tau[a] += 1.0
        self.mu[a] = (self.mu[a] * (self.tau[a] - 1) + r) / self.tau[a]

    def save(self, path):
        np.savez(path, mu=self.mu, tau=self.tau)