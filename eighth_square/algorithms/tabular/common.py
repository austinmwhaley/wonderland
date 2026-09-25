import numpy as np

from ..base import BaseAgent, EpsilonScheduler


def n_states(env):
    return int(env.observation_space.n)


def n_actions(env):
    return int(env.action_space.n)


_tie_rng = np.random.default_rng()


def greedy_action(Q, state, rng=None):
    q = Q[state]
    best = np.flatnonzero(q == q.max())
    if len(best) == 1:
        return int(best[0])
    rng = rng if rng is not None else _tie_rng
    return int(best[rng.integers(len(best))])


def eps_greedy_action(Q, state, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return greedy_action(Q, state, rng)


def episode_stats(tracker, t, ep, ret, loss=None):
    row = {"timestep": t, "episode": ep, "return": ret}
    if loss is not None:
        row["loss"] = loss
    tracker.log(**row)


class TabularAgent(BaseAgent):
    action_space = "discrete"
    state_space = "tabular"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.nS = n_states(env)
        self.nA = n_actions(env)
        self.Q = np.zeros((self.nS, self.nA))
        self.eps_sched = EpsilonScheduler(
            config.get("eps_start", 0.1),
            config.get("eps_end", 0.01),
            config.get("eps_decay_steps", config.get("steps", 50_000)),
            self.rng,
        )

    def act(self, state, eval=False):
        if eval:
            return greedy_action(self.Q, state, self.rng)
        return eps_greedy_action(self.Q, state, self.eps_sched.epsilon(self.t), self.rng)

    def select(self, state):
        return self.act(state)

    def save(self, path):
        np.savez(path, Q=self.Q, eps=self.eps_sched.epsilon(self.t))

    def load(self, path):
        self.Q = np.load(path)["Q"]
