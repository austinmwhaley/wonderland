"""Synthetic logged datasets for agnostic-input testing.

Real deployments ingest logs of unknown origin. These generators produce
data with KNOWN structure (so tests can assert correct behavior) in the two
shapes the library must handle:

- make_bandit: a contextual bandit (human-email-like) — one decision per row,
  context -> action by an unknown logging policy, reward depends on action.
  Propensities may be returned (exact) or stripped (estimated path).

- make_sequential: a tiny deterministic MDP with episodes, to exercise the
  sequential/RL path.

Policies:
- SoftmaxPolicy: pi(action|context) = softmax(W x). The candidate the library
  evaluates.
- GreedyPolicy: wraps a SoftmaxPolicy to ship its argmax.
"""

from __future__ import annotations

import numpy as np


class SoftmaxPolicy:
    """Log-linear stochastic policy over a discrete action space."""

    def __init__(self, W, nA=None):
        self.W = np.asarray(W, dtype=np.float64)  # (nA, d)
        self.nA = int(nA or self.W.shape[0])

    def _logits(self, obs):
        o = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        return o @ self.W.T

    def action_probs(self, obs, temperature=1.0):
        z = self._logits(obs) / max(float(temperature), 1e-9)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return (p / p.sum(1, keepdims=True)).astype(np.float32)

    def act(self, state, eval=True):
        return int(np.argmax(self.action_probs(np.asarray(state)[None, :])[0]))


class GreedyPolicy(SoftmaxPolicy):
    def action_probs(self, obs, temperature=1.0):
        p = super().action_probs(obs, temperature=1.0)
        out = np.zeros_like(p)
        out[np.arange(len(p)), p.argmax(1)] = 1.0
        return out


def make_bandit(n=4000, d=6, nA=4, seed=0, reward_scale=1.0,
                logging_temp=1.0, include_propensity=True, noise=0.5,
                behavior_W=None, reward_W=None):
    """Contextual-bandit log with a KNOWN logging policy and reward model.

    Returns (canonical_dict, info) where info carries the logging policy, the
    reward model, and the best action per row (for assertions).
    """
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, d)).astype(np.float32)
    behavior_W = (rng.normal(size=(nA, d)) * 0.7 if behavior_W is None
                  else np.asarray(behavior_W))
    reward_W = (rng.normal(size=(nA, d)) if reward_W is None
                else np.asarray(reward_W))
    # logging policy: softmax over behavior_W . x, with temperature.
    beh = SoftmaxPolicy(behavior_W / max(logging_temp, 1e-9), nA)
    probs = beh.action_probs(obs)
    act = np.array([rng.choice(nA, p=p) for p in probs])
    # reward: action-specific linear payoff + noise (unknown to the library).
    mu_r = (obs @ reward_W.T)[np.arange(n), act] * reward_scale
    rew = (mu_r + rng.normal(0, noise, n)).astype(np.float32)
    best_a = (obs @ reward_W.T).argmax(1)

    canon = {"obs": obs, "act": act.astype(np.int64), "rew": rew}
    if include_propensity:
        canon["propensity"] = probs[np.arange(n), act]
    info = {"logging_W": behavior_W, "reward_W": reward_W,
            "best_action": best_a, "nA": nA,
            "best_policy": GreedyPolicy(reward_W, nA),
            "logging_policy": beh,
            "mean_reward_behavior": float(rew.mean()),
            "mean_reward_best": float((obs @ reward_W.T).max(1).mean())}
    return canon, info


class OptimalActionPolicy:
    """Deterministic a* = argmax(state[:nA]) policy (the target in
    make_sequential)."""

    def __init__(self, nA):
        self.nA = int(nA)

    def act(self, state, eval=True):
        return int(np.argmax(np.asarray(state)[:self.nA]))

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs))
        a = o[:, :self.nA].argmax(1)
        out = np.zeros((len(o), self.nA), dtype=np.float32)
        out[np.arange(len(o)), a] = 1.0
        return out


def make_sequential(n=6000, d=6, nA=4, T=15, seed=0, behavior_eps=0.4,
                    noise=0.05):
    """Sequential RL log with a KNOWN better policy (the real target shape).

    Optimal action a* = argmax(state[:nA]); reward 1 iff the chosen action is
    a*. Behavior is a* with prob (1-eps) else uniform — so the log contains all
    actions (coverage) but is suboptimal. A good offline-RL policy can recover
    a* and beat behavior; OPE should certify that. Returns (canonical, info).
    """
    rng = np.random.default_rng(seed)
    obs, act, rew, done = [], [], [], []
    n_ep = max(n // T, 2)
    for e in range(n_ep):
        s = rng.normal(size=d).astype(np.float32)
        for tstep in range(T):
            a_star = int(np.argmax(s[:nA]))
            if rng.random() < behavior_eps:
                a = int(rng.integers(nA))
            else:
                a = a_star
            r = 1.0 if a == a_star else 0.0
            obs.append(s.copy())
            act.append(a)
            rew.append(r)
            done.append(1.0 if tstep == T - 1 else 0.0)
            s = (s + noise * rng.normal(size=d)).astype(np.float32)
    canon = {"obs": np.array(obs, dtype=np.float32),
             "act": np.array(act, dtype=np.int64),
             "rew": np.array(rew, dtype=np.float32),
             "done": np.array(done, dtype=np.float32)}
    info = {"nA": nA, "T": T, "optimal_policy": OptimalActionPolicy(nA),
            "behavior_eps": behavior_eps,
            "optimal_step_reward": 1.0,
            "behavior_step_reward": (1 - behavior_eps) + behavior_eps / nA}
    return canon, info



class GaussianPolicy:
    """Diagonal-Gaussian continuous policy: a ~ N(W s, sigma^2). Exposes the
    continuous candidate protocol (action_mean, log_prob_fn, act)."""

    def __init__(self, W, sigma=0.5, nA=None):
        self.W = np.asarray(W, dtype=np.float64)  # (a_dim, d)
        self.sigma = float(sigma)

    def action_mean(self, obs):
        o = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        return o @ self.W.T

    def log_prob_fn(self, obs, act):
        o = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        a = np.atleast_2d(np.asarray(act, dtype=np.float64))
        mean = o @ self.W.T
        z = (a - mean) / self.sigma
        return (-0.5 * z ** 2 - np.log(self.sigma)
                - 0.5 * float(np.log(2 * np.pi))).sum(1)

    def act(self, state, eval=True):
        return (self.action_mean(np.asarray(state)[None, :])[0])

    def sample(self, obs, rng):
        return self.action_mean(obs) + self.sigma * rng.normal(size=self.action_mean(obs).shape)


def make_continuous(n=4000, d=5, a_dim=2, T=15, seed=0, sigma_behavior=1.0,
                    include_logp=True, sequential=True):
    """Sequential continuous-action log with a KNOWN better policy.

    Optimal action a* = W* s; reward = -||a - a*||^2 (so a* earns ~0, a
    mismatched policy earns negative). Behavior is a DIFFERENT Gaussian
    (mean = W_b s, sigma_behavior), with its exact log-density recorded.
    """
    rng = np.random.default_rng(seed)
    Wstar = rng.normal(size=(a_dim, d)) * 1.5
    Wb = rng.normal(size=(a_dim, d)) * 0.5   # suboptimal behavior
    best = GaussianPolicy(Wstar, sigma_behavior)  # same variance: ratios identify the mean shift
    beh = GaussianPolicy(Wb, sigma_behavior)
    obs, act, rew, done, logp = [], [], [], [], []
    n_ep = max(n // T, 2)
    for e in range(n_ep):
        s = rng.normal(size=d).astype(np.float32)
        for tstep in range(T):
            mean = s @ Wb.T
            a = mean + sigma_behavior * rng.normal(size=a_dim)
            star = s @ Wstar.T
            r = -float(np.sum((a - star) ** 2))
            obs.append(s.copy()); act.append(a.astype(np.float32))
            rew.append(r); done.append(1.0 if tstep == T - 1 else 0.0)
            if include_logp:
                z = (a - mean) / sigma_behavior
                logp.append(float(-0.5 * np.sum(z ** 2) - a_dim *
                                  np.log(sigma_behavior)
                                  - 0.5 * a_dim * np.log(2 * np.pi)))
            s = (s + 0.05 * rng.normal(size=d)).astype(np.float32)
    canon = {"obs": np.array(obs, dtype=np.float32),
             "act": np.array(act, dtype=np.float32),
             "rew": np.array(rew, dtype=np.float32)}
    if sequential:
        canon["done"] = np.array(done, dtype=np.float32)
    if include_logp:
        canon["log_prob"] = np.array(logp, dtype=np.float32)
    info = {"Wstar": Wstar, "Wb": Wb, "a_dim": a_dim,
            "optimal_policy": best, "behavior_policy": beh,
            "optimal_step_reward": 0.0}
    return canon, info
