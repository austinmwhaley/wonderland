import numpy as np

from ..base import BaseAgent
from ..tabular.common import n_actions, n_states


class LinearPrediction(BaseAgent):
    """Off-policy linear value prediction on tabular envs.

    Behavior policy is uniform over all actions; the target policy is uniform
    over the first two actions only, so the importance-sampling ratio
    rho = pi(a|s) / b(a|s) is genuinely off-policy (2 or 0).  States are
    encoded as one-hot features, and theta is the linear weight vector.
    """

    family = "value-based"
    policy = "off-policy"
    action_space = "discrete"
    state_space = "tabular"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.nS = n_states(env)
        self.nA = n_actions(env)
        self.theta = np.zeros(self.nS)
        self.alpha = config.get("alpha", 0.05)
        self.gamma = config.get("gamma", 0.99)
        self.t = 0

    def _features(self, s):
        phi = np.zeros(self.nS)
        phi[s] = 1.0
        return phi

    def behavior(self, state):
        return int(self.rng.integers(self.nA))

    def target_prob(self, state, a):
        return 1.0 / 2 if a < 2 else 0.0

    def behavior_prob(self, state, a):
        return 1.0 / self.nA

    def act(self, state, eval=False):
        return self.behavior(state)

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
                a = self.behavior(state)
                ns, r, term, trunc, _ = env.step(a)
                done = bool(term or trunc)
                loss = self.update(state, a, r, ns, done)
                losses.append(loss)
                state = ns
                ret += r
                self.t += 1
            ep += 1
            tracker.log(timestep=self.t, episode=ep, ret=ret,
                        loss=float(np.mean(losses)) if losses else None)
            losses = []

    def save(self, path):
        np.savez(path, theta=self.theta)


class GTD(LinearPrediction):
    """GTD2 (Gradient TD): off-policy correction via a second weight vector w
    that estimates the covariance term, so the update is a true gradient of
    the projected Bellman error (Sutton et al. 2009)."""

    def __init__(self, env, config):
        super().__init__(env, config)
        self.w = np.zeros(self.nS)
        self.alpha2 = config.get("alpha2", self.alpha)

    def update(self, s, a, r, s_next, done):
        rho = self.target_prob(s, a) / self.behavior_prob(s, a)
        phi = self._features(s)
        phi2 = np.zeros(self.nS) if done else self._features(s_next)
        delta = r + self.gamma * self.theta @ phi2 - self.theta @ phi
        self.w += self.alpha2 * (delta * phi - self.gamma * (self.w @ phi2) * phi)
        self.theta += self.alpha * (phi - self.gamma * phi2) * (self.w @ phi)
        return abs(delta)

    def save(self, path):
        np.savez(path, theta=self.theta, w=self.w)


class TrueOnlineTDLambda(LinearPrediction):
    """True Online TD(lambda): a dutch eligibility trace plus the correction
    term alpha*rho*(theta.T*phi2)*(E.T*phi) that makes the update exactly
    equivalent to offline TD(lambda) per step (van Seijen et al. 2014).

    On-policy here: behavior and target are both the uniform random policy,
    so the IS ratio is 1 and the dutch trace stays bounded."""

    def behavior(self, state):
        return int(self.rng.integers(self.nA))

    def target_prob(self, state, a):
        return 1.0 / self.nA

    def behavior_prob(self, state, a):
        return 1.0 / self.nA

    def __init__(self, env, config):
        super().__init__(env, config)
        self.lam = config.get("lambda", 0.9)
        self.E = np.zeros(self.nS)
        self.v_old = 0.0

    def train(self, env, config, tracker):
        self.E[:] = 0.0
        self.v_old = 0.0
        super().train(env, config, tracker)

    def update(self, s, a, r, s_next, done):
        rho = self.target_prob(s, a) / self.behavior_prob(s, a)
        phi = self._features(s)
        phi2 = np.zeros(self.nS) if done else self._features(s_next)
        v_old = self.theta @ phi
        delta = r + self.gamma * (self.theta @ phi2) - v_old
        v_new = self.theta @ phi
        self.E = rho * (self.gamma * self.lam * self.E + phi)
        self.theta += self.alpha * (rho * delta + v_new - v_old) * self.E
        self.theta -= self.alpha * rho * (self.theta @ phi2) * (self.E @ phi)
        return abs(delta)

    def save(self, path):
        np.savez(path, theta=self.theta)


class EmphaticTD(LinearPrediction):
    """Emphatic TD: off-policy correction by re-weighting the TD update with
    an emphasis M_t = lambda * rho_{t-1} * M_{t-1} + 1 (Sutton et al. 2016)."""

    def __init__(self, env, config):
        super().__init__(env, config)
        self.lam = config.get("lambda", 0.9)
        self.M = 0.0

    def train(self, env, config, tracker):
        self.M = 0.0
        super().train(env, config, tracker)

    def update(self, s, a, r, s_next, done):
        rho = self.target_prob(s, a) / self.behavior_prob(s, a)
        phi = self._features(s)
        phi2 = np.zeros(self.nS) if done else self._features(s_next)
        delta = r + self.gamma * (self.theta @ phi2) - self.theta @ phi
        self.M = self.lam * rho * self.M + 1.0
        self.theta += self.alpha * self.M * delta * phi
        return abs(delta)

    def save(self, path):
        np.savez(path, theta=self.theta)