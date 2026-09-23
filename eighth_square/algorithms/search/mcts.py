import numpy as np

from ..base import BaseAgent, evaluate
from ..tabular.common import episode_stats


class EnvModel:
    """Deterministic model of a Gym env obtained by cloning the internal
    state (works for the classic-control environments, whose dynamics are
    deterministic given the state vector)."""

    def __init__(self, env):
        self.env = env.unwrapped
        self.nA = int(env.action_space.n)
        self._stack = []

    def save(self):
        self._stack.append(np.array(self.env.unwrapped.state, copy=True))

    def restore(self):
        self.env.unwrapped.state = np.array(self._stack.pop(), copy=True)
        if hasattr(self.env.unwrapped, "steps_beyond_terminated"):
            self.env.unwrapped.steps_beyond_terminated = 0

    def step(self, a):
        s2, r, term, trunc, _ = self.env.step(a)
        return np.asarray(s2, dtype=np.float32), float(r), bool(term or trunc)


class MCTS:
    """Plain Monte Carlo Tree Search over an EnvModel: UCB1 selection,
    expansion, random rollouts, and mean-return backup."""

    def __init__(self, model, gamma=0.99, c=1.0, iterations=200, max_depth=100, rng=None):
        self.model = model
        self.gamma = gamma
        self.c = c
        self.iterations = iterations
        self.max_depth = max_depth
        self.rng = rng or np.random.default_rng(0)
        self.root = {}

    def _node(self, s):
        if s not in self.root:
            self.root[s] = {"N": np.zeros(self.model.nA), "W": np.zeros(self.model.nA),
                            "children": [None] * self.model.nA}
        return self.root[s]

    def search(self, s):
        self.model.save()
        for _ in range(self.iterations):
            self._simulate(s, 0)
        self.model.restore()
        node = self._node(s)
        N = node["N"]
        best = int(np.argmax(N))
        return best, N / max(N.sum(), 1.0)

    def _simulate(self, s, depth):
        node = self._node(s)
        N, W = node["N"], node["W"]
        total = N.sum()
        if total == 0:
            self.model.save()
            val = self._rollout(s, depth)
            self.model.restore()
            node["N"] = np.ones(self.model.nA)
            node["W"] = np.full(self.model.nA, val)
            return val
        ucb = W / np.maximum(N, 1e-6) + self.c * np.sqrt(np.log(total + 1.0) / (N + 1e-6))
        a = int(np.argmax(ucb))
        self.model.save()
        s2, r, done = self.model.step(a)
        s2 = tuple(s2)
        if done or depth + 1 >= self.max_depth:
            val = r
        else:
            child = node["children"][a]
            if child is None:
                child = self._node(s2)
                node["children"][a] = child
            val = self._simulate(s2, depth + 1)
        self.model.restore()
        val = r + self.gamma * val
        node["N"][a] += 1
        node["W"][a] += val
        return val

    def _rollout(self, s, depth):
        val = 0.0
        g = 1.0
        d = depth
        while d < self.max_depth:
            a = int(self.rng.integers(self.model.nA))
            s, r, done = self.model.step(a)
            val += g * r
            if done:
                break
            g *= self.gamma
            d += 1
        return val


class MCTSAgent(BaseAgent):
    """Model-based planning agent: uses MCTS with the real environment as
    the model (via state cloning), no learning. The tree is rebuilt each
    step and the root action with the highest visit count is played."""

    family = "model-based"
    policy = "on-policy"
    action_space = "discrete"
    state_space = "continuous"

    def __init__(self, env, config):
        super().__init__(env, config)
        self.iterations = config.get("mcts_iterations", 200)
        self.c = config.get("mcts_c", 1.0)
        self.max_depth = config.get("mcts_max_depth", 60)
        self.gamma = config.get("gamma", 0.99)
        self.model = EnvModel(env)

    def act(self, state, eval=True):
        s = tuple(np.asarray(state, dtype=np.float32))
        best, _ = MCTS(self.model, self.gamma, self.c, self.iterations,
                       self.max_depth, self.rng).search(s)
        return best

    def train(self, env, config, tracker):
        ep = 0
        for _ in range(config.get("eval_episodes", 10)):
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
                if t > config.get("max_episode_steps", 10_000):
                    break
            ep += 1
            episode_stats(tracker, t, ep, ret)
        self.episodes = ep

    def save(self, path):
        with open(path, "wb") as f:
            np.savez(f, iterations=self.iterations)

    def load(self, path):
        pass