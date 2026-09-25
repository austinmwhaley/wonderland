# Eighth Square

A **standalone PyTorch reinforcement-learning library**. Point your projects at it
and import clean implementations of classic and modern RL algorithms across all
major families — tabular, deep, approximate, offline, bandits, multi-agent, and
search.

## Install

From the project root (editable):

```bash
pip install -e .
# optional: Box2D environments (LunarLander, BipedalWalker) and test deps
pip install -e ".[box2d,dev]"
```

Requires Python >= 3.10, PyTorch, Gymnasium, NumPy, Matplotlib, tqdm.

## Use it as a library

```python
import eighth_square as es

es.families()  # ['tabular', 'deep', 'approx', 'offline',
#  'bandits', 'multiagent', 'search']
from eighth_square import deep, offline, approx, tabular, bandits

agent = deep.DQN(env, config)  # PyTorch implementation
agent = deep.SAC(env, config)  # continuous control
agent = approx.PPO(env, config)
agent = offline.IQL(env, config)  # offline RL
agent = tabular.QLearning(env, config)
```

Legacy top-level imports also work (the packages are installed directly):

```python
from algorithms.deep.dqn import DQN
from algorithms.offline.iql import IQL
from algorithms.approx.ppo import PPO
from algorithms.registry import ALGORITHMS  # full class catalogue (69)
from algorithms.registry import ALGO_REGISTRY  # comparison trainers (20)
```

Every agent follows the same protocol: `__init__(env, config)`,
`act(state, eval=False)`, `train(env, config, tracker)`, `save(path)`,
`load(path)`.

## Algorithms

### `algorithms.tabular`
Q-Learning, DoubleQ-Learning, SARSA, Expected SARSA, n-step SARSA / Expected
SARSA / Tree Backup / off-policy, SARSA(λ), Q(λ), MC prediction/control
(on/off-policy), TD(0), TD(λ), policy evaluation/iteration, value iteration,
Dyna-Q, Dyna-Q+, Prioritized Sweeping.

### `algorithms.deep` (PyTorch)
DQN, Double DQN, Dueling DQN, Prioritized DQN, C51, QR-DQN, Rainbow,
Rainbow+HER, HER-DQN, R2D2, Agent57, DDPG, TD3, SAC, ICM, RND
(+ `networks`, `replay` buffers).

### `algorithms.approx` (PyTorch)
REINFORCE, Actor-Critic, A2C, PPO, IMPALA, ACER, ACKTR, TRPO, semi-gradient
SARSA, naive Q-learning, GTD, True Online TD(λ), Emphatic TD.

### `algorithms.offline` (PyTorch)
IQL, CQL, Continuous IQL, Decision Transformer.

### `algorithms.bandits`
ε-greedy, Optimistic, UCB, Gradient, Thompson Sampling.

### `algorithms.multiagent`
MAPPO, QMIX, MADDPG, multi-agent IQL.

### `algorithms.search` (model-based / planning)
MCTS, AlphaZero, MuZero, Dreamer, World Models (RSSM, VAE+MDN-RNN).

### Comparison trainers
`algorithms.registry.ALGO_REGISTRY` (20 names) maps the classic
`core.config`-driven trainers used by the comparison runner:
`q_learning, sarsa, expected_sarsa, dyna_q, vanilla_dqn, dqn, double_dqn,
dueling_dqn, prioritized_dqn, qr_dqn, rainbow_dqn, reinforce, a2c, ppo, grpo,
trpo, ddpg, td3, sac, cem`.

## Environments

Custom environments + a registry live in `environments/` (bandits, gridworlds,
goal-conditioned, multi-agent, cooperative tasks, wrappers). Gymnasium
environments are used directly.

## Comparison framework (included)

`main.py` is a CLI to run and compare the 20 comparison trainers across
environments with plateau stopping, sweeps, variants, parallel workers, and
matplotlib reports:

```bash
eighth-square --list-algos
python main.py --envs CartPole-v1 --max-episodes 5000
python main.py --sweep "dqn:lr=[1e-4,3e-4,1e-3]" --workers 4
```

See `README_comparison_framework.md` for details.

## Layout

```
eighth_square/         library facade (import eighth_square as es)
algorithms/
  tabular/  deep/  approx/  offline/  bandits.py  multiagent/  search/
  registry.py  base.py  stubs.py
environments/          custom envs + registry
core/                  config, networks, replay buffers, trainer utilities
runners/  plotting/  tracking/  visualization/  experiments/
main.py                comparison CLI
```

## Status

Standalone and installable (`pip install -e .`). Importable from any project:
`import eighth_square`. Both import surfaces are supported and verified:
`eighth_square.<family>` and the direct `algorithms.<family>` packages.
