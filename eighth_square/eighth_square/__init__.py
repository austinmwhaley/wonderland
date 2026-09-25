"""Eighth Square — a standalone PyTorch reinforcement-learning library.

Algorithm families (all importable packages):

    algorithms.tabular    Q-learning, SARSA, Expected SARSA, Dyna-Q, MC, TD(λ),
                          n-step, eligibility traces, planning  (+ the ES
                          ``train_*`` tabular trainers in ``tabular_es``)
    algorithms.deep       DQN, Double/Dueling/Prioritized/QR/Rainbow, C51,
                          R2D2, Agent57, DDPG, TD3, SAC, HER, ICM, RND
    algorithms.approx     REINFORCE, A2C, PPO, IMPALA, ACER, ACKTR, TRPO,
                          actor-critic, semi-gradient SARSA, off-policy TD
    algorithms.offline    IQL, CQL, continuous IQL, Decision Transformer
    algorithms.bandits    ε-greedy, UCB, optimistic, gradient, Thompson
    algorithms.multiagent (MAPPO, QMIX, MADDPG, multi-agent IQL)
    algorithms.search     MCTS, AlphaZero, MuZero, Dreamer, world models

Usage::

    import eighth_square as es

    es.families()                 # ['tabular', 'deep', ...]
    from eighth_square import deep, offline
    agent = deep.DQN(env, config)          # PyTorch implementation
    agent = offline.IQL(env, config)

The distribution also installs the top-level packages directly, so the legacy
imports keep working::

    from algorithms.deep.dqn import DQN
    from algorithms.offline.iql import IQL
    from algorithms.registry import ALGORITHMS
    from algorithms.registry import ALGO_REGISTRY   # the comparison trainers
"""

from importlib import import_module

__version__ = "0.1.0"

# The algorithm families exposed at the top level.
_FAMILIES = ("tabular", "deep", "approx", "offline", "bandits", "multiagent", "search")
# Other public packages.
_PACKAGES = (
    "algorithms",
    "environments",
    "core",
    "runners",
    "plotting",
    "tracking",
    "visualization",
    "experiments",
)


def families():
    """Return the list of algorithm families in the library."""
    return list(_FAMILIES)


def __getattr__(name):
    # Lazy attribute access so `import eighth_square` stays cheap and only the
    # family you touch pulls in torch.
    if name in _FAMILIES:
        return import_module(f"algorithms.{name}")
    if name in _PACKAGES:
        return import_module(name)
    raise AttributeError(f"module 'eighth_square' has no attribute {name!r}")


def __dir__():
    return sorted(list(_FAMILIES) + list(_PACKAGES) + ["families", "__version__"])
