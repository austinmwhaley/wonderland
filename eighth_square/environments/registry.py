import gymnasium as gym

from .bandit import BanditEnv
from .goal import PointReach
from .multiagent import CooperativeCartPole
from .multi_urban import CooperativeUrban
from .coop_tug import CooperativeTug
from .coop_balance import CooperativeBalanceDeliver
from .wrappers import Tabularize

# NOTE: OFFSET imports are function-local (lazy) on purpose: this registry
# must import anywhere (CI, offline judging) without the OFFSET package
# installed. Only offset_* make_env branches import it.

ENVIRONMENTS = {
    "cooperative_cartpole": {
        "gym_id": None,
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "multi-agent",
        "description": "Two agents control a shared cart; forces sum, reward and state are shared. Cooperative coordination benchmark.",
        "use_cases": ["multi-agent RL", "cooperative coordination", "centralized training"],
        "episode_length": 500,
        "n_agents": 2,
    },
    "point_reach": {
        "gym_id": None,
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "goal-conditioned",
        "description": "1-D point mass with a random per-episode goal and a sparse +1 reach reward. The canonical HER testbed.",
        "use_cases": ["goal-conditioned RL", "hindsight relabeling", "sparse reward learning"],
        "episode_length": 50,
        "n_actions": 3,
    },
    "cartpole": {
        "gym_id": "CartPole-v1",
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "classic-control",
        "description": "Balance a pole on a cart. The canonical first RL problem.",
        "use_cases": ["sanity checks", "debugging agents", "benchmarking"],
        "episode_length": 500,
    },
    "lunar": {
        "gym_id": "LunarLander-v3",
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "classic-control",
        "description": "Land a lander on a landing pad. Rewards shaping for touchdown.",
        "use_cases": ["discrete-action deep RL", "intermediate difficulty"],
        "episode_length": 1000,
    },
    "pendulum": {
        "gym_id": "Pendulum-v1",
        "action_space": "continuous",
        "state_space": "continuous",
        "family": "classic-control",
        "description": "Swing up and balance a pendulum. The canonical continuous-control problem.",
        "use_cases": ["continuous action algorithms", "DDPG/TD3/SAC testing"],
        "episode_length": 200,
    },
    "mountaincar": {
        "gym_id": "MountainCar-v0",
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "classic-control",
        "description": "Drive a car up a hill with an underpowered engine. Sparse reward, needs exploration.",
        "use_cases": ["sparse reward exploration", "on-policy algorithms"],
        "episode_length": 200,
    },
    "mountaincar_continuous": {
        "gym_id": "MountainCarContinuous-v0",
        "action_space": "continuous",
        "state_space": "continuous",
        "family": "classic-control",
        "description": "Continuous-action variant of MountainCar.",
        "use_cases": ["continuous control", "HER-style goals"],
        "episode_length": 1000,
    },
    "acrobot": {
        "gym_id": "Acrobot-v1",
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "classic-control",
        "description": "Swing up an underactuated two-link robot.",
        "use_cases": ["discrete-action deep RL", "benchmarking"],
        "episode_length": 500,
    },
    "cliffwalking": {
        "gym_id": "CliffWalking-v1",
        "action_space": "discrete",
        "state_space": "tabular",
        "family": "gridworld",
        "description": "4x12 grid; stepping into the cliff yields -100. The classic S&B Example 6.6.",
        "use_cases": ["tabular TD methods", "SARSA vs Q-learning comparison"],
        "episode_length": 1000,
    },
    "frozenlake": {
        "gym_id": "FrozenLake-v1",
        "action_space": "discrete",
        "state_space": "tabular",
        "family": "gridworld",
        "description": "Navigate a frozen lake; ice is slippery (stochastic transitions).",
        "use_cases": ["tabular methods", "model-based planning", "stochastic dynamics"],
        "episode_length": 100,
    },
    "blackjack": {
        "gym_id": "Blackjack-v1",
        "action_space": "discrete",
        "state_space": "tabular",
        "family": "cards",
        "description": "Sutton & Barto's blackjack. Observation is a tuple, tabularized automatically.",
        "use_cases": ["tabular MC and TD", "off-policy methods"],
        "episode_length": 100,
    },
    "bandit": {
        "gym_id": None,
        "action_space": "discrete",
        "state_space": "bandit",
        "family": "bandit",
        "description": "10-armed testbed from S&B Ch. 2. One state, n actions, stationary Gaussian rewards.",
        "use_cases": ["bandit algorithms", "exploration vs exploitation"],
        "episode_length": 1,
    },
    "offset_l0": {
        "gym_id": None,
        "action_space": "continuous",
        "state_space": "continuous",
        "family": "offset",
        "description": "OFFSET L0: single drone point-nav continuous in empty 10x10, dense+progress reward. Gate: 95% reach.",
        "use_cases": ["OFFSET ladder L0", "single-agent urban baseline"],
        "episode_length": 150,
    },
    "offset_l0_discrete": {
        "gym_id": None,
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "offset",
        "description": "OFFSET L0 discrete: 5-action (stay/N/S/E/W) point-nav in 10x10. Discrete baseline for DQN.",
        "use_cases": ["OFFSET ladder L0 discrete", "DQN sanity check"],
        "episode_length": 150,
    },
    "offset_l1": {
        "gym_id": None,
        "action_space": "continuous",
        "state_space": "continuous",
        "family": "offset",
        "description": "OFFSET L1: lidar8 + 5 buildings, point-nav. Tests obstacle avoidance.",
        "use_cases": ["OFFSET ladder L1", "partial observability"],
        "episode_length": 150,
    },
    "offset_l1_discrete": {
        "gym_id": None,
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "offset",
        "description": "OFFSET L1 discrete: lidar8 + 5 buildings. DQN baseline.",
        "use_cases": ["OFFSET ladder L1 discrete"],
        "episode_length": 150,
    },
    "offset_l2_discrete": {
        "gym_id": None,
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "offset",
        "description": "OFFSET L2 discrete: sparse reward + random spawn/goal, lidar8 + 5 buildings. Tests credit assignment.",
        "use_cases": ["OFFSET ladder L2 discrete", "sparse reward"],
        "episode_length": 150,
    },
    "offset_l3": {
        "gym_id": None,
        "action_space": "discrete",
        "state_space": "continuous",
        "family": "multi-agent",
        "description": "OFFSET L3: two UAVs carry a balanced payload to a common goal via summed control (CooperativeCartPole-style anti-correlated force). Coordination required; CTDE benchmark.",
        "use_cases": ["OFFSET ladder L3", "cooperative MARL", "CTDE", "summed control"],
        "episode_length": 400,
        "n_agents": 2,
    },
}


def make_env(name, seed=None):
    meta = ENVIRONMENTS[name]
    if name == "bandit":
        env = BanditEnv(seed=seed)
        if seed is not None:
            env.reset(seed=seed)
        return env
    if name == "cooperative_cartpole":
        env = CooperativeCartPole(seed=seed)
        env.reset(seed=seed)
        return env
    if name == "point_reach":
        env = PointReach(seed=seed)
        env.reset(seed=seed)
        return env
    if name == "offset_l0":
        from OFFSET.environments.urban_world import UrbanWorld
        env = UrbanWorld(n_agents=1, buildings=[], world_size=10, world_height=10, max_episode_steps=meta["episode_length"], seed=seed, discrete=False)
        env.reset(seed=seed)
        return env
    if name == "offset_l0_discrete":
        from OFFSET.environments.urban_world import UrbanWorld
        env = UrbanWorld(n_agents=1, buildings=[], world_size=10, world_height=10, max_episode_steps=meta["episode_length"], seed=seed, discrete=True)
        env.reset(seed=seed)
        return env
    if name == "offset_l1":
        from OFFSET.environments.urban_world import L1_BUILDINGS
        env = UrbanWorld(n_agents=1, buildings=L1_BUILDINGS, world_size=10, world_height=10, max_episode_steps=meta["episode_length"], seed=seed, discrete=False, obs_mode="lidar8")
        env.reset(seed=seed)
        return env
    if name == "offset_l1_discrete":
        from OFFSET.environments.urban_world import L1_BUILDINGS
        env = UrbanWorld(n_agents=1, buildings=L1_BUILDINGS, world_size=10, world_height=10, max_episode_steps=meta["episode_length"], seed=seed, discrete=True, obs_mode="lidar8")
        env.reset(seed=seed)
        return env
    if name == "offset_l2_discrete":
        from OFFSET.environments.urban_world import L1_BUILDINGS
        env = UrbanWorld(n_agents=1, buildings=L1_BUILDINGS, world_size=10, world_height=10, max_episode_steps=meta["episode_length"], seed=seed, discrete=True, obs_mode="lidar8", reward_mode="sparse")
        env.reset(seed=seed)
        return env
    if name == "offset_l3":
        env = CooperativeBalanceDeliver(seed=seed, world_width=10.0)
        env.reset(seed=seed)
        return env
    try:
        env = gym.make(meta["gym_id"])
    except gym.error.DeprecatedEnv as e:
        alt = meta["gym_id"].rsplit("-", 1)[0] + "-v1"
        env = gym.make(alt)
    if name == "blackjack":
        env = Tabularize(env)
    if seed is not None:
        env.reset(seed=seed)
    return env