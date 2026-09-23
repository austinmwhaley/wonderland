from typing import Callable

from algorithms.base import Result
from algorithms.cem import train_cem
from algorithms.continuous import train_ddpg, train_sac, train_td3
from algorithms.dqn import train_double_dqn, train_dqn, train_dueling_dqn
from algorithms.dyna_q import train_dyna_q
from algorithms.grpo import train_grpo
from algorithms.policy_gradient import train_a2c, train_ppo, train_reinforce
from algorithms.prioritized_dqn import train_prioritized_dqn
from algorithms.qr_dqn import train_qr_dqn
from algorithms.rainbow_dqn import train_rainbow_dqn
from algorithms.tabular_es import train_expected_sarsa, train_q_learning, train_sarsa
from algorithms.trpo import train_trpo
from algorithms.vanilla_dqn import train_vanilla_dqn
from core.config import AlgorithmConfig
from environments.base import EnvWrapper

ALGO_REGISTRY: dict[str, Callable[[EnvWrapper, AlgorithmConfig], Result]] = {
    "q_learning": train_q_learning,
    "sarsa": train_sarsa,
    "expected_sarsa": train_expected_sarsa,
    "dyna_q": train_dyna_q,
    "vanilla_dqn": train_vanilla_dqn,
    "dqn": train_dqn,
    "double_dqn": train_double_dqn,
    "dueling_dqn": train_dueling_dqn,
    "prioritized_dqn": train_prioritized_dqn,
    "qr_dqn": train_qr_dqn,
    "rainbow_dqn": train_rainbow_dqn,
    "reinforce": train_reinforce,
    "a2c": train_a2c,
    "ppo": train_ppo,
    "grpo": train_grpo,
    "trpo": train_trpo,
    "ddpg": train_ddpg,
    "td3": train_td3,
    "sac": train_sac,
    "cem": train_cem,
}

TABULAR_ALGOS = {"q_learning", "sarsa", "expected_sarsa", "dyna_q"}
CONTINUOUS_ALGOS = {"ddpg", "td3", "sac"}
DISCRETE_ALGOS = ALGO_REGISTRY.keys() - CONTINUOUS_ALGOS


def get_algorithm(name: str) -> Callable:
    if name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGO_REGISTRY.keys())}")
    return ALGO_REGISTRY[name]


def list_algorithms() -> list:
    return list(ALGO_REGISTRY.keys())


# ==================== merged from reinforcement_learning ====================

from .bandits import (
    BanditAgent, EpsilonGreedyBandit, OptimisticBandit, UCB,
    GradientBandit, ThompsonSampling,
)
from .tabular.mc import MCPrediction, MCControlOnPolicy, MCControlOffPolicy
from .tabular.prediction import TD0Prediction, TDLambdaPrediction
from .tabular.td import Sarsa, ExpectedSarsa, QLearning, DoubleQLearning
from .tabular.nstep import NStepSarsa, NStepExpectedSarsa, TreeBackup, NStepOffPolicy
from .tabular.traces import SarsaLambda, QLambda
from .tabular.dyna import DynaQ, DynaQPlus, PrioritizedSweeping
from .tabular.planning import PolicyEvaluation, PolicyIteration, ValueIteration
from .approx.sarsa import SemiGradientSarsa, NaiveQLearning
from .approx.reinforce import Reinforce
from .approx.actor_critic import ActorCritic
from .approx.a2c import A2C
from .approx.impala import IMPALA
from .approx.acer import ACER
from .approx.acktr import ACKTR
from .approx.ppo import PPO
from .approx.trpo import TRPO
from .approx.offpolicy_td import GTD, TrueOnlineTDLambda, EmphaticTD
from .deep.dqn import DQN, DoubleDQN, DuelingDQN, PrioritizedDQN, C51, QRDQN, Rainbow, HERDQN, RainbowHER
from .deep.icm import ICM
from .deep.rnd import RND
from .deep.r2d2 import R2D2
from .deep.agent57 import Agent57
from .deep.her import HERAgent
from .offline.cql import CQL
from .offline.iql import IQL
from .offline.decision_transformer import DecisionTransformer
from .deep.ddpg import DDPG
from .deep.td3 import TD3
from .deep.sac import SAC
from .search.mcts import MCTSAgent
from .search.alphazero import AlphaZeroAgent
from .search.muzero import MuZeroAgent
from .search.world_models import WorldModelsAgent
from .search.dreamer import DreamerAgent
from .multiagent.maddpg import MADDPGAgent
from .multiagent.mappo import MAPPOAgent
from .multiagent.qmix import QMIXAgent
from .multiagent.iql import IQLAgent
from .stubs import STUBS, make_stub

TABULAR_ENVS = ["cliffwalking", "frozenlake", "blackjack", "bandit"]
PLANNING_ENVS = ["frozenlake", "cliffwalking"]
CLASSIC_ENVS = ["cartpole", "lunar", "mountaincar", "acrobot", "offset_l0_discrete", "offset_l1_discrete", "offset_l2_discrete"]
STATE_ENVS = ["cartpole", "mountaincar", "acrobot"]  # expose env.state for one-ply lookahead (not box2d)
MARL_ENVS = ["cooperative_cartpole", "offset_l3"]
CONTINUOUS_ENVS = ["pendulum", "mountaincar_continuous", "offset_l0", "offset_l1"]
BOTH_ENVS = CLASSIC_ENVS + CONTINUOUS_ENVS

S_B = "http://incompleteideas.net/book/RLbook2020.pdf"

PAPERS = {
    "epsilon_greedy_bandit": S_B,
    "optimistic_bandit": S_B,
    "ucb_bandit": "https://link.springer.com/article/10.1023/A:1013689704352",
    "gradient_bandit": S_B,
    "thompson_sampling": "https://doi.org/10.2307/2332286",
    "policy_evaluation": "https://www.rand.org/pubs/papers/P534.html",
    "policy_iteration": "https://www.rand.org/pubs/papers/P534.html",
    "value_iteration": "https://www.rand.org/pubs/papers/P534.html",
    "mc_prediction": S_B,
    "mc_control_onpolicy": S_B,
    "mc_control_offpolicy": S_B,
    "td0_prediction": "http://incompleteideas.net/papers/sutton-88-with-erratum.pdf",
    "td_lambda_prediction": "https://link.springer.com/article/10.1007/BF00115009",
    "sarsa": S_B,
    "expected_sarsa": "https://www.marcgbellemare.com/papers/expected_sarsa.pdf",
    "nstep_sarsa": S_B,
    "nstep_expected_sarsa": S_B,
    "tree_backup": "https://www.cs.ualberta.ca/~sutton/papers/PSD00.pdf",
    "nstep_off_policy": S_B,
    "qlearning": "https://link.springer.com/article/10.1007/BF00992698",
    "double_qlearning": "https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html",
    "sarsa_lambda": S_B,
    "q_lambda": S_B,
    "dyna_q": "http://incompleteideas.net/papers/1990-dyna.pdf",
    "dyna_q_plus": S_B,
    "prioritized_sweeping": "https://link.springer.com/article/10.1007/BF00993104",
    "semi_gradient_sarsa": S_B,
    "naive_qlearning": S_B,
    "reinforce": "https://link.springer.com/article/10.1007/BF00992696",
    "reinforce_baseline": S_B,
    "actor_critic": "https://proceedings.neurips.cc/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    "a2c": "https://arxiv.org/abs/1602.01783",
    "impala": "https://arxiv.org/abs/1802.01561",
    "acer": "https://arxiv.org/abs/1611.01224",
    "acktr": "https://arxiv.org/abs/1708.05144",
    "trpo": "https://arxiv.org/abs/1502.05477",
    "ppo": "https://arxiv.org/abs/1707.06347",
    "gtd": "https://arxiv.org/abs/1106.2434",
    "tde": "https://proceedings.mlr.press/v32/seijen14.html",
    "emphatic_td": "https://arxiv.org/abs/1503.04209",
    "dqn": "https://www.nature.com/articles/nature14236",
    "double_dqn": "https://arxiv.org/abs/1509.06461",
    "dueling_dqn": "https://arxiv.org/abs/1511.06581",
    "prioritized_dqn": "https://arxiv.org/abs/1511.05952",
    "c51": "https://arxiv.org/abs/1707.06887",
    "qr_dqn": "https://arxiv.org/abs/1710.10044",
    "rainbow": "https://arxiv.org/abs/1710.02298",
    "icm": "https://arxiv.org/abs/1705.05363",
    "rnd": "https://arxiv.org/abs/1810.12894",
    "r2d2": "https://arxiv.org/abs/1905.09262",
    "agent57": "https://arxiv.org/abs/2003.13350",
    "cql": "https://arxiv.org/abs/2006.04779",
    "iql": "https://arxiv.org/abs/2110.06169",
    "decision_transformer": "https://arxiv.org/abs/2106.01345",
    "ddpg": "https://arxiv.org/abs/1509.02971",
    "td3": "https://arxiv.org/abs/1802.09477",
    "sac": "https://arxiv.org/abs/1801.01290",
    "mcts": "https://link.springer.com/chapter/10.1007/11871842_29",
    "alphazero": "https://arxiv.org/abs/1712.01815",
    "muzero": "https://www.nature.com/articles/s41586-020-03051-4",
    "world_models": "https://arxiv.org/abs/1803.10122",
    "dreamer": "https://arxiv.org/abs/1912.01603",
    "maddpg": "https://arxiv.org/abs/1706.02275",
    "mappo": "https://arxiv.org/abs/2103.01955",
    "qmix": "https://arxiv.org/abs/1803.11485",
    "her": "https://arxiv.org/abs/1707.01495",
}


def _entry(cls, source, family, policy, action_space, state_space, description,
           use_cases, compatible_envs, notes, status="implemented"):
    return {
        "class": cls, "status": status, "source": source, "family": family,
        "policy": policy, "action_space": action_space, "state_space": state_space,
        "description": description, "use_cases": use_cases,
        "compatible_envs": compatible_envs, "notes": notes,
    }


ALGORITHMS = {
    "epsilon_greedy_bandit": _entry(
        EpsilonGreedyBandit, "Sutton & Barto Ch. 2", "bandit", "none",
        "discrete", "bandit",
        "Sample-average action values with epsilon-greedy action selection.",
        ["exploration vs exploitation", "baseline bandit behavior"], ["bandit"],
        "The default baseline every other bandit algorithm is compared against."),
    "optimistic_bandit": _entry(
        OptimisticBandit, "Sutton & Barto Ch. 2", "bandit", "none",
        "discrete", "bandit",
        "Greedy action selection with optimistically initialized Q values.",
        ["exploration via optimism", "evaluating exploration mechanisms"], ["bandit"],
        "Simple and effective: initial optimism alone drives exploration."),
    "ucb_bandit": _entry(
        UCB, "Sutton & Barto Ch. 2", "bandit", "none",
        "discrete", "bandit",
        "Upper-confidence-bound action selection: Q + c * sqrt(ln t / N).",
        ["principled exploration", "regret minimization"], ["bandit"],
        "Selects actions by uncertainty bonus; no explicit epsilon needed."),
    "gradient_bandit": _entry(
        GradientBandit, "Sutton & Barto Ch. 2", "bandit", "none",
        "discrete", "bandit",
        "Softmax action preferences learned via stochastic gradient ascent with baseline.",
        ["preference-based selection", "studying baselines"], ["bandit"],
        "Uses average reward as baseline; sensitive to step size."),
    "thompson_sampling": _entry(
        ThompsonSampling, "Thompson 1933", "bandit", "none",
        "discrete", "bandit",
        "Sample actions from a posterior over arm means (Gaussian posterior).",
        ["Bayesian exploration", "regret minimization"], ["bandit"],
        "Modern addition; the gold standard for bandit exploration."),

    "policy_evaluation": _entry(
        PolicyEvaluation, "Sutton & Barto Ch. 4", "planning", "none",
        "discrete", "tabular",
        "Iterative DP policy evaluation on the environment model (.P).",
        ["computing value functions", "model-based planning", "FrozenLake"], PLANNING_ENVS,
        "Model-based: requires an env with explicit transition probabilities."),
    "policy_iteration": _entry(
        PolicyIteration, "Sutton & Barto Ch. 4", "planning", "none",
        "discrete", "tabular",
        "Alternates policy evaluation and greedy improvement until stable.",
        ["exact planning", "small MDPs"], PLANNING_ENVS,
        "Converges to the optimal policy for finite MDPs."),
    "value_iteration": _entry(
        ValueIteration, "Sutton & Barto Ch. 4", "planning", "none",
        "discrete", "tabular",
        "In-place Bellman optimality backups until convergence.",
        ["exact planning", "small MDPs"], PLANNING_ENVS,
        "Equivalent to policy iteration with truncated evaluation; faster in practice."),

    "mc_prediction": _entry(
        MCPrediction, "Sutton & Barto Ch. 5", "tabular-value", "none",
        "discrete", "tabular",
        "First- or every-visit Monte Carlo prediction of a fixed (random) policy.",
        ["learning value functions from episodes", "understanding MC returns"], TABULAR_ENVS,
        "Evaluates the random policy; use the control variants to learn behavior."),
    "mc_control_onpolicy": _entry(
        MCControlOnPolicy, "Sutton & Barto Ch. 5", "tabular-value", "on-policy",
        "discrete", "tabular",
        "On-policy first-visit MC control with epsilon-soft policies.",
        ["episodic tasks", "high-variance but unbiased learning"], TABULAR_ENVS,
        "Waits for episode completion before updating; unbiased but noisy."),
    "mc_control_offpolicy": _entry(
        MCControlOffPolicy, "Sutton & Barto Ch. 5", "tabular-value", "off-policy",
        "discrete", "tabular",
        "Off-policy MC control via importance sampling toward a greedy target.",
        ["learning greedy policy from exploratory data"], TABULAR_ENVS,
        "Only episodes consistent with the greedy target contribute updates."),

    "td0_prediction": _entry(
        TD0Prediction, "Sutton & Barto Ch. 6", "tabular-value", "none",
        "discrete", "tabular",
        "TD(0) prediction: bootstrap value estimates from the next state.",
        ["learning with fewer samples than MC"], TABULAR_ENVS,
        "Evaluates the random policy; the control workhorses are SARSA/Q-learning."),
    "sarsa": _entry(
        Sarsa, "Sutton & Barto Ch. 6", "tabular-value", "on-policy",
        "discrete", "tabular",
        "On-policy TD(0) control: learns Q for the policy being followed.",
        ["classic control", "SARSA vs Q-learning cliff comparison"], TABULAR_ENVS,
        "Safer than Q-learning in risky environments (e.g., CliffWalking)."),
    "expected_sarsa": _entry(
        ExpectedSarsa, "Sutton & Barto Ch. 6", "tabular-value", "on-policy",
        "discrete", "tabular",
        "SARSA variant that bootstraps with the expected value over actions.",
        ["lower-variance updates", "studying bootstrapping"], TABULAR_ENVS,
        "Reduces variance vs SARSA at the cost of one expectation per update."),
    "qlearning": _entry(
        QLearning, "Sutton & Barto Ch. 6", "tabular-value", "off-policy",
        "discrete", "tabular",
        "Off-policy TD(0): learns the optimal Q independent of behavior policy.",
        ["classic control", "off-policy learning"], TABULAR_ENVS,
        "The most famous TD control algorithm; optimistic, so risk-prone in cliffs."),
    "double_qlearning": _entry(
        DoubleQLearning, "Sutton & Barto Ch. 6.7", "tabular-value", "off-policy",
        "discrete", "tabular",
        "Uses two Q tables to decouple action selection and evaluation, reducing maximization bias.",
        ["maximization bias", "stochastic environments"], TABULAR_ENVS,
        "The tabular ancestor of Double DQN."),
    "td_lambda_prediction": _entry(
        TDLambdaPrediction, "Sutton & Barto Ch. 12", "tabular-value", "none",
        "discrete", "tabular",
        "TD(lambda) prediction with eligibility traces.",
        ["bridging MC and TD", "eligibility traces"], TABULAR_ENVS,
        "lambda=0 is TD(0), lambda=1 approaches MC."),

    "nstep_sarsa": _entry(
        NStepSarsa, "Sutton & Barto Ch. 7", "tabular-value", "on-policy",
        "discrete", "tabular",
        "n-step SARSA: bootstraps after n steps instead of one.",
        ["tuning bootstrap depth", "unifying TD and MC"], TABULAR_ENVS,
        "n=1 is SARSA; large n approaches MC control."),
    "nstep_expected_sarsa": _entry(
        NStepExpectedSarsa, "Sutton & Barto Ch. 7", "tabular-value", "on-policy",
        "discrete", "tabular",
        "n-step SARSA bootstrapping with the expected next-state value.",
        ["lower-variance n-step learning"], TABULAR_ENVS,
        "Combines n-step returns with expected-value bootstrapping."),
    "tree_backup": _entry(
        TreeBackup, "Sutton & Barto Ch. 7", "tabular-value", "off-policy",
        "discrete", "tabular",
        "n-step off-policy TD control without importance sampling.",
        ["off-policy control", "clipping-free updates"], TABULAR_ENVS,
        "Bootstraps with an expectation over the target policy; no IS ratios."),
    "nstep_off_policy": _entry(
        NStepOffPolicy, "Sutton & Barto Ch. 7", "tabular-value", "off-policy",
        "discrete", "tabular",
        "n-step off-policy SARSA with per-decision importance sampling.",
        ["importance sampling", "off-policy control"], TABULAR_ENVS,
        "Updates only when the behavior actions match the greedy target."),

    "sarsa_lambda": _entry(
        SarsaLambda, "Sutton & Barto Ch. 12", "tabular-value", "on-policy",
        "discrete", "tabular",
        "SARSA(lambda) with accumulating or replacing eligibility traces.",
        ["trace-based credit assignment", "faster tabular learning"], TABULAR_ENVS,
        "One update per step but propagates credit across many steps."),
    "q_lambda": _entry(
        QLambda, "Sutton & Barto Ch. 12", "tabular-value", "off-policy",
        "discrete", "tabular",
        "Watkins Q(lambda): off-policy traces that cut at exploratory actions.",
        ["off-policy traces", "greedy trajectories"], TABULAR_ENVS,
        "Traces are reset when the behavior action is non-greedy."),

    "dyna_q": _entry(
        DynaQ, "Sutton & Barto Ch. 8", "model-based", "off-policy",
        "discrete", "tabular",
        "Q-learning augmented with a learned model and simulated planning steps.",
        ["model-based learning", "sample efficiency"], TABULAR_ENVS,
        "The simplest integration of learning and planning."),
    "dyna_q_plus": _entry(
        DynaQPlus, "Sutton & Barto Ch. 8", "model-based", "off-policy",
        "discrete", "tabular",
        "Dyna-Q with a bonus for states/actions not visited recently.",
        ["non-stationary environments", "exploration bonus"], TABULAR_ENVS,
        "The bonus encourages revisiting stale model entries."),
    "prioritized_sweeping": _entry(
        PrioritizedSweeping, "Sutton & Barto Ch. 8", "model-based", "off-policy",
        "discrete", "tabular",
        "Model-based planning that replays the transitions with the largest TD errors.",
        ["efficient planning", "focused backups"], TABULAR_ENVS,
        "Backs up predecessors of high-error transitions via a priority queue."),

    "semi_gradient_sarsa": _entry(
        SemiGradientSarsa, "Sutton & Barto Ch. 10", "value-based", "on-policy",
        "discrete", "continuous",
        "Online semi-gradient TD(0) control with a neural Q approximator.",
        ["function approximation", "on-policy value learning"], CLASSIC_ENVS,
        "No replay or target network: the honest, unstable ancestor of DQN."),
    "naive_qlearning": _entry(
        NaiveQLearning, "Sutton & Barto Ch. 11 (the naive method)", "value-based", "off-policy",
        "discrete", "continuous",
        "Online semi-gradient Q-learning: naive off-policy bootstrapping with a neural Q, no replay or target network.",
        ["studying the deadly triad", "off-policy divergence"], CLASSIC_ENVS,
        "Off-policy + bootstrapping + function approximation: the classic recipe for divergence. "
        "Compare with dqn to see what replay and target networks fix."),
    "reinforce": _entry(
        Reinforce, "Sutton & Barto Ch. 13", "policy-gradient", "on-policy",
        "both", "both",
        "REINFORCE: unbiased policy gradient using full-episode returns.",
        ["policy gradients 101", "episodic tasks"], BOTH_ENVS,
        "High variance; needs small learning rates. baseline=True removes the baseline."),
    "reinforce_baseline": _entry(
        Reinforce, "Sutton & Barto Ch. 13", "policy-gradient", "on-policy",
        "both", "both",
        "REINFORCE with a learned value-function baseline to reduce variance.",
        ["variance reduction", "policy gradients"], BOTH_ENVS,
        "Same update rule as reinforce but subtracts a critic estimate."),
    "actor_critic": _entry(
        ActorCritic, "Sutton & Barto Ch. 13", "actor-critic", "on-policy",
        "both", "both",
        "One-step actor-critic: policy and value trained together on TD error.",
        ["bootstrapping policy gradients", "continuous control"], BOTH_ENVS,
        "Lower variance than REINFORCE but biased."),
    "a2c": _entry(
        A2C, "Mnih et al. 2016 (A3C paper)", "actor-critic", "on-policy",
        "both", "both",
        "Synchronous advantage actor-critic over N parallel environments.",
        ["parallel training", "stabilizing actor-critics"], BOTH_ENVS,
        "Uses gymnasium vector envs (default 8); n-step advantage estimates."),
    "ppo": _entry(
        PPO, "Schulman et al. 2017, arXiv:1707.06347", "policy-gradient", "on-policy",
        "both", "both",
        "Proximal policy optimization with clipped surrogate objective and GAE.",
        ["general-purpose RL", "the default modern baseline"], BOTH_ENVS,
        "The workhorse of modern RL; robust across seeds and tasks."),
    "trpo": _entry(
        TRPO, "Schulman et al. 2015, arXiv:1502.05477", "policy-gradient", "on-policy",
        "both", "both",
        "Trust region policy optimization: KL-constrained updates via conjugate gradient.",
        ["stable monotonic improvements", "theory-minded experiments"], BOTH_ENVS,
        "The theoretical predecessor of PPO; slower but provably monotone."),

    "dqn": _entry(
        DQN, "Mnih et al. 2015, Nature", "value-based", "off-policy",
        "discrete", "continuous",
        "Deep Q-network: neural Q with replay buffer and target network.",
        ["discrete-action deep RL", "Atari-style control"], CLASSIC_ENVS,
        "The algorithm that started deep RL."),
    "double_dqn": _entry(
        DoubleDQN, "van Hasselt et al. 2016, AAAI", "value-based", "off-policy",
        "discrete", "continuous",
        "Decouples action selection (online net) from evaluation (target net) to reduce overestimation.",
        ["reducing Q overestimation"], CLASSIC_ENVS,
        "A one-line change to DQN with measurable improvement."),
    "dueling_dqn": _entry(
        DuelingDQN, "Wang et al. 2016, ICML", "value-based", "off-policy",
        "discrete", "continuous",
        "Separates value and advantage streams in the Q network.",
        ["action-independent value learning"], CLASSIC_ENVS,
        "Helps in environments where many actions have similar values."),
    "prioritized_dqn": _entry(
        PrioritizedDQN, "Schaul et al. 2016, ICLR", "value-based", "off-policy",
        "discrete", "continuous",
        "Samples transitions by TD-error magnitude with importance correction (also uses double Q).",
        ["sample efficiency", "focused learning"], CLASSIC_ENVS,
        "A sum-tree buffer drives replay toward surprising transitions."),
    "c51": _entry(
        C51, "Bellemare et al. 2017, ICML", "distributional", "off-policy",
        "discrete", "continuous",
        "Categorical DQN: learns the full return distribution over 51 atoms.",
        ["distributional RL", "risk-sensitive behavior"], CLASSIC_ENVS,
        "Learns value distributions instead of point estimates; Rainbow ingredient."),
    "qr_dqn": _entry(
        QRDQN, "Dabney et al. 2018, ICML", "distributional", "off-policy",
        "discrete", "continuous",
        "Quantile regression DQN: learns quantiles of the return distribution.",
        ["distributional RL", "quantile methods"], CLASSIC_ENVS,
        "A flexible successor to C51 with quantile Huber loss."),
    "rainbow": _entry(
        Rainbow, "Hessel et al. 2018, AAAI", "value-based", "off-policy",
        "discrete", "continuous",
        "Combines double Q, dueling, prioritized replay, noisy nets, n-step returns, and C51.",
        ["state-of-the-art discrete control", "ablation studies"], CLASSIC_ENVS,
        "Each ingredient contributes; run ablations by turning flags off."),
    "her_dqn": _entry(
        HERDQN, "Andrychowicz et al. 2017, NeurIPS (HER) + DQN", "off-policy value", "off-policy",
        "discrete", "continuous",
        "DQN with Hindsight Experience Replay for sparse-reward goal reaching (lidar8 urban obs: goal = last 2 channels).",
        ["sparse reward", "goal reaching"], ["offset_l2_discrete"],
        "Relabels failed episodes with the reached position as a hindsight goal; turns sparse +1/-1 into a shaped signal."),
    "rainbow_her": _entry(
        RainbowHER, "HER + Rainbow (Hessel 2018 / Andrychowicz 2017)", "off-policy value", "off-policy",
        "discrete", "continuous",
        "HER with prioritized replay, double Q, and dueling for sparse-reward goal reaching.",
        ["sparse reward", "sample efficiency"], ["offset_l2_discrete"],
        "Prioritized replay re-samples the rare +1 hindsight transitions, giving the credit signal that plain HER/DQN miss."),

    "ddpg": _entry(
        DDPG, "Lillicrap et al. 2016, ICLR", "actor-critic", "off-policy",
        "continuous", "continuous",
        "Deep deterministic policy gradient: off-policy actor-critic for continuous actions.",
        ["continuous control", "sample-efficient actor-critics"], CONTINUOUS_ENVS,
        "Fragile to hyperparameters; TD3 and SAC are its descendants."),
    "td3": _entry(
        TD3, "Fujimoto et al. 2018, ICML", "actor-critic", "off-policy",
        "continuous", "continuous",
        "Twin delayed DDPG: clipped double-Q, delayed policy updates, target smoothing.",
        ["continuous control", "fixing DDPG's overestimation"], CONTINUOUS_ENVS,
        "Addresses DDPG's value overestimation with three targeted fixes."),
    "sac": _entry(
        SAC, "Haarnoja et al. 2018, ICML", "actor-critic", "off-policy",
        "continuous", "continuous",
        "Soft actor-critic: maximum-entropy RL with stochastic policy and automatic temperature.",
        ["continuous control", "robust exploration"], CONTINUOUS_ENVS,
        "The default choice for continuous control."),
    "impala": _entry(
        IMPALA, "Espeholt et al. 2018, ICML", "actor-critic", "off-policy",
        "discrete", "continuous",
        "V-trace: parallel actor-learners with clipped importance ratios for the value target (rho) and the trace (c).",
        ["scalable on-policy learning", "off-policy correction"], CLASSIC_ENVS,
        "Single-learner port: ratios are ~1, so the policy gradient is the on-policy one; validated on cartpole and acrobot."),
    "acer": _entry(
        ACER, "Wang et al. 2017, ICML", "actor-critic", "off-policy",
        "discrete", "continuous",
        "Sample-efficient actor-critic: episode retrace targets, truncated IS with bias correction, efficient-TRPO trust region.",
        ["sample-efficient actor-critics", "off-policy stabilization"], CLASSIC_ENVS,
        "Works well on cartpole; weak on acrobot - the Q-based advantage gives no signal while the behavior policy is uniform (constant rewards)."),
    "acktr": _entry(
        ACKTR, "Wu et al. 2017, ICML", "actor-critic", "on-policy",
        "discrete", "continuous",
        "A2C with Kronecker-factored natural-gradient updates (K-FAC) instead of Adam.",
        ["curvature-aware updates", "natural gradient RL"], CLASSIC_ENVS,
        "Works on cartpole; weak on acrobot - KFAC conditioning amplifies noise on tiny action spaces. Fisher per Linear layer: nat = G^-1 * grad * A^-1."),
    "mcts": _entry(
        MCTSAgent, "Coulom 2006 / S&B Ch. 8", "model-based", "on-policy",
        "discrete", "continuous",
        "Monte Carlo Tree Search: UCB1 selection, expansion, random rollouts, mean-return backup. Plans in the real environment via state cloning.",
        ["tree search", "planning with a known model"], STATE_ENVS,
        "Pure planning, no learning; cartpole ~160-250 returns with 100 iterations. The real env (state cloning) is the model."),
    "alphazero": _entry(
        AlphaZeroAgent, "Silver et al. 2018, Science", "model-based", "on-policy",
        "discrete", "continuous",
        "AlphaZero: self-play MCTS with PUCT guided by a learned policy/value network trained on visit-count policies and episode returns.",
        ["self-play search", "policy/value-guided MCTS"], STATE_ENVS,
        "Leaf values blend the network with real model rollouts - a naive value-only backup gives no gradient on dense-reward tasks. Cartpole ~224-244 after 6k steps."),
    "muzero": _entry(
        MuZeroAgent, "Schrittwieser et al. 2019, Nature", "model-based", "on-policy",
        "discrete", "continuous",
        "MuZero: learns a latent model (representation, dynamics with reward and continuation, policy/value prediction) and plans in it with MCTS.",
        ["learned world models", "planning without a model"], STATE_ENVS,
        "Latent-consistency loss and a continuation head are required for dense-reward tasks; a one-ply real lookahead breaks the random-policy bootstrap. Cartpole ~100-500."),
    "world_models": _entry(
        WorldModelsAgent, "Ha & Schmidhuber 2018", "model-based", "off-policy",
        "discrete", "continuous",
        "World Models: a VAE compresses observations, an MDN-RNN learns the latent dynamics, and a linear controller is trained by CEM inside dreams.",
        ["generative world models", "evolutionary control in imagination"], CLASSIC_ENVS,
        "Works purely from random data: cartpole ~40-130 with the CEM controller on imagined rollouts."),
    "dreamer": _entry(
        DreamerAgent, "Hafner et al. 2020, ICLR", "model-based", "off-policy",
        "discrete", "continuous",
        "Dreamer: an RSSM world model (deterministic GRU state + stochastic latent) is trained with reconstruction, reward, continuation and free-bits KL; an actor-critic then learns in imagination.",
        ["recurrent world models", "policy learning in imagination"], CLASSIC_ENVS,
        "The continuation head needs class weighting and the terminal transition in the data, or dreams never end and the actor gets no signal. Cartpole ~30-130."),
    "maddpg": _entry(
        MADDPGAgent, "Lowe et al. 2017, NeurIPS", "multi-agent", "off-policy",
        "discrete", "continuous",
        "MADDPG: each agent runs a policy while a centralized critic sees the joint observation and joint actions (CTDE), with target networks and a joint TD target.",
        ["centralized training", "decentralized execution"], MARL_ENVS,
        "Discrete-action port: double-Q joint targets (argmax from the online critic, value from the target) and per-agent softmax actors with greedy-Q targets. On cooperative_cartpole (anti-correlated optimal) the batch-averaged policy gradient averages out symmetric state advantages and the actors collapse to correlated same-direction play (ret ~8-12 vs random 23, oscillate 43, QMIX 93)."),
    "mappo": _entry(
        MAPPOAgent, "Yu et al. 2022, ICLR", "multi-agent", "on-policy",
        "discrete", "continuous",
        "MAPPO: independent PPO policies per agent trained on the same GAE advantages from a centralized critic over the joint state (CTDE).",
        ["centralized critics", "trust-region policy updates"], MARL_ENVS,
        "Shared advantages make both policies drift symmetrically, so anti-correlated coordination is never discovered; ret ~9-12 on cooperative_cartpole even at 60k steps."),
    "qmix": _entry(
        QMIXAgent, "Rashid et al. 2018, ICML", "multi-agent", "off-policy",
        "discrete", "continuous",
        "QMIX: per-agent Q-networks whose values are mixed by a state-dependent nonnegative (IGM-preserving) hypernetwork into a joint Q trained with joint TD.",
        ["value decomposition", "cooperative MARL"], MARL_ENVS,
        "Learns the anti-correlated cooperative policy that the independent-policy CTDE methods cannot: ret ~91-93 on cooperative_cartpole. Per-sample Q fitting preserves the state-conditional action ordering that batch-averaged actor losses destroy."),
    "iql_marl": _entry(
        IQLAgent, "Tan 1993 (independent Q-learning)", "multi-agent", "off-policy",
        "discrete", "continuous",
        "Independent Q-Learning: each agent runs its own DQN on the shared team reward with no centralized mixing.",
        ["baseline", "CTDE ablations"], MARL_ENVS,
        "Baseline that isolates the value of QMIX's mixing network; the L3 gate requires QMIX to beat this by 20%."),
    "her": _entry(
        HERAgent, "Andrychowicz et al. 2017, NeurIPS", "value-based", "off-policy",
        "discrete", "continuous",
        "Hindsight Experience Replay: failed episodes are relabeled with goals the agent actually reached, converting sparse rewards into dense ones for goal-conditioned value learning.",
        ["goal-conditioned RL", "sparse-reward learning"], ["point_reach"],
        "DQN core with goal as part of the observation; future-state relabeling (k=4 copies per transition). Learns point_reach (~40-50 step horizon) where plain sparse DQN stays at ~0 reward."),
    "icm": _entry(
        ICM, "Pathak et al. 2017, ICML", "value-based", "off-policy",
        "discrete", "both",
        "Intrinsic Curiosity Module: DQN plus a learned forward model whose prediction error is an intrinsic reward.",
        ["sparse-reward exploration", "curiosity-driven learning"], CLASSIC_ENVS + TABULAR_ENVS,
        "Inverse model ties the features to actionability; intrinsic reward is std-normalized."),
    "rnd": _entry(
        RND, "Burda et al. 2018, ICLR", "value-based", "off-policy",
        "discrete", "both",
        "Random Network Distillation: DQN plus an intrinsic reward from a frozen random network the predictor cannot match on novel states.",
        ["sparse-reward exploration", "novelty search"], CLASSIC_ENVS + TABULAR_ENVS,
        "Simpler than ICM: no inverse model, bonus uses only the next state."),
    "r2d2": _entry(
        R2D2, "Kapturowski et al. 2019, ICLR", "value-based", "off-policy",
        "discrete", "continuous",
        "Recurrent Replay DQN: LSTM Q-network trained on prioritized sequences with burn-in, n-step returns, and value rescaling.",
        ["partially observable tasks", "recurrent value learning"], CLASSIC_ENVS,
        "Single-learner port: sequences replayed a bounded number of times; targets computed in the rescaled space h(x)."),
    "agent57": _entry(
        Agent57, "Badia et al. 2020, Nature", "value-based", "off-policy",
        "discrete", "continuous",
        "R2D2 with a meta-controller that balances exploration (beta) and long-term credit (gamma) via a UCB bandit over parameterizations.",
        ["long-horizon tasks", "exploration/exploitation balance"], CLASSIC_ENVS,
        "RND-style intrinsic reward (std-normalized) scaled by beta; one value head per discount; round-robin seeding then sliding-window UCB selects the active parameterization."),
    "cql": _entry(
        CQL, "Kumar et al. 2020, NeurIPS", "value-based", "off-policy",
        "discrete", "continuous",
        "Conservative Q-Learning: offline Q-learning with a penalty on out-of-distribution actions (log-sum-exp Q minus the Q of the data action).",
        ["offline RL", "safe policy learning from fixed data"], CLASSIC_ENVS,
        "Dataset collected once from an epsilon-greedy DQN behavior policy (eps 0.3 keeps action diversity); polyak target Q; verified on cartpole (107) and acrobot (-75)."),
    "iql": _entry(
        IQL, "Kostrikov et al. 2022, NeurIPS", "value-based", "off-policy",
        "discrete", "continuous",
        "Implicit Q-Learning: the value function is fit with expectile regression so OOD actions never inflate the target; the policy is advantage-weighted behavioral cloning.",
        ["offline RL", "penalty-free conservative value learning"], CLASSIC_ENVS,
        "Expectile tau=0.7 with a polyak target-V; AWR weights use mean-abs-scaled advantages; verified on cartpole (500); acrobot is unstable at smoke scale (needs more diverse data than the 40k budget)."),
    "decision_transformer": _entry(
        DecisionTransformer, "Chen et al. 2021, NeurIPS", "other", "off-policy",
        "discrete", "continuous",
        "Decision Transformer: causal transformer over (return-to-go, state, action) tokens that predicts actions conditioned on a target return, framing RL as sequence modeling.",
        ["offline RL", "sequence-modeling RL"], CLASSIC_ENVS,
        "Trained on behavior-policy episodes; eval conditions on the best dataset return; trajectories tracked internally (reset_episode hook); verified on cartpole (~105)."),
    "gtd": _entry(
        GTD, "Sutton et al. 2009", "value-based", "off-policy",
        "discrete", "tabular",
        "GTD2 gradient TD: off-policy linear prediction with a second weight vector for the covariance correction.",
        ["off-policy value prediction", "divergence-free TD"], TABULAR_ENVS,
        "Behavior is uniform random; target is uniform over the first two actions, so IS ratios are 2/0."),
    "tde": _entry(
        TrueOnlineTDLambda, "van Seijen et al. 2014", "value-based", "on-policy",
        "discrete", "tabular",
        "True Online TD(lambda): dutch eligibility trace with the correction term that makes the per-step update exact.",
        ["on-policy value prediction", "efficient TD(lambda)"], TABULAR_ENVS,
        "O(d) per step with the dutch trace; on-policy here so the trace stays bounded."),
    "emphatic_td": _entry(
        EmphaticTD, "Sutton et al. 2016, AIJ", "value-based", "off-policy",
        "discrete", "tabular",
        "Emphatic TD: re-weights the TD update with an emphasis M_t = lambda*rho*M_{t-1} + 1.",
        ["off-policy value prediction", "stable off-policy learning"], TABULAR_ENVS,
        "Emphasis grows when the target policy visits a state more than behavior does."),
}

for name, (description, source) in STUBS.items():
    ALGORITHMS[name] = {
        "class": make_stub(name, description, source, "discrete", "both", "model-based", "off-policy"),
        "status": "stub", "source": source, "family": "model-based", "policy": "off-policy",
        "action_space": "discrete", "state_space": "both",
        "description": description, "use_cases": ["planned", "not yet implemented"],
        "compatible_envs": [], "notes": "Stub: registered so the table is complete; running raises NotImplementedError.",
    }


def check_compat(algo_name, env_name):
    algo = ALGORITHMS[algo_name]
    env = _env_meta(env_name)
    if env_name not in algo["compatible_envs"]:
        return False, (f"{algo_name} is not listed as compatible with {env_name}. "
                       f"Compatible: {algo['compatible_envs'] or 'none yet'}")
    if algo["action_space"] != "both" and algo["action_space"] != env["action_space"]:
        return False, f"{algo_name} needs {algo['action_space']} actions but {env_name} has {env['action_space']}"
    if algo["state_space"] != "both" and algo["state_space"] != env["state_space"]:
        return False, f"{algo_name} needs {algo['state_space']} states but {env_name} has {env['state_space']}"
    return True, ""


def _env_meta(env_name):
    from environments.registry import ENVIRONMENTS
    return ENVIRONMENTS[env_name]


def list_implemented():
    return [n for n, a in ALGORITHMS.items() if a["status"] == "implemented"]


def list_stubs():
    return [n for n, a in ALGORITHMS.items() if a["status"] == "stub"]


def render_table(implemented_only=False):
    rows = []
    for name, a in sorted(ALGORITHMS.items()):
        if implemented_only and a["status"] != "implemented":
            continue
        rows.append([
            name, "implemented" if a["status"] == "implemented" else "stub",
            a["family"], a["policy"], a["action_space"], a["state_space"], a["source"],
        ])
    return rows