# Eighth Square

A modular reinforcement learning framework comparing **20 algorithms** across multiple environments. Features plateau-based early stopping, progress bars, parameter sweep mode, parallel execution, save/load, and automated summary reports.

## Quick Start

```bash
pip install -r requirements.txt
python3 main.py --list-algos                           # Show 20 algorithms
python3 main.py --envs CartPole-v1 --max-episodes 5000  # Run all on CartPole
python3 main.py --variants "ppo:ppo_epochs=4,10"       # Compare hyperparams
python3 main.py --sweep "dqn:lr=[1e-4,3e-4,1e-3]"     # Grid sweep
python3 main.py --workers 4                            # Parallel execution
python3 main.py --load --output results/               # Re-plot saved results
```

## Algorithms (20 total)

| Family | Algorithms |
|---|---|
| **Tabular** (4) | Q-Learning, SARSA, Expected SARSA, Dyna-Q |
| **Value-Based** (7) | Vanilla DQN, DQN, Double DQN, Dueling DQN, Prioritized DQN, QR-DQN, Rainbow DQN |
| **Policy Gradient** (5) | REINFORCE, A2C, PPO, GRPO, TRPO |
| **Actor-Critic** (3) | DDPG, TD3, SAC |
| **Black-Box** (1) | CEM |

**Rainbow DQN** combines 6 improvements: double Q-learning, dueling architecture, prioritized replay, multi-step returns (n=3), categorical distributional RL (C51), and noisy networks for exploration.

## Results Summary

| Environment | Best Algorithm | Solved | Key Insight |
|---|---|---|---|
| **CartPole-v1** | A2C (486) | 6/14 | Policy gradient dominates. Replay buffers hurt simple tasks. |
| **LunarLander-v3** | PPO (197) | 1/4 | PPO is the only solver. DQN fails (-3 avg). |
| **Pendulum-v1** | TD3 (-176) | 0/4 | TD3/DDPG close to solve (-150 threshold). CEM fails. |

## Project Structure

```
eighth_square/
├── main.py                     # CLI: --envs, --algos, --variants, --sweep, --workers
├── core/
│   ├── config.py               # AlgorithmConfig, EnvConfig, defaults, GPU auto-detect
│   ├── networks.py             # MLP, Dueling, Gaussian, NoisyLinear, NoisyDueling
│   ├── replay_buffer.py        # Uniform + Prioritized replay buffers
│   └── trainer.py              # PlateauTracker (early stopping)
├── environments/
│   └── base.py                 # EnvWrapper, DiscretizedWrapper, OneHotWrapper
├── algorithms/                 # 20 algorithm implementations
│   ├── tabular/                # Q-Learning, SARSA, Expected SARSA (+ tabular_es.py)
│   ├── dyna_q.py               # Dyna-Q (model-based planning)
│   ├── vanilla_dqn.py          # Vanilla DQN (no target, no replay)
│   ├── dqn.py                  # DQN, Double DQN, Dueling DQN
│   ├── prioritized_dqn.py      # Prioritized Experience Replay DQN
│   ├── qr_dqn.py               # Quantile Regression DQN (distributional)
│   ├── rainbow_dqn.py          # Rainbow DQN (6 improvements combined)
│   ├── policy_gradient.py      # REINFORCE, A2C, PPO
│   ├── grpo.py                 # Group Relative Policy Optimization
│   ├── trpo.py                 # TRPO (KL-penalty adaptive)
│   ├── continuous.py           # DDPG, TD3, SAC
│   └── cem.py                  # Cross-Entropy Method
├── runners/
│   └── experiment.py           # run_experiment, run_all_algorithms, sweep/variants
├── plotting/
│   ├── plotter.py              # Single-graph comparison plots
│   ├── summary.py              # Automated report generator
│   └── analysis.py             # Macro + micro comprehensive analysis
└── results/                    # Output: plots (*.png), reports (report.txt), data (results.pkl)
```

## Key Features

- **20 algorithms** spanning all major RL paradigms
- **Save/Load**: Auto-saves to `results.pkl`. Use `--load` to re-plot.
- **Parameter variants**: `--variants "algo:param=v1,v2"` — labeled lines on one graph
- **Grid sweep**: `--sweep "algo:lr=[1e-4,1e-3],eps=[0.99,0.999]"` — all combinations
- **Multiprocessing**: `--workers 4` — parallel execution with spawn context
- **Plateau stopping**: Auto-detects convergence, stops early
- **Progress bars**: Live tqdm with avg reward, epsilon, buffer size
- **Auto reports**: `report.txt` per run with rankings and family breakdowns
- **GPU auto-detect**: CUDA > MPS > CPU

## Dependencies

```
gymnasium>=1.0.0  torch>=2.0.0  numpy>=1.24.0  matplotlib>=3.7.0  tqdm>=4.65.0
```

## TODO

- [ ] Run BipedalWalker-v3 (hardest continuous env)
- [ ] Atari support (CNN architectures, frame stacking)
- [ ] YAML config files for reproducible experiments
- [ ] TensorBoard/WandB logging
- [ ] Test suite
- [ ] Investigate DQN replay buffer issue on simple environments
