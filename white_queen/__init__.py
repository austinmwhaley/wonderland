"""White Queen: toy Offline RL + OPE loop on CartPole.

Stage 1 "Behavior Colony" (colony/): online agents of many families train on
CartPole; checkpoints from birth to mastery roll out trajectories into SQLite.
Stage 2 "The Tribunal" (tribunal/): offline candidates train on that data and
a full OPE validation suite delivers a go/no-go verdict, scored against live
ground truth.

All RL reuses the neighbouring `reinforcement_learning` lab (algorithms/,
environments/) through duck-typed seams. Nothing outside white_queen/ is
modified. See config.py for QUICK_LOOK vs PUBLICATION presets.
"""
