import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from algorithms.base import Result

COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def plot_results(
    results: dict[str, list[Result]],
    env_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    window: int = 100,
    max_episodes: Optional[int] = None,
):
    fig, ax = plt.subplots(figsize=(14, 8))

    color_idx = 0
    for algo_name, result_list in results.items():
        color = COLORS[color_idx % len(COLORS)]
        color_idx += 1

        for i, result in enumerate(result_list):
            if not result.avg_rewards:
                result.compute_running_avg(window)

            rewards = result.avg_rewards
            if max_episodes and len(rewards) > max_episodes:
                rewards = rewards[:max_episodes]

            episodes = list(range(window, window + len(rewards)))

            label = f"{algo_name}"
            if len(result_list) > 1:
                label += f" (run {i + 1})"

            ax.plot(episodes, rewards, color=color, alpha=0.85, linewidth=1.5, label=label)

    ax.set_xlabel("Episodes", fontsize=13)
    ax.set_ylabel(f"Average Reward (window={window})", fontsize=13)
    ax.set_title(title or f"Algorithm Comparison — {env_name}", fontsize=15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_all_environments(
    all_results: dict[str, dict[str, list[Result]]],
    output_dir: str = "results",
    window: int = 100,
    max_episodes: Optional[int] = None,
):
    for env_name, env_results in all_results.items():
        safe_name = env_name.replace("/", "_").replace("-", "_")
        save_path = os.path.join(output_dir, f"{safe_name}.png")
        plot_results(
            results=env_results,
            env_name=env_name,
            save_path=save_path,
            window=window,
            max_episodes=max_episodes,
        )
