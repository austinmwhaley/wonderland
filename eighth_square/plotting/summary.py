"""
Generates a summary report comparing algorithm performance across environments.
Reads Result objects saved as pickle or JSON, or can be called programmatically
with a list of (env_name, algo_name, Result) tuples.
"""

import os
from typing import Optional

import numpy as np

from algorithms.base import Result


def generate_summary(
    all_results: dict[str, dict[str, Result]],
    output_path: Optional[str] = None,
) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("EIGHTH SQUARE — RL ALGORITHM COMPARISON REPORT")
    lines.append("=" * 80)

    for env_name, env_results in sorted(all_results.items()):
        lines.append("")
        lines.append(f"ENVIRONMENT: {env_name}")
        lines.append("-" * 40)

        rows = []
        for algo_name, result in sorted(env_results.items()):
            rewards = result.episode_rewards
            if not rewards:
                rows.append((algo_name, 0, 0, 0, "N/A", "FAILED"))
                continue

            max_reward = float(np.max(rewards))
            final_avg = float(np.mean(rewards[-min(100, len(rewards)) :]))
            wall_time = result.wall_time
            episodes = len(rewards)

            if result.converged:
                status = f"SOLVED @ ep {result.episodes_to_solve}"
            else:
                status = "did not converge"

            rows.append((algo_name, max_reward, final_avg, wall_time, episodes, status))

        rows.sort(key=lambda r: r[2], reverse=True)

        lines.append(
            f"{'Algorithm':<20} {'Max R':>8} {'Final Avg':>10} {'Time':>8} {'Episodes':>10}  Status"
        )
        lines.append("-" * 80)

        for algo_name, max_r, final_avg, wall_time, episodes, status in rows:
            lines.append(
                f"{algo_name:<20} {max_r:>8.1f} {final_avg:>10.1f} {wall_time:>7.1f}s {episodes:>10}  {status}"
            )

        lines.append("-" * 40)

        best = rows[0] if rows else None
        if best:
            lines.append(f"BEST: {best[0]} (final avg = {best[2]:.1f})")

        converged_count = sum(1 for r in env_results.values() if r.converged)
        total = len(env_results)
        lines.append(f"Converged: {converged_count}/{total} algorithms")

        if all_results[env_name]:
            best_algo = max(
                env_results.items(),
                key=lambda kv: kv[1].avg_rewards[-1] if kv[1].avg_rewards else 0,
            )
            lines.append("")
            lines.append("ANALYSIS:")
            lines.append(f"  Top performer: {best_algo[0]}")

            dqn_variants = [n for n in env_results if "dqn" in n]
            if len(dqn_variants) >= 2:
                dqn_scores = {
                    n: np.mean(env_results[n].episode_rewards[-100:]) for n in dqn_variants
                }
                best_dqn = max(dqn_scores, key=dqn_scores.get)
                lines.append(f"  Best DQN variant: {best_dqn} ({dqn_scores[best_dqn]:.1f})")
                if "vanilla_dqn" in dqn_scores and "dqn" in dqn_scores:
                    improvement = (
                        (dqn_scores.get("dqn", 0) - dqn_scores.get("vanilla_dqn", 0))
                        / max(abs(dqn_scores.get("vanilla_dqn", 1)), 1)
                    ) * 100
                    lines.append(f"  DQN improvements over Vanilla DQN: {improvement:+.1f}%")

            pg_variants = [n for n in env_results if n in ("reinforce", "a2c", "ppo", "grpo")]
            if len(pg_variants) >= 2:
                pg_scores = {n: np.mean(env_results[n].episode_rewards[-100:]) for n in pg_variants}
                best_pg = max(pg_scores, key=pg_scores.get)
                lines.append(f"  Best policy gradient: {best_pg} ({pg_scores[best_pg]:.1f})")

            tabular_variants = [
                n for n in env_results if n in ("q_learning", "sarsa", "expected_sarsa")
            ]
            if len(tabular_variants) >= 2:
                tab_scores = {
                    n: np.mean(env_results[n].episode_rewards[-100:]) for n in tabular_variants
                }
                best_tab = max(tab_scores, key=tab_scores.get)
                lines.append(f"  Best tabular method: {best_tab} ({tab_scores[best_tab]:.1f})")

            times = {n: r.wall_time for n, r in env_results.items() if r.wall_time > 0}
            if times:
                fastest = min(times, key=times.get)
                slowest = max(times, key=times.get)
                lines.append(f"  Fastest: {fastest} ({times[fastest]:.1f}s)")
                lines.append(f"  Slowest: {slowest} ({times[slowest]:.1f}s)")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report
