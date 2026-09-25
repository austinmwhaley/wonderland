import argparse
import os
import pickle

from algorithms.registry import list_algorithms
from plotting.plotter import plot_all_environments
from plotting.summary import generate_summary
from runners.experiment import run_all_algorithms


def save_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.pkl")
    with open(path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {path}")


def load_results(output_dir):
    path = os.path.join(output_dir, "results.pkl")
    if not os.path.exists(path):
        print(f"No saved results at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Eighth Square — RL Algorithm Comparison")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["CartPole-v1"],
        help="Environments to run (default: CartPole-v1)",
    )
    parser.add_argument("--algos", nargs="+", default=None, help="Algorithms to run (default: all)")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help='Parameter variants, e.g. "dqn:lr=1e-3,3e-4;ppo:ppo_epochs=4,10"',
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help='Grid sweep, e.g. "dqn:lr=[1e-4,3e-4,1e-3],eps_decay=[0.99,0.995]"',
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of parallel workers (0 = serial)"
    )
    parser.add_argument(
        "--output", default="results", help="Output directory for plots and reports"
    )
    parser.add_argument("--list-algos", action="store_true", help="List available algorithms")
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="Limit training episodes per algorithm"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--no-report", action="store_true", help="Skip summary report")
    parser.add_argument("--no-save", action="store_true", help="Skip saving results to disk")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load saved results and re-plot/report without re-running",
    )
    args = parser.parse_args()

    if args.list_algos:
        print("Available algorithms:")
        for a in list_algorithms():
            print(f"  - {a}")
        return

    if args.load:
        all_results = load_results(args.output)
        if all_results is None:
            return
        print(f"Loaded results from {args.output}/results.pkl")
    else:
        print(f"Running algorithms on environments: {args.envs}")
        print(f"Algorithms: {args.algos or 'all'}")
        if args.variants:
            print(f"Variants: {args.variants}")
        print(f"Seed: {args.seed}")
        print()

        all_results = run_all_algorithms(
            env_names=args.envs,
            algo_names=args.algos,
            max_episodes=args.max_episodes,
            variants_str=args.variants,
            sweep_str=args.sweep,
            workers=args.workers,
        )

        if not args.no_save:
            save_results(all_results, args.output)

    os.makedirs(args.output, exist_ok=True)

    if not args.no_plot:
        plot_results = {
            env_name: {algo: [result] for algo, result in env_results.items()}
            for env_name, env_results in all_results.items()
        }
        plot_all_environments(plot_results, output_dir=args.output, max_episodes=args.max_episodes)

    if not args.no_report:
        report_path = os.path.join(args.output, "report.txt")
        report = generate_summary(all_results, output_path=report_path)
        print(report)

    print(f"\nDone. Output saved to {args.output}/")


if __name__ == "__main__":
    main()
