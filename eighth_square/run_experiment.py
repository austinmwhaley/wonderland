import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.registry import ALGORITHMS, check_compat, render_table
from environments.registry import ENVIRONMENTS
from experiments.configs import DEFAULT_CONFIGS, ENV_STEPS, EXPERIMENTS
from experiments.run import RESULTS_DIR, run_experiment, run_id

COLUMNS = ["algorithm", "status", "family", "policy", "action", "state", "source"]


def print_algorithms(args):
    rows = render_table(implemented_only=args.implemented_only)
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(COLUMNS)]
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(COLUMNS)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


def print_environments():
    cols = ["env", "gym id", "action", "state", "family", "description"]
    rows = [[name, e["gym_id"], e["action_space"], e["state_space"], e["family"],
             e["description"][:60]] for name, e in ENVIRONMENTS.items()]
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(cols)]
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(cols)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


def main():
    parser = argparse.ArgumentParser(
        description="Run reinforcement learning experiments (algorithm x environment).")
    parser.add_argument("--algo", help="algorithm name (see --list)")
    parser.add_argument("--env", help="environment name (see --list)")
    parser.add_argument("--experiment", help="named experiment preset (see experiments/configs.py)")
    parser.add_argument("--steps", type=int, default=None, help="total environment steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None, help="optional run tag (appears in run name)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu", "mps", None])
    parser.add_argument("--eval-freq", type=int, default=None, help="override eval frequency")
    parser.add_argument("--list", action="store_true", help="print the algorithm table")
    parser.add_argument("--implemented-only", action="store_true",
                        help="with --list, show only implemented algorithms")
    parser.add_argument("--envs", action="store_true", help="print the environment table")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nALGORITHMS (implemented x stub):\n")
        print_algorithms(args)
        print("\n\nENVIRONMENTS:\n")
        print_environments()
        return

    if args.envs:
        print_environments()
        return

    algo, env = args.algo, args.env
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            sys.exit(f"unknown experiment '{args.experiment}'; available: {list(EXPERIMENTS)}")
        preset = EXPERIMENTS[args.experiment]
        algo = preset.pop("algo", algo)
        env = preset.pop("env", env)
    if not algo or not env:
        parser.print_help()
        sys.exit("--algo and --env are required (or use --experiment)")

    if algo not in ALGORITHMS:
        sys.exit(f"unknown algorithm '{algo}'; see --list")
    if env not in ENVIRONMENTS:
        sys.exit(f"unknown environment '{env}'; see --envs")
    if ALGORITHMS[algo]["status"] == "stub":
        sys.exit(f"'{algo}' is a planned stub; it is not implemented yet (see --list)")

    ok, why = check_compat(algo, env)
    if not ok:
        sys.exit(f"incompatible combo: {why}")

    config = dict(DEFAULT_CONFIGS.get(algo, {}))
    config.update(DEFAULT_CONFIGS.get("bandit", {})) if env == "bandit" else None
    if args.experiment:
        config.update(EXPERIMENTS[args.experiment])
    config["steps"] = args.steps if args.steps else config.get("steps", ENV_STEPS.get(env, 50_000))
    config["seed"] = args.seed
    if args.eval_freq:
        config["eval_freq"] = args.eval_freq
    if args.device:
        config["device"] = args.device

    run_dir = os.path.join(RESULTS_DIR, "runs", run_id(algo, env, args.tag, args.seed))
    run_experiment(algo, env, config, run_dir, quiet=args.quiet)


if __name__ == "__main__":
    main()