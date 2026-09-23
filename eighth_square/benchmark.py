import argparse
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("combo exceeded wall-clock budget")

from algorithms.registry import ALGORITHMS, check_compat, list_implemented
from environments.registry import ENVIRONMENTS
from experiments.configs import DEFAULT_CONFIGS, ENV_STEPS
from experiments.run import RESULTS_DIR, run_experiment, run_id


def main():
    parser = argparse.ArgumentParser(description="Run every implemented algorithm x compatible environment.")
    parser.add_argument("--steps", type=int, default=20_000, help="steps per combo (default 20000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", default=None, help="restrict to one algorithm")
    parser.add_argument("--env", default=None, help="restrict to one environment")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip (algorithm, environment) pairs whose run directory already exists")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="max seconds per combo before it is aborted (default 3600)")
    args = parser.parse_args()

    pairs = []
    for algo in list_implemented():
        if args.algo and algo != args.algo:
            continue
        for env in ENVIRONMENTS:
            if args.env and env != args.env:
                continue
            if check_compat(algo, env)[0]:
                pairs.append((algo, env))

    if args.resume:
        existing = set(os.listdir(os.path.join(RESULTS_DIR, "runs")))
        pairs = [(a, e) for a, e in pairs
                 if not any(d.startswith(f"{a}_{e}_bench_") for d in existing)]
        print(f"benchmark grid: {len(pairs)} remaining pairs (resume mode)")
    else:
        print(f"benchmark grid: {len(pairs)} (algorithm, environment) pairs")
    failures = []
    signal.signal(signal.SIGALRM, _alarm_handler)
    for algo, env in pairs:
        config = dict(DEFAULT_CONFIGS.get(algo, {}))
        config["steps"] = args.steps
        config["seed"] = args.seed
        if args.device:
            config["device"] = args.device
        run_dir = os.path.join(RESULTS_DIR, "runs", run_id(algo, env, "bench", args.seed))
        signal.alarm(args.timeout)
        try:
            run_experiment(algo, env, config, run_dir, quiet=args.quiet)
        except _Timeout as exc:
            msg = f"TIMEOUT ({args.timeout}s): {exc}"
            print(f"[{algo} x {env}] FAILED -> {msg}")
            failures.append((algo, env, msg))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[{algo} x {env}] FAILED -> {msg}")
            failures.append((algo, env, msg))
        finally:
            signal.alarm(0)

    if failures:
        print(f"\n{len(failures)} failed combos:")
        for algo, env, msg in failures:
            print(f"  {algo} x {env}: {msg}")
    print("\nall done. compare with:")
    print("  python visualize.py --metric return --ema 20")


if __name__ == "__main__":
    main()