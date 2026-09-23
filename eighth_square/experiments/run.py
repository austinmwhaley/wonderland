import os
import time

from algorithms.base import BaseAgent
from algorithms.registry import ALGORITHMS, check_compat
from environments.registry import make_env
from tracking.tracker import MetricTracker

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_id(algo_name, env_name, tag=None, seed=0):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [algo_name, env_name, stamp]
    if tag:
        parts.insert(2, tag)
    return "_".join(parts)


def run_experiment(algo_name, env_name, config, run_dir, quiet=False):
    ok, why = check_compat(algo_name, env_name)
    if not ok:
        raise ValueError(why)
    env = make_env(env_name, seed=config.get("seed", 0))
    config["device"] = config.get("device") or detect_device()
    agent_cls = ALGORITHMS[algo_name]["class"]
    agent = agent_cls(env, config)
    tracker = MetricTracker(run_dir)
    tracker.save_config(config, algo_name, env_name)
    start = time.time()
    if not quiet:
        print(f"[{algo_name} x {env_name}] training for {config.get('steps')} steps...")
    try:
        agent.train(env, config, tracker)
    finally:
        elapsed = time.time() - start
        path = os.path.join(run_dir, "checkpoint.pt")
        agent.save(path)
        tracker.close()
        env.close()
    if not quiet:
        print(f"done in {elapsed:.1f}s -> {run_dir}")
    return run_dir, elapsed


def detect_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"