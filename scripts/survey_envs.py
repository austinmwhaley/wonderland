"""Multi-environment survey: build a small colony pool per env, train offline
candidates on that single pool, run OPE, and report. Surfaces shortcomings
across environments rather than overfitting to CartPole.

Usage: python survey_envs.py <env1> <env2> ...
Writes one verdict JSON per env under verdicts/survey/ and prints a table.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import sys
import time

sys.path.insert(0, str(ROOT))

from white_queen.config import QUICK_LOOK
from white_queen.colony.collect import run as colony_run
from white_queen.tribunal.ope import pipeline


def survey_env(env_name, seed=0, out_dir="/tmp/opencode/wq_survey"):
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, f"{env_name}.db")
    ckpt = os.path.join(out_dir, f"{env_name}_ckpt")
    if os.path.exists(db_path):
        os.remove(db_path)
    cfg = dict(QUICK_LOOK)
    cfg.update(
        {
            "env": env_name,
            "seeds": [seed],
            "collect_episodes": 25,
            "collect_eps": 0.2,
            "dqn_train_steps": 20000,
            "pg_train_steps": 20000,
            "checkpoint_fracs": (0.1, 1.0),
            "db_path": db_path,
            "ckpt_dir": ckpt,
        }
    )
    t0 = time.perf_counter()
    colony_run(cfg, db_path, ckpt)
    t_colony = time.perf_counter() - t0
    # Load the whole pool agnostically through DuckDB/Arrow and run offline RL.
    t1 = time.perf_counter()
    rep = pipeline.run(
        db_path,
        gamma=cfg["gamma"],
        estimate_propensity=True,
        offline_steps=6000,
        fast=True,
        ensemble_K=3,
        fqe_cfg={"steps_max": 3000, "eval_every": 500, "patience": 6, "batch": 512, "hidden": 96},
        out_dir=os.path.join(out_dir, f"{env_name}_cands"),
        source_name=env_name,
    )
    t_ope = time.perf_counter() - t1
    rep["env"] = env_name
    rep["t_colony_s"] = round(t_colony, 1)
    rep["t_ope_s"] = round(t_ope, 1)
    with open(os.path.join(out_dir, f"verdict_{env_name}.json"), "w") as f:
        json.dump({k: v for k, v in rep.items() if k != "estimates"}, f, indent=1, default=str)
    return rep


if __name__ == "__main__":
    envs = sys.argv[1:] or ["cartpole", "acrobot", "mountaincar"]
    for e in envs:
        try:
            r = survey_env(e)
            print(
                f"\n=== {e} === behavior={r['behavior_mean']} bar={r['bar']} "
                f"deployed={r['deployed']} colony={r['t_colony_s']}s ope={r['t_ope_s']}s"
            )
            print("  rank:", r["rank"])
            for n in r["rank"]:
                est = r["estimates"][n]
                print(
                    f"    {n:8s} FQE={est['fqe_dm']} FQEarg={est['sharp_dm']} "
                    f"MBarg={est['mb_sharp']} ESS={est['ess_frac']} "
                    f"{'DEPLOY' if r['decisions'][n]['deploy'] else 'hold'}"
                )
        except Exception as ex:
            import traceback

            print(f"\n=== {e} FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
