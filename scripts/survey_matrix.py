"""Multi-environment offline-RL + OPE survey (whole-pool, no diet partition).

For each env:
  1. generate a behavior colony (mixed online policies) -> logged data
  2. train offline-RL candidates on ALL of it (iql, cql, bc, random)
  3. measure each candidate's TRUE discounted return in the live env
  4. run OPE (panel -> gate -> judge) and report whether the estimates and
     DEPLOY/HOLD decisions match truth

Usage: python survey_matrix.py [env ...]
Writes /tmp/opencode/wq_matrix/summary.json and prints a table.
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

import numpy as np

sys.path.insert(0, str(ROOT))

from white_queen.config import QUICK_LOOK
from white_queen.colony.collect import run as colony_run
from white_queen.tribunal.ope import data as odata
from white_queen.tribunal.ope import pipeline
from white_queen.tribunal.adjudicate import _discounted_rollout
from environments.registry import make_env
from white_queen.tribunal.ope.receipts import spearman

OUT = "/tmp/opencode/wq_matrix"
ALGOS = ("iql", "cql", "bc")  # random is an input/logging policy, not an output


def _fin(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def survey(
    env_name,
    seed=0,
    collect_episodes=25,
    dqn_steps=20000,
    pg_steps=20000,
    gt_eps=20,
    offline_steps=6000,
):
    try:
        from white_queen.tribunal.ope.training import seed_all as _sa
        import torch as _t

        _sa(seed)
        np.random.seed(seed)
        _t.manual_seed(seed)
        _t.backends.cudnn.deterministic = True
        _t.backends.cudnn.benchmark = False
        _t.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    # Freeze the per-seed LOG so re-runs measure OPE variance, not data-gen
    # noise (GPU colony training is not bit-reproducible). regen=True rebuilds.
    db_path = os.path.join(OUT, f"{env_name}#s{seed}.db")
    ckpt = os.path.join(OUT, f"{env_name}#s{seed}_ckpt")
    _regen = os.environ.get("WQ_REGEN_LOGS") == "1"
    if _regen and os.path.exists(db_path):
        os.remove(db_path)
    cfg = dict(QUICK_LOOK)
    cfg.update(
        {
            "env": env_name,
            "seeds": [seed],
            "collect_episodes": collect_episodes,
            "collect_eps": 0.2,
            "dqn_train_steps": dqn_steps,
            "pg_train_steps": pg_steps,
            "checkpoint_fracs": (0.1, 1.0),
            "db_path": db_path,
            "ckpt_dir": ckpt,
        }
    )
    t0 = time.perf_counter()
    if not os.path.exists(db_path):
        colony_run(cfg, db_path, ckpt)
    t_col = time.perf_counter() - t0

    d = odata.to_canonical(db_path, estimate_propensity=True, source_name=env_name)
    env = make_env(env_name, seed=999)
    cand_dir = os.path.join(OUT, f"{env_name}_cands")
    handles = pipeline.train_candidates(
        d, ALGOS, gamma=cfg["gamma"], offline_steps=offline_steps, out_dir=cand_dir
    )
    truth = {n: _discounted_rollout(h, env, gt_eps, cfg["gamma"]) for n, h in handles.items()}

    t1 = time.perf_counter()
    import os as _os

    _fqe = {}
    if _os.environ.get("WQ_FQE_STEPS"):
        _fqe["steps_max"] = int(_os.environ["WQ_FQE_STEPS"])
    if _os.environ.get("WQ_DYN_K"):
        _fqe["dyn_ensemble"] = int(_os.environ["WQ_DYN_K"])
    _ek = int(_os.environ.get("WQ_ENSEMBLE_K", "3"))
    rep = pipeline.evaluate_pool(
        d, handles, gamma=cfg["gamma"], fast=True, ensemble_K=_ek, fqe_cfg=(_fqe or None)
    )
    t_ope = time.perf_counter() - t1

    anchor = float(rep["behavior_mean"])
    est, dec = rep["estimates"], rep["decisions"]
    names = list(est)
    sharp = [(_fin(est[n].get("sharp_dm")) or -1e9) for n in names]
    rho_sharp = spearman(sharp, [truth[n] for n in names])
    rho_dr = spearman([float(est[n]["dr"]) for n in names], [truth[n] for n in names])
    best_true = max(names, key=lambda n: truth[n])
    picked_best = bool(rep["rank"] and rep["rank"][0] == best_true)
    dep = set(rep["deployed"])
    tp = [n for n in names if n in dep and truth[n] > anchor]
    fp = [n for n in names if n in dep and truth[n] <= anchor]
    fn = [n for n in names if n not in dep and truth[n] > anchor]
    return {
        "env": env_name,
        "obs_dim": int(d["obs"].shape[1]),
        "nA": int(d["nA"]),
        "N": int(len(d["obs"])),
        "behavior_mean": round(anchor, 2),
        "bar": rep["bar"],
        "t_colony_s": round(t_col, 1),
        "t_ope_s": round(t_ope, 1),
        "rank": rep["rank"],
        "deployed": rep["deployed"],
        "true_best": best_true,
        "picked_best": picked_best,
        "rho_sharp": round(rho_sharp, 3),
        "rho_dr": round(rho_dr, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "certificates": {n: (dec[n].get("certificate") or {}) for n in names},
        "candidates": {
            n: {
                "truth": round(truth[n], 1),
                "sharp": _fin(est[n].get("sharp_dm")),
                "fqe_soft": _fin(est[n].get("fqe_dm")),
                "dr": _fin(est[n].get("dr")),
                "mb": _fin(est[n].get("mb")),
                "mb_sharp": _fin(est[n].get("mb_sharp")),
                "ess": _fin(est[n].get("ess_frac")),
                "deploy": bool(dec[n]["deploy"]),
                "better": bool(truth[n] > anchor),
            }
            for n in names
        },
    }


if __name__ == "__main__":
    envs = sys.argv[1:] or ["cartpole", "mountaincar", "acrobot"]
    summary_path = os.path.join(OUT, "summary.json")
    merged = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                merged = {r["env"]: r for r in json.load(f)}
        except Exception:
            merged = {}
    for e in envs:
        try:
            r = survey(e)
            merged[e] = r
            with open(os.path.join(OUT, f"verdict_{e}.json"), "w") as f:
                json.dump(r, f, indent=1, default=str)
            print(
                f"\n=== {e} === obs={r['obs_dim']} nA={r['nA']} N={r['N']} "
                f"behavior={r['behavior_mean']} bar={r['bar']} "
                f"({r['t_colony_s']}s colony + {r['t_ope_s']}s ope)"
            )
            print(
                f"  deployed={r['deployed']} true_best={r['true_best']} "
                f"picked_best={r['picked_best']} rho_sharp={r['rho_sharp']} "
                f"rho_dr={r['rho_dr']} FP={r['fp']} FN={r['fn']}"
            )
            for n, c in r["candidates"].items():
                print(
                    f"    {n:7s} truth={c['truth']:7.1f} "
                    f"sharp={c['sharp']} softFQE={c['fqe_soft']} "
                    f"DR={c['dr']} MBs={c['mb_sharp']} "
                    f"ess={c['ess']} {'DEPLOY' if c['deploy'] else 'hold'}"
                )
        except Exception as ex:
            import traceback

            print(f"\n=== {e} FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
    with open(summary_path, "w") as f:
        json.dump(list(merged.values()), f, indent=1, default=str)
