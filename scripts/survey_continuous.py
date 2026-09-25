"""Continuous-action offline-RL + OPE survey.

Logs come from ONLINE (short DDPG) + RANDOM agents. We then train an offline
IQL (continuous) on those logs and ask the simple question: does the offline
policy beat the logging (behavior) policy's true return, and does OPE agree?

Usage: python survey_continuous.py [env ...]   (default: mountaincar_continuous pendulum)
Writes /tmp/opencode/wq_matrix/continuous.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import sys

import numpy as np

sys.path.insert(0, str(ROOT))

from environments.registry import make_env
from white_queen.tribunal.ope import data as odata
from white_queen.tribunal.ope import continuous as C
from white_queen.tribunal.ope import gate as G
from white_queen.tribunal.ope import judge as J
from white_queen.tribunal.ope.receipts import behavior_stats
from white_queen.tribunal.candidates import train_candidate
from white_queen.tribunal.ope.protocols import EnvStub

OUT = "/tmp/opencode/wq_matrix"
GAMMA = 0.99


class _Tracker:
    def log(self, **kw):
        pass


def _rollout(env, act_fn, episodes, max_steps=1200):
    """Collect transitions; act_fn(state) -> action."""
    obs, act, rew, obs2, done, ep_id, t = [], [], [], [], [], [], []
    for e in range(episodes):
        try:
            s, _ = env.reset(seed=1000 + e)
        except TypeError:
            s, _ = env.reset()
        for step in range(max_steps):
            a = np.asarray(act_fn(s), dtype=np.float32).ravel()
            ns, r, term, trunc, _ = env.step(a)
            d = bool(term or trunc)
            obs.append(np.asarray(s, dtype=np.float32))
            act.append(a)
            rew.append(float(r))
            obs2.append(np.asarray(ns, dtype=np.float32))
            done.append(1.0 if d else 0.0)
            ep_id.append(e)
            t.append(step)
            s = ns
            if d:
                break
    return {
        k: np.asarray(v)
        for k, v in {
            "obs": obs,
            "action": act,
            "reward": rew,
            "next_obs": obs2,
            "done": done,
            "episode": ep_id,
            "t": t,
        }.items()
    }


def _disc_return(env, cand, episodes, gamma=GAMMA, max_steps=1200):
    rets = []
    for e in range(episodes):
        try:
            s, _ = env.reset(seed=5000 + e)
        except TypeError:
            s, _ = env.reset()
        disc, g = 1.0, 0.0
        for _ in range(max_steps):
            a = np.asarray(
                cand.action_mean(np.asarray(s)[None, :])[0]
                if hasattr(cand, "action_mean")
                else cand.act(s, eval=True),
                dtype=np.float32,
            ).ravel()
            s, r, term, trunc, _ = env.step(a)
            g += disc * float(r)
            disc *= gamma
            if term or trunc:
                break
        rets.append(g)
    return float(np.mean(rets))


def survey_env(
    env_name,
    seed=0,
    online_steps=8000,
    n_ep_rand=15,
    n_ep_online=15,
    offline_steps=6000,
    gt_episodes=10,
):
    try:
        from white_queen.tribunal.ope.training import seed_all as _sa
        import torch as _t

        _sa(seed if "seed" in locals() else 0)
        np.random.seed(seed if "seed" in locals() else 0)
        _t.manual_seed(seed if "seed" in locals() else 0)
    except Exception:
        pass
    env = make_env(env_name, seed=0)
    low = np.asarray(env.action_space.low, dtype=np.float64).ravel()
    high = np.asarray(env.action_space.high, dtype=np.float64).ravel()
    rng = np.random.default_rng(0)

    # --- RANDOM logging agent ---
    rand_logs = _rollout(env, lambda s: rng.uniform(low, high), n_ep_rand)

    # --- ONLINE logging agent (short DDPG) ---
    on_logs = None
    try:
        from algorithms.deep.ddpg import DDPG

        agent = DDPG(
            env,
            {
                "device": "cuda",
                "gamma": GAMMA,
                "batch_size": 128,
                "hidden": 128,
                "lr": 3e-4,
                "buffer_size": 100000,
                "warmup": 1000,
                "update_freq": 1,
                "seed": 0,
                "eval_freq": 10**9,
            },
        )
        agent.train(env, {"steps": online_steps}, _Tracker())
        on_logs = _rollout(env, lambda s: agent.act(s, eval=True), n_ep_online)
    except Exception as ex:
        print(f"  [online agent skipped: {type(ex).__name__}: {ex}]")

    merged = {}
    srcs = [rand_logs] + ([on_logs] if on_logs is not None else [])
    for k in rand_logs:
        merged[k] = np.concatenate([s[k] for s in srcs], 0)
    # unique episode ids across sources
    merged["episode"] = np.concatenate(
        [np.asarray(s["episode"]) + i * 10000 for i, s in enumerate(srcs)]
    )

    canon = odata.to_canonical(
        merged, nA=int(low.size), estimate_propensity=True, source_name=env_name
    )
    b = behavior_stats(canon, GAMMA)

    # --- offline continuous IQL ---
    a_dim = int(canon["act"].shape[1])
    stub = EnvStub(int(canon["obs"].shape[1]), a_dim, a_dim=a_dim, continuous=True)
    cfg = {
        "gamma": GAMMA,
        "hidden": 128,
        "batch_size": 256,
        "seed": 0,
        "offline_steps": offline_steps,
        "device": "cuda",
    }
    iql = train_candidate(
        "iql_cont", stub, canon, cfg, os.path.join(OUT, f"{env_name}_iql_cont.pt")
    )
    iql.name = "iql_cont"

    # --- truth: live discounted return of the offline policy vs behavior ---
    truth_iql = _disc_return(env, iql, gt_episodes)
    # behavior truth = logged behavior's own discounted return (anchor)
    anchor = float(b["mean"])

    # --- OPE ---
    panels = {
        "iql_cont": C.panel_continuous(
            canon,
            iql,
            gamma=GAMMA,
            fast=True,
            cand_id="iql_cont",
            fqe_cfg={
                "steps_max": 20000,
                "eval_every": 2500,
                "patience": 8,
                "batch": 512,
                "hidden": 96,
            },
        )
    }
    rows = G.adjudicate(
        panels, b["mean"], b["std"], None, None, n_episodes=len(np.unique(canon["episode"]))
    )
    v = J.judge_diet(rows, b["mean"], b["std"], None, 0.5)
    dec = v["decisions"]["iql_cont"]
    return {
        "env": env_name,
        "obs_dim": int(canon["obs"].shape[1]),
        "a_dim": a_dim,
        "N": int(canon["N"]),
        "behavior_mean": round(anchor, 2),
        "bar": v["bar"],
        "iql_truth": round(truth_iql, 2),
        "improved": bool(truth_iql > anchor),
        "fqe": panels["iql_cont"]["fqe_dm"],
        "dr": panels["iql_cont"]["dr"],
        "sharp": panels["iql_cont"]["sharp_dm"],
        "ess": panels["iql_cont"]["ess_frac"],
        "deploy": bool(dec["deploy"]),
        "rule": dec["reasons"][0],
        "cert_lo": (dec.get("certificate") or {}).get("lo"),
        "cert_hi": (dec.get("certificate") or {}).get("hi"),
    }


if __name__ == "__main__":
    envs = sys.argv[1:] or ["mountaincar_continuous", "pendulum"]
    os.makedirs(OUT, exist_ok=True)
    res = []
    for e in envs:
        try:
            r = survey_env(e)
            res.append(r)
            print(f"\n=== {e} === obs={r['obs_dim']} a_dim={r['a_dim']} N={r['N']}")
            print(
                f"  behavior(anchor)={r['behavior_mean']} bar={r['bar']} "
                f"iql_truth={r['iql_truth']} improved={r['improved']}"
            )
            print(
                f"  OPE: FQE={r['fqe']} DR={r['dr']} sharp={r['sharp']} "
                f"ess={r['ess']} deploy={r['deploy']} :: {r['rule'][:60]}"
            )
        except Exception as ex:
            import traceback

            print(f"\n=== {e} FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
    with open(os.path.join(OUT, "continuous.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
