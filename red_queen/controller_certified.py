"""red_queen controller driven by a white_queen-CERTIFIED policy.

Trains offline-RL candidates on the multi-cadence decision log, certifies them
(DEPLOY/HOLD), and — only if a candidate is deployed — uses that policy to choose
the number of actions per epoch across cadences, under constraints.

This is the future-state wiring: white_queen LEARNS+CERTIFIES, red_queen CONTROLS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import os

LOG = Path(
    os.environ.get(
        "DECISION_LOG", Path(__file__).resolve().parents[0] / "artifacts" / "decision_log.npz"
    )
)
OUT = Path(__file__).resolve().parents[0] / "artifacts" / "nba_schedule_certified.json"
GAMMA = 0.99
PER_EPOCH_CAP = 8


def _discretize(n, nA=4):
    edges = np.unique(np.quantile(n, np.linspace(0, 1, nA + 1)[1:-1]))
    b = np.digitize(n, edges)
    rep = {i: float(n[b == i].mean()) if (b == i).any() else 0.0 for i in range(int(b.max()) + 1)}
    return b.astype(np.int64), rep, int(b.max()) + 1


def run(seed=0, global_budget=None):
    from white_queen.tribunal.ope import data as _data
    from white_queen.tribunal.ope.pipeline import train_candidates, evaluate_pool

    z = np.load(LOG)
    S = z["state"]
    S2 = z["next_state"]
    R = z["reward"]
    D = z["done"]
    n = z["action"][:, 0]
    act, rep, nA = _discretize(n)
    # subsample for training/certification (speed); apply policy to ALL rows
    sub = np.arange(len(S))
    if len(sub) > 15000:
        sub = np.random.default_rng(seed).choice(sub, 15000, replace=False)
    diet = _data.to_canonical(
        {"obs": S[sub], "act": act[sub], "rew": R[sub], "next_obs": S2[sub], "done": D[sub]}, nA=nA
    )
    handles = train_candidates(
        diet, algorithms=("iql",), gamma=GAMMA, offline_steps=1500, seed=seed
    )
    rep_report = evaluate_pool(diet, handles, gamma=GAMMA, fast=True, ensemble_K=2)
    raw = rep_report.get("deployed") or []
    decs = rep_report.get("decisions", {})
    # ROBUSTNESS GATE (real-world hardening): require a corroborating witness
    # (>=1) in addition to the certificate; white_queen's deploy is certificate-
    # based, so a 0-witness deploy is treated as HOLD here.
    deployed = [n for n in raw if decs.get(n, {}).get("witnesses", 0) >= 1]
    if raw and not deployed:
        print("  [robustness] certificate-only deploy (0 witnesses) -> HOLD")
    policy = handles.get("iql")
    if global_budget is None:
        global_budget = float(len(S) * 1.0)
    ep_cad = {0: "daily", 1: "weekly", 2: "monthly"}
    plan, used = [], 0.0
    # CERTIFICATION GATE: only act if a candidate is DEPLOYED (else HOLD).
    if not deployed:
        print("  HOLD: no candidate certified better than logging -> no actions scheduled")
    else:
        for i in range(len(S)):
            a = int(policy.act(S[i]))
            desired = min(rep.get(a, 0.0), PER_EPOCH_CAP)
            take = int(min(desired, max(0.0, global_budget - used)))  # floor: no overshoot
            used += take
            plan.append(
                {
                    "epoch": int(i),
                    "cadence": ep_cad[int(z["cadence"][i])],
                    "action_bucket": a,
                    "actions": take,
                }
            )
            if used >= global_budget:
                break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"deployed": deployed, "budget": global_budget, "used": used, "plan": plan})
    )
    by_cad = {}
    for p in plan:
        by_cad[p["cadence"]] = by_cad.get(p["cadence"], 0) + p["actions"]
    print("== RED_QUEEN driven by CERTIFIED white_queen policy ==")
    print(f"  certified deployed    : {deployed}")
    print(f"  behavior / bar        : {rep_report.get('behavior_mean')} / {rep_report.get('bar')}")
    print(f"  epochs planned        : {len(plan)}")
    print(f"  touches used / budget : {round(used, 1)} / {round(global_budget, 1)}")
    print(f"  touches by cadence    : {by_cad}")
    print(f"  schedule -> {OUT}")
    return {"deployed": deployed, "epochs": len(plan), "used": used, "by_cadence": by_cad}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    run(global_budget=a.budget)
