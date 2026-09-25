"""red_queen sequential controller.

Consumes the multi-cadence, multi-action DECISION LOG and emits, per
(customer, epoch), an ACTION SET (n touches) across daily/weekly/monthly epochs,
maximising predicted discounted incremental margin under constraints:
  * per-epoch cap
  * global send budget (allocation across epochs)
  * fail-safe (act only where predicted incremental value > 0)

Currently the policy is a value model over (state, n_actions) -> reward; it is
policy-agnostic (can be swapped for a certified white_queen policy or red_king
counterfactual rollouts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
LOG = Path(__file__).resolve().parents[0] / "artifacts" / "decision_log.npz"
OUT = Path(__file__).resolve().parents[0] / "artifacts" / "nba_schedule.json"
GAMMA = 0.999
PER_EPOCH_CAP = 8  # max touches per epoch (hard business cap)
N_GRID = 16  # candidate action frequencies


def _fit_value(z, seed=0):
    """Predict incremental GP from (state, n_actions). Ridge (fast, calibrated)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    S = z["state"]
    n = z["action"][:, 0:1]
    X = np.hstack([S, n]).astype(np.float32)
    y = z["reward"].astype(np.float32)
    sc = StandardScaler().fit(X)
    m = Ridge(alpha=1.0).fit(sc.transform(X), y)
    return sc, m


def run(global_budget=None, seed=0):
    z = np.load(LOG)
    S = z["state"]
    TID = z["traj"]
    DT = z["dt_days"]
    CAD = z["cadence"]
    sc, m = _fit_value(z, seed)
    # per-epoch candidate values across a grid of action counts
    grid = np.linspace(0, PER_EPOCH_CAP, N_GRID).astype(np.float32)
    # group epochs by trajectory
    order = np.argsort(TID, kind="stable")
    if global_budget is None:
        global_budget = float(len(order) * 1.0)  # derived capacity: ~1 touch/epoch
    plan = []
    used = 0.0
    episode_cad = {0: "daily", 1: "weekly", 2: "monthly"}
    for t in np.unique(TID):
        idx = order[TID[order] == t]
        for i in idx:
            Xg = np.hstack([np.repeat(S[i][None, :], N_GRID, 0), grid[:, None]]).astype(np.float32)
            vg = m.predict(sc.transform(Xg))
            # choose the best affordable count with positive value
            best_n, best_v = 0.0, 0.0
            for gi in np.argsort(-vg):
                n = float(grid[gi])
                if vg[gi] > 0 and n <= (global_budget - used) and n <= PER_EPOCH_CAP:
                    best_n, best_v = n, float(vg[gi])
                    break
            used += best_n
            plan.append(
                {
                    "epoch": int(i),
                    "cadence": episode_cad[int(CAD[i])],
                    "actions": int(round(best_n)),
                    "expected_incremental_gp": round(best_v, 2),
                    "discount": round(float(GAMMA ** float(DT[i])), 4),
                }
            )
            if used >= global_budget:
                break
        if used >= global_budget:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"plan": plan, "budget": global_budget, "used": used}))
    acted = sum(1 for p in plan if p["actions"] > 0)
    by_cad = {}
    for p in plan:
        by_cad[p["cadence"]] = by_cad.get(p["cadence"], 0) + p["actions"]
    print("== RED_QUEEN SEQUENTIAL CONTROLLER ==")
    print(f"  epochs planned        : {len(plan)}")
    print(f"  epochs with actions   : {acted}")
    print(f"  touches used / budget : {round(used, 1)} / {round(global_budget, 1)}")
    print(f"  touches by cadence    : {by_cad}")
    print(f"  schedule -> {OUT}")
    return {"epochs": len(plan), "acted": acted, "used": used, "by_cadence": by_cad}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    run(a.budget)
