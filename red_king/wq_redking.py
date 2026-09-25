"""white_queen x red_king: inject the VALIDATED red_king rollout value into
white_queen's native panel['mb'] and compare the decision (with vs without).
red_king ordering was validated against the known causal effect (spearman 1.0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[0] / "data" / "seq_email.npz"
GAMMA = 0.99


def main(seed=0):
    from white_queen.tribunal.ope import (
        data as _data,
        estimators as _E,
        gate as _gate,
        judge as _judge,
    )
    from white_queen.tribunal.ope.receipts import behavior_stats
    from red_king.rssm import rollout_arm_values

    z = np.load(DATA)
    S, A, R, S2, D = z["S"], z["A"], z["R"], z["S2"], z["D"]
    nA = int(A.max()) + 1
    means = np.array([R[A == a].mean() if (A == a).any() else -1e9 for a in range(nA)])
    astar = int(means.argmax())

    class ConstPolicy:
        def act(self, state, eval=True):
            return astar

        def action_probs(self, obs, temperature=1.0):
            o = np.asarray(obs)
            n = len(o) if o.ndim > 1 else 1
            p = np.zeros((n, nA), np.float32)
            p[:, astar] = 1.0
            return p

    diet = _data.to_canonical({"obs": S, "act": A, "rew": R, "next_obs": S2, "done": D}, nA=nA)
    b = behavior_stats(diet, GAMMA)
    cand = ConstPolicy()
    panel = _E.panel(diet, cand, GAMMA, fast=True, ensemble_K=2)
    ne = len(np.unique(diet["episode"]))
    rows = _gate.adjudicate({"c": panel}, b["mean"], b["std"], None, None, n_episodes=ne)
    dec0 = _judge.judge_diet(rows, b["mean"], b["std"], None)["decisions"]["c"]
    # validated red_king value of the constant-arm candidate
    V, SD = rollout_arm_values(S)
    rk_val = float(V[:, astar].mean())
    rk_se = float(SD[:, astar].mean())
    panel2 = dict(panel)
    panel2["mb"] = {"mb": rk_val, "se": rk_se, "sims": 3}
    rows2 = _gate.adjudicate({"c": panel2}, b["mean"], b["std"], None, None, n_episodes=ne)
    dec1 = _judge.judge_diet(rows2, b["mean"], b["std"], None)["decisions"]["c"]
    print("== WHITE_QUEEN WITH VALIDATED red_king ==  candidate = arm", astar)
    print(f"  white_queen mb (without): {panel['mb'].get('mb')}")
    print(f"  red_king     mb (with)  : {rk_val:.2f} (se {rk_se:.2f})")
    print(f"  WITHOUT red_king: deploy={dec0['deploy']} witnesses={dec0.get('witnesses')}")
    print(f"  WITH    red_king: deploy={dec1['deploy']} witnesses={dec1.get('witnesses')}")


if __name__ == "__main__":
    main()
