"""HARD-OPE test — the decisive red_king <-> white_queen experiment.

Low-overlap contextual bandit (sharp logging) with KNOWN ground truth. The
optimal policy is genuinely better than logging. Question:

  WITHOUT red_king : can white_queen's own estimators certify and DEPLOY the
                     optimal policy? (model-free fails under poor overlap)
  WITH    red_king : does an external model-based value (a reward model fit on
                     logs) let white_queen make the CORRECT decision?

Ground truth is known, so we can score the decision, not just stability.
"""

from __future__ import annotations

import argparse

import numpy as np


def _red_king_mb(canon, cand, nA, seed=0, K=5, steps=2000):
    """Model-based value: ensemble reward model r(s,a), averaged at the candidate."""
    import torch
    import torch.nn as nn

    obs = np.asarray(canon["obs"], np.float32)
    act = np.asarray(canon["act"], np.int64)
    rew = np.asarray(canon["rew"], np.float32)
    dim = obs.shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    St = torch.tensor(obs, device=dev)
    At = torch.tensor(act, device=dev)
    Rt = torch.tensor(rew, device=dev)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + nA, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, nA),
            )

        def out(self, s, oh):
            return self.net(torch.cat([s, oh], -1))

    ens = [M().to(dev) for _ in range(K)]
    opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in ens]
    for m, opt in zip(ens, opts):
        torch.manual_seed(seed)
        for _ in range(steps):
            b = torch.randint(0, len(obs), (256,), device=dev)
            oh = nn.functional.one_hot(At[b], nA).float()
            pred = (m.out(St[b], oh) * oh).sum(-1)
            loss = nn.functional.mse_loss(pred, Rt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        probs = np.asarray(cand.action_probs(obs))
        a_star = probs.argmax(1)
        oh = nn.functional.one_hot(torch.tensor(a_star, device=dev), nA).float()
        vals = torch.stack([(m.out(St, oh) * oh).sum(-1) for m in ens])
        return float(vals.mean(0).mean()), float(vals.std(0).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logging_temp", type=float, default=0.2)
    a = ap.parse_args(argv)
    from white_queen.tribunal.ope.synthetic import make_bandit, GreedyPolicy
    from white_queen.tribunal.ope import (
        data as _data,
        estimators as _E,
        gate as _gate,
        judge as _judge,
    )
    from white_queen.tribunal.ope.receipts import behavior_stats

    logs, info = make_bandit(
        n=4000, d=6, nA=4, seed=1, reward_scale=3.0, noise=0.3, logging_temp=a.logging_temp
    )
    nA = info["nA"]
    diet = _data.to_canonical(logs, nA=nA)
    b = behavior_stats(diet, 0.99)
    cand = GreedyPolicy(info["reward_W"], nA)  # the GENUINELY optimal policy
    panel = _E.panel(diet, cand, 0.99, fast=True, ensemble_K=2)
    ne = len(np.unique(diet["episode"]))
    rows = _gate.adjudicate({"opt": panel}, b["mean"], b["std"], None, None, n_episodes=ne)
    dec0 = _judge.judge_diet(rows, b["mean"], b["std"], None)["decisions"]["opt"]
    rk_val, rk_se = _red_king_mb(logs, cand, nA)
    panel2 = dict(panel)
    panel2["mb"] = {"mb": rk_val, "se": rk_se, "sims": 5}
    rows2 = _gate.adjudicate({"opt": panel2}, b["mean"], b["std"], None, None, n_episodes=ne)
    dec1 = _judge.judge_diet(rows2, b["mean"], b["std"], None)["decisions"]["opt"]
    print("== HARD-OPE TEST (low overlap) ==")
    print(f"  logging_temp         : {a.logging_temp}")
    print(
        f"  TRUE best(value)     : {info['mean_reward_best']:.3f}  behavior {info['mean_reward_behavior']:.3f}"
    )
    print(f"  white_queen mb       : {panel['mb'].get('mb')}")
    print(f"  red_king mb          : {rk_val:.3f} (se {rk_se:.3f})")
    print(f"  WITHOUT red_king     : deploy={dec0['deploy']} witnesses={dec0.get('witnesses')}")
    print(f"  WITH    red_king     : deploy={dec1['deploy']} witnesses={dec1.get('witnesses')}")
    print("  CORRECT decision     : DEPLOY (optimal is truly better)")
    return {
        "true_best": info["mean_reward_best"],
        "behavior": info["mean_reward_behavior"],
        "without": dec0["deploy"],
        "with": dec1["deploy"],
    }


if __name__ == "__main__":
    main()
