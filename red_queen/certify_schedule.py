"""Certify the red_queen schedule with white_queen (offline RL + OPE).

Builds a canonical sequential dataset from the multi-action decision log
(action = a discrete bucket of actions/epoch), then runs white_queen's offline-RL
pipeline to LEARN and CERTIFY a policy with a DEPLOY/HOLD verdict + certificate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LOG = Path(__file__).resolve().parents[0] / "artifacts" / "decision_log.npz"


def _discretize(n, nA=4):
    edges = np.unique(np.quantile(n, np.linspace(0, 1, nA + 1)[1:-1]))
    return np.digitize(n, edges).astype(np.int64)


def run(seed=0, max_steps=12000):

    z = np.load(LOG)
    S = z["state"]
    S2 = z["next_state"]
    R = z["reward"]
    D = z["done"]
    A = z["action"][:, 0]
    act = _discretize(A)
    nA = int(act.max()) + 1
    idx = np.arange(len(S))
    if len(idx) > max_steps:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, max_steps, replace=False)
    src = {"obs": S[idx], "act": act[idx], "rew": R[idx], "next_obs": S2[idx], "done": D[idx]}
    from white_queen.tribunal.ope.pipeline import run as wq_run

    rep = wq_run(
        src,
        algorithms=("iql", "cql", "bc"),
        nA=nA,
        fast=True,
        ensemble_K=2,
        offline_steps=1500,
        seed=seed,
    )
    print("== WHITE_QUEEN CERTIFICATION of the red_queen sequential task ==")
    print(f"  candidates        : {rep.get('n_candidates')}")
    print(f"  deployed          : {rep.get('deployed')}")
    print(f"  behavior value    : {rep.get('behavior_mean')}")
    print(f"  bar               : {rep.get('bar')}")
    decs = rep.get("decisions", {})
    for name, d in decs.items():
        print(f"    {name:6s}: deploy={d.get('deploy')} witnesses={d.get('witnesses')}")
    return {
        "deployed": rep.get("deployed"),
        "behavior": rep.get("behavior_mean"),
        "bar": rep.get("bar"),
        "n_candidates": rep.get("n_candidates"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run()
