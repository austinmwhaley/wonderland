"""Decisive red_king <-> white_queen A/B — long-horizon sequential, KNOWN truth.

The last unproven cell of v1 DoD #4. Prior art established:
  * bandit + hard-OPE: decisions IDENTICAL with/without (STATUS "HARD-OPE TEST");
  * sequential email: value estimate better (rho +0.10) but NO decision flip
    (witness count drops 4->3, deploy carried by DR/FQE);
  * open question on record: "red_king's marginal value must live where
    white_queen's internal MB is weak — LONG-HORIZON SEQUENTIAL MDPs —
    which we have not built a ground-truth test for."

This script builds that missing test:
  * generator: synthetic.make_sequential (known dynamics, exact truth via
    Monte-Carlo under the TRUE model: a* = argmax(s[:nA]), r = 1[a==a*]);
  * cells: horizon T in {5, 20, 50} x overlap behavior_eps in {0.4, 0.15, 0.05}
    (long-horizon + low-overlap regimes);
  * candidates: optimal / near(0.8) / mimic-behavior / uniform / anti —
    scored per cell: SHOULD_DEPLOY iff true value beats true behavior value
    by > 2 MC SEs;
  * arms: WITHOUT (stock panel->adjudicate->judge) |
          WITH (panel["mb"] <- red_king ensemble multi-step rollout, pessimistic se) |
          WITH_BOTH (also mb_sharp <- red_king argmax rollout).
    Integration point identical to prior art (red_king/wq_native.py).

PRE-COMMITTED SCORING RULE (recorded before running — receipts doctrine):
  SHIP red_king into the decision path iff the WITH arm, vs WITHOUT:
    (a) fixes >=1 decision toward truth (catches a false deploy OR flips a
        correct hold->deploy), AND
    (b) introduces ZERO new errors on any candidate in any cell
        (never ships worse than logging; never newly misses a true deploy).
  OTHERWISE REMOVE red_king from the decision path (analyst tool only).

Receipt -> red_king/artifacts/ab_witness.json; verdict printed mechanically
from the rule (no post-hoc judgement).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "artifacts" / "ab_witness.json"
GAMMA = 0.99
CELLS = [(5, 0.4), (20, 0.4), (50, 0.4), (20, 0.15), (50, 0.05)]
N_MC = 20_000
RK_K = 3
RK_STEPS = 2500
WQ_FQE = {
    "device": "cpu",
    "steps_max": 8000,
    "eval_every": 500,
    "patience": 5,
    "batch": 256,
    "hidden": 64,
    "allow_under_budget": True,
}


# --------------------------------------------------------------------------
# candidates (all expose act + action_probs)
# --------------------------------------------------------------------------
class UniformPolicy:
    def __init__(self, nA):
        self.nA = int(nA)

    def act(self, state, eval=True):
        return 0

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs))
        return np.full((len(o), self.nA), 1.0 / self.nA, dtype=np.float32)


class NearOptPolicy:
    """a* w.p. mix, else uniform (soft version of the optimal)."""

    def __init__(self, nA, mix=0.8):
        self.nA, self.mix = int(nA), float(mix)

    def act(self, state, eval=True):
        return int(np.argmax(np.asarray(state)[: self.nA]))

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs))
        a = o[:, : self.nA].argmax(1)
        out = np.full((len(o), self.nA), (1.0 - self.mix) / self.nA, dtype=np.float32)
        out[np.arange(len(o)), a] += self.mix
        return out


class BehaviorPolicy:
    """Clone of make_sequential's behavior for the cell's eps (true value == bar)."""

    def __init__(self, nA, eps):
        self.nA, self.eps = int(nA), float(eps)

    def act(self, state, eval=True):
        return int(np.argmax(np.asarray(state)[: self.nA]))

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs))
        a = o[:, : self.nA].argmax(1)
        out = np.full((len(o), self.nA), self.eps / self.nA, dtype=np.float32)
        out[np.arange(len(o)), a] += 1.0 - self.eps
        return out


class AntiPolicy:
    """argmin(s[:nA]): never the optimal action (true value ~ 0)."""

    def __init__(self, nA):
        self.nA = int(nA)

    def act(self, state, eval=True):
        return int(np.argmin(np.asarray(state)[: self.nA]))

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs))
        a = o[:, : self.nA].argmin(1)
        out = np.zeros((len(o), self.nA), dtype=np.float32)
        out[np.arange(len(o)), a] = 1.0
        return out


def candidates(nA, eps):
    from white_queen.tribunal.ope.synthetic import OptimalActionPolicy

    return {
        "optimal": OptimalActionPolicy(nA),
        "near": NearOptPolicy(nA, mix=0.8),
        "mimic": BehaviorPolicy(nA, eps),
        "uniform": UniformPolicy(nA),
        "anti": AntiPolicy(nA),
    }


# --------------------------------------------------------------------------
# exact truth under the TRUE generator
# --------------------------------------------------------------------------
def true_value(cand, T, eps, noise=0.05, d=6, nA=4, n_mc=N_MC, seed=0):
    """Monte-Carlo value under make_sequential's true dynamics (exact model)."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n_mc, d)).astype(np.float32)
    total = np.zeros(n_mc, dtype=np.float64)
    for t in range(T):
        p = np.asarray(cand.action_probs(s), dtype=np.float64)
        a_star = s[:, :nA].argmax(1)
        r = p[np.arange(n_mc), a_star]
        total += (GAMMA**t) * r
        s = s + noise * rng.normal(size=(n_mc, d)).astype(np.float32)
    return float(total.mean()), float(total.std() / np.sqrt(n_mc))


def behavior_value(T, eps, noise=0.05, d=6, nA=4, n_mc=N_MC, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n_mc, d)).astype(np.float32)
    total = np.zeros(n_mc, dtype=np.float64)
    for t in range(T):
        # E[r] = P(chosen == a*) under behavior = 1 - eps + eps/nA
        total += (GAMMA**t) * (1.0 - eps + eps / nA)
        s = s + noise * rng.normal(size=(n_mc, d)).astype(np.float32)
    return float(total.mean()), float(total.std() / np.sqrt(n_mc))


# --------------------------------------------------------------------------
# red_king-style witness: ensemble (s, a) -> (ds, r) multi-step rollout
# --------------------------------------------------------------------------
def train_rk(diet, nA, seed=0, K=RK_K, steps=RK_STEPS):
    import torch
    import torch.nn as nn

    obs = np.asarray(diet["obs"], np.float32)
    act = np.asarray(diet["act"], np.int64)
    rew = np.asarray(diet["rew"], np.float32)
    nxt = np.asarray(diet["obs2"], np.float32)
    dim = obs.shape[1]
    dstd = float((nxt - obs).std() + 1e-6)
    rstd = float(rew.std() + 1e-6)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + nA, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, dim + 1),
            )

        def forward(self, s, oh):
            o = self.net(torch.cat([s, oh], -1))
            return o[:, :dim], o[:, dim:]

    St = torch.tensor(obs)
    At = torch.tensor(act)
    Rt = torch.tensor((rew / rstd).astype(np.float32)).unsqueeze(-1)
    DS = torch.tensor(((nxt - obs) / dstd).astype(np.float32))
    ens, opts = [], []
    for i in range(K):
        torch.manual_seed(seed + i)
        m = M()
        ens.append(m)
        opts.append(torch.optim.Adam(m.parameters(), lr=1e-3))
    g = torch.Generator().manual_seed(seed)
    for m, opt in zip(ens, opts):
        for _ in range(steps):
            ix = torch.randint(0, len(obs), (256,), generator=g)
            oh = nn.functional.one_hot(At[ix], nA).float()
            ds, r = m(St[ix], oh)
            loss = nn.functional.mse_loss(ds, DS[ix]) + nn.functional.mse_loss(r, Rt[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {"ens": ens, "dstd": dstd, "rstd": rstd, "dim": dim}


def rk_rollout(rk, starts, cand, T, nA, gamma=GAMMA):
    """Closed-loop expected-action rollout from episode starts; returns (mb, se)."""
    import torch
    import torch.nn as nn

    ens, dstd, rstd = rk["ens"], rk["dstd"], rk["rstd"]
    s0 = torch.tensor(np.asarray(starts, np.float32))
    B = len(s0)
    s = s0.clone()
    acc = torch.zeros(B)
    with torch.no_grad():
        for t in range(T):
            p = torch.tensor(np.asarray(cand.action_probs(s.numpy()), np.float32))
            # all-action batch: (B*nA) inputs -> per-action (ds, r), then mix by p
            s_rep = s.unsqueeze(1).expand(-1, nA, -1).reshape(B * nA, -1)
            oh = nn.functional.one_hot(torch.arange(nA).repeat(B), nA).float()
            ds_all, r_all = [], []
            for m in ens:
                ds, r = m(s_rep, oh)
                ds_all.append(ds)
                r_all.append(r)
            ds = torch.stack(ds_all).mean(0).view(B, nA, -1)
            r = torch.stack(r_all).mean(0).view(B, nA)
            ds = (ds * p.unsqueeze(-1)).sum(1)
            r = (r * p).sum(1) * rstd
            acc = r + gamma * acc
            s = s + ds * dstd
    vals = acc.numpy()
    return float(vals.mean()), float(vals.std() / max(np.sqrt(len(vals)), 1.0))


# --------------------------------------------------------------------------
# one A/B cell
# --------------------------------------------------------------------------
def run_cell(T, eps, nA=4, d=6, seed=0):
    from white_queen.tribunal.ope import (
        data as _data,
        estimators as _E,
        gate as _gate,
        judge as _judge,
    )
    from white_queen.tribunal.ope.receipts import behavior_stats
    from white_queen.tribunal.ope.synthetic import make_sequential

    t0 = time.perf_counter()
    n = max(6000, 300 * T)
    logs, _info = make_sequential(n=n, d=d, nA=nA, T=T, seed=seed, behavior_eps=eps, noise=0.05)
    # LAB HONESTY: the generator's logging policy is known exactly — inject the
    # TRUE propensity column instead of estimating it (estimated propensity on
    # this generator wrecks IS/DR: ESS ~5%, certificates can never clear and
    # nothing can deploy — the battery would be vacuous).
    logs = dict(logs)
    a_star = logs["obs"][:, :nA].argmax(1)
    logs["propensity"] = np.where(logs["act"] == a_star, 1.0 - eps + eps / nA, eps / nA).astype(
        np.float32
    )
    diet = _data.to_canonical(logs, nA=nA)
    prov = diet.get("provenance", {}).get("propensity")
    assert prov == "provided", f"propensity provenance must be provided, got {prov}"
    b = behavior_stats(diet, GAMMA)
    ne = len(np.unique(diet["episode"]))
    starts = diet["obs"][diet["t"] == 0]
    if len(starts) > 300:
        starts = starts[:300]

    rk = train_rk(diet, nA, seed=seed)
    bar_true, bar_se = behavior_value(T, eps, d=d, nA=nA, seed=seed + 7)

    records = []
    for name, cand in candidates(nA, eps).items():
        panel = _E.panel(diet, cand, GAMMA, fast=True, ensemble_K=2, fqe_cfg=dict(WQ_FQE))
        rk_mb, rk_se = rk_rollout(rk, starts, cand, T, nA)

        class _Sharp:
            def act(self, state, eval=True):
                return int(np.asarray(cand.action_probs([state])).argmax())

            def action_probs(self, obs, temperature=1.0):
                o = np.atleast_2d(np.asarray(obs))
                a = np.asarray(cand.action_probs(o)).argmax(1)
                out = np.zeros((len(o), nA), dtype=np.float32)
                out[np.arange(len(o)), a] = 1.0
                return out

        rk_sharp_mb, rk_sharp_se = rk_rollout(rk, starts, _Sharp(), T, nA)

        arms = {}
        for arm in ("without", "with", "with_both"):
            p = dict(panel)
            if arm in ("with", "with_both"):
                p["mb"] = {"mb": rk_mb, "se": rk_se, "sims": RK_K}
            if arm == "with_both":
                p["mb_sharp"] = {"mb": rk_sharp_mb, "se": rk_sharp_se, "sims": RK_K}
            rows = _gate.adjudicate({name: p}, b["mean"], b["std"], None, None, n_episodes=ne)
            dec = _judge.judge_diet(rows, b["mean"], b["std"], None)["decisions"][name]
            arms[arm] = {"deploy": bool(dec["deploy"]), "witnesses": dec.get("witnesses")}

        v_true, v_se = true_value(cand, T, eps, d=d, nA=nA, seed=seed + 11)
        # genuinely better: > 2 MC-SEs AND a practical margin (>=1% of bar) so
        # float32-epsilon ties (mimic == behavior) can never count as deploys.
        thr = max(2.0 * np.sqrt(v_se**2 + bar_se**2), 0.01 * bar_true)
        should = bool((v_true - bar_true) > thr)
        rec = {
            "candidate": name,
            "T": T,
            "eps": eps,
            "truth_value": round(v_true, 3),
            "truth_se": round(v_se, 3),
            "bar_true": round(bar_true, 3),
            "should_deploy": should,
            "wq_mb": panel["mb"].get("mb") if isinstance(panel["mb"], dict) else None,
            "rk_mb": round(rk_mb, 3),
            "rk_se": round(rk_se, 3),
            **{f"{a}_{k}": v for a, arm in arms.items() for k, v in arm.items()},
        }
        records.append(rec)
        print(
            f"  T={T:<3} eps={eps:<5} {name:<8} should={'DEPLOY' if should else 'HOLD':<7} "
            f"without={arms['without']['deploy']!s:<5}(w{arms['without']['witnesses']}) "
            f"with={arms['with']['deploy']!s:<5}(w{arms['with']['witnesses']}) "
            f"both={arms['with_both']['deploy']!s:<5}"
            f"(w{arms['with_both']['witnesses']})",
            flush=True,
        )
    print(f"  cell T={T} eps={eps} wall {time.perf_counter() - t0:.1f}s", flush=True)
    return records


# --------------------------------------------------------------------------
# pre-committed rule application
# --------------------------------------------------------------------------
def apply_rule(records, arm):
    fixed, broken = [], []
    for r in records:
        truth = r["should_deploy"]
        wo, wi = r["without_deploy"], r[f"{arm}_deploy"]
        if wo != truth and wi == truth:
            fixed.append(f"{r['candidate']}(T{r['T']},e{r['eps']})")
        if wo == truth and wi != truth:
            broken.append(f"{r['candidate']}(T{r['T']},e{r['eps']})")
    ship = bool(fixed) and not broken
    return {"fixed": fixed, "broken": broken, "ship": ship}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=int, default=len(CELLS), help="run first N cells")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print("== DECISIVE red_king x white_queen A/B (sequential, known truth) ==")
    print(
        f"pre-committed rule: SHIP iff WITH fixes >=1 AND breaks 0 (cells: {CELLS[: args.cells]})",
        flush=True,
    )
    records = []
    for i, (T, eps) in enumerate(CELLS[: args.cells]):
        print(f"-- cell T={T} eps={eps} --", flush=True)
        records.extend(run_cell(T, eps, seed=args.seed + i))

    verdict_with = apply_rule(records, "with")
    verdict_both = apply_rule(records, "with_both")
    err_without = sum(r["should_deploy"] != r["without_deploy"] for r in records)
    err_with = sum(r["should_deploy"] != r["with_deploy"] for r in records)
    err_both = sum(r["should_deploy"] != r["with_both_deploy"] for r in records)
    print("== SCORECARD (vs exact truth) ==")
    print(f"  candidates        : {len(records)}")
    print(f"  errors WITHOUT    : {err_without}")
    print(
        f"  errors WITH (mb)  : {err_with}  fixed={verdict_with['fixed']}  broken={verdict_with['broken']}"
    )
    print(
        f"  errors WITH_BOTH  : {err_both}  fixed={verdict_both['fixed']}  broken={verdict_both['broken']}"
    )
    print(
        f"  VERDICT (rule on WITH)     : {'SHIP red_king' if verdict_with['ship'] else 'REMOVE from decision path'}"
    )
    print(
        f"  VERDICT (rule on WITH_BOTH): {'SHIP red_king' if verdict_both['ship'] else 'REMOVE from decision path'}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "rule": "SHIP iff WITH fixes >=1 decision toward truth AND breaks 0",
                "cells": [list(c) for c in CELLS[: args.cells]],
                "records": records,
                "errors": {"without": err_without, "with": err_with, "with_both": err_both},
                "verdict_with": verdict_with,
                "verdict_both": verdict_both,
            },
            indent=1,
            default=str,
        )
    )
    print(f"receipt -> {OUT}")
    return 0 if verdict_with["ship"] or verdict_both["ship"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
