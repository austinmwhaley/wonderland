"""Decisive A/B v2 — the WORLD-ENGINE test (action-dependent, partially observed).

Why v1 was structurally unfair to red_king (the product hypothesis):
`make_sequential` has action-INDEPENDENT dynamics (`s = s + noise*randn`) and
one-step rewards — there is nothing to simulate forward, so a dedicated
counterfactual world engine could not demonstrate value there. The REMOVE
verdict of v1 stands for that problem class; it does NOT settle whether a
world engine pays off when actions shape the future and the relevant state is
hidden.

This world is built so a world engine is NECESSARY:
  * stock-and-fatigue marketing MDP: contact intensity builds "stock"
    (future revenue) AND "fatigue" (damps future revenue) -> investing now
    pays off later, overspending hurts later: real action->state->reward
    chains;
  * PARTIAL OBSERVABILITY: stock/fatigue are hidden; observations are noisy
    projections + distractors. white_queen's internal MB is a feedforward
    one-step MLP (structurally blind to hidden state); red_king's RSSM-style
    witness is a recurrent GRU ensemble that INFERS the latent stock from
    history — the dedicated-engine hypothesis under test.

Arms: WITHOUT (stock panel incl. internal MLP MB) | WITH (panel["mb"] <-
red_king GRU-ensemble rollout) | WITH_BOTH (+ mb_sharp).
Exact truth: Monte-Carlo under the TRUE dynamics (policy sees observations
only), same scoring rule as v1 (2 MC-SEs AND >=1% of bar).

PRE-COMMITTED SCORING RULE (unchanged, recorded before running):
  SHIP red_king into the decision path iff WITH, vs WITHOUT:
    (a) fixes >=1 decision toward truth, AND
    (b) introduces ZERO new errors (never ships worse, never newly misses).
  OTHERWISE REMOVE (analyst tool only).

Receipt -> red_king/artifacts/ab_world_engine.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "artifacts" / "ab_world_engine.json"
GAMMA = 0.99
NA = 4
D = 6  # [stock_obs, fatigue_obs, 4 distractors]
CELLS = [(30, 0.4), (60, 0.4), (30, 0.15), (60, 0.15)]
N_EP = 400
N_MC = 8000
OBS_NOISE = 0.25
RK_K = 3
RK_EPOCHS = 60
RK_SIMS = 8
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
# TRUE world (hidden stock/fatigue; action-dependent dynamics)
# --------------------------------------------------------------------------
def step_true(x, f, a, rng):
    """Vectorized true transition. x: stock, f: fatigue, a: intensity 0..3."""
    xn = np.clip(0.85 * x + 0.9 * a - 0.4, 0.0, 10.0)
    fn = np.clip(0.7 * f + 0.6 * a, 0.0, 5.0)
    r = 2.0 * x / (1.0 + f) - 0.8 * a * (1.0 + 0.3 * f)
    return xn, fn, r


def observe(x, f, rng):
    ox = (x / 10.0 + OBS_NOISE * rng.standard_normal(x.shape)).reshape(-1, 1)
    of = (f / 5.0 + OBS_NOISE * rng.standard_normal(x.shape)).reshape(-1, 1)
    u = rng.standard_normal((len(x), D - 2))
    return np.concatenate([ox, of, u], 1).astype(np.float32)


def behavior_action(obs, eps, rng):
    """Logging policy: ONE-STEP greedy on obs (revenue is action-independent,
    cost is not -> the myopic logger never invests), eps-uniform for overlap."""
    u = rng.integers(0, NA, size=len(obs))
    return np.where(rng.random(len(obs)) < eps, u, 0).astype(np.int64)


def behavior_probs(obs, eps):
    """Exact P(a | obs) of the logging policy (obs-independent: base = a=0)."""
    p = np.full((len(obs), NA), eps / NA, dtype=np.float64)
    p[:, 0] += 1.0 - eps
    return p


def make_logs(T, eps, seed, n_ep=N_EP):
    rng = np.random.default_rng(seed)
    obs_l, act_l, rew_l, done_l = [], [], [], []
    for _ in range(n_ep):
        x = rng.uniform(0.0, 2.0)
        f = rng.uniform(0.0, 1.0)
        for tstep in range(T):
            obs = observe(np.array([x]), np.array([f]), rng)[0]
            a = int(behavior_action(obs[None, :], eps, rng)[0])
            xn, fn, r = step_true(np.array([x]), np.array([f]), a, rng)
            obs_l.append(obs)
            act_l.append(a)
            rew_l.append(float(r[0]))
            done_l.append(1.0 if tstep == T - 1 else 0.0)
            x, f = float(xn[0]), float(fn[0])
    return {
        "obs": np.stack(obs_l),
        "act": np.asarray(act_l, dtype=np.int64),
        "rew": np.asarray(rew_l, dtype=np.float32),
        "done": np.asarray(done_l, dtype=np.float32),
    }


def _sample_actions(p, rng):
    """Draw one action per row from each row's categorical p."""
    cdf = np.cumsum(p, axis=1)
    u = rng.random(len(p))[:, None]
    return (u > cdf).sum(axis=1).clip(0, NA - 1)


def true_value(cand, T, n_mc=N_MC, seed=0):
    """MC value under TRUE dynamics; the candidate sees observations only.
    Actions are SAMPLED from the policy (dynamics are non-linear in a)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0, size=n_mc)
    f = rng.uniform(0.0, 1.0, size=n_mc)
    total = np.zeros(n_mc, dtype=np.float64)
    for t in range(T):
        obs = observe(x, f, rng)
        p = np.asarray(cand.action_probs(obs), dtype=np.float64)
        p = p / p.sum(1, keepdims=True)
        a = _sample_actions(p, rng)
        xn, fn, r = step_true(x, f, a, rng)
        total += (GAMMA**t) * r
        x, f = xn, fn
    return float(total.mean()), float(total.std() / np.sqrt(n_mc))


def behavior_true_value(T, eps, n_mc=N_MC, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0, size=n_mc)
    f = rng.uniform(0.0, 1.0, size=n_mc)
    total = np.zeros(n_mc, dtype=np.float64)
    for t in range(T):
        obs = observe(x, f, rng)
        p = behavior_probs(obs, eps)
        a = _sample_actions(p, rng)
        xn, fn, r = step_true(x, f, a, rng)
        total += (GAMMA**t) * r
        x, f = xn, fn
    return float(total.mean()), float(total.std() / np.sqrt(n_mc))


# --------------------------------------------------------------------------
# candidates (policy on OBSERVATIONS only — same interface as v1)
# --------------------------------------------------------------------------
class ThresholdPolicy:
    """a = lo when ox < t1; mid when ox < t2; else hi (build/ coast/ harvest)."""

    def __init__(self, t1, t2, lo, mid, hi):
        self.t1, self.t2, self.a = t1, t2, (lo, mid, hi)

    def act(self, state, eval=True):
        o = np.atleast_2d(np.asarray(state))
        return int(
            np.where(
                o[:, 0] < self.t1, self.a[0], np.where(o[:, 0] < self.t2, self.a[1], self.a[2])
            )[0]
        )

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs))
        a = np.where(
            o[:, 0] < self.t1, self.a[0], np.where(o[:, 0] < self.t2, self.a[1], self.a[2])
        )
        out = np.zeros((len(o), NA), dtype=np.float32)
        out[np.arange(len(o)), a] = 1.0
        return out


class ConstPolicy:
    def __init__(self, a):
        self.a = int(a)

    def act(self, state, eval=True):
        return self.a

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs))
        out = np.zeros((len(o), NA), dtype=np.float32)
        out[:, self.a] = 1.0
        return out


class MyopicPolicy:
    """One-step greedy on observed revenue minus cost -> structurally never
    invests (revenue term is action-independent): the shortsighted foil."""

    def act(self, state, eval=True):
        return 0

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs))
        out = np.zeros((len(o), NA), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class MimicPolicy:
    """Exact clone of the logging heuristic (eps-mix) -> true value == bar."""

    def __init__(self, eps):
        self.eps = eps

    def act(self, state, eval=True):
        return 0

    def action_probs(self, obs, temperature=1.0):
        o = np.atleast_2d(np.asarray(obs))
        return behavior_probs(o, self.eps).astype(np.float32)


def candidates(eps):
    return {
        # build early, gentle mid-step, harvest: tuned to beat the heuristic
        "smart": ThresholdPolicy(0.45, 0.75, 3, 1, 0),
        "myopic": MyopicPolicy(),
        "mimic": MimicPolicy(eps),
        "always3": ConstPolicy(3),
        # anti: only "invests" when stock is already high (wastes cost, piles
        # fatigue for zero revenue gain) -> strictly worse than the myopic log
        "anti": ThresholdPolicy(0.6, 0.9, 0, 1, 3),
    }


# --------------------------------------------------------------------------
# red_king-style witness: recurrent GRU ensemble (the world-engine hypothesis)
# --------------------------------------------------------------------------
def train_rk(logs, seed=0, K=RK_K, epochs=RK_EPOCHS):
    import torch
    import torch.nn as nn

    T = int(np.ceil(len(logs["obs"]) / N_EP))
    n_ep = N_EP
    obs = logs["obs"].reshape(n_ep, T, D)
    act = logs["act"].reshape(n_ep, T)
    rew = logs["rew"].reshape(n_ep, T, 1)
    obs_t = torch.tensor(obs)
    act_oh = nn.functional.one_hot(torch.tensor(act), NA).float()
    inp = torch.cat([obs_t, act_oh], -1)  # (E, T, D+NA)
    tgt_o = torch.tensor(obs)  # predict next obs; shift handled by teacher input at t -> out t
    tgt_r = torch.tensor(rew)

    class G(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(D + NA, 64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, D + 1))

        def forward(self, x):
            h, _ = self.gru(x)
            return self.head(h)  # (E, T, D+1)

    ens = []
    for i in range(K):
        torch.manual_seed(seed + i)
        m = G()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(epochs):
            out = m(inp)
            # out[:, t] predicts obs/rew of step t+1 from inputs through t
            loss_o = nn.functional.mse_loss(out[:, :-1, :D], tgt_o[:, 1:, :])
            loss_r = nn.functional.mse_loss(out[:, :-1, D : D + 1], tgt_r[:, 1:, :])
            loss = loss_o + loss_r
            opt.zero_grad()
            loss.backward()
            opt.step()
        ens.append(m)
    return ens


def rk_rollout(ens, starts_obs, cand, T, sims=RK_SIMS, seed=0):
    """Closed-loop simulation of the CANDIDATE inside the GRU world model.

    starts_obs: (n_starts, D) episode-start observations (h0 = 0, fresh episodes).
    For each start: S sampled rollouts under the candidate's action_probs,
    every ensemble member; value = discounted return. Returns (mb, se) with
    se across start-means (episode-level sampling error, same convention as v1).
    """
    import torch

    rng = np.random.default_rng(seed)
    n = len(starts_obs)
    vals = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for i in range(n):
            accs = []
            for _ in range(sims):
                member_returns = []
                for m in ens:
                    h = None
                    o = torch.tensor(starts_obs[i][None, :])  # (1, D)
                    acc = 0.0
                    for t in range(T):
                        p = np.asarray(cand.action_probs(o.numpy()), dtype=np.float64)[0]
                        a = int(rng.choice(NA, p=p / p.sum()))
                        x = torch.cat(
                            [o, torch.nn.functional.one_hot(torch.tensor([a]), NA).float()], -1
                        )
                        out, h = m.gru(x, h)
                        head = m.head(out)  # (1, D+1)
                        r = float(head[0, D])
                        acc = r + GAMMA * acc
                        o = head[0, :D][None, :]
                    member_returns.append(acc)
                accs.append(float(np.mean(member_returns)))  # ensemble mean this sim
            vals[i] = float(np.mean(accs))
    return float(vals.mean()), float(vals.std() / max(np.sqrt(n), 1.0))


# --------------------------------------------------------------------------
# one A/B cell
# --------------------------------------------------------------------------
def run_cell(T, eps, seed=0):
    from white_queen.tribunal.ope import (
        data as _data,
        estimators as _E,
        gate as _gate,
        judge as _judge,
    )
    from white_queen.tribunal.ope.receipts import behavior_stats

    t0 = time.perf_counter()
    logs = make_logs(T, eps, seed)
    # inject TRUE logged propensity (behavior on obs is known exactly)
    p = behavior_probs(logs["obs"], eps)
    logs = dict(logs)
    logs["propensity"] = p[np.arange(len(logs["act"])), logs["act"]].astype(np.float32)
    diet = _data.to_canonical(logs, nA=NA)
    prov = diet.get("provenance", {}).get("propensity")
    assert prov == "provided", f"propensity provenance must be provided, got {prov}"
    b = behavior_stats(diet, GAMMA)
    ne = len(np.unique(diet["episode"]))
    starts_obs = diet["obs"][diet["t"] == 0]

    ens = train_rk(logs, seed=seed)
    bar_true, bar_se = behavior_true_value(T, eps, seed=seed + 7)

    records = []
    for name, cand in candidates(eps).items():
        panel = _E.panel(diet, cand, GAMMA, fast=True, ensemble_K=2, fqe_cfg=dict(WQ_FQE))
        rk_mb, rk_se = rk_rollout(ens, starts_obs[:150], cand, T, seed=seed + 3)

        class _Sharp:
            def act(self, state, eval=True):
                return int(np.asarray(cand.action_probs([state])).argmax())

            def action_probs(self, obs, temperature=1.0):
                o = np.atleast_2d(np.asarray(obs))
                a = np.asarray(cand.action_probs(o)).argmax(1)
                out = np.zeros((len(o), NA), dtype=np.float32)
                out[np.arange(len(o)), a] = 1.0
                return out

        rk_sharp_mb, rk_sharp_se = rk_rollout(ens, starts_obs[:150], _Sharp(), T, seed=seed + 3)

        arms = {}
        for arm in ("without", "with", "with_both"):
            pnl = dict(panel)
            if arm in ("with", "with_both"):
                pnl["mb"] = {"mb": rk_mb, "se": rk_se, "sims": RK_K * RK_SIMS}
            if arm == "with_both":
                pnl["mb_sharp"] = {"mb": rk_sharp_mb, "se": rk_sharp_se, "sims": RK_K * RK_SIMS}
            rows = _gate.adjudicate({name: pnl}, b["mean"], b["std"], None, None, n_episodes=ne)
            dec = _judge.judge_diet(rows, b["mean"], b["std"], None)["decisions"][name]
            arms[arm] = {"deploy": bool(dec["deploy"]), "witnesses": dec.get("witnesses")}

        v_true, v_se = true_value(cand, T, seed=seed + 11)
        thr = max(2.0 * np.sqrt(v_se**2 + bar_se**2), 0.01 * abs(bar_true))
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
            f"  T={T:<3} eps={eps:<5} {name:<8} truth={v_true:8.2f} bar={bar_true:7.2f} "
            f"should={'DEPLOY' if should else 'HOLD':<7} "
            f"without={arms['without']['deploy']!s:<5}(w{arms['without']['witnesses']}) "
            f"with={arms['with']['deploy']!s:<5}(w{arms['with']['witnesses']}) "
            f"both={arms['with_both']['deploy']!s:<5}(w{arms['with_both']['witnesses']})",
            flush=True,
        )
    print(f"  cell T={T} eps={eps} wall {time.perf_counter() - t0:.1f}s", flush=True)
    return records


def truth_only():
    """Pre-flight: verify the candidate set has BOTH directions vs exact truth."""
    print("== TRUTH PREFLIGHT (no panels) ==")
    ok = True
    for T, eps in CELLS:
        bar, bse = behavior_true_value(T, eps, seed=7)
        line = [f"T={T} eps={eps} bar={bar:7.2f} |"]
        n_deploy = 0
        for name, cand in candidates(eps).items():
            v, se = true_value(cand, T, seed=11)
            thr = max(2.0 * np.sqrt(se**2 + bse**2), 0.01 * abs(bar))
            should = (v - bar) > thr
            n_deploy += int(should)
            line.append(f"{name}={v:7.2f}{'*' if should else ' '}")
        if n_deploy == 0:
            ok = False
            line.append("<< NO should-DEPLOY candidate")
        print("  " + "  ".join(line), flush=True)
    print("PREFLIGHT", "OK" if ok else "NEEDS TUNING")
    return 0 if ok else 1


def apply_rule(records, arm):
    fixed, broken = [], []
    for r in records:
        truth = r["should_deploy"]
        wo, wi = r["without_deploy"], r[f"{arm}_deploy"]
        if wo != truth and wi == truth:
            fixed.append(f"{r['candidate']}(T{r['T']},e{r['eps']})")
        if wo == truth and wi != truth:
            broken.append(f"{r['candidate']}(T{r['T']},e{r['eps']})")
    return {"fixed": fixed, "broken": broken, "ship": bool(fixed) and not broken}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth-only", action="store_true", help="pre-flight truth check")
    ap.add_argument("--cells", type=int, default=len(CELLS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.truth_only:
        return truth_only()

    print("== A/B v2: WORLD-ENGINE test (action-dependent, partial observability) ==")
    print("rule: SHIP iff WITH fixes >=1 AND breaks 0", flush=True)
    records = []
    for i, (T, eps) in enumerate(CELLS[: args.cells]):
        print(f"-- cell T={T} eps={eps} --", flush=True)
        records.extend(run_cell(T, eps, seed=args.seed + i))

    vw = apply_rule(records, "with")
    vb = apply_rule(records, "with_both")
    e0 = sum(r["should_deploy"] != r["without_deploy"] for r in records)
    e1 = sum(r["should_deploy"] != r["with_deploy"] for r in records)
    e2 = sum(r["should_deploy"] != r["with_both_deploy"] for r in records)
    print("== SCORECARD (vs exact truth) ==")
    print(f"  candidates        : {len(records)}")
    print(f"  errors WITHOUT    : {e0}")
    print(f"  errors WITH (mb)  : {e1}  fixed={vw['fixed']}  broken={vw['broken']}")
    print(f"  errors WITH_BOTH  : {e2}  fixed={vb['fixed']}  broken={vb['broken']}")
    print(
        f"  VERDICT (rule on WITH)     : {'SHIP red_king' if vw['ship'] else 'REMOVE from decision path'}"
    )
    print(
        f"  VERDICT (rule on WITH_BOTH): {'SHIP red_king' if vb['ship'] else 'REMOVE from decision path'}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "world": "stock-fatigue MDP: action-dependent dynamics + partial observability",
                "rule": "SHIP iff WITH fixes >=1 decision toward truth AND breaks 0",
                "cells": [list(c) for c in CELLS[: args.cells]],
                "records": records,
                "errors": {"without": e0, "with": e1, "with_both": e2},
                "verdict_with": vw,
                "verdict_both": vb,
            },
            indent=1,
            default=str,
        )
    )
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
