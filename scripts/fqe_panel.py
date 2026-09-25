"""FQE experiment panel — the consolidated home of the one-off FQE sweeps.

Formerly 10 near-copy scripts (fqe_budget, fqe_govern, fqe_validate, fqe_argmax2,
fqe_sharp, fqe_sweep, fqe_stable, fqe_toy, fqe_sync, fqe_polyak). Each former
script is now one subcommand; run history/results live in STATUS.md.

Usage
-----
  python scripts/fqe_panel.py --list
  python scripts/fqe_panel.py budget
  python scripts/fqe_panel.py toy sync polyak   # several in one process
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

from algorithms.deep.networks import QNetwork
from environments.registry import make_env
from white_queen import db
from white_queen.config import PRESETS
from white_queen.tribunal.candidates import load_candidate
from white_queen.tribunal.ope import estimators as E

DB = ROOT / "white_queen" / "data" / "white_queen_quick.duckdb"
VERDICTS_V15 = ROOT / "white_queen" / "verdicts" / "v15"
DIETS = ["novice_only", "mixed", "expert_only"]


# --------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------
def make_cfg_env():
    return dict(PRESETS["quick"]), make_env("cartpole", seed=999)


def load_ctx(diet: str = "novice_only"):
    cfg, env = make_cfg_env()
    data = db.load_diet(str(DB), diet)
    return cfg, env, data


def load_iql(cfg, env, data, diet: str = "novice_only"):
    return load_candidate("iql", env, data, cfg, str(VERDICTS_V15 / f"iql_{diet}.pt"))


def load_named(name: str, env, data, cfg, diet: str):
    if name == "uniform":
        return UniformPolicy()
    return load_candidate(name, env, data, cfg, str(VERDICTS_V15 / f"{name}_{diet}.pt"))


class UniformPolicy:
    def act(self, s, eval=True):
        return 0

    def action_probs(self, o, temperature=1.0):
        o = np.asarray(o)
        return np.full((len(o), 2), 0.5, dtype=np.float32)


class ArgmaxWrap:
    """Evaluate the SHIPPED argmax policy: action_probs = one-hot at argmax."""

    def __init__(self, inner):
        self.inner = inner

    def act(self, s, eval=True):
        return self.inner.act(s, eval=eval)

    def action_probs(self, o, temperature=1.0):
        p = np.asarray(self.inner.action_probs(o, temperature=1.0), dtype=np.float64)
        out = np.zeros_like(p)
        out[np.arange(len(p)), p.argmax(1)] = 1.0
        return out


def fit(data, cand, cfg_fqe, temperature=1.0):
    torch.manual_seed(0)
    return E.fit_fqe(data, cand, 0.99, cfg_fqe, temperature=temperature)


# --------------------------------------------------------------------------
# real-data experiments (were: budget, govern, validate, argmax2, sharp)
# --------------------------------------------------------------------------
def exp_budget():
    cfg, env, data = load_ctx("novice_only")
    iql = load_iql(cfg, env, data, "novice_only")
    print("proxy live ~52; bar 30.0", flush=True)
    for sm in (11428, 20000, 40000, 80000):
        _, dm, info = fit(
            data,
            iql,
            {
                "device": "cpu",
                "steps_max": sm,
                "eval_every": max(500, sm // 20),
                "patience": 20,
                "target_tau": 0.02,
            },
        )
        print(
            "  steps_max=%6d DM=%7.1f stopped=%s steps=%d"
            % (sm, dm, info["stopped"], info["steps"]),
            flush=True,
        )
    print("BUDGET DONE", flush=True)


def exp_govern():
    cfg, env, data = load_ctx("novice_only")
    iql = load_iql(cfg, env, data, "novice_only")
    print("proxy live ~52; bar 30.0", flush=True)
    for label, base in [
        ("governed default", {"device": "cpu"}),
        ("patience=999 (no early stop)", {"device": "cpu", "patience": 999}),
        ("patience=999 tau=0.02", {"device": "cpu", "patience": 999, "target_tau": 0.02}),
        ("patience=999 tau=0.05", {"device": "cpu", "patience": 999, "target_tau": 0.05}),
    ]:
        _, dm, info = fit(data, iql, base)
        print(
            "  %-30s DM=%7.1f tau=%.4f steps=%s stopped=%s"
            % (label, dm, info.get("target_tau", -1), info["steps"], info["stopped"]),
            flush=True,
        )
    print("GOVERN DONE", flush=True)


def exp_validate():
    cfg, env = make_cfg_env()
    for diet in DIETS:
        data = db.load_diet(str(DB), diet)
        for name in ["uniform", "iql"]:
            cand = load_named(name, env, data, cfg, diet)
            _, dm, info = fit(
                data,
                cand,
                {"device": "cpu", "steps_max": 80000, "eval_every": 4000, "patience": 20},
            )
            print(
                "%-12s %-8s governed FQE DM=%7.1f stopped=%s tau=%.3f"
                % (diet, name, dm, info["stopped"], info.get("target_tau", -1)),
                flush=True,
            )
    print("VALIDATE DONE", flush=True)


def exp_argmax2():
    cfg, env = make_cfg_env()
    for diet in DIETS:
        data = db.load_diet(str(DB), diet)
        for name in ["uniform", "iql"]:
            cand = ArgmaxWrap(load_named(name, env, data, cfg, diet))
            _, dm, _info = fit(
                data,
                cand,
                {"device": "cpu", "steps_max": 80000, "eval_every": 4000, "patience": 20},
            )
            print("%-12s %-8s FQE-ARGMAX DM=%7.1f" % (diet, name, dm), flush=True)
    print("ARGMAX2 DONE", flush=True)


def exp_sharp():
    cfg, env = make_cfg_env()
    for diet in DIETS:
        data = db.load_diet(str(DB), diet)
        for name in ["uniform", "iql", "bc", "cql"]:
            cand = ArgmaxWrap(load_named(name, env, data, cfg, diet))
            _, dm, info = fit(
                data,
                cand,
                {
                    "steps_max": 120000,
                    "eval_every": 5000,
                    "patience": 10,
                    "batch": 512,
                    "hidden": 128,
                },
            )
            print(
                f"{diet:12s} {name:8s} FQE-argmax DM={dm:6.1f} stopped={info['stopped']}",
                flush=True,
            )
    print("FQE SHARP DONE", flush=True)


# --------------------------------------------------------------------------
# real-data manual Q sweeps (were: sweep, stable)
# --------------------------------------------------------------------------
def _real_tensors(data, iql):
    n = len(data["obs"])
    o = torch.as_tensor(data["obs"])
    a = torch.as_tensor(data["act"], dtype=torch.long)
    r = torch.as_tensor(data["rew"]).unsqueeze(1)
    o2 = torch.as_tensor(data["obs2"])
    d = torch.as_tensor(data["done"]).unsqueeze(1)
    p = iql.action_probs(data["obs"], temperature=1.0)
    p2 = torch.as_tensor(iql.action_probs(data["obs2"], temperature=1.0))
    ep = data["episode"]
    starts = np.unique(ep, return_index=True)[1]
    return n, o, a, r, o2, d, p, p2, starts


def _run_real_q(
    data,
    iql,
    steps=40000,
    *,
    update="polyak",
    cadence=0,
    tau=0.02,
    lr=1e-3,
    clip=0.0,
    huber=False,
    batch=256,
    hidden=128,
    seed=0,
    sample="step",
):
    """Manual FQE-style Q training on the real diet.

    update: 'polyak' | 'hard'; sample: 'step' (per-step seed) | 'stream'.
    """
    n, o, a, r, o2, d, p, p2, starts = _real_tensors(data, iql)
    torch.manual_seed(seed)
    q = QNetwork(4, hidden, 2)
    qt = QNetwork(4, hidden, 2)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    stream = np.random.default_rng(seed)
    for s in range(steps):
        if sample == "step":
            idx = np.random.default_rng(s).integers(0, n, batch)
        else:
            idx = stream.integers(0, n, batch)
        i = torch.as_tensor(idx)
        with torch.no_grad():
            v2 = (p2[i] * qt(o2[i])).sum(1, keepdim=True)
            tgt = r[i] + 0.99 * (1 - d[i]) * v2
        pred = q(o[i]).gather(1, a[i].unsqueeze(1))
        loss = F.smooth_l1_loss(pred, tgt) if huber else F.mse_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(q.parameters(), clip)
        opt.step()
        if update == "polyak":
            with torch.no_grad():
                for pq, pqt in zip(q.parameters(), qt.parameters()):
                    pqt.mul_(1 - tau).add_(pq, alpha=tau)
        elif update == "hard" and cadence and s % cadence == 0:
            qt.load_state_dict(q.state_dict())
    with torch.no_grad():
        v = (torch.as_tensor(p) * q(o)).sum(1)
        return float(v[starts].mean())


def exp_sweep():
    cfg, env, data = load_ctx("novice_only")
    iql = load_iql(cfg, env, data, "novice_only")
    print("context: candidate live argmax truth ~81; this soft proxy live ~52", flush=True)
    for label, kw in [
        ("hard every 2242 (current)", dict(update="hard", cadence=2242)),
        ("hard every 1000", dict(update="hard", cadence=1000)),
        ("hard every 250", dict(update="hard", cadence=250)),
        ("hard every 100", dict(update="hard", cadence=100)),
        ("polyak 0.02", dict(update="polyak", tau=0.02)),
        ("polyak 0.01", dict(update="polyak", tau=0.01)),
        ("polyak 0.005", dict(update="polyak", tau=0.005)),
    ]:
        try:
            print("  %-28s DM=%7.1f" % (label, _run_real_q(data, iql, **kw)), flush=True)
        except Exception as e:
            print("  %-28s ERROR %r" % (label, e), flush=True)
    print("SWEEP DONE", flush=True)


def exp_stable():
    cfg, env, data = load_ctx("novice_only")
    iql = load_iql(cfg, env, data, "novice_only")
    print("proxy live ~52; bar 30.0", flush=True)
    for label, kw in [
        ("baseline 80k", dict(steps=80000, batch=512, sample="stream")),
        ("clip=1.0", dict(steps=80000, batch=512, sample="stream", clip=1.0)),
        ("lr=5e-4", dict(steps=80000, batch=512, sample="stream", lr=5e-4)),
        ("clip+lr5e-4", dict(steps=80000, batch=512, sample="stream", clip=1.0, lr=5e-4)),
        ("huber", dict(steps=80000, batch=512, sample="stream", huber=True)),
        (
            "huber+clip+lr5e-4",
            dict(steps=80000, batch=512, sample="stream", huber=True, clip=1.0, lr=5e-4),
        ),
    ]:
        print("  %-20s DM=%7.1f" % (label, _run_real_q(data, iql, **kw)), flush=True)
    print("STABLE DONE", flush=True)


# --------------------------------------------------------------------------
# toy experiments (were: toy, sync, polyak)
# --------------------------------------------------------------------------
def _toy_setup(g=0.99):
    rng = np.random.default_rng(0)
    n = 2000
    obs = rng.normal(size=(n, 4)).astype(np.float32)
    o = torch.as_tensor(obs)
    r = torch.ones(n, 1)
    o2 = torch.as_tensor(obs)
    d = torch.zeros(n, 1)
    return rng, n, o, r, o2, d


def _run_toy_q(
    rng,
    n,
    o,
    r,
    o2,
    d,
    steps,
    *,
    g=0.99,
    lr=1e-3,
    hidden=128,
    update="sync",
    cadence=1000,
    tau=0.0,
    batch=256,
    eval_n=100,
):
    """Toy FQE loop: reward 1 every step, uniform target (0.5 * two arms).

    update: 'sync' (hard copy every cadence) | 'polyak' | 'none'.
    """
    torch.manual_seed(0)
    q = QNetwork(4, hidden, 2)
    qt = QNetwork(4, hidden, 2)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    for s in range(steps):
        i = torch.as_tensor(rng.integers(0, n, batch))
        with torch.no_grad():
            v2 = (0.5 * qt(o2[i])).sum(1, keepdim=True)
            tgt = r[i] + g * (1 - d[i]) * v2
        pred = q(o[i]).mean(1, keepdim=True)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if update == "sync" and s % cadence == 0:
            qt.load_state_dict(q.state_dict())
        elif update == "polyak":
            with torch.no_grad():
                for pq, pqt in zip(q.parameters(), qt.parameters()):
                    pqt.mul_(1 - tau).add_(pq, alpha=tau)
    with torch.no_grad():
        return float(q(o[:eval_n]).mean(1).mean())


def exp_toy():
    g, h = 0.99, 100
    truth = float(sum(g**t for t in range(h)))
    print("exact DM =", round(truth, 2), flush=True)
    rng, n, o, r, o2, d = _toy_setup(g)
    for label, kw in [
        ("baseline 5k", dict(steps=5000)),
        ("20k", dict(steps=20000)),
        ("50k", dict(steps=50000)),
        ("50k lr3e-3", dict(steps=50000, lr=3e-3)),
        ("50k no-target", dict(steps=50000, update="none")),
        ("50k wide256", dict(steps=50000, hidden=256)),
    ]:
        print(
            "%16s DM=%7.2f (exact %.1f)" % (label, _run_toy_q(rng, n, o, r, o2, d, **kw), truth),
            flush=True,
        )
    print("TOY DONE", flush=True)


def exp_toy_sync():
    g, h = 0.99, 100
    truth = float(sum(g**t for t in range(h)))
    print("exact DM =", round(truth, 2), flush=True)
    rng, n, o, r, o2, d = _toy_setup(g)
    print("steps=50000, vary target sync cadence:", flush=True)
    for se in (50, 100, 200, 500, 1000):
        dm = _run_toy_q(rng, n, o, r, o2, d, 50000, update="sync", cadence=se, g=g, eval_n=h)
        print(
            "  sync_every=%5d DM=%7.2f  (%.0f%% of exact)" % (se, dm, 100 * dm / truth),
            flush=True,
        )
    print("TOY2 DONE", flush=True)


def exp_toy_polyak():
    truth = 100.0  # infinite-horizon uniform policy, reward 1: 1/(1-0.99)
    print("true fixed point =", truth, flush=True)
    rng, n, o, r, o2, d = _toy_setup()
    print("steps=50000, Polyak soft target (tau: smaller = faster tracking):", flush=True)
    for tau in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
        dm = _run_toy_q(rng, n, o, r, o2, d, 50000, update="polyak", tau=tau)
        print("  tau=%.3f DM=%7.2f  (%.0f%% of exact)" % (tau, dm, 100 * dm / truth), flush=True)
    print("TOY3 DONE", flush=True)


EXPERIMENTS = {
    "budget": exp_budget,
    "govern": exp_govern,
    "validate": exp_validate,
    "argmax2": exp_argmax2,
    "sharp": exp_sharp,
    "sweep": exp_sweep,
    "stable": exp_stable,
    "toy": exp_toy,
    "toy-sync": exp_toy_sync,
    "toy-polyak": exp_toy_polyak,
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "experiments",
        nargs="*",
        metavar="NAME",
        help=f"experiments to run (default: all): {', '.join(EXPERIMENTS)}",
    )
    ap.add_argument("--list", action="store_true", help="list experiments and exit")
    args = ap.parse_args(argv)
    if args.list:
        for name in EXPERIMENTS:
            print(name)
        return 0
    unknown = [name for name in args.experiments if name not in EXPERIMENTS]
    if unknown:
        ap.error(f"unknown experiment(s): {', '.join(unknown)}")
    todo = args.experiments or list(EXPERIMENTS)
    print("cuda", torch.cuda.is_available(), flush=True)
    for name in todo:
        print(f"== fqe_panel: {name} ==", flush=True)
        EXPERIMENTS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
