"""Standing OPE acceptance scorecard.

Runs the acceptance suite (or reprints from accumulated results) and prints the
contract table. Results accumulate under BENCH_RESULTS so the table can be
reprinted cheaply between runs.

Usage
-----
  python -m white_queen.tribunal.bench.scorecard --run cartpole acrobot
  python -m white_queen.tribunal.bench.scorecard --print
"""
from __future__ import annotations

import json
import os
import sys

from .acceptance import CONTRACT, render, rows_from_cases
from .harness import Case, Cell

BENCH_RESULTS = "/tmp/opencode/wq_matrix/bench_results"
_SCRIPTS = "/home/austin-whaley/wq/scripts"
_ROOT = "/home/austin-whaley/wq"
FAMILY = {}
for _fam, _envs in CONTRACT["suite"].items():
    for _e in _envs:
        FAMILY[_e] = _fam


def _save(case):
    os.makedirs(BENCH_RESULTS, exist_ok=True)
    rec = {"name": case.name, "meta": case.meta,
           "cells": [{"candidate": c.candidate, "deploy": c.deploy,
                      "truth": c.truth, "anchor": c.anchor, "bar": c.bar,
                      "rank": c.rank, "lo": c.lo, "hi": c.hi}
                     for c in case.cells]}
    with open(os.path.join(BENCH_RESULTS, f"{case.name}.json"), "w") as f:
        json.dump(rec, f, indent=1)


def _load_all():
    out = {}
    if not os.path.isdir(BENCH_RESULTS):
        return []
    for fn in sorted(os.listdir(BENCH_RESULTS)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(BENCH_RESULTS, fn)) as f:
            r = json.load(f)
        out[r["name"]] = Case(r["name"],
                              [Cell(c["candidate"], c["deploy"], c["truth"],
                                    c["anchor"], c["bar"], c["rank"],
                                    c.get("lo"), c.get("hi"))
                               for c in r["cells"]], meta=r["meta"])
    return list(out.values())


class _RandomPolicy:
    def __init__(self, nA, seed=0):
        import numpy as np
        self.nA, self._rng = int(nA), np.random.default_rng(seed)

    def action_probs(self, obs, temperature=1.0):
        import numpy as np
        o = np.asarray(obs)
        n = len(o) if o.ndim > 1 else 1
        return np.full((n, self.nA), 1.0 / self.nA, dtype=np.float32)

    def act(self, state, eval=True):
        return int(self._rng.integers(self.nA))


def _bandit_case(seed=0):
    import numpy as np
    sys.path.insert(0, _ROOT)
    from white_queen.tribunal.ope.synthetic import make_bandit
    from white_queen.tribunal.ope.api import evaluate
    logs, info = make_bandit(n=4000, d=6, nA=4, seed=seed)
    best = info["best_policy"]
    best.name = "best"
    hold = info["logging_policy"]
    hold.name = "logging"
    tiny = {"steps_max": 500, "eval_every": 250, "patience": 4, "batch": 128,
            "hidden": 64, "allow_under_budget": True}
    rb = evaluate(logs, best, nA=info["nA"], estimate_propensity=True,
                  fast=True, fqe_cfg=dict(tiny))
    rh = evaluate(logs, hold, nA=info["nA"], estimate_propensity=True,
                  fast=True, fqe_cfg=dict(tiny))
    anchor = float(rb["behavior_mean"])
    bar = float(rb["bar"])
    truth_best = float(info["mean_reward_best"])
    truth_hold = anchor   # the logging policy's value IS the anchor -> HOLD
    cb = rb["decision"]["certificate"]
    ch = rh["decision"]["certificate"]
    cells = [Cell("best", bool(rb["deploy"]), truth_best, anchor, bar, 1,
                  cb.get("lo"), cb.get("hi")),
             Cell("logging", bool(rh["deploy"]), truth_hold, anchor, bar, 2,
                  ch.get("lo"), ch.get("hi"))]
    return Case(f"bandit_synth#s{seed}", cells,
                meta={"family": "bandit", "seed": seed})


def _matrix_case(env, seed=0):
    sys.path.insert(0, _SCRIPTS)
    import survey_matrix as SM
    r = SM.survey(env, seed=seed)
    rank = r.get("rank") or list(r["candidates"])
    order = {n: i + 1 for i, n in enumerate(rank)}
    certs = r.get("certificates", {})
    cells = []
    for n, c in r["candidates"].items():
        cert = certs.get(n) or {}
        cells.append(Cell(n, bool(c["deploy"]), float(c["truth"]),
                          float(r["behavior_mean"]), float(r["bar"]),
                          order.get(n, 999), cert.get("lo"), cert.get("hi")))
    return Case(f"{env}#s{seed}", cells,
                meta={"family": FAMILY[env], "seed": seed})


def _continuous_case(env, seed=0):
    sys.path.insert(0, _SCRIPTS)
    import survey_continuous as SC
    r = SC.survey_env(env, seed=seed)
    return Case(f"{env}#s{seed}", [Cell("iql_cont", bool(r["deploy"]),
                                      float(r["iql_truth"]),
                                      float(r["behavior_mean"]), float(r["bar"]),
                                      1, r.get("cert_lo"), r.get("cert_hi"))],
                meta={"family": "continuous", "seed": seed, "improved":
                      bool(r["improved"])})


def run(env, seed=0):
    if FAMILY[env] == "bandit":
        case = _bandit_case(seed)
    elif FAMILY[env] == "continuous":
        case = _continuous_case(env, seed)
    else:
        case = _matrix_case(env, seed)
    _save(case)
    return case


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    seed_only = None
    if "--seed" in args:
        i = args.index("--seed"); seed_only = int(args[i+1]); del args[i:i+2]
    tests = None
    if "--tests" in args:
        i = args.index("--tests")
        tests = int(args[i + 1])
        del args[i:i + 2]
    if "--run" in args:
        i = args.index("--run")
        envs = args[i + 1:] or [e for v in CONTRACT["suite"].values() for e in v]
        seeds = ([seed_only] if seed_only is not None
                 else CONTRACT["seeds"])
        for e in envs:
            for sd in seeds:
                fn = os.path.join(BENCH_RESULTS, f"{e}#s{sd}.json")
                if os.path.exists(fn) and not force:
                    print(f"[scorecard] skip {e} seed {sd} (saved)", flush=True)
                    continue
                print(f"[scorecard] running {e} seed {sd} ...", flush=True)
                run(e, sd)
    cases = _load_all()
    rows, agg = rows_from_cases(cases, tests_passed=tests)
    print(render(rows, agg))
