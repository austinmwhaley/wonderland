"""Full offline-RL + OPE pipeline over an agnostic log pool.

This is the library's core entry point for the real use case: hand it ONE
pool of logged trajectories (from humans, an online policy, or another
offline policy, in any of the formats ope.data accepts), and it:

  1. ingests + canonicalizes the pool (sequential/RL mode),
  2. trains offline-RL candidates on that pool (IQL / CQL / BC by default),
  3. runs the OPE panel on each candidate against the pool,
  4. gates + judges -> per-candidate DEPLOY/HOLD, ranked, with receipts.

No live environment is needed (EnvStub from the data shape). No diet splits,
no colony coupling. `source` can be a path to parquet/csv/db, an Arrow table,
Polars/pandas DataFrame, DuckDB relation, or a dict of arrays.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from .protocols import EnvStub, validate_diet


def _default_train_cfg(gamma, offline_steps, seed):
    return {"gamma": gamma, "offline_steps": offline_steps, "seed": seed}


def train_candidates(
    data,
    algorithms=("iql", "cql", "bc"),
    *,
    gamma=0.99,
    offline_steps=None,
    seed=0,
    out_dir=None,
    cfg=None,
):
    """Train offline-RL candidates on a canonical dataset. `algorithms` may
    mix registered/built-in names (str) and already-trained policy objects
    (anything implementing act + action_probs). Returns {name: handle}."""
    from white_queen.tribunal.candidates import train_candidate, CANDIDATES

    d = int(data["obs"].shape[1])
    if data.get("continuous"):
        for entry in algorithms:
            if isinstance(entry, str) and entry in ("iql", "cql", "bc", "random"):
                if entry in ("iql", "cql", "bc"):
                    raise ValueError(
                        "bundled iql/cql/bc trainers are discrete-action; for "
                        "continuous logs pass a trained policy object (with "
                        "log_prob_fn) or register a continuous trainer via "
                        "register_candidate()."
                    )
    env = EnvStub(
        d,
        int(data["nA"]),
        a_dim=int(data["act"].shape[1]) if data.get("continuous") else None,
        continuous=bool(data.get("continuous")),
    )
    base = _default_train_cfg(gamma, offline_steps, seed)
    if cfg:
        base.update(cfg)
    out_dir = out_dir or tempfile.mkdtemp(prefix="wq_cands_")
    os.makedirs(out_dir, exist_ok=True)
    handles = {}
    for i, entry in enumerate(algorithms):
        if not isinstance(entry, str):
            # already-trained policy object (from any framework)
            name = getattr(entry, "name", None) or f"policy_{i}"
            handles[name] = entry
            continue
        if entry not in CANDIDATES:
            # registry lookup happens inside train_candidate; only hard-fail
            # if it is neither built-in nor registered.
            from white_queen.tribunal.candidates import CANDIDATE_TRAINERS

            if entry not in CANDIDATE_TRAINERS and entry not in ("random", "iql_cont", "bc_cont"):
                raise ValueError(
                    f"unknown algorithm {entry!r}; built-ins {CANDIDATES}, "
                    f"registered {sorted(CANDIDATE_TRAINERS)}"
                )
        ckpt = os.path.join(out_dir, f"{entry}.pt")
        handles[entry] = train_candidate(entry, env, data, base, ckpt)
    return handles


def evaluate_pool(
    data,
    handles,
    *,
    gamma=0.99,
    fast=True,
    ensemble_K=None,
    fqe_cfg=None,
    risk_aversion=0.5,
    cache_dir=None,
    behavior_cfg=None,
):
    """Run the OPE panel + gate + judge on already-trained candidate handles.
    Returns a ranked report with per-candidate decisions."""
    from . import estimators as E
    from . import gate as G
    from . import judge as J
    from .receipts import behavior_stats
    from .drift import drift_report

    b = behavior_stats(data, gamma)
    names = list(handles)
    panels, rows = {}, {}
    per = {}
    for name in names:
        if data.get("continuous"):
            from .continuous import panel_continuous

            p = panel_continuous(
                data,
                handles[name],
                gamma,
                fqe_cfg=fqe_cfg,
                fast=fast,
                cand_id=f"{name}",
                cache_dir=cache_dir,
            )
        else:
            p = E.panel(
                data,
                handles[name],
                gamma,
                meta=None,
                fqe_cfg=fqe_cfg,
                cand_id=f"{name}",
                cache_dir=cache_dir,
                ensemble_K=ensemble_K,
                fast=fast,
            )
        panels[name] = p
        per[name] = p
    rows = G.adjudicate(
        panels, b["mean"], b["std"], None, None, n_episodes=len(np.unique(data["episode"]))
    )
    verdict = J.judge_diet(rows, b["mean"], b["std"], None, risk_aversion=risk_aversion)
    report = {
        "behavior_mean": round(b["mean"], 2),
        "behavior_std": round(b["std"], 2),
        "bar": verdict["bar"],
        "rank": verdict["rank"],
        "deployed": verdict["deployed"],
        "decisions": {n: verdict["decisions"][n] for n in names},
        "estimates": {
            n: {
                k: rows[n].get(k)
                for k in (
                    "fqe_dm",
                    "sharp_dm",
                    "mb",
                    "mb_sharp",
                    "dr",
                    "dr_ci",
                    "level_est",
                    "ess_frac",
                    "lstdq",
                    "wis",
                    "is",
                    "dr_step",
                    "step_ess_frac",
                )
            }
            for n in names
        },
        "sensitivity": {n: rows[n].get("sensitivity") for n in names},
        "n_candidates": len(names),
        "lone_candidate": verdict.get("lone_candidate", False),
        "provenance": data.get("provenance", {}),
        "drift": drift_report(data, gamma),
        "rationale": J.explain_diet("pool", b["mean"], verdict, rows),
    }
    return report


def run(
    source,
    *,
    algorithms=("iql", "cql", "bc"),
    gamma=0.99,
    columns=None,
    nA=None,
    estimate_propensity=True,
    behavior_cfg=None,
    behavior_seed=0,
    offline_steps=None,
    seed=0,
    fast=True,
    ensemble_K=None,
    fqe_cfg=None,
    risk_aversion=0.5,
    cache_dir=None,
    out_dir=None,
    source_name=None,
):
    """One call: log pool -> trained offline-RL candidates -> OPE decisions.

    Requires sequential logs (provide next-state + done). For pure bandit logs
    use ope.api.evaluate per candidate instead.
    """
    from . import data as _data

    d = _data.to_canonical(
        source,
        columns=columns,
        nA=nA,
        estimate_propensity=estimate_propensity,
        behavior_cfg=behavior_cfg,
        behavior_seed=behavior_seed,
        source_name=source_name,
    )
    validate_diet(d)
    if d["mode"] != "rl":
        raise ValueError(
            "offline-RL pipeline needs sequential logs "
            "(next_obs + done). For bandit logs use evaluate()."
        )
    handles = train_candidates(
        d, algorithms, gamma=gamma, offline_steps=offline_steps, seed=seed, out_dir=out_dir
    )
    return evaluate_pool(
        d,
        handles,
        gamma=gamma,
        fast=fast,
        ensemble_K=ensemble_K,
        fqe_cfg=fqe_cfg,
        risk_aversion=risk_aversion,
        cache_dir=cache_dir,
        behavior_cfg=behavior_cfg,
    )
