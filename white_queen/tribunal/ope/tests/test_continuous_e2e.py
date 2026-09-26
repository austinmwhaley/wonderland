"""Continuous-action offline RL + OPE.

Per-decision continuous logs (a ~ N(W s, sigma^2), the real high-dim/robot
shape): the library must ingest them, accept or estimate the behavior
density, and evaluate a continuous candidate via density-ratio OPE
(IS/WIS/DR) + continuous FQE + model-based rollout. Ground truth is known,
so we assert the decision is correct.

Sequential continuous logs are also exercised end-to-end (structural), but
they inherit the same long-horizon boundary as discrete: trajectory density
ratios collapse, which the panel reports via ESS.

Run: pytest .../tests/test_continuous_e2e.py -q
"""

import numpy as np

import pytest

from white_queen.tribunal.ope.synthetic import make_continuous

pytestmark = pytest.mark.slow  # integration: continuous E2E suites

_TINY = {
    "steps_max": 2000,
    "eval_every": 500,
    "patience": 5,
    "batch": 256,
    "hidden": 64,
    # tests deliberately under-train for speed; opt out of the derived-
    # budget reliability guard (which is for production panels).
    "allow_under_budget": True,
}


def test_continuous_bandit_better_deploys_behavior_holds():
    from white_queen.tribunal.ope.api import evaluate

    logs, info = make_continuous(
        n=4000, d=5, a_dim=2, T=1, seed=0, sigma_behavior=1.0, sequential=False
    )
    rep_best = evaluate(
        logs, info["optimal_policy"], nA=info["a_dim"], fast=True, fqe_cfg=dict(_TINY)
    )
    rep_beh = evaluate(
        logs, info["behavior_policy"], nA=info["a_dim"], fast=True, fqe_cfg=dict(_TINY)
    )
    print(
        "\nCONT BEST deploy=",
        rep_best["deploy"],
        "FQE=",
        rep_best["estimates"]["sharp_dm"],
        "MB=",
        rep_best["estimates"]["mb"],
        "ESS=",
        rep_best["estimates"]["ess_frac"],
    )
    print("CONT BEH deploy=", rep_beh["deploy"], "FQE=", rep_beh["estimates"]["sharp_dm"])
    assert rep_best["provenance"]["continuous"] is True
    assert rep_best["provenance"]["mode"] == "bandit"
    assert rep_best["deploy"] is True, rep_best
    assert rep_beh["deploy"] is False, rep_beh


def test_continuous_estimated_density_path():
    from white_queen.tribunal.ope.api import evaluate

    logs, info = make_continuous(
        n=4000, d=5, a_dim=2, T=1, seed=1, sigma_behavior=1.0, include_logp=False, sequential=False
    )
    rep = evaluate(
        logs,
        info["optimal_policy"],
        nA=info["a_dim"],
        fast=True,
        estimate_propensity=True,
        behavior_cfg={"steps_max": 400, "device": "cpu", "hidden": 64, "batch": 128},
        fqe_cfg=dict(_TINY),
    )
    assert rep["provenance"]["propensity"] == "estimated"
    assert rep["provenance"]["continuous"] is True
    assert rep["deploy"] is True, rep


def test_continuous_sequential_runs_and_decides():
    from white_queen.tribunal.ope.api import evaluate

    logs, info = make_continuous(
        n=3000, d=5, a_dim=2, T=10, seed=2, sigma_behavior=1.0, sequential=True
    )
    rep = evaluate(logs, info["optimal_policy"], nA=info["a_dim"], fast=True, fqe_cfg=dict(_TINY))
    assert rep["provenance"]["mode"] == "rl"
    assert rep["provenance"]["continuous"] is True
    assert isinstance(rep["deploy"], bool)
    # Value estimates must be finite-and-in-range, OR falsified by the FQE
    # calibration gate (a tiny budget under-trains the ensemble, so a None here
    # is the gate correctly refusing to trust it — fail-safe, not a bug).
    e = rep["estimates"]
    assert e["sharp_dm"] is None or np.isfinite(e["sharp_dm"])
    assert e["mb"] is None or np.isfinite(e["mb"])


def test_continuous_offline_iql_beats_behavior():
    # Continuous offline RL: train IQL on a logged continuous pool and check it
    # improves the (known, reward-implied) objective over the logging policy.
    # reward = -||a - a*||^2, so higher mean immediate reward = better policy.
    import numpy as np
    from white_queen.tribunal.ope.synthetic import make_continuous
    from white_queen.tribunal.ope.data import to_canonical
    from algorithms.offline.continuous_iql import ContinuousIQL

    logs, info = make_continuous(n=3000, d=5, a_dim=2, T=15, seed=0, sigma_behavior=1.0)
    diet = to_canonical(logs)
    a_dim = int(diet["act"].shape[1])
    agent = ContinuousIQL(
        diet,
        {
            "device": "cpu",
            "gamma": 0.99,
            "hidden": 64,
            "batch_size": 256,
            "seed": 0,
            "lr": 3e-4,
            "action_low": -5 * np.ones(a_dim),
            "action_high": 5 * np.ones(a_dim),
        },
    )
    agent.fit(1500)
    obs = diet["obs"]
    astar = obs @ info["Wstar"].T

    def mean_r(pol):
        return float(np.mean(-np.sum((pol.action_mean(obs) - astar) ** 2, 1)))

    class _Beh:
        def action_mean(self, obs):
            return np.atleast_2d(obs) @ info["Wb"].T

    assert mean_r(agent) > mean_r(_Beh()), (mean_r(agent), mean_r(_Beh()))
    # The continuous candidate protocol must be complete for OPE.
    lp = agent.log_prob_fn(diet["obs"][:5], diet["act"][:5])
    assert lp.shape == (5,) and np.isfinite(lp).all()
    assert agent.sample(diet["obs"][:5], np.random.default_rng(0)).shape == (5, a_dim)
