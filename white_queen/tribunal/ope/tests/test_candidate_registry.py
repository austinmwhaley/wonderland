"""Candidate pluggability: custom trainers + externally-supplied policies.

The library must evaluate ANY protocol-compliant policy, not just its bundled
iql/cql/bc. These tests pin the registry and the object path.

Run: pytest .../tests/test_candidate_registry.py -q
"""

import numpy as np
import pytest


from white_queen.tribunal.ope.synthetic import make_sequential, OptimalActionPolicy
from white_queen.tribunal.ope.protocols import check_candidate

pytestmark = pytest.mark.slow  # integration: registry training paths (~110s)

_TINY_FQE = {
    "steps_max": 2000,
    "eval_every": 500,
    "patience": 4,
    "batch": 256,
    "hidden": 64,
    "allow_under_budget": True,
}


class _ConstantPolicy:
    def __init__(self, nA, action=0):
        self.nA, self.action = int(nA), int(action)

    def act(self, state, eval=True):
        return self.action

    def action_probs(self, obs, temperature=1.0):
        o = np.asarray(obs)
        n = len(o) if o.ndim > 1 else 1
        p = np.zeros((n, self.nA), dtype=np.float32)
        p[:, self.action] = 1.0
        return p


def test_register_custom_trainer():
    from white_queen.tribunal.candidates import register_candidate, CANDIDATE_TRAINERS

    def _train(name, env, diet, cfg, out_path):
        c = _ConstantPolicy(diet["nA"], action=0)
        c.name = name
        check_candidate(c)
        return c

    register_candidate("const0", _train, overwrite=True)
    assert "const0" in CANDIDATE_TRAINERS

    from white_queen.tribunal.ope.pipeline import run

    logs, info = make_sequential(n=3000, d=6, nA=4, T=12, seed=3, behavior_eps=0.5)
    rep = run(
        logs,
        algorithms=("iql", "const0"),
        nA=info["nA"],
        estimate_propensity=True,
        offline_steps=2000,
        fast=True,
        ensemble_K=2,
        fqe_cfg=dict(_TINY_FQE),
        out_dir="/tmp/opencode/wq_reg",
    )
    assert set(rep["decisions"]) == {"iql", "const0"}
    assert "const0" in rep["rank"]


def test_external_policy_object_directly():
    from white_queen.tribunal.ope.pipeline import run

    logs, info = make_sequential(n=3000, d=6, nA=4, T=12, seed=4, behavior_eps=0.5)
    optimal = OptimalActionPolicy(info["nA"])
    optimal.name = "oracle"
    rep = run(
        logs,
        algorithms=("bc", optimal),
        nA=info["nA"],
        estimate_propensity=True,
        offline_steps=1500,
        fast=True,
        ensemble_K=2,
        fqe_cfg=dict(_TINY_FQE),
        out_dir="/tmp/opencode/wq_ext",
    )
    assert "oracle" in rep["decisions"]
    # The oracle (true optimal) should be ranked at or near the top.
    assert rep["rank"][0] in ("oracle", "bc")


def test_unknown_algorithm_raises():
    from white_queen.tribunal.ope.pipeline import run

    logs, info = make_sequential(n=1000, d=4, nA=3, T=8, seed=5)
    with pytest.raises(ValueError, match="unknown algorithm"):
        run(logs, algorithms=("nope",), nA=info["nA"], offline_steps=100)
