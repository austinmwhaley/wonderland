"""Full offline-RL + OPE pipeline test on a sequential pool.

Trains IQL/CQL/BC on a logged sequential pool (unknown origin, agnostic
source), runs the OPE panel, and asserts the pipeline returns sane, ranked
decisions. This exercises the library's core: pool in -> trained offline-RL
candidates -> DEPLOY/HOLD out.

Run: pytest .../tests/test_pipeline.py -q  (slower: trains 3 candidates)
"""

from white_queen.tribunal.ope.synthetic import make_sequential

import pytest

pytestmark = pytest.mark.slow  # integration: full pipeline train+decide (~300s)


_TINY_FQE = {
    "steps_max": 8000,
    "eval_every": 500,
    "patience": 5,
    "batch": 256,
    "hidden": 64,
    "allow_under_budget": True,
}


def test_pipeline_trains_and_decides():
    from white_queen.tribunal.ope.pipeline import run

    logs, info = make_sequential(n=6000, d=6, nA=4, T=15, seed=0, behavior_eps=0.5)
    rep = run(
        logs,
        gamma=0.99,
        nA=info["nA"],
        estimate_propensity=True,
        offline_steps=6000,
        fast=True,
        ensemble_K=2,
        fqe_cfg=dict(_TINY_FQE),
        out_dir="/tmp/opencode/wq_pipe",
    )
    # Structure of a decision report.
    assert rep["n_candidates"] == 3
    assert set(rep["decisions"]) == {"iql", "cql", "bc"}
    assert isinstance(rep["deployed"], list)
    assert rep["rank"][0] in rep["decisions"]
    assert "rationale" in rep and rep["provenance"]["mode"] == "rl"
    print("\nPIPELINE deployed:", rep["deployed"], "rank:", rep["rank"])
    for n in rep["rank"]:
        e = rep["estimates"][n]
        print(
            f"  {n}: FQE-soft={e['fqe_dm']} FQE-argmax={e['sharp_dm']} "
            f"MB-argmax={e['mb_sharp']} ESS={e['ess_frac']} "
            f"{'DEPLOY' if rep['decisions'][n]['deploy'] else 'hold'}"
        )
    # At least one candidate should be certified as better than behavior on a
    # pool where the optimal policy is learnable and clearly better.
    assert len(rep["deployed"]) >= 1, rep


def test_pipeline_rejects_bandit_logs():
    from white_queen.tribunal.ope.pipeline import run
    from white_queen.tribunal.ope.synthetic import make_bandit

    logs, info = make_bandit(n=500, d=4, nA=3, seed=0, include_propensity=True)
    import pytest

    with pytest.raises(ValueError, match="sequential"):
        run(logs, nA=info["nA"], offline_steps=200)
