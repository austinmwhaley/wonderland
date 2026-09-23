"""Agnostic data adapter tests: any source -> canonical, both modes.
CPU, no torch (except the estimation test). Run: pytest .../tests/test_data.py -q
"""
import numpy as np
import pytest


def _raw_bandit(n=200, d=4, nA=3, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "c0": rng.normal(size=n).astype(np.float32),
        "c1": rng.normal(size=n).astype(np.float32),
        "c2": rng.normal(size=n).astype(np.float32),
        "c3": rng.normal(size=n).astype(np.float32),
        "action": rng.integers(0, nA, n),
        "reward": rng.normal(size=n).astype(np.float32),
        "propensity": rng.uniform(0.1, 0.5, n).astype(np.float32),
    }


def test_dict_bandit_canonicalizes():
    from white_queen.tribunal.ope.data import to_canonical
    d = to_canonical(_raw_bandit(), columns={"context": ["c0", "c1", "c2", "c3"]})
    assert d["mode"] == "bandit"
    assert d["N"] == 200 and d["nA"] == 3 and d["obs"].shape == (200, 4)
    assert d["N"] == len(np.unique(d["episode"]))  # one episode per row
    assert (d["done"] == 1).all()
    assert d["mu_take"].shape == (200,)
    assert d["provenance"]["propensity"] == "provided"


def test_alias_and_inference():
    from white_queen.tribunal.ope.data import to_canonical
    raw = _raw_bandit()
    raw["a"] = raw.pop("action")
    raw["r"] = raw.pop("reward")
    raw["p"] = raw.pop("propensity")
    d = to_canonical(raw, columns={"context": ["c0", "c1", "c2", "c3"]})
    assert d["nA"] == 3 and d["N"] == 200


def test_arrow_polars_pandas_duckdb_equivalent():
    import pyarrow as pa
    import polars as pl
    import duckdb
    from white_queen.tribunal.ope.data import to_canonical
    raw = _raw_bandit()
    ctx = ["c0", "c1", "c2", "c3"]
    cols = {"context": ctx}
    ref = to_canonical(raw, columns=cols)
    table = pa.table({k: pa.array(v) for k, v in raw.items()})
    d_arrow = to_canonical(table, columns=cols)
    d_polars = to_canonical(pl.DataFrame(raw), columns=cols)
    rel = duckdb.connect().execute("SELECT * FROM (VALUES (1)) t(x)")  # placeholder
    con = duckdb.connect()
    rel = con.from_arrow(table)
    d_duck = to_canonical(rel, columns=cols)
    for d in (d_arrow, d_polars, d_duck):
        assert np.array_equal(d["act"], ref["act"])
        assert np.allclose(d["obs"], ref["obs"])
        assert np.allclose(d["mu_take"], ref["mu_take"])
        assert d["mode"] == "bandit"


def test_sequential_mode_infers_episode_and_t():
    from white_queen.tribunal.ope.data import to_canonical
    rng = np.random.default_rng(0)
    T, n_ep, d = 5, 40, 3
    N = T * n_ep
    obs = rng.normal(size=(N, d)).astype(np.float32)
    done = np.zeros(N, dtype=np.float32)
    done[T - 1::T] = 1.0
    raw = {"obs": obs, "act": rng.integers(0, 2, N),
           "rew": rng.normal(size=N).astype(np.float32),
           "next_obs": obs, "done": done,
           "propensity": rng.uniform(0.2, 0.8, N).astype(np.float32)}
    d = to_canonical(raw)
    assert d["mode"] == "rl"
    assert len(np.unique(d["episode"])) == n_ep
    assert d["t"].min() == 0 and d["t"].max() == T - 1


def test_missing_propensity_estimated():
    from white_queen.tribunal.ope.data import to_canonical
    raw = _raw_bandit(n=600)
    raw.pop("propensity")
    d = to_canonical(raw, columns={"context": ["c0", "c1", "c2", "c3"]},
                     behavior_cfg={"steps_max": 200, "device": "cpu",
                                   "hidden": 32, "batch": 64})
    assert d["provenance"]["propensity"] == "estimated"
    assert (d["mu_take"] > 0).all() and (d["mu_take"] <= 1).all()
    assert d["mu"] is not None and d["mu"].shape == (600, 3)


def test_missing_propensity_raises_when_disabled():
    from white_queen.tribunal.ope.data import to_canonical
    raw = _raw_bandit()
    raw.pop("propensity")
    with pytest.raises(ValueError, match="propensity"):
        to_canonical(raw, columns={"context": ["c0", "c1", "c2", "c3"]},
                     estimate_propensity=False)


def test_prefixed_obs_group_and_behavior_metadata_excluded():
    # Colony schema: vector obs split as o0..o3, next state as n0..n3, plus
    # behavior metadata (greedy_action, eps, prob_taken). The agnostic ingest
    # must (a) take obs = [o0..o3] only, (b) build obs2 from [n0..n3], and
    # (c) NOT sweep next-state/metadata into the context (which caused an
    # 11-dim context and leaked the next state into training).
    from white_queen.tribunal.ope.data import to_canonical
    n = 60
    rng = np.random.default_rng(0)
    raw = {}
    for i in range(4):
        raw[f"o{i}"] = rng.normal(size=n).astype(np.float32)
        raw[f"n{i}"] = rng.normal(size=n).astype(np.float32)
    raw["action"] = rng.integers(0, 2, n)
    raw["reward"] = rng.normal(size=n).astype(np.float32)
    raw["done"] = np.zeros(n, dtype=np.float32)
    raw["greedy_action"] = rng.integers(0, 2, n)
    raw["eps"] = np.full(n, 0.2, dtype=np.float32)
    raw["prob_taken"] = np.full(n, 0.8, dtype=np.float32)
    d = to_canonical(raw)
    assert d["obs"].shape == (n, 4)
    assert d["obs2"].shape == (n, 4)
    assert d["mode"] == "rl"
    # obs2 must come from n0..n3, not a copy of obs.
    assert not np.allclose(d["obs"], d["obs2"])
    assert np.allclose(d["obs2"][0], [raw[f"n{i}"][0] for i in range(4)])
