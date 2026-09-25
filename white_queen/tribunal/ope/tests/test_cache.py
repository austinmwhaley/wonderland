"""Cache + screen-tier tests. CPU only. Run: pytest .../tests/test_cache.py -q"""

import numpy as np


def _diet(N=600, n_ep=12, seed=0):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), N // n_ep)
    N = len(ep)
    obs = rng.normal(size=(N, 4)).astype(np.float32)
    return {
        "obs": obs,
        "obs2": obs,
        "act": rng.integers(0, 2, N),
        "rew": rng.normal(size=N).astype(np.float32),
        "done": np.zeros(N, dtype=np.float32),
        "mu": np.full((N, 2), 0.5, dtype=np.float32),
        "episode": ep,
        "t": np.zeros(N),
        "nA": 2,
        "N": N,
    }


class _CountingUniform:
    calls = 0

    def __init__(self, nA=2):
        self.nA = nA

    def act(self, state, eval=True):
        return 0

    def action_probs(self, obs, temperature=1.0):
        type(self).calls += 1
        o = np.asarray(obs)
        n = len(o) if o.ndim > 1 else 1
        return np.full((n, self.nA), 1.0 / self.nA, dtype=np.float32)


def test_cache_roundtrip_miss_and_key():
    import torch
    from white_queen.tribunal.ope import cache as C
    import tempfile

    d = _diet()
    h1, h2 = C.diet_hash(d), C.diet_hash(dict(d))
    assert h1 == h2  # content, not identity
    d2 = dict(d, rew=d["rew"] + 1.0)
    assert C.diet_hash(d2) != h1
    k1 = C.make_key("fqe", h1, "iql_mixed", {"steps_max": 100}, 1.0, "w")
    assert C.make_key("fqe", h1, "iql_mixed", {"steps_max": 100}, 1.0, "w") == k1
    assert C.make_key("fqe", h1, "iql_mixed", {"steps_max": 101}, 1.0, "w") != k1
    with tempfile.TemporaryDirectory() as td:
        assert C.load(td, "nope") is None  # miss -> None, never raises
        net = torch.nn.Linear(4, 2)
        assert C.save(td, k1, {"w": net.state_dict()}) is not None
        back = C.load(td, k1)
        for k in back["w"]:
            assert bool((back["w"][k] == net.state_dict()[k]).all())


def test_candidate_precompute_surgery_engaged():
    import torch
    from white_queen.tribunal.ope import estimators as E

    torch.manual_seed(0)
    torch.set_num_threads(1)
    _CountingUniform.calls = 0
    E.fit_fqe(
        _diet(),
        _CountingUniform(),
        0.99,
        {
            "steps_max": 120,
            "eval_every": 40,
            "patience": 5,
            "batch": 32,
            "hidden": 16,
            "device": "cpu",
        },
        temperature=1.0,
        cand_id="u",
        cache_dir=None,
    )
    # Precompute (2 full-array forwards) + DM eval loop (1 per episode).
    # Per-batch training calls are gone: old code made O(steps) calls here.
    assert _CountingUniform.calls <= 2 + 12 + 2, _CountingUniform.calls


def test_fqe_cache_hit_skips_training():
    import torch
    from white_queen.tribunal.ope import estimators as E
    import tempfile

    torch.manual_seed(0)
    torch.set_num_threads(1)
    tiny = {
        "steps_max": 120,
        "eval_every": 40,
        "patience": 5,
        "batch": 32,
        "hidden": 16,
        "device": "cpu",
    }
    with tempfile.TemporaryDirectory() as td:
        _, dm1, i1 = E.fit_fqe(
            _diet(),
            _CountingUniform(),
            0.99,
            dict(tiny),
            temperature=1.0,
            cand_id="u",
            cache_dir=td,
            weights_hash="w",
        )
        assert not i1["cache_hit"] and i1["steps"] > 0
        _, dm2, i2 = E.fit_fqe(
            _diet(),
            _CountingUniform(),
            0.99,
            dict(tiny),
            temperature=1.0,
            cand_id="u",
            cache_dir=td,
            weights_hash="w",
        )
        assert i2["cache_hit"] and i2["steps"] == 0
        assert dm1 == dm2  # bit-exact, not approximate


def test_screen_tier_contests_never_deploys():
    import torch
    from white_queen.tribunal.ope import estimators as E
    from white_queen.tribunal.ope import judge as J

    torch.manual_seed(0)
    torch.set_num_threads(1)
    tiny = {
        "steps_max": 120,
        "eval_every": 40,
        "patience": 2,
        "batch": 32,
        "hidden": 16,
        "device": "cpu",
    }
    p = E.panel(
        _diet(),
        _CountingUniform(),
        0.99,
        meta={"bootstrap_B": 40, "temps": (0.5, 1.0)},
        fqe_cfg=tiny,
        ensemble_K=2,
        dice_steps=100,
        magic_B=40,
    )
    rows = {
        "u": {
            "fqe_dm": p["fqe_dm"],
            "dr": p["dr"],
            "dr_ci": [0.0, 1.0],
            "efqe": p["efqe"],
            "ess_frac": 0.5,
            "wis": 0.0,
            "mb": p["mb"],
            "temperature": 1.0,
        }
    }
    v = J.judge_diet(
        rows,
        0.0,
        20.0,
        {"rel_edge_std": 0.2, "min_ess_frac": 0.01},
        risk_aversion=0.0,
        allow_deploy=False,
    )
    assert v["deployed"] == []  # screen tier cannot ship, by construction
    assert set(v["contested"]) <= {"u"}
