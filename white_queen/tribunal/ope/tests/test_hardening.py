"""Hardening tests: validation, registry contract, golden determinism, save/load.

Run: pytest white_queen/tribunal/ope/tests/test_hardening.py -q
(panel golden takes ~1-2 min CPU; everything else instant.)
"""
import numpy as np


def _diet(N=600, n_ep=12, seed=0):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), N // n_ep)
    N = len(ep)
    obs = rng.normal(size=(N, 4)).astype(np.float32)
    return {
        "obs": obs, "obs2": np.roll(obs, -1, axis=0).astype(np.float32),
        "act": rng.integers(0, 2, N), "rew": rng.normal(size=N).astype(np.float32),
        "done": np.zeros(N, dtype=np.float32),
        "mu": np.full((N, 2), 0.5, dtype=np.float32),
        "episode": ep, "t": np.tile(np.arange(N // n_ep), n_ep)[:N],
        "nA": 2, "N": N,
    }


class _Uniform:
    def __init__(self, nA=2):
        self.nA = nA

    def act(self, state, eval=True):
        return 0

    def action_probs(self, obs, temperature=1.0):
        o = np.asarray(obs)
        n = len(o) if o.ndim > 1 else 1
        return np.full((n, self.nA), 1.0 / self.nA, dtype=np.float32)


def test_validate_diet():
    from white_queen.tribunal.ope.protocols import validate_diet
    fp = validate_diet(_diet())
    assert fp["N"] == 600 and fp["n_episodes"] == 12
    import pytest
    d = _diet()
    del d["mu"]  # no propensity at all
    with pytest.raises(ValueError, match="propensity"):
        validate_diet(d)
    d = _diet()
    d["obs"] = d["obs"][:100]  # ragged vs N
    with pytest.raises(ValueError, match="obs"):
        validate_diet(d)
    d = _diet()
    d["obs"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_diet(d)
    d = _diet()
    d["mu"] = np.full_like(d["mu"], 0.1)  # rows sum 0.2
    with pytest.raises(ValueError, match="sum to 1"):
        validate_diet(d)
    # bandit/one-step canonical (episode = arange) is now VALID
    d = _diet()
    d["episode"] = np.arange(len(d["obs"]), dtype=int)
    fp2 = validate_diet(d)
    assert fp2["n_episodes"] == 600
    # action out of range is rejected
    d = _diet()
    d["act"] = np.full(len(d["obs"]), 9, dtype=int)
    with pytest.raises(ValueError, match="act"):
        validate_diet(d)
    # mu_take alone (no full mu) is accepted
    d = _diet()
    d["mu_take"] = d["mu"][np.arange(len(d["obs"])), d["act"]]
    del d["mu"]
    validate_diet(d)


def test_panel_rejects_bad_diet_fast():
    from white_queen.tribunal.ope import estimators as E
    import pytest
    with pytest.raises((ValueError, TypeError)):
        E.panel({"obs": []}, _Uniform(), 0.99)


def test_registry_keys_match_panel():
    import torch
    from white_queen.tribunal.ope import estimators as E
    torch.manual_seed(0)
    torch.set_num_threads(1)
    d = _diet()
    tiny = {"steps_max": 120, "eval_every": 40, "patience": 2, "batch": 32,
            "hidden": 16, "device": "cpu"}
    meta = {"bootstrap_B": 40, "temps": (0.5, 1.0)}
    p = E.panel(d, _Uniform(), 0.99, meta=meta, fqe_cfg=tiny)
    need = set()
    for keys, _, _ in E.ESTIMATORS.values():
        need.update(keys)
    missing = need - set(p)
    assert not missing, missing  # registry contract: add estimator? update table
    assert set(p["timing"]) >= {"fqe_single", "ensemble", "dr", "mis",
                                 "dynamics", "sharp", "total"}
    assert all(v >= 0 for v in p["timing"].values())
    # DM headline = VAL-WEIGHTED ensemble mean (collapsed members earn less
    # weight). Must lie within member range; weights sum to 1.
    ms = p["efqe"]["members"]
    assert min(ms) - 1e-6 <= p["fqe_dm"] <= max(ms) + 1e-6
    assert abs(sum(p["efqe"]["val_weights"]) - 1.0) < 1e-6
    # Sharp (argmax) evaluation present; may be finite or divergence-guarded.
    assert ("sharp_dm" in p) and ("sharp_info" in p)
    # Gate surfaces both through rows + soft-vs-sharp advisory.
    from white_queen.tribunal.ope import gate as G
    rows = G.adjudicate({"u": p}, 50.0, 20.0, None, None, n_episodes=12)
    assert np.isfinite(rows["u"]["sharp_dm"])
    assert any("soft-vs-sharp" in a for a in rows["u"]["advisories"])
    # New surfaces: support map, sensitivity receipt, evidence passthrough.
    assert set(p["support"]) >= {"w_p90", "top_decile_share",
                                 "top_decile_obs_center"}
    assert len(p["ep_returns"]) == len(p["ep_weights"])
    assert "sensitivity" in rows["u"]
    assert rows["u"]["sensitivity"] is not None
    assert rows["u"]["sensitivity"]["target"] == "dr"


def test_golden_determinism():
    import torch
    from white_queen.tribunal.ope import estimators as E
    tiny = {"steps_max": 120, "eval_every": 40, "patience": 2, "batch": 32,
            "hidden": 16, "device": "cpu"}
    meta = {"bootstrap_B": 40, "temps": (0.5, 1.0)}
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        outs.append(E.panel(_diet(), _Uniform(), 0.99, meta=meta,
                            fqe_cfg=dict(tiny)))
    a, b = outs
    assert a["temperature"] == b["temperature"]  # numpy paths bit-identical
    assert a["ess_frac"] == b["ess_frac"]
    assert abs(a["blended"] - b["blended"]) < 1e-9  # torch paths stable CPU
    assert a["slope_pick"] == b["slope_pick"]
    for k in ("blended", "dr", "fqe_dm", "mis", "wdr", "magic"):
        assert np.isfinite(a[k]), k


def test_bc_save_load_roundtrip():
    import torch
    from types import SimpleNamespace
    from white_queen.tribunal.candidates import _BCWrapper
    env = SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)),
                          action_space=SimpleNamespace(n=2))
    d = _diet(N=400, n_ep=8)
    w = _BCWrapper(env, {"device": "cpu", "hidden": 16, "batch_size": 32,
                         "lr": 1e-3}, d).fit(d, 60)
    assert w.val_info["steps"] <= 60 and np.isfinite(w.val_info["val_nll"])
    import io
    buf = io.BytesIO()
    torch.save(w.net.state_dict(), buf)
    buf.seek(0)
    w2 = _BCWrapper(env, {"device": "cpu", "hidden": 16, "batch_size": 32,
                          "lr": 1e-3}, d)
    w2.net.load_state_dict(torch.load(buf, weights_only=True))
    p1 = w.action_probs(d["obs"][:10])
    p2 = w2.action_probs(d["obs"][:10])
    assert np.allclose(p1, p2)


def test_registry_import_needs_no_offset():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['OFFSET'] = None; "
         "sys.path.insert(0, '/home/austin-whaley/wq'); "
         "import environments.registry as R; "
         "e = R.make_env('cartpole', seed=0); print('cartpole ok')"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert "cartpole ok" in r.stdout
