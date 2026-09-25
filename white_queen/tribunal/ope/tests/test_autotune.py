"""Autotune invariants: same data -> same knobs; bigger data -> saner knobs.

No torch needed. Run: pytest white_queen/tribunal/ope/tests -q
"""

import numpy as np

from white_queen.tribunal.ope import autotune as A


def _toy_diet(N=2000, n_ep=40, obs_dim=4, nA=2, seed=0):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), N // n_ep)
    N = len(ep)
    return {
        "obs": rng.normal(size=(N, obs_dim)).astype(np.float32),
        "obs2": rng.normal(size=(N, obs_dim)).astype(np.float32),
        "act": rng.integers(0, nA, N),
        "rew": rng.normal(size=N).astype(np.float32),
        "done": np.zeros(N, dtype=np.float32),
        "mu": np.full((N, nA), 0.5, dtype=np.float32),
        "episode": ep,
        "t": np.tile(np.arange(N // n_ep), n_ep)[:N],
        "nA": nA,
        "N": N,
    }


def test_fingerprint_counts():
    d = _toy_diet()
    fp = A.diet_fingerprint(d)
    assert fp["N"] == len(d["obs"])
    assert fp["n_episodes"] == 40
    assert fp["max_len"] == 50


def test_bootstrap_grows_with_n():
    b_small, a_small = A.resolve_bootstrap(10, None, None)
    b_big, a_big = A.resolve_bootstrap(500, None, None)
    assert b_big >= b_small
    assert a_small == 0.10  # tiny-n honesty
    assert a_big == 0.05
    # Explicit overrides always win.
    assert A.resolve_bootstrap(10, B=123, alpha=0.01) == (123, 0.01)


def test_target_ess_is_K_over_n():
    assert A.resolve_target_ess(100) == 0.05  # 5/100
    assert A.resolve_target_ess(10) == 0.30  # clipped (5/10=0.5 -> 0.3 cap)
    assert A.resolve_target_ess(1000) == 0.02  # floored
    assert A.resolve_target_ess(100, user_val=0.11) == 0.11


def test_clip_quantile_matches_old_at_10k():
    q = A.resolve_clip_quantile(10_000)
    assert abs(q - 0.99) < 1e-9
    assert A.resolve_clip_quantile(100) < q < A.resolve_clip_quantile(1_000_000)


def test_temps_derive_from_horizon():
    short = A.resolve_temps(20)
    long = A.resolve_temps(500)
    assert short[-1] == 1.0 and long[-1] == 1.0
    assert len(long) >= len(short)  # longer horizon -> finer grid
    assert A.resolve_temps(100, user_temps=(0.3, 1.0)) == (0.3, 1.0)


def test_blend_range_scales():
    lo_s, hi_s = A.resolve_blend_range(25)
    lo_b, hi_b = A.resolve_blend_range(400)
    assert lo_b < lo_s  # more data -> trust DR earlier
    assert hi_b < hi_s
    assert lo_b < hi_b


def test_gate_autotunes():
    g_small = A.resolve_gate(10, 5.0, 50.0, None)
    g_big = A.resolve_gate(500, 5.0, 50.0, None)
    assert g_small["min_ess_frac"] > g_big["min_ess_frac"]
    assert g_small["rel_edge_std"] >= g_big["rel_edge_std"]
    # Pinning wins.
    g = A.resolve_gate(10, 1.0, 0.0, {"min_ess_frac": 0.07})
    assert g["min_ess_frac"] == 0.07


def test_bar_avoids_degenerate_std():
    # Expert diet with ~0 std still demands a relative lift, not 1e-6 absolute.
    bar = A.resolve_bar(80.0, 1e-9, 0.5)
    assert bar > 80.0 + 1.0
    assert A.resolve_bar(50.0, 10.0, 0.5) == 55.0


def test_fqe_cfg_scales_with_data():
    small = _toy_diet(N=1000, n_ep=20)
    big = _toy_diet(N=40000, n_ep=200, obs_dim=8)
    cs = A.resolve_fqe_cfg(small, None)
    cb = A.resolve_fqe_cfg(big, None)
    assert cb["batch"] >= cs["batch"]
    assert cb["steps_max"] >= cs["steps_max"]
    assert cb["hidden"] >= cs["hidden"]
    assert 200 <= cs["holdout"] <= 2000 or small["N"] < 400


def test_magic_horizons_follow_lengths():
    h = A.resolve_magic_horizons([10] * 50 + [500] * 50)
    assert h[-1] is None
    assert len(h) >= 3  # quantiles + max + DM
    h2 = A.resolve_magic_horizons([5, 5, 5, 6])
    assert h2[-1] is None


def test_resolve_meta_is_deterministic_and_overridable():
    d = _toy_diet()
    m1, i1 = A.resolve_meta(d, 0.99, None)
    m2, _ = A.resolve_meta(d, 0.99, None)
    assert m1 == m2
    assert "derived" in i1
    m3, i3 = A.resolve_meta(d, 0.99, {"bootstrap_B": 999})
    assert m3["bootstrap_B"] == 999
    assert "bootstrap_B" not in i3["derived"]


def test_legacy_meta_warns():
    import warnings
    from white_queen.tribunal.ope import estimators as E
    import numpy as np

    rng = np.random.default_rng(0)
    d = {"obs": rng.normal(size=(200, 4)).astype(np.float32)}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            E._resolve_meta(d, 0.99, E.META)
        except Exception:
            pass  # tiny diet may fail resolve; warning fires first
    assert any(issubclass(x.category, FutureWarning) for x in w)


def test_steps_scale_with_horizon():
    from white_queen.tribunal.ope.autotune import _shared_net_cfg

    small = _shared_net_cfg(1500, 4, 2, None, gamma=0.99)
    big = _shared_net_cfg(153064, 4, 2, None, gamma=0.99)
    # Long-horizon toy from the asymptote probe (gamma .9933 -> H=150):
    # horizon term must bind above N//2.
    toy = _shared_net_cfg(1500, 4, 2, None, gamma=0.9933)
    assert toy["steps_max"] > 1500 // 2  # N//2 alone would starve it
    assert big["steps_max"] > 153064 // 2  # horizon term binds at scale too
    assert small["steps_max"] >= 5000  # floor holds for tiny diets
    # Explicit override always wins.
    assert _shared_net_cfg(1500, 4, 2, {"steps_max": 123}, gamma=0.99)["steps_max"] == 123
