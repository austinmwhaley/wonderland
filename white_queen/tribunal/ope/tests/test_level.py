"""Level-composite tests: median(FQE, LSTDQ, MB) as the level headline.

Backtest table: 15 full-budget cells (v15 academic + v16cert), columns
(diet, cand, truth, fqe, lstdq, mb). Median must beat FQE alone on bias/RMSE
without wrecking rank — this pins the "stop asking FQE to be the level" fix.
Run: pytest .../tests/test_level.py -q (instant, no torch).
"""

import numpy as np

# (diet, cand, truth, fqe_dm, lstdq_dm, mb)
CELLS = [
    ("expert", "iql", 99.3, 17.1, 63.2, 90.1),
    ("expert", "bc", 98.3, 12.5, 63.5, 98.6),
    ("expert", "cql", 99.1, 11.5, 63.3, 99.3),
    ("mixed", "iql", 94.4, 28.1, 55.6, 99.3),
    ("mixed", "bc", 87.7, 24.0, 54.4, 99.3),
    ("mixed", "cql", 98.4, 26.6, 53.2, 99.3),
    ("novice", "iql", 80.7, 28.0, 30.5, 92.5),
    ("novice", "bc", 80.7, 18.8, 28.9, 93.3),
    ("novice", "cql", 81.3, 24.9, 29.4, 93.1),
    ("mixed", "iql", 94.4, 29.2, 55.6, 99.3),
    ("mixed", "bc", 87.7, 18.7, 54.4, 99.3),
    ("mixed", "cql", 98.4, 23.6, 53.2, 99.3),
    ("novice", "iql", 80.7, 23.2, 30.5, 93.3),
    ("novice", "bc", 80.7, 13.1, 28.9, 93.3),
    ("novice", "cql", 81.3, 18.2, 29.4, 93.3),
]


def _stats(vals):
    v = np.array(vals, float)
    t = np.array([c[2] for c in CELLS], float)
    bias = float((v - t).mean())
    rmse = float(np.sqrt(((v - t) ** 2).mean()))
    rx, ry = np.argsort(np.argsort(v)).astype(float), np.argsort(np.argsort(t)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    rho = float((rx * ry).sum() / max(np.sqrt((rx**2).sum() * (ry**2).sum()), 1e-12))
    return bias, rmse, rho


def level_of(fqe, lstdq, mb):
    return float(np.median([fqe, lstdq, mb]))


def test_median_level_beats_fqe_level():
    lv = [level_of(c[3], c[4], c[5]) for c in CELLS]
    fq = [c[3] for c in CELLS]
    b0, r0, p0 = _stats(fq)
    b1, r1, p1 = _stats(lv)
    assert abs(b1) < abs(b0), (b1, b0)  # -68 -> ~-40
    assert r1 < r0, (r1, r0)  # ~69 -> ~40
    assert p1 >= 0.6, p1  # rank survives aggregation


def test_median_ignores_single_extreme():
    # Collapsed FQE (0.5) beside sane LSTDQ/MB, and ecstatic MB (500.0)
    # beside sane FQE/LSTDQ: median stays with the sane pair both ways.
    assert level_of(0.5, 55.0, 99.0) == 55.0
    assert level_of(50.0, 55.0, 500.0) == 55.0


def test_panel_reports_level_receipts():
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(1)
    from white_queen.tribunal.ope import estimators as E

    rng = np.random.default_rng(0)
    N, n_ep = 600, 12
    ep = np.repeat(np.arange(n_ep), N // n_ep)
    obs = rng.normal(size=(N, 4)).astype(np.float32)

    class U:
        def act(self, s, eval=True):
            return 0

        def action_probs(self, o, temperature=1.0):
            o = np.asarray(o)
            return np.full((len(o), 2), 0.5, dtype=np.float32)

    d = {
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
    tiny = {
        "steps_max": 120,
        "eval_every": 40,
        "patience": 2,
        "batch": 32,
        "hidden": 16,
        "device": "cpu",
    }
    p = E.panel(d, U(), 0.99, meta={"bootstrap_B": 40, "temps": (0.5, 1.0)}, fqe_cfg=tiny)
    assert set(p["level_parts"]) == {"fqe", "lstdq", "mb"}
    assert p["level_est"] == float(
        np.median([p["level_parts"]["fqe"], p["level_parts"]["lstdq"], p["level_parts"]["mb"]])
    )
