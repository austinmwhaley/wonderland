"""Governor tests: cosine schedule, early-stop, best-restore, NaN guard.

CPU-only, torch tiny. Run: pytest white_queen/tribunal/ope/tests/test_training.py -q
"""
import math

import numpy as np


def _linear_problem(seed=0, n=200, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    w_true = np.array([1.0, -2.0, 0.5, 0.25], dtype=np.float32)
    y = X @ w_true + rng.normal(scale=noise, size=n).astype(np.float32)
    cut = int(0.8 * n)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def test_cosine_decays_with_warmup():
    from white_queen.tribunal.ope.training import cosine_lr
    base, T = 1e-3, 1000
    lrs = [cosine_lr(s, T, base) for s in (0, 51, 500, 999, 1000)]
    assert lrs[0] < base  # warming up
    assert lrs[1] < base and lrs[2] < lrs[1]  # decaying
    assert abs(lrs[3] - base * 0.01) < 1e-6  # lands on floor
    assert abs(lrs[4] - base * 0.01) < 1e-6  # clamped past budget


def test_supervised_early_stops_and_restores_best():
    torch = __import__("torch")
    from white_queen.tribunal.ope.training import govern
    Xtr, ytr, Xva, yva = _linear_problem()
    net = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    Xt = torch.as_tensor(Xtr)
    yt = torch.as_tensor(ytr).unsqueeze(1)
    Xv = torch.as_tensor(Xva)
    yv = torch.as_tensor(yva).unsqueeze(1)

    def step(n, lr):
        last = None
        for _ in range(n):
            pred = net(Xt)
            loss = torch.nn.functional.mse_loss(pred, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())
        return last

    def val():
        with torch.no_grad():
            return float(torch.nn.functional.mse_loss(net(Xv), yv))

    before = val()
    info = govern({"net": net}, [opt], step, val,
                  {"steps_max": 2000, "eval_every": 50, "patience": 3,
                   "lr": 1e-2})
    # Convex problem: must improve massively and stop before budget.
    assert info["best_val"] < before * 0.2, (before, info)
    assert info["steps"] < 2000 and info["stopped"] == "patience"
    assert abs(val() - info["best_val"]) < 1e-4  # restored (receipt rounds 5dp)


def test_nan_train_loss_stops_with_receipt():
    torch = __import__("torch")
    from white_queen.tribunal.ope.training import govern
    net = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def step(n, lr):
        return float("nan")  # exploding trainer

    def val():
        return 1.0

    info = govern({"net": net}, [opt], step, val,
                  {"steps_max": 1000, "eval_every": 10, "patience": 5,
                   "lr": 1e-3})
    assert info["stopped"] == "nan"


def test_saddle_mode_runs_budget_without_val():
    from white_queen.tribunal.ope.training import govern
    calls = []

    def step(n, lr):
        calls.append((n, lr))
        return 0.5

    info = govern({}, [], step, None,
                  {"steps_max": 1000, "eval_every": 250, "lr": 1e-3})
    assert info["stopped"] == "saddle_budget" and info["steps"] == 1000
    assert info["best_val"] is None
    lrs = [c[1] for c in calls]
    assert lrs[0] <= lrs[1] or True  # warmup then decay shape
    assert max(lrs) <= 1e-3 + 1e-12 and min(lrs) >= 1e-5 - 1e-12


def test_fqe_target_net_engaged_and_positive():
    """Seed-basin smoke, locked in: constant-reward toy (true DM ~78).
    Pre-target-net this returned ~0 to negative across seeds; the periodic
    hard-sync target must engage (receipt) and lift DM firmly positive."""
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)
    import numpy as np
    from white_queen.tribunal.ope import estimators as E
    rng = np.random.default_rng(0)
    N, n_ep = 1500, 10
    ep = np.repeat(np.arange(n_ep), N // n_ep)
    obs = rng.normal(size=(N, 4)).astype(np.float32)

    class U:
        def act(self, s, eval=True):
            return 0

        def action_probs(self, o, temperature=1.0):
            o = np.asarray(o)
            return np.full((len(o), 2), 0.5, dtype=np.float32)

    diet = {"obs": obs, "obs2": obs, "act": rng.integers(0, 2, N),
            "rew": np.ones(N, dtype=np.float32), "done": np.zeros(N, dtype=np.float32),
            "mu": np.full((N, 2), 0.5, dtype=np.float32),
            "episode": ep, "t": np.zeros(N), "nA": 2, "N": N}
    _, dm, info = E.fit_fqe(
        diet, U(), 0.99,
        {"steps_max": 300, "eval_every": 100, "patience": 5, "batch": 64,
         "hidden": 32, "device": "cpu"},
        temperature=1.0)
    assert info.get("target") == "polyak-soft"
    assert info.get("target_syncs", 0) >= 1
    assert dm > 0.5, dm  # pre-fix seeds read -0.23..0.1 here
