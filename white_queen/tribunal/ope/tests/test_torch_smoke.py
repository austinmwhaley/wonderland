"""Torch smoke: core PyTorch path in <2 min (no 30-min tribunal).

Exercises fit_fqe + ensemble_fqe + learn_dynamics on a tiny synthetic diet
with a uniform mock candidate. Catches import breaks, NaN crashes, schema
drift. Run: pytest white_queen/tribunal/ope/tests/test_torch_smoke.py -q
"""
import numpy as np


class UniformCandidate:
    def __init__(self, nA=2):
        self.nA = nA

    def act(self, state, eval=True):
        return 0

    def action_probs(self, obs, temperature=1.0):
        o = np.asarray(obs)
        n = len(o) if o.ndim > 1 else 1
        return np.full((n, self.nA), 1.0 / self.nA, dtype=np.float32)


def _diet(N=800, n_ep=10, seed=0):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), N // n_ep)
    N = len(ep)
    obs = rng.normal(size=(N, 4)).astype(np.float32)
    return {
        "obs": obs,
        "obs2": np.roll(obs, -1, axis=0),
        "act": rng.integers(0, 2, N),
        "rew": rng.normal(size=N).astype(np.float32),
        "done": np.zeros(N, dtype=np.float32),
        "mu": np.full((N, 2), 0.5, dtype=np.float32),
        "episode": ep,
        "t": np.tile(np.arange(N // n_ep), n_ep)[:N],
        "nA": 2,
        "N": N,
    }


def test_fit_fqe_tiny():
    torch = __import__("torch")
    from white_queen.tribunal.ope import estimators as E
    d = _diet()
    cand = UniformCandidate()
    q, dm, info = E.fit_fqe(d, cand, 0.99,
                            {"steps_max": 200, "eval_every": 50,
                             "patience": 2, "batch": 64, "hidden": 32},
                            temperature=1.0)
    assert np.isfinite(dm), f"FQE DM not finite: {dm}"
    assert info["steps"] <= 200


def test_ensemble_and_dynamics_tiny():
    from white_queen.tribunal.ope.direct import ensemble_fqe
    from white_queen.tribunal.ope.model_based import learn_dynamics, rollout_estimate
    d = _diet()
    cand = UniformCandidate()
    ef = ensemble_fqe(d, cand, 0.99,
                      {"steps_max": 200, "eval_every": 50, "patience": 2,
                       "batch": 64, "hidden": 32}, temperature=1.0, K=2)
    assert np.isfinite(ef["mean"])
    step_fn, info = learn_dynamics(d, hidden=32, batch=64, steps_max=200,
                                   eval_every=50, patience=2)
    assert np.isfinite(info["val_mse"])
    mb = rollout_estimate(d, cand, 0.99, step_fn, temperature=1.0,
                          sim_min=10, sim_max=20)
    assert np.isfinite(mb["mb"]), f"MB not finite: {mb}"
