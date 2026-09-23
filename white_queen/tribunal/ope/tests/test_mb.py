"""World-model termination-head tests.

Before this head, the rollout never ended early, so on CartPole (score =
survival) MB read ~99 for every policy including random — it measured
max_len reward, not performance, and could not corroborate anything. These
pin the fix: rollout must terminate on the learned done signal, so a
short-survival policy scores far below a long-survival one.

Run: pytest .../tests/test_mb.py -q  (numpy only, instant)
"""
import numpy as np


def _diet(N=400, n_ep=10, seed=0):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), N // n_ep)
    N = len(ep)
    obs = rng.normal(size=(N, 1)).astype(np.float32)
    return {
        "obs": obs, "obs2": obs, "act": rng.integers(0, 2, N),
        "rew": np.ones(N, dtype=np.float32),
        "done": np.zeros(N, dtype=np.float32),
        "mu": np.full((N, 2), 0.5, dtype=np.float32),
        "episode": ep, "t": np.zeros(N), "nA": 2, "N": N,
    }


class _FixedPolicy:
    def __init__(self, action, nA=2):
        self.a, self.nA = action, nA

    def act(self, s, eval=True):
        return self.a

    def action_probs(self, o, temperature=1.0):
        o = np.asarray(o)
        n = len(o) if o.ndim > 1 else 1
        p = np.zeros((n, self.nA), dtype=np.float32)
        p[:, self.a] = 1.0
        return p


def test_rollout_terminates_on_done_head():
    from white_queen.tribunal.ope.model_based import rollout_estimate
    # Synthetic world: action 0 survives to max_len; action 1 terminates at
    # step 3. reward = 1/step. Survival *is* the score.
    def step_fn(obs, act):
        o = np.asarray(obs, dtype=float)
        r = np.ones(len(o))
        dp = np.where(np.asarray(act) == 1, 0.9, 0.0)
        return o, r, dp

    d = _diet()
    good = rollout_estimate(d, _FixedPolicy(0), 0.99, step_fn,
                            sim_min=20, sim_max=20)
    bad = rollout_estimate(d, _FixedPolicy(1), 0.99, step_fn,
                           sim_min=20, sim_max=20)
    assert good["mb"] > 10 * max(bad["mb"], 1e-9), (good, bad)
    assert bad["done_hits"] > 0 and good["done_hits"] == 0
    # Discriminating again: the whole point of the termination head.
    assert good["mb"] - bad["mb"] > 0.5


def test_dynamics_reports_termination_head():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)
    from white_queen.tribunal.ope.model_based import learn_dynamics
    d = _diet()
    step_fn, info = learn_dynamics(d, hidden=16, batch=32, steps_max=60,
                                   eval_every=30, patience=2, seed=0)
    assert info["termination_head"] is True
    _, r, dp = step_fn(d["obs"][:5], d["act"][:5])
    assert r.shape == (5,) and dp.shape == (5,)
    assert np.all((dp >= 0.0) & (dp <= 1.0))
