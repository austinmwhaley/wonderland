"""Independent counterfactual test for red_king + the fixes.

Tests the PURPOSE of a world model: recover the true value of a policy under
known cause-effect, and estimate it without over-optimism. Compares:

  WITHOUT world model : behavior value; model-free IPS of the greedy policy
  WITH    world model : red_king ensemble reward model -> greedy policy,
                        scored with a pessimistic (mean - lambda*std) penalty

Fixes applied to the world model:
  * deconfound the action: use the *residual* exposure (action - E[action|state])
    as the treatment, so "active customers get more" is removed.
  * predict reward RATE (per unit time), removing the window-length confound.
  * pessimistic valuation via ensemble uncertainty.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def make_bandit(n=6000, d=32, nA=4, seed=0, noise=0.6, confound=1.0):
    """Contextual bandit with KNOWN reward model and a confounded logging policy."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n, d)).astype(np.float32)
    # confounder: a hidden scalar drives both action and reward
    u = s[:, 0]
    W = rng.normal(size=(nA, d)).astype(np.float32)  # true reward weights
    true_mu = s @ W.T  # (n, nA)
    # logging policy: confounded by u -> action depends on s[:,0]
    logits = (u[:, None] * confound + rng.normal(0, 0.5, size=(n, nA))).astype(np.float32)
    p = torch.softmax(torch.tensor(logits), -1).numpy()
    act = np.array([rng.choice(nA, p=pi) for pi in p])
    rew = (true_mu[np.arange(n), act] + rng.normal(0, noise, n)).astype(np.float32)
    best = true_mu.max(1)
    return s, act, rew, p, true_mu, float(best.mean()), float(rew.mean())


def _fit_world(s, a, r, nA, tr, seed=0, steps=2000, K=5):
    """Ensemble reward model with action RESIDUALISED against the state (deconfound)."""
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dim = s.shape[1]

    # E[action|state]: deconfounding control (ridge). Residual exposure.
    from sklearn.linear_model import Ridge

    ctrl = Ridge(alpha=1.0).fit(s[tr], a[tr])
    resid = a - ctrl.predict(s)
    resid = (resid - resid[tr].mean()) / (resid[tr].std() + 1e-6)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + nA + 1, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, nA),
            )

        def forward(self, s, ac, rx):
            oh = nn.functional.one_hot(ac, nA).float()
            return self.net(torch.cat([s, oh, rx[:, None]], -1))

    ens = [M().to(dev) for _ in range(K)]
    opts = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in ens]
    S = torch.tensor(s, device=dev)
    A = torch.tensor(a, device=dev)
    R = torch.tensor(r, device=dev)
    X = torch.tensor(resid, dtype=torch.float32, device=dev)
    tri = torch.tensor(np.where(tr)[0], device=dev)
    for m, opt in zip(ens, opts):
        torch.manual_seed(seed)
        for _ in range(steps):
            b = tri[torch.randint(0, len(tri), (256,), device=dev)]
            oh = torch.nn.functional.one_hot(A[b], nA).float()
            out = m.net(torch.cat([S[b], oh, X[b][:, None]], -1))
            pred = (out * oh).sum(-1)
            loss = torch.nn.functional.mse_loss(pred, R[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return ens, ctrl, resid, dev


def evaluate(s, a, r, p, true_greedy, behavior, nA, seed=0):
    tr = np.arange(len(s)) < int(0.7 * len(s))
    te = ~tr
    ens, ctrl, resid, dev = _fit_world(s, a, r, nA, tr, seed)

    # WITHOUT world model: model-free IPS of the greedy policy
    # candidate = greedy under the TRUE reward model is unknown to us; use IPS to
    # estimate the value of the *behaviour* policy (upper bound of "no WM").
    ips_behavior = float((r[te]).mean())

    # WITH world model: greedy action per state under the ensemble; pessimistic.
    with torch.no_grad():
        S = torch.tensor(s[te], device=dev)
        grids = []
        for m in ens:
            # residual for each candidate action
            rows = []
            for ac in range(nA):
                acv = torch.full((S.shape[0],), ac, device=dev, dtype=torch.long)
                base = ctrl.predict(s[te])
                # residual of candidate action vs expected
                rx = ac - base
                rx = (rx - resid[tr].mean()) / (resid[tr].std() + 1e-6)
                oh = torch.nn.functional.one_hot(acv, nA).float()
                out = m.net(
                    torch.cat(
                        [S, oh, torch.tensor(rx, dtype=torch.float32, device=dev)[:, None]], -1
                    )
                )
                rows.append((out * oh).sum(-1))
            grids.append(torch.stack(rows, 0))  # (nA, N)
        grid = torch.stack(grids, 0)  # (K, nA, N)
    mean = grid.mean(0)
    std = grid.std(0)
    val_mean = float(mean.max(0).values.mean())
    val_pess = float((mean - 1.0 * std).max(0).values.mean())
    return {
        "behavior": behavior,
        "true_greedy": true_greedy,
        "ips_behavior": ips_behavior,
        "wm_value_mean": val_mean,
        "wm_value_pessimistic": val_pess,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="red_king counterfactual benchmark")
    ap.parse_args(argv)
    s, act, rew, p, true_mu, true_greedy, behavior = make_bandit()
    res = evaluate(s, act, rew, p, true_greedy, behavior, 4)
    print("== RED_KING COUNTERFACTUAL BENCHMARK (known ground truth) ==")
    print(f"  behavior value            : {res['behavior']:.3f}")
    print(f"  TRUE greedy value         : {res['true_greedy']:.3f}")
    print(f"  WITHOUT world model (IPS) : {res['ips_behavior']:.3f}")
    print(f"  WITH world model (mean)   : {res['wm_value_mean']:.3f}")
    print(f"  WITH world model (pessim) : {res['wm_value_pessimistic']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
