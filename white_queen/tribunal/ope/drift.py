"""Distribution-drift diagnostics for time-ordered logs.

Real logs span years; behavior, population, and reward scale drift. Pooling
them silently invalidates OPE. We cannot "fix" non-stationarity from old
data, but we can detect it and refuse to hide it: split the log by time into
periods, report per-period behavior return and the standardized shift in
context/reward between the earliest and latest period.

A `drift_flag` (obs or reward shift > 1 std) means the pool is not stationary
and per-period behavior/conclusions should be recomputed rather than pooled.
"""
from __future__ import annotations

import numpy as np

from .receipts import _segment_disc_returns


def _period_index(diet, n_periods):
    N = diet["N"]
    ts = diet.get("timestamp")
    if ts is not None:
        ts = np.asarray(ts)
        try:
            order = np.argsort(ts)
        except Exception:
            order = np.arange(N)
    else:
        order = np.arange(N)
    return np.array_split(order, n_periods)


def drift_report(diet, gamma=0.99, n_periods=4):
    """Return a drift receipt over time-ordered periods (timestamp if present,
    else input order)."""
    N = diet["N"]
    n_periods = max(int(n_periods), 2)
    obs = np.asarray(diet["obs"], dtype=np.float64)
    rew = np.asarray(diet["rew"], dtype=np.float64)
    ep = np.asarray(diet["episode"])
    parts = [p for p in _period_index(diet, n_periods) if len(p) > 0]
    period_returns, period_mean_rew = [], []
    for p in parts:
        # per-episode discounted returns restricted to episodes present here
        sub = {"rew": rew[p], "episode": ep[p]}
        vals = _segment_disc_returns(sub, gamma)
        period_returns.append(round(float(vals.mean()), 2) if len(vals) else None)
        period_mean_rew.append(float(rew[p].mean()))
    obs_sd = obs.std(0) + 1e-9
    obs_shift = float(np.max(np.abs(obs[parts[-1]].mean(0)
                                    - obs[parts[0]].mean(0)) / obs_sd))
    rew_sd = float(np.std(rew)) + 1e-9
    rew_shift = abs(period_mean_rew[-1] - period_mean_rew[0]) / rew_sd
    drift_flag = bool(obs_shift > 1.0 or rew_shift > 1.0)
    return {
        "n_periods": len(parts),
        "period_returns": period_returns,
        "period_mean_reward": [round(x, 3) for x in period_mean_rew],
        "obs_shift_std": round(obs_shift, 3),
        "reward_shift_std": round(rew_shift, 3),
        "drift_flag": drift_flag,
        "note": ("pool is non-stationary — recompute behavior/conclusions per "
                 "period rather than pooling" if drift_flag
                 else "no strong drift detected between first and last period"),
    }
