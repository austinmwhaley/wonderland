"""Bridge: certify a looking_glass `E`-policy with the REAL white_queen OPE library.

`looking_glass.decision_certify` is an in-house OPE; this module puts the **white_queen**
tribunal (`white_queen.tribunal.ope.api.evaluate`) directly in the loop, so the apex report
card reports the actual white_queen verdict alongside the in-house one.

Design: build the logged dataset the tribunal expects from the stream —
    context = the frozen embedding `E` (so FQE/DR can use it), action = the randomized ARM,
    reward = logged future margin, propensity = the empirical marginal (randomized arms).
The candidate policy reads `E` (a value model `V(E,a)` -> argmax). `oracle_policy` optionally
certifies the KNOWN-truth policy (the apex's `arm_pref`) as a positive control.

Deploy is decided by white_queen (never-ship-worse + its conservative bar); we only pass it E.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from looking_glass.cfm_state import load_frozen_encoder
from looking_glass.head_battery import _b_states


class EPolicy:
    """A policy defined by a value model V(E, a): act = argmax_a V(E, a)."""

    def __init__(self, vmodel, mean, std, q, pref=None):
        self._m = vmodel
        self._mean = np.asarray(mean, dtype=np.float64)
        self._std = np.asarray(std, dtype=np.float64) + 1e-9
        self._Q = int(q)
        self._pref = pref  # optional per-row index policy override (the oracle control)

    def _scores(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        cols = o[:, 1:] if (o.ndim == 2 and o.shape[1] > 1) else o
        E = (cols - self._mean) / self._std
        return np.column_stack(
            [
                self._m.predict(np.column_stack([E, np.full(len(E), float(a))]))
                for a in range(self._Q)
            ]
        )

    def action_probs(self, obs, temperature: float = 1.0):
        o = np.asarray(obs, dtype=np.float64)
        if self._pref is not None:
            idx = o[:, 0].astype(int) if o.ndim == 2 else np.arange(len(o))
            p = np.zeros((len(idx), self._Q))
            p[np.arange(len(idx)), self._pref[idx]] = 1.0
            return p
        S = self._scores(o)
        p = np.zeros_like(S)
        p[np.arange(len(S)), np.argmax(S, 1)] = 1.0
        return p

    def act(self, state, eval: bool = True):
        o = np.atleast_2d(np.asarray(state, dtype=np.float64))
        return int(np.argmax(self.action_probs(o)[0]))


def certify_period(
    model,
    cfg,
    action_key: str = "channel",
    risk_aversion: float = 0.5,
    fast=True,
    pref_by_customer=None,
):
    """white_queen on a PER-PERIOD randomized action (e.g. the weekly `channel`/`offer`/`freq`):
    context = the customer's frozen `E`; action = the logged action that period; reward = whether
    the outcome fired that period. The candidate policy reads `E` (V(E,a)->argmax)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from looking_glass.use_case_gate import _period_rows
    from white_queen.tribunal.ope.api import evaluate

    cust, Z = _b_states(model, cfg)
    j = _period_rows(cfg, cust, action_key)
    ci = {c: i for i, c in enumerate(cust)}
    cu = j["customer_key"].to_numpy()
    A = j["action"].to_numpy().astype(int)
    y = (j["y"].to_numpy().astype(float) > 0).astype(float)
    # white_queen's panel cost scales with rows (FQE budget ~ H*N/batch); subsample periods to
    # a bounded, seeded set (still an unbiased OPE sample).
    _cap = 4000
    if len(A) > _cap:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(A), _cap, replace=False))
        cu, A, y = cu[sel], A[sel], y[sel]
    Q = int(A.max()) + 1
    N = len(A)
    Zrow = Z[[ci[c] for c in cu]]
    mean, std = Z.mean(0), Z.std(0)
    sc = ((Zrow - mean) / (std + 1e-9)).astype(np.float32)
    vm = HistGradientBoostingRegressor(max_iter=300, min_samples_leaf=50).fit(
        np.column_stack([sc, A.astype(float)]), y
    )
    if pref_by_customer is not None:
        # ORACLE control: a fixed per-customer policy (e.g. the truth channel_pref).
        pref_arr = np.array([int(pref_by_customer.get(str(c), 0)) for c in cu])
        cand = EPolicy(None, mean, std, Q, pref=pref_arr)
        _name = f"ORACLE-{action_key}"
    else:
        cand = EPolicy(vm, mean, std, Q)
        _name = f"E-{action_key}"
    ctx = np.column_stack([np.arange(N), sc]).astype(np.float32)
    # NO logged propensity (the real-world default: the human/logging policy is unrecorded).
    # white_queen ESTIMATES P(action|context) from the stream. Pass a real column only if you
    # truly have it.
    src = {"obs": ctx, "action": A, "reward": y}
    rep = evaluate(
        src,
        cand,
        candidate_name=_name,
        fast=fast,
        risk_aversion=risk_aversion,
        fqe_cfg={"steps_max": (1500 if fast else 4000), "patience": 3},
    )
    return {
        "deploy": bool(rep["deploy"]),
        "behavior_mean": rep.get("behavior_mean"),
        "bar": rep.get("bar"),
        "estimates": {
            k: rep["estimates"].get(k) for k in ("dr", "fqe", "wis") if k in rep["estimates"]
        },
    }


def certify(model, cfg, horizon_days: int = 90, risk_aversion: float = 0.5, fast: bool = True):
    """Run the white_queen tribunal on an `E`-policy + the apex arm log. Returns its verdict."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from looking_glass.decision_certify import _arms_rewards
    from white_queen.tribunal.ope.api import evaluate

    cust, Z = _b_states(model, cfg)
    arm, rew = _arms_rewards(cfg, cust, horizon_days)
    j = (
        pl.DataFrame({"customer": cust})
        .join(arm, left_on="customer", right_on="customer_key", how="left")
        .join(rew, left_on="customer", right_on="customer_key", how="left")
        .fill_null(0.0)
    )
    a = j["arm"].to_numpy().astype(int)
    r = j["gm"].to_numpy().astype(float)
    Q = int(a.max()) + 1
    N = len(a)
    mean, std = Z.mean(0), Z.std(0)
    sc = ((Z - mean) / (std + 1e-9)).astype(np.float32)
    # context = [row_index, E] so the policy can either look up V(E,a) or, for the oracle
    # control, index a known per-row action.
    ctx = np.column_stack([np.arange(N), sc]).astype(np.float32)
    vm = HistGradientBoostingRegressor(max_iter=300, min_samples_leaf=50).fit(
        np.column_stack([sc, a.astype(float)]), r
    )
    cand = EPolicy(vm, mean, std, Q)
    # NO logged propensity (estimated by white_queen); pass a real column only if logged.
    src = {"obs": ctx, "action": a, "reward": r}
    rep = evaluate(src, cand, candidate_name="E-policy", fast=fast, risk_aversion=risk_aversion)
    return {
        "deploy": bool(rep["deploy"]),
        "behavior_mean": rep.get("behavior_mean"),
        "bar": rep.get("bar"),
        "estimates": {
            k: rep["estimates"].get(k) for k in ("dr", "fqe", "wis") if k in rep["estimates"]
        },
        "rationale": rep.get("rationale", ""),
    }


def oracle(model, cfg, truth_json, horizon_days: int = 90, risk_aversion: float = 0.5, fast=True):
    """white_queen on the KNOWN-truth policy (arm_pref) — the positive control."""
    import json

    from looking_glass.decision_certify import _arms_rewards
    from white_queen.tribunal.ope.api import evaluate

    cust, Z = _b_states(model, cfg)
    truth = {c["customer"]: c for c in json.loads(open(truth_json).read())["customers"]}
    pref = np.array([truth[c].get("arm_pref", 0) for c in cust])
    arm, rew = _arms_rewards(cfg, cust, horizon_days)
    j = (
        pl.DataFrame({"customer": cust})
        .join(arm, left_on="customer", right_on="customer_key", how="left")
        .join(rew, left_on="customer", right_on="customer_key", how="left")
        .fill_null(0.0)
    )
    a = j["arm"].to_numpy().astype(int)
    r = j["gm"].to_numpy().astype(float)
    Q = int(a.max()) + 1
    N = len(a)
    mean, std = Z.mean(0), Z.std(0)
    sc = ((Z - mean) / (std + 1e-9)).astype(np.float32)
    ctx = np.column_stack([np.arange(N), sc]).astype(np.float32)
    pol = EPolicy(None, mean, std, Q, pref=pref)
    # NO logged propensity (estimated by white_queen); pass a real column only if logged.
    src = {"obs": ctx, "action": a, "reward": r}
    rep = evaluate(src, pol, candidate_name="ORACLE", fast=fast, risk_aversion=risk_aversion)
    return {
        "deploy": bool(rep["deploy"]),
        "behavior_mean": rep.get("behavior_mean"),
        "bar": rep.get("bar"),
        "estimates": {
            k: rep["estimates"].get(k) for k in ("dr", "fqe", "wis") if k in rep["estimates"]
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument("--risk", type=float, default=0.5)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--customers", type=int, default=12000)
    a = ap.parse_args()
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    print("# WHITE_QUEEN (real tribunal) — E-policy")
    r = certify(model, cfg, a.horizon_days, a.risk)
    print(r)
    if a.truth:
        print("# WHITE_QUEEN — ORACLE (arm_pref, positive control)")
        print(oracle(model, cfg, a.truth, a.horizon_days, a.risk))
