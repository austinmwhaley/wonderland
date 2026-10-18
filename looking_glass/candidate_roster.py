"""Candidate roster (Layer C): several offline policies that read the frozen `E`, each a
white_queen `CandidateProtocol` (act / action_probs). white_queen ranks + certifies them.

For a per-period contextual-bandit decision (the reference case) the offline-RL families collapse to
`argmax_a f(E, a)` with different `f`:
  * value  — regression V(E,a)->reward, act = argmax V
  * bc     — behaviour cloning: P(logged action | E), act = argmax P (imitate the log)
  * iql    — implicit-Q: expectile-weighted regression Q(E,a)->reward, act = argmax Q
  * cql    — conservative-Q: a bootstrap ensemble, act = argmax (mean - alpha*std)  [pessimistic]
  * dt     — decision-transformer (bandit form): P(reward in the top quantile | E,a), act = argmax
Plus controls: the logging/behaviour policy (must HOLD) and, if truth is given, the oracle.

No logged propensity is required (white_queen estimates it — the real-world default).
"""

from __future__ import annotations

import numpy as np


class ScoreCandidate:
    """CandidateProtocol driven by per-action scores f(E, a)."""

    def __init__(self, mean, std, q, scores_fn, name="cand"):
        self._mean = np.asarray(mean, dtype=np.float64)
        self._std = np.asarray(std, dtype=np.float64) + 1e-9
        self._Q = int(q)
        self._f = scores_fn
        self.name = name

    def _scores(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        E = (o[:, 1:] - self._mean) / self._std if o.ndim == 2 and o.shape[1] > 1 else o
        return self._f(np.atleast_2d(E))

    def action_probs(self, obs, temperature: float = 1.0):
        S = self._scores(obs)
        if temperature <= 0 or temperature == 1.0:
            p = np.zeros_like(S)
            p[np.arange(len(S)), np.argmax(S, 1)] = 1.0
            return p
        Z = S / temperature
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(1, keepdims=True)

    def act(self, state, eval: bool = True):
        return int(np.argmax(self._scores(np.atleast_2d(state))[0]))


def build_roster(E, A, y, *, alpha=0.5, seed=0):
    """Fit the roster on the logged (E, A, y). E is already reduced/standardized by the caller."""
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    Q = int(A.max()) + 1
    mean = np.zeros(E.shape[1])
    std = np.ones(E.shape[1])
    cols = np.column_stack([E, A.astype(float)])

    def _reg(Xtr, ytr, w=None):
        m = HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=30)
        m.fit(Xtr, ytr, sample_weight=w)
        return m

    def _score_from_reg(m):
        return lambda Ei: np.column_stack(
            [m.predict(np.column_stack([Ei, np.full(len(Ei), float(a))])) for a in range(Q)]
        )

    out = {}
    # 1) value: regression
    mv = _reg(cols, y)
    out["value"] = ScoreCandidate(mean, std, Q, _score_from_reg(mv), "value")

    # 2) bc: behaviour cloning (imitate the logged action)
    mc = HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=30).fit(E, A)
    classes = list(mc.classes_)

    def _bc(Ei):
        P = mc.predict_proba(Ei)  # (n, n_classes)
        S = np.full((len(Ei), Q), -50.0)
        for j, c in enumerate(classes):
            if 0 <= int(c) < Q:
                S[:, int(c)] = np.log(np.clip(P[:, j], 1e-6, 1.0))
        return S

    out["bc"] = ScoreCandidate(mean, std, Q, _bc, "bc")

    # 3) iql: expectile-weighted regression (up-weight high returns)
    tau = 0.7
    w = np.where(y >= np.quantile(y, tau), 1.0, 0.25)
    mi = _reg(cols, y, w)
    out["iql"] = ScoreCandidate(mean, std, Q, _score_from_reg(mi), "iql")

    # 4) cql: pessimistic bootstrap ensemble (mean - alpha*std over actions' uncertainty)
    K = 4
    rng = np.random.default_rng(seed)
    preds = []
    for k in range(K):
        idx = rng.integers(0, len(y), len(y))
        mk = _reg(cols[idx], y[idx])
        preds.append(_score_from_reg(mk))

    def _cql(Ei):
        St = np.stack([f(Ei) for f in preds], 0)  # (K, n, Q)
        return St.mean(0) - alpha * St.std(0)

    out["cql"] = ScoreCandidate(mean, std, Q, _cql, "cql")

    # 5) dt: decision-transformer bandit form — P(top-quantile reward | E, a)
    thr = np.quantile(y, 0.75)
    lbl = (y >= thr).astype(int)
    if lbl.min() != lbl.max():
        md = HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=30).fit(cols, lbl)

        def _dt(Ei):
            return np.column_stack(
                [
                    md.predict_proba(np.column_stack([Ei, np.full(len(Ei), float(a))]))[:, 1]
                    for a in range(Q)
                ]
            )

        out["dt"] = ScoreCandidate(mean, std, Q, _dt, "dt")
    return out, mean, std


def certify_roster(
    model, cfg, action_key="channel", risk_aversion=0.5, fast=True, oracle_pref=None, names=None
):
    """Run white_queen on the whole roster for one per-period action key. Returns per-candidate
    verdicts. No logged propensity is passed (white_queen estimates it)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from looking_glass.use_case_gate import _period_rows
    from looking_glass.head_battery import _b_states
    from white_queen.tribunal.ope.api import evaluate

    cust, Z = _b_states(model, cfg)
    j = _period_rows(cfg, cust, action_key)
    ci = {c: i for i, c in enumerate(cust)}
    cu = j["customer_key"].to_numpy()
    A = j["action"].to_numpy().astype(int)
    y = (j["y"].to_numpy().astype(float) > 0).astype(float)
    _cap = 4000
    if len(A) > _cap:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(A), _cap, replace=False))
        cu, A, y = cu[sel], A[sel], y[sel]
    Zrow = Z[[ci[c] for c in cu]]
    Zs = StandardScaler().fit_transform(Zrow)
    # DEC-194: a 64-dim reduction is LOSSY (drops the response); keep up to 128 dims so the
    # candidates see the same signal the bridge's full-E policy does.
    Er = PCA(n_components=min(128, Zs.shape[1], len(A) - 1), random_state=0).fit_transform(Zs)
    roster, mean, std = build_roster(Er, A, y)
    ctx = np.column_stack([np.arange(len(A)), Er]).astype(np.float32)
    src = {"obs": ctx, "action": A, "reward": y}  # no propensity (estimated)
    results = {}
    for name, cand in roster.items():
        if names and name not in names:
            continue
        try:
            rep = evaluate(
                src,
                cand,
                candidate_name=name,
                fast=fast,
                risk_aversion=risk_aversion,
                fqe_cfg={"steps_max": 4000, "patience": 3},
            )
            results[name] = {
                "deploy": bool(rep["deploy"]),
                "dr": rep["estimates"].get("dr"),
                "fqe": rep["estimates"].get("fqe"),
                "wis": rep["estimates"].get("wis"),
            }
        except Exception as e:  # noqa: BLE001
            results[name] = {"deploy": False, "error": str(e)[:80]}
    return results


if __name__ == "__main__":  # pragma: no cover
    import argparse

    from looking_glass.cfm_state import load_frozen_encoder

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--action", default="channel")
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--customers", type=int, default=4000)
    a = ap.parse_args()
    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    print("# CANDIDATE ROSTER (white_queen, no logged propensity)")
    _nm = tuple(a.candidates.split(",")) if a.candidates else None
    for name, r in certify_roster(model, cfg, a.action, names=_nm).items():
        print(f"  {name:6s} deploy={r['deploy']!s:5s} dr={r.get('dr')} fqe={r.get('fqe')}")
