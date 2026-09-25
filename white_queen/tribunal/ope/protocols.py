"""Candidate + estimator protocols for the white_queen OPE library.

This module is dependency-free (numpy only, no torch) so tests and
downstream libraries can import it without the RL lab.

Anything the Tribunal can judge implements CandidateProtocol:
  - act(state, eval=True) -> int
  - action_probs(obs_batch, temperature=1.0) -> (N, nA) float matrix
  - optional log_prob_fn(obs, act) -> (N,) for continuous density ratios

Any OPE estimator implements EstimatorProtocol:
  - estimate(diet, candidate, gamma, cfg) -> dict with at least 'point'
    and per-episode 'values' for bootstrapping.

Ratio learners implement RatioProtocol:
  - learn(diet, candidate, gamma, cfg) -> (w_fn, info)
"""

from typing import Any, Callable, Dict, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class CandidateProtocol(Protocol):
    def act(self, state: Any, eval: bool = True) -> int: ...  # noqa: D102

    def action_probs(self, obs: Any, temperature: float = 1.0) -> Any: ...  # noqa: D102


@runtime_checkable
class EstimatorProtocol(Protocol):
    def estimate(
        self,
        diet: Dict[str, Any],
        candidate: CandidateProtocol,
        gamma: float,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]: ...  # noqa: D102


RatioFn = Callable[[Any, Any], Any]


@runtime_checkable
class RatioProtocol(Protocol):
    def learn(
        self,
        diet: Dict[str, Any],
        candidate: CandidateProtocol,
        gamma: float,
        cfg: Dict[str, Any],
    ) -> Tuple[RatioFn, Dict[str, Any]]: ...  # noqa: D102


def check_candidate(cand: Any) -> None:
    """Raise TypeError with a helpful message if cand misses the protocol."""
    missing = []
    if not hasattr(cand, "act") or not callable(getattr(cand, "act")):
        missing.append("act(state, eval=True)")
    if not hasattr(cand, "action_probs") or not callable(getattr(cand, "action_probs")):
        missing.append("action_probs(obs_batch, temperature=1.0)")
    if missing:
        raise TypeError(
            f"Candidate {type(cand).__name__!r} misses CandidateProtocol: "
            + ", ".join(missing)
            + ". See protocols.CandidateProtocol."
        )


def safe_probs(p: Any) -> Any:
    """Sanitize a (N, nA) probability matrix: NaN/Inf/negatives -> uniform.

    Training can emit NaN logits on narrow diets (e.g. expert_only); sampling
    must never crash the panel — it must degrade to uniform and be reported
    via low ESS downstream. Pure numpy, no tuning.
    """
    import numpy as _np

    q = _np.asarray(p, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    q = _np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q = _np.maximum(q, 0.0)
    s = q.sum(axis=1, keepdims=True)
    nA = q.shape[1]
    # Rows with zero mass (all-NaN / all-zero) -> uniform.
    bad = (s.squeeze(1) <= 0) | ~_np.isfinite(s.squeeze(1))
    q[bad] = 1.0 / nA
    s = q.sum(axis=1, keepdims=True)
    return q / s


def sample_actions(rng: Any, probs: Any) -> Any:
    """Sample one action per row with NaN-safe normalization."""
    import numpy as _np

    q = safe_probs(probs)
    return _np.array([rng.choice(q.shape[1], p=row) for row in q])


REQUIRED_DIET_KEYS = ("obs", "obs2", "act", "rew", "done", "mu", "episode", "t", "nA", "N")


def validate_diet(diet: Any) -> dict:
    """Validate a canonical dataset at panel entry. Before: missing keys
    surfaced as KeyError 6 frames deep; ragged/NaN arrays as cryptic matmul
    errors. Returns a fingerprint summary {N, n_episodes, obs_dim, nA}.

    Requires obs/act/rew + (mu_take OR mu). Accepts bandit data where each
    row is its own episode (n_episodes == N) and single-episode RL.
    """
    import numpy as _np

    if not isinstance(diet, dict):
        raise TypeError(f"diet must be dict, got {type(diet).__name__}")
    for k in ("obs", "act", "rew", "nA", "N"):
        if k not in diet:
            raise ValueError(f"diet missing required key {k!r}")
    continuous = bool(diet.get("continuous"))
    if continuous:
        if "logp_take" not in diet:
            raise ValueError("continuous diet needs 'logp_take' (N,) behavior log-density")
    elif "mu_take" not in diet and "mu" not in diet:
        raise ValueError("diet needs a behavior propensity: 'mu_take' (N,) or 'mu' (N,nA)")
    N = int(diet["N"])
    obs = _np.asarray(diet["obs"])
    if obs.ndim != 2 or obs.shape[0] != N:
        raise ValueError(f"diet['obs'] shape {obs.shape} vs N={N}")
    if not _np.isfinite(obs).all():
        raise ValueError("diet['obs'] contains NaN/Inf")
    nA = int(diet["nA"])
    if continuous:
        lp = _np.asarray(diet["logp_take"], dtype=float)
        if lp.shape != (N,) or not _np.isfinite(lp).all():
            raise ValueError("diet['logp_take'] must be finite (N,)")
        act = _np.asarray(diet["act"], dtype=float)
        if act.ndim != 2 or act.shape[0] != N:
            raise ValueError(f"continuous diet['act'] must be (N,a_dim), got {act.shape}")
    else:
        if "mu_take" in diet:
            mt = _np.asarray(diet["mu_take"], dtype=float)
            if mt.shape != (N,):
                raise ValueError(f"diet['mu_take'] shape {mt.shape} vs ({N},)")
            if not _np.isfinite(mt).all() or (mt <= 0).any() or (mt > 1 + 1e-6).any():
                raise ValueError("diet['mu_take'] must be in (0,1]")
        if "mu" in diet and diet["mu"] is not None:
            mu = _np.asarray(diet["mu"], dtype=float)
            if mu.shape != (N, nA):
                raise ValueError(f"diet['mu'] shape {mu.shape} vs (N,nA)=({N},{nA})")
            if not _np.isfinite(mu).all() or (mu < 0).any():
                raise ValueError("diet['mu'] must be finite non-negative probs")
            s = mu.sum(axis=1)
            if not ((abs(s - 1.0) < 1e-4) | (s == 0)).all():
                bad = int((~((abs(s - 1.0) < 1e-4) | (s == 0))).sum())
                raise ValueError(f"diet['mu'] rows must sum to 1 ({bad} violators)")
        act = _np.asarray(diet["act"])
        if act.min() < 0 or act.max() >= nA:
            raise ValueError(f"diet['act'] outside [0,{nA})")
    for k in ("act", "rew"):
        a = _np.asarray(diet[k])
        if a.shape[0] != N:
            raise ValueError(f"diet[{k!r}] len {a.shape[0]} vs N={N}")
    rew = _np.asarray(diet["rew"], dtype=float)
    if not _np.isfinite(rew).all():
        raise ValueError("diet['rew'] contains NaN/Inf")
    rew = _np.asarray(diet["rew"], dtype=float)
    if not _np.isfinite(rew).all():
        raise ValueError("diet['rew'] contains NaN/Inf")
    # Optional structural keys: validate length when present.
    for k in ("done", "episode", "t", "obs2"):
        if k in diet and diet[k] is not None:
            a = _np.asarray(diet[k])
            if a.shape[0] != N:
                raise ValueError(f"diet[{k!r}] len {a.shape[0]} vs N={N}")
    n_ep = len(_np.unique(_np.asarray(diet["episode"]))) if diet.get("episode") is not None else N
    return {"N": N, "n_episodes": n_ep, "obs_dim": int(obs.shape[1]), "nA": nA}


class ArgmaxPolicy:
    """Wrap a candidate to expose its SHIPPED policy: action_probs is one-hot
    at argmax(logits). FQE on this evaluand bootstraps the deployable policy's
    value and needs no trajectory support (unlike reweighting), so it is the
    right target for a sharp-policy estimate. Ignores temperature by design.
    """

    def __init__(self, inner):
        self.inner = inner

    def act(self, state, eval=True):
        return self.inner.act(state, eval=eval)

    def action_probs(self, obs, temperature=1.0):
        import numpy as _np

        p = _np.asarray(self.inner.action_probs(obs, temperature=1.0), dtype=_np.float64)
        out = _np.zeros_like(p)
        out[_np.arange(len(p)), p.argmax(1)] = 1.0
        return out


class EnvStub:
    """Minimal environment descriptor from a logged dataset (obs_dim, nA).

    Lets the offline-RL trainers work on data of unknown origin without a live
    environment object — the library only needs the action-space size and
    observation shape to build candidate networks.
    """

    def __init__(self, obs_dim, nA, a_dim=None, continuous=False):
        shape = (int(obs_dim),)
        self.observation_space = type("_Space", (), {"shape": shape})()
        if continuous or a_dim is not None:
            a = int(a_dim if a_dim is not None else nA)
            self.action_space = type(
                "_Space",
                (),
                {
                    "shape": (a,),
                    "n": a,
                    "low": np.full(a, -1.0, dtype=np.float32),
                    "high": np.full(a, 1.0, dtype=np.float32),
                    "continuous": True,
                },
            )()
        else:
            self.action_space = type("_Space", (), {"n": int(nA), "continuous": False})()
