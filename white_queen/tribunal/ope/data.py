"""Agnostic data ingestion: any logged dataset -> canonical schema.

The library must not care where data came from (human team, online RL,
another offline policy) or what format it is in. This module is the single
translation layer:

    source  ->  to_canonical(...)  ->  {obs, act, rew, nA, N, mu_take, ...}

Accepted sources: a dict of numpy arrays, a pyarrow.Table, a Polars DataFrame,
a pandas DataFrame (converted via Arrow, never used natively), a DuckDB
relation, or a path to .parquet/.csv/.db (read through DuckDB/Arrow).

Two structural modes, auto-detected:
- "bandit" (default): each row is an independent decision. We canonicalize by
  giving every row its own one-step episode (episode=arange(N), t=0, done=1).
  The existing sequential panel then collapses exactly to per-decision
  estimators (IPS/DR over the taken action) — no separate math needed.
- "rl": sequential logs. Provide next-context + done (and optionally episode/
  t); trajectories are inferred from episode ids or done boundaries.

Behavior propensity is required to compute any importance ratio. It is taken,
in order of preference, from:
  1. an explicit per-row propensity column (`propensity`/`prob_taken`), or a
     full `mu` matrix;
  2. a supervised behavior model fit on (context -> action) when
     estimate_propensity=True (industry path; sets provenance="estimated").
If neither is available we raise, rather than silently producing unusable
ratios.
"""

from __future__ import annotations


import numpy as np

# User-facing column aliases -> canonical role.
_ACTION_NAMES = ("action", "act", "a")
_REWARD_NAMES = ("reward", "rew", "r", "return")
_PROP_NAMES = ("propensity", "prob_taken", "prob", "mu_take", "behavior_prob")
_NEXT_NAMES = ("next_obs", "next_context", "obs2", "next_state", "s2")
_DONE_NAMES = ("done", "terminal", "episode_done")
_EP_NAMES = ("episode", "episode_id", "trajectory_id", "traj_id", "session_id")
_T_NAMES = ("t", "step", "timestep", "time_step")
_TS_NAMES = ("timestamp", "time", "date", "datetime")
_OBS_NAMES = ("obs", "observation", "state", "context", "features", "x")
_MU_NAMES = ("mu", "behavior_probs", "propensities")
_LOGPROB_NAMES = ("log_prob", "logp", "behavior_logp", "log_prob_take", "logprob", "log_density")

_RESERVED = set(
    _ACTION_NAMES
    + _REWARD_NAMES
    + _PROP_NAMES
    + _NEXT_NAMES
    + _DONE_NAMES
    + _EP_NAMES
    + _T_NAMES
    + _TS_NAMES
    + _MU_NAMES
    + _LOGPROB_NAMES
    + ("greedy_action", "greedy", "eps", "epsilon")
)

# Declared source-schema adapters (schema.py) own the role detection for known
# log layouts (e.g. colony o0../n0..). This module no longer guesses in-line.
from .schema import detect_schema as _detect_schema


def _as_columns(source):
    """Return {column_name: 1D-or-2D numpy array} for any accepted source."""
    # dict of arrays
    if isinstance(source, dict):
        return {k: np.asarray(v) for k, v in source.items()}
    # pyarrow Table
    if hasattr(source, "num_rows") and hasattr(source, "column_names"):
        return {n: np.asarray(source.column(n)) for n in source.column_names}
    # Polars DataFrame
    if hasattr(source, "to_arrow") and hasattr(source, "columns"):
        t = source.to_arrow()
        return {n: np.asarray(t.column(n)) for n in t.column_names}
    # pandas DataFrame (converted via Arrow; pandas never used natively)
    if (
        hasattr(source, "to_dict")
        and hasattr(source, "columns")
        and not hasattr(source, "num_rows")
    ):
        try:
            import pyarrow as pa

            t = pa.Table.from_pandas(source)
            return {n: np.asarray(t.column(n)) for n in t.column_names}
        except Exception:
            return {c: np.asarray(source[c].values) for c in source.columns}
    # DuckDB relation
    if hasattr(source, "fetchall"):
        return _duckdb_to_columns(source)
    # path string
    if isinstance(source, (str, bytes)):
        return _duckdb_to_columns(source)
    raise TypeError(f"unsupported data source type: {type(source).__name__}")


def _duckdb_to_columns(source):
    import duckdb

    if isinstance(source, (str, bytes)):
        p = source if isinstance(source, str) else source.decode()
        low = p.lower()
        if low.endswith(".duckdb") or low.endswith(".ddb"):
            con = duckdb.connect(p, read_only=True)
            try:
                tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
                if not tables:
                    raise ValueError(f"no tables in {p}")
                rel = con.execute(f'SELECT * FROM "{tables[0]}"')
                return {k: np.asarray(v) for k, v in rel.fetchnumpy().items()}
            finally:
                con.close()
        con = duckdb.connect()
        try:
            if low.endswith(".parquet"):
                q = f"SELECT * FROM read_parquet('{p}')"
            elif low.endswith(".csv"):
                q = f"SELECT * FROM read_csv_auto('{p}')"
            elif low.endswith(".json") or low.endswith(".jsonl") or low.endswith(".ndjson"):
                q = f"SELECT * FROM read_json_auto('{p}')"
            else:
                raise ValueError(f"unrecognized file type: {p}")
            rel = con.execute(q)
            return {k: np.asarray(v) for k, v in rel.fetchnumpy().items()}
        finally:
            con.close()
    return {k: np.asarray(v) for k, v in source.fetchnumpy().items()}


def _pick(cols, names, required=True, role=""):
    for n in names:
        if n in cols:
            return n
    if required:
        raise ValueError(f"no column found for {role or names[0]} (looked for {names})")
    return None


def _context_matrix(cols, context_cols=None):
    """Context as (N, d). Either a single 2D 'obs' column, or a set of scalar
    feature columns (context_cols or all non-reserved numeric columns)."""
    obs_name = _pick(cols, _OBS_NAMES, required=False)
    if obs_name is not None:
        arr = np.asarray(cols[obs_name])
        if arr.dtype == object:  # list-valued column
            arr = np.stack([np.asarray(x, dtype=np.float32) for x in arr])
        return arr.astype(np.float32), [obs_name]
    # scalar feature columns
    if context_cols is None:
        context_cols = []
        for c, v in cols.items():
            if c in _RESERVED:
                continue
            if np.asarray(v).dtype.kind in "fiub":
                context_cols.append(c)
    if not context_cols:
        raise ValueError("no context features found (provide 'obs' or context_cols)")
    mat = np.stack([np.asarray(cols[c], dtype=np.float32) for c in context_cols], 1)
    return mat, list(context_cols)


def to_canonical(
    source,
    *,
    columns=None,
    nA=None,
    estimate_propensity=True,
    behavior_cfg=None,
    behavior_seed=0,
    source_name=None,
):
    """Normalize any accepted source into the canonical dataset dict.

    columns: optional overrides mapping role -> column name, e.g.
      {"action": "send_type", "reward": "replied", "propensity": "p"}
      or {"context": ["f1","f2","f3"]}.
    Returns dict with obs, act, rew, mu_take, nA, N, episode, t, obs2, done,
    mode, provenance.
    """
    cols = _as_columns(source)
    if not cols:
        raise ValueError("empty dataset")
    columns = dict(columns or {})
    # resolve roles
    a_name = columns.get("action") or _pick(cols, _ACTION_NAMES, role="action")
    r_name = columns.get("reward") or _pick(cols, _REWARD_NAMES, role="reward")
    p_name = columns.get("propensity") or _pick(cols, _PROP_NAMES, required=False)
    mu_name = columns.get("mu") or _pick(cols, _MU_NAMES, required=False)
    n_name = columns.get("next_context") or _pick(cols, _NEXT_NAMES, required=False)
    d_name = columns.get("done") or _pick(cols, _DONE_NAMES, required=False)
    e_name = columns.get("episode") or _pick(cols, _EP_NAMES, required=False)
    t_name = columns.get("t") or _pick(cols, _T_NAMES, required=False)
    ts_name = columns.get("timestamp") or _pick(cols, _TS_NAMES, required=False)
    ctx_cols = columns.get("context")

    act_raw = np.asarray(cols[a_name])
    continuous = (act_raw.dtype.kind == "f") or (act_raw.ndim == 2 and act_raw.shape[1] > 1)
    if continuous:
        act = act_raw.astype(np.float32)
        if act.ndim == 1:
            act = act.reshape(-1, 1)
    else:
        act = act_raw.astype(np.int64).ravel()
    rew = np.asarray(cols[r_name]).astype(np.float64).ravel()
    N = len(act)
    if len(rew) != N:
        raise ValueError(f"action len {N} != reward len {len(rew)}")

    # Real-world reward censoring: outcomes may be unobserved (NaN) or
    # explicitly masked. Drop those ROWS from every column (aligned), receipt
    # the fraction. A drop here is honest: unobserved rewards carry no signal.
    rm_name = columns.get("reward_mask") or _pick(
        cols, ("reward_observed", "reward_mask", "observed", "converted"), required=False
    )
    if rm_name is not None:
        mask = np.asarray(cols[rm_name]).astype(bool).ravel()
        if mask.shape[0] != N:
            raise ValueError("reward_mask length != N")
    else:
        mask = ~np.isnan(rew)
    dropped = int((~mask).sum())
    if dropped:
        keep = np.flatnonzero(mask)
        new_cols = {}
        for k, v in cols.items():
            a = np.asarray(v)
            if a.shape[0] == N:
                new_cols[k] = a[keep]
            else:
                new_cols[k] = a
        cols = new_cols
        N = N - dropped
    act = (
        np.asarray(cols[a_name]).astype(np.float32)
        if continuous
        else np.asarray(cols[a_name]).astype(np.int64).ravel()
    )
    if continuous and act.ndim == 1:
        act = act.reshape(-1, 1)
    rew = np.asarray(cols[r_name]).astype(np.float32).ravel()

    # Declared schema adapter (schema.py) supplies context/next columns for
    # known layouts (e.g. colony o0../n0..), so behavior metadata and
    # next-state columns are never swept into the context by the fallback.
    _schema = _detect_schema(cols, explicit=ctx_cols)
    if _schema is not None and ctx_cols is None:
        ctx_cols = _schema.get("context")
    obs, ctx_used = _context_matrix(cols, ctx_cols)
    if obs.shape[0] != N:
        raise ValueError(f"context len {obs.shape[0]} != N={N}")
    if continuous:
        nA = int(act.shape[1])
    else:
        if nA is None:
            nA = int(act.max()) + 1 if N else 1
        nA = int(nA)

    # structural mode
    if d_name is not None and (n_name is not None or obs is not None):
        mode = "rl"
    else:
        mode = "bandit"

    if mode == "rl":
        if n_name is not None:
            obs2 = np.asarray(cols[n_name]).astype(np.float32)
        elif _schema is not None and _schema.get("next_context"):
            _ngrp = _schema["next_context"]
            obs2 = np.stack([np.asarray(cols[c], dtype=np.float32) for c in _ngrp], 1)
        else:
            obs2 = obs.copy()
        done = (
            np.asarray(cols[d_name]).astype(np.float32)
            if d_name is not None
            else np.zeros(N, dtype=np.float32)
        )
        if e_name is not None:
            episode = np.asarray(cols[e_name]).astype(np.int64).ravel()
        else:
            episode = np.cumsum(done.astype(np.int64))
            episode = np.concatenate([[0], episode[:-1]]) if N else episode
        if t_name is not None:
            t = np.asarray(cols[t_name]).astype(np.int64).ravel()
        else:
            # step within episode via boundaries
            chg = np.flatnonzero(np.diff(episode) != 0) + 1
            starts = np.concatenate([[0], chg])
            seg = np.cumsum(np.concatenate([[0], np.diff(episode) != 0]))
            t = (np.arange(N) - starts[seg]).astype(np.int64)
    else:
        # bandit: each row an independent one-step episode
        obs2 = obs.copy()
        done = np.ones(N, dtype=np.float32)
        episode = np.arange(N, dtype=np.int64)
        t = np.zeros(N, dtype=np.int64)

    canon = {
        "obs": obs.astype(np.float32),
        "act": act,
        "rew": rew,
        "obs2": np.asarray(obs2, dtype=np.float32),
        "done": done,
        "episode": episode,
        "t": t,
        "nA": nA,
        "N": N,
        "mode": mode,
        "continuous": bool(continuous),
    }
    provenance = "provided"
    if continuous:
        # behavior DENSITY at the taken action (log p(a|s)).
        lp_name = columns.get("log_prob") or _pick(cols, _LOGPROB_NAMES, required=False)
        if lp_name is not None:
            logp_take = np.asarray(cols[lp_name]).astype(np.float64).ravel()
        else:
            if not estimate_propensity:
                raise ValueError("no behavior log_prob provided and estimate_propensity=False")
            from .behavior import estimate_behavior_continuous

            logp_take, _info = estimate_behavior_continuous(obs, act, behavior_cfg, behavior_seed)
            provenance = "estimated"
        canon["logp_take"] = logp_take.astype(np.float32)
    else:
        if p_name is not None:
            mu_take = np.asarray(cols[p_name]).astype(np.float64).ravel()
            if mu_take.shape[0] != N:
                raise ValueError("propensity length != N")
            mu = None
        elif mu_name is not None:
            mu = np.asarray(cols[mu_name]).astype(np.float64)
            if mu.ndim == 1:
                mu = mu.reshape(N, -1)
            if mu.shape != (N, nA):
                raise ValueError(f"mu shape {mu.shape} vs (N,nA)=({N},{nA})")
            mu_take = mu[np.arange(N), act]
        else:
            if not estimate_propensity:
                raise ValueError("no behavior propensity provided and estimate_propensity=False")
            mu_take, mu = _estimate_propensity(obs, act, nA, behavior_cfg, behavior_seed)
            provenance = "estimated"
        canon["mu_take"] = mu_take.astype(np.float32)
        if mu is not None:
            canon["mu"] = mu.astype(np.float32)
    if ts_name is not None:
        canon["timestamp"] = np.asarray(cols[ts_name])
    canon["provenance"] = {
        "source": source_name or "memory",
        "mode": mode,
        "continuous": bool(continuous),
        "propensity": provenance,
        "context_columns": ctx_used,
        "dropped_unobserved": dropped,
    }
    return canon


def _estimate_propensity(obs, act, nA, behavior_cfg, seed):
    """Supervised behavior model over (context -> action). Returns
    (mu_take (N,), mu (N,nA)). Reuses the governed estimator in behavior.py."""
    from .behavior import estimate_behavior

    stub = {
        "obs": obs,
        "obs2": obs,
        "act": act,
        "rew": np.ones(len(act), dtype=np.float32),
        "done": np.zeros(len(act), dtype=np.float32),
        "mu": np.full((len(act), nA), 1.0 / nA, dtype=np.float32),
        "episode": np.arange(len(act)),
        "t": np.zeros(len(act)),
        "nA": nA,
        "N": len(act),
    }
    probs, _info = estimate_behavior(stub, behavior_cfg, seed)
    mu_take = probs[np.arange(len(act)), act]
    return mu_take, probs
