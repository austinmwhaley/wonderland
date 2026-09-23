"""Estimator checkpoint cache: never pay twice for the same fit.

Re-judging after a gate/judge change is our most common operation, and it
needs zero refits — but every panel run retrained FQE x6, FVE, MIS, DICE and
dynamics from scratch (~1h of the 75-min tribunal). This store persists
fitted weights keyed by exactly what determines them:

  key = sha1(code_version, diet content hash, candidate weights hash,
             resolved cfg, temperature, tag)

Determinism (seeded training) makes cache hits bit-exact, not approximate.
Miss/corruption/device-mismatch -> train normally (fail-open, never crash
a tribunal). Disabled unless a cache dir is passed explicitly, so unit tests
and casual calls never touch disk.
"""

import hashlib
import json
import os

OPE_CACHE_VERSION = "v4"  # BUMP when any estimator's math changes.
# v2: dynamics gained a termination head (in_dim+2).
# v3: FQE target net switched from once-per-eval hard sync to per-step Polyak
#     soft updates (propagation was ~40% of truth).
# v4: FQE runs budget-mode (no self-referential best-restore); sharp witness
#     is now FQE of the ARGMAX policy (not a low-temperature proxy).


def diet_hash(diet):
    """Content hash of what training actually reads. Milliseconds on 150k
    rows (single sha1 pass); exact, not a fingerprint. Includes mu because
    the industry track replaces exact propensities with estimated ones on
    otherwise-identical rows — without mu in the hash, a logged-mu fit would
    wrongly cache-hit for an estimated-mu judgement (and vice versa)."""
    import numpy as _np
    h = hashlib.sha1()
    for k in ("obs", "act", "rew", "obs2", "done", "episode", "mu"):
        h.update(_np.ascontiguousarray(diet[k]).tobytes())
    h.update(str(int(diet["nA"])).encode())
    h.update(str(diet.get("_mu_source", "logged")).encode())
    return h.hexdigest()[:16]


def file_hash(path):
    """sha1 of a weights file (candidate .pts are KBs — microseconds)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _stable(cfg):
    if cfg is None:
        return ""
    def _norm(x):
        if isinstance(x, dict):
            return {k: _norm(x[k]) for k in sorted(x)}
        if isinstance(x, float):
            return round(x, 10)
        if isinstance(x, (list, tuple)):
            return [_norm(v) for v in x]
        return x
    return json.dumps(_norm(dict(cfg)), sort_keys=True, default=str)


def make_key(tag, diet_h, cand_id, cfg, temperature=None, extra=""):
    raw = "|".join([OPE_CACHE_VERSION, str(tag), str(diet_h), str(cand_id),
                    _stable(cfg), str(temperature), str(extra)])
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


def path_for(cache_dir, key):
    return os.path.join(cache_dir, key + ".pt")


def save(cache_dir, key, payload):
    """payload: {name: tensor/state_dict} + plain receipts. Returns path or
    None on any failure (fail-open)."""
    if not cache_dir:
        return None
    try:
        import torch
        os.makedirs(cache_dir, exist_ok=True)
        p = path_for(cache_dir, key)
        torch.save(payload, p)
        return p
    except Exception:
        return None


def load(cache_dir, key, map_location="cpu"):
    """Returns payload dict or None (miss, corrupt, or incompatible)."""
    if not cache_dir:
        return None
    try:
        import os as _os
        import torch
        p = path_for(cache_dir, key)
        if not _os.path.exists(p):
            return None
        return torch.load(p, map_location=map_location, weights_only=True)
    except Exception:
        return None
