"""CFM configuration, constants, shared helpers, and autotune wiring."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from looking_glass import autotune as AT
except Exception:  # run as a standalone script
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("cfm_autotune", Path(__file__).with_name("autotune.py"))
    AT = _ilu.module_from_spec(_spec)
    import sys as _sys

    _sys.modules["cfm_autotune"] = AT
    _spec.loader.exec_module(AT)

STREAM_TABLE = "customer_events"
EMBED_DIM = 64
LN2 = math.log(2.0)
# Registry defaults for event vocabulary: the ORDER event name and the company
# action (send) names. Sources with different vocabularies override via config
# (`order_event`, `company_actions`) / `--set`, never by editing call sites.
ORDER_EVENT = "order_placed"
# Successor features: predict the DISCOUNTED future at a continuously-sampled
# discount gamma. No human-chosen horizons — the model learns all timescales.
TIME_UNIT_SECONDS = 86400.0  # a day (unit scaling only, not a horizon)
SF_PHI = 4  # discounted [value, count, is_order, order*value]
GAMMA_MAX = 0.999


def apply_set_overrides(cfg, receipt: dict | None = None) -> dict:
    """Apply explicit ``--set key=value`` overrides AFTER data-derivation.

    Derived values must never silently overwrite an explicit user setting
    (recorded-then-clobbered). Unknown fields fail safe. Returns what was set.
    """
    import ast

    applied = {}
    for raw in getattr(cfg, "set_overrides", None) or {}:
        k, _, v = str(raw).partition("=")
        k = k.strip()
        if not hasattr(cfg, k) or k in ("set_overrides", "tag"):
            raise SystemExit(f"--set: unknown or reserved config field: {k!r}")
        try:
            val = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            val = v.strip()
        setattr(cfg, k, val)
        applied[k] = val
        if receipt is not None:
            receipt.setdefault("overrides", {})[k] = val
    if applied:
        print(f"[set] explicit overrides applied: {applied}", flush=True)
    return applied


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class CFMConfig:
    db: str = "data/arrow/customer_event_stream.feather"
    table: str = STREAM_TABLE
    out_dir: str = "artifacts/cfm"
    # v2.2.0: portfolio-rigorous training — held-out COMBINED objective selects
    # the kept state (was: next-CE alone), two new self-supervised objectives
    # (`query` = read-at-time, `agg` = exact multi-horizon window targets),
    # `sf` phi is event-agnostic by default, and every train emits a portfolio
    # receipt (per-objective structure-skill + geometry, DEC-008/DEC-009).
    version: str = "v2.2.0"  # encoder code version
    revision: int = 1  # data/score revision (r)
    sample_customers: int | None = 500  # working base: first N customers (populations live here)
    split_a_frac: float = 0.7
    # Populations (assign_split): disjoint monthly pools A and B.
    # Samples: drawn FROM each population for compute efficiency — as small as
    # possible, large enough for signal (sizes come from the ladders). None = use
    # the whole population (previous behavior).
    sample_a_customers: int | None = None  # encoder training sample from population A
    sample_b_customers: int | None = None  # plugin-training sample from population B
    # Warm-start / continual: "auto" continues from the most recent COMPATIBLE
    # checkpoint with as_of <= this run's as_of (never future-trained data),
    # "none" trains from scratch, or name a tag explicitly. Same objective as
    # scratch (all data <= as_of), so the governor still decides convergence —
    # warm init just gets there in fewer steps (no quality sacrifice by design).
    warm_start: str = "auto"
    warm_from: str | None = None  # recorded: which checkpoint we continued from
    split_seed: int = 7
    # Point-in-time training: when set (ISO date/datetime), the stream is cut to
    # events <= as_of before keys/split/anchors/training. Production monthly runs
    # pass the 1st of the month. None = full stream (backward compatible).
    as_of: str | None = None
    seq_len: int = 128
    n_anchors: int = 6  # exact number of sample-B anchor days per customer
    # Company actions are EXOGENOUS (interventions/treatments), not customer
    # behavior: they are covariates and are never predicted as tokens.
    company_actions: tuple = ("email_send", "sms_send", "push_send")
    # The ORDER event name (registry default; override per source via config /
    # --set). Objectives, evaluation probes and battery raw-features read THIS,
    # never a literal.
    order_event: str = ORDER_EVENT
    # Explicit CLI overrides (`--set key=value`, repeatable), applied AFTER
    # derivation and recorded in the registry receipt.
    set_overrides: list = field(default_factory=list)
    dim: int = EMBED_DIM
    n_experts: int = 1  # K=1: M1 multi-timescale gave no gain (speed)
    epochs: int = 3
    batch: int = 64
    lr: float = 3e-3
    gamma_contrast: float = 0.5
    gamma_mask: float = 0.5
    gamma_redundancy: float = 0.1
    mask_frac: float = 0.15
    contrast_tau: float = 0.1
    state_half_life_days: float = 30.0
    # Multi-objective control. Adaptive (uncertainty) weighting learns each
    # task's weight, so we can enable many objectives without hand-tuning and
    # with less gradient interference.
    use_uncertainty_weighting: bool = True
    # Successor-feature phi shape: "event_types" = agnostic (one discounted
    # component per event type + value — the operator's ruling; no objective
    # may name a purchase event); "purchase" = legacy 4-dim phi.
    sf_mode: str = "event_types"
    # Exact window targets (count/value over (t, t+h]) at horizons DERIVED
    # from within-customer gap quantiles (filled by resolve_cfm; `--set` can
    # override and it wins over derivation, recorded as an override).
    agg_horizons_days: list = field(default_factory=list)
    objectives: tuple = (
        "next",
        "entity",
        "dt",
        "value",
        "mask",
        "contrast",
        "redundancy",
        "occur",
        "order",
        "jepa",
        "sf",
        "query",  # read-at-time: grade the FADED state (the serving path)
        "agg",  # long-horizon integration: exact counts/value per window
    )
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def tag(self) -> str:
        return f"{self.version}r{self.revision}"


def sample_a(n_customers: int, n_anchors: int, **kw):
    """The ONLY knobs for the encoder's training sample. Everything else
    (seq_len, dim, batch, budget) is hardware-bounded and predictable, so
    runtime scales ~linearly with n_customers (anchors affect embedding gen)."""
    return CFMConfig(sample_customers=n_customers, n_anchors=n_anchors, **kw)


def _h(key: str, seed: int) -> int:
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16)


def monthly_split_seed(base_seed: int, as_of: str) -> int:
    """Monthly re-randomization of the sample-A/B membership.

    The split seed is derived from (base_seed, calendar month of as_of), so it
    rotates every month and is fully reproducible: same month -> same split,
    next month -> a new disjoint A/B assignment. Anchors follow the same seed
    (`_random_anchor_epochs`), so both member lists and anchor dates refresh
    on each monthly encoder rebuild.
    """
    d = datetime.fromisoformat(str(as_of))
    return int(hashlib.md5(f"{base_seed}:{d.year:04d}-{d.month:02d}".encode()).hexdigest()[:12], 16)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def as_of_epoch(s) -> float:
    """Day-boundary helper for as_of DATES: naive ISO dates are UTC midnight —
    the same interpretation as the point-in-time stream cut (_cut_as_of), so
    the encoder cut, the daily job window, and inference lookups share ONE
    boundary (no local-timezone skew). Aware datetimes pass through."""
    from datetime import datetime, timezone

    d = datetime.fromisoformat(str(s))
    if d.tzinfo is None:
        return d.replace(tzinfo=timezone.utc).timestamp()
    return d.timestamp()


def _to_epoch(s) -> float:
    try:
        return datetime.fromisoformat(str(s)).timestamp()
    except Exception:
        return 0.0


def _seed_everything(seed: int) -> None:
    """Reproducibility: seed all RNGs and force deterministic kernels so the
    same version gives the same battery number (doctrine: identity = behavior)."""
    import random

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _expert_biases(K, gap_days):
    """Timescale decay scales are LEARNED free parameters (delta_bias), not
    derived from human periods. Initialised equal; the loss discovers scales."""
    return [0.0] * K
