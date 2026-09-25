"""CFM configuration, constants, shared helpers, and autotune wiring."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
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
# Successor features: predict the DISCOUNTED future at a continuously-sampled
# discount gamma. No human-chosen horizons — the model learns all timescales.
TIME_UNIT_SECONDS = 86400.0  # a day (unit scaling only, not a horizon)
SF_PHI = 4  # discounted [value, count, is_order, order*value]
GAMMA_MAX = 0.999


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class CFMConfig:
    db: str = "data/arrow/customer_event_stream.feather"
    table: str = STREAM_TABLE
    out_dir: str = "artifacts/cfm"
    version: str = "v2.0.0"  # encoder code version
    revision: int = 1  # data/score revision (r)
    sample_customers: int | None = 500
    split_a_frac: float = 0.7
    split_seed: int = 7
    seq_len: int = 128
    n_anchors: int = 6  # exact number of sample-B anchor days per customer
    # Company actions are EXOGENOUS (interventions/treatments), not customer
    # behavior: they are covariates and are never predicted as tokens.
    company_actions: tuple = ("email_send", "sms_send", "push_send")
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


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


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
