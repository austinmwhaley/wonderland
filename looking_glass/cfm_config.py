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
# Successor features: predict the DISCOUNTED future at a continuously-sampled
# discount gamma. No human-chosen horizons — the model learns all timescales.
TIME_UNIT_SECONDS = 86400.0  # a day (unit scaling only, not a horizon)
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
        cur = getattr(cfg, k)
        s = v.strip()
        # `--set flag=true/false` arrives as text; ast uses capitalised
        # True/False, so coerce against the FIELD's own type. Silently keeping
        # the string made bool fields always-truthy ("false" is truthy) — every
        # boolean override was a no-op (DEC-047).
        if isinstance(cur, bool):
            low = s.lower()
            if low in ("true", "1", "yes", "on"):
                val = True
            elif low in ("false", "0", "no", "off"):
                val = False
            else:
                raise SystemExit(f"--set {k}: expected a boolean, got {s!r}")
        else:
            try:
                val = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                if isinstance(cur, int):
                    val = int(float(s))
                elif isinstance(cur, float):
                    val = float(s)
                else:
                    val = s
        if cur is not None and not isinstance(val, type(cur)):
            raise SystemExit(
                f"--set {k}: {val!r} ({type(val).__name__}) does not match field"
                f" type {type(cur).__name__}"
            )
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
    # v2.3.0: Kendall uncertainty weighting replaced by DWA (rate-of-change
    # weights) as the default balancer — the production run measured s-runaway
    # (redundancy s ~ -9, negative combined contributions) and geometry
    # collapse; DWA is scale-free and works on non-bounded losses (DEC-014).
    # v2.4.0: variance-floor objective added (VICReg-style hinge on per-dim
    # std of the projected state) — twice-measured geometry collapse
    # (eff-rank 0.23-0.26 x null) showed decorrelation alone never treats
    # scale collapse; the floor is the direct lever (DEC-015).
    # v2.5.0: `rank` objective — the graded metric (participation ratio of the
    # centered batch covariance) becomes a training pressure; governor selects
    # on the STATIONARY equal-weight held-out sum (DWA weights drive training
    # only — the selection metric must not change definition between evals).
    # v2.6.0: scale-free DWA (each task's loss divided by its own EMA before
    # weighting — dimensionless contributions; sf's discounted-sum scale
    # measured at 94% of gradient mass while failing to learn) + sf yardstick
    # fixed via target-variance R² (DEC-018) -> sf re-enters the gate.
    # v2.7.0: cross-batch memory bank + closed-loop geometry governor
    # (DEC-019): redundancy/variance penalties now ALSO computed over a FIFO
    # bank of recent states (population-global; the batch-only versions
    # couldn't see population collapse — measured ratio 0.213 vs bar 0.30),
    # and the bank geometry multiplier is driven in closed loop by the bank's
    # own eff-rank (ramp below 0.32, back off above — replaces the manual
    # geometry_boost sweep).
    # v2.8.0: barrier geometry (log-det on the bank covariance — infinite
    # wall at collapse, replaces the soft tau-hinge as the primary) +
    # grouped PCGrad (geometry-family vs predictive-family conflict
    # resolution; 2 grouped backwards, not 13 pairwise — DEC-020).
    # v2.8.1: bank FIFO fixed to ROWS (was 8 chunks = 512 rows on a 17k
    # population); rank_lambda_max effectively unclamped (1e6) — the closed
    # loop applies real pressure (measured: lambda pinned at 50 the whole
    # run with rank 0.10 vs target 0.32). (DEC-021)
    # v2.9.0: whitened donor readout (DEC-022). Conclusive evidence (v2.8.1:
    # unclamped lambda 1e6 + barrier + full bank, predictive skills at
    # all-time bests, eff-rank STILL 2.59): collapse is the recurrence's
    # solution structure — training-side pressure cannot fix it. The donor
    # boundary whitens the CONSUMED representation (z = (h-mu) Sigma^-1/2,
    # frozen from held-out states): linear heads span the same class, the
    # graded geometry becomes full-rank by construction, zero training risk.
    # v3.0.0: dual-velocity encoder (DEC-025) — n_experts=2 with spread
    # delta_bias init (fast/slow timescales), ortho-loss between expert state
    # components, rolling EMA whitening, trajectory graded on the slow state.
    version: str = "v6.3.0"  # encoder code version
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
    # Exogenous events (interventions/treatments, not customer behaviour) are
    # declared by the DATA/schema at runtime, never hardcoded here. Default empty
    # (treat every event as customer behaviour — the safe, domain-free fallback);
    # a source declares its exogenous `source_table`s via `--set
    # exogenous_events=...`, resolved at runtime and recorded in the receipt.
    exogenous_events: tuple = ()
    # Explicit CLI overrides (`--set key=value`, repeatable), applied AFTER
    # derivation and recorded in the registry receipt.
    set_overrides: list = field(default_factory=list)
    final_task_weights: dict = field(default_factory=dict)  # recorded post-train
    final_loss_scales: dict = field(default_factory=dict)  # the EMA unit system
    final_bank_stats: dict = field(default_factory=dict)  # bank rank/lambda trail
    dim: int = EMBED_DIM
    # DEC-025: dual-velocity (K=2) — fast expert tracks token transitions,
    # slow expert accumulates behavioral trends; the donor concatenates both.
    # The v2.9.0 half-life experiment proved single-state trajectory is
    # structural (cos -0.30 at any half-life); the fix is architectural.
    n_experts: int = 2
    # v3.1.0 (DEC-027): low-pass intent filter on the slow expert — smooths the
    # content stream so the slow state's velocity no longer zigzags. Default on;
    # opt out with --set slow_intent_filter=false.
    slow_intent_filter: bool = False
    # v4.0 (DEC-029): macro shifts.
    # (1) unified wide selective SSM with a per-channel timescale spectrum,
    #     replacing the hand-split expert bank + intent filter.
    unified_ssm: bool = True
    # (2) differentiable Newton-Schulz ZCA at the donor boundary (isotropy by
    #     construction; gradients shape the consumed representation).
    zca: bool = True
    # (3) adaptive information bottleneck: a compression term with a
    #     self-tuned Lagrange multiplier (dual ascent on the batch rank).
    aib: bool = True
    aib_lr: float = 0.05
    aib_target_rank: float = 0.5  # target batch PR/dim the controller holds
    aib_beta_max: float = 5.0  # cap on the compression multiplier
    # v4.2 native self-governing mechanics (DEC-031):
    tau_mp: float = 0.70  # Marchenko-Pastur PR/dim floor the iso barrier targets
    mp_floor: float = 0.25  # fallback floor when no per-stream capacity is known
    traj_tau_mult: float = 1.0  # slow-band cutoff = tau_mult x median learned half-life
    # (separation of concerns: keeps the slow band free to be continuous)
    geom_coverage: float = 0.6  # per-stream: held-out PR must retain this
    # fraction of the stream's achievable whitened PR (from the receipt)
    iso_gamma: float = 8.0  # barrier sigmoid steepness (self-throttling)
    ortho_weight: float = 0.1  # cross-subspace ortho-loss weight
    epochs: int = 3  # legacy (governor ignores); see max_steps
    max_steps: int = 0  # 0 = derived budget; >0 caps the governor (FAST iteration)
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
    # Task-balance mode (DEC-014): "dwa" = Dynamic Weight Average (scale-free,
    # learned from loss improvement rates — no hyperparameter tuning, works on
    # non-bounded geometric losses); "uncertainty" = Kendall log-var weights
    # (kept for likelihood-pure stacks; measured to s-runaway on geometric
    # losses); "equal" = plain sum. Weights are dynamic either way.
    weight_mode: str = "equal"  # FR v2.0: no dynamic re-weighting
    dwa_temp: float = 2.0  # DWA temperature (paper default; the ONLY knob)
    dwa_scale_free: bool = True  # divide each loss by its own EMA (DEC-018):
    # dimensionless contributions; without it sf consumed 94% of grad mass
    # Geometry guard (DEC-019): the closed-loop population geometry
    bank_size: int = 8192  # FIFO of recent projected states (0 = off)
    rank_target: float = 0.35  # bank eff-rank target (v4.3: push final readout to capacity)
    rank_alpha: float = 0.25  # governor gain (PID-like, EMA-damped)
    rank_lambda_max: float = 1e6  # effectively unclamped (EMA alpha bounds rate)
    tau_eig: float = 0.05  # singular-value floor on the bank covariance
    #    (soft floor; the PRIMARY rank guard is now the log-det barrier)
    pcgrad: bool = True  # grouped PCGrad (DEC-020): geometry vs predictive
    donor_whiten: bool = True  # whitened readout at the donor boundary (legacy)
    isometric_boundary: bool = True  # v6 Stage1: per-sample orthogonal readout (kappa=1)
    bilinear_recurrence: bool = False  # v7: multiplicative state-input term (rank-preserving fix)
    # v6.1 (DEC-047): variance-preserving recurrence. A recurrent readout has no
    # intrinsic scale; over long horizons the drives accumulate and the readout
    # magnitude drifts (measured std ~50, max ~900), which made every linear head
    # ill-conditioned and untrainable (head 4x worse than a probe on its own
    # state). `readout_norm` RMS-normalizes the per-step readout (heads always see
    # unit-scale features); `input_norm` RMS-normalizes the token stream so the
    # accumulated drive cannot grow with horizon. Both are per-sample (no batch
    # statistics -> cannot fake rank). Structural invariants, not losses.
    readout_norm: bool = True  # RMSNorm on the SSM readout before heads
    input_norm: bool = True  # RMSNorm on the token stream (bounded drive)
    # v6.2 (DEC-048): direct token->readout skip (the SSM `D` term). The recurrent
    # state is a low-pass aggregate; the per-event transition (cart->purchase)
    # lives in the CURRENT token. A skip restores it to the readout so a linear
    # head can express the transition table (measured: state->next 0.37 vs
    # 1-gram 0.25). Identity-init, learnable, per-sample.
    readout_skip: bool = True
    # DEC-028: condition cap for the boundary whitening — eigenvalues of the
    # state covariance are floored at this fraction of the largest, bounding
    # kappa(Sigma^{-1/2}) <= 1/sqrt(whiten_cond_floor). Prevents near-null
    # amplification that made the geometry/OOT gates non-reproducible.
    whiten_cond_floor: float = 1e-2
    # Frontier sampler (DEC-017): multiplier on the geometry-family losses
    # (variance, rank, redundancy). 1.0 = the balanced point; >1 trades a bit
    # of predictive skill for state headroom — the explicit Pareto coordinate,
    # recorded in the registry so every point on the curve is a receipt.
    geometry_boost: float = 1.0
    use_uncertainty_weighting: bool = True  # legacy flag (uncertainty mode)
    # Successor-feature phi: agnostic only — one discounted component per event
    # type plus value. No objective may name a business event (DEC-009). The
    # legacy purchase-named phi was deleted.
    sf_mode: str = "event_types"
    # Exact window targets (count/value over (t, t+h]) at horizons DERIVED
    # from within-customer gap quantiles (filled by resolve_cfm; `--set` can
    # override and it wins over derivation, recorded as an override).
    agg_horizons_days: list = field(default_factory=list)
    # FR v2.0 (DEC-044): the SOFT loss-invariants (redundancy/decorr/variance/
    # rank/spectrum/volume/iso/trajectory/commutation/ortho) are REMOVED from the
    # objective — invariants must be structural (the bilinear recurrence), not
    # soft penalties (they stiffened the landscape: L~2e6 -> lr collapse). Only
    # predictive self-supervised objectives remain; rank is enforced in the update.
    objectives: tuple = (
        "next",
        "entity",
        "dt",
        "value",
        "mask",
        "contrast",
        "occur",
        "order",
        "jepa",
        "sf",
        "query",
        "agg",
    )
    # kept as the (deprecated) soft-invariant family for reference / opt-in
    soft_invariants: tuple = (
        "redundancy",
        "decorr",
        "variance",
        "rank",
        "spectrum",
        "volume",
        "iso",
        "trajectory",
        "commutation",
        "ortho",
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
    """Dual-velocity init (DEC-025): spread expert delta_biases log-uniformly
    so the K experts START at different timescales (fast first, slow last)
    instead of from the same point (which makes them redundant). The learned
    W_delta adapts; the init breaks the symmetry. gap_days = median gap
    (the typical event cadence, used to center the spread)."""
    if K <= 1:
        return [0.0]
    # log-uniform spread from -1.5 (slow) to +1.5 (fast) around 0
    lo, hi = -1.5, 1.5
    return [lo + (hi - lo) * i / max(K - 1, 1) for i in range(K)]
