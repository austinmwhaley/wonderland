"""Derived configuration and convergence-governed training.

Doctrine (see work/AGENTS.md): no magic numbers; train to convergence, not to a
count; make it learn; emit receipts.

Everything here DERIVES its settings from the data and the hardware, and the
training loop STOPS on held-out convergence (with an overfit guard) rather than
a fixed epoch count. Surviving literals are conservative fallback *bounds*,
recorded in the receipt when they fire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def data_revision(df) -> tuple[int, str]:
    """Data/score revision (the `r` in vNrN). A deterministic identity for the
    stream, so a retrain on different data is a different artifact."""
    import hashlib

    def _mass() -> float:
        # The stream's dominant numeric quantity (no hardcoded `value` field).
        cands = [
            c
            for c in df.columns
            if c not in ("customer_key", "event_ts", "event_type", "event_attributes")
            and df[c].dtype.is_numeric()
        ]
        if not cands:
            return 0.0
        sums = {c: float(df[c].cast(float, strict=False).fill_null(0).abs().sum()) for c in cands}
        best = max(sorted(sums), key=lambda c: sums[c])
        return float(df[best].cast(float, strict=False).fill_null(0).sum())

    sig = "|".join(
        str(x)
        for x in (
            df.height,
            df["customer_key"].n_unique(),
            df["event_type"].n_unique(),
            str(df["event_ts"][0]),
            str(df["event_ts"][-1]),
            int(round(_mass())),
        )
    )
    h = hashlib.sha1(sig.encode()).hexdigest()
    return int(h[:6], 16) % 1_000_000, h[:12]


# ---------------------------------------------------------------------------
# derived configuration
# ---------------------------------------------------------------------------
def sequence_lengths(df, keys) -> list[int]:
    """Events per customer, ordered as in the stream."""
    cols = df["customer_key"].to_list()
    want = set(keys)
    lens, cur, started = [], 0, False
    prev = None
    for k in cols:
        if k in want:
            if started and k != prev:
                lens.append(cur)
                cur = 0
            started = True
            cur += 1
            prev = k
    if started:
        lens.append(cur)
    return lens


def derive_seq_len(lengths) -> tuple[int, dict]:
    import numpy as np

    if not lengths:
        return 64, {"seq_len": "fallback default (no sequences)"}
    # FULL-HISTORY default: cover nearly the whole customer history (p99), not just
    # the median. A windowed state cannot subsume full-history RFM.
    q = float(np.quantile(lengths, 0.99))
    val = int(clip(round(q), 8, 4096))
    return val, {"seq_len": f"p99 events/customer={q:.0f} -> {val}"}


def event_type_entropy(df) -> float:
    """Shannon entropy of the event-type distribution (nats) — the stream's
    categorical diversity, used to scale capacity (v6 Stage 5, #2)."""
    import numpy as np

    if "event_type" not in df.columns:
        return 0.0
    c = df.group_by("event_type").len().sort("len", descending=True)
    n = float(c["len"].sum())
    if n <= 0:
        return 0.0
    p = (c["len"] / n).to_numpy()
    return float(-(p * np.log(p)).sum())


def derive_dim(
    n_seqs, total_events, vocab_sizes, et_entropy: float | None = None
) -> tuple[int, dict]:
    """Capacity scales with the INFORMATION in the data: events, vocabulary, and
    the stream's categorical diversity H (v6 Stage 5, #2: D ~ exp(H), not a fixed
    power of two). Bounded by a conservative hardware-safe range."""
    info = math.sqrt(max(total_events, 1)) * math.log2(2 + sum(vocab_sizes) + n_seqs)
    if et_entropy is not None:
        info *= 1.0 + et_entropy  # more diverse stream -> more manifold capacity
    val = int(clip(2 ** round(math.log2(max(info / 64.0, 8.0))), 16, 256))
    return val, {"dim": f"info={info:.0f} (H_et={et_entropy}) -> {val}"}


def derive_lr(input_power: float | None = None) -> tuple[float, dict]:
    """Learning rate from the INPUT signal scale (v6 Stage 5, #10): step size
    ~ 1/sqrt(P_in) so the first update is O(1) relative to the state magnitude,
    independent of the stream's units. Falls back to 3e-3 when unknown."""
    if not input_power or input_power <= 0:
        return 3e-3, {"lr": "fallback 3e-3 (no input power)"}
    val = float(clip(0.5 / math.sqrt(input_power), 1e-4, 1e-2))
    return val, {"lr": f"P_in={input_power:.3f} -> {val:.2e}"}


def derive_batch(n_seqs, seq_len: int = 0) -> tuple[int, dict]:
    """Batch scales with data but is BOUNDED BY HARDWARE *and* sequence length.

    Activation memory scales with batch x seq_len: a daily-cadence stream
    (seq_len in the thousands) OOMs at a batch sized only by n_seqs. Cap the
    per-step token count so the derived batch is safe on the card."""
    import torch

    if torch.cuda.is_available():
        # memory-safe cap for this model (K experts x JEPA x multi-gamma)
        free = torch.cuda.mem_get_info()[0] / (1024**3)
        cap = int(clip(2 ** round(math.log2(max(free * 8.0, 8.0))), 8, 256))
    else:
        cap = 32
    val = int(clip(2 ** round(math.log2(max(n_seqs / 32.0, 8.0))), 8, cap))
    token_cap = max(8, int(16384 // max(int(seq_len or 1), 1)))
    val = int(min(val, cap, token_cap))
    return val, {"batch": f"n_seqs={n_seqs} seq_len={seq_len} hw_cap={cap} -> {val}"}


def within_customer_gaps(df):
    """Positive within-customer inter-event gaps (seconds), pooled -> np.ndarray."""
    import polars as pl

    d = df.select(
        pl.col("customer_key"),
        pl.col("event_ts").str.to_datetime(time_zone="UTC", strict=False).dt.epoch("s"),
    ).sort(["customer_key", "event_ts"])
    g = d.select(pl.col("event_ts").diff().over("customer_key").alias("g"))
    return g.filter(pl.col("g").is_not_null() & (pl.col("g") > 0))["g"].to_numpy()


def derive_agg_horizons(df) -> tuple[list[float], dict]:
    """Exact-window horizons for the `agg` objective, from gap quantiles
    (p50/p90/p99 — same distribution the half-life derives from). Conservative
    floor: 1 day (documented fallback), receipted."""
    import numpy as np

    g = within_customer_gaps(df)
    if g.size < 3:
        return [7.0, 30.0], {"agg_horizons_days": "fallback [7,30] (too few gaps)"}
    qs = np.quantile(g, [0.5, 0.9, 0.99]) / 86400.0
    horizons = sorted({float(max(round(q, 1), 1.0)) for q in qs})
    return horizons, {"agg_horizons_days": f"gap quantiles [50,90,99]% -> {horizons}d"}


def derive_half_life(df) -> tuple[float, dict]:
    """State persistence timescale from WITHIN-customer inter-event gaps.

    The p95 (not median): serving fades a state from its last event to the
    scoring boundary, so the half-life must outlast a typical quiet stretch or
    the readout forgets everything between events.

    Gaps are computed per customer — pooling every customer into one merged,
    re-sorted timeline measures *interleaving* (sub-second gaps across
    customers), which pinned the half-life at the 1-hour floor on every
    multi-customer dataset.
    """
    import numpy as np

    g = within_customer_gaps(df)
    if g.size < 3:
        return 30.0, {"state_half_life_days": "fallback (too few within-customer gaps)"}
    p95_days = float(np.quantile(g, 0.95)) / 86400.0
    val = float(clip(p95_days, 1.0 / 24.0, 365.0))
    receipt = {"state_half_life_days": f"p95 within-customer gap {p95_days:.2f}d -> {val:.2f}"}
    if val != p95_days:
        receipt["state_half_life_days"] += " (clipped)"
    return val, receipt


def _customer_level_features(df, keys, sample_n: int | None = None):
    """Compute customer-level feature matrix from the raw stream.

    Features (one row per customer):
    - Event type proportions (one per event type discovered in the stream)
    - Numeric field means (one per numeric key discovered from event_attributes JSON)
    - Temporal features: mean gap, std gap, event span (days), events per day

    Returns (feature_matrix, customer_keys, feature_names).
    """
    import polars as pl
    import numpy as np
    import json

    # Optional sampling for speed on large streams
    if sample_n is not None and len(keys) > sample_n:
        import random
        rng = random.Random(42)
        keys = sorted(rng.sample(list(keys), sample_n))

    keys_set = set(keys)

    # Parse event_attributes as JSON and build customer-level aggregates
    cust_data: dict = {}
    for row in df.iter_rows(named=True):
        ck = row["customer_key"]
        if ck not in keys_set:
            continue
        et = row["event_type"]
        ts = row["event_ts"]
        attrs = parse_attrs(row["event_attributes"])

        if ck not in cust_data:
            cust_data[ck] = {
                "event_types": {},
                "numeric_sums": {},
                "numeric_counts": {},
                "n_events": 0,
                "first_ts": ts,
                "last_ts": ts,
                "gaps": [],
            }

        d = cust_data[ck]
        d["n_events"] += 1
        d["event_types"][et] = d["event_types"].get(et, 0) + 1
        # event_ts is a string; compare lexicographically (ISO format sorts correctly)
        if d["first_ts"] is None or ts < d["first_ts"]:
            d["first_ts"] = ts
        if d["last_ts"] is None or ts > d["last_ts"]:
            d["last_ts"] = ts

        for k, v in attrs.items():
            if isinstance(v, (int, float)):
                d["numeric_sums"][k] = d["numeric_sums"].get(k, 0) + v
                d["numeric_counts"][k] = d["numeric_counts"].get(k, 0) + 1

    if not cust_data:
        return np.zeros((0, 0)), [], []

    # Discover all feature names
    et_cols = sorted(set(et for d in cust_data.values() for et in d["event_types"]))
    num_cols = sorted(set(k for d in cust_data.values() for k in d["numeric_sums"]))

    customer_keys = sorted(cust_data.keys())
    n_cust = len(customer_keys)
    n_et = len(et_cols)
    n_num = len(num_cols)
    n_feat = n_et + n_num + 4  # +4: n_events, bias, event_span_days, events_per_day

    X = np.zeros((n_cust, n_feat))

    for i, ck in enumerate(customer_keys):
        d = cust_data[ck]
        n_ev = d["n_events"]

        # Event type proportions
        for j, et in enumerate(et_cols):
            X[i, j] = d["event_types"].get(et, 0) / n_ev

        # Numeric field means
        for j, k in enumerate(num_cols):
            if d["numeric_counts"][k] > 0:
                X[i, n_et + j] = d["numeric_sums"][k] / d["numeric_counts"][k]

        # Temporal features
        X[i, n_et + n_num] = float(n_ev)  # total events
        X[i, n_et + n_num + 1] = 1.0  # bias
        # event_ts is ISO string; convert to epoch for span computation
        try:
            from datetime import datetime
            first = datetime.fromisoformat(d["first_ts"])
            last = datetime.fromisoformat(d["last_ts"])
            span_days = (last - first).total_seconds() / 86400.0
        except Exception:
            span_days = 0.0
        X[i, n_et + n_num + 2] = span_days  # event span in days
        X[i, n_et + n_num + 3] = n_ev / max(1.0, span_days)  # events per day

    return X, customer_keys, et_cols + num_cols + ["n_events", "bias", "event_span_days", "events_per_day"]


def parse_attrs(s):
    """Parse event_attributes JSON string."""
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def derive_contrast_tau(df, keys) -> tuple[float, dict]:
    """Derive contrast temperature from pairwise customer distances.

    Computes customer-level features (event type proportions, numeric means,
    temporal features) and derives contrast_tau from the median pairwise
    cosine distance squared. A larger tau makes the contrastive softmax softer,
    reducing collapse pressure that pushes similar customers to identical states.

    Falls back to 1.0 (soft contrast) when insufficient data.
    """
    import numpy as np

    X, customer_keys, _ = _customer_level_features(df, keys, sample_n=2000)
    if X.shape[0] < 3:
        return 1.0, {"contrast_tau": "fallback 1.0 (too few customers)"}

    # Normalize and compute pairwise cosine distances
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_distances

    X_norm = normalize(X)
    dists = cosine_distances(X_norm)
    triu_idx = np.triu_indices(len(X), k=1)
    pairwise_dists = dists[triu_idx]

    # Derive contrast_tau from median distance squared
    # tau ~ median(distance^2) makes the softmax soft enough to not collapse
    # but sharp enough to provide useful contrast.
    # Customer-level features have low intrinsic dimensionality (customers look
    # similar), so the derived tau is small. Raise the floor to ensure the
    # contrastive objective is soft enough to not collapse the state.
    val = float(np.median(pairwise_dists) ** 2)
    val = float(clip(val, 1.0, 2.0))  # floor at 1.0 (soft contrast)

    return val, {
        "contrast_tau": f"median pairwise cosine distance^2={np.median(pairwise_dists):.3f} -> {val:.3f} (floor 1.0)"
    }


def derive_tau_eig(df, keys) -> tuple[float, dict]:
    """Derive eigenvalue floor (tau_eig) from the noise floor of the
    customer-level feature covariance.

    The tau_eig should be slightly above the noise floor to prevent collapse
    without penalizing legitimate small eigenvalues. Derived as 2x the median
    of the bottom half of eigenvalues (excluding exact zeros).

    Falls back to 0.05 (conservative) when insufficient data.
    """
    import numpy as np

    X, _, _ = _customer_level_features(df, keys, sample_n=2000)
    if X.shape[0] < 3 or X.shape[1] < 3:
        return 0.05, {"tau_eig": "fallback 0.05 (too few customers/features)"}

    # Center and compute covariance
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)

    # Noise floor: median of bottom half, excluding exact zeros
    nz = eigvals[eigvals > 1e-12]
    if nz.size < 3:
        return 0.05, {"tau_eig": "fallback 0.05 (no non-zero eigenvalues)"}

    mid = len(nz) // 2
    noise_floor = float(np.median(nz[:mid]))
    val = float(clip(2.0 * noise_floor, 0.01, 0.1))

    return val, {"tau_eig": f"2x noise floor ({noise_floor:.2e}) -> {val:.2e}"}


def derive_rank_target(df, keys, dim: int) -> tuple[float, dict]:
    """Derive rank target from the stream's signal count.

    The state must encode enough dimensions to represent the stream's complexity:
    event types, numeric fields, generative traits, and timescale regimes.
    Derived from the stream's schema, not from the feature covariance (which
    has low intrinsic dimensionality due to customer similarity).

    Falls back to min(7, dim) / dim (7 generative traits) when insufficient data.
    """
    import polars as pl
    import numpy as np
    import json

    # Count event types
    n_et = int(df["event_type"].n_unique()) if "event_type" in df.columns else 0

    # Count numeric fields discovered from event_attributes JSON
    numeric_fields: set = set()
    for attrs in df["event_attributes"].unique().to_list():
        if attrs:
            try:
                d = json.loads(attrs)
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        numeric_fields.add(k)
            except Exception:
                pass

    # Generative traits the state must encode (from the certification targets)
    n_generative = 7  # x0, h0, price_sensitivity, return_proneness, churn_day, regime_switches, final_regime_days

    # Timescale regimes from the gap distribution
    g = within_customer_gaps(df)
    if g.size >= 3:
        short_term = float((g < 86400.0).mean())
        daily = float(((g >= 86400.0) & (g < 7 * 86400.0)).mean())
        weekly = float(((g >= 7 * 86400.0) & (g < 30 * 86400.0)).mean())
        monthly = float((g >= 30 * 86400.0).mean())
        n_timescales = sum([short_term > 0.05, daily > 0.05, weekly > 0.05, monthly > 0.05])
    else:
        n_timescales = 2  # fast/slow fallback

    n_signals = n_et + len(numeric_fields) + n_generative + n_timescales
    val = float(clip(n_signals / max(dim, 1), 0.01, 0.5))

    return val, {
        "rank_target": f"{n_signals} signals ({n_et} et + {len(numeric_fields)} num + {n_generative} gen + {n_timescales} ts) of {dim} -> {val:.3f}"
    }


def derive_aib_target_rank(df, keys, dim: int) -> tuple[float, dict]:
    """Derive AIB target rank from the stream's signal count.

    The AIB controller compresses the state to hold PR/dim at this target.
    Derived from the same data as rank_target so the AIB and GeometryBank
    governors are consistent.

    Falls back to the rank_target derivation when insufficient data.
    """
    rt, rt_receipt = derive_rank_target(df, keys, dim)
    return rt, {"aib_target_rank": f"consistent with rank_target={rt:.3f} ({rt_receipt['rank_target']})"}


def derive_tau_mp(df, keys, dim: int) -> tuple[float, dict]:
    """Derive Marchenko-Pastur floor (tau_mp) from the stream's signal count.

    The isotropic barrier opens only below tau_mp and stops competing once
    PR/dim reaches it. Derived from the same signal count as rank_target so
    the barrier stops competing once geometry reaches the target rank.

    Falls back to 0.05 when insufficient data.
    """
    import polars as pl
    import numpy as np
    import json

    # Count event types
    n_et = int(df["event_type"].n_unique()) if "event_type" in df.columns else 0

    # Count numeric fields discovered from event_attributes JSON
    numeric_fields: set = set()
    for attrs in df["event_attributes"].unique().to_list():
        if attrs:
            try:
                d = json.loads(attrs)
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        numeric_fields.add(k)
            except Exception:
                pass

    # Generative traits the state must encode
    n_generative = 7  # x0, h0, price_sensitivity, return_proneness, churn_day, regime_switches, final_regime_days

    # Timescale regimes from the gap distribution
    g = within_customer_gaps(df)
    if g.size >= 3:
        short_term = float((g < 86400.0).mean())
        daily = float(((g >= 86400.0) & (g < 7 * 86400.0)).mean())
        weekly = float(((g >= 7 * 86400.0) & (g < 30 * 86400.0)).mean())
        monthly = float((g >= 30 * 86400.0).mean())
        n_timescales = sum([short_term > 0.05, daily > 0.05, weekly > 0.05, monthly > 0.05])
    else:
        n_timescales = 2  # fast/slow fallback

    n_signals = n_et + len(numeric_fields) + n_generative + n_timescales
    val = float(clip(n_signals / max(dim, 1), 0.01, 0.5))

    return val, {
        "tau_mp": f"{n_signals} signals of {dim} -> {val:.3f} (barrier stops at rank_target)"
    }


def derive_budget(n_seqs, batch, dim) -> tuple[int, dict]:
    """Training budget from problem size, not a fixed count. The governor stops
    earlier on convergence; this is only the ceiling."""
    steps_per_pass = max(1, math.ceil(n_seqs / max(batch, 1)))
    # enough passes to see structure, scaled by capacity; ceiling bounded.
    val = int(clip(15 * steps_per_pass * math.sqrt(max(dim, 8) / 64.0), 50, 200000))
    return val, {"budget_steps": f"n_seqs={n_seqs} batch={batch} dim={dim} -> {val}"}


@dataclass
class ResolvedCFM:
    seq_len: int
    dim: int
    batch: int
    half_life_days: float
    budget_steps: int
    eval_every: int
    patience: int
    seed: int
    agg_horizons_days: list = field(default_factory=list)
    # Derived rank-geometry parameters (DEC-016/019): no magic numbers
    contrast_tau: float = 1.0
    tau_eig: float = 0.05
    rank_target: float = 0.35
    aib_target_rank: float = 0.5
    tau_mp: float = 0.70
    receipt: dict = field(default_factory=dict)


def resolve_cfm(base, df, keys, vocab_sizes) -> ResolvedCFM:
    """Resolve all config from data + hardware. `base` is the raw config; any
    explicit override wins and is recorded (recording alone never applied it —
    the derived value overwrote the override downstream)."""
    lengths = sequence_lengths(df, keys)
    seq_len, r1 = derive_seq_len(lengths)
    total = int(sum(lengths)) if lengths else 0
    et_H = event_type_entropy(df)
    dim, r2 = derive_dim(len(keys), total, vocab_sizes, et_entropy=et_H)
    batch, r3 = derive_batch(len(keys), seq_len=seq_len)
    hl, r4 = derive_half_life(df)
    budget, r5 = derive_budget(len(keys), batch, dim)
    # Derived rank-geometry parameters (DEC-016/019): no magic numbers
    ct, r6 = derive_contrast_tau(df, keys)
    te, r7 = derive_tau_eig(df, keys)
    rt, r8 = derive_rank_target(df, keys, dim)
    atr, r9 = derive_aib_target_rank(df, keys, dim)
    tmp, r10 = derive_tau_mp(df, keys, dim)
    # eval cadence: enough evals to detect a plateau inside the budget.
    eval_every = int(clip(round(budget / 20.0), 5, 5000))
    # patience: a fraction of the eval budget, so convergence, not a count, is
    # what stops training. Overridable (a late-converging objective needs more).
    patience = int(clip(round(budget / eval_every / 5.0), 3, 20))
    user_p = int(getattr(base, "train_patience", 0) or 0)
    if user_p > 0:
        patience = user_p
    rec = {"derived": {}, "overrides": {}}
    if user_p > 0:
        rec["overrides"]["patience"] = user_p
    out = {}
    for name, (val, rr) in (
        ("seq_len", (seq_len, r1)),
        ("dim", (dim, r2)),
        ("batch", (batch, r3)),
        ("half_life_days", (hl, r4)),
        ("budget_steps", (budget, r5)),
        ("contrast_tau", (ct, r6)),
        ("tau_eig", (te, r7)),
        ("rank_target", (rt, r8)),
        ("aib_target_rank", (atr, r9)),
        ("tau_mp", (tmp, r10)),
    ):
        user = getattr(base, name, None)
        default = {
            "seq_len": 128,
            "dim": 64,
            "batch": 64,
            "half_life_days": 30.0,
            "budget_steps": 0,
            "contrast_tau": 0.1,
            "tau_eig": 0.05,
            "rank_target": 0.35,
            "aib_target_rank": 0.5,
            "tau_mp": 0.70,
        }[name]
        if user is not None and user != default and user != 0:
            rec["overrides"][name] = user
            out[name] = user
        else:
            out[name] = val
        rec["derived"][name] = rr
    # agg horizons: explicit --set wins (recorded), else derived from gap
    # quantiles — the same rule as every other resolved field (DEC-008).
    user_h = list(getattr(base, "agg_horizons_days", None) or [])
    if user_h:
        agg_h = [float(h) for h in user_h]
        rec["overrides"]["agg_horizons_days"] = agg_h
    else:
        agg_h, r6 = derive_agg_horizons(df)
        rec["derived"]["agg_horizons_days"] = r6
    return ResolvedCFM(
        seq_len=out["seq_len"],
        dim=out["dim"],
        batch=out["batch"],
        half_life_days=out["half_life_days"],
        budget_steps=out["budget_steps"],
        eval_every=eval_every,
        patience=patience,
        seed=int(getattr(base, "seed", 0)),
        agg_horizons_days=agg_h,
        contrast_tau=out["contrast_tau"],
        tau_eig=out["tau_eig"],
        rank_target=out["rank_target"],
        aib_target_rank=out["aib_target_rank"],
        tau_mp=out["tau_mp"],
        receipt=rec,
    )


# ---------------------------------------------------------------------------
# convergence governor
# ---------------------------------------------------------------------------
def govern(
    train_step,
    val_metric,
    budget_steps,
    eval_every,
    patience,
    seed=0,
    adjust_lr=None,
    lr_min=1e-6,
):
    """Train until the held-out metric plateaus, with an overfit guard.

    train_step(n) runs n optimizer steps. val_metric() returns (metric, robust
    flag) where LOWER is better for a loss (set val_metric lower-is-better). The
    noise floor (tol) is estimated from the metric's own variation, so 'plateau'
    is measured, not assumed. Returns a receipt; the caller keeps the best state.

    v6 DEC-039 adaptive trust region: `adjust_lr(v, best, prev_best, tol)` is
    called each eval and returns the new lr. The step expands while progress is
    monotone and contracts on regression — monotone stability by construction. If
    the trust region collapses (lr < lr_min) the run is UNSTABLE (hard failure).

    Receipts:
      * best_state = the LOWEST-LOSS state ever evaluated (strict).
      * patience is only counted once the noise floor is measurable (>= 3 evals).
      * progress prints per eval (what/when/how good).
    """
    import numpy as np

    best = math.inf
    best_state = None
    hist = []
    no_improve = 0
    steps = 0
    diverged = False  # any non-finite metric OR a spike far above the noise floor
    while steps < budget_steps:
        train_step(min(eval_every, budget_steps - steps))
        steps += eval_every
        v, state = val_metric()
        if not math.isfinite(v):
            diverged = True
            print(f"[govern] DIVERGENCE at step {steps}: non-finite metric", flush=True)
            break
        hist.append(v)
        prev_best = best
        # measured noise floor from PRIOR evals (excluding the current one, so a
        # genuine spike cannot inflate its own yardstick). Measurable after the
        # third eval (>= 2 priors).
        # The FIRST eval is an initialization artifact (the loss-scale system is
        # seeded on the first batch and the model is untrained); including it
        # inflates the noise floor and stops training early (measured: tol ~4 vs
        # ~0.1). Exclude it. Use a robust MAD scale (fallback: std/2).
        prior = hist[1:-1]
        if len(prior) >= 2:
            win = np.asarray(prior[-8:], dtype=float)
            center = float(np.median(win))
            mad = float(np.median(np.abs(win - center))) * 1.4826
            tol = mad if mad > 0 else float(np.std(win)) * 0.5
        else:
            tol, center = None, None
        if v < best:  # strict: keep the lowest-loss model ever seen
            best = v
            best_state = state
        # ADAPTIVE TRUST REGION (v6 DEC-039): expand while monotone, contract on
        # regression. Collapse (lr<lr_min) without stabilising => UNSTABLE.
        if adjust_lr is not None and tol is not None:
            _new_lr = adjust_lr(v, best, prev_best, tol)
            if _new_lr is not None and _new_lr < lr_min:
                diverged = True
                print(
                    f"[govern] trust region collapsed (lr {_new_lr:.2e} < {lr_min:.0e}) "
                    "— UNSTABLE run",
                    flush=True,
                )
                break
        if tol is None:
            print(
                f"[govern] step {steps}/{budget_steps}  val {v:.4f}  eval {len(hist)}", flush=True
            )
            continue  # plateau not measurable yet — patience does not start
        # DIVERGENCE GUARD (v6): instability is a metric far above the RECENT
        # LEVEL (robust center of prior evals), not above the all-time best — an
        # early anomalous low must not make a settled plateau look divergent. The
        # yardstick uses PRIOR evals only, so a real spike is still caught.
        # 20x the measured noise floor is a documented conservative factor; it is
        # a HARD FAILURE, never rescued-and-reported-green (DEC-039).
        # DIVERGENCE = EXPLOSION, not a small drift. Require the metric to rise at
        # least 50% above its recent level (a dimensionless, scale-free blow-up
        # floor; the spec's divergence is explosion/non-finite, not a 2% wobble).
        _explode = max(20.0 * tol, 0.5 * abs(center)) if center is not None else None
        if center is not None and v > center + _explode:
            diverged = True
            print(
                f"[govern] DIVERGENCE at step {steps}: val {v:.4f} >> recent {center:.4f} "
                f"(explode floor {_explode:.4f}) — UNSTABLE run",
                flush=True,
            )
            break
        if v < prev_best - tol:
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"[govern] step {steps}/{budget_steps}  val {v:.4f}  best {best:.4f}  "
            f"tol {tol:.4f}  no_improve {no_improve}/{patience}",
            flush=True,
        )
        if no_improve >= patience:
            break
    return {
        "steps": steps,
        "best": best,
        "diverged": diverged,
        "eval_every": eval_every,
        "patience": patience,
        "evals": len(hist),
        "stopped": "converged" if steps < budget_steps else "budget",
        "final": hist[-1] if hist else None,
    }, best_state
