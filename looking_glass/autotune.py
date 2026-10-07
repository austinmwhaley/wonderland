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

    sig = "|".join(
        str(x)
        for x in (
            df.height,
            df["customer_key"].n_unique(),
            df["event_type"].n_unique(),
            str(df["event_ts"][0]),
            str(df["event_ts"][-1]),
            int(round(float(df["value"].fill_null(0).sum()))),
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
    q = float(np.quantile(lengths, 0.9))
    val = int(clip(round(q), 8, 256))
    return val, {"seq_len": f"p90 events/customer={q:.0f} -> {val}"}


def derive_dim(n_seqs, total_events, vocab_sizes) -> tuple[int, dict]:
    """Capacity scales with the information in the data (events, vocab), not a
    fixed width. Bounded by a conservative hardware-safe range."""
    info = math.sqrt(max(total_events, 1)) * math.log2(2 + sum(vocab_sizes) + n_seqs)
    val = int(clip(2 ** round(math.log2(max(info / 64.0, 8.0))), 16, 256))
    return val, {"dim": f"info={info:.0f} -> {val}"}


def derive_batch(n_seqs) -> tuple[int, dict]:
    """Batch scales with data but is BOUNDED BY HARDWARE (speed principle).
    An unbounded derived batch makes large runs blow up superlinearly."""
    import torch

    if torch.cuda.is_available():
        # memory-safe cap for this model (K experts x JEPA x multi-gamma)
        free = torch.cuda.mem_get_info()[0] / (1024**3)
        cap = int(clip(2 ** round(math.log2(max(free * 8.0, 8.0))), 8, 256))
    else:
        cap = 32
    val = int(clip(2 ** round(math.log2(max(n_seqs / 32.0, 8.0))), 8, cap))
    return val, {"batch": f"n_seqs={n_seqs} hw_cap={cap} -> {val}"}


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
    receipt: dict = field(default_factory=dict)


def resolve_cfm(base, df, keys, vocab_sizes) -> ResolvedCFM:
    """Resolve all config from data + hardware. `base` is the raw config; any
    explicit override wins and is recorded (recording alone never applied it —
    the derived value overwrote the override downstream)."""
    lengths = sequence_lengths(df, keys)
    seq_len, r1 = derive_seq_len(lengths)
    total = int(sum(lengths)) if lengths else 0
    dim, r2 = derive_dim(len(keys), total, vocab_sizes)
    batch, r3 = derive_batch(len(keys))
    hl, r4 = derive_half_life(df)
    budget, r5 = derive_budget(len(keys), batch, dim)
    # eval cadence: enough evals to detect a plateau inside the budget.
    eval_every = int(clip(round(budget / 20.0), 5, 5000))
    # patience: a fraction of the eval budget, so convergence, not a count, is
    # what stops training. Overridable.
    patience = int(clip(round(budget / eval_every / 5.0), 3, 20))
    rec = {"derived": {}, "overrides": {}}
    out = {}
    for name, (val, rr) in (
        ("seq_len", (seq_len, r1)),
        ("dim", (dim, r2)),
        ("batch", (batch, r3)),
        ("half_life_days", (hl, r4)),
        ("budget_steps", (budget, r5)),
    ):
        user = getattr(base, name if name != "half_life_days" else "state_half_life_days", None)
        default = {
            "seq_len": 128,
            "dim": 64,
            "batch": 64,
            "half_life_days": 30.0,
            "budget_steps": 0,
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
        receipt=rec,
    )


# ---------------------------------------------------------------------------
# convergence governor
# ---------------------------------------------------------------------------
def govern(train_step, val_metric, budget_steps, eval_every, patience, seed=0):
    """Train until the held-out metric plateaus, with an overfit guard.

    train_step(n) runs n optimizer steps. val_metric() returns (metric, robust
    flag) where LOWER is better for a loss (set val_metric lower-is-better). The
    noise floor (tol) is estimated from the metric's own variation, so 'plateau'
    is measured, not assumed. Returns a receipt; the caller keeps the best state.

    Receipts:
      * best_state = the LOWEST-LOSS state ever evaluated (strict) — the old
        rule only recorded a new best when it beat the old one by more than the
        noise tol, so training returned an older, worse model.
      * patience is only counted once the noise floor is measurable (>= 3 evals):
        calling a run "converged" off one or two points is not a measurement.
      * progress prints per eval (what/when/how good).
    """
    import numpy as np

    best = math.inf
    best_state = None
    hist = []
    no_improve = 0
    steps = 0
    while steps < budget_steps:
        train_step(min(eval_every, budget_steps - steps))
        steps += eval_every
        v, state = val_metric()
        if not math.isfinite(v):
            break
        hist.append(v)
        prev_best = best
        # measured noise floor from recent variation (needs >= 3 points)
        tol = float(np.std(hist[-3:])) * 0.5 if len(hist) >= 3 else None
        if v < best:  # strict: keep the lowest-loss model ever seen
            best = v
            best_state = state
        if tol is None:
            print(
                f"[govern] step {steps}/{budget_steps}  val {v:.4f}  eval {len(hist)}", flush=True
            )
            continue  # plateau not measurable yet — patience does not start
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
        "eval_every": eval_every,
        "patience": patience,
        "evals": len(hist),
        "stopped": "converged" if steps < budget_steps else "budget",
        "final": hist[-1] if hist else None,
    }, best_state
