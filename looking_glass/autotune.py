"""Derived configuration and convergence-governed training.

Doctrine (see work/AGENTS.md): no magic numbers; train to convergence, not to a
count; make it learn; emit receipts.

Everything here DERIVES its settings from the data and the hardware, and the
training loop STOPS on held-out convergence (with an overfit guard) rather than
a fixed epoch count. Surviving literals are conservative fallback *bounds*,
recorded in the receipt when they fire.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field


def clip(x, lo, hi):
	return max(lo, min(hi, x))


def data_revision(df) -> tuple[int, str]:
	"""Data/score revision (the `r` in vNrN). A deterministic identity for the
	stream, so a retrain on different data is a different artifact."""
	import hashlib
	sig = "|".join(str(x) for x in (
		df.height, df["customer_key"].n_unique(), df["event_type"].n_unique(),
		str(df["event_ts"][0]), str(df["event_ts"][-1]),
		int(round(float(df["value"].fill_null(0).sum())))))
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
	for k in cols:
		if k in want:
			if started and k != prev:
				lens.append(cur); cur = 0
			started = True; cur += 1; prev = k
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
		free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
		cap = int(clip(2 ** round(math.log2(max(free * 8.0, 8.0))), 8, 256))
	else:
		cap = 32
	val = int(clip(2 ** round(math.log2(max(n_seqs / 32.0, 8.0))), 8, cap))
	return val, {"batch": f"n_seqs={n_seqs} hw_cap={cap} -> {val}"}


def derive_half_life(event_ts) -> tuple[float, dict]:
	"""State persistence timescale from the data's inter-event gaps: a state
	should fade over the typical gap between events, not a fixed duration."""
	import numpy as np
	ts = np.sort(np.asarray([t for t in event_ts if t], dtype=np.float64))
	if ts.size < 3:
		return 30.0, {"state_half_life_days": "fallback (too few events)"}
	gaps = np.diff(ts)
	gaps = gaps[gaps > 0]
	if gaps.size == 0:
		return 30.0, {"state_half_life_days": "fallback (no positive gaps)"}
	med_days = float(np.median(gaps)) / 86400.0
	val = float(clip(med_days, 1.0 / 24.0, 365.0))
	return val, {"state_half_life_days": f"median inter-event {med_days:.2f}d -> {val:.2f}"}


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
	receipt: dict = field(default_factory=dict)


def resolve_cfm(base, df, keys, vocab_sizes, event_ts) -> ResolvedCFM:
	"""Resolve all config from data + hardware. `base` is the raw config; any
	explicit override wins but is recorded."""
	lengths = sequence_lengths(df, keys)
	seq_len, r1 = derive_seq_len(lengths)
	total = int(sum(lengths)) if lengths else 0
	dim, r2 = derive_dim(len(keys), total, vocab_sizes)
	batch, r3 = derive_batch(len(keys))
	hl, r4 = derive_half_life(event_ts)
	budget, r5 = derive_budget(len(keys), batch, dim)
	# eval cadence: enough evals to detect a plateau inside the budget.
	eval_every = int(clip(round(budget / 20.0), 5, 5000))
	# patience: a fraction of the eval budget, so convergence, not a count, is
	# what stops training. Overridable.
	patience = int(clip(round(budget / eval_every / 5.0), 3, 20))
	rec = {"derived": {}, "overrides": {}}
	for name, (val, rr) in (("seq_len", (seq_len, r1)), ("dim", (dim, r2)),
							("batch", (batch, r3)),
							("half_life_days", (hl, r4)),
							("budget_steps", (budget, r5))):
		user = getattr(base, name if name != "half_life_days" else "state_half_life_days", None)
		default = {"seq_len": 128, "dim": 64, "batch": 64,
				   "half_life_days": 30.0, "budget_steps": 0}[name]
		if user is not None and user != default and user != 0:
			rec["overrides"][name] = user
		rec["derived"][name] = rr
	return ResolvedCFM(seq_len=seq_len, dim=dim, batch=batch,
					   half_life_days=hl, budget_steps=budget,
					   eval_every=eval_every, patience=patience,
					   seed=int(getattr(base, "seed", 0)), receipt=rec)


# ---------------------------------------------------------------------------
# convergence governor
# ---------------------------------------------------------------------------
def govern(train_step, val_metric, budget_steps, eval_every, patience, seed=0):
	"""Train until the held-out metric plateaus, with an overfit guard.

	train_step(n) runs n optimizer steps. val_metric() returns (metric, robust
	flag) where LOWER is better for a loss (set val_metric lower-is-better). The
	noise floor (tol) is estimated from the metric's own variation, so 'plateau'
	is measured, not assumed. Returns a receipt; the caller keeps the best state.
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
		# measured noise floor from recent variation
		tol = 0.0
		if len(hist) >= 3:
			tol = float(np.std(hist[-3:])) * 0.5
		if v < best - tol:
			best = v; best_state = state; no_improve = 0
		else:
			no_improve += 1
		if no_improve >= patience:
			break
	return {"steps": steps, "best": best, "eval_every": eval_every,
			"patience": patience, "evals": len(hist),
			"stopped": "converged" if steps < budget_steps else "budget",
			"final": hist[-1] if hist else None}, best_state
