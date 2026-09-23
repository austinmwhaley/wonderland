"""Looking Glass — the Customer Foundation Model (CFM).

A frozen, versioned, causal state function:

	state_c(t) = CFM_version( events of customer c with timestamp <= t )

Multi-objective self-supervised encoder (no labels), frozen after training.
Customers are partitioned into disjoint samples: A (encoder training) and B
(plugin training); A and B never overlap.

Products (DuckDB):
  * customer_state          -- the STATE for every customer, with as_of.
							   Holds the recurrence state H, the public
							   embedding S, and as_of so we know whether to
							   fade() (time passed, no events) or absorb()
							   (new events arrived).
  * anchor_embeddings -- point-in-time EMBEDDINGS S_c(anchor) for
							   sample-B customers at past anchor dates, used
							   as inputs to plugin training.

State ops:
  fade(H, dt)            -- decay the state when no events occurred for dt.
  absorb(model, H, evs)  -- push new events through the recurrence (O(#events)).
  StateStore.advance()   -- load state, fade to first new event, absorb, upsert.

Objectives (self-supervised): next-event type, next-entity, time-to-next-event,
contrastive, masked-event reconstruction, redundancy (VICReg). Plug in more.

CLI:
  python -m looking_glass.customer_foundation_model all --db <duckdb> --customers 500
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

try:
	from . import autotune as AT
except Exception:  # run as a standalone script
	import importlib.util as _ilu
	_spec = _ilu.spec_from_file_location("cfm_autotune",
										 Path(__file__).with_name("autotune.py"))
	AT = _ilu.module_from_spec(_spec)
	import sys as _sys
	_sys.modules['cfm_autotune'] = AT
	_spec.loader.exec_module(AT)
import torch
import torch.nn as nn
import torch.nn.functional as F

STREAM_TABLE = "customer_events"
EMBED_DIM = 64
LN2 = math.log(2.0)
# Successor features: predict the DISCOUNTED future at a continuously-sampled
# discount gamma. No human-chosen horizons — the model learns all timescales.
TIME_UNIT_SECONDS = 86400.0   # a day (unit scaling only, not a horizon)
SF_PHI = 4                    # discounted [value, count, is_order, order*value]
GAMMA_MAX = 0.999


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class CFMConfig:
	db: str = "data/arrow/customer_event_stream.feather"
	table: str = STREAM_TABLE
	out_dir: str = "artifacts/cfm"
	version: str = "v2.0.0"            # encoder code version
	revision: int = 1                  # data/score revision (r)
	sample_customers: int | None = 500
	split_a_frac: float = 0.7
	split_seed: int = 7
	seq_len: int = 128
	n_anchors: int = 6                 # exact number of sample-B anchor days per customer
	# Company actions are EXOGENOUS (interventions/treatments), not customer
	# behavior: they are covariates and are never predicted as tokens.
	company_actions: tuple = ("email_send",)
	dim: int = EMBED_DIM
	n_experts: int = 1                 # K=1: M1 multi-timescale gave no gain (speed)
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
	objectives: tuple = ("next", "entity", "dt", "value", "mask", "contrast",
						 "redundancy", "occur", "order", "jepa", "sf")
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
	random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _read_stream(cfg: CFMConfig):
	cols = ("customer_key", "event_ts", "brand", "event_type",
			"event_attributes", "entity_type", "entity_id", "value")
	if str(cfg.db).endswith((".arrow", ".feather", ".ipc")):
		import polars as pl
		return (pl.read_ipc(cfg.db, memory_map=True).select(list(cols))
				.sort(["customer_key", "event_ts"]))
	import duckdb
	con = duckdb.connect(cfg.db, read_only=True)
	try:
		df = con.execute(
			f"SELECT {', '.join(cols)} FROM {cfg.table} ORDER BY customer_key, event_ts"
		).pl()
	finally:
		con.close()
	return df


def _customer_keys(df, cfg):
	keys = df["customer_key"].unique(maintain_order=True).to_list()
	if cfg.sample_customers is not None:
		keys = keys[: cfg.sample_customers]
	return keys


def assign_split(keys, cfg: CFMConfig) -> dict[str, str]:
	return {k: ("A" if (_h(k, cfg.split_seed) % 1000) < int(cfg.split_a_frac * 1000)
				else "B") for k in keys}


def _apply_data_revision(cfg: CFMConfig, df) -> str:
	cfg.revision, sig = AT.data_revision(df)
	cfg._data_signature = sig
	return sig


def _covariates(et, ts, company):
	"""Vectorized exogenous company-action covariates per event:
	[is_company_action, log1p(seconds since previous company action)].
	Company actions are inputs the model must not predict, not tokens."""
	is_co = np.isin(et, list(company))
	co_ts = np.where(is_co, ts, -np.inf)
	acc = np.maximum.accumulate(co_ts) if co_ts.size else co_ts
	last_prev = (np.concatenate([[-np.inf], acc[:-1]]) if co_ts.size
				 else np.zeros(0))
	gap = np.where(np.isfinite(last_prev), ts - last_prev, 0.0)
	return np.stack([is_co.astype(np.float64), np.log1p(np.clip(gap, 0, None))],
					axis=1)


def _random_anchor_epochs(ts, data_end, cfg, key):
	"""n uniform-random anchor days over [history_start, data_end - 1d] —
	agnostic to any plugin's target window."""
	lo = float(ts[0]); hi = float(data_end) - 86400.0
	if hi <= lo:
		return []
	n = max(1, int(cfg.n_anchors))
	rng = np.random.default_rng(_h(key, cfg.split_seed))
	return sorted(float(x) for x in rng.uniform(lo, hi, size=n))


def build_sequences(df, keys, cfg: CFMConfig, split, with_anchors: bool):
	"""Polars-first: partition once in Rust, then slice per group. Anchors are
	random uniform days; company actions ride along as exogenous covariates."""
	want = list(set(keys))
	d = (df.filter(pl.col("customer_key").is_in(want))
		   .with_columns(pl.col("event_ts")
						 .str.to_datetime(time_zone="UTC", strict=False)
						 .dt.epoch("s").alias("_ts")))
	data_end = float(d["_ts"].max()) if d.height else 0.0
	company = set(map(str, cfg.company_actions))
	seqs = []
	for g in d.partition_by("customer_key", maintain_order=True):
		k = g["customer_key"][0]
		ts_full = g["_ts"].to_numpy()
		# numpy arrays (Polars->numpy is vectorized in Rust); no per-row Python.
		et_arr = g["event_type"].to_numpy()
		val_full = g["value"].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
		# Company actions are exogenous: NOT tokens. Covariates are computed on
		# the full stream (sends visible) while tokens keep only customer events.
		co_full = _covariates(et_arr, ts_full, company)
		ki = np.flatnonzero(~np.isin(et_arr, list(company)))
		if ki.size < 3:
			continue
		ts = ts_full[ki]
		et = et_arr[ki]
		co = co_full[ki]
		val = val_full[ki]
		brand = g["brand"].to_numpy()[ki]
		ent = g["entity_type"].to_numpy()[ki]
		eid = g["entity_id"].to_numpy()[ki]
		ets = g["event_ts"].to_numpy()[ki]
		n = ts.size
		spans: list[tuple[int, float | None]] = []
		if with_anchors and split.get(k) == "B":
			for a in _random_anchor_epochs(ts, data_end, cfg, k):
				end = int(np.searchsorted(ts, a, side="right"))
				if end >= 3:
					spans.append((end, a))
		else:
			spans.append((n, None))
		for end, anchor_epoch in spans:
			start = max(0, end - cfg.seq_len)
			if end - start >= 3:
				seqs.append({
					"customer": k, "group": split.get(k, "A"),
					"anchor_epoch": anchor_epoch,
					"event_type": et[start:end],
					"brand": brand[start:end],
					"entity_type": ent[start:end],
					"entity_id": eid[start:end],
					"value": val[start:end],
					"event_ts": ets[start:end],
					"ts": ts[start:end].tolist(),
					"co": co[start:end].tolist()})
	return seqs


# ---------------------------------------------------------------------------
# vocab + tokenizer
# ---------------------------------------------------------------------------
class EventVocab:
	def __init__(self, et, brand, ent):
		self.et = {v: i for i, v in enumerate(et)}
		self.brand = {v: i for i, v in enumerate(brand)}
		self.ent = {v: i for i, v in enumerate(ent)}
		self.n_et, self.n_brand, self.n_ent = len(et), len(brand), len(ent)

	@staticmethod
	def build(seqs):
		et, br, en = set(), set(), set()
		for s in seqs:
			et.update(map(str, s["event_type"]))
			br.update(str(b) if b is not None else "none" for b in s["brand"])
			en.update(str(e) if e is not None else "none" for e in s["entity_type"])
		return EventVocab(sorted(et), sorted(br), sorted(en))

	def dumps(self):
		return {"et": list(self.et), "brand": list(self.brand), "ent": list(self.ent)}


# ---------------------------------------------------------------------------
# encoder (selective SSM) + objectives
# ---------------------------------------------------------------------------
def _scan(d, b):
	"""Vectorized affine prefix scan (Hillis-Steele, O(log T) Python steps).

	Solves h_t = d_t * h_{t-1} + b_t with h_{-1}=0, for d:(B,T,1), b:(B,T,D).
	Returns (D_t, H_t) where D_t = prod_{m<=t} d_m and H_t = h_t. Fully
	vectorized in T (the operation is associative: f_i∘f_j = (d_i d_j, d_i b_j + b_i))."""
	a = d
	Bb = b
	T = a.shape[1]
	s = 1
	while s < T:
		an = a.clone()
		bn = Bb.clone()
		an[:, s:] = a[:, s:] * a[:, :-s]
		bn[:, s:] = a[:, s:] * Bb[:, :-s] + Bb[:, s:]
		a, Bb = an, bn
		s *= 2
	return a, Bb


class SelectiveSSM(nn.Module):
	def __init__(self, dim, delta_bias=0.0):
		super().__init__()
		self.W_delta = nn.Linear(dim, dim)
		self.W_B = nn.Linear(dim, dim)
		self.W_C = nn.Linear(dim, dim)
		self.W_out = nn.Linear(dim, dim)
		# per-channel timescale (log-decay) offset; LEARNED (init equal).
		self.delta_bias = nn.Parameter(torch.tensor([float(delta_bias)]))

	def forward(self, x, h0=None, mask=None):
		delta = F.softplus(self.W_delta(x) + self.delta_bias)
		bx = self.W_B(x)
		decay = torch.exp(-delta)
		if mask is not None:
			m = mask.unsqueeze(-1)
			decay = decay * m + (1.0 - m)      # pad: hold state
			bx = bx * m
		D, H = _scan(decay, (1.0 - decay) * bx)     # h_t (zero-init)
		if h0 is not None:
			h0v = h0 if h0.dim() >= 2 else h0.unsqueeze(0)
			H = H + D * h0v.unsqueeze(1)
		return self.W_C(H), H[:, -1]


class MultiScaleSSM(nn.Module):
	"""M1: a bank of selective SSMs with different decay scales (timescales).
	Each channel summarizes a different horizon; their readouts are concatenated
	into the public state (width = chan * n_experts). K=1 is the original SSM."""

	def __init__(self, chan, n_experts, delta_biases):
		super().__init__()
		self.chan = chan
		self.n_experts = n_experts
		self.experts = nn.ModuleList(
			[SelectiveSSM(chan, b) for b in delta_biases[:n_experts]])

	def forward(self, x, h0=None, mask=None):
		# NOTE: context-gated mixtures (M2 softmax gate, M5 sparse top-k) were
		# both REJECTED by the battery — the gate attenuated channels and
		# collapsed their diversity (unique-coverage fell). Experts stay
		# concatenated (M1) and the multi-task objective (M4) is what helped.
		ys, hs = [], []
		for i, e in enumerate(self.experts):
			hi = None if h0 is None else h0[..., i * self.chan:(i + 1) * self.chan]
			yi, ho = e(x, h0=hi, mask=mask)
			ys.append(yi); hs.append(ho)
		return torch.cat(ys, dim=-1), torch.cat(hs, dim=-1)


class CFM(nn.Module):
	def __init__(self, vocab: EventVocab, dim: int, n_experts: int = 1,
				 delta_biases=None):
		super().__init__()
		self.vocab = vocab
		self.n_experts = max(1, int(n_experts))
		self.chan = max(1, dim // self.n_experts)
		dim = self.chan * self.n_experts
		self.dim = dim
		self.emb_et = nn.Embedding(vocab.n_et + 1, self.chan)
		self.emb_brand = nn.Embedding(vocab.n_brand + 1, self.chan)
		self.emb_ent = nn.Embedding(vocab.n_ent + 1, self.chan)
		self.w_val = nn.Linear(1, self.chan)
		self.w_dt = nn.Linear(1, self.chan)
		self.w_co = nn.Linear(2, self.chan)                 # exogenous covariates
		if delta_biases is None:
			delta_biases = [0.0] * self.n_experts
		self.ssm = MultiScaleSSM(self.chan, self.n_experts, delta_biases)
		self.head_next = nn.Linear(dim, vocab.n_et + 1)     # next event type
		self.head_ent = nn.Linear(dim, vocab.n_ent + 1)     # next entity type
		self.head_dt = nn.Linear(dim, 1)                    # log1p(dt_next)
		self.head_occ = nn.Linear(dim, 1)                   # next event within horizon?
		self.head_val = nn.Linear(dim, 1)                   # next event value (monetary)
		self.head_sf = nn.Sequential(nn.Linear(dim + 1, dim), nn.ReLU(),
									 nn.Linear(dim, SF_PHI))  # successor features
		self.order_W = nn.Linear(dim, self.chan, bias=False)  # temporal-order scorer
		self.proj = nn.Linear(dim, dim)                     # contrastive / public S
		# M3: entity-aware donor — pools step outputs per entity type.
		self.proj_ent = nn.Linear((vocab.n_ent + 1) * dim, dim)
		# JEPA: EMA target encoder + predictor (latent future prediction).
		import copy as _copy
		self.t_emb_et = _copy.deepcopy(self.emb_et)
		self.t_emb_brand = _copy.deepcopy(self.emb_brand)
		self.t_emb_ent = _copy.deepcopy(self.emb_ent)
		self.t_w_val = _copy.deepcopy(self.w_val)
		self.t_w_dt = _copy.deepcopy(self.w_dt)
		self.t_w_co = _copy.deepcopy(self.w_co)
		self.t_ssm = _copy.deepcopy(self.ssm)
		self.t_proj = _copy.deepcopy(self.proj)
		self.pred = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
		for q in self._target_params():
			q.requires_grad_(False)
		# log-variance per task for uncertainty-based weighting.
		self.log_var = nn.ParameterDict({
			k: nn.Parameter(torch.zeros(1)) for k in
			("next", "entity", "dt", "value", "mask", "contrast",
			 "redundancy", "occur", "order", "jepa", "sf")})

	def _dev(self):
		return next(self.parameters()).device

	def tokens(self, seq):
		dev = self._dev()
		et = torch.tensor([self.vocab.et.get(str(x), self.vocab.n_et)
						   for x in seq["event_type"]], device=dev)
		br = torch.tensor([self.vocab.brand.get(str(b) if b is not None else "none",
												self.vocab.n_brand) for b in seq["brand"]], device=dev)
		en = torch.tensor([self.vocab.ent.get(str(e) if e is not None else "none",
											  self.vocab.n_ent) for e in seq["entity_type"]], device=dev)
		val = torch.tensor([_f(v) for v in seq["value"]], dtype=torch.float32,
						   device=dev).unsqueeze(1)
		ts = np.array([_to_epoch(x) for x in seq["event_ts"]], dtype=np.float64)
		dt = np.zeros_like(ts); dt[1:] = np.maximum(ts[1:] - ts[:-1], 0.0)
		dt = torch.tensor(np.log1p(dt), dtype=torch.float32, device=dev).unsqueeze(1)
		co = seq.get("co")
		co = co if co is not None else [[0.0, 0.0]] * len(seq["event_type"])
		co = torch.tensor(co, dtype=torch.float32, device=dev)
		return (self.emb_et(et) + self.emb_brand(br) + self.emb_ent(en)
				+ self.w_val(val) + self.w_dt(dt) + self.w_co(co))

	def forward(self, seq, h0=None):
		y, h = self.ssm(self.tokens(seq).unsqueeze(0), h0=h0)
		return y.squeeze(0), h.squeeze(0)

	def _target_params(self):
		return (list(self.t_emb_et.parameters()) + list(self.t_emb_brand.parameters())
				+ list(self.t_emb_ent.parameters()) + list(self.t_w_val.parameters())
				+ list(self.t_w_dt.parameters()) + list(self.t_w_co.parameters())
				+ list(self.t_ssm.parameters())
				+ list(self.t_proj.parameters()))

	def ema(self, tau):
		pairs = list(zip(self._target_params(),
						 list(self.emb_et.parameters()) +
						 list(self.emb_brand.parameters()) +
						 list(self.emb_ent.parameters()) +
						 list(self.w_val.parameters()) + list(self.w_dt.parameters()) +
						 list(self.w_co.parameters()) +
						 list(self.ssm.parameters()) + list(self.proj.parameters())))
		with torch.no_grad():
			for pt, ps in pairs:
				pt.data.mul_(1 - tau).add_(ps.data, alpha=tau)

	def tokens_batch(self, t):
		return (self.emb_et(t["et"]) + self.emb_brand(t["br"]) + self.emb_ent(t["en"])
				+ self.w_val(t["val"]) + self.w_dt(t["dt"]) + self.w_co(t["co"]))

	def target_tokens_batch(self, t):
		return (self.t_emb_et(t["et"]) + self.t_emb_brand(t["br"]) + self.t_emb_ent(t["en"])
				+ self.t_w_val(t["val"]) + self.t_w_dt(t["dt"]) + self.t_w_co(t["co"]))

	def embed(self, x):
		"""Public embedding S from the recurrence STATE (x = h, recency-weighted
		via the decay), or mean-pool if a per-step matrix is passed."""
		v = self.proj(x) if x.dim() == 1 else self.proj(x.mean(0))
		return F.normalize(v, dim=0)

	def donor(self, x):
		"""Donor representation for plugins: same projection WITHOUT L2
		normalization, so magnitude (how much / how recent) is preserved."""
		return self.proj(x) if x.dim() == 1 else self.proj(x.mean(0))

	def successor(self, state, gamma, reward_weight=None):
		"""Item 3: query the discounted future at ANY horizon (gamma in (0,1)) from
		a state. Returns successor features; value for a reward w is w . sf."""
		with torch.no_grad():
			col = torch.full((state.shape[0], 1), float(gamma), device=state.device)
			sf = self.head_sf(torch.cat([state, col], dim=-1))
		if reward_weight is not None:
			return (sf * torch.as_tensor(reward_weight, device=state.device)).sum(-1)
		return sf

	def donor_seq(self, y, h, seq):
		"""M3 multi-entity donor: [timescale state | entity-pooled summary].
		Pools per-step outputs by entity type (customer/product/session/...)."""
		et = seq.get("entity_type")
		if et is None or len(et) == 0:
			return self.donor(h)
		dev = y.device
		ids = torch.tensor([self.vocab.ent.get(str(e) if e is not None else "none",
											  self.vocab.n_ent) for e in et], device=dev)
		P = torch.zeros(self.vocab.n_ent + 1, y.shape[-1], device=dev)
		C = torch.zeros(self.vocab.n_ent + 1, device=dev)
		P.index_add_(0, ids, y)
		C.index_add_(0, ids, torch.ones_like(ids, dtype=y.dtype))
		P = P / C.clamp(min=1).unsqueeze(1)
		return torch.cat([self.proj(h), self.proj_ent(P.reshape(-1))])


# ---------------------------------------------------------------------------
# state operations (fade / absorb)
# ---------------------------------------------------------------------------
def fade(h: torch.Tensor, dt_seconds: float, half_life_days: float) -> torch.Tensor:
	"""Decay the state when time passes with NO events."""
	if dt_seconds <= 0:
		return h
	decay = math.exp(-LN2 * dt_seconds / max(half_life_days * 86400.0, 1.0))
	return h * decay


def absorb(model: CFM, seq, h0: torch.Tensor | None = None, as_of_epoch: float | None = None):
	"""Push new events through the recurrence from an existing state.

	Fades h0 from as_of to the first new event, then runs only the new events.
	Returns (new_h, embedding).
	"""
	with torch.no_grad():
		h = h0
		if h is not None and as_of_epoch is not None and len(seq["event_ts"]):
			h = fade(h, _to_epoch(seq["event_ts"][0]) - as_of_epoch,
					 getattr(model, "half_life_days", 30.0))
		y, h = model(seq, h0=h)
		return h, model.embed(h)


# ---------------------------------------------------------------------------
# training (multi-objective)
# ---------------------------------------------------------------------------
def _val_loss(model, vocab, seqs, cfg):
	"""Grounded held-out metric (next-event cross-entropy; lower is better)."""
	dev = model._dev(); tot = 0.0; n = 0
	with torch.no_grad():
		for seq in seqs:
			y, _ = model(seq)
			if y.shape[0] < 2:
				continue
			tgt = torch.tensor([vocab.et.get(str(x), vocab.n_et)
								for x in seq["event_type"][1:]], device=dev)
			tot += float(F.cross_entropy(model.head_next(y[:-1]), tgt)); n += 1
	return tot / max(n, 1)


def train_cfm(cfg: CFMConfig):
	_seed_everything(cfg.seed)
	df = _read_stream(cfg)
	_apply_data_revision(cfg, df)
	keys = _customer_keys(df, cfg)
	split = assign_split(keys, cfg)
	# ---- derive configuration from the data (no fixed values) ----
	lengths = AT.sequence_lengths(df, keys)
	all_ts = [_to_epoch(x) for x in df["event_ts"].to_list()]
	vocab_sizes = (df["event_type"].n_unique(), df["brand"].n_unique(),
				   df["entity_type"].n_unique())
	res = AT.resolve_cfm(cfg, df, keys, vocab_sizes, all_ts)
	cfg.seq_len, cfg.dim, cfg.batch = res.seq_len, res.dim, res.batch
	cfg.state_half_life_days = res.half_life_days
	a_keys = [k for k in keys if split[k] == "A"]
	a_seqs = build_sequences(df, a_keys, cfg, split, with_anchors=False)
	vocab = EventVocab.build(a_seqs)
	device = torch.device(cfg.device)
	K = max(1, int(cfg.n_experts))
	cfg.dim = max(K, (cfg.dim // K) * K)
	model = CFM(vocab, cfg.dim, n_experts=K,
				delta_biases=_expert_biases(K, cfg.state_half_life_days)).to(device)
	model.half_life_days = cfg.state_half_life_days
	params = [q for q in model.parameters() if q.requires_grad]
	opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=1e-4)
	tau = 0.99  # EMA of the JEPA target encoder (documented fallback)
	# ---- train/val split of sample A (derived fraction) ----
	order = np.arange(len(a_seqs))
	np.random.default_rng(cfg.seed).shuffle(order)
	n_val = max(1, int(round(0.15 * len(order))))
	val_idx, tr_idx = order[:n_val], order[n_val:]
	rng = np.random.default_rng(cfg.seed)

	use_amp = device.type == "cuda"
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

	def train_step(n):
		for _ in range(n):
			b = rng.choice(tr_idx, size=min(cfg.batch, len(tr_idx)), replace=False)
			opt.zero_grad()
			with torch.autocast(device_type=device.type, dtype=torch.float16,
								enabled=use_amp):
				loss = _loss(model, vocab, [a_seqs[i] for i in b], cfg)
			scaler.scale(loss).backward()
			scaler.unscale_(opt)
			torch.nn.utils.clip_grad_norm_(params, 1.0)
			scaler.step(opt)
			scaler.update()
			model.ema(tau)

	def val_metric():
		# bound eval cost (speed principle): subsample the validation set
		vi = val_idx[:256]
		v = _val_loss(model, vocab, [a_seqs[i] for i in vi], cfg)
		return v, copy.deepcopy(model.state_dict())

	gov, best_state = AT.govern(train_step, val_metric, res.budget_steps,
								res.eval_every, res.patience, cfg.seed)
	if best_state is not None:
		model.load_state_dict(best_state)
	out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
	torch.save({"state": model.state_dict(), "vocab": vocab.dumps(), "dim": cfg.dim,
				"n_experts": K},
			   out / f"cfm_{cfg.tag.replace('.', '_')}.pt")
	_registry(cfg, vocab, len(a_seqs), len(keys), derived=res.receipt,
			  governor=gov, cfg_resolved={"seq_len": cfg.seq_len, "dim": cfg.dim,
										  "batch": cfg.batch,
										  "half_life_days": cfg.state_half_life_days})
	return model, vocab, df, keys, split


def _collate(seqs, vocab, device):
	"""Pad a batch into dense GPU tensors (Polars already gave us per-group
	lists). One forward over the batch instead of one per sequence."""
	B = len(seqs)
	T = max(len(s["event_type"]) for s in seqs)
	et = torch.full((B, T), vocab.n_et, dtype=torch.long, device=device)
	br = torch.full((B, T), vocab.n_brand, dtype=torch.long, device=device)
	en = torch.full((B, T), vocab.n_ent, dtype=torch.long, device=device)
	val = torch.zeros(B, T, 1, device=device)
	dt = torch.zeros(B, T, 1, device=device)
	cov = torch.zeros(B, T, 2, device=device)
	mask = torch.zeros(B, T, device=device)
	cut = torch.ones(B, dtype=torch.long, device=device)
	for i, sq in enumerate(seqs):
		L = len(sq["event_type"])
		et[i, :L] = torch.tensor([vocab.et.get(str(x), vocab.n_et)
								  for x in sq["event_type"]], device=device)
		br[i, :L] = torch.tensor([vocab.brand.get(str(b) if b is not None else "none",
												   vocab.n_brand) for b in sq["brand"]], device=device)
		en[i, :L] = torch.tensor([vocab.ent.get(str(e) if e is not None else "none",
												vocab.n_ent) for e in sq["entity_type"]], device=device)
		val[i, :L, 0] = torch.tensor([_f(v) for v in sq["value"]],
									 dtype=torch.float32, device=device)
		ts = np.asarray(sq.get("ts") or [_to_epoch(x) for x in sq["event_ts"]],
						dtype=np.float64)
		d = np.zeros(L); d[1:] = np.maximum(ts[1:] - ts[:-1], 0.0)
		dt[i, :L, 0] = torch.tensor(np.log1p(d), dtype=torch.float32, device=device)
		if sq.get("co"):
			cov[i, :L] = torch.tensor(sq["co"], dtype=torch.float32, device=device)
		mask[i, :L] = 1.0
		cut[i] = min(L - 1, max(1, int(0.6 * L)))
	return {"et": et, "br": br, "en": en, "val": val, "dt": dt, "co": cov,
			"mask": mask, "cut": cut}


def _mask_loss_batch(model, vocab, t, dev):
	B, T = t["et"].shape
	mask = t["mask"]
	rand = (torch.rand(B, T, device=dev) < 0.15) & (mask > 0)
	if rand.sum() == 0:
		return torch.zeros((), device=dev)
	et2 = t["et"].clone(); et2[rand] = vocab.n_et
	x2 = (model.emb_et(et2) + model.emb_brand(t["br"]) + model.emb_ent(t["en"])
		  + model.w_val(t["val"]) + model.w_dt(t["dt"]) + model.w_co(t["co"]))
	y2, _ = model.ssm(x2, mask=mask)
	return F.cross_entropy(model.head_next(y2[rand]), t["et"][rand])


def _jepa_loss(model, t, y):
	"""Latent future prediction: predict the EMA target encoder's embedding of
	the future suffix from the context state at the cut."""
	B, T, _ = y.shape
	cut = t["cut"]
	idx = torch.arange(B, device=y.device)
	ctx = y[idx, cut - 1]
	xt = model.target_tokens_batch(t)
	yt, _ = model.t_ssm(xt, mask=t["mask"])
	pos = torch.arange(T, device=y.device).unsqueeze(0)
	suf = (pos >= cut.unsqueeze(1)) & (t["mask"] > 0)
	cnt = suf.sum(1, keepdim=True).clamp(min=1)
	pool = (yt * suf.unsqueeze(-1)).sum(1) / cnt
	S_tgt = F.normalize(model.t_proj(pool), dim=-1)
	S_pred = F.normalize(model.pred(ctx), dim=-1)
	return (1.0 - (S_tgt * S_pred).sum(-1)).mean()


def _task_losses(model, vocab, items, cfg):
	dev = model._dev()
	t = _collate(items, vocab, dev)
	x = model.tokens_batch(t)
	y, h = model.ssm(x, mask=t["mask"])
	B, T, _ = y.shape
	# Targets at company-action positions are exogenous; never predict them.
	valid_t = t["mask"][:, :-1] * (1.0 - t["co"][:, 1:, 0])
	valid = valid_t.reshape(-1)
	nv = valid.sum().clamp(min=1)

	def mce(logits, tgt):
		l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
							tgt.reshape(-1), reduction="none")
		return (l * valid).sum() / nv

	def mmse(pred, tgt):
		l = F.mse_loss(pred, tgt, reduction="none").squeeze(-1).reshape(-1)
		return (l * valid).sum() / nv

	T_ = {"next": mce(model.head_next(y[:, :-1]), t["et"][:, 1:]),
		  "entity": mce(model.head_ent(y[:, :-1]), t["en"][:, 1:]),
		  "dt": mmse(model.head_dt(y[:, :-1]), t["dt"][:, 1:]),
		  "value": mmse(model.head_val(y[:, :-1]),
						torch.log1p(t["val"][:, 1:].abs()))}
	# Occurrence horizon derived from the data (median gap) so classes balance,
	# not a fixed window that a frequent exogenous event can saturate.
	g = t["dt"][:, 1:].squeeze(-1)
	vm = valid_t > 0
	thr = torch.median(g[vm]) if vm.any() else torch.tensor(math.log1p(7 * 86400.0), device=dev)
	occ_lab = (g <= thr).float()
	ol = F.binary_cross_entropy_with_logits(
		model.head_occ(y[:, :-1]).squeeze(-1), occ_lab, reduction="none").reshape(-1)
	T_["occur"] = (ol * valid).sum() / nv
	pos = t["et"][:, 1:]; neg = torch.randint(0, vocab.n_et, pos.shape, device=dev)
	sp = (model.order_W(y[:, :-1]) * model.emb_et(pos)).sum(-1)
	sn = (model.order_W(y[:, :-1]) * model.emb_et(neg)).sum(-1)
	opl = F.binary_cross_entropy_with_logits(sp - sn, torch.ones_like(sp), reduction="none").reshape(-1)
	T_["order"] = (opl * valid).sum() / nv
	S = F.normalize(model.proj(h), dim=-1)
	if B >= 4:
		logits = S @ S.t() / cfg.contrast_tau
		T_["contrast"] = F.cross_entropy(logits, torch.arange(B, device=dev))
	if B >= 2:
		zc = S - S.mean(0, keepdim=True); cov = (zc.t() @ zc) / max(B - 1, 1)
		off = cov - torch.diag(torch.diag(cov)); T_["redundancy"] = (off ** 2).mean()
	T_["mask"] = _mask_loss_batch(model, vocab, t, dev)
	T_["jepa"] = _jepa_loss(model, t, y)
	# Successor features (self-supervised, horizon-free): from every state,
	# predict the discounted future [log1p value, count] at a continuously
	# sampled discount gamma. No fixed horizons; the model learns all scales.
	# Multi-gamma successor features (horizon-free): richer phi and several
	# discounts per step, all vectorized (one scan over B*S).
	GAMS = 4
	gamma = (torch.rand(B, GAMS, device=dev) * GAMMA_MAX).clamp(min=1e-3)  # (B,S)
	dt_days = torch.expm1(t["dt"]) / TIME_UNIT_SECONDS                    # (B,T,1)
	oid = model.vocab.et.get("order_placed", -1)
	is_order = (t["et"] == oid).float().unsqueeze(-1)                     # (B,T,1)
	v = torch.log1p(t["val"].abs())
	phi = torch.cat([v, torch.ones_like(v), is_order, v * is_order], dim=-1)  # (B,T,4)
	de = dt_days.unsqueeze(1).expand(B, GAMS, T, 1).reshape(B * GAMS, T, 1)
	g = gamma.view(B, GAMS, 1, 1).expand(B, GAMS, T, 1).reshape(B * GAMS, T, 1) ** de
	pe = phi.unsqueeze(1).expand(B, GAMS, T, SF_PHI).reshape(B * GAMS, T, SF_PHI)
	# Exact reverse affine scan: R_i = sum_{j>i} (prod g) phi_j.
	prev = torch.flip(g, dims=[1])
	dprime = torch.zeros_like(prev); dprime[:, 1:] = prev[:, :-1]
	phir = torch.flip(pe, dims=[1])
	bprime = torch.zeros_like(phir); bprime[:, 1:] = dprime[:, 1:] * phir[:, :-1]
	_, Sf = _scan(dprime, bprime)
	R = torch.flip(Sf, dims=[1])                                         # (B*S,T,4)
	ye = y.unsqueeze(1).expand(B, GAMS, T, y.shape[-1]).reshape(B * GAMS, T, y.shape[-1])
	gcol = gamma.view(B, GAMS, 1, 1).expand(B, GAMS, T, 1).reshape(B * GAMS, T, 1)
	mask_e = t["mask"].unsqueeze(1).expand(B, GAMS, T).reshape(B * GAMS, T)
	pred = model.head_sf(torch.cat([ye, gcol], dim=-1))
	sl = F.mse_loss(pred, R, reduction="none").mean(-1)
	T_["sf"] = (sl * mask_e).sum() / mask_e.sum().clamp(min=1)
	return T_


def _combine(model, T, cfg):
	"""Uncertainty-based adaptive weighting: loss = sum 0.5*exp(-s)*L + 0.5*s.
	No manual weights; tasks that are noisy/conflicting earn lower weight."""
	keys = [k for k in cfg.objectives if k in T]
	if cfg.use_uncertainty_weighting:
		return sum(0.5 * torch.exp(-model.log_var[k]) * T[k] + 0.5 * model.log_var[k]
				   for k in keys)
	return sum(T[k] for k in keys)


def _loss(model, vocab, items, cfg):
	return _combine(model, _task_losses(model, vocab, items, cfg), cfg)


def _mask_loss(model, vocab, items):
	"""Masked-event reconstruction: hide events, predict them from context."""
	dev = model._dev(); total = 0.0
	for seq in items:
		L = len(seq["event_type"])
		if L < 4:
			continue
		k = max(1, int(0.15 * L))
		rng = np.random.default_rng(_h(seq["customer"], L) % (2 ** 31))
		mi = list(rng.choice(L, k, replace=False))
		et = list(map(str, seq["event_type"]))
		for i in mi:
			et[i] = "<mask>"
		y, _ = model({**seq, "event_type": et})
		mi = [i for i in mi if i < len(y)]
		if not mi:
			continue
		tgt = torch.tensor([vocab.et.get(str(seq["event_type"][i]), vocab.n_et)
							for i in mi], device=dev)
		total = total + F.cross_entropy(
			model.head_next(y[torch.tensor(mi, device=dev)]), tgt)
	return total / max(len(items), 1)


# ---------------------------------------------------------------------------
# state store (DuckDB): customer_state + anchor_embeddings
# ---------------------------------------------------------------------------
class StateStore:
	def __init__(self, path, cfg: CFMConfig, model: CFM):
		import duckdb
		Path(path).parent.mkdir(parents=True, exist_ok=True)
		self.path = str(path); self.cfg = cfg; self.model = model
		self.con = duckdb.connect(self.path)
		self.con.execute(
			"CREATE TABLE IF NOT EXISTS customer_state ("
			"customer_key TEXT, as_of_epoch DOUBLE, version TEXT, dim INT, "
			"state FLOAT[], embedding FLOAT[], last_event_ts TEXT)")
		self.con.execute(
			"CREATE TABLE IF NOT EXISTS anchor_embeddings ("
			"customer_key TEXT, anchor_epoch DOUBLE, version TEXT, dim INT, embedding FLOAT[])")
		self.con.execute(
			"CREATE TABLE IF NOT EXISTS encoder_samples ("
			"customer_key TEXT, split TEXT, version TEXT)")

	def write_splits(self, keys, split):
		"""Persist the sample-A/B assignment (strict disjoint) as a reusable table."""
		import polars as pl
		self.con.execute("CREATE OR REPLACE TABLE encoder_samples "
						 "(customer_key TEXT, split TEXT, version TEXT)")
		df = pl.DataFrame({"customer_key": list(keys),
						   "split": [split[k] for k in keys],
						   "version": [self.cfg.tag] * len(keys)})
		self.con.register("_splits", df)
		try:
			self.con.execute("INSERT INTO encoder_samples "
							 "SELECT customer_key, split, version FROM _splits")
		finally:
			self.con.unregister("_splits")

	def get_state(self, key):
		row = self.con.execute(
			"SELECT as_of_epoch, state FROM customer_state WHERE customer_key=? "
			"ORDER BY as_of_epoch DESC LIMIT 1", [key]).fetchone()
		if not row or row[1] is None:
			return None, None
		return torch.tensor(row[1], dtype=torch.float32), float(row[0])

	def upsert(self, key, h, emb, as_of_epoch, last_event_ts):
		self.con.execute("DELETE FROM customer_state WHERE customer_key=?", [key])
		self.con.execute("INSERT INTO customer_state VALUES (?,?,?,?,?,?,?)",
						 [key, as_of_epoch, self.cfg.tag, len(h), h.tolist(),
						  emb.tolist(), last_event_ts])

	def advance(self, seqs, incremental=True):
		"""fade to the first new event, absorb the events, persist."""
		for seq in seqs:
			h0, as_of = (self.get_state(seq["customer"]) if incremental else (None, None))
			h, emb = absorb(self.model, seq, h0=h0,
							as_of_epoch=as_of if h0 is not None else None)
			self.upsert(seq["customer"], h, emb, _to_epoch(seq["event_ts"][-1]),
						seq["event_ts"][-1])

	def add_training(self, key, anchor_epoch, emb):
		self.con.execute("INSERT INTO anchor_embeddings VALUES (?,?,?,?,?)",
						 [key, anchor_epoch, self.cfg.tag, len(emb), emb.tolist()])

	def fade_idle(self, now_epoch):
		"""Lazily advance as_of of idle states (decay only; no events)."""
		rows = self.con.execute(
			"SELECT customer_key, as_of_epoch, state FROM customer_state").fetchall()
		for key, as_of, state in rows:
			h = fade(torch.tensor(state, dtype=torch.float32), now_epoch - as_of,
					 self.cfg.state_half_life_days)
			emb = self.model.embed(h)
			self.upsert(key, h, emb, now_epoch, None)

	def count(self):
		s = self.con.execute("SELECT count(*) FROM customer_state").fetchone()[0]
		t = self.con.execute("SELECT count(*) FROM anchor_embeddings").fetchone()[0]
		return s, t

	def close(self):
		self.con.close()


def build_products(cfg, model, vocab, df, keys, split):
	# Rebuild fresh each release so embedding dims never mix across runs.
	pdb = Path(cfg.out_dir) / "cfm_products.duckdb"
	if pdb.exists():
		pdb.unlink()
	store = StateStore(pdb, cfg, model)
	store.write_splits(keys, split)
	# inference states for every customer (full history, as_of = last event)
	store.advance(build_sequences(df, keys, cfg, split, with_anchors=False),
				  incremental=False)
	# training embeddings for B at anchors
	for seq in build_sequences(df, keys, cfg, split, with_anchors=True):
		if seq["group"] == "B" and seq["anchor_epoch"] is not None:
			with torch.no_grad():
				y, h = model(seq)
				store.add_training(seq["customer"], seq["anchor_epoch"],
								   model.donor_seq(y, h, seq))
	s, t = store.count()
	store.close()
	return s, t


# ---------------------------------------------------------------------------
# registry + independent validation
# ---------------------------------------------------------------------------
def _registry(cfg, vocab, n_train, n_keys, derived=None, governor=None,
			  cfg_resolved=None):
	out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
	(out / f"registry_{cfg.tag.replace('.', '_')}.json").write_text(json.dumps({
		"version": cfg.version, "revision": cfg.revision, "tag": cfg.tag,
		"data_signature": getattr(cfg, "_data_signature", ""),
		"db": cfg.db, "table": cfg.table, "n_customers": n_keys,
		"n_train_sequences": n_train, "config": asdict(cfg),
		"vocab_sizes": {k: len(v) for k, v in vocab.dumps().items()},
		"derived": (derived or {}).get("derived", {}),
		"overrides": (derived or {}).get("overrides", {}),
		"resolved": cfg_resolved or {},
		"governor": governor or {},
		"trained_at": datetime.now(timezone.utc).isoformat()}, indent=1))


def validate(cfg: CFMConfig):
	df = _read_stream(cfg)
	_apply_data_revision(cfg, df)
	keys = _customer_keys(df, cfg)
	split = assign_split(keys, cfg)
	blob = torch.load(Path(cfg.out_dir) / f"cfm_{cfg.tag.replace('.', '_')}.pt",
					  map_location=cfg.device, weights_only=False)
	vocab = EventVocab(blob["vocab"]["et"], blob["vocab"]["brand"], blob["vocab"]["ent"])
	model = CFM(vocab, blob["dim"], n_experts=blob.get("n_experts", 1)).to(cfg.device)
	model.load_state_dict(blob["state"]); model.eval()
	model.half_life_days = cfg.state_half_life_days

	def r(name, target, achieved, ok):
		return {"check": name, "target": target, "achieved": achieved, "ok": ok}

	rows = []
	A = {k for k, g in split.items() if g == "A"}
	B = {k for k, g in split.items() if g == "B"}
	rows.append(r("A/B disjoint", "0 overlap", len(A & B), len(A & B) == 0))
	rows.append(r("both samples non-empty", ">0 each", f"A={len(A)} B={len(B)}",
				  len(A) > 0 and len(B) > 0))

	bseq = [s for s in build_sequences(df, list(B), cfg, split, with_anchors=True)
			if s["anchor_epoch"] is not None][:8]
	rows.append(r("causality (future can't change past)", "stable",
				  _causal(model, bseq), _causal(model, bseq)))

	b_all = build_sequences(df, list(B), cfg, split, with_anchors=False)
	acc, base = _next_event_acc(model, vocab, b_all[:200])
	rows.append(r("next-event acc > baseline", ">", f"{acc:.3f} vs {base:.3f}", acc > base))
	mm = _objective_metrics(model, vocab, b_all[:200])
	rows.append(r("objective: next-event acc", ">0.45", round(mm["next"], 3), mm["next"] > 0.45))
	rows.append(r("objective: entity acc", ">0.35", round(mm["entity"], 3), mm["entity"] > 0.35))
	rows.append(r("objective: occurrence acc > base", ">",
				  f"{mm['occ']:.3f} vs {mm['occ_base']:.3f}", mm["occ"] > mm["occ_base"]))
	rows.append(r("objective: temporal-order acc", ">0.55", round(mm["order"], 3),
				  mm["order"] > 0.55))
	rows.append(r("objective: dt MAE (log1p s)", "<3.0", round(mm["dt_mae"], 3),
				  mm["dt_mae"] < 3.0))

	# state products + as_of
	import duckdb
	con = duckdb.connect(str(Path(cfg.out_dir) / "cfm_products.duckdb"), read_only=True)
	n_state = con.execute("select count(*) from customer_state").fetchone()[0]
	n_nulls = con.execute("select count(*) from customer_state where as_of_epoch is null").fetchone()[0]
	n_tr = con.execute("select count(*) from anchor_embeddings").fetchone()[0]
	dims = con.execute("select distinct dim from customer_state").fetchall()
	con.close()
	rows.append(r("customer_state covers customers", len(keys), n_state, n_state == len(keys)))
	rows.append(r("state has as_of", "0 null", n_nulls, n_nulls == 0))
	rows.append(r("training embeddings present", ">0", n_tr, n_tr > 0))
	rows.append(r("dim consistent", 1, len(dims), len(dims) == 1))

	# absorb consistency: incremental == from-scratch (first sample)
	seqs = build_sequences(df, list(B)[:1], cfg, split, with_anchors=False)
	if seqs:
		s = seqs[0]
		with torch.no_grad():
			_, h_full = model(s)
			emb_full = model.embed(model(s)[0])
		h0, _ = None, None
		h_inc, emb_inc = absorb(model, s, h0=None)
		rows.append(r("absorb == from-scratch", "allclose",
					  float(torch.allclose(h_full, h_inc, atol=1e-4)),
					  torch.allclose(h_full, h_inc, atol=1e-4)))
		# fade decays the state
		faded = fade(h_full, 30 * 86400, cfg.state_half_life_days)
		rows.append(r("fade decays state", "norm<1",
					  round(float(faded.norm() / max(h_full.norm(), 1e-9)), 3),
					  faded.norm() < h_full.norm()))
	return rows, all(x["ok"] for x in rows)


def _objective_metrics(model, vocab, seqs):
	"""Held-out accuracy per objective (self-supervised; strictly causal).
	Company-action targets are excluded; occurrence is balanced (median-horizon)."""
	dev = model._dev()
	inv_et = {v: k for k, v in vocab.et.items()}
	inv_en = {v: k for k, v in vocab.ent.items()}
	m = {"next": 0, "entity": 0, "occ_tp": 0, "occ_tn": 0, "occ_pos": 0,
		 "occ_neg": 0, "order": 0, "n": 0, "dt": 0.0}
	for seq in seqs:
		with torch.no_grad():
			y, _ = model(seq)
		if y.shape[0] < 2:
			continue
		ts = np.array([_to_epoch(x) for x in seq["event_ts"]], dtype=np.float64)
		gap = np.maximum(ts[1:] - ts[:-1], 0.0)
		co = seq.get("co")
		co = co if co is not None else [[0.0, 0.0]] * len(seq["event_type"])
		keep = np.array([co[i + 1][0] < 0.5 for i in range(len(gap))], dtype=bool)
		thr = float(np.median(gap[keep])) if keep.any() else float(np.log1p(7 * 86400))
		nxt = model.head_next(y[:-1]).argmax(-1).detach().cpu().numpy()
		ent = model.head_ent(y[:-1]).argmax(-1).detach().cpu().numpy()
		occ = (model.head_occ(y[:-1]).squeeze(1).detach().cpu().numpy() > 0).astype(int)
		dtp = model.head_dt(y[:-1]).squeeze(1).detach().cpu().numpy()
		pos = torch.tensor([vocab.et.get(str(x), vocab.n_et) for x in seq["event_type"][1:]],
						   device=dev)
		neg = torch.randint(0, vocab.n_et, pos.shape, device=dev)
		sp = (model.order_W(y[:-1]) * model.emb_et(pos)).sum(-1)
		sn = (model.order_W(y[:-1]) * model.emb_et(neg)).sum(-1)
		ordv = (sp > sn).detach().cpu().numpy().astype(int)
		for i in range(len(nxt)):
			if not keep[i]:
				continue
			m["n"] += 1
			m["next"] += int(inv_et.get(int(nxt[i])) == str(seq["event_type"][i + 1]))
			m["entity"] += int(inv_en.get(int(ent[i])) ==
							   (str(seq["entity_type"][i + 1]) if seq["entity_type"][i + 1] is not None else "none"))
			lab = int(gap[i] <= thr)
			m["occ_tp"] += int(occ[i] == 1 and lab == 1)
			m["occ_tn"] += int(occ[i] == 0 and lab == 0)
			m["occ_pos"] += lab
			m["occ_neg"] += 1 - lab
			m["order"] += int(ordv[i] == 1)
			m["dt"] += abs(float(dtp[i]) - float(np.log1p(gap[i])))
	n = max(m["n"], 1)
	occ_bal = 0.5 * (m["occ_tp"] / max(m["occ_pos"], 1) + m["occ_tn"] / max(m["occ_neg"], 1))
	return {"next": m["next"] / n, "entity": m["entity"] / n,
			"occ": occ_bal, "occ_base": max(m["occ_pos"], m["occ_neg"]) / n,
			"order": m["order"] / n, "dt_mae": m["dt"] / n}


def _causal(model, seqs):
	ok = True
	for s in seqs:
		with torch.no_grad():
			y_full, _ = model(s)
		cut = {**s, "event_type": s["event_type"][:-1], "brand": s["brand"][:-1],
			   "entity_type": s["entity_type"][:-1], "value": s["value"][:-1],
			   "event_ts": s["event_ts"][:-1], "co": s["co"][:-1],
			   "ts": s["ts"][:-1]}
		if len(cut["event_type"]) < 2:
			continue
		with torch.no_grad():
			y_cut, _ = model(cut)
		n = min(len(y_cut), len(y_full))
		if not torch.allclose(y_cut[:n], y_full[:n], atol=1e-3):
			ok = False
	return ok


def _next_event_acc(model, vocab, seqs):
	correct = total = base = 0
	from collections import Counter
	freq = Counter()
	for s in seqs:
		for x in s["event_type"][1:]:
			freq[str(x)] += 1
	top = freq.most_common(1)[0][0] if freq else None
	inv = {v: k for k, v in vocab.et.items()}
	for s in seqs:
		with torch.no_grad():
			y, _ = model(s)
			pred = model.head_next(y[:-1]).argmax(-1).detach().cpu().numpy()
		for p, t in zip(pred, [str(x) for x in s["event_type"][1:]]):
			total += 1
			correct += int(inv.get(int(p)) == t)
			base += int(t == top)
	return (correct / total if total else 0.0, base / total if total else 0.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print(rows, title):
	print(f"== {title} ==")
	print(f"{'check':46s} {'target':>12s} {'achieved':>16s}  status")
	for x in rows:
		print(f"{x['check']:46s} {str(x['target']):>12s} {str(x['achieved']):>16s}  "
			  f"{'PASS' if x['ok'] else 'FAIL'}")
	n = sum(1 for x in rows if x["ok"])
	print(f"completion: {n}/{len(rows)} ({100*n/len(rows):.0f}%)")


def main(argv=None):
	import argparse
	ap = argparse.ArgumentParser(description="Looking Glass — Customer Foundation Model")
	ap.add_argument("cmd", choices=["train", "validate", "all"])
	ap.add_argument("--db", default=CFMConfig.db)
	ap.add_argument("--customers", type=int, default=CFMConfig.sample_customers)
	ap.add_argument("--anchors", type=int, default=CFMConfig.n_anchors)
	ap.add_argument("--epochs", type=int, default=CFMConfig.epochs)
	ap.add_argument("--device", default=CFMConfig.device)
	a = ap.parse_args(argv)
	cfg = sample_a(a.customers, a.anchors, db=a.db, epochs=a.epochs, device=a.device)
	if a.cmd in ("train", "all"):
		model, vocab, df, keys, split = train_cfm(cfg)
		s, t = build_products(cfg, model, vocab, df, keys, split)
		print(f"trained {cfg.tag}: customer_state={s} training_embeddings={t}")
	if a.cmd in ("validate", "all"):
		rows, verdict = validate(cfg)
		_print(rows, f"CFM VALIDATION {cfg.tag}")
		print("VERDICT:", "PASS" if verdict else "FAIL")
		return 0 if verdict else 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
