"""Relational REFERENCE generator: normal business tables -> one customer event stream.

Why this exists (DEC-236): the flat reference was additive/measure-like, so a wide table was
sufficient by construction and could not test a sequence encoder. This generator builds the
tables a real business has — customers, products, orders, order_items, events, campaigns,
returns, support — with latent regimes, path dependence, interactions, long memory, and
missing logs, then unifies them into the canonical 4-column event stream. It also emits the
traditional wide customer table (the baseline), so the same source data supports:
    table + GBM   vs   raw events + sequence encoder   vs   E + table.

Writes:
  <out>/customer_event_stream.parquet     canonical stream (customer_key, event_ts,
                                          event_type, event_attributes, ...)
  <out>/customer_wide_table.parquet       traditional per-customer aggregates
  <out>/relational/customers.parquet      customers, products, orders, order_items, events,
                                          campaigns, returns, support
  <out>/relational_truth.json             known-truth receipt (latent optima/regimes)
  <out>/reference_truth.json              alias for the certification pipeline
  <out>/runtime_receipt.json              device + wall-clock receipt

Speed-first / GPU-first (AGENTS.md §16, operator GPU policy): the core simulation is
VECTORIZED across (customers x days) with torch on CUDA when available (numpy fallback),
simulating ALL customers simultaneously as arrays; PATH A aggregation runs on the
polars-CUDF GPU engine through the dedicated `.venv-cudf` process boundary, falling back to
CPU polars with the reason recorded.

Sequence dependence (DEC-283/284): the label/future signal lives in the ORDER/TIMING of
events so a bag-of-events / aggregate table cannot capture it:
  1. next organic event is a variable-order (2nd-order) Markov transition over the recent k-gram;
  2. future purchase/value/churn depend on the recent sequence SHAPE + inter-event GAPS;
  3. self-excitation is ORDER-GATED (triggered only by specific hot k-grams), never count-based;
  4. email_send actions affect the outcome with a DELAY conditioned on the sequence;
  5. all existing latents/truth and PATH A columns are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rabbit_hole.generators.path_a_build import build_path_a  # noqa: E402

CATEGORIES = ["apparel", "footwear", "accessories", "beauty", "home", "outdoor"]
BRANDS = [f"brand_{i}" for i in range(10)]
CHANNELS = ["email", "sms", "push", "direct_mail"]
TOUCH = ["web", "app", "store"]
REGIMES = ["browse", "intent", "loyal", "at_risk"]
REGIONS = ["north", "south", "east", "west"]
LOYALTY = ["none", "silver", "gold", "platinum"]
ACQ = ["organic", "paid_search", "social", "referral"]
ORG_EVENTS = ["view", "search", "add_to_cart", "remove_from_cart", "support_ticket", "review"]
ACTS = ("freq", "hour", "dow", "channel", "offer")

_BASE_COLS = [
    "customer_key",
    "event_ts",
    "event_type",
    "product_id",
    "category_code",
    "brand",
    "channel",
    "campaign_id",
    "price",
    "qty",
    "discount",
    "value",
    "margin",
    "session_id",
    "attrs_json",
]


@dataclass
class RelationalSpec:
    n_customers: int = 2000
    window_days: int = 540
    seed: int = 17
    snr: float = 1.0
    start_date: str = "2024-01-01"
    as_of: str = "2025-01-01"
    missing_rate: float = 0.15
    n_products: int = 80
    base_activity: float = 1.2
    base_purchase: float = 0.065


def _clip01_scalar(x):
    return float(np.clip(x, 0.0, 1.0))


def _cuda_ok():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


class _TensorBackend:
    """Minimal tensor ops: torch on CUDA when available, else numpy."""

    def __init__(self, seed, use_cuda):
        self.use_cuda = bool(use_cuda)
        if self.use_cuda:
            import torch

            self.t = torch
            self.dev = torch.device("cuda")
            self.gen = torch.Generator(device=self.dev)
            self.gen.manual_seed(int(seed) & 0x7FFFFFFF)
            self.name = "torch-cuda"
        else:
            import numpy as np

            self.np = np
            self.gen = np.random.default_rng(int(seed))
            self.name = "numpy-cpu"

    def _dt(self, kind):
        if self.use_cuda:
            return {
                "f32": self.t.float32,
                "f64": self.t.float64,
                "i64": self.t.int64,
                "bool": self.t.bool,
            }[kind]
        return {
            "f32": self.np.float32,
            "f64": self.np.float64,
            "i64": self.np.int64,
            "bool": self.np.bool_,
        }[kind]

    def zeros(self, shape, kind="f32"):
        if self.use_cuda:
            return self.t.zeros(shape, dtype=self._dt(kind), device=self.dev)
        return self.np.zeros(shape, dtype=self._dt(kind))

    def full(self, shape, val, kind="f32"):
        if self.use_cuda:
            return self.t.full(shape, val, dtype=self._dt(kind), device=self.dev)
        return self.np.full(shape, val, dtype=self._dt(kind))

    def arange(self, n):
        if self.use_cuda:
            return self.t.arange(n, dtype=self.t.int64, device=self.dev)
        return self.np.arange(n, dtype=self.np.int64)

    def rand(self, shape):
        if self.use_cuda:
            return self.t.rand(shape, generator=self.gen, device=self.dev, dtype=self.t.float32)
        return self.gen.random(shape).astype(self.np.float32)

    def to_dev(self, arr, kind=None):
        a = np.asarray(arr)
        if self.use_cuda:
            t = self.t.as_tensor(a, device=self.dev)
            return t if kind is None else t.to(self._dt(kind))
        return a if kind is None else a.astype(self._dt(kind))

    def to_numpy(self, x):
        if self.use_cuda:
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def to_f32(self, x):
        return x.to(self.t.float32) if self.use_cuda else x.astype(self.np.float32)

    def to_i64(self, x):
        return x.to(self.t.int64) if self.use_cuda else x.astype(self.np.int64)

    def where(self, c, a, b):
        return self.t.where(c, a, b) if self.use_cuda else self.np.where(c, a, b)

    def clip(self, x, lo, hi):
        return self.t.clamp(x, lo, hi) if self.use_cuda else self.np.clip(x, lo, hi)

    def exp(self, x):
        return self.t.exp(x) if self.use_cuda else self.np.exp(x)

    def abs(self, x):
        return self.t.abs(x) if self.use_cuda else self.np.abs(x)

    def round_(self, x, decimals=0):
        return self.t.round(x, decimals=decimals) if self.use_cuda else self.np.round(x, decimals)

    def floor_i(self, x):
        if self.use_cuda:
            return self.t.floor(x).to(self.t.int64)
        return self.np.floor(x).astype(self.np.int64)

    def maximum(self, a, b):
        if not self.use_cuda:
            return self.np.maximum(a, b)
        a_t, b_t = self.t.is_tensor(a), self.t.is_tensor(b)
        if a_t and b_t:
            return self.t.maximum(a, b)
        if a_t:
            return self.t.clamp(a, min=b)
        return self.t.clamp(b, max=a)

    def minimum(self, a, b):
        if not self.use_cuda:
            return self.np.minimum(a, b)
        a_t, b_t = self.t.is_tensor(a), self.t.is_tensor(b)
        if a_t and b_t:
            return self.t.minimum(a, b)
        if a_t:
            return self.t.clamp(a, max=b)
        return self.t.clamp(b, max=a)

    def sum(self, x, dim=-1, keepdim=False):
        if self.use_cuda:
            return x.sum(dim=dim, keepdim=keepdim)
        return x.sum(axis=dim, keepdims=keepdim)

    def amax(self, x, dim=-1):
        return self.t.amax(x, dim=dim) if self.use_cuda else x.max(axis=dim)

    def cumsum(self, x, dim=-1):
        return self.t.cumsum(x, dim=dim) if self.use_cuda else self.np.cumsum(x, axis=dim)

    def cat_probs(self, probs, u):
        cdf = self.cumsum(probs, -1)
        if self.use_cuda:
            idx = (cdf < u[:, None]).sum(dim=-1)
            return self.t.clamp(idx, max=probs.shape[-1] - 1)
        idx = (cdf < u[:, None]).sum(axis=-1).astype(self.np.int64)
        return self.np.minimum(idx, probs.shape[-1] - 1)

    def cat_shared(self, cdf_row, u):
        if self.use_cuda:
            idx = (cdf_row[None, :] < u[:, None]).sum(dim=-1)
            return self.t.clamp(idx, max=cdf_row.shape[-1] - 1)
        idx = (cdf_row[None, :] < u[:, None]).sum(axis=-1).astype(self.np.int64)
        return self.np.minimum(idx, cdf_row.shape[-1] - 1)

    def nonzero2d(self, mask):
        if self.use_cuda:
            return mask.nonzero(as_tuple=False).cpu().numpy()
        return self.np.argwhere(mask)

    def masked(self, field, mask):
        return self.to_numpy(field[mask])

    def mem_info(self):
        if not self.use_cuda:
            return {}
        free, total = self.t.cuda.mem_get_info()
        return {
            "mem_free_bytes": int(free),
            "mem_total_bytes": int(total),
            "max_memory_allocated_bytes": int(self.t.cuda.max_memory_allocated()),
            "torch_version": self.t.__version__,
            "device_name": self.t.cuda.get_device_name(0),
        }


def _iso(start64, day_float):
    ns = np.rint(np.asarray(day_float, np.float64) * 86400.0 * 1e9).astype(np.int64)
    dt = start64 + ns.astype("timedelta64[ns]")
    return np.datetime_as_string(dt.astype("datetime64[us]"), unit="us")


def _regime_runs_np(reg_by_day, regime_names):
    T, C = reg_by_day.shape
    change = np.empty(reg_by_day.shape, dtype=bool)
    change[0] = True
    if T > 1:
        change[1:] = reg_by_day[1:] != reg_by_day[:-1]
    run_id = np.cumsum(change, axis=0) - 1
    nrun = int(run_id.max()) + 1
    keys = np.arange(C, dtype=np.int64)[None, :] * nrun + run_id
    flat_keys = keys.ravel()
    counts = np.bincount(flat_keys, minlength=C * nrun)
    st = np.flatnonzero(change.ravel())
    st_keys = flat_keys[st]
    st_reg = reg_by_day.ravel()[st]
    order = np.argsort(st_keys, kind="stable")
    st_keys = st_keys[order]
    st_reg = st_reg[order]
    custs = (st_keys // nrun).astype(np.int64).tolist()
    regs = st_reg.tolist()
    ks = st_keys.tolist()
    out = [[] for _ in range(C)]
    for i in range(len(ks)):
        out[custs[i]].append([regime_names[regs[i]], int(counts[ks[i]])])
    return out


def _camp_strs(camp_idx, camp_str):
    out = np.full(len(camp_idx), None, dtype=object)
    m = camp_idx >= 0
    out[m] = camp_str[camp_idx[m]]
    return out.tolist()


def simulate(spec: RelationalSpec, use_cuda: bool = True):
    """Vectorized (customers x days) simulation with optional CUDA."""
    t_sim = time.perf_counter()
    C = int(spec.n_customers)
    T = int(spec.window_days)
    s = float(spec.snr)
    P = int(spec.n_products)
    start64 = np.datetime64(spec.start_date).astype("datetime64[ns]")
    cust_keys = np.array([f"rc{i:07d}" for i in range(C)])

    B = _TensorBackend(spec.seed * 2654435761 + 13, use_cuda and _cuda_ok())

    # ---- latents (vectorized over customers; numpy; not a hot path) --------
    lrng = np.random.default_rng(spec.seed)
    prod_cat = lrng.integers(0, len(CATEGORIES), P).astype(np.int64)
    prod_brand = lrng.integers(0, len(BRANDS), P).astype(np.int64)
    prod_price = np.round(np.exp(lrng.normal(np.log(55.0), 0.55, P)), 2)
    prod_margin = np.clip(lrng.normal(0.38, 0.10, P), 0.10, 0.65)
    prod_pop = lrng.dirichlet(np.full(P, 0.8))

    n_camp = 24
    camp_disc = np.round(lrng.uniform(0.05, 0.35, n_camp), 2)
    camp_start = lrng.integers(0, T, n_camp)
    camp_dur = lrng.integers(5, 30, n_camp)

    cust_cat = lrng.dirichlet(np.full(len(CATEGORIES), 0.7), size=C)
    price_sens = lrng.beta(2.0, 2.5, C)
    chan_pref = lrng.integers(0, len(CHANNELS), C).astype(np.int64)
    touch_pref = lrng.dirichlet(np.array([2.0, 1.5, 1.0]), size=C)
    loyal = lrng.integers(0, len(LOYALTY), C).astype(np.int64)
    acq = lrng.integers(0, len(ACQ), C).astype(np.int64)
    segment = lrng.integers(0, 5, C).astype(np.int64)
    return_prone = np.clip(lrng.beta(1.5, 6.0, C), 0.0, 1.0)
    churn_day = np.where(lrng.random(C) < 0.18, lrng.uniform(0.45 * T, T, C), float(T)).astype(
        np.float32
    )
    x0 = lrng.uniform(0, 3, C).astype(np.float32)
    h0 = lrng.uniform(0, 24, C).astype(np.float32)
    d0 = lrng.uniform(0, 7, C).astype(np.float32)
    offer_opt = lrng.integers(0, 4, C).astype(np.int64)
    propensity = np.exp(lrng.normal(0.0, float(os.environ.get("WQ_PROP_SD", "0.40")), C))
    propensity = propensity / propensity.mean()
    Pc = lrng.dirichlet(np.full(len(REGIMES), 1.2), size=(C, len(REGIMES)))
    Pc = 0.93 * np.eye(len(REGIMES))[None] + 0.07 * Pc
    regime0 = lrng.integers(0, len(REGIMES), C).astype(np.int64)
    shock_day = lrng.uniform(0.1 * T, 0.8 * T, C).astype(np.float32)
    shock_amp = lrng.uniform(0.3, 1.2, C).astype(np.float32)
    shock_tau = lrng.uniform(10, 45, C).astype(np.float32)
    region = lrng.integers(0, 4, C).astype(np.int64)
    signup_days = lrng.integers(1, 400, C).astype(np.int64)

    # ---- order model: variable-order (2nd-order) organic Markov chain ------
    ALPHABET = ["customer_signup", *ORG_EVENTS]
    nA = len(ALPHABET)
    ATC = ALPHABET.index("add_to_cart")
    ATC_ORG = ORG_EVENTS.index("add_to_cart")
    FUNNEL = np.array([ALPHABET.index(e) for e in ORG_EVENTS], dtype=np.int64)
    _rm = np.random.default_rng(spec.seed * 1000003 + 7)
    BASE = _rm.dirichlet(np.full(nA, 0.8))
    T1 = 0.30 * BASE[None, :] + 0.70 * _rm.dirichlet(np.full(nA, 0.22), size=nA)
    draw2 = _rm.dirichlet(np.full(nA, 0.10), size=nA * nA).reshape(nA, nA, nA)
    T2 = 0.15 * T1[None, :, :] + 0.85 * draw2
    DELAY2 = _rm.uniform(4.0, 22.0, size=(nA, nA))
    ACT_DELAY = _rm.uniform(1.0, 6.0, size=len(ACTS))
    ATC_MASS = T2[:, :, ATC]
    HOT = float(np.quantile(ATC_MASS, 0.80))
    PAT = np.exp(_rm.normal(0.0, 0.75, size=(nA, nA)))
    PAT = PAT / PAT.mean()
    PATH_LUT = np.ones(nA)
    PATH_LUT[ALPHABET.index("view")] = 1.15
    PATH_LUT[ATC] = 1.6
    REGIME_BUY = np.array([0.75, 1.55, 1.25, 0.45], dtype=np.float32)

    # all-event k-gram: the purchase hazard is ALSO gated by the last two EVENTS
    # (any type), so the label lives in the recent ORDER -- invisible to aggregates.
    ALL_EV = ["customer_signup", *ORG_EVENTS, "purchase", "return", "email_send"]
    nAll = len(ALL_EV)
    ORG_TO_ALL = np.array([ALL_EV.index(e) for e in ORG_EVENTS], dtype=np.int64)
    IDX_SEND = ALL_EV.index("email_send")
    IDX_BUY = ALL_EV.index("purchase")
    IDX_RET = ALL_EV.index("return")
    PAT2 = np.exp(_rm.normal(0.0, 0.75, size=(nAll, nAll)))
    PAT2 = np.clip(PAT2 / PAT2.mean(), 0.35, 2.8)

    TAU = 12.0
    G_ORD = int(np.ceil(float(DELAY2.max()) + 6.0 * TAU)) + 2
    G_ACT = int(np.ceil(float(ACT_DELAY.max()) + 6.0 * TAU)) + 2

    # device copies
    T2d = B.to_dev(T2, "f32")
    DELAY2d = B.to_dev(DELAY2, "f32")
    PATd = B.to_dev(PAT, "f32")
    PATH_LUTd = B.to_dev(PATH_LUT, "f32")
    ACT_DELAYd = B.to_dev(ACT_DELAY, "f32")
    FUNNELd = B.to_dev(FUNNEL, "i64")
    reg_cdf = B.to_dev(np.cumsum(Pc, axis=-1), "f32")
    touch_pref_d = B.to_dev(touch_pref, "f32")
    churn_day_d = B.to_dev(churn_day, "f32")
    x0d = B.to_dev(x0, "f32")
    h0d = B.to_dev(h0, "f32")
    d0d = B.to_dev(d0, "f32")
    chan_pref_d = B.to_dev(chan_pref, "i64")
    offer_opt_d = B.to_dev(offer_opt, "i64")
    regime0_d = B.to_dev(regime0, "i64")
    shock_day_d = B.to_dev(shock_day, "f32")
    shock_amp_d = B.to_dev(shock_amp, "f32")
    shock_tau_d = B.to_dev(shock_tau, "f32")
    return_prone_d = B.to_dev(return_prone, "f32")
    price_sens_d = B.to_dev(price_sens, "f32")
    prod_price_d = B.to_dev(prod_price, "f32")
    prod_margin_d = B.to_dev(prod_margin, "f32")
    prod_str = np.array([f"p{i:03d}" for i in range(P)])
    camp_str = np.array([f"cmp{i:03d}" for i in range(n_camp)])
    cat_str = np.array(CATEGORIES)
    brand_str = np.array(BRANDS)
    touch_str = np.array(TOUCH)
    chan_str = np.array(CHANNELS)
    ar = B.arange(C)

    W = cust_cat[:, prod_cat] * (0.5 + prod_pop)[None, :]
    W = W / W.sum(1, keepdims=True)
    Wd = B.to_dev(W, "f32")
    CCP = B.to_dev(cust_cat[:, prod_cat], "f32")
    PROD_POP2 = B.to_dev(np.broadcast_to(prod_pop, (C, P)).copy(), "f32")
    regime_act_lut = B.to_dev(1.0 + 0.6 * s * (REGIME_BUY - 1.0), "f32")
    propensity_d = B.to_dev(propensity, "f32")
    PAT2d = B.to_dev(PAT2, "f32")
    ORG_TO_ALLd = B.to_dev(ORG_TO_ALL, "i64")

    days = np.arange(T)
    act_mask = (camp_start[None, :] <= days[:, None]) & (
        days[:, None] < camp_start[None, :] + camp_dur[None, :]
    )
    camp_row = act_mask.astype(np.float64)
    tot = camp_row.sum(1, keepdims=True)
    camp_probs = np.where(tot > 0, camp_row / np.maximum(tot, 1.0), 0.0)
    CAMP_CDF = B.to_dev(np.cumsum(camp_probs, axis=1), "f32")
    CAMP_HAS = B.to_dev(act_mask.any(1).astype(bool), "bool")
    CAMP_DISC = B.to_dev(camp_disc, "f32")

    U = B.rand((T, C, 24))  # vectorized uniform pool; sliced per day

    send_mask = B.zeros((T, C), "bool")
    send_akey = B.zeros((T, C), "i64")
    send_freq = B.zeros((T, C), "i64")
    send_hour = B.zeros((T, C), "i64")
    send_dow = B.zeros((T, C), "i64")
    send_chan = B.zeros((T, C), "i64")
    send_off = B.zeros((T, C), "i64")
    send_camp = B.full((T, C), -1, "i64")
    buy_mask = B.zeros((T, C), "bool")
    buy_pid = B.zeros((T, C), "i64")
    buy_qty = B.zeros((T, C), "i64")
    buy_unit = B.zeros((T, C), "f32")
    buy_val = B.zeros((T, C), "f32")
    buy_margin = B.zeros((T, C), "f32")
    buy_disc = B.zeros((T, C), "f32")
    buy_touch = B.zeros((T, C), "i64")
    buy_camp = B.full((T, C), -1, "i64")
    buy_chan = B.zeros((T, C), "i64")
    buy_hour = B.zeros((T, C), "i64")
    org_mask = B.zeros((T, C), "bool")
    org_et = B.zeros((T, C), "i64")
    org_pid = B.zeros((T, C), "i64")
    org_touch = B.zeros((T, C), "i64")
    org_camp = B.full((T, C), -1, "i64")
    ret_mask = B.zeros((T, C), "bool")
    ret_dayv = B.zeros((T, C), "f32")
    ret_pid = B.zeros((T, C), "i64")
    ret_val = B.zeros((T, C), "f32")
    ret_reason = B.zeros((T, C), "i64")
    ret_src = B.zeros((T, C), "i64")

    regime = regime0_d
    o1 = B.full((C,), 0, "i64")
    o2 = B.full((C,), 0, "i64")
    a1 = B.zeros((C,), "i64")
    a2 = B.zeros((C,), "i64")
    refr_until = B.full((C,), -1e9, "f32")
    since_atc = B.full((C,), 10**6, "i64")
    reg_traj = B.zeros((T, C), "i64")

    or_t0 = B.full((C, G_ORD), -1e9, "f32")
    or_delay = B.full((C, G_ORD), 1.0, "f32")
    or_amp = B.zeros((C, G_ORD), "f32")
    ac_t0 = B.full((C, G_ACT), -1e9, "f32")
    ac_delay = B.full((C, G_ACT), 1.0, "f32")
    ac_amp = B.zeros((C, G_ACT), "f32")

    rday = B.zeros((C, 7), "f32")
    rval = B.zeros((C, 7), "f32")
    rpid = B.zeros((C, 7), "i64")
    rsrc_day = B.zeros((C, 7), "f32")
    rreason = B.zeros((C, 7), "i64")
    rmask = B.zeros((C, 7), "bool")

    freq_def = B.to_i64(B.clip(B.round_(x0d), 0, 3))
    hour_def = B.to_i64(B.clip(B.round_(h0d), 0, 23))
    dow_def = B.to_i64(B.clip(B.round_(d0d), 0, 6))
    zero_f = B.zeros((C,), "f32")

    def kernel(dd_float, t0c, delayc, ampc):
        age = dd_float - t0c
        delayc = B.maximum(delayc, 1e-6)
        b = B.where(age < delayc, ampc * (age / delayc), ampc * B.exp(-(age - delayc) / TAU))
        live = (age >= 0.0) & (age <= delayc + 6.0 * TAU)
        return B.where(ampc > 0.0, B.where(live, b, ampc * 0.0), ampc * 0.0)

    for d in range(T):
        u = U[d]
        fd = float(d)
        gate2 = PAT2d[a1, a2]  # all-event (prev2, prev1) order gate
        active = d < churn_day_d
        slot_r = d % 7
        slot_o = d % G_ORD
        slot_a = d % G_ACT

        # ---- due returns ----
        due = active & rmask[:, slot_r] & (B.floor_i(rday[:, slot_r]) == d)
        ret_mask[d] = due
        ret_dayv[d] = B.where(due, rday[:, slot_r], 0.0)
        ret_pid[d] = B.where(due, rpid[:, slot_r], 0)
        ret_val[d] = B.where(due, rval[:, slot_r], 0.0)
        ret_reason[d] = B.where(due, rreason[:, slot_r], 0)
        ret_src[d] = B.to_i64(B.where(due, rsrc_day[:, slot_r], 0.0))
        rmask[:, slot_r] = rmask[:, slot_r] & (~due)

        # ---- regime switch ----
        switch = (u[:, 0] < 0.035) & active
        typed = regime.to(B._dt("i64")) if B.use_cuda else regime
        cdf_row = reg_cdf[ar, typed]
        newreg = B.cat_probs(cdf_row, u[:, 1])
        regime = B.where(switch, newreg, typed)

        # ---- long-memory shock ----
        shock = B.where(
            d >= shock_day_d,
            shock_amp_d * B.exp(-(fd - shock_day_d) / B.maximum(shock_tau_d, 1.0)),
            zero_f,
        )

        # ---- order-gated, delayed self-excitation ----
        ob = B.maximum(
            B.amax(kernel(fd, or_t0, or_delay, or_amp), -1),
            B.amax(kernel(fd, ac_t0, ac_delay, ac_amp), -1),
        )

        # ---- campaign per day ----
        camp_choice = B.cat_shared(CAMP_CDF[d], u[:, 3])
        sent_camp = CAMP_HAS[d] & (u[:, 2] < 0.25)
        camp_idx = B.where(sent_camp, camp_choice, B.full((C,), -1, "i64"))
        camp_discount = B.where(sent_camp, CAMP_DISC[camp_choice], zero_f)

        # ---- one-factor daily action ----
        akey = B.floor_i(u[:, 4] * len(ACTS))
        ov = u[:, 5]
        freq_act = B.where(akey == 0, B.floor_i(ov * 4), freq_def)
        hour_act = B.where(akey == 1, B.floor_i(ov * 24), hour_def)
        dow_act = B.where(akey == 2, B.floor_i(ov * 7), dow_def)
        chan_act = B.where(akey == 3, B.floor_i(ov * len(CHANNELS)), chan_pref_d)
        off_act = B.where(akey == 4, B.floor_i(ov * 4), offer_opt_d)

        q = B.exp(-0.5 * ((B.to_f32(freq_act) - x0d) / 1.1) ** 2)
        q = q * B.exp(-0.5 * ((B.to_f32(hour_act) - h0d) / 3.0) ** 2)
        q = q * B.exp(-0.5 * ((B.to_f32(dow_act) - d0d) / 1.6) ** 2)
        q = q * B.where(
            chan_act == chan_pref_d, B.full((C,), 1.0, "f32"), B.full((C,), 0.35, "f32")
        )
        q = q * (1.0 - 0.22 * B.abs(B.to_f32(off_act) - B.to_f32(offer_opt_d)))
        q = B.clip(1.0 + s * (q - 1.0), 0.05, 1.0)

        p_send = B.minimum(0.95, 0.05 + 0.05 * B.to_f32(freq_act))
        sent = active & (u[:, 6] < p_send)

        send_mask[d] = sent
        send_akey[d] = B.where(sent, akey, 0)
        send_freq[d] = B.where(sent, freq_act, 0)
        send_hour[d] = B.where(sent, hour_act, 0)
        send_dow[d] = B.where(sent, dow_act, 0)
        send_chan[d] = B.where(sent, chan_act, 0)
        send_off[d] = B.where(sent, off_act, 0)
        send_camp[d] = B.where(sent, camp_idx, B.full((C,), -1, "i64"))
        a2 = B.where(sent, a1, a2)
        a1 = B.where(sent, B.full((C,), IDX_SEND, "i64"), a1)

        row = T2d[o1, o2]
        ready = row[:, ATC]

        # schedule DELAYED action trigger conditioned on the sequence
        ac_t0[:, slot_a] = B.where(sent, B.full((C,), fd, "f32"), ac_t0[:, slot_a])
        ac_delay[:, slot_a] = B.where(sent, ACT_DELAYd[akey], B.full((C,), 1.0, "f32"))
        ac_amp[:, slot_a] = B.where(sent, s * 0.12 * q * (0.35 + ready), zero_f)

        # ---- purchase (MODERATE aggregate-visible base + BOUNDED sequence term) ----
        pid = B.cat_probs(Wd, u[:, 8])
        path = 1.0 + s * (PATH_LUTd[o2] - 1.0)
        ogate = B.clip(PATd[o1, o2], 0.35, 2.8)
        renew = B.where(
            (since_atc >= 1) & (since_atc <= 3),
            B.full((C,), 2.2, "f32"),
            B.where(since_atc <= 10, B.full((C,), 1.5, "f32"), B.full((C,), 1.0, "f32")),
        )
        regime_act = regime_act_lut[typed]
        chan_regime = B.where(
            sent,
            1.0
            + s * B.where((chan_act == chan_pref_d) == ((typed == 1) | (typed == 2)), 0.25, -0.15),
            B.full((C,), 1.0, "f32"),
        )
        action_discount = B.where(off_act == 0, 0.0, 0.05 + 0.06 * B.to_f32(off_act))
        price_fit = B.where(
            sent,
            1.0 + s * (0.35 * (action_discount * price_sens_d * 4.0) - 0.15 * price_sens_d),
            B.full((C,), 1.0, "f32"),
        )
        q_eff = B.where(sent, q, B.full((C,), 1.0, "f32"))
        discount = B.maximum(action_discount, camp_discount)
        cat_prod = CCP[ar, pid]
        # aggregate-visible base: what a wide table (counts/sums/recency) can see
        base_mult = (
            propensity_d
            * regime_act
            * (0.45 + 0.55 * q_eff)
            * chan_regime
            * price_fit
            * (1.0 + 0.40 * shock)
            * (1.0 + 0.25 * cat_prod)
        )
        # sequence-only term: recent all-event/organic k-gram shape, last-event path,
        # gap/renewal since add_to_cart, and order-gated self-excitation -- combined into
        # ONE bounded multiplier so the aggregate base stays learnable while the sequence
        # stays strongly order-dependent (aggregates are blind to the ORDER).
        seq_score = (
            0.5 * (gate2 - 1.0)
            + 0.6 * (ogate - 1.0)
            + 0.5 * (path - 1.0)
            + 0.6 * (renew - 1.0)
            + 1.0 * ob
        )
        seq_mult = B.clip(1.0 + float(os.environ.get("WQ_SEQ_AMP", "2.0")) * seq_score, 0.25, 3.5)
        p_buy = (
            spec.base_purchase
            * base_mult
            * seq_mult
            * B.where(d >= refr_until, B.full((C,), 1.0, "f32"), B.full((C,), 0.25, "f32"))
        )
        p_buy = B.clip(p_buy, 0.0, 1.0)
        buy = active & (u[:, 7] < p_buy)

        qty = 1 + B.floor_i(u[:, 9] * 3)
        unit = prod_price_d[pid]
        # sequence-shape gives a MILD value tilt only (a strong order->value term just adds
        # unlearnable variance at this scale and hurts the encoder)
        seq_val = 1.0 + s * 0.30 * B.clip(gate2 - 1.0, -0.8, 1.5)
        val = B.round_(unit * B.to_f32(qty) * (1.0 - discount) * seq_val, 2)
        margin = B.round_(val * prod_margin_d[pid], 2)
        touch = B.cat_probs(touch_pref_d, u[:, 10])
        buy_mask[d] = buy
        buy_pid[d] = B.where(buy, pid, 0)
        buy_qty[d] = B.where(buy, qty, 0)
        buy_unit[d] = B.where(buy, unit, 0.0)
        buy_val[d] = B.where(buy, val, 0.0)
        buy_margin[d] = B.where(buy, margin, 0.0)
        buy_disc[d] = B.where(buy, discount, 0.0)
        buy_touch[d] = B.where(buy, touch, 0)
        buy_camp[d] = B.where(buy, camp_idx, B.full((C,), -1, "i64"))
        buy_chan[d] = B.where(buy, chan_act, 0)
        buy_hour[d] = B.where(buy, hour_act, 0)
        refr_until = B.where(buy, fd + 2.0, refr_until)
        a2 = B.where(buy, a1, a2)
        a1 = B.where(buy, B.full((C,), IDX_BUY, "i64"), a1)
        a2 = B.where(due, a1, a2)
        a1 = B.where(due, B.full((C,), IDX_RET, "i64"), a1)

        # ---- schedule returns ----
        make_ret = buy & (u[:, 11] < s * return_prone_d * 0.6)
        rday_new = B.to_f32(B.floor_i(fd + 1.0 + u[:, 12] * 5.0)) + 0.998
        ok = make_ret & (rday_new < float(T))
        rslot = B.to_i64(rday_new) % 7
        rem_cur = rmask[ar, rslot]
        rmask[ar, rslot] = ok | rem_cur
        rday[ar, rslot] = B.where(ok, rday_new, rday[ar, rslot])
        rval[ar, rslot] = B.where(ok, -val, rval[ar, rslot])
        rpid[ar, rslot] = B.where(ok, pid, rpid[ar, rslot])
        rsrc_day[ar, rslot] = B.where(ok, B.full((C,), fd, "f32"), rsrc_day[ar, rslot])
        rreason[ar, rslot] = B.where(ok, B.floor_i(u[:, 18] * 4.0), rreason[ar, rslot])

        # ---- organic event: variable-order Markov transition ----
        organic = active & (~buy) & (u[:, 13] < s * spec.base_activity * regime_act)
        row_o = T2d[o1, o2][:, FUNNELd]
        tot_o = B.sum(row_o, -1, keepdim=True)
        pr = row_o / B.maximum(tot_o, B.full(tot_o.shape, 1e-9, "f32"))
        et_idx = B.cat_probs(pr, u[:, 14])
        is_missing_class = et_idx < 4
        emit_org = organic & (~is_missing_class | (u[:, 16] > spec.missing_rate))
        opid = B.cat_probs(PROD_POP2, u[:, 15])
        otouch = B.cat_probs(touch_pref_d, u[:, 19])
        org_mask[d] = emit_org
        org_et[d] = B.where(emit_org, et_idx, 0)
        org_pid[d] = B.where(emit_org, opid, 0)
        org_touch[d] = B.where(emit_org, otouch, 0)
        org_camp[d] = B.where(emit_org, camp_idx, B.full((C,), -1, "i64"))

        et_a = et_idx + 1
        o1 = B.where(emit_org, o2, o1)
        o2 = B.where(emit_org, et_a, o2)
        a2 = B.where(emit_org, a1, a2)
        a1 = B.where(emit_org, ORG_TO_ALLd[et_idx], a1)
        since_atc = B.where(
            emit_org & (et_idx == ATC_ORG),
            B.zeros((C,), "i64"),
            since_atc + 1,
        )

        # order-gated delayed trigger: only SPECIFIC hot k-grams fire
        ready_new = T2d[o1, o2][:, ATC]
        hot = emit_org & (ready_new > HOT)
        or_t0[:, slot_o] = B.where(hot, fd, or_t0[:, slot_o])
        or_delay[:, slot_o] = B.where(hot, DELAY2d[o1, o2], B.full((C,), 1.0, "f32"))
        or_amp[:, slot_o] = B.where(hot, B.maximum(ready_new - HOT, 0.0), B.zeros((C,), "f32"))

    if B.use_cuda:
        B.t.cuda.synchronize()
    sim_seconds = time.perf_counter() - t_sim

    # ======================= assemble columnar events =======================
    frames = []

    # signup (every customer at day 0)
    signup_ts = _iso(start64, -signup_days.astype(np.float64))
    frames.append(
        pl.DataFrame(
            {
                "customer_key": cust_keys,
                "event_day": np.zeros(C),
                "event_type": "customer_signup",
                "product_id": [None] * C,
                "category_code": [None] * C,
                "brand": [None] * C,
                "channel": [None] * C,
                "campaign_id": [None] * C,
                "price": [None] * C,
                "qty": [None] * C,
                "discount": [None] * C,
                "margin": [None] * C,
                "session_id": [None] * C,
                "value": np.zeros(C),
                "acquisition_channel": np.array(ACQ)[acq],
                "region": np.array(REGIONS)[region],
                "segment": segment,
                "loyalty_tier": np.array(LOYALTY)[loyal],
                "signup_ts": signup_ts,
            }
        )
        .with_columns(pl.Series("event_ts", _iso(start64, np.zeros(C))))
        .with_columns(
            pl.struct(["acquisition_channel", "region", "segment", "loyalty_tier", "signup_ts"])
            .struct.json_encode()
            .alias("attrs_json")
        )
        .select(_BASE_COLS)
    )

    def emit_frames(mask, cols):
        idx = B.nonzero2d(mask)
        if idx.size == 0:
            return None
        d_idx = idx[:, 0]
        c_idx = idx[:, 1]
        fields = {k: B.masked(v, mask) for k, v in cols.items()}
        return d_idx, c_idx, fields

    # sends
    r = emit_frames(
        send_mask,
        {
            "akey": send_akey,
            "freq": send_freq,
            "hour": send_hour,
            "dow": send_dow,
            "chan": send_chan,
            "off": send_off,
            "camp": send_camp,
        },
    )
    if r is not None:
        d_idx, c_idx, f = r
        ch = f["chan"]
        day = d_idx + (f["hour"].astype(np.float64) + 0.25) / 24.0
        frames.append(
            pl.DataFrame(
                {
                    "customer_key": cust_keys[c_idx],
                    "event_day": day,
                    "event_type": "email_send",
                    "product_id": [None] * len(d_idx),
                    "category_code": [None] * len(d_idx),
                    "brand": [None] * len(d_idx),
                    "channel": chan_str[ch],
                    "campaign_id": _camp_strs(f["camp"], camp_str),
                    "price": [None] * len(d_idx),
                    "qty": [None] * len(d_idx),
                    "discount": [None] * len(d_idx),
                    "margin": [None] * len(d_idx),
                    "session_id": [None] * len(d_idx),
                    "value": np.zeros(len(d_idx)),
                    "touch": chan_str[ch],
                    "target_key": np.array(ACTS)[f["akey"]],
                    "freq": f["freq"],
                    "hour": f["hour"],
                    "dow": f["dow"],
                    "channel_a": ch,
                    "offer": f["off"],
                }
            )
            .with_columns(pl.Series("event_ts", _iso(start64, day)))
            .with_columns(
                pl.struct(
                    [
                        "touch",
                        "target_key",
                        "campaign_id",
                        "freq",
                        "hour",
                        "dow",
                        pl.col("channel_a").alias("channel"),
                        "offer",
                    ]
                )
                .struct.json_encode()
                .alias("attrs_json")
            )
            .drop("channel_a")
            .select(_BASE_COLS)
        )

    # purchases
    r = emit_frames(
        buy_mask,
        {
            "pid": buy_pid,
            "qty": buy_qty,
            "unit": buy_unit,
            "val": buy_val,
            "margin": buy_margin,
            "disc": buy_disc,
            "touch": buy_touch,
            "camp": buy_camp,
            "chan": buy_chan,
            "hour": buy_hour,
        },
    )
    purchase_rows = None
    if r is not None:
        d_idx, c_idx, f = r
        pid = f["pid"]
        day = d_idx + (f["hour"].astype(np.float64) + 0.75) / 24.0
        frames.append(
            pl.DataFrame(
                {
                    "customer_key": cust_keys[c_idx],
                    "event_day": day,
                    "event_type": "purchase",
                    "product_id": prod_str[pid],
                    "category_code": cat_str[prod_cat[pid]],
                    "brand": brand_str[prod_brand[pid]],
                    "channel": touch_str[f["touch"]],
                    "campaign_id": _camp_strs(f["camp"], camp_str),
                    "price": f["unit"],
                    "qty": f["qty"],
                    "discount": np.round(f["disc"], 3),
                    "margin": f["margin"],
                    "session_id": [None] * len(d_idx),
                    "value": f["val"],
                    "order_id": [None] * len(d_idx),
                    "touch": touch_str[f["touch"]],
                    "value_a": f["val"],
                    "campaign_a": _camp_strs(f["camp"], camp_str),
                }
            )
            .with_columns(pl.Series("event_ts", _iso(start64, day)))
            .with_columns(
                pl.struct(
                    [
                        "brand",
                        "order_id",
                        "product_id",
                        "category_code",
                        "price",
                        "qty",
                        "discount",
                        "touch",
                        pl.col("campaign_a").alias("campaign_id"),
                        "margin",
                        pl.col("value_a").alias("value"),
                    ]
                )
                .struct.json_encode()
                .alias("attrs_json")
            )
            .drop("value_a", "campaign_a")
            .select(_BASE_COLS)
        )
        purchase_rows = (
            d_idx,
            c_idx,
            pid,
            f,
            day,
        )

    # organic
    r = emit_frames(
        org_mask,
        {"et": org_et, "pid": org_pid, "touch": org_touch, "camp": org_camp},
    )
    org_rows = None
    if r is not None:
        d_idx, c_idx, f = r
        et = f["et"]
        pid = f["pid"]
        org_rows = (d_idx, c_idx, et, f)
        day = d_idx.astype(np.float64) + 0.999
        et_name = np.array(ORG_EVENTS)[et]
        frames.append(
            pl.DataFrame(
                {
                    "customer_key": cust_keys[c_idx],
                    "event_day": day,
                    "event_type": et_name,
                    "product_id": prod_str[pid],
                    "category_code": cat_str[prod_cat[pid]],
                    "brand": brand_str[prod_brand[pid]],
                    "channel": touch_str[f["touch"]],
                    "campaign_id": _camp_strs(f["camp"], camp_str),
                    "price": [None] * len(d_idx),
                    "qty": [None] * len(d_idx),
                    "discount": [None] * len(d_idx),
                    "margin": [None] * len(d_idx),
                    "session_id": [None] * len(d_idx),
                    "value": np.zeros(len(d_idx)),
                    "touch": touch_str[f["touch"]],
                    "product_id_a": prod_str[pid],
                    "category_a": cat_str[prod_cat[pid]],
                    "campaign_a": _camp_strs(f["camp"], camp_str),
                }
            )
            .with_columns(pl.Series("event_ts", _iso(start64, day)))
            .with_columns(
                pl.struct([pl.col("brand"), pl.col("product_id_a").alias("product_id"), pl.col("category_a").alias("category_code"), pl.col("touch"), pl.col("campaign_a").alias("campaign_id")])
                .struct.json_encode()
                .alias("attrs_json")
            )
            .drop("product_id_a", "category_a", "campaign_a")
            .select(_BASE_COLS)
        )

    # returns
    r = emit_frames(
        ret_mask,
        {"pid": ret_pid, "val": ret_val, "reason": ret_reason, "dayv": ret_dayv, "rsrc": ret_src},
    )
    return_rows = None
    if r is not None:
        d_idx, c_idx, f = r
        pid = f["pid"]
        top = f["dayv"].astype(np.float64)
        frames.append(
            pl.DataFrame(
                {
                    "customer_key": cust_keys[c_idx],
                    "event_day": top,
                    "event_type": "return",
                    "product_id": prod_str[pid],
                    "category_code": cat_str[prod_cat[pid]],
                    "brand": brand_str[prod_brand[pid]],
                    "channel": [None] * len(d_idx),
                    "campaign_id": [None] * len(d_idx),
                    "price": [None] * len(d_idx),
                    "qty": [None] * len(d_idx),
                    "discount": [None] * len(d_idx),
                    "margin": [None] * len(d_idx),
                    "session_id": [None] * len(d_idx),
                    "value": f["val"],
                    "order_id": [None] * len(d_idx),
                    "touch": [None] * len(d_idx),
                    "reason": np.array(["size", "fit", "damaged", "changed_mind"])[f["reason"]],
                    "value_a": f["val"],
                }
            )
            .with_columns(pl.Series("event_ts", _iso(start64, top)))
            .with_columns(
                pl.struct(
                    [
                        "brand",
                        "order_id",
                        "product_id",
                        "category_code",
                        "reason",
                        pl.col("value_a").alias("value"),
                    ]
                )
                .struct.json_encode()
                .alias("attrs_json")
            )
            .drop("value_a")
            .select(_BASE_COLS)
        )
        return_rows = (d_idx, c_idx, pid, f, top)

    base = pl.concat(frames, how="vertical_relaxed").sort(["customer_key", "event_ts"])
    base = base.with_columns(pl.int_range(1, pl.len() + 1).alias("event_id"))

    # ---- relational tables -------------------------------------------------
    customers = pl.DataFrame(
        {
            "customer_key": cust_keys,
            "signup_ts": signup_ts,
            "segment": segment,
            "region": np.array(REGIONS)[region],
            "acquisition_channel": np.array(ACQ)[acq],
            "loyalty_tier": np.array(LOYALTY)[loyal],
            # true marginal discount effect (dq/ddiscount = 0.35*4*price_sens from the action effect)
            "discount_sensitivity": np.round(1.4 * price_sens, 4),
        }
    )
    products = pl.DataFrame(
        {
            "product_id": prod_str,
            "category_code": cat_str[prod_cat],
            "brand": brand_str[prod_brand],
            "price": prod_price,
            "margin": np.round(prod_margin, 4),
        }
    )
    campaigns = pl.DataFrame(
        {
            "campaign_id": camp_str,
            "channel": np.array([CHANNELS[i % len(CHANNELS)] for i in range(n_camp)]),
            "discount": camp_disc,
            "start_day": camp_start,
            "duration_days": camp_dur,
        }
    )

    # orders / items / returns / support built from the extracted events
    orders = pl.DataFrame(
        schema={
            "order_id": pl.Utf8,
            "customer_key": pl.Utf8,
            "event_ts": pl.Utf8,
            "channel": pl.Utf8,
            "discount": pl.Float64,
            "total": pl.Float64,
            "margin": pl.Float64,
            "status": pl.Utf8,
        }
    )
    items = pl.DataFrame(
        schema={
            "order_id": pl.Utf8,
            "product_id": pl.Utf8,
            "qty": pl.Int64,
            "unit_price": pl.Float64,
            "discount": pl.Float64,
            "value": pl.Float64,
        }
    )
    returns = pl.DataFrame(
        schema={
            "return_id": pl.Utf8,
            "order_id": pl.Utf8,
            "customer_key": pl.Utf8,
            "event_ts": pl.Utf8,
            "reason": pl.Utf8,
        }
    )
    support = pl.DataFrame(
        schema={
            "ticket_id": pl.Utf8,
            "customer_key": pl.Utf8,
            "event_ts": pl.Utf8,
            "reason": pl.Utf8,
        }
    )

    if purchase_rows is not None:
        d_idx, c_idx, pid, f, day = purchase_rows
        oid = np.char.add("o", np.char.zfill((np.arange(len(d_idx)) + 1).astype(str), 9))
        ord_ts = _iso(start64, day)
        orders = pl.DataFrame(
            {
                "order_id": oid,
                "customer_key": cust_keys[c_idx],
                "event_ts": ord_ts,
                "channel": chan_str[f["chan"]],
                "discount": np.round(f["disc"], 3),
                "total": f["val"],
                "margin": f["margin"],
                "status": "completed",
            }
        )
        items = pl.DataFrame(
            {
                "order_id": oid,
                "product_id": prod_str[pid],
                "qty": f["qty"],
                "unit_price": f["unit"],
                "discount": np.round(f["disc"], 3),
                "value": f["val"],
            }
        )
        # link returns to the order from their source (customer, floor(day))
        src = pl.DataFrame(
            {
                "customer_key": cust_keys[c_idx],
                "src_day": d_idx.astype(np.int64),
                "order_id": oid,
            }
        )
        if return_rows is not None:
            rd_idx, rc_idx, rpid_, rf, rtop = return_rows
            rsrc = pl.DataFrame(
                {
                    "customer_key": cust_keys[rc_idx],
                    "src_day": rf["rsrc"].astype(np.int64),
                    "reason": np.array(["size", "fit", "damaged", "changed_mind"])[rf["reason"]],
                    "event_ts": _iso(start64, rtop),
                }
            ).join(src, on=["customer_key", "src_day"], how="left")
            returns = pl.DataFrame(
                {
                    "return_id": np.char.add(
                        "r", np.char.zfill((np.arange(rsrc.height) + 1).astype(str), 8)
                    ),
                    "order_id": rsrc["order_id"].to_list(),
                    "customer_key": rsrc["customer_key"].to_list(),
                    "event_ts": rsrc["event_ts"].to_list(),
                    "reason": rsrc["reason"].to_list(),
                }
            )

    if org_rows is not None:
        od_idx, oc_idx, oet, _of = org_rows
        sup_m = oet == ATC_ORG + 2
        if sup_m.any():
            oday = od_idx.astype(np.float64) + 0.999
            support = pl.DataFrame(
                {
                    "ticket_id": np.char.add(
                        "t", np.char.zfill((np.arange(int(sup_m.sum())) + 1).astype(str), 8)
                    ),
                    "customer_key": cust_keys[oc_idx][sup_m].tolist(),
                    "event_ts": np.asarray(_iso(start64, oday))[sup_m].tolist(),
                    "reason": ["delivery"] * int(sup_m.sum()),
                }
            )

    stream = base.select(
        [
            pl.col("customer_key"),
            pl.col("event_ts"),
            pl.col("event_type"),
            pl.col("attrs_json").alias("event_attributes"),
        ]
    ).sort(["customer_key", "event_ts"])

    # ---- truth receipt -----------------------------------------------------
    regime_runs = _regime_runs_np(B.to_numpy(reg_traj), REGIMES)
    cat_pref = cat_str[np.argmax(cust_cat, axis=1)]
    truth_customers = [
        {
            "customer": cust_keys[i],
            "x0": round(float(x0[i]), 3),
            "h0": round(float(h0[i]), 3),
            "d0": round(float(d0[i]), 3),
            "offer_opt": int(offer_opt[i]),
            "channel_pref": CHANNELS[int(chan_pref[i])],
            "segment": int(segment[i]),
            "churn_day": round(float(churn_day[i]), 2),
            "regime0": REGIMES[int(regime0[i])],
            "regime_runs": regime_runs[i],
            "price_sensitivity": round(float(price_sens[i]), 4),
            "discount_sensitivity": round(1.4 * float(price_sens[i]), 4),
            "return_proneness": round(float(return_prone[i]), 4),
            "touch_pref": [round(float(v), 4) for v in touch_pref[i]],
            "category_pref": str(cat_pref[i]),
        }
        for i in range(C)
    ]

    tables = {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": items,
        "events": base,
        "campaigns": campaigns,
        "returns": returns,
        "support": support,
    }
    info = {
        "device": B.name,
        "sim_seconds": round(sim_seconds, 3),
        "n_customers": C,
        "window_days": T,
        "seed": spec.seed,
        "events": stream.height,
    }
    info.update(B.mem_info())
    return stream, tables, truth_customers, info


def build_relational(spec: RelationalSpec, path_engine=None):
    """In-process builder (CPU PATH A) retained as the module's public API."""
    stream, tables, truth_customers, info = simulate(spec, use_cuda=True)
    as_of = datetime.fromisoformat(spec.as_of)
    features, labels, wide, engine_state = build_path_a(
        tables["events"], tables["customers"], as_of, engine=path_engine
    )
    return (
        stream,
        {"features": features, "labels": labels, "wide": wide},
        tables,
        {
            "customers": truth_customers,
            "_runtime": info,
            "_path_a_engine": engine_state,
        },
    )


def _path_a_via_gpu(base_events, customers, as_of):
    venv_py = _ROOT / ".venv-cudf" / "bin" / "python"
    script = _ROOT / "scripts" / "gpu_path_a.py"
    if not venv_py.exists() or not script.exists():
        return None, {
            "requested": "gpu",
            "used": "cpu",
            "note": "`.venv-cudf` or gpu_path_a.py missing",
        }
    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            base_events.write_parquet(tdp / "base.parquet")
            customers.write_parquet(tdp / "customers.parquet")
            proc = subprocess.run(
                [
                    str(venv_py),
                    str(script),
                    "--base",
                    str(tdp / "base.parquet"),
                    "--customers",
                    str(tdp / "customers.parquet"),
                    "--as-of",
                    as_of.date().isoformat(),
                    "--out",
                    str(tdp),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0:
                return None, {
                    "requested": "gpu",
                    "used": "cpu",
                    "note": f"gpu_path_a failed: {proc.stderr.strip()[-400:]}",
                }
            state = json.loads((tdp / "engine.json").read_text())
            feats = pl.read_parquet(tdp / "features.parquet")
            labs = pl.read_parquet(tdp / "labels.parquet")
            wide = pl.read_parquet(tdp / "wide.parquet")
            return (feats, labs, wide), state
    except Exception as exc:  # noqa: BLE001
        return None, {"requested": "gpu", "used": "cpu", "note": f"{type(exc).__name__}: {exc}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=2000, dest="n_customers")
    ap.add_argument("--window-days", type=int, default=540, dest="window_days")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--snr", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="relational_reference")
    ap.add_argument("--start-date", default="2024-01-01", dest="start_date")
    ap.add_argument("--as-of", default="2025-01-01", dest="as_of")
    ap.add_argument("--no-validate", action="store_true", dest="no_validate")
    a = ap.parse_args(argv)
    t_main = time.perf_counter()
    spec = RelationalSpec(
        n_customers=a.n_customers,
        window_days=a.window_days,
        seed=a.seed,
        snr=a.snr,
        start_date=a.start_date,
        as_of=a.as_of,
    )
    stream, tables, truth_customers, info = simulate(spec, use_cuda=True)

    as_of = datetime.fromisoformat(spec.as_of)
    gpu_res, engine_state = _path_a_via_gpu(tables["events"], tables["customers"], as_of)
    if gpu_res is None:
        features, labels, wide, engine_state2 = build_path_a(
            tables["events"], tables["customers"], as_of, engine=None
        )
        engine_state = {**engine_state, **{k: engine_state2[k] for k in ("used", "note")}}
    else:
        features, labels, wide = gpu_res
    path_a = {"features": features, "labels": labels, "wide": wide}

    d = Path(a.out)
    (d / "relational").mkdir(parents=True, exist_ok=True)
    (d / "path_a").mkdir(parents=True, exist_ok=True)
    (d / "path_b").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_parquet(d / "relational" / f"{name}.parquet", compression="zstd")
    path_a["features"].write_parquet(d / "path_a" / "customer_features.parquet", compression="zstd")
    path_a["labels"].write_parquet(d / "path_a" / "customer_labels.parquet", compression="zstd")
    path_a["wide"].write_parquet(d / "path_a" / "path_a_wide_table.parquet", compression="zstd")
    stream.write_parquet(d / "path_b" / "path_b_customer_event_stream.parquet", compression="zstd")
    stream.write_parquet(d / "customer_event_stream.parquet", compression="zstd")
    path_a["wide"].write_parquet(d / "customer_wide_table.parquet", compression="zstd")

    meta = {
        "name": "relational_reference",
        "spec": asdict(spec),
        "n_customers": spec.n_customers,
        "window_days": spec.window_days,
        "seed": spec.seed,
        "as_of": as_of.date().isoformat(),
        "as_of_day": (as_of - datetime.fromisoformat(spec.start_date)).days,
        "paths": {
            "A": "traditional supervised wide table: features <= as_of, labels > as_of (30/90d)",
            "B": "customer event stream (full window; the encoder must use <= as_of)",
        },
        "categories": CATEGORIES,
        "brands": BRANDS,
        "contact_channels": CHANNELS,
        "event_types": [
            "customer_signup",
            "view",
            "search",
            "add_to_cart",
            "remove_from_cart",
            "purchase",
            "return",
            "email_send",
            "support_ticket",
            "review",
            "loyalty_change",
        ],
        "action_keys": list(ACTS),
        "arm_key": "channel",
        "action": {
            "action_period_seconds": 86400.0,
            "action_design": "one_factor_daily",
            "note": "daily one-factor randomized action; other keys at the customer optimum; "
            "purchase depends on action quality x regime x path x interactions x long-memory shock, "
            "with DELAYED sequence-conditioned response.",
        },
    }
    truth = {"meta": meta, "customers": truth_customers}
    (d / f"{a.name}_truth.json").write_text(json.dumps(truth, indent=1))
    (d / "reference_truth.json").write_text(json.dumps(truth, indent=1))

    manifest = {
        "name": a.name,
        "actions": [{"event_type": "email_send", "fields": ["channel", "offer", "freq", "hour"]}],
        "join_key": "customer_key",
        "as_of": spec.as_of,
        "window": {"start": spec.start_date, "days": spec.window_days},
        "paths": {
            "A": {
                "description": "traditional supervised wide table",
                "features": "path_a/customer_features.parquet",
                "labels": "path_a/customer_labels.parquet",
                "wide": "path_a/path_a_wide_table.parquet",
                "rows": path_a["wide"].height,
                "columns": path_a["wide"].width,
            },
            "B": {
                "description": "customer event stream (full window; encoder uses <= as_of)",
                "stream": "path_b/path_b_customer_event_stream.parquet",
                "rows": stream.height,
                "columns": stream.width,
            },
        },
        "relational": {k: f"relational/{k}.parquet" for k in tables},
        "counts": {
            "customers": spec.n_customers,
            "events": stream.height,
            "orders": tables["orders"].height,
            "returns": tables["returns"].height,
        },
        "truth": f"{a.name}_truth.json",
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    cpu_old = os.environ.get("GEN_CPU_OLD_SECONDS")
    total_wall = round(time.perf_counter() - t_main, 3)
    receipt = {
        "device": info.get("device"),
        "n_customers": spec.n_customers,
        "window_days": spec.window_days,
        "seed": spec.seed,
        "sim_wall_s": info.get("sim_seconds"),
        "gpu_new_total_wall_s": total_wall,
        "events": stream.height,
        "path_a_engine": engine_state,
        "one_gpu_job_at_a_time": True,
        "gpu_mem": {
            k: info[k]
            for k in (
                "mem_free_bytes",
                "mem_total_bytes",
                "max_memory_allocated_bytes",
                "device_name",
            )
            if k in info
        },
    }
    if info.get("torch_version"):
        receipt["torch_version"] = info["torch_version"]
    if cpu_old is not None:
        old_s = float(cpu_old)
        receipt["cpu_old_wall_s"] = old_s
        receipt["gpu_new_wall_s"] = info["sim_seconds"]
        receipt["speedup_total"] = round(old_s / max(total_wall, 1e-9), 2)
        receipt["speedup_sim"] = round(old_s / max(float(info["sim_seconds"]), 1e-9), 2)
    (d / "runtime_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(manifest["counts"] | {"wide_columns": path_a["wide"].width}, indent=1))
    print(json.dumps(receipt, indent=1))

    if not a.no_validate:
        rc = subprocess.run(
            [sys.executable, str(_ROOT / "scripts/validate_path_a.py"), "--dir", str(d)],
            check=False,
        ).returncode
        if rc != 0:
            raise SystemExit("PATH A validation failed; see path_a_validation.json")


if __name__ == "__main__":
    main()
