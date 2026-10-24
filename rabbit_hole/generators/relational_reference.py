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
VECTORIZED across customers (all customers advance together, one vectorized pass per day)
with torch on CUDA when available (numpy fallback). PATH A aggregation runs on the
polars-CUDF GPU engine through the dedicated `.venv-cudf` process boundary, falling back to
CPU polars with the reason recorded.

This is a faithful vectorization of the per-customer reference dynamics: the SAME probability
expressions, distributions, regime chain, path/order effects, action effects, long-memory
shock, purchase/value/return/churn generation and latents. No mechanism is added or removed.
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

# event-state alphabet (only events that update the path state)
SIGNUP, VIEW, SEARCH, ATC, REMOVE, SUPPORT, REVIEW, PURCHASE, RETURN = range(9)
NSTATE = 9
# ORG_EVENTS -> state codes (view, search, atc, remove, support, review)
ORG_CODES = np.array([VIEW, SEARCH, ATC, REMOVE, SUPPORT, REVIEW], dtype=np.int64)
PATH_LUT = np.array([1.0, 1.15, 1.0, 1.6, 1.0, 1.0, 1.0, 0.85, 0.65], dtype=np.float32)
REGIME_BUY = np.array([0.75, 1.55, 1.25, 0.45], dtype=np.float32)
RETURN_REASONS = ["size", "fit", "damaged", "changed_mind"]

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
    base_activity: float = 0.55
    base_purchase: float = 0.10


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

    def cumsum(self, x, dim=-1):
        return self.t.cumsum(x, dim=dim) if self.use_cuda else self.np.cumsum(x, axis=dim)

    def cat_cdf(self, cdf_rows, u):
        """Sample a category per row from per-row cumulative densities (C, K)."""
        if self.use_cuda:
            idx = (cdf_rows < u[:, None]).sum(dim=-1)
            return self.t.clamp(idx, max=cdf_rows.shape[-1] - 1)
        idx = (cdf_rows < u[:, None]).sum(axis=-1).astype(self.np.int64)
        return self.np.minimum(idx, cdf_rows.shape[-1] - 1)

    def cat_shared(self, cdf_row, u):
        """Sample a category per row from a shared cumulative density (K,)."""
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
    out = [[] for _ in range(C)]
    for i in range(len(st_keys)):
        out[custs[i]].append([regime_names[regs[i]], int(counts[st_keys[i]])])
    return out


def _camp_strs(camp_idx, camp_str):
    out = np.full(len(camp_idx), None, dtype=object)
    m = camp_idx >= 0
    out[m] = camp_str[camp_idx[m]]
    return out.tolist()


def simulate(spec: RelationalSpec, use_cuda: bool = True):
    """Vectorized (customers advance together) simulation with optional CUDA."""
    t_sim = time.perf_counter()
    C = int(spec.n_customers)
    T = int(spec.window_days)
    s = float(spec.snr)
    P = int(spec.n_products)
    start64 = np.datetime64(spec.start_date).astype("datetime64[ns]")
    cust_keys = np.array([f"rc{i:07d}" for i in range(C)])

    B = _TensorBackend(spec.seed * 2654435761 + 13, use_cuda and _cuda_ok())

    # ---- latents (drawn once; not a hot path) ------------------------------
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
    churn_day = np.where(
        lrng.random(C) < 0.18, lrng.uniform(0.45 * T, T, C), float(T)
    ).astype(np.float32)
    x0 = lrng.uniform(0, 3, C).astype(np.float32)
    h0 = lrng.uniform(0, 24, C).astype(np.float32)
    d0 = lrng.uniform(0, 7, C).astype(np.float32)
    offer_opt = lrng.integers(0, 4, C).astype(np.int64)
    Pc = lrng.dirichlet(np.full(len(REGIMES), 1.2), size=(C, len(REGIMES)))
    Pc = 0.93 * np.eye(len(REGIMES))[None] + 0.07 * Pc
    regime0 = lrng.integers(0, len(REGIMES), C).astype(np.int64)
    shock_day = lrng.uniform(0.1 * T, 0.8 * T, C).astype(np.float32)
    shock_amp = lrng.uniform(0.3, 1.2, C).astype(np.float32)
    shock_tau = lrng.uniform(10, 45, C).astype(np.float32)
    region = lrng.integers(0, 4, C).astype(np.int64)
    signup_days = lrng.integers(1, 400, C).astype(np.int64)

    prod_str = np.array([f"p{i:03d}" for i in range(P)])
    camp_str = np.array([f"cmp{i:03d}" for i in range(n_camp)])
    cat_str = np.array(CATEGORIES)
    brand_str = np.array(BRANDS)
    touch_str = np.array(TOUCH)
    chan_str = np.array(CHANNELS)
    ar = B.arange(C)

    # product-choice weights per customer (fixed) and shared organic population
    W = cust_cat[:, prod_cat] * (0.5 + prod_pop)[None, :]
    W = W / W.sum(1, keepdims=True)
    W_CDF = B.to_dev(np.cumsum(W, axis=1), "f32")
    PROD_POP_CDF = B.to_dev(np.cumsum(prod_pop), "f32")
    TOUCH_CDF = B.to_dev(np.cumsum(touch_pref, axis=1), "f32")
    REG_CDF = B.to_dev(np.cumsum(Pc, axis=2), "f32")
    PATH_LUT_d = B.to_dev(PATH_LUT, "f32")
    CCP = B.to_dev(cust_cat[:, prod_cat], "f32")  # (C, P) category affinity of chosen product
    prod_price_d = B.to_dev(prod_price, "f32")
    prod_margin_d = B.to_dev(prod_margin, "f32")

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
    REGIME_BUY_d = B.to_dev(REGIME_BUY, "f32")

    # campaign membership per day (global)
    days = np.arange(T)
    camp_active_mask = (camp_start[None, :] <= days[:, None]) & (
        days[:, None] < camp_start[None, :] + camp_dur[None, :]
    )
    camp_lists = [
        B.to_dev(np.flatnonzero(camp_active_mask[d]).astype(np.int64), "i64") for d in range(T)
    ]
    camp_disc_d = B.to_dev(camp_disc, "f32")

    freq_def = B.to_i64(B.clip(B.round_(x0d), 0, 3))
    hour_def = B.to_i64(B.clip(B.round_(h0d), 0, 23))
    dow_def = B.to_i64(B.clip(B.round_(d0d), 0, 6))
    zero = B.zeros((C,), "f32")

    # uniform pool: (T, C, 24) independent channels, sliced per day
    U = B.rand((T, C, 24))

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
    org_sec = B.zeros((T, C), "i64")

    ret_mask = B.zeros((T, C), "bool")
    ret_dayv = B.zeros((T, C), "f32")
    ret_pid = B.zeros((T, C), "i64")
    ret_val = B.zeros((T, C), "f32")
    ret_reason = B.zeros((T, C), "i64")
    ret_src = B.zeros((T, C), "f32")

    regime = regime0_d
    last = B.full((C,), SIGNUP, "i64")
    prev1 = B.full((C,), SIGNUP, "i64")
    prev2 = B.full((C,), SIGNUP, "i64")
    reg_traj = B.zeros((T, C), "i64")

    for d in range(T):
        u = U[d]
        fd = float(d)
        active = d < churn_day_d

        # long-memory shock (computed for every day; used only when active)
        shock = B.where(
            d >= shock_day_d,
            shock_amp_d * B.exp(-(fd - shock_day_d) / B.maximum(shock_tau_d, 1.0)),
            zero,
        )

        # regime switch (sticky chain; customer-specific transition)
        switch = active & (u[:, 0] < 0.035)
        cdf_rows = REG_CDF[ar, regime]
        newreg = B.cat_cdf(cdf_rows, u[:, 1])
        regime = B.where(switch, newreg, regime)
        regime_mult = 1.0 + s * (REGIME_BUY_d[regime] - 1.0)

        # campaign for the day
        active_camps = camp_lists[d]
        if len(active_camps):
            pick = active_camps[B.floor_i(u[:, 3] * len(active_camps))]
            sent_camp = u[:, 2] < 0.25
            camp_idx = B.where(sent_camp, pick, B.full((C,), -1, "i64"))
            camp_discount = B.where(sent_camp, camp_disc_d[pick], zero)
        else:
            camp_idx = B.full((C,), -1, "i64")
            camp_discount = zero

        # one-factor daily action (others at the customer optimum)
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
        q = q * B.where(chan_act == chan_pref_d, B.full((C,), 1.0, "f32"), B.full((C,), 0.35, "f32"))
        q = q * (1.0 - 0.22 * B.abs(B.to_f32(off_act) - B.to_f32(offer_opt_d)))
        q = B.clip(1.0 + s * (q - 1.0), 0.05, 1.0)

        p_send = B.minimum(0.95, 0.20 + 0.20 * B.to_f32(freq_act))
        sent = active & (u[:, 6] < p_send)

        send_mask[d] = sent
        send_akey[d] = B.where(sent, akey, 0)
        send_freq[d] = B.where(sent, freq_act, 0)
        send_hour[d] = B.where(sent, hour_act, 0)
        send_dow[d] = B.where(sent, dow_act, 0)
        send_chan[d] = B.where(sent, chan_act, 0)
        send_off[d] = B.where(sent, off_act, 0)
        send_camp[d] = B.where(sent, camp_idx, B.full((C,), -1, "i64"))

        # path factor: last observed event changes the purchase propensity
        path = PATH_LUT_d[last]
        is_atc = last == ATC
        is_view = last == VIEW
        pat = B.where(
            is_atc & (prev1 == VIEW) & (prev2 == SEARCH),
            1.15,
            B.where(
                is_atc & (prev1 == VIEW),
                1.35,
                B.where((last == VIEW) & (prev1 == ATC), 0.75, B.where((last == PURCHASE) & (prev1 == ATC), 1.20, 1.0)),
            ),
        )
        pat = 1.0 + s * (pat - 1.0)
        path = path * pat
        path = 1.0 + s * (path - 1.0)

        # action effects only when a send actually occurs
        chan_regime = B.where(
            sent,
            1.0 + s * B.where((chan_act == chan_pref_d) == ((regime == 1) | (regime == 2)), 0.25, -0.15),
            B.full((C,), 1.0, "f32"),
        )
        action_discount = B.where(
            sent, B.where(off_act == 0, 0.0, 0.05 + 0.06 * B.to_f32(off_act)), B.full((C,), 0.0, "f32")
        )
        price_fit = B.where(
            sent,
            1.0 + s * (0.35 * (action_discount * price_sens_d * 4.0) - 0.15 * price_sens_d),
            B.full((C,), 1.0, "f32"),
        )
        q_eff = B.where(sent, q, B.full((C,), 1.0, "f32"))
        discount = B.maximum(action_discount, camp_discount)

        pid = B.cat_cdf(W_CDF, u[:, 7])
        p_buy = (
            spec.base_purchase
            * regime_mult
            * (0.55 + 0.45 * q_eff)
            * path
            * chan_regime
            * price_fit
            * (1.0 + 0.9 * shock)
            * (1.0 + 0.35 * CCP[ar, pid])
        )
        p_buy = B.clip(p_buy, 0.0, 1.0)
        buy = active & (u[:, 8] < p_buy)

        qty = 1 + B.floor_i(u[:, 9] * 3)
        unit = prod_price_d[pid]
        val = B.round_(unit * B.to_f32(qty) * (1.0 - discount), 2)
        margin = B.round_(val * prod_margin_d[pid], 2)
        touch = B.cat_cdf(TOUCH_CDF, u[:, 10])

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

        # returns are scheduled by a purchase but take effect on state at the purchase day
        make_ret = buy & (u[:, 11] < s * return_prone_d * 0.6)
        rday = fd + 1.0 + u[:, 12] * 5.0
        ret_ok = make_ret & (rday < float(T))
        ret_mask[d] = ret_ok
        ret_dayv[d] = B.where(ret_ok, rday, 0.0)
        ret_pid[d] = B.where(ret_ok, pid, 0)
        ret_val[d] = B.where(ret_ok, -val, 0.0)
        ret_reason[d] = B.where(ret_ok, B.floor_i(u[:, 18] * 4.0), 0)
        ret_src[d] = B.where(ret_ok, fd, 0.0)

        # purchase state update
        last_p = B.where(buy, PURCHASE, last)
        prev1_p = B.where(buy, last, prev1)
        prev2_p = B.where(buy, prev1, prev2)
        # return state update (after the purchase, same day)
        prev2_r = B.where(ret_ok, prev1_p, prev2_p)
        prev1_r = B.where(ret_ok, last_p, prev1_p)
        last_r = B.where(ret_ok, RETURN, last_p)

        # organic funnel (independent of purchase: made_purchase is never set)
        organic = active & (u[:, 13] < s * spec.base_activity * regime_mult)
        u_et = u[:, 14]
        pick3 = B.floor_i(u_et * 3)
        et = B.where(
            last_r == ATC,
            B.where(u_et < 0.35, PURCHASE, REMOVE),
            B.where(
                last_r == VIEW,
                B.where(pick3 == 0, SEARCH, B.where(pick3 == 1, ATC, PURCHASE)),
                B.floor_i(u_et * 6) + 1,
            ),
        )
        is_missing_class = (et == VIEW) | (et == SEARCH) | (et == ATC) | (et == REMOVE)
        keep = ~(is_missing_class & (u[:, 16] < spec.missing_rate))
        org_pid_v = B.cat_shared(PROD_POP_CDF, u[:, 15])
        org_touch_v = B.cat_cdf(TOUCH_CDF, u[:, 19])
        org_sec_v = B.floor_i(u[:, 17] * 86400.0)
        state_org = organic & (et != PURCHASE)
        org_emit = state_org & keep
        org_mask[d] = org_emit
        org_et[d] = B.where(org_emit, et, 0)
        org_pid[d] = B.where(org_emit, org_pid_v, 0)
        org_touch[d] = B.where(org_emit, org_touch_v, 0)
        org_camp[d] = B.where(org_emit, camp_idx, B.full((C,), -1, "i64"))
        org_sec[d] = B.where(org_emit, org_sec_v, 0)

        # organic state update (applies to missing events too)
        last = B.where(state_org, et, last_r)
        prev1 = B.where(state_org, last_r, prev1_r)
        prev2 = B.where(state_org, prev1_r, prev2_r)
        reg_traj[d] = regime

    if B.use_cuda:
        B.t.cuda.synchronize()
    sim_seconds = time.perf_counter() - t_sim

    # ======================= assemble columnar events =======================
    frames = []

    signup_ts = _iso(start64, -signup_days.astype(np.float64))
    frames.append(
        pl.DataFrame(
            {
                "customer_key": cust_keys,
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
                pl.when(pl.col("campaign_id").is_null())
                .then(
                    pl.struct(
                        [
                            "touch",
                            "target_key",
                            "freq",
                            "hour",
                            "dow",
                            pl.col("channel_a").alias("channel"),
                            "offer",
                        ]
                    )
                    .struct.json_encode()
                )
                .otherwise(
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
                )
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
    purchase_oid_map: dict[tuple, str] = {}
    if r is not None:
        d_idx, c_idx, f = r
        pid = f["pid"]
        day = d_idx + (f["hour"].astype(np.float64) + 0.75) / 24.0
        oid = np.char.add("o", np.char.zfill((np.arange(len(d_idx)) + 1).astype(str), 9))
        for _k, _sd, _o in zip(cust_keys[c_idx], d_idx.astype(np.int64), oid):
            purchase_oid_map[(str(_k), int(_sd))] = str(_o)
        frames.append(
            pl.DataFrame(
                {
                    "customer_key": cust_keys[c_idx],
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
                    "order_id": oid,
                    "touch": touch_str[f["touch"]],
                }
            )
            .with_columns(pl.Series("event_ts", _iso(start64, day)))
            .with_columns(
                pl.when(pl.col("campaign_id").is_null())
                .then(
                    pl.struct(
                        [
                            "product_id",
                            "category_code",
                            "price",
                            "qty",
                            "discount",
                            "touch",
                            "margin",
                            "brand",
                            "order_id",
                            "value",
                        ]
                    )
                    .struct.json_encode()
                )
                .otherwise(
                    pl.struct(
                        [
                            "product_id",
                            "category_code",
                            "price",
                            "qty",
                            "discount",
                            "touch",
                            "margin",
                            "campaign_id",
                            "brand",
                            "order_id",
                            "value",
                        ]
                    )
                    .struct.json_encode()
                )
                .alias("attrs_json")
            )
            .select(_BASE_COLS)
        )
        purchase_rows = (d_idx, c_idx, pid, f, day)

    # organic
    r = emit_frames(
        org_mask,
        {"et": org_et, "pid": org_pid, "touch": org_touch, "camp": org_camp, "sec": org_sec},
    )
    org_rows = None
    if r is not None:
        d_idx, c_idx, f = r
        et = f["et"]
        pid = f["pid"]
        org_rows = (d_idx, c_idx, et, f)
        day = d_idx.astype(np.float64) + f["sec"].astype(np.float64) / 86400.0
        et_name = np.array(ORG_EVENTS)[et - 1]
        frames.append(
            pl.DataFrame(
                {
                    "customer_key": cust_keys[c_idx],
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
                }
            )
            .with_columns(pl.Series("event_ts", _iso(start64, day)))
            .with_columns(
                pl.when(pl.col("campaign_id").is_null())
                .then(pl.struct(["product_id", "category_code", "touch", "brand"]).struct.json_encode())
                .otherwise(
                    pl.struct(
                        ["product_id", "category_code", "touch", "campaign_id", "brand"]
                    ).struct.json_encode()
                )
                .alias("attrs_json")
            )
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
        ret_oid = [
            purchase_oid_map.get((str(_k), int(_sd)))
            for _k, _sd in zip(cust_keys[c_idx], f["rsrc"].astype(np.int64))
        ]
        frames.append(
            pl.DataFrame(
                {
                    "customer_key": cust_keys[c_idx],
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
                    "order_id": ret_oid,
                    "touch": [None] * len(d_idx),
                    "reason": np.array(RETURN_REASONS)[f["reason"]],
                }
            )
            .with_columns(pl.Series("event_ts", _iso(start64, top)))
            .with_columns(
                pl.struct(
                    [
                        "product_id",
                        "category_code",
                        "reason",
                        "brand",
                        "order_id",
                        "value",
                    ]
                )
                .struct.json_encode()
                .alias("attrs_json")
            )
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
                    "reason": ["size"] * rsrc.height,
                }
            )

    if org_rows is not None:
        od_idx, oc_idx, oet, _of = org_rows
        sup_m = oet == SUPPORT
        if sup_m.any():
            oday = od_idx.astype(np.float64)
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
    """In-process builder retained as the module's public API."""
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
        return None, {"requested": "gpu", "used": "cpu", "note": "`.venv-cudf` or gpu_path_a.py missing"}
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

    t_path = time.perf_counter()
    gpu_res, engine_state = _path_a_via_gpu(tables["events"], tables["customers"], as_of)
    if gpu_res is None:
        features, labels, wide, engine_state2 = build_path_a(
            tables["events"], tables["customers"], as_of, engine=None
        )
        engine_state = {**engine_state, **{k: engine_state2[k] for k in ("used", "note")}}
    else:
        features, labels, wide = gpu_res
    path_a = {"features": features, "labels": labels, "wide": wide}
    path_seconds = time.perf_counter() - t_path

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
            "purchase depends on action quality x regime x path x interactions x long-memory shock.",
        },
    }
    truth = {"meta": meta, "customers": truth_customers}
    (d / f"{a.name}_truth.json").write_text(json.dumps(truth, indent=1))
    (d / "reference_truth.json").write_text(json.dumps(truth, indent=1))

    total_seconds = time.perf_counter() - t_main
    cpu_old = os.environ.get("GEN_CPU_OLD_SECONDS")
    receipt = {
        "device": info.get("device"),
        "sim_seconds": info.get("sim_seconds"),
        "path_a_seconds": round(path_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "path_a_engine": engine_state,
        "n_customers": spec.n_customers,
        "window_days": spec.window_days,
        "seed": spec.seed,
        "events": stream.height,
    }
    if cpu_old is not None:
        try:
            old = float(cpu_old)
            receipt["cpu_old_seconds"] = old
            receipt["speedup_vs_cpu_old"] = round(old / total_seconds, 2)
        except ValueError:
            pass
    for k in (
        "mem_free_bytes",
        "mem_total_bytes",
        "max_memory_allocated_bytes",
        "torch_version",
        "device_name",
    ):
        if k in info:
            receipt[k] = info[k]
    (d / "runtime_receipt.json").write_text(json.dumps(receipt, indent=2))

    manifest = {
        "name": a.name,
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
    print(json.dumps(manifest["counts"] | {"wide_columns": path_a["wide"].width, **receipt}, indent=1))
    if not a.no_validate:
        root = Path(__file__).resolve().parents[2]
        rc = subprocess.run(
            [sys.executable, str(root / "scripts/validate_path_a.py"), "--dir", str(d)],
            check=False,
        ).returncode
        if rc != 0:
            raise SystemExit("PATH A validation failed; see path_a_validation.json")


if __name__ == "__main__":
    main()
