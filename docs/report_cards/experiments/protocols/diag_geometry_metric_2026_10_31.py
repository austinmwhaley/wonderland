"""DEC-494/495 diagnostic: geometry is metric-limited, and the distance-pressure proxy is a lookalike.

Two questions, both before any protocol (DEC-462/464 rule), read-only on M2:

1. PROXY VALIDITY. A distance-regression objective needs a self-supervised target that tracks
   the judged trait distance. Measure the observable future-outcome discrepancy (prefix/suffix
   split per customer: last 20% of events) against the lab's trait distance (x0/h0/price/
   return/churn). If the proxy is far below the current state geometry, training on it would
   optimize the wrong target (protocol F's failure mode).

2. METRIC HEADROOM. The judge's distance tasks use Euclidean distance on z-scored columns.
   Fit a PCA-whitening on split A ONLY, freeze it, transform B, and rescore the same tasks.
   This is a Mahalanobis-style metric, not a change to the encoder; any adoption is a
   consumer/artifact contract ruling (stop-and-ask), so this receipt only measures.

No GPU, no writes outside the receipt. Same population/split/statistic as the judge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from measure_judge_law import (  # noqa: E402
    BOOT_PAIRS,
    SEED,
    TARGETS,
    _alignment,
    _arms,
    _fit_scale,
    _paired_stat_ci,
    _z1,
)

from looking_glass.cfm_data import _read_stream  # noqa: E402
from looking_glass.cfm_state import load_frozen_encoder  # noqa: E402
from looking_glass.customer_foundation_model import (  # noqa: E402
    _replay_run_cfg,
    _resolve_registry,
)

PROD = Path("/home/austin_whaley/wonderland_work/cfm_m2_clean")
DEMO = Path("/home/austin_whaley/wonderland_work/demo")
ACTION_FIELDS = ("freq", "hour", "dow", "channel", "offer")
K_SWEEP = (16, 20)


def _trait_distance(truth, akeys, bkeys, fields):
    vec = lambda k: np.asarray([truth[k][f] for f in fields], float)  # noqa: E731
    mu, sd = _fit_scale({k: vec(k) for k in akeys})
    return pdist(np.vstack([_z1(vec(k), mu, sd) for k in bkeys]))


def _ci(arm_d, path_d, truth_d, rng):
    cap = min(truth_d.size, BOOT_PAIRS)
    idx = (
        rng.choice(truth_d.size, size=cap, replace=False)
        if truth_d.size > cap
        else np.arange(truth_d.size)
    )
    lo, hi = _paired_stat_ci(
        lambda q: (
            _alignment(arm_d[idx][q], truth_d[idx][q]),
            _alignment(path_d[idx][q], truth_d[idx][q]),
        ),
        cap,
    )
    return {
        "brain": round(float(_alignment(arm_d, truth_d)), 6),
        "pathA": round(float(_alignment(path_d, truth_d)), 6),
        "ci_lo": round(float(lo), 6),
        "ci_hi": round(float(hi), 6),
        "n_pairs": int(truth_d.size),
        "n_bootstrap_pairs": int(cap),
    }


a = _arms(PROD / "cfm_products.duckdb", DEMO)
truth, akeys, bkeys = a["truth"], a["akeys"], a["bkeys"]
sample = next(iter(truth.values()))
numeric = tuple(sorted(k for k, v in sample.items() if isinstance(v, (int, float))))
future_fields = [f for f in TARGETS if f in numeric]
persona_fields = [f for f in numeric if f not in set(TARGETS) | {"offer_opt"}]
Dt_future = _trait_distance(truth, akeys, bkeys, future_fields)
Dt_persona = _trait_distance(truth, akeys, bkeys, persona_fields)

# ---- 1. proxy validity: the last 20% of each customer's own history ---------------------
tag, meta = _resolve_registry(PROD, None)
model, cfg = load_frozen_encoder(tag, PROD)
_replay_run_cfg(meta, cfg, PROD, cfg)
cfg.db = str(DEMO / "customer_event_stream.parquet")
cfg.out_dir = str(PROD)
df = _read_stream(cfg).filter(pl.col("customer_key").is_in(list(bkeys)))
df = df.sort(["customer_key", "event_ts"]).with_columns(
    pl.int_range(pl.len()).over("customer_key").alias("_i"),
    pl.len().over("customer_key").alias("_n"),
)
fut = df.filter(pl.col("_i") >= (0.8 * pl.col("_n")).cast(pl.Int64))
frames = []
for name in ("event_type", *[f for f in ACTION_FIELDS if f in fut.columns]):
    sub = fut.filter(pl.col(name).is_not_null())
    piv = (
        sub.group_by(["customer_key", name])
        .len()
        .pivot(values="len", index="customer_key", on=name, aggregate_function="sum")
        .fill_null(0)
        .sort("customer_key")
    )
    orig = [c for c in piv.columns if c != "customer_key"]
    piv = piv.rename({c: f"{name}::{c}" for c in orig})
    frames.append(piv)
proxy = frames[0]
for frame in frames[1:]:
    proxy = proxy.join(frame, on="customer_key", how="full", coalesce=True).fill_null(0)
proxy = proxy.sort("customer_key")
proxy_keys = proxy["customer_key"].to_numpy()
proxy_X = np.log1p(
    proxy.select([c for c in proxy.columns if c != "customer_key"]).to_numpy().astype(float)
)
idx = np.searchsorted(proxy_keys, np.asarray(bkeys))
present = idx < len(proxy_keys)
present[present] &= proxy_keys[idx[present]] == np.asarray(bkeys)
proxy_kb = [k for k, p in zip(bkeys, present) if p]

# trait distance restricted to the customers present in the proxy
vec = lambda k: np.asarray([truth[k][f] for f in future_fields], float)  # noqa: E731
mu, sd = _fit_scale({k: vec(k) for k in akeys})
Dt_proxy = pdist(np.vstack([_z1(vec(k), mu, sd) for k in proxy_kb]))
proxy_score = _alignment(pdist(proxy_X[idx[present]]), Dt_proxy)
# event-type-only comparator (no action fields)
etype_cols = [c for c in proxy.columns if c.startswith("event_type::")]
etype_X = np.log1p(proxy.select(etype_cols).to_numpy().astype(float))
etype_score = _alignment(pdist(etype_X[idx[present]]), Dt_proxy)

# ---- 2. metric headroom: frozen PCA-whitening fit on A -----------------------------------
muW, sdW = _fit_scale({k: a["table"][k] for k in akeys})
Dw_future = pdist(_z1(np.vstack([a["table"][k] for k in bkeys]), muW, sdW))
Dw_persona = Dw_future
rng = np.random.default_rng(SEED)
metric = {}
for col in ("brain", "brain_ledger"):
    XA = np.vstack([a["columns"][col]["A"][k] for k in akeys])
    XB = np.vstack([a["columns"][col]["B"][k] for k in bkeys])
    mu, sd = _fit_scale({k: a["columns"][col]["A"][k] for k in akeys})
    ZA, ZB = _z1(XA, mu, sd), _z1(XB, mu, sd)
    metric[col] = {}
    for k in K_SWEEP:
        p = PCA(n_components=k, whiten=True, random_state=SEED).fit(ZA)
        Db = pdist(p.transform(ZB))
        metric[col][f"pca_whiten_{k}"] = {
            "future_geometry": _ci(Db, Dw_future, Dt_future, rng),
            "persona_geometry": _ci(Db, Dw_persona, Dt_persona, rng),
        }

out = {
    "diagnostic": "geometry metric + distance-pressure proxy (diagnostic only)",
    "artifact": tag,
    "splits": {"n_fit_A": len(akeys), "n_score_B": len(bkeys)},
    "trait_fields": {"future": future_fields, "persona": persona_fields},
    "proxy": {
        "definition": "per-customer last 20% of events; log1p counts by event type + action field levels",
        "action_plus_type_alignment": round(float(proxy_score), 6),
        "event_type_only_alignment": round(float(etype_score), 6),
        "n_customers": len(proxy_kb),
    },
    "metric": metric,
}
receipt = Path(__file__).with_suffix(".json")
receipt.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
