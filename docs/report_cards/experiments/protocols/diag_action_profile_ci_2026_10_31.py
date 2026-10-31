"""DEC-492 diagnostic: paired CI of a compact action-profile view vs PATH A.

Pre-verdict diagnostic (DEC-462/464 rule), NOT a protocol and NOT an adoption: the
operator's pending ruling asks whether a compact action-profile view may be a separate
judge arm. Before that contract question is worth a ruling, the arm is measured on the
exact frozen judge geometry: fit on split A, score on split B, paired per-customer
squared-error CI against the SAME GBM incumbent (48-column PATH A table, xgboost,
DEC-428), for every supervised trait.

Views are mechanical and history-only (G2): per-level counts of the declared randomized
action fields at/before `as_of`, plus each field's observed mode. No target is read to
build them. The freq-only block is reported as the DEC-464 CEILING, not as a proposal
(a block chosen because it is known to generate x0 would violate G3).

Read-only. No GPU. One JSON receipt beside this script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

T0 = time.perf_counter()

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from measure_battery_law import _ridge_oos  # noqa: E402
from measure_judge_law import (  # noqa: E402
    _arms,
    _gbm_oos,
    _paired_ci,
    _r2,
    _target_matrix,
    controls_matched,
)

from looking_glass.cfm_data import _read_stream  # noqa: E402
from looking_glass.cfm_state import load_frozen_encoder  # noqa: E402
from looking_glass.customer_foundation_model import (  # noqa: E402
    _replay_run_cfg,
    _resolve_registry,
)

PROD = Path("/home/austin_whaley/wonderland_work/cfm_m2_clean")
DEMO = Path("/home/austin_whaley/wonderland_work/demo")
#: DEC-484's declared action family. A shipped view would declare these from the
#: stream's randomization metadata, not from a literal; kept in sync with DEC-484 here.
ACTION_FIELDS = ("freq", "hour", "dow", "channel", "offer")
SEED = 65537  # principle-ok: evaluation-contract seed (same as the frozen judge, DEC-410)


def tick(label: str) -> None:
    print(f"[t] {label}: {time.perf_counter() - T0:.1f}s", flush=True)


manifest = json.loads((DEMO / "manifest.json").read_text())
as_of = float(np.datetime64(manifest["as_of"]).astype("datetime64[s]").astype(np.int64))

a = _arms(PROD / "cfm_products.duckdb", DEMO)
truth, akeys, bkeys = a["truth"], a["akeys"], a["bkeys"]
usable, excluded = _target_matrix(truth, akeys, bkeys)
tick(f"judge arms loaded (A={len(akeys)} B={len(bkeys)}; excluded={excluded})")

tag, meta = _resolve_registry(PROD, None)
model, cfg = load_frozen_encoder(tag, PROD)
_replay_run_cfg(meta, cfg, PROD, cfg)
cfg.db = str(DEMO / "customer_event_stream.parquet")
cfg.out_dir = str(PROD)
df = _read_stream(cfg)
df = df.with_columns(
    pl.col("event_ts").str.to_datetime(time_zone="UTC", strict=False).dt.epoch("s").alias("_ts")
).filter(pl.col("_ts") <= as_of)
tick("stream read + as_of cut")

# One frame per field: customer_key × one column per observed level, renamed to
# "<field>::<level>" so joins across fields cannot collide. Full join over customers;
# a customer with no event in a field gets zero counts (history-only, G2).
levels: dict[str, list[str]] = {}
frames: list[pl.DataFrame] = []
field_cols: dict[str, list[str]] = {}
for name in ACTION_FIELDS:
    if name not in df.columns:
        continue
    sub = df.filter(pl.col(name).is_not_null())
    vals = sorted(sub[name].unique().to_list())
    levels[name] = [str(v) for v in vals]
    piv = (
        sub.group_by(["customer_key", name])
        .len()
        .pivot(values="len", index="customer_key", on=name, aggregate_function="sum")
        .fill_null(0)
    )
    orig = [c for c in piv.columns if c != "customer_key"]
    renamed = {c: f"{name}::{c}" for c in orig}
    piv = piv.rename(renamed)
    field_cols[name] = [renamed[c] for c in orig]
    frames.append(piv)
    tick(f"field {name}: {len(vals)} levels")

prof = frames[0]
for frame in frames[1:]:
    prof = prof.join(frame, on="customer_key", how="full", coalesce=True).fill_null(0)
prof = prof.sort("customer_key")
prof_keys = prof["customer_key"].to_numpy()
count_cols = [c for c in prof.columns if c != "customer_key"]
counts = prof.select(count_cols).to_numpy().astype(np.float64)
col_at = {c: i for i, c in enumerate(count_cols)}
# Per-field mode as a feature: the level index of the most frequent observed value.
mode_cols = []
for name in field_cols:
    idx = [col_at[c] for c in field_cols[name]]
    block = counts[:, idx]
    mode_cols.append(np.where(block.sum(axis=1) > 0, block.argmax(axis=1), -1).astype(np.float64))
modes = np.column_stack(mode_cols) if mode_cols else np.empty((counts.shape[0], 0))
tick(f"profile matrix: {counts.shape[0]} customers × {counts.shape[1]} counts")


def gather(keys, matrix):
    idx = np.searchsorted(prof_keys, np.asarray(keys))
    if not np.array_equal(prof_keys[idx], np.asarray(keys)):
        raise ValueError("a judged customer is missing from the action-profile matrix")
    return matrix[idx]


def view(keys, fields):
    cols = [c for f in fields for c in field_cols[f]]
    return gather(keys, counts[:, [col_at[c] for c in cols]])


def view_profile(keys, fields):
    cols = [c for f in fields for c in field_cols[f]]
    mcols = [i for i, f in enumerate(field_cols) if f in fields]
    return np.hstack(
        [
            gather(keys, counts[:, [col_at[c] for c in cols]]),
            gather(keys, modes[:, mcols]),
        ]
    )


FIELDS = list(field_cols)
views = {
    # DEC-464 ceiling: one declared field's counts (reported, never proposed)
    "freq_counts": (view(akeys, ["freq"]), view(bkeys, ["freq"])),
    # target-blind declared family
    "action_counts": (view(akeys, FIELDS), view(bkeys, FIELDS)),
    # compact profile: counts + observed mode per field
    "action_profile": (view_profile(akeys, FIELDS), view_profile(bkeys, FIELDS)),
}


def _mat(side, source):
    keys = akeys if side == "A" else bkeys
    return np.vstack([source[side][k] for k in keys])


brain_a, brain_b = _mat("A", a["columns"]["brain"]), _mat("B", a["columns"]["brain"])
led_a, led_b = _mat("A", a["ledger"]), _mat("B", a["ledger"])
table_a = np.vstack([a["table"][k] for k in akeys])
table_b = np.vstack([a["table"][k] for k in bkeys])

arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for vname, (va, vb) in views.items():
    arms[vname] = (va, vb)
    arms[f"brain+{vname}"] = (np.hstack([brain_a, va]), np.hstack([brain_b, vb]))
    arms[f"ledger+{vname}"] = (np.hstack([led_a, va]), np.hstack([led_b, vb]))
    arms[f"brain+ledger+{vname}"] = (
        np.hstack([brain_a, led_a, va]),
        np.hstack([brain_b, led_b, vb]),
    )
arms["brain"] = (brain_a, brain_b)
arms["brain+ledger"] = (np.hstack([brain_a, led_a]), np.hstack([brain_b, led_b]))

print("views: " + "; ".join(f"{k}={v[0].shape[1]}d" for k, v in views.items()))
print(f"arms: {len(arms)}; targets: {list(usable)}")

rng = np.random.default_rng(SEED)
results: dict = {}
for tname, (ya, yb) in usable.items():
    p_tab = _gbm_oos(table_a, ya, table_b)
    r2_tab = _r2(yb, p_tab)
    per_arm = {}
    for aname, (xa, xb) in arms.items():
        p, r2 = _ridge_oos(xa, ya, xb, yb)
        if p is None:
            per_arm[aname] = {"r2": None}
            continue
        _mean, lo, hi = _paired_ci((yb - p_tab) ** 2 - (yb - p) ** 2)
        rand, _ = controls_matched(rng, xa, xb)
        _, r2_rand = _ridge_oos(xa, ya, rand, yb)
        per_arm[aname] = {
            "dim": int(xa.shape[1]),
            "r2": round(float(r2), 6),
            "delta_vs_pathA": round(float(r2 - r2_tab), 6),
            "ci_lo": round(float(lo), 6),
            "ci_hi": round(float(hi), 6),
            "random_r2": None if r2_rand is None else round(float(r2_rand), 6),
        }
    results[tname] = {
        "pathA_r2": round(float(r2_tab), 6),
        "n": int(len(yb)),
        "arms": per_arm,
    }
    print(f"\n== {tname} (PATH A {r2_tab:.4f}) ==")
    for aname, r in per_arm.items():
        if r.get("r2") is None:
            print(f"  {aname:28s} degenerate")
            continue
        star = "  CI-EXCLUDES-0" if r["ci_lo"] > 0 else ""
        print(
            f"  {aname:28s} d={r['dim']:4d} R2={r['r2']:.4f} "
            f"delta={r['delta_vs_pathA']:+.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
            f"{star}"
        )

out = {
    "diagnostic": "action-profile view vs PATH A, frozen judge geometry (diagnostic only)",
    "artifact": tag,
    "as_of": manifest["as_of"],
    "splits": {"n_fit_A": len(akeys), "n_score_B": len(bkeys)},
    "targets_excluded": excluded,
    "levels": levels,
    "views": {k: int(v[0].shape[1]) for k, v in views.items()},
    "results": results,
}
receipt = Path(__file__).with_suffix(".json")
receipt.write_text(json.dumps(out, indent=2) + "\n")
tick(f"done -> {receipt}")
