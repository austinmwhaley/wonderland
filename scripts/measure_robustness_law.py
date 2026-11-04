"""B7.2 robustness law: uncertainty and missingness are reported, never faked.

B7.2 is a REPORTING gate: the served object must carry an uncertainty statement for the
unobserved future, and the producer must publish its measured coverage. It does not invent
a new quality bar.

Measured on split B against the full stream after the products' as_of:

  next_time     the published 1-dim `uncertainty:next_time` point forecast (days to the
                next event). An 80% interval is calibrated on split A by the 10/90%
                residual quantiles, then applied unchanged to B; empirical coverage, MAE,
                and the residual quantiles are recorded.
  occurrence    the published `uncertainty:occurrence` logit for an event within the
                median split-A next-event gap (a data-derived horizon); Brier score, AUC,
                and five calibration bins are recorded.
  response      the response head's own calibration block from the products faceplate
                (nominal/empirical coverage, sigma, n) is quoted, not recomputed.
  missingness   the same `stat_bank.level_block` is recomputed from the pre-as_of stream
                with 10% and 25% of each customer's events dropped (seeded); the relative
                L2 change of the explicit block and the share of finite rows are recorded.
                This is the sensitivity of the served statistic to unlogged events, not a
                claim that the dropped events are recovered.

Writes docs/report_cards/experiments/encoder_robustness_law.json (stamped).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cfm_products import _db_path, resolve_served_tag  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_robustness_law.json"
DEFAULT_PRODUCTS = ROOT / "looking_glass/artifacts/cfm_v2k_vf2"
DEFAULT_DEMO = ROOT / "looking_glass/artifacts/demo_v2"
SEED = 0


def _next_event_days(demo: Path, as_of_epoch: float) -> dict[str, float]:
    raw = pl.read_parquet(demo / "customer_event_stream.parquet").with_columns(
        pl.col("event_ts").cast(pl.Datetime).alias("ts")
    )
    fut = (
        raw.filter(pl.col("ts").dt.epoch("s") > as_of_epoch)
        .group_by("customer_key")
        .agg(pl.col("ts").min().alias("next_ts"))
    )
    return {
        str(r["customer_key"]): (float(r["next_ts"].timestamp()) - as_of_epoch) / 86400.0
        for r in fut.iter_rows(named=True)
    }


def _view(products: Path, view: str) -> tuple[dict[str, dict[str, float]], list[str]]:
    import duckdb

    con = duckdb.connect(str(_db_path(str(products))), read_only=True)
    try:
        rows = con.execute(
            "select customer_key, part, embedding from view_embeddings where view = ?", [view]
        ).fetchall()
    finally:
        con.close()
    out: dict[str, dict[str, float]] = {}
    parts: set[str] = set()
    for k, part, emb in rows:
        out.setdefault(str(k), {})[str(part)] = float(np.asarray(emb, dtype=float)[0])
        parts.add(str(part))
    return out, sorted(parts)


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    ap.add_argument("--demo", default=str(DEFAULT_DEMO))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from looking_glass.cfm_config import _to_epoch
    from looking_glass.stat_bank import level_block

    t0 = time.perf_counter()
    demo = Path(a.demo)
    products = Path(a.products)
    tag, reg_path = resolve_served_tag(str(products))
    meta = json.loads(reg_path.read_text())
    as_of = str(meta.get("as_of"))
    as_of_epoch = float(_to_epoch(as_of)) + 86399.0

    import duckdb

    con = duckdb.connect(str(_db_path(str(products))), read_only=True)
    try:
        split = dict(con.execute("select customer_key, split from encoder_samples").fetchall())
    finally:
        con.close()
    akeys = sorted(k for k, v in split.items() if v == "A")
    bkeys = sorted(k for k, v in split.items() if v == "B")

    unc, parts = _view(products, "uncertainty")
    if "next_time" not in parts:
        raise SystemExit("uncertainty view has no next_time part")
    next_days = _next_event_days(demo, as_of_epoch)
    a_common = [k for k in akeys if k in unc and k in next_days]
    b_common = [k for k in bkeys if k in unc and k in next_days]
    if len(a_common) < 30 or len(b_common) < 30:
        raise SystemExit(f"too few customers for coverage: A {len(a_common)} B {len(b_common)}")
    pred_a = np.asarray([unc[k]["next_time"] for k in a_common])
    true_a = np.asarray([next_days[k] for k in a_common])
    resid = true_a - pred_a
    q_lo, q_hi = (float(x) for x in np.quantile(resid, [0.1, 0.9]))
    pred_b = np.asarray([unc[k]["next_time"] for k in b_common])
    true_b = np.asarray([next_days[k] for k in b_common])
    lo, hi = pred_b + q_lo, pred_b + q_hi
    next_block = {
        "part": "uncertainty:next_time",
        "units": "days",
        "nominal": 0.8,
        "calibration": "10/90% residual quantiles fit on split A, applied to split B",
        "q_lo": round(q_lo, 6),
        "q_hi": round(q_hi, 6),
        "mae": round(float(np.mean(np.abs(true_b - pred_b))), 6),
        "coverage": round(_coverage(true_b, lo, hi), 6),
        "n_A": len(a_common),
        "n_B": len(b_common),
        "n": len(b_common),
        "finite": bool(np.isfinite(pred_b).all() and np.isfinite(true_b).all()),
    }

    occ_block = {"part": "uncertainty:occurrence", "complete": False}
    if "occurrence" in parts:
        # data-derived horizon: the median next-event gap on split A (no 30/90 literal)
        horizon = float(np.median(true_a))
        y_occ = np.asarray([1.0 if next_days[k] <= horizon else 0.0 for k in b_common])
        logit = np.asarray([unc[k]["occurrence"] for k in b_common])
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
        bins = np.linspace(0.0, 1.0, 6)
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, 4)
        cal = []
        for b in range(5):
            m = idx == b
            if m.any():
                cal.append(
                    {
                        "bin": b,
                        "n": int(m.sum()),
                        "pred": round(float(p[m].mean()), 6),
                        "obs": round(float(y_occ[m].mean()), 6),
                    }
                )
        occ_block = {
            "part": "uncertainty:occurrence",
            "horizon_days": round(horizon, 6),
            "horizon_rule": "median next-event gap on split A",
            "brier": round(float(brier_score_loss(y_occ, p)), 6),
            "auc": round(float(roc_auc_score(y_occ, p)), 6) if len(np.unique(y_occ)) > 1 else None,
            "calibration_bins": cal,
            "n": int(len(y_occ)),
            "complete": True,
        }

    face = None
    con = duckdb.connect(str(_db_path(str(products))), read_only=True)
    try:
        row = con.execute("select stamp from view_faceplate limit 1").fetchone()
        face = json.loads(row[0]) if row and row[0] else None
    finally:
        con.close()
    resp = (face or {}).get("response_calibration") or {}
    response_block = {
        "source": "view_faceplate.response_calibration",
        "coverage_nominal": resp.get("coverage_nominal"),
        "coverage_empirical": resp.get("coverage_empirical"),
        "sigma_randomized": resp.get("sigma_randomized"),
        "sigma_unrandomized": resp.get("sigma_unrandomized"),
        "n_fit": resp.get("n_fit"),
        "n_calibrate": resp.get("n_calibrate"),
        "complete": bool(resp),
    }

    raw = pl.read_parquet(demo / "customer_event_stream.parquet").with_columns(
        pl.col("event_ts").cast(pl.Datetime).alias("ts")
    )
    raw = raw.filter(pl.col("ts").dt.epoch("s") <= as_of_epoch)
    full_block = level_block(raw, bkeys, as_of_epoch)
    rng = np.random.default_rng(SEED)
    miss = {"complete": True, "seed": SEED, "fractions": [], "n": len(bkeys)}
    for frac in (0.1, 0.25):
        keep = rng.random(raw.height) >= frac
        thin = raw.filter(pl.Series("_keep", keep))
        thin_block = level_block(thin, bkeys, as_of_epoch)
        denom = np.linalg.norm(full_block, axis=1)
        rel = np.linalg.norm(thin_block - full_block, axis=1) / np.maximum(denom, 1e-12)
        miss["fractions"].append(
            {
                "dropped": frac,
                "mean_rel_l2": round(float(np.mean(rel)), 6),
                "median_rel_l2": round(float(np.median(rel)), 6),
                "finite_share": round(float(np.mean(np.isfinite(rel))), 6),
                "n_events_before": int(raw.height),
                "n_events_after": int(thin.height),
            }
        )

    body = {
        "complete": bool(next_block["finite"] and occ_block["complete"] and miss["complete"]),
        "coverage_reported": True,
        "next_time": next_block,
        "occurrence": occ_block,
        "response_head": response_block,
        "missingness": miss,
        "products": products_identity(str(products)),
        "tag": tag,
        "as_of": as_of,
        "n_A": int(len(akeys)),
        "n_B": int(len(bkeys)),
        "why": (
            "B7.2 reports the served uncertainty's measured coverage and its sensitivity to "
            "unlogged events: next-time 80% interval calibrated on A and scored on B; "
            "occurrence calibration on B; the response head's own calibration quoted; the "
            "level block recomputed under seeded 10%/25% event dropping. No missing event "
            "is treated as observed."
        ),
        "device": "cpu",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stamp_merge_write(Path(a.out), body, " ".join(sys.argv), stamp_blocks=False)
    print(json.dumps({k: v for k, v in body.items() if k != "missingness"}, indent=1, default=str))
    print("WROTE", a.out)
    return 0 if body["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
