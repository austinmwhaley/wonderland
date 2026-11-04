#!/usr/bin/env python3
"""THE SHAPE BATTERY (DEC-588 Phase 0) — which shape does the frozen E carry?

A development diagnostic, not a B-row producer: no card row names this receipt, and no lab
receipt flips a B-row verdict (DEC-576). It measures, on cfm_v2k, how much of a customer's
FUTURE SHAPE (the next 90 days) and PERSONA is linearly decodable from three feature sets:

  * E       — the frozen served state (state_embeddings, 256-d),
  * counter — the exact history distributional statistics (type histogram, transition matrix,
              gap/value quantiles, counts, recency/tenure) — the B3.2 "exact counter",
  * path_a  — the PATH A wide-table behavioural columns (no labels, no persona).

Targets (future window `(as_of, as_of+90d]`):
  * future_type_hist      — normalized counts over 6 type classes (5 recurring + other)
  * future_transition     — per-customer row-normalized transition matrix (6x6, +1 smoothing)
  * future_gap_q          — quantiles of log1p(inter-event gap days)
  * future_value_q        — quantiles of log1p(purchase value)
  * persona_segment/tier  — one-hot classification accuracy (they are a bijection; both kept)
  * persona_region        — DEGENERATE (constant "north") -> excluded and reported.

Protocol: frozen 70/30 A/B split from `encoder_samples`, fit on A, score on B, paired
bootstrap CI (2000) of E minus the counter. A shuffled-target fit on A is the control
(the head must read ~0 on B). No B data is used for fitting or tuning.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cfm_products as CP  # noqa: E402
from looking_glass.stream_adapter import to_ts_expr  # noqa: E402
from measure_battery_law import N_BOOT, SEED, _ridge_head  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from safe_run import ensure_boundary  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "shape_battery.json"
AS_OF = datetime(2025, 1, 1)  # naive: the canonical stream ts is naive-UTC
FUTURE_DAYS = 90
TYPES = ("view", "search", "email_send", "purchase", "return")
K = len(TYPES) + 1  # + other
QUANTILES = (0.25, 0.5, 0.75, 0.9)
A_FRAC = 0.7


def _round(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def _load_stream(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path).select(["customer_key", "event_ts", "event_type", "value"])
    df = df.with_columns(to_ts_expr(pl.col("event_ts")).alias("ts")).drop("event_ts")
    return df.with_columns(pl.col("value").fill_null(0.0))


def _index(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    keys = df["customer_key"].unique().sort()
    kmap = pl.DataFrame({"customer_key": keys, "ci": np.arange(keys.len(), dtype=np.int64)})
    tmap = pl.DataFrame(
        {
            "event_type": list(TYPES) + ["__other__"],
            "ti": np.arange(K, dtype=np.int64),
        }
    )
    out = (
        df.join(kmap, on="customer_key")
        .join(tmap, on="event_type", how="left")
        .with_columns(pl.col("ti").fill_null(K - 1))
        .sort(["ci", "ts"])
    )
    return out, keys


def _quantiles(ci: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """Per-customer quantiles of x (vectorized group_by; NaN where a customer has none)."""
    if len(x) == 0:
        return np.full((n, len(QUANTILES)), np.nan)
    sub = pl.DataFrame({"ci": ci, "x": x})
    agg = sub.group_by("ci").agg(
        [pl.col("x").quantile(q).alias(f"q{i}") for i, q in enumerate(QUANTILES)]
    )
    out = np.full((n, len(QUANTILES)), np.nan)
    idx = agg["ci"].to_numpy()
    out[idx] = agg.drop("ci").to_numpy()
    return out


def _aggregate(df: pl.DataFrame, n: int) -> dict:
    ci = df["ci"].to_numpy()
    ti = df["ti"].to_numpy()
    val = df["value"].to_numpy().astype(float)

    hist = np.zeros((n, K))
    np.add.at(hist, (ci, ti), 1.0)

    same = np.zeros(len(df), dtype=bool)
    same[1:] = ci[1:] == ci[:-1]
    prev = np.full(len(df), -1, dtype=np.int64)
    prev[1:] = ti[:-1]
    trans = np.zeros((n, K * K))
    m = same
    np.add.at(trans, (ci[m], prev[m] * K + ti[m]), 1.0)

    cnt = np.bincount(ci, minlength=n).astype(float)
    is_purchase = ti == TYPES.index("purchase")
    vsum = np.bincount(ci[is_purchase], weights=val[is_purchase], minlength=n)

    t = df["ts"]
    gap_days = (t.diff().dt.total_seconds() / 86400.0).to_numpy()  # principle-ok: unit
    ok_gap = same & (gap_days > 0) & np.isfinite(gap_days)
    gap_q = _quantiles(ci[ok_gap], np.log1p(gap_days[ok_gap]), n)

    ok_val = is_purchase & (val > 0)
    val_q = _quantiles(ci[ok_val], np.log1p(val[ok_val]), n)

    bounds = df.group_by("ci").agg([pl.col("ts").min().alias("t0"), pl.col("ts").max().alias("t1")])
    t0 = np.full(n, np.nan)
    t1 = np.full(n, np.nan)
    idx = bounds["ci"].to_numpy()
    t0[idx] = bounds["t0"].dt.epoch("s").to_numpy().astype(float)
    t1[idx] = bounds["t1"].dt.epoch("s").to_numpy().astype(float)

    per_type = np.column_stack(
        [np.bincount(ci[ti == k], minlength=n).astype(float) for k in range(K)]
    )
    return {
        "hist": hist,
        "trans": trans.reshape(n, K, K),
        "cnt": cnt,
        "vsum": vsum,
        "gap_q": gap_q,
        "val_q": val_q,
        "t0": t0,
        "t1": t1,
        "per_type": per_type,
    }


def _counter_features(h: dict, as_of_epoch: float) -> np.ndarray:
    hist = h["hist"] / np.clip(h["hist"].sum(1, keepdims=True), 1.0, None)
    tr = h["trans"] + 1.0
    tr = tr / tr.sum(2, keepdims=True)
    gap = np.nan_to_num(h["gap_q"], nan=0.0)
    valq = np.nan_to_num(h["val_q"], nan=0.0)
    ten = (as_of_epoch - h["t0"]) / 86400.0
    rec = (as_of_epoch - h["t1"]) / 86400.0
    have_gap = (~np.isnan(h["gap_q"])).any(1).astype(float)[:, None]
    have_val = (~np.isnan(h["val_q"])).any(1).astype(float)[:, None]
    counts = np.column_stack(
        [
            np.log1p(h["cnt"]),
            np.log1p(h["per_type"]),
            np.log1p(np.clip(h["vsum"], 0, None)),
            np.log1p(np.clip(ten, 0, None)),
            np.log1p(np.clip(rec, 0, None)),
        ]
    )
    return np.nan_to_num(
        np.column_stack([hist, tr.reshape(len(hist), -1), gap, valq, have_gap, have_val, counts])
    )


def _shape_targets(f: dict) -> dict:
    hist = f["hist"] / np.clip(f["hist"].sum(1, keepdims=True), 1.0, None)
    tr = f["trans"] + 1.0
    tr = tr / tr.sum(2, keepdims=True)
    return {
        "type_hist": (hist, False),
        "transition": (tr.reshape(len(tr), -1), False),
        "gap_q": (f["gap_q"], False),
        "value_q": (f["val_q"], False),
    }


def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(y), k))
    ok = (y >= 0) & (y < k)
    out[np.arange(len(y))[ok], y[ok]] = 1.0
    return out


def _r2_parts(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-customer SSE/SST (uniform over outputs) so the bootstrap is a vectorized gather."""
    sse = ((y - p) ** 2).mean(1)
    sst = ((y - y.mean(0, keepdims=True)) ** 2).mean(1)
    return sse, sst


def _boot_r2(sse: np.ndarray, sst: np.ndarray, idx: np.ndarray) -> np.ndarray:
    a = sse[idx].sum(1)
    b = sst[idx].sum(1)
    return 1.0 - a / np.clip(b, 1e-12, None)


def _boot_acc(correct: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return correct[idx].mean(1)


def _fit_score(Xa, ya, Xb, yb, rng, classification: bool):
    head = _ridge_head()
    head.fit(Xa, ya)
    pred = head.predict(Xb)
    if classification:
        got = pred.argmax(1)
        want = yb.argmax(1)
        correct = (got == want).astype(float)
        return float(correct.mean()), correct, None, None
    sse, sst = _r2_parts(yb, pred)
    return float(1.0 - sse.sum() / max(sst.sum(), 1e-12)), None, sse, sst


def main(argv=None) -> int:
    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="/home/austin_whaley/wonderland_work/cfm_v2k")
    ap.add_argument(
        "--stream",
        default="/home/austin_whaley/wonderland_work/demo_v2/customer_event_stream.parquet",
    )
    ap.add_argument(
        "--path-a",
        default="/home/austin_whaley/wonderland_work/demo_v2/path_a/path_a_wide_table.parquet",
    )
    ap.add_argument(
        "--features",
        default="/home/austin_whaley/wonderland_work/demo_v2/path_a/customer_features.parquet",
    )
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    as_of_epoch = AS_OF.replace(tzinfo=timezone.utc).timestamp()
    rng = np.random.default_rng(SEED)

    raw = _load_stream(Path(a.stream))
    df, keys = _index(raw)
    n = len(keys)
    hist_df = df.filter(pl.col("ts") <= AS_OF)
    fut_df = df.filter(
        (pl.col("ts") > AS_OF) & (pl.col("ts") <= AS_OF + timedelta(days=FUTURE_DAYS))
    )
    h = _aggregate(hist_df, n)
    f = _aggregate(fut_df, n)
    X_counter = _counter_features(h, as_of_epoch)

    tables = CP.stored_tables(a.products)
    key_list = keys.to_list()
    emb = tables["embeddings"]
    missing = [k for k in key_list if k not in emb]
    if missing:
        raise SystemExit(f"{len(missing)} customers have no served E (first: {missing[0]})")
    X_E = np.vstack([emb[k] for k in key_list])

    def _split_mask(want: str) -> np.ndarray:
        return np.array([tables["split"].get(k) == want for k in key_list])

    ma, mb = _split_mask("A"), _split_mask("B")

    # PATH A behavioural columns only (no persona, no labels, no trailing-window values).
    wa = pl.read_parquet(a.path_a)
    drop = {"customer_key", "segment", "loyalty_tier", "signup_ts", "region"}
    cols = [
        c
        for c in wa.columns
        if c not in drop and not c.startswith("y_") and c not in ("value_30d", "value_90d")
    ]
    wa = wa.select(["customer_key", *cols])
    wa = pl.DataFrame({"customer_key": key_list}).join(wa, on="customer_key", how="left")
    X_path = wa.select(cols).fill_null(0.0).to_numpy().astype(float)

    # persona targets
    feat = pl.read_parquet(a.features).select(["customer_key", "segment", "loyalty_tier", "region"])
    feat = pl.DataFrame({"customer_key": key_list}).join(feat, on="customer_key", how="left")
    seg = feat["segment"].cast(pl.Int64).fill_null(0).to_numpy() - 1
    tier_map = {"bronze": 0, "silver": 1, "gold": 2, "platinum": 3}
    tier = feat["loyalty_tier"].replace_strict(tier_map, default=-1).to_numpy()

    targets = {f"future_{k}": v for k, v in _shape_targets(f).items()}
    targets = {k: (v[0], v[1]) for k, v in targets.items() if np.isfinite(v[0]).any()}
    targets["persona_segment"] = (_one_hot(seg, 4), True)
    targets["persona_tier"] = (_one_hot(tier, 4), True)

    arms = {"E": X_E, "counter": X_counter, "path_a": X_path}
    results: dict = {}
    excluded: dict = {"persona_region": "degenerate: constant 'north' (no variance)"}

    for name, (y, classification) in targets.items():
        if classification:
            ok = np.isfinite(y).all(1) & (y.sum(1) > 0)
            lo = 4
        else:
            ok = np.isfinite(y).all(1)
            lo = int(y.shape[1])
        ra, rb = ma & ok, mb & ok
        if int(ra.sum()) < 100 or int(rb.sum()) < 50:
            excluded[name] = f"too few rows (A {int(ra.sum())}, B {int(rb.sum())})"
            continue
        out = {"dims": lo, "n_A": int(ra.sum()), "n_B": int(rb.sum()), "arms": {}}
        idx_boot = rng.integers(0, int(rb.sum()), size=(N_BOOT, int(rb.sum())))
        parts = {}
        for arm, X in arms.items():
            point, correct, sse, sst = _fit_score(X[ra], y[ra], X[rb], y[rb], rng, classification)
            if classification:
                boot = _boot_acc(correct, idx_boot)
            else:
                boot = _boot_r2(sse, sst, idx_boot)
            out["arms"][arm] = {
                "point": _round(point),
                "ci95": [_round(np.percentile(boot, 2.5)), _round(np.percentile(boot, 97.5))],
                "boot": boot,
            }
            parts[arm] = (point, boot)
        # paired delta E - counter (same resamples) and E - path_a
        for other in ("counter", "path_a"):
            d = parts["E"][1] - parts[other][1]
            out[f"delta_E_minus_{other}"] = {
                "point": _round(parts["E"][0] - parts[other][0]),
                "ci95": [_round(np.percentile(d, 2.5)), _round(np.percentile(d, 97.5))],
            }
        # shuffled-target control: fit E on A with y permuted, score on B
        perm = rng.permutation(int(ra.sum()))
        shuf, _, _, _ = _fit_score(
            arms["E"][ra], y[ra][perm], arms["E"][rb], y[rb], rng, classification
        )
        out["shuffle_control_E"] = _round(shuf)
        for arm in arms:
            out["arms"][arm].pop("boot")
        results[name] = out

    # Does E encode the HISTORY's own shape at all? (sufficiency, not forecasting.)
    # The counter arm is the target itself for these statistics (R^2 = 1 by construction),
    # so it is omitted; the test is E against a shuffled-target control.
    for name, (y, _cls) in _shape_targets(h).items():
        ok = np.isfinite(y).all(1)
        ra, rb = ma & ok, mb & ok
        if int(ra.sum()) < 100 or int(rb.sum()) < 50:
            continue
        point, _c, sse, sst = _fit_score(X_E[ra], y[ra], X_E[rb], y[rb], rng, False)
        idx_h = rng.integers(0, int(rb.sum()), size=(N_BOOT, int(rb.sum())))
        boot = _boot_r2(sse, sst, idx_h)
        perm = rng.permutation(int(ra.sum()))
        shuf, _c2, _s2, _t2 = _fit_score(X_E[ra], y[ra][perm], X_E[rb], y[rb], rng, False)
        results[f"history_{name}"] = {
            "dims": int(y.shape[1]),
            "n_A": int(ra.sum()),
            "n_B": int(rb.sum()),
            "arms": {
                "E": {
                    "point": _round(point),
                    "ci95": [
                        _round(np.percentile(boot, 2.5)),
                        _round(np.percentile(boot, 97.5)),
                    ],
                }
            },
            "shuffle_control_E": _round(shuf),
            "target_defining": True,
        }

    body = {
        "kind": "shape_battery_diagnostic",
        "protocol": (
            "DEC-588 Phase 0: linear decodability of future shape and persona from "
            "E vs the exact counter vs PATH A, on cfm_v2k; fit on split A, score on B, "
            "paired bootstrap; a development instrument, never B-row evidence (DEC-576)"
        ),
        "as_of": AS_OF.date().isoformat(),
        "future_days": FUTURE_DAYS,
        "n_customers": n,
        "n_A": int(ma.sum()),
        "n_B": int(mb.sum()),
        "types": list(TYPES) + ["other"],
        "quantiles": list(QUANTILES),
        "n_boot": N_BOOT,
        "seed": SEED,
        "targets": results,
        "excluded": excluded,
        "caveats": {
            "persona": (
                "CONTAMINATED on cfm_v2k: segment/loyalty_tier/region are carried in every "
                "event's event_attributes, so E reads them directly. This is identity "
                "leakage (cf. B4.3 check-2), NOT a shape measure; the counter arm has no "
                "persona fields. Do not read the persona rows as learned shape."
            ),
            "future_targets": (
                "aligned with the training objectives (rate/value/survival heads), so E's "
                "absolute R^2 is inflated; the comparator (counter) is the claim, not the "
                "absolute value."
            ),
            "history_targets": (
                "the counter arm is target-defining (R^2=1 by construction) and omitted; "
                "the E row tests whether the state encodes the history's own statistics."
            ),
        },
        "products_identity": products_identity(Path(a.products)),
        "discipline": (
            "a development diagnostic; no B-row source names this receipt; a movement here "
            "can only SUGGEST the next real-data run (DEC-576/588)"
        ),
        "complete": bool(results),
        "seconds": 0.0,
    }
    body["seconds"] = round(time.perf_counter() - t0, 1)
    stamp_merge_write(
        Path(a.out), body, " ".join([Path(__file__).name] + sys.argv[1:]), stamp_blocks=False
    )
    print(f"shape battery -> {a.out} ({body['seconds']}s)")
    for name, r in results.items():
        e = r["arms"]["E"]
        if "counter" in r["arms"]:
            c = r["arms"]["counter"]
            print(
                f"  {name:20s} n_B {r['n_B']:5d}  E {e['point']:+.4f} {e['ci95']}  "
                f"counter {c['point']:+.4f}  dE-counter "
                f"{r['delta_E_minus_counter']['point']:+.4f} "
                f"{r['delta_E_minus_counter']['ci95']}  shuffle {r['shuffle_control_E']:+.4f}"
            )
        else:
            print(
                f"  {name:20s} n_B {r['n_B']:5d}  E {e['point']:+.4f} {e['ci95']}  "
                f"(target-defining)  shuffle {r['shuffle_control_E']:+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
