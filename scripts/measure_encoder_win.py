#!/usr/bin/env python3
"""The gate that certifies the ENCODER (DEC-635) — the encoder beats the 1-gram baseline.

The frozen gates certify the instrument — the pipeline produces a frozen, certifiable donor.
This producer measures the OUTCOME: the encoder's frozen output beats the 1-gram baseline
on the customer event stream (the demo_v2 prototype) — the ridge head (the shipping consumer
head) beats the pooled 1-gram on held-out B rows, with a paired bootstrap CI.

The protocol is the frozen lab_measure._measure_pair protocol — the ridge head, the A/B
split, the paired CI — applied to the certified demo_v2 customer event stream. No
re-train; the encoder is the frozen bank.

    python3 scripts/measure_encoder_win.py [--products DIR] [--demo DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import safe_run as SR  # noqa: E402
from looking_glass.stream_adapter import to_ts_expr  # noqa: E402
from measure_battery_law import N_BOOT, SEED, _ridge_oos  # noqa: E402

# frozen A/B customer partition (mirrors the frozen split protocol — DEC-576/611)
A_FRAC = 0.7
P_FLOOR = 1e-15


def _split_customers(keys: list[str]) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(keys))
    cut = int(A_FRAC * len(keys))
    return [keys[i] for i in sorted(order[:cut])], [keys[i] for i in sorted(order[cut:])]


def _load_served(products: Path) -> dict:
    import duckdb

    db = products / "cfm_products.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    emb = {
        k: np.asarray(v, float)
        for k, v in con.execute("select customer_key, embedding from state_embeddings").fetchall()
    }
    con.close()
    return emb


def _bigram_counts(hist: pl.DataFrame, classes: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    idx = {c: i for i, c in enumerate(classes)}
    T = np.ones((len(classes), len(classes)))
    if hist.height == 0:
        return T, {}
    keys_h = hist["customer_key"].to_numpy()
    types_h = np.array([idx[t] for t in hist["event_type"].to_numpy()])
    boundary = np.ones(len(types_h), bool)
    boundary[1:] = keys_h[1:] != keys_h[:-1]
    pos = np.nonzero(~boundary & (np.arange(len(types_h)) > 0))[0]
    np.add.at(T, (types_h[pos - 1], types_h[pos]), 1.0)
    is_last = np.zeros(len(types_h), bool)
    is_last[:-1] = boundary[1:]
    is_last[-1] = True
    last_hist = {str(k): int(t) for k, t in zip(keys_h[is_last], types_h[is_last], strict=False)}
    return T, last_hist


def _paired_ci(deltas: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    if len(deltas) < 2:
        return None, None
    rng = np.random.default_rng(SEED)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = deltas[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _round(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def main() -> int:
    SR.ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="looking_glass/artifacts/cfm_v2k_vf2_bank")
    ap.add_argument("--demo", default="looking_glass/artifacts/demo_v2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.perf_counter()
    demo = Path(a.demo)
    products = Path(a.products)
    manifest = json.loads((demo / "manifest.json").read_text())
    as_of = datetime.strptime(manifest["as_of"], "%Y-%m-%d")

    stream = pl.read_parquet(demo / "path_b" / "path_b_customer_event_stream.parquet")
    keys = sorted(str(k) for k in stream["customer_key"].unique().to_list())
    akeys, bkeys = _split_customers(keys)

    fut = stream.filter(to_ts_expr(pl.col("event_ts")) > as_of).sort(
        ["customer_key", "event_ts", "event_type"]
    )
    nxt = fut.unique("customer_key", keep="first")
    next_of = dict(zip(nxt["customer_key"].to_list(), nxt["event_type"].to_list(), strict=False))
    classes = sorted(set(next_of.values()))

    emb = _load_served(products)
    hist = (
        stream.filter(to_ts_expr(pl.col("event_ts")) <= as_of)
        .filter(pl.col("event_type").is_in(classes))
        .sort(["customer_key", "event_ts", "event_type"])
    )
    T, last_hist = _bigram_counts(hist, classes)
    P = T / T.sum(1, keepdims=True)

    b_rows = [k for k in bkeys if k in next_of and k in emb]
    a_rows = [k for k in akeys if k in next_of and k in emb]
    n_A = {c: int(sum(1 for k in a_rows if next_of[k] == c)) for c in classes}

    if not a_rows or not b_rows or not any(n >= 8 for n in n_A.values()):
        print("fatal: no A/B rows with a next type", file=sys.stderr)
        return 1

    nll_base = np.array(
        [
            -math.log(P[last_hist[k], classes.index(next_of[k])])
            if k in last_hist and next_of[k] in classes
            else np.nan
            for k in b_rows
        ]
    )

    ya = np.array([next_of[k] for k in a_rows])
    yb = np.array([next_of[k] for k in b_rows])
    Ea = np.vstack([emb[k] for k in a_rows])
    Eb = np.vstack([emb[k] for k in b_rows])
    excluded = [c for c in classes if n_A[c] < 8]
    kept = [c for c in classes if c not in excluded]
    scores = []
    for c in kept:
        p, _ = _ridge_oos(Ea, (ya == c).astype(float), Eb, (yb == c).astype(float))
        if p is None:
            excluded.append(c)
            kept = [k2 for k2 in kept if k2 != c]
            continue
        scores.append(p)
    if not kept:
        print("fatal: no class head fit", file=sys.stderr)
        return 1
    S = np.column_stack(scores)
    S = S - S.max(1, keepdims=True)
    Pm = np.exp(S)
    Pm = np.clip(Pm, P_FLOOR, None)
    Pm /= Pm.sum(1, keepdims=True)
    ki = np.array([kept.index(next_of[k]) for k in b_rows])
    nll_enc = -np.log(Pm[np.arange(len(b_rows)), ki])

    m = ~np.isnan(nll_base) & ~np.isnan(nll_enc)
    nll_base, nll_enc = nll_base[m], nll_enc[m]
    delta = nll_enc - nll_base
    ci_lo, ci_hi = _paired_ci(delta) if len(delta) else (None, None)
    beats_unigram = bool(ci_hi is not None and ci_hi < 0)

    body = {
        "kind": "encoder_win",
        "protocol": "DEC-635: the ridge head (the shipping consumer head) beats the pooled 1-gram baseline on the customer event stream — the ridge head is fit on split A, scored on split B, the paired bootstrap CI of (nll_enc - nll_base) must exclude 0. The outcome: the encoder is a foundation, not a green receipt.",
        "encoder_win": {
            "complete": True,
            "beats_unigram": beats_unigram,
            "beats_gram1": None,
            "paired_ci_enc_minus_base": [ci_lo, ci_hi],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "nll_enc": _round(nll_enc.mean()) if len(m) else None,
            "nll_base": _round(nll_base.mean()) if len(m) else None,
            "n_A": len(a_rows),
            "n_B_rows": int(len(m)),
            "n_customers": len(keys),
            "classes": classes,
            "classes_excluded": excluded,
            "per_class_n_A": n_A,
        },
        "seconds": round(time.perf_counter() - t0, 1),
        "products": {"path": str(products), "tag": None},
        "complete": True,
    }

    out_path = (
        Path(a.out)
        if a.out
        else ROOT / "docs" / "report_cards" / "experiments" / "encoder_win.json"
    )
    from measure_input_law import stamp_merge_write

    stamp_merge_write(
        out_path, body, " ".join([Path(__file__).name] + sys.argv[1:]), stamp_blocks=False
    )
    print(f"encoder-win -> {out_path} ({body['seconds']}s)")
    print(
        f"  beats_unigram={beats_unigram}  paired CI [{ci_lo}, {ci_hi}]  nll_enc={body['encoder_win']['nll_enc']}  nll_base={body['encoder_win']['nll_base']}  n_B_rows={body['encoder_win']['n_B_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
