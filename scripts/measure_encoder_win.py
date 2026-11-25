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


def _recent_block(hist: pl.DataFrame, keys: list[str], classes: list[str], k: int) -> np.ndarray:
    """Exact recent-event identity: one-hot of each customer's last k event types.

    A sufficient statistic for next-event is a Markov state — it must carry the exact
    recent-event identity. The frozen embedding recovers the last type only partially
    (measured 0.6253 vs 0.3560 majority on demo_v2), so on the sharpest transition it is
    structurally behind a 1-gram. This block is deterministic and serving-side (history
    <= as_of, the stream's own vocabulary); it adds no fitted parameter.
    """
    idx = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    out = np.zeros((len(keys), k * n_cls), dtype=float)
    if hist.height == 0 or k <= 0:
        return out
    tails = hist.group_by("customer_key", maintain_order=True).agg(pl.col("event_type").tail(k))
    pos = {str(kk): i for i, kk in enumerate(keys)}
    for key, types in zip(
        tails["customer_key"].to_list(), tails["event_type"].to_list(), strict=False
    ):
        i = pos.get(str(key))
        if i is None:
            continue
        tail = list(types)[-k:]
        for j, tp in enumerate(tail):
            c = idx.get(tp)
            if c is not None:
                out[i, j * n_cls + c] = 1.0
    return out


def _softmax_nll(S: np.ndarray, ki: np.ndarray, temp: float) -> float:
    Z = S / max(float(temp), 1e-6)
    Z = Z - Z.max(1, keepdims=True)
    Pm = np.clip(np.exp(Z), P_FLOOR, None)
    Pm = Pm / Pm.sum(1, keepdims=True)
    return float(-np.log(Pm[np.arange(len(ki)), ki]).mean())


def _fit_temperature(S_cal: np.ndarray, ki_cal: np.ndarray) -> float:
    """One scalar softmax temperature, fit on a held-out slice of split A ONLY.

    The shipped consumer fits one-vs-rest ridge on a 0/1 target (squared error) and then
    softmaxes the scores. Squared-error regression on a Bernoulli target shrinks logits
    toward the base rate, so the softmax at T=1 is systematically under-confident and the
    comparison against a proper probability model (the pooled 1-gram) is not like-for-like
    (DEC-646). The temperature restores comparability; it adds no fitted parameter that sees
    split B.
    """
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        lambda tt: _softmax_nll(S_cal, ki_cal, tt), bounds=(0.02, 10.0), method="bounded"
    )
    return float(res.x)


def _fit_vector_scaling(S_cal: np.ndarray, ki_cal: np.ndarray) -> np.ndarray:
    """One multiplicative weight per class (vector scaling, Guo et al. 2017).

    A single softmax temperature is one scalar for classes with very different prevalence
    and sharpness; it fixes the dominant classes and mis-scales the rest (measured on
    demo_v2: view fixed, purchase/search broken — DEC-648). Vector scaling is the canonical
    next step: K weights, fit on a held-out slice of split A only, minimizing NLL there.
    """
    from scipy.optimize import minimize

    K = S_cal.shape[1]

    def nll(w):
        Z = S_cal * w
        Z = Z - Z.max(1, keepdims=True)
        Pm = np.clip(np.exp(Z), P_FLOOR, None)
        Pm = Pm / Pm.sum(1, keepdims=True)
        return float(-np.log(Pm[np.arange(len(ki_cal)), ki_cal]).mean())

    res = minimize(nll, np.ones(K), method="L-BFGS-B", bounds=[(0.02, 50.0)] * K)
    return np.asarray(res.x, dtype=float)


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
    ap.add_argument(
        "--head",
        choices=["ridge", "logistic"],
        default="ridge",
        help="consumer head family: one-vs-rest ridge + softmax (default, DEC-646) or the "
        "canonical multinomial logistic (DEC-649); both linear in the served object",
    )
    ap.add_argument(
        "--calibration",
        choices=["scalar", "vector"],
        default="scalar",
        help="consumer-head calibration form: one temperature (default) or one weight per "
        "class (DEC-648); both fit on a split-A holdout only",
    )
    ap.add_argument(
        "--recent-k",
        type=int,
        default=0,
        help="append the exact last-k event-type indicators to the served object "
        "(0 = off; DEC-645)",
    )
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
    if a.recent_k > 0:
        Ra = _recent_block(hist, a_rows, classes, a.recent_k)
        Rb = _recent_block(hist, b_rows, classes, a.recent_k)
        Ea = np.hstack([Ea, Ra])
        Eb = np.hstack([Eb, Rb])
    excluded = [c for c in classes if n_A[c] < 8]
    kept = [c for c in classes if c not in excluded]
    n_cal = int(round(0.3 * len(a_rows)))
    a_fit, a_cal = a_rows[: len(a_rows) - n_cal], a_rows[len(a_rows) - n_cal :]
    temp = 1.0
    w_vec = None
    if a.head == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def _mk(cstrength):
            return make_pipeline(StandardScaler(), LogisticRegression(C=cstrength, max_iter=4000))

        Eaf = np.vstack([emb[k] for k in a_fit])
        Eac = np.vstack([emb[k] for k in a_cal])
        yaf = np.array([next_of[k] for k in a_fit])
        yac = np.array([next_of[k] for k in a_cal])
        best_c, best_nll = 0.1, float("inf")
        for cstrength in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            m = _mk(cstrength).fit(Eaf, yaf)
            cls_m = list(m.classes_)
            Pp = m.predict_proba(Eac)
            kk = np.array([cls_m.index(v) for v in yac])
            nn = float(-np.log(np.clip(Pp[np.arange(len(yac)), kk], 1e-15, None)).mean())
            if nn < best_nll:
                best_c, best_nll = cstrength, nn
        head = _mk(best_c).fit(Ea, ya)
        cls = list(head.classes_)
        kept = [c for c in kept if c in cls]
        Pp = head.predict_proba(Eb)
        S = np.log(np.clip(Pp[:, [cls.index(c) for c in kept]], 1e-15, None))
        S = S - S.max(1, keepdims=True)
        logit_c = best_c
    else:
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
        logit_c = None
    # DEC-646: one softmax temperature, fit on a held-out slice of split A only.
    # A1 fits the per-class ridge, A2 fits the scalar; split B is never touched.
    if a.head == "ridge" and len(a_cal) >= 50:
        Eaf = np.vstack([emb[k] for k in a_fit])
        Eac = np.vstack([emb[k] for k in a_cal])
        if a.recent_k > 0:
            Eaf = np.hstack([Eaf, _recent_block(hist, a_fit, classes, a.recent_k)])
            Eac = np.hstack([Eac, _recent_block(hist, a_cal, classes, a.recent_k)])
        yaf = np.array([next_of[k] for k in a_fit])
        yac = np.array([next_of[k] for k in a_cal])
        cal_scores = []
        for c in kept:
            p, _ = _ridge_oos(Eaf, (yaf == c).astype(float), Eac, (yac == c).astype(float))
            cal_scores.append(p if p is not None else np.zeros(len(a_cal)))
        S_cal = np.column_stack(cal_scores)
        ki_cal = np.array([kept.index(next_of[k]) for k in a_cal])
        temp = _fit_temperature(S_cal, ki_cal)
        if a.calibration == "vector":
            w_vec = _fit_vector_scaling(S_cal, ki_cal)

    if a.calibration == "vector" and w_vec is not None:
        S = S * w_vec
    else:
        S = S / max(temp, 1e-6)
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
            "served_object": (
                f"frozen bank embedding ({Eb.shape[1] - a.recent_k * len(kept)} dims)"
                + (
                    f" || exact last-{a.recent_k} event-type indicators "
                    f"({a.recent_k * len(kept)} dims, serving-side, history <= as_of)"
                    if a.recent_k > 0
                    else ""
                )
            ),
            "recent_k": a.recent_k,
            "per_class_n_B": {
                c: int((np.array([next_of[k] for k in b_rows]) == c).sum()) for c in kept
            },
            "per_class_nll_enc": {
                c: round(float(nll_enc[np.array([next_of[k] for k in b_rows]) == c].mean()), 6)
                for c in kept
                if bool((np.array([next_of[k] for k in b_rows]) == c).any())
            },
            "per_class_nll_base": {
                c: round(float(nll_base[np.array([next_of[k] for k in b_rows]) == c].mean()), 6)
                for c in kept
                if bool((np.array([next_of[k] for k in b_rows]) == c).any())
            },
            "temperature": round(float(temp), 6),
            "temperature_fit_on": "split A holdout (30%), never split B",
            "head": a.head,
            "logistic_C": logit_c,
            "calibration": a.calibration,
            "calibration_weights": (
                {c: round(float(w), 6) for c, w in zip(kept, w_vec, strict=False)}
                if w_vec is not None
                else None
            ),
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
