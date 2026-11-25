#!/usr/bin/env python3
"""WHERE do purchase / search / view lose to the pooled 1-gram? — a read-only DIAGNOSTIC.

B9.1 passes (the served bank beats the pooled 1-gram by 0.250 nats on split B) but three classes
still trail their per-class 1-gram baseline, and every head variant trades one class for another.
This producer localises the shortfall with numbers instead of another head:

  1. Reproduction — the frozen split (`measure_encoder_win._split_customers`, SEED) and the exact
     served object (179-dim `state_embeddings`), asserted against the certified receipt.
  2. Per class {purchase, search, view}: the adopted multinomial-logistic head's confusion (which
     classes steal c's mass), the mean predicted probability of c on true-c rows, the same three
     numbers for the pooled 1-gram, and the exact level/dispersion decomposition of the gap.
     Reference comparators (NOT gates) — the customer-level marginal, the customer-level bigram,
     the frequency-only head, and a serving-side frequency+ELAPSED-TIME head — separate "the
     1-gram is better informed" from "the 1-gram is tilted" and from "the bank lost recency".
  3. Nonlinear diagnostic: `HistGradientBoostingClassifier` (multiclass + per-class binary) fit on
     split A, scored on split B. If GBM closes a per-class gap the information is present but not
     linearly readable; if it does not, the linear head is already at the accessible ceiling.
     *** DEC-407: MLP/GBM never ship. This is a DIAGNOSTIC, not a consumer, not a gate. ***
  4. Block ablation: the `bank_receipt.json` block slices (shape/level/response/rfm/action/counter/
     profile/numeric_stats/event_mass/return_ratio); leave-one-block-out AND block-only, so we see
     which block carries (or dilutes) each class's signal.
  5. Last-event oracle: a probe from the bank to the exact last event type, its confident subset
     (threshold derived on the split-A holdout, never B), the 1-gram's per-class nll restricted to
     that subset, the head's gap on confident vs unconfident rows, and the direct oracle head
     (bank || exact last-event one-hot) — is the residual gap last-event uncertainty?

Deterministic (SEED = the frozen evaluation-contract seed), CPU only, read-only: no file is
written (--json dumps the same body to stdout). Under ~10 minutes.

    .venv/bin/python scripts/diagnose_class_gap.py [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Small matrices on a 20-core box: the default BLAS/OpenMP pools oversubscribe and can be up to
# ~200x SLOWER than one thread (measured here: one 2450x179 multinomial-logistic fit 0.011 s at
# OPENBLAS_NUM_THREADS=1 vs 2.12 s at 8). Cap the pools BEFORE numpy/sklearn are imported; the
# operator can still override. This is a cost/determinism fix, not a quality trade.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_battery_law import SEED  # noqa: E402  (frozen evaluation-contract seed)
from measure_encoder_win import (  # noqa: E402
    _bigram_counts,
    _load_served,
    _split_customers,
)

#: the head's C grid, identical to `measure_encoder_win` so the adopted head is REPRODUCED, not
#: re-tuned (the frozen receipt selected C=0.1 on the split-A holdout).
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)  # principle-ok: protocol grid, reproduces the receipt
P_FLOOR = 1e-15
SECONDS_PER_DAY = 86400.0  # principle-ok: physical unit (s -> days), not a cadence assumption
TARGET_CLASSES = ("purchase", "search", "view")
#: probe precision that defines "the bank can read the exact last event on this row" (evaluation
#: contract for the confident subset; the threshold itself is derived on the A holdout).
PROBE_PRECISION = 0.95  # principle-ok: evaluation-contract precision target for the subset
PROBE_MIN_COVERAGE = 0.05  # principle-ok: a subset below this is not a subset, it is an anecdote
MIN_CLAIM_N = (
    8  # principle-ok: per-class floor below which a per-class nll is reported, not claimed
)


def _r(x, k: int = 6):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, k)


def _lr(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))


def _nll(proba: np.ndarray, cols: list[int], ki: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(proba[np.arange(len(ki)), np.asarray(cols)[ki]], P_FLOOR, None))


def _select_c(E: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    """Reproduce the adopted head's C selection (split-A tail holdout, never split B)."""
    n_cal = int(round(0.3 * len(y)))
    n_fit = len(y) - n_cal
    scored = {}
    for c in C_GRID:
        m = _lr(c).fit(E[:n_fit], y[:n_fit])
        cls = list(m.classes_)
        Pp = m.predict_proba(E[n_fit:])
        kk = np.array([cls.index(v) for v in y[n_fit:]])
        scored[c] = float(-np.log(np.clip(Pp[np.arange(n_cal), kk], P_FLOOR, None)).mean())
    best = min(scored, key=scored.get)
    return float(best), {str(c): _r(v) for c, v in scored.items()}


def _confusion(pred: np.ndarray, yb: np.ndarray, classes: list[str]) -> dict:
    out = {}
    for c in classes:
        m = yb == c
        out[c] = {p: int((pred[m] == p).sum()) for p in classes}
    return out


def _per_class(nll: np.ndarray, ki: np.ndarray, classes: list[str]) -> dict:
    return {
        c: (_r(nll[ki == i].mean()) if bool((ki == i).any()) else None)
        for i, c in enumerate(classes)
    }


def _counts(ki: np.ndarray, classes: list[str]) -> dict:
    return {c: int((ki == i).sum()) for i, c in enumerate(classes)}


def _paired_ci(d: np.ndarray, n_boot: int = 2000) -> tuple[float | None, float | None]:
    """Percentile CI of the mean paired per-row difference (split B only, seeded)."""
    if len(d) < 2:
        return None, None
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _blocks(receipt: Path) -> list[dict]:
    meta = json.loads(receipt.read_text())
    blocks = [dict(b) for b in meta["blocks"]]
    blocks.sort(key=lambda b: b["start"])
    return blocks


def _load_context(products: Path, demo: Path) -> dict:
    import polars as pl

    from looking_glass.stream_adapter import to_ts_expr

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
        .with_columns(to_ts_expr(pl.col("event_ts")).alias("_ts"))
    )
    T, last_hist = _bigram_counts(hist, classes)

    # customer-level event-type composition in history: the naive frequency-only comparator
    # (the repo's canonical "is the bank more than a propensity summary?" control).
    keymap = pl.DataFrame({"customer_key": keys, "row": np.arange(len(keys), dtype=np.int32)})
    hh = hist.select("customer_key", "event_type").join(keymap, on="customer_key", how="left")
    cidx = {c: i for i, c in enumerate(classes)}
    hcls = hh["event_type"].replace_strict(cidx, return_dtype=pl.Int32).to_numpy()
    cnt = np.zeros((len(keys), len(classes)))
    np.add.at(cnt, (hh["row"].to_numpy(), hcls), 1.0)

    # ELAPSED-TIME state (serving-side, no fitted parameter): log days since the last event, and
    # log days since the last event of EACH class. A customer who never had type c is censored at
    # the stream's own history window — derived from the data, not a typed sentinel.
    hist_min = hist["_ts"].min()
    window_days = (as_of - hist_min).total_seconds() / SECONDS_PER_DAY
    last_agg = hist.group_by("customer_key").agg(pl.col("_ts").max().alias("last_ts"))
    last_map = dict(
        zip(last_agg["customer_key"].to_list(), last_agg["last_ts"].to_list(), strict=False)
    )
    gap_days = np.array(
        [(as_of - last_map.get(k, hist_min)).total_seconds() / SECONDS_PER_DAY for k in keys]
    )
    per_days = np.empty((len(keys), len(classes)))
    for i, c in enumerate(classes):
        agg = (
            hist.filter(pl.col("event_type") == c)
            .group_by("customer_key")
            .agg(pl.col("_ts").max().alias("last_ts"))
        )
        m = dict(zip(agg["customer_key"].to_list(), agg["last_ts"].to_list(), strict=False))
        per_days[:, i] = [
            (as_of - m[k]).total_seconds() / SECONDS_PER_DAY if k in m else window_days
            for k in keys
        ]

    a_rows = [k for k in akeys if k in next_of and k in emb]
    b_rows = [k for k in bkeys if k in next_of and k in emb]
    # one stack, then a positional gather — no per-row embedding materialisation
    all_keys = sorted(emb)
    M = np.stack([np.asarray(emb[k], float) for k in all_keys])
    pos = {k: i for i, k in enumerate(all_keys)}
    ia = np.array([pos[k] for k in a_rows])
    ib = np.array([pos[k] for k in b_rows])
    tot = cnt.sum(1, keepdims=True)
    F = np.hstack([cnt / np.maximum(tot, 1.0), np.log1p(tot)])
    R = np.hstack([np.log1p(gap_days)[:, None], np.log1p(per_days)])

    return {
        "classes": classes,
        "T": T,
        "last_hist": last_hist,
        "a_rows": a_rows,
        "b_rows": b_rows,
        "Ea": M[ia],
        "Eb": M[ib],
        "Fa": F[ia],
        "Fb": F[ib],
        "Ra": R[ia],
        "Rb": R[ib],
        "window_days": window_days,
        "ya": np.array([next_of[k] for k in a_rows]),
        "yb": np.array([next_of[k] for k in b_rows]),
        "lasta": np.array([last_hist[k] for k in a_rows], dtype=int),
        "lastb": np.array([last_hist[k] for k in b_rows], dtype=int),
        "n_customers": len(keys),
        "as_of": manifest["as_of"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="looking_glass/artifacts/cfm_v2k_vf2_bank")
    ap.add_argument("--demo", default="looking_glass/artifacts/demo_v2")
    ap.add_argument("--certified", default="docs/report_cards/experiments/encoder_win.json")
    ap.add_argument("--json", action="store_true", help="dump the report body as JSON to stdout")
    a = ap.parse_args()

    t0 = time.perf_counter()
    products = ROOT / a.products
    demo = ROOT / a.demo
    ctx = _load_context(products, demo)
    classes, T = ctx["classes"], ctx["T"]
    ki = np.array([classes.index(v) for v in ctx["yb"]])
    ka = np.array([classes.index(v) for v in ctx["ya"]])
    n_A = _counts(ka, classes)
    n_B = _counts(ki, classes)
    P = T / T.sum(1, keepdims=True)
    nll_base = -np.log(P[ctx["lastb"], ki])

    # ------------------------------------------------------------------ section 1: reproduction
    C, c_grid = _select_c(ctx["Ea"], ctx["ya"])
    head = _lr(C).fit(ctx["Ea"], ctx["ya"])
    cls = list(head.classes_)
    if cls != classes:
        raise SystemExit(f"head classes {cls} != target classes {classes} — refusing to compare")
    Pp = head.predict_proba(ctx["Eb"])
    nll_enc = _nll(Pp, [cls.index(v) for v in classes], ki)

    # naive frequency-only comparator: same head family, same C protocol, 6 features (the
    # customer's event-type shares + log history length). If the bank adds nothing over it, the
    # served object is a propensity summary for that class.
    Cf, _ = _select_c(ctx["Fa"], ctx["ya"])
    fhead = _lr(Cf).fit(ctx["Fa"], ctx["ya"])
    nll_freq = _nll(
        fhead.predict_proba(ctx["Fb"]), [list(fhead.classes_).index(v) for v in classes], ki
    )

    # ELAPSED-TIME control: the same frequency state plus serving-side recency (log days since the
    # last event of each class). No fitted parameter, no label, history <= as_of — the exact kind
    # of statistic the bank is supposed to carry. It answers "is the bank losing RECENCY?".
    rec_heads = {}
    nll_rec = None
    for tag, Xa, Xb in (
        ("freq", ctx["Fa"], ctx["Fb"]),
        (
            "freq+gap",
            np.hstack([ctx["Fa"], ctx["Ra"][:, :1]]),
            np.hstack([ctx["Fb"], ctx["Rb"][:, :1]]),
        ),
        ("freq+elapsed", np.hstack([ctx["Fa"], ctx["Ra"]]), np.hstack([ctx["Fb"], ctx["Rb"]])),
    ):
        Cr, _ = _select_c(Xa, ctx["ya"])
        m = _lr(Cr).fit(Xa, ctx["ya"])
        px = _nll(m.predict_proba(Xb), [list(m.classes_).index(v) for v in classes], ki)
        rec_heads[tag] = {
            "C": Cr,
            "n_features": int(Xa.shape[1]),
            "nll_overall": _r(px.mean()),
            "per_class_nll": _per_class(px, ki, classes),
            "per_class_delta_vs_bank": {
                c: _r(px[ki == i].mean() - nll_enc[ki == i].mean())
                for i, c in enumerate(classes)
                if bool((ki == i).any())
            },
            "per_class_delta_ci95": {
                c: [_r(v) for v in _paired_ci((px - nll_enc)[ki == i])]
                for i, c in enumerate(classes)
                if bool((ki == i).any())
            },
            "delta_overall": _r(px.mean() - nll_enc.mean()),
            "delta_overall_ci95": [_r(v) for v in _paired_ci(px - nll_enc)],
        }
        if tag == "freq+elapsed":
            nll_rec = px

    # the head's logit geometry: how many effective discriminative directions does it have?
    # One dominant direction = every reweighting can only TRADE mass between classes.
    Z = head.decision_function(ctx["Eb"])
    Z = Z - Z.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(np.cov(Z, rowvar=False))[::-1]
    ev = np.clip(ev, 0.0, None)
    ev_share = ev / ev.sum() if ev.sum() > 0 else ev

    certified = {}
    cert_path = ROOT / a.certified
    if cert_path.exists():
        certified = json.loads(cert_path.read_text()).get("encoder_win", {})
    d_enc = (
        abs(float(certified["nll_enc"]) - float(nll_enc.mean()))
        if certified.get("nll_enc") is not None
        else None
    )
    d_base = (
        abs(float(certified["nll_base"]) - float(nll_base.mean()))
        if certified.get("nll_base") is not None
        else None
    )
    repro = {
        "n_customers": ctx["n_customers"],
        "n_A": len(ctx["ya"]),
        "n_B": len(ctx["yb"]),
        "dim": int(ctx["Eb"].shape[1]),
        "classes": classes,
        "n_A_per_class": n_A,
        "n_B_per_class": n_B,
        "adopted_C": C,
        "C_selection_nll": c_grid,
        "nll_enc": _r(nll_enc.mean()),
        "nll_base": _r(nll_base.mean()),
        "certified_nll_enc": certified.get("nll_enc"),
        "certified_nll_base": certified.get("nll_base"),
        "delta_vs_certified_enc": _r(d_enc),
        "delta_vs_certified_base": _r(d_base),
        # 1e-4, not 1e-6: the receipt rounds to 6 dp and BLAS reduction order moves the last
        # digits; a cross-thread refit cannot be bit-identical. The deltas are reported.
        "matches_certified": bool(d_enc is not None and d_enc < 1e-4 and d_base < 1e-4),
        "frequency_only_head": {"C": Cf, "nll_overall": _r(nll_freq.mean())},
        "logit_geometry": {
            "note": "eigen-share of the fitted head's logit covariance on split B",
            "eigen_share": [_r(v) for v in ev_share],
            "top1_share": _r(ev_share[0]),
            "top2_share": _r(ev_share[:2].sum()),
        },
    }

    # ------------------------------------------- section 1: confusion, mean p, gap decomposition
    pred_enc = np.array(classes)[Pp.argmax(1)]
    pred_base = np.array(classes)[P.argmax(1)[ctx["lastb"]]]
    conf = {"head": _confusion(pred_enc, ctx["yb"], classes)}
    conf["base"] = _confusion(pred_base, ctx["yb"], classes)
    share = {c: _r(np.mean(pred_enc == c)) for c in classes}
    share_base = {c: _r(np.mean(pred_base == c)) for c in classes}

    # reference comparators (NOT gates): a customer-level marginal and a customer-level bigram.
    # The certified 1-gram is estimated from ALL event transitions in history but scored on the
    # FIRST customer-level event after as_of; these two rows measure how much of its per-class
    # edge is that level tilt rather than information about the customer.
    prior = np.bincount(ka, minlength=len(classes)) / len(ka)  # split-A frequencies, never B
    nll_marg = -np.log(np.clip(prior[ki], P_FLOOR, None))
    Cc = np.ones((len(classes), len(classes)))  # Laplace, same convention as _bigram_counts
    np.add.at(Cc, (ctx["lasta"], ka), 1.0)
    Pc = Cc / Cc.sum(1, keepdims=True)
    nll_custgram = -np.log(np.clip(Pc[ctx["lastb"], ki], P_FLOOR, None))

    ptab = {}
    for c in classes:
        i = classes.index(c)
        m = ki == i
        if not bool(m.any()):
            continue
        pe = Pp[m, i]
        pb = P[ctx["lastb"][m], i]
        n_e = float(nll_enc[m].mean())
        n_b = float(nll_base[m].mean())
        lvl = float(np.log(pb.mean() / pe.mean()))
        disp_e = n_e - float(-np.log(pe.mean()))
        disp_b = n_b - float(-np.log(pb.mean()))
        ptab[c] = {
            "n_B": int(m.sum()),
            "nll_enc": _r(n_e),
            "nll_base": _r(n_b),
            "gap_enc_minus_base": _r(n_e - n_b),
            "meanp_enc_on_true": _r(pe.mean()),
            "meanp_base_on_true": _r(pb.mean()),
            "meanp_enc_all_rows": _r(Pp[:, i].mean()),
            "head_lift_on_true": _r(pe.mean() / max(Pp[:, i].mean(), P_FLOOR)),
            "sd_p_enc_all_rows": _r(Pp[:, i].std()),
            "gap_level": _r(lvl),
            "gap_disp_enc": _r(disp_e),
            "gap_disp_base": _r(disp_b),
            "head_pred_share": share[c],
            "base_pred_share": share_base[c],
            "head_steals": {
                p: int(conf["head"][c][p]) for p in classes if p != c and conf["head"][c][p] > 0
            },
            "base_steals": {
                p: int(conf["base"][c][p]) for p in classes if p != c and conf["base"][c][p] > 0
            },
            "nll_marginal_customer": _r(nll_marg[m].mean()),
            "nll_custgram_customer": _r(nll_custgram[m].mean()),
            "nll_freq_head": _r(nll_freq[m].mean()),
            "nll_recency_head": _r(nll_rec[m].mean()),
            "claimable": bool(m.sum() >= MIN_CLAIM_N),
        }

    # ------------------------------------------------------ section 2: nonlinear DIAGNOSTIC only
    from sklearn.ensemble import HistGradientBoostingClassifier

    gn = {
        "diagnostic_only": True,
        "dec_407": "MLP/GBM never ship; measured here only to test linear readability",
        "model": "HistGradientBoostingClassifier",
        "params": {
            "max_iter": 300,
            "learning_rate": 0.06,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "early_stopping": "internal 20% of split A",
            "n_iter_no_change": 25,
            "random_state": SEED,
        },
    }
    t1 = time.perf_counter()
    gbm = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=25,
        random_state=SEED,
    ).fit(ctx["Ea"], ctx["ya"])
    Pg = gbm.predict_proba(ctx["Eb"])
    gcls = list(gbm.classes_)
    nll_gbm = _nll(Pg, [gcls.index(v) for v in classes], ki)
    gn["seconds"] = _r(time.perf_counter() - t1, 1)
    gn["n_iter"] = int(gbm.n_iter_)
    gn["nll_overall"] = _r(nll_gbm.mean())
    gn["nll_overall_enc"] = _r(nll_enc.mean())
    gn["nll_overall_base"] = _r(nll_base.mean())
    gn["per_class_nll"] = _per_class(nll_gbm, ki, classes)
    gn["per_class_delta_vs_linear"] = {
        c: _r(nll_gbm[ki == i].mean() - nll_enc[ki == i].mean())
        for i, c in enumerate(classes)
        if bool((ki == i).any())
    }
    gn["binary"] = {}
    for i, c in enumerate(classes):
        m = ki == i
        if m.sum() < MIN_CLAIM_N:
            continue
        t2 = time.perf_counter()
        g = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=SEED,
        ).fit(ctx["Ea"], (ctx["ya"] == c).astype(int))
        q = g.predict_proba(ctx["Eb"])[:, 1]
        gn["binary"][c] = {
            "nll": _r(-np.log(np.clip(q[m], P_FLOOR, None)).mean()),
            "meanq_on_true": _r(q[m].mean()),
            "meanq_all_rows": _r(q.mean()),
            "n_iter": int(g.n_iter_),
            "seconds": _r(time.perf_counter() - t2, 1),
        }

    # ------------------------------------------------------------- section 3: block ablation
    blocks = _blocks(products / "bank_receipt.json")
    total_dim = int(ctx["Eb"].shape[1])
    covered = np.zeros(total_dim, int)
    for b in blocks:
        covered[b["start"] : b["start"] + b["dim"]] += 1
    if not bool((covered == 1).all()):
        raise SystemExit("bank_receipt blocks do not tile the served object — refusing ablation")
    all_cols = list(range(total_dim))
    full = _per_class(nll_enc, ki, classes)
    abl = {
        "fixed_C": C,
        "note": "C held at the adopted head's value so capacity does not confound the ablation",
        "blocks": [{"name": b["name"], "start": b["start"], "dim": b["dim"]} for b in blocks],
        "full": full,
        "full_nll_overall": _r(nll_enc.mean()),
        "leave_one_out": {},
        "block_only": {},
    }
    for b in blocks:
        loo = [j for j in all_cols if not (b["start"] <= j < b["start"] + b["dim"])]
        m_loo = _lr(C).fit(ctx["Ea"][:, loo], ctx["ya"])
        pl_loo = _nll(
            m_loo.predict_proba(ctx["Eb"][:, loo]),
            [list(m_loo.classes_).index(v) for v in classes],
            ki,
        )
        only = list(range(b["start"], b["start"] + b["dim"]))
        m_only = _lr(C).fit(ctx["Ea"][:, only], ctx["ya"])
        pl_only = _nll(
            m_only.predict_proba(ctx["Eb"][:, only]),
            [list(m_only.classes_).index(v) for v in classes],
            ki,
        )
        abl["leave_one_out"][b["name"]] = {
            "nll_overall": _r(pl_loo.mean()),
            "delta_overall": _r(pl_loo.mean() - nll_enc.mean()),
            "per_class_nll": _per_class(pl_loo, ki, classes),
            "per_class_delta": {
                c: _r(pl_loo[ki == i].mean() - nll_enc[ki == i].mean())
                for i, c in enumerate(classes)
                if bool((ki == i).any())
            },
        }
        abl["block_only"][b["name"]] = {
            "nll_overall": _r(pl_only.mean()),
            "per_class_nll": _per_class(pl_only, ki, classes),
        }

    # ------------------------------------------------------- section 4: last-event oracle probe
    probe_grid = {}
    n_cal = int(round(0.3 * len(ctx["ya"])))
    n_fit = len(ctx["ya"]) - n_cal
    best_c, best_acc = None, -1.0
    for c in C_GRID:
        m = _lr(c).fit(ctx["Ea"][:n_fit], ctx["lasta"][:n_fit])
        acc = float((m.predict(ctx["Ea"][n_fit:]) == ctx["lasta"][n_fit:]).mean())
        probe_grid[str(c)] = _r(acc)
        if acc > best_acc:
            best_c, best_acc = float(c), acc
    probe_cal = _lr(best_c).fit(ctx["Ea"][:n_fit], ctx["lasta"][:n_fit])
    Qcal = probe_cal.predict_proba(ctx["Ea"][n_fit:])
    pcls = list(probe_cal.classes_)
    cal_correct = np.array(pcls)[Qcal.argmax(1)] == ctx["lasta"][n_fit:]
    cal_conf = Qcal.max(1)
    thr = None
    for t in np.linspace(0.30, 0.995, 140):
        sel = cal_conf >= t
        if sel.mean() >= PROBE_MIN_COVERAGE and cal_correct[sel].mean() >= PROBE_PRECISION:
            thr = float(t)
            break
    probe = _lr(best_c).fit(ctx["Ea"], ctx["lasta"])
    Qb = probe.predict_proba(ctx["Eb"])
    bcls = list(probe.classes_)
    b_pred = np.array(bcls)[Qb.argmax(1)]
    b_conf = Qb.max(1)
    b_correct = b_pred == ctx["lastb"]

    def _sub(mask: np.ndarray) -> dict:
        out = {
            "n": int(mask.sum()),
            "coverage": _r(mask.mean()),
            "probe_accuracy": _r(b_correct[mask].mean()) if mask.any() else None,
            "nll_overall": {
                "base": _r(nll_base[mask].mean()) if mask.any() else None,
                "enc": _r(nll_enc[mask].mean()) if mask.any() else None,
                "gbm": _r(nll_gbm[mask].mean()) if mask.any() else None,
            },
            "per_class": {},
        }
        for i, c in enumerate(classes):
            mm = mask & (ki == i)
            if mm.sum() == 0:
                continue
            out["per_class"][c] = {
                "n": int(mm.sum()),
                "nll_base": _r(nll_base[mm].mean()),
                "nll_enc": _r(nll_enc[mm].mean()),
                "nll_gbm": _r(nll_gbm[mm].mean()),
                "gap": _r((nll_enc - nll_base)[mm].mean()),
                "claimable": bool(mm.sum() >= MIN_CLAIM_N),
            }
        return out

    conf_mask = b_conf >= (thr if thr is not None else 1.0)
    # a sweep, so the single threshold is not the whole argument: does the gap move with the
    # bank's own confidence that it knows the last event?
    sweep = []
    for t in (0.50, 0.70, 0.90, 0.95, 0.97, 0.99):
        s = b_conf >= t
        row = {
            "threshold": t,
            "n": int(s.sum()),
            "coverage": _r(s.mean()),
            "probe_precision": _r(b_correct[s].mean()) if bool(s.any()) else None,
            "per_class_gap": {},
        }
        for c in TARGET_CLASSES:
            mm = s & (ki == classes.index(c))
            row["per_class_gap"][c] = (
                _r((nll_enc - nll_base)[mm].mean()) if bool(mm.any()) else None
            )
        sweep.append(row)

    # leverage: how many nats does the CERTIFIED 1-gram itself spend on last-event identity?
    leverage = {}
    for i, c in enumerate(classes):
        vals = -np.log(P[:, i])
        leverage[c] = {
            "min": _r(vals.min()),
            "max": _r(vals.max()),
            "range_nats": _r(vals.max() - vals.min()),
            "sd_weighted_on_B": _r(vals[ctx["lastb"]].std()),
        }

    # direct oracle head: bank || exact last-event one-hot (the rejected recent_k=1 variant),
    # fixed C and fixed capacity protocol, so this section answers the question on its own.
    La = np.zeros((len(ctx["ya"]), len(classes)))
    La[np.arange(len(ctx["ya"])), ctx["lasta"]] = 1.0
    Lb = np.zeros((len(ctx["yb"]), len(classes)))
    Lb[np.arange(len(ctx["yb"])), ctx["lastb"]] = 1.0
    oracle = _lr(C).fit(np.hstack([ctx["Ea"], La]), ctx["ya"])
    nll_oracle = _nll(
        oracle.predict_proba(np.hstack([ctx["Eb"], Lb])),
        [list(oracle.classes_).index(v) for v in classes],
        ki,
    )

    last_event = {
        "probe": {
            "C": best_c,
            "C_selection_accuracy": probe_grid,
            "accuracy_on_B": _r(b_correct.mean()),
            "accuracy_on_B_per_true_last": _per_class(
                np.where(b_correct, 1.0, 0.0), ctx["lastb"], classes
            ),
            "threshold": _r(thr),
            "threshold_rule": (
                f"smallest max-prob cut on the split-A holdout with precision >= "
                f"{PROBE_PRECISION} and coverage >= {PROBE_MIN_COVERAGE}; derived on A, applied to B"
            ),
            "threshold_sweep": sweep,
        },
        "confident_subset": _sub(conf_mask),
        "unconfident_subset": _sub(~conf_mask),
        "oracle_subset_probe_correct": _sub(b_correct),
        "base_1gram_last_event_leverage": leverage,
        "oracle_head_with_exact_last_event": {
            "note": "diagnostic replica of the rejected recent_k=1 variant (bank || one-hot(last))",
            "nll_overall": _r(nll_oracle.mean()),
            "per_class_nll": _per_class(nll_oracle, ki, classes),
            "per_class_delta_vs_linear": {
                c: _r(nll_oracle[ki == i].mean() - nll_enc[ki == i].mean())
                for i, c in enumerate(classes)
                if bool((ki == i).any())
            },
            "per_class_delta_vs_base": {
                c: _r(nll_oracle[ki == i].mean() - nll_base[ki == i].mean())
                for i, c in enumerate(classes)
                if bool((ki == i).any())
            },
        },
    }

    # overall reference comparators: how much of the certified 0.250-nat win is the baseline's
    # event-level tilt?  Descriptive only — the certified gate is frozen and unchanged.
    ref = {}
    for name, arr in (
        ("base_event_level", nll_base),
        ("marginal_customer", nll_marg),
        ("custgram_customer", nll_custgram),
        ("freq_head", nll_freq),
        ("recency_head", nll_rec),
    ):
        lo, hi = _paired_ci(arr - nll_enc)
        ref[name] = {
            "nll": _r(arr.mean()),
            "delta_minus_head": _r(arr.mean() - nll_enc.mean()),
            "delta_ci95": [_r(lo), _r(hi)],
        }

    body = {
        "kind": "class_gap_diagnosis",
        "diagnostic_only": True,
        "dec_407": "MLP/GBM never ship; nonlinear results here are diagnostic, never a consumer",
        "gate": False,
        "protocol": (
            "Read-only localisation of the B9.1 per-class shortfall. The frozen split "
            "(measure_encoder_win._split_customers, SEED) and the served 179-dim bank are "
            "reproduced and asserted against the certified receipt; the adopted multinomial "
            "logistic head (C re-selected on the split-A tail holdout, never B) is the object "
            "under diagnosis. Target classes are the FIRST event type after as_of per customer."
        ),
        "reproduction": repro,
        "reference_comparators": ref,
        "per_class": ptab,
        "recency_control": {
            "note": (
                "serving-side reference control (NOT a gate, NOT a consumer): frequency state + "
                "log days since the last event of each class, no fitted parameter, history <= as_of"
            ),
            "window_days": _r(ctx["window_days"], 1),
            "heads": rec_heads,
        },
        "nonlinear_diagnostic": gn,
        "block_ablation": abl,
        "last_event_oracle": last_event,
        "seconds": _r(time.perf_counter() - t0, 1),
    }

    if a.json:
        print(json.dumps(body, indent=2))
        return 0

    # ------------------------------------------------------------------------------ text report
    print("CLASS-GAP DIAGNOSIS (read-only). NOT a gate. Sections 3-5 = DEC-407 DIAGNOSTIC ONLY.")

    print("\n[1] reproduction: frozen split + exact served object")
    print(
        f"  n_customers={repro['n_customers']} n_A={repro['n_A']} n_B={repro['n_B']} "
        f"dim={repro['dim']} classes={classes}"
    )
    print(f"  adopted head: logistic C={C} (re-selected on the split-A tail holdout: {c_grid})")
    print(
        f"  nll enc={repro['nll_enc']:.6f} base={repro['nll_base']:.6f}  "
        f"certified {repro['certified_nll_enc']} / {repro['certified_nll_base']}  "
        f"delta {repro['delta_vs_certified_enc']:.2e} / {repro['delta_vs_certified_base']:.2e}  "
        f"reproduced={repro['matches_certified']}"
    )
    print(f"  n_A per class: {n_A}")

    print("\n[2] per class: gap, mass allocation, confusion, and reference comparators")
    hdr = (
        f"  {'class':12s} {'n_B':>4s} {'base':>7s} {'head':>7s} {'gap':>7s} | "
        f"{'level':>7s} {'disp_e':>7s} {'disp_b':>7s} | "
        f"{'p_true':>6s} {'p_base':>6s} {'p_all':>6s} {'lift':>4s} | "
        f"{'p_shr':>5s} | {'marg':>7s} {'custg':>7s} {'freq':>7s} {'rec':>7s}"
    )
    print(hdr)
    for c in classes:
        d = ptab.get(c)
        if not d:
            continue
        print(
            f"  {c:12s} {d['n_B']:4d} {d['nll_base']:7.4f} {d['nll_enc']:7.4f} "
            f"{d['gap_enc_minus_base']:+7.4f} | {d['gap_level']:+7.4f} {d['gap_disp_enc']:+7.4f} "
            f"{d['gap_disp_base']:+7.4f} | {d['meanp_enc_on_true']:6.4f} "
            f"{d['meanp_base_on_true']:6.4f} {d['meanp_enc_all_rows']:6.4f} "
            f"{d['head_lift_on_true']:4.2f} | {d['head_pred_share']:5.3f} | "
            f"{d['nll_marginal_customer']:7.4f} {d['nll_custgram_customer']:7.4f} "
            f"{d['nll_freq_head']:7.4f} {d['nll_recency_head']:7.4f}"
            + ("" if d["claimable"] else "  (n<8: report, do not claim)")
        )
    print(
        "  gap = level + disp_head - disp_base ; level>0 = the head puts LESS mass on c on c's own "
        "rows than the 1-gram (a prior tilt, not information)"
    )
    print(
        "  p_shr = share of B rows the head's argmax gives to c (actual share is in n_B/1500); "
        "p_true/p_all = mean p_c on true-c rows vs all rows"
    )
    print(
        "  marg/custg/freq/rec = customer-level marginal, customer-level bigram, frequency-only "
        "head, frequency+elapsed-time head (all split-A fit; reference comparators, NOT gates)"
    )
    rh = body["recency_control"]["heads"]
    fe = rh["freq+elapsed"]
    print(
        f"  recency control (serving-side, 0 fitted params, history<=as_of): freq={rh['freq']['nll_overall']:.4f} "
        f"-> +gap={rh['freq+gap']['nll_overall']:.4f} -> +elapsed={fe['nll_overall']:.4f} "
        f"(bank {repro['nll_enc']:.4f}; window={body['recency_control']['window_days']}d)"
    )
    print(
        "    delta vs bank: "
        + " ".join(
            f"{c}={fe['per_class_delta_vs_bank'][c]:+.4f}"
            for c in ("email_send", *TARGET_CLASSES)
            if c in fe["per_class_delta_vs_bank"]
        )
        + "  (return excluded: n_B=3)"
    )
    print(
        "    paired 95% CI vs bank: "
        + " ".join(
            f"{c}=[{fe['per_class_delta_ci95'][c][0]:+.3f},{fe['per_class_delta_ci95'][c][1]:+.3f}]"
            for c in TARGET_CLASSES
            if c in fe["per_class_delta_ci95"]
        )
        + f"; overall {fe['delta_overall']:+.4f} "
        f"[{fe['delta_overall_ci95'][0]:+.3f},{fe['delta_overall_ci95'][1]:+.3f}]"
    )
    print(
        f"  overall comparators: base(1-gram)={repro['nll_base']:.4f} "
        f"marg={nll_marg.mean():.4f} custg={nll_custgram.mean():.4f} freq={nll_freq.mean():.4f} "
        f"rec={nll_rec.mean():.4f} head={repro['nll_enc']:.4f}"
    )
    print(
        "  comparator minus head (positive = head better), paired 95% CI: "
        + " ".join(
            f"{k}={v['delta_minus_head']:+.3f}[{v['delta_ci95'][0]:+.3f},{v['delta_ci95'][1]:+.3f}]"
            for k, v in ref.items()
        )
    )
    for c in TARGET_CLASSES:
        d = ptab.get(c)
        if not d:
            continue
        hs = ", ".join(f"{p}:{n}" for p, n in sorted(d["head_steals"].items(), key=lambda t: -t[1]))
        bs = ", ".join(f"{p}:{n}" for p, n in sorted(d["base_steals"].items(), key=lambda t: -t[1]))
        print(f"  {c:12s} head steals -> {hs or 'none'} | base steals -> {bs or 'none'}")
    lg = repro["logit_geometry"]
    print(
        f"  head logit geometry: top-1 eigen-share={lg['top1_share']:.4f} "
        f"top-2={lg['top2_share']:.4f} (all 5: {lg['eigen_share']})"
    )

    print("\n[3] nonlinear DIAGNOSTIC ONLY (DEC-407: MLP/GBM never ship) — is the info readable?")
    print(
        f"  HistGBM multiclass: nll={gn['nll_overall']:.4f} (head {gn['nll_overall_enc']:.4f}, "
        f"base {gn['nll_overall_base']:.4f}) n_iter={gn['n_iter']} fit={gn['seconds']}s"
    )
    for c in classes:
        d = ptab.get(c)
        if not d:
            continue
        b = gn["binary"].get(c)
        bin_txt = f"{b['nll']:7.4f}" if b else "    n/a"
        q_txt = f"  q_true={b['meanq_on_true']:.4f} q_all={b['meanq_all_rows']:.4f}" if b else ""
        print(
            f"  {c:12s} head={d['nll_enc']:7.4f} gbm_mc={gn['per_class_nll'][c]:7.4f} "
            f"gbm_bin={bin_txt} base={d['nll_base']:7.4f} "
            f"delta_gbm={gn['per_class_delta_vs_linear'][c]:+7.4f}{q_txt}"
        )
    print("  delta_gbm < 0 = the nonlinear model beats the linear head on that class")

    print("\n[4] block ablation (fixed C; delta>0 = removing the block HELPS that class)")
    print(
        f"  {'block':14s} {'dim':>4s} | "
        f"{'d_purch':>8s} {'d_srch':>8s} {'d_view':>8s} {'d_all':>8s} | "
        f"{'only_pur':>8s} {'only_srch':>8s} {'only_view':>9s} {'only_all':>8s}"
    )
    for b in blocks:
        loo = abl["leave_one_out"][b["name"]]
        only = abl["block_only"][b["name"]]
        print(
            f"  {b['name']:14s} {b['dim']:4d} | "
            f"{loo['per_class_delta']['purchase']:+8.4f} "
            f"{loo['per_class_delta']['search']:+8.4f} "
            f"{loo['per_class_delta']['view']:+8.4f} {loo['delta_overall']:+8.4f} | "
            f"{only['per_class_nll']['purchase']:8.4f} "
            f"{only['per_class_nll']['search']:8.4f} "
            f"{only['per_class_nll']['view']:9.4f} {only['nll_overall']:8.4f}"
        )
    print(
        f"  full bank: purchase={full['purchase']:.4f} search={full['search']:.4f} "
        f"view={full['view']:.4f} all={abl['full_nll_overall']:.4f}"
    )

    print("\n[5] last-event oracle: is the residual gap last-event uncertainty?")
    pr = last_event["probe"]
    print(
        f"  probe bank->last event: C={pr['C']} acc_B={pr['accuracy_on_B']:.4f} "
        f"threshold={pr['threshold']}  confident subset: n={last_event['confident_subset']['n']} "
        f"coverage={last_event['confident_subset']['coverage']:.3f} "
        f"precision={last_event['confident_subset']['probe_accuracy']}"
    )
    for tag in ("confident_subset", "unconfident_subset"):
        sub = last_event[tag]
        print(
            f"  {tag:18s} n={sub['n']:4d} overall: base={sub['nll_overall']['base']:.4f} "
            f"head={sub['nll_overall']['enc']:.4f} gbm={sub['nll_overall']['gbm']:.4f}"
        )
        for c in TARGET_CLASSES:
            d = sub["per_class"].get(c)
            if d:
                print(
                    f"    {c:12s} n={d['n']:4d} base={d['nll_base']:.4f} head={d['nll_enc']:.4f} "
                    f"gap={d['gap']:+.4f}" + ("" if d["claimable"] else "  (n<8: no claim)")
                )
    print("  sweep (threshold -> coverage, probe precision, gap per class):")
    for row in pr["threshold_sweep"]:
        g = row["per_class_gap"]

        def _g(x):
            return f"{x:+.4f}" if x is not None else "  n/a "

        prec = row["probe_precision"]
        print(
            f"    t={row['threshold']:.2f} n={row['n']:4d} cov={row['coverage']:.3f} "
            f"prec={prec if prec is not None else 'n/a'} "
            f"gap purch={_g(g['purchase'])} srch={_g(g['search'])} view={_g(g['view'])}"
        )
    print("  1-gram's own spend on last-event identity (range of -log P over last types):")
    for c in TARGET_CLASSES:
        lv = last_event["base_1gram_last_event_leverage"][c]
        print(
            f"    {c:12s} range={lv['range_nats']:.4f} nats  sd_on_B={lv['sd_weighted_on_B']:.4f}"
        )
    oh = last_event["oracle_head_with_exact_last_event"]
    print(
        f"  oracle head (bank || exact last-event one-hot): nll={oh['nll_overall']:.4f} "
        f"vs head {repro['nll_enc']:.4f}"
    )
    for c in TARGET_CLASSES:
        print(
            f"    {c:12s} oracle={oh['per_class_nll'][c]:.4f} "
            f"vs_head={oh['per_class_delta_vs_linear'][c]:+.4f} "
            f"vs_base={oh['per_class_delta_vs_base'][c]:+.4f}"
        )
    print(f"\nseconds={body['seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
