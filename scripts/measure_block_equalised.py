#!/usr/bin/env python3
"""DEC-669 AXIS D: does block-equalising the served bank rescue the narrow blocks?

Pre-registration (DEC-669): one uniform L2 over the 179-dim served bank treats every standardized
column alike, so a narrow block (``return_ratio``, 6 dims) is penalized as if its aggregate
contribution mattered as much as ``numeric_stats`` (60 dims). The consumer-side fix is to scale
every served BLOCK to unit total variance before the head's penalty,
``x_block <- x_block / sqrt(trace(cov(x_block)))``, with the scaling fit on split A only. Still
linear, still the shipping multinomial-logistic family (DEC-407/649).

One algebraic point the measurement must respect: the adopted head's ``StandardScaler`` z-scores
every column, and a uniform per-block scalar applied BEFORE that scaler is exactly cancelled by
it. The pre-registered rule is therefore applied where it can act — after the per-column
z-scoring: each block of the standardized matrix is divided by ``sqrt(trace(cov(z_block)))``
(trace on split A), which is precisely "each block enters with unit total variance". The third
arm (``block_z``: z-scoring only) is the control that demonstrates the cancellation, so a
``block_equalised`` gain cannot be attributed to rescaling.

Arms, all fit on split A (C chosen on the A tail holdout, never B) and scored on split B:
  plain          the adopted multinomial-logistic pipeline (StandardScaler + L2 logistic); it
                 must reproduce the certified nll 1.173099 within 1e-5 (self-check)
  block_z        per-column z-scoring only (the rescaling control; algebraically == plain)
  block_equalised block_z, then each block scaled to unit total variance (equalising)

Reports aggregate and per-class nll, the paired bootstrap CI of (arm - plain) on split B, the
selected C, and the frozen DEC-669 acceptance verdict. Read-only: no file is written; ``--json``
dumps the same body to stdout.

    .venv/bin/python scripts/measure_block_equalised.py [--json]
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

# Small matrices on a 20-core box: oversubscribed BLAS/OpenMP pools are slower (measured ~200x on
# a 2450x179 multinomial fit) and move the last digits. Cap BEFORE numpy/sklearn are imported;
# the operator can still override. This is a cost/determinism fix, not a quality trade.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_battery_law import N_BOOT, SEED  # noqa: E402  (frozen evaluation-contract seed)
from measure_encoder_win import (  # noqa: E402
    _load_served,
    _paired_ci,
    _population_check,
    _split_customers,
)

#: the certified B9.1 numbers this probe must reproduce (DEC-649) — self-check, not a target.
CERTIFIED_NLL = 1.173099
CERTIFIED_C = 0.1
CERTIFIED_PER_CLASS = {
    "email_send": 0.762128,
    "purchase": 2.158392,
    "search": 1.726758,
    "view": 0.956927,
}
REPRO_TOL = 1e-5  # principle-ok: pre-registered self-check tolerance

#: the adopted head's protocol (diagnose_class_gap / measure_encoder_win), reproduced, not tuned.
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)  # principle-ok: protocol grid, reproduces the receipt
HOLDOUT_FRAC = 0.3  # principle-ok: the adopted head's split-A tail holdout
MAX_ITER = 4000  # principle-ok: the adopted head's solver budget
P_FLOOR = 1e-15

#: DEC-669 frozen acceptance.
TARGET_CLASSES = ("purchase", "search", "view")
TARGET_GAIN = 0.02  # principle-ok: pre-registered effect floor
MAX_REGRESSION = 0.02  # principle-ok: pre-registered per-class regression cap
MIN_CLAIM_N = (
    8  # principle-ok: per-class floor below which a per-class nll is reported, not claimed
)

ARMS = ("plain", "block_z", "block_equalised")


def _r(x, k: int = 6):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), k)


def _blocks(receipt: Path, width: int) -> list[dict]:
    """Read the block slices from the bank receipt and ASSERT they tile the served width."""
    meta = json.loads(receipt.read_text())
    blocks = sorted((dict(b) for b in meta["blocks"]), key=lambda b: b["start"])
    bank_dim = int(meta.get("bank_dim", width))
    if bank_dim != width:
        raise SystemExit(f"bank_receipt says bank_dim={bank_dim}, served object is {width}-dim")
    covered = np.zeros(width, dtype=int)
    for b in blocks:
        covered[b["start"] : b["start"] + b["dim"]] += 1
    if not bool((covered == 1).all()):
        bad = [int(j) for j in np.nonzero(covered != 1)[0][:8]]
        raise SystemExit(
            f"bank_receipt blocks do not tile the {width}-dim served object "
            f"(first bad columns: {bad}) — refusing"
        )
    return blocks


def _load_context(products: Path, demo: Path) -> dict:
    """The exact frozen split + served object of measure_encoder_win, nothing re-derived."""
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
    a_rows = [k for k in akeys if k in next_of and k in emb]
    b_rows = [k for k in bkeys if k in next_of and k in emb]
    if not a_rows or not b_rows:
        raise SystemExit("no A/B rows with a next event type")
    return {
        "as_of": manifest["as_of"],
        "n_customers": len(keys),
        "classes": classes,
        "next_of": next_of,
        "a_rows": a_rows,
        "b_rows": b_rows,
        "Ea": np.vstack([emb[k] for k in a_rows]),
        "Eb": np.vstack([emb[k] for k in b_rows]),
        "ya": np.array([next_of[k] for k in a_rows]),
        "yb": np.array([next_of[k] for k in b_rows]),
    }


def _fit_tf(X: np.ndarray, blocks: list[dict], equalise: bool) -> dict:
    """Per-column z-scoring fit on X, optionally followed by block equalisation.

    The trace is the population covariance trace (ddof=0), i.e. the sum of the block's per-column
    variances. After z-scoring every non-constant column has variance 1, so the trace equals the
    block's non-constant width; the scaling is still computed from the trace, not from the width.
    """
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Z = (X - mu) / sd
    scales = np.ones(X.shape[1], dtype=float)
    traces: dict[str, float] = {}
    for b in blocks:
        sl = slice(b["start"], b["start"] + b["dim"])
        trace = float(np.var(Z[:, sl], axis=0, ddof=0).sum())
        traces[b["name"]] = trace
        if equalise:
            scales[sl] = 1.0 / math.sqrt(max(trace, 1e-12))
    return {"mu": mu, "sd": sd, "scales": scales, "traces": traces, "equalise": equalise}


def _apply_tf(tf: dict, X: np.ndarray) -> np.ndarray:
    return ((X - tf["mu"]) / tf["sd"]) * tf["scales"]


def _fit_head(kind: str, X: np.ndarray, y: np.ndarray, C: float, blocks: list[dict]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "plain":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=MAX_ITER))
        model.fit(X, y)
        return {"kind": kind, "C": C, "model": model, "tf": None}
    if kind not in ("block_z", "block_equalised"):
        raise SystemExit(f"unknown arm {kind!r}")
    tf = _fit_tf(X, blocks, equalise=(kind == "block_equalised"))
    model = LogisticRegression(C=C, max_iter=MAX_ITER)
    model.fit(_apply_tf(tf, X), y)
    return {"kind": kind, "C": C, "model": model, "tf": tf}


def _classes_of(fit: dict) -> list[str]:
    est = fit["model"]
    if hasattr(est, "named_steps"):
        est = est.named_steps["logisticregression"]
    return [str(c) for c in est.classes_]


def _n_iter(fit: dict) -> int:
    est = fit["model"]
    if hasattr(est, "named_steps"):
        est = est.named_steps["logisticregression"]
    return int(np.ravel(est.n_iter_)[0])


def _predict_proba(fit: dict, X: np.ndarray) -> np.ndarray:
    Xt = X if fit["tf"] is None else _apply_tf(fit["tf"], X)
    return fit["model"].predict_proba(Xt)


def _nll_rows(fit: dict, X: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    cls = _classes_of(fit)
    if cls != classes:
        raise SystemExit(f"head classes {cls} != target classes {classes} — refusing to compare")
    P = _predict_proba(fit, X)
    ki = np.array([classes.index(v) for v in y])
    return -np.log(np.clip(P[np.arange(len(y)), ki], P_FLOOR, None))


def _select_c(kind: str, X: np.ndarray, y: np.ndarray, classes: list[str], blocks: list[dict]):
    """The adopted C selection: fit on the A head, score the A tail, never split B."""
    n_cal = int(round(HOLDOUT_FRAC * len(y)))
    n_fit = len(y) - n_cal
    grid: dict[str, float] = {}
    for C in C_GRID:
        fit = _fit_head(kind, X[:n_fit], y[:n_fit], C, blocks)
        grid[str(C)] = float(_nll_rows(fit, X[n_fit:], y[n_fit:], classes).mean())
    best = float(min(C_GRID, key=lambda c: grid[str(c)]))
    return best, {k: _r(v) for k, v in grid.items()}


def _per_class(rows: np.ndarray, ki: np.ndarray, classes: list[str]) -> dict:
    return {
        c: (float(rows[ki == i].mean()) if bool((ki == i).any()) else None)
        for i, c in enumerate(classes)
    }


def _counts(ki: np.ndarray, classes: list[str]) -> dict:
    return {c: int((ki == i).sum()) for i, c in enumerate(classes)}


def _protocol_text(dim: int) -> str:
    return (
        f"DEC-669 AXIS D: scale each served block of the {dim}-dim bank to unit total variance "
        "before the multinomial-logistic head's L2 penalty (trace of the block covariance fit on "
        "split A only), keeping the head linear and the C/A-tail protocol frozen. Pre-registered "
        "acceptance: aggregate split-B nll improves AND at least one of "
        f"{list(TARGET_CLASSES)} improves by >= {TARGET_GAIN} with no class regressing > "
        f"{MAX_REGRESSION}. Read-only measurement; nothing is adopted."
    )


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="looking_glass/artifacts/cfm_v2k_vf2_bank")
    ap.add_argument("--demo", default="looking_glass/artifacts/demo_v2")
    ap.add_argument("--json", action="store_true", help="dump the report body as JSON to stdout")
    return ap


def main() -> int:
    a = _parser().parse_args()
    t0 = time.perf_counter()

    products = ROOT / a.products
    demo = ROOT / a.demo
    ctx = _load_context(products, demo)
    classes = ctx["classes"]
    Ea, Eb = ctx["Ea"], ctx["Eb"]
    ya, yb = ctx["ya"], ctx["yb"]
    dim = int(Ea.shape[1])
    blocks = _blocks(products / "bank_receipt.json", dim)
    pop = _population_check(products, demo, int(Ea.shape[0] + Eb.shape[0]))
    if not pop["match"]:
        raise SystemExit(
            f"POPULATION MISMATCH: demo {pop['demo_customers']} vs products "
            f"{pop['products_n_customers']} — refusing a non-comparable measurement"
        )

    ki = np.array([classes.index(v) for v in yb])
    n_A = _counts(np.array([classes.index(v) for v in ya]), classes)
    n_B = _counts(ki, classes)

    # ------------------------------------------------------------------ arm fits + scored rows
    rows: dict[str, np.ndarray] = {}
    fits: dict[str, dict] = {}
    selected: dict[str, float] = {}
    grids: dict[str, dict] = {}
    for kind in ARMS:
        C, grid = _select_c(kind, Ea, ya, classes, blocks)
        fit = _fit_head(kind, Ea, ya, C, blocks)
        rows[kind] = _nll_rows(fit, Eb, yb, classes)
        fits[kind] = fit
        selected[kind] = C
        grids[kind] = grid

    plain_rows = rows["plain"]
    plain_nll = float(plain_rows.mean())
    # the rescaling control must be a numerical no-op: the plain pipeline's StandardScaler
    # already z-scores every column, so only the trace normalisation can move the geometry.
    control_max_abs = float(np.max(np.abs(rows["block_z"] - plain_rows)))

    # ------------------------------------------------------------------ reproduction self-check
    repro_delta = abs(plain_nll - CERTIFIED_NLL)
    plain_pc = _per_class(plain_rows, ki, classes)
    pc_deltas = {
        c: abs(float(plain_pc[c]) - v)
        for c, v in CERTIFIED_PER_CLASS.items()
        if plain_pc.get(c) is not None
    }
    if repro_delta > REPRO_TOL:
        raise SystemExit(
            f"FATAL: plain arm nll={plain_nll:.9f} does not reproduce certified "
            f"{CERTIFIED_NLL} (|delta|={repro_delta:.3e} > {REPRO_TOL})"
        )
    if selected["plain"] != CERTIFIED_C:
        raise SystemExit(
            f"FATAL: plain arm selected C={selected['plain']} != certified C={CERTIFIED_C}"
        )

    # ------------------------------------------------------------------ arm summaries
    arms: dict[str, dict] = {}
    for kind in ARMS:
        r = rows[kind]
        pc = _per_class(r, ki, classes)
        delta = r - plain_rows
        lo, hi = _paired_ci(delta)
        pc_delta = {c: float(pc[c]) - float(plain_pc[c]) for c in classes if pc[c] is not None}
        pc_ci = {
            c: list(_paired_ci(delta[ki == i]))
            for i, c in enumerate(classes)
            if bool((ki == i).any())
        }
        fit = fits[kind]
        arm = {
            "C": selected[kind],
            "C_selection_nll": grids[kind],
            "nll": _r(float(r.mean())),
            "per_class_nll": {c: _r(v) for c, v in pc.items()},
            "delta_vs_plain": _r(float(delta.mean())),
            "delta_ci95_vs_plain": [_r(lo), _r(hi)],
            "per_class_delta_vs_plain": {c: _r(v) for c, v in pc_delta.items()},
            "per_class_delta_ci95_vs_plain": {c: [_r(ci[0]), _r(ci[1])] for c, ci in pc_ci.items()},
            "n_iter": _n_iter(fit),
        }
        if fit["tf"] is not None:
            arm["transform_fit_on"] = "the same split-A rows as the head"
            arm["block_scale"] = {
                b["name"]: _r(float(fit["tf"]["scales"][b["start"]])) for b in blocks
            }
            arm["trace_z_on_A"] = {c: _r(v) for c, v in fit["tf"]["traces"].items()}
        arms[kind] = arm

    # ------------------------------------------------------------------ frozen acceptance
    # unrounded means: the verdict must not depend on the 6-dp JSON rendering
    eq_pc = _per_class(rows["block_equalised"], ki, classes)
    agg_delta = float(rows["block_equalised"].mean()) - plain_nll
    agg_improves = bool(agg_delta < 0)
    target_gain = {c: float(plain_pc[c]) - float(eq_pc[c]) for c in TARGET_CLASSES}
    targets_hit = [c for c in TARGET_CLASSES if target_gain[c] >= TARGET_GAIN]
    pc_delta_eq = {c: float(eq_pc[c]) - float(plain_pc[c]) for c in classes}
    worst_class = max(pc_delta_eq, key=lambda c: pc_delta_eq[c])
    worst_regression = float(pc_delta_eq[worst_class])
    no_regression = bool(worst_regression <= MAX_REGRESSION)
    accepted = bool(agg_improves and targets_hit and no_regression)

    # raw block traces (context: the equalised scale is width-driven after z-scoring)
    block_table = []
    raw_trace = {}
    for b in blocks:
        sl = slice(b["start"], b["start"] + b["dim"])
        raw_trace[b["name"]] = float(np.var(Ea[:, sl], axis=0, ddof=0).sum())
        block_table.append(
            {
                "name": b["name"],
                "start": int(b["start"]),
                "dim": int(b["dim"]),
                "trace_on_A_raw": _r(raw_trace[b["name"]]),
                "trace_on_A_z": _r(fits["block_equalised"]["tf"]["traces"][b["name"]]),
                "scale": _r(float(fits["block_equalised"]["tf"]["scales"][b["start"]])),
            }
        )

    body = {
        "kind": "block_equalised",
        "dec": "DEC-669",
        "axis": "D",
        "gate": False,
        "adopted": False,
        "protocol": _protocol_text(dim),
        "object": {
            "dim": dim,
            "blocks": block_table,
            "products": str(a.products),
            "demo": str(a.demo),
            "as_of": ctx["as_of"],
            "n_customers": ctx["n_customers"],
            "n_A": len(ya),
            "n_B": len(yb),
            "classes": classes,
            "n_A_per_class": n_A,
            "n_B_per_class": n_B,
            "per_class_claimable": {c: bool(n_B[c] >= MIN_CLAIM_N) for c in classes},
            "population": pop,
        },
        "reproduction": {
            "arm": "plain",
            "nll": _r(plain_nll),
            "certified_nll": CERTIFIED_NLL,
            "abs_delta": _r(repro_delta, 9),
            "tolerance": REPRO_TOL,
            "matches": bool(repro_delta <= REPRO_TOL),
            "C": selected["plain"],
            "certified_C": CERTIFIED_C,
            "per_class_abs_delta_vs_certified": {c: _r(v, 9) for c, v in pc_deltas.items()},
            "max_per_class_abs_delta": _r(max(pc_deltas.values()), 9),
        },
        "seeds": {"SEED": SEED, "N_BOOT": N_BOOT},
        "rescaling_control": {
            "arm": "block_z",
            "max_abs_row_delta_vs_plain": _r(control_max_abs, 12),
            "expected_zero": True,
            "passed": bool(control_max_abs < 1e-9),
            "why": (
                "StandardScaler in the adopted pipeline already z-scores every column; the "
                "block_z transform is that same map, so only trace normalisation can move the "
                "fitted geometry"
            ),
        },
        "arms": arms,
        "acceptance": {
            "aggregate_improves": agg_improves,
            "aggregate_delta": _r(agg_delta),
            "aggregate_ci95_vs_plain": arms["block_equalised"]["delta_ci95_vs_plain"],
            "target_gains": {c: _r(v) for c, v in target_gain.items()},
            "target_gain_floor": TARGET_GAIN,
            "targets_hit": targets_hit,
            "worst_class": worst_class,
            "worst_class_delta": _r(worst_regression),
            "max_regression": MAX_REGRESSION,
            "no_class_regresses": no_regression,
        },
        "verdict": "PASS" if accepted else "FAIL",
        "note": (
            "block_z is algebraically identical to plain (the adopted pipeline's StandardScaler "
            "already z-scores every column); it is the rescaling control, so only a "
            "block_equalised gain can be attributed to equalising aggregate block variance."
        ),
        "seconds": _r(time.perf_counter() - t0, 1),
    }

    if a.json:
        print(json.dumps(body, indent=2))
        return 0

    # ------------------------------------------------------------------------------ text report
    print("DEC-669 AXIS D — BLOCK-EQUALISED CONSUMER FEATURES (read-only; nothing adopted)")
    print(
        f"n_customers={ctx['n_customers']} n_A={len(ya)} n_B={len(yb)} dim={dim} "
        f"classes={classes}  blocks={len(blocks)}"
    )
    print(
        f"plain self-check: nll={plain_nll:.9f} certified={CERTIFIED_NLL} "
        f"|delta|={repro_delta:.2e} C={selected['plain']} reproduced="
        f"{repro_delta <= REPRO_TOL and selected['plain'] == CERTIFIED_C}"
    )

    print("\nper-arm fit (C chosen on the split-A tail holdout; scored on split B)")
    hdr = (
        f"  {'arm':16s} {'C':>6s} {'nll':>9s} {'d_nll':>9s} "
        f"{'paired 95% CI (d_nll)':>26s} | " + " ".join(f"{c:>10s}" for c in classes)
    )
    print(hdr)
    for kind in ARMS:
        arm = arms[kind]
        ci = arm["delta_ci95_vs_plain"]
        ci_txt = "—" if kind == "plain" or ci[0] is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        print(
            f"  {kind:16s} {arm['C']:6.3f} {arm['nll']:9.6f} {arm['delta_vs_plain']:+9.6f} "
            f"{ci_txt:>26s} | " + " ".join(f"{arm['per_class_nll'][c]:10.6f}" for c in classes)
        )
    print("  d_nll = arm - plain (negative = arm better); nll columns are per-class nll on B")
    print(
        f"  rescaling control (block_z): max |row delta| vs plain = {control_max_abs:.2e} "
        f"(expected 0: StandardScaler already z-scores every column) "
        f"passed={body['rescaling_control']['passed']}"
    )

    print("\nper-class delta vs plain (negative = arm better)")
    print(f"  {'arm':16s} " + " ".join(f"{c:>10s}" for c in classes))
    for kind in ARMS:
        if kind == "plain":
            continue
        print(
            f"  {kind:16s} "
            + " ".join(f"{arms[kind]['per_class_delta_vs_plain'][c]:+10.6f}" for c in classes)
            + ("  (return n<8: report, do not claim)" if n_B.get("return", 0) < MIN_CLAIM_N else "")
        )

    print("\nblock equalisation (trace of the block covariance on split A)")
    print(f"  {'block':14s} {'dim':>4s} {'trace_raw':>14s} {'trace_z':>8s} {'1/sqrt':>8s}")
    for b in block_table:
        print(
            f"  {b['name']:14s} {b['dim']:4d} {b['trace_on_A_raw']:14.4f} "
            f"{b['trace_on_A_z']:8.3f} {b['scale']:8.4f}"
        )
    print(
        "  trace_z = trace after per-column z-scoring (non-constant width); 1/sqrt = the applied "
        "block scale"
    )

    acc = body["acceptance"]
    print("\nDEC-669 frozen acceptance (block_equalised vs plain)")
    print(
        f"  aggregate improves: {acc['aggregate_improves']}  "
        f"(d_nll={acc['aggregate_delta']:+.6f}, CI "
        f"[{acc['aggregate_ci95_vs_plain'][0]:+.6f}, {acc['aggregate_ci95_vs_plain'][1]:+.6f}])"
    )
    print(
        "  target gains (plain - arm): "
        + " ".join(f"{c}={acc['target_gains'][c]:+.6f}" for c in TARGET_CLASSES)
        + f"  floor={TARGET_GAIN}  hit={acc['targets_hit']}"
    )
    print(
        f"  worst class regression: {acc['worst_class']}={acc['worst_class_delta']:+.6f} "
        f"(cap {MAX_REGRESSION})  no_class_regresses={acc['no_class_regresses']}"
    )
    print(f"  VERDICT: {body['verdict']}   (no production file changed; nothing adopted)")
    print(f"seconds={body['seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
