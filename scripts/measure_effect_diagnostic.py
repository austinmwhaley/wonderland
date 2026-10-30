"""Effect-axis diagnostic (DEC-454 follow-up): WHERE does the uplift failure live? CPU-only receipt.

Three protocols (B/C/D) all scored NEGATIVE out-of-sample R² on the identified per-customer
uplift — the state has never ranked WHO responds better than noise. Before asking for another
retrain, this script separates the axes using the ALREADY-trained artifact:

  (A) arm_response_by_level        the randomized offer arm must move response in the LOGGED
                                   data at all, or no encoder can be graded on uplift here
  (F) head_factual_spearman        does the trained effect head fit the FACTUAL per-(customer,
                                   level) response rates? (objective/capacity axis)
  (V) head_uplift_spearman         does the head's own per-customer contrast (mirroring the
                                   identified uplift exactly: opt vs levels ≥2 away, day-weighted)
                                   rank the known uplifts? (contrast axis)
  (W) contrast_norm_ratio          did the fitted head even differentiate arms?
  (R) head_vs_ridge_cosine         is the head's contrast direction (mapped into E space through
                                   the exact serving inverse, self-verified by round-trip)
                                   aligned with the ridge direction fit on split A?
  (P) probe_r2_check               the probe's own brain R² recomputed here must match the card
                                   (a diagnostic reading different arms than the judge graded is
                                   a second judge — DEC-428's drift rule)

Interpretation (recorded as guide strings, not a verdict — this is a diagnostic, not a gate):
  A ≈ flat            -> no measurable treatment effect in this sample; the probe cannot grade
                         ANY encoder (data axis, not encoder axis)
  F high, V ≈ 0       -> the state carries response LEVEL but not the arm CONTRAST: the failure
                         is in what the state carries about the treatment (input/tokenizer axis)
  F low               -> the head does not fit even the factuals: objective/capacity axis
  V high, probe R² < 0 -> the CONTRAST is present in the state but the frozen linear probe
                         cannot use it; the probe would need to consume the head — that is an
                         eval-contract change, i.e. an operator ruling, not a loop change

Usage (inside the memory-capped container):
  scripts/wonder_container.sh -- .venv/bin/python scripts/measure_effect_diagnostic.py \
      --products /work/cfm_d_qte --demo /work/demo --out /tmp/effect_diag.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_battery_law import (  # noqa: E402  (one definition for every arm, head, and CI)
    MIN_ARM_DAYS,
    _ridge_head,
    _ridge_oos,
    arm_cells,
    identified_uplift,
)
from measure_judge_law import _arms, _common_populations  # noqa: E402  (the judge's OWN geometry)


def _spearman(x, y):
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return None, int(m.sum())
    return float(spearmanr(x[m], y[m]).statistic), int(m.sum())


def _state_inverse(model):
    """(G, c0) with head-state `s = E @ G + c0` — the EXACT inverse of the serving map.

    The head was trained on the pre-projection readout state s; products publish
    E = boundary(proj(s)). Both boundary branches are linear and invertible:
      isometry   E = (v @ R.T) * scale      =>  v = (E / scale) @ R
      whitening  E = (v - mu) @ Wh           =>  v = E @ inv(Wh) + mu
    and proj(s) = s @ Wp.T + bp              =>  s = (v - bp) @ inv(Wp.T).
    The caller verifies the composition by round-trip before trusting any number.
    """
    import torch

    with torch.no_grad():
        Wp = model.proj.weight.double()
        bp = model.proj.bias.double()
        if getattr(model, "isometric_boundary", False):
            Gb = model.iso_matrix().double() / float(model.boundary_scale)
            mu = torch.zeros(model.proj.weight.shape[1], dtype=torch.float64)
        else:
            Gb = torch.linalg.inv(model.whiten_W.double())
            mu = model.whiten_mean.double()
        invWpT = torch.linalg.inv(Wp.T)
        G = (Gb @ invWpT).cpu().numpy()
        c0 = ((mu - bp) @ invWpT).cpu().numpy()
    return G, c0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--products", required=True, help="dir holding cfm_products.duckdb + cfm_*.pt")
    ap.add_argument("--demo", required=True, help="demo dir with reference_truth.json")
    ap.add_argument(
        "--out",
        default=str(ROOT / "docs/report_cards/experiments/effect_diagnostic.json"),
    )
    a = ap.parse_args()
    t0 = time.time()
    products = Path(a.products)
    db = products / "cfm_products.duckdb"
    if not db.exists():
        db = products  # producers also accept the direct path
    rep: dict = {"working": False, "products": str(products)}
    try:
        _run(products, db, Path(a.demo), rep)
    except Exception as exc:  # noqa: BLE001 - a diagnostic receipts its own failure, loudly
        rep["note"] = f"{type(exc).__name__}: {exc}"
    _write(rep, a.out, t0)
    return 0


def _run(products: Path, db: Path, demo: Path, rep: dict) -> None:
    import torch

    from looking_glass.cfm_state import load_frozen_encoder

    arms = _arms(db, demo)
    if arms["problems"]:
        rep["note"] = "arms not aligned: " + "; ".join(arms["problems"][:3])
        return
    ckpt = sorted(products.glob("cfm_*.pt"))
    if not ckpt:
        rep["note"] = f"no cfm_*.pt checkpoint under {products}"
        return
    tag = ckpt[0].stem[len("cfm_") :].replace("_", ".", 2)
    model, _cfg = load_frozen_encoder(tag, products)
    rep["tag"] = tag
    if getattr(model, "head_effect", None) is None or getattr(model, "effect_head", "mlp") != (
        "bilinear"
    ):
        rep["note"] = (
            f"{tag}: effect_head={getattr(model, 'effect_head', None)} — this diagnostic "
            "reads the QTE head (protocol D)"
        )
        return

    cells = arm_cells(arms["stream"], arms["truth"])
    up = identified_uplift(cells)
    Q = int(model.effect_dim)

    # (A) does the randomized arm move response in the LOGGED data at all?
    rates = (
        cells.group_by("off")
        .agg(pl.col("buys").sum().alias("b"), pl.col("days").sum().alias("d"))
        .sort("off")
    )
    rep["arm_response_by_level"] = {
        int(r["off"]): {"rate": round(float(r["b"]) / max(r["d"], 1), 5), "days": int(r["d"])}
        for r in rates.iter_rows(named=True)
    }

    # population: the judge's OWN split and intersection — (P) must be able to match the card
    col = arms["columns"]["brain"]
    ak, bk = _common_populations(arms, "brain", set(up))
    if len(ak) < 8 or len(bk) < 8:
        rep["note"] = f"only {len(ak)} A / {len(bk)} B identified customers — too few to grade"
        return
    Ea = np.vstack([col["A"][k] for k in ak])
    Eb = np.vstack([col["B"][k] for k in bk])
    ya = np.array([up[k] for k in ak])
    yb = np.array([up[k] for k in bk])
    rep["n_identified_A"] = len(ak)
    rep["n_identified_B"] = len(bk)

    # (P) the probe recomputed here must equal the card's brain-uplift value
    _p, r2 = _ridge_oos(Ea, ya, Eb, yb)
    rep["probe_r2_check"] = None if r2 is None else round(float(r2), 4)

    # (R) the ridge direction on A — the linear functional the frozen probe actually learned.
    # _ridge_head() is a StandardScaler->RidgeCV pipeline (one definition, DEC-407), so the
    # fitted coef_ lives in scaled space; divide by the scale to land back in E space.
    rm = _ridge_head()
    rm.fit(Ea, ya)
    sc, rd = rm.steps[0][1], rm.steps[1][1]
    w_ridge = rd.coef_.ravel() / sc.scale_

    # inverse-map E -> head state, SELF-VERIFIED by round-trip on the scored population
    G, c0 = _state_inverse(model)
    E_all = np.vstack([Ea, Eb])
    s_all = E_all @ G + c0
    with torch.no_grad():
        back = model.donor_batch(torch.from_numpy(s_all).float()).numpy().astype(np.float64)
    rt = float(np.max(np.abs(back - E_all)))
    rep["roundtrip_max_err"] = rt
    if not np.isfinite(rt) or rt > 1e-2:
        rep["note"] = f"E->state->E round-trip off by {rt:.2e}: the inverse is not the serving map"
        return

    W = model.head_effect.proj.weight.detach().numpy().astype(np.float64)  # (Q, dim)
    b = model.head_effect.proj.bias.detach().numpy().astype(np.float64)  # (Q,)
    sB = Eb @ G + c0

    # (F) head fit on FACTUALS: level-q head score vs the logged per-cell response rate
    cl = cells.with_columns((pl.col("buys") / pl.col("days")).alias("rate"))
    key_to_rowB = {k: i for i, k in enumerate(bk)}
    fs: dict = {}
    for q in range(Q):
        sub = cl.filter((pl.col("off") == q) & (pl.col("days") >= MIN_ARM_DAYS))
        idx, tgt = [], []
        for r in sub.iter_rows(named=True):
            i = key_to_rowB.get(r["customer_key"])
            if i is not None:
                idx.append(i)
                tgt.append(r["rate"])
        if not idx:
            continue
        rho, n = _spearman(sB[np.array(idx)] @ W[q] + b[q], tgt)
        fs[str(q)] = {"rho": None if rho is None else round(rho, 4), "n_cells": n}
    rep["head_factual_spearman"] = fs

    # (W) did the fitted head differentiate the arms at all?
    spread = float(np.mean(np.linalg.norm(W - W.mean(0), axis=1)))
    scale = float(np.mean(np.linalg.norm(W, axis=1)))
    rep["contrast_norm_ratio"] = round(spread / max(scale, 1e-12), 4)

    # (V)+(R) the head's OWN contrast on the scored population, mirroring identified_uplift
    # exactly: opt vs the day-weighted mean over levels >= 2 away.
    idk = [k for k in bk if k in up]
    days_by: dict = {}
    want = pl.DataFrame({"customer_key": idk})
    for r in cells.join(want, on="customer_key", how="inner").iter_rows(named=True):
        days_by.setdefault(r["customer_key"], []).append((int(r["off"]), int(r["days"])))
    pred = np.zeros(len(idk))
    dirs = np.zeros((len(idk), W.shape[1]))
    for i, k in enumerate(idk):
        opt = int(arms["truth"][k]["offer_opt"])
        present = {q: d for q, d in days_by.get(k, [])}
        bad = [q for q in range(Q) if abs(q - opt) >= 2 and present.get(q, 0) > 0]
        if not bad:
            continue
        wts = np.array([present[q] for q in bad], dtype=np.float64)
        wts /= wts.sum()
        dvec = W[opt] - (wts[:, None] * W[np.array(bad)]).sum(0)
        dirs[i] = dvec
        pred[i] = sB[key_to_rowB[k]] @ dvec + (b[opt] - float(wts @ b[np.array(bad)]))
    yv = np.array([up[k] for k in idk])
    good = np.abs(dirs).sum(1) > 0
    rho, n = _spearman(pred[good], yv[good])
    rep["head_uplift_spearman"] = None if rho is None else round(rho, 4)
    rep["n_head_uplift"] = n
    vE = dirs[good] @ G.T
    vE = vE / np.maximum(np.linalg.norm(vE, axis=1, keepdims=True), 1e-12)
    wn = w_ridge / max(float(np.linalg.norm(w_ridge)), 1e-12)
    cos = vE @ wn
    rep["head_vs_ridge_cosine_mean"] = None if cos.size == 0 else round(float(cos.mean()), 4)
    rep["head_vs_ridge_cosine_absmean"] = (
        None if cos.size == 0 else round(float(np.abs(cos).mean()), 4)
    )

    # interpretation guide (strings, not a verdict — the card's gates are untouched here)
    lv = list(rep["arm_response_by_level"].values())
    guide: list = []
    if lv and max(x["rate"] for x in lv) - min(x["rate"] for x in lv) < 0.005:
        guide.append(
            "A flat: the randomized arm does not move response in this sample — uplift is "
            "not gradeable on it (data axis, not encoder axis)"
        )
    fs_vals = [v["rho"] for v in fs.values() if v["rho"] is not None]
    if fs_vals and min(fs_vals) > 0.15 and (rho is None or abs(rho) < 0.1):
        guide.append(
            "F fits, V ~ 0: the state carries response LEVEL but not the arm CONTRAST "
            "(input/tokenizer axis)"
        )
    if fs_vals and max(fs_vals) < 0.05:
        guide.append(
            "F low: the head does not fit even factual responses (objective/capacity axis)"
        )
    if (
        rho is not None
        and abs(rho) > 0.15
        and rep["probe_r2_check"] is not None
        and rep["probe_r2_check"] < 0
    ):
        guide.append(
            "V present but probe R2 negative: the contrast exists in the state and the frozen "
            "linear probe cannot use it — probe-consumes-head is an eval-contract change "
            "(operator ruling)"
        )
    rep["interpretation"] = guide
    rep["working"] = True


def _write(rep: dict, out: str, t0: float) -> None:
    rep["seconds"] = round(time.time() - t0, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(f"effect_diagnostic working={rep['working']} -> {out} ({rep['seconds']}s)", flush=True)
    for k in (
        "arm_response_by_level",
        "head_factual_spearman",
        "head_uplift_spearman",
        "n_head_uplift",
        "contrast_norm_ratio",
        "head_vs_ridge_cosine_mean",
        "probe_r2_check",
        "roundtrip_max_err",
        "interpretation",
        "note",
    ):
        if k in rep:
            print(f"  {k}: {rep[k]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
