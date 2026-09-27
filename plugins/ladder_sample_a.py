"""Sample-A ladder — the smallest encoder training sample that keeps the signal.

Question: sample_A = 17,531 of 25,000 customers — is that the minimum? Each
rung retrains the encoder with `--sample-a N` (populations, cutoff, anchors and
the derived architecture are all held FIXED — only the training sample size
changes, and draws are nested), into its own archive dir, then measures:

  * `auc` — the frozen purchase-propensity target on the rung's own products
    (same split / same metric as plugins.ladder; identical eligible rows and
    test set across rungs, so the curve isolates encoder sample size)
  * `ce`  — the encoder's held-out next-event loss (training governor receipt)

Rule (same convention as plugins.ladder): CHOSEN = smallest rung within
2 x max(fold SE) of the best AUC. `ce` is reported as corroboration.

Receipts: <rung_dir>/ladder_receipt.json each + summary_<as_of>.json.

  python -m plugins.ladder_sample_a --as-of 2025-11-01 [--skip-done]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .base import STREAM_DB, load_dataset
from .head_template import _binary_family, _metrics_binary, _split
from .ladder import _fold_se, _metric
from .targets import PURCHASE_PROPENSITY_30D

WORK = Path(__file__).resolve().parents[1]
LADDER_DIR = WORK / "looking_glass" / "artifacts" / "cfm_ladder"
RUNGS = (250, 500, 1000, 2000, 4000, 8000, 12000, 20000)


def _train_rung(rung: int, as_of: str, customers: int, anchors: int, out: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "looking_glass.customer_foundation_model",
        "train",
        "--customers",
        str(customers),
        "--anchors",
        str(anchors),
        "--sample-a",
        str(rung),
        "--as-of",
        as_of,
        "--db",
        str(STREAM_DB),
        "--out-dir",
        str(out),
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORK))
    wall = round(time.perf_counter() - t0, 1)
    log = out / "train.log"
    log.write_text((r.stdout or "") + "\n" + (r.stderr or ""))
    if r.returncode != 0:
        raise RuntimeError(f"rung {rung} training failed (see {log}): {r.stderr[-400:]}")
    m = re.search(r"trained (\S+):", r.stdout or "")
    if not m:
        raise RuntimeError(f"rung {rung}: could not parse tag (see {log})")
    reg = json.loads((out / f"registry_{m.group(1).replace('.', '_')}.json").read_text())
    return {"tag": m.group(1), "train_wall_s": wall, "registry": reg}


def _downstream_auc(as_of: str, out: Path) -> dict:
    """Purchase-propensity AUC on the rung's own products (fixed split/metric)."""
    ds = load_dataset(
        30,
        cfm_products=out / "cfm_products.duckdb",
        target=PURCHASE_PROPENSITY_30D,
        as_of=as_of,
    )
    tr, te = _split(ds.keys, 0)
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(ds.X[tr])
    Xtr, Xte = sc.transform(ds.X[tr]), sc.transform(ds.X[te])
    clf = _binary_family("logistic", 0, ds.X.shape[1])
    clf.fit(Xtr, ds.y[tr].astype(int))
    pred = clf.predict_proba(Xte)[:, 1]
    yte = ds.y[te]
    m = _metrics_binary(pred, yte)
    se = _fold_se(pred, yte, np.asarray(ds.keys)[te], _metric("binary"))
    return {
        "auc": m["auc"],
        "auc_se": se,
        "top_decile_lift": m["top_decile_lift"],
        "n_rows": int(len(ds.y)),
        "n_customers": int(ds.meta["n_customers"]),
    }


def run(as_of: str, rungs=RUNGS, customers: int = 25000, anchors: int = 6, skip_done: bool = False):
    print(f"== SAMPLE-A LADDER (as_of={as_of}, base={customers}, anchors={anchors}) ==")
    print("architecture + populations + anchors are FIXED; only sample_A size varies\n")
    rows = []
    for rung in rungs:
        out = LADDER_DIR / f"r{rung}"
        receipt_p = out / "ladder_receipt.json"
        if skip_done and receipt_p.exists():
            rec = json.loads(receipt_p.read_text())
            rows.append(rec)
            print(
                f"  r{rung:>6} (cached) auc={rec['auc']:.4f} +/-{2 * rec['auc_se']:.4f} "
                f"ce={rec['ce']:.4f} train={rec['train_wall_s']:.0f}s",
                flush=True,
            )
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"  r{rung:>6}: training ...", flush=True)
        tr_info = _train_rung(rung, as_of, customers, anchors, out)
        gov = tr_info["registry"].get("governor", {})
        auc = _downstream_auc(as_of, out)
        rec = {
            "rung": rung,
            "tag": tr_info["tag"],
            "train_wall_s": tr_info["train_wall_s"],
            "n_train_sequences": tr_info["registry"].get("n_train_sequences"),
            "budget_steps": tr_info["registry"].get("resolved", {}).get("budget_steps"),
            "governor_steps": gov.get("steps"),
            "governor_stopped": gov.get("stopped"),
            "ce": gov.get("best"),
            **auc,
        }
        receipt_p.write_text(json.dumps(rec, indent=1, default=float))
        rows.append(rec)
        print(
            f"    tag={rec['tag']} n_train_seqs={rec['n_train_sequences']} "
            f"budget={rec['budget_steps']} stopped={rec['governor_stopped']} "
            f"train={rec['train_wall_s']:.0f}s ce={rec['ce']:.4f}",
            f"auc={rec['auc']:.4f} +/-{2 * rec['auc_se']:.4f} "
            f"lift={rec['top_decile_lift']:.3f} rows={rec['n_rows']}",
            flush=True,
        )

    finite = [r for r in rows if r.get("auc") is not None and np.isfinite(r["auc"])]
    tol = 2.0 * float(np.nanmax([r["auc_se"] for r in finite]))
    best = max(r["auc"] for r in finite)
    chosen = next(r for r in rows if r.get("auc") is not None and r["auc"] >= best - tol)
    print(f"\nmeasured noise 2*SE(AUC) = {tol:.4f}   best AUC = {best:.4f}")
    print(
        f"CHOSEN sample_A = {chosen['rung']} "
        f"(effective n_train_seqs={chosen['n_train_sequences']}, auc={chosen['auc']:.4f})"
    )
    summary = {
        "as_of": as_of,
        "customers_base": customers,
        "anchors": anchors,
        "tol": tol,
        "best_auc": best,
        "chosen_rung": chosen["rung"],
        "chosen_n_train_sequences": chosen["n_train_sequences"],
        "rungs": rows,
    }
    LADDER_DIR.mkdir(parents=True, exist_ok=True)
    path = LADDER_DIR / f"summary_{as_of}.json"
    path.write_text(json.dumps(summary, indent=1, default=float))
    print(f"receipt -> {path}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sample-A (encoder training) size ladder")
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--rungs", default=None, help="comma-separated sizes (default 250..20000)")
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--anchors", type=int, default=6)
    ap.add_argument("--skip-done", action="store_true", help="resume: reuse rungs with receipts")
    a = ap.parse_args(argv)
    rungs = tuple(int(x) for x in a.rungs.split(",")) if a.rungs else RUNGS
    run(a.as_of, rungs=rungs, customers=a.customers, anchors=a.anchors, skip_done=a.skip_done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
