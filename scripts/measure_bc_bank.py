"""B+C judge protocol v2 on the served block bank (DEC-598).

The flat judge asks one question of one vector. The B battery showed the donor is a block
bank; this producer gives that bank a simple consumer:

  supervised   standardized ridge alpha=10 on the bank (progress) and on the bank + compact
               judge_ledger (delivery). The canonical GBM v1 is recorded beside it.
  geometry     per-block pairwise distances, z-scored on split A; non-negative block weights
               fit on split-A pairs to the truth distance, then scored on split B. The segment
               task uses the served profile block (the categorical mass). The same matched
               random/destroyed controls as the flat leg are carried.
  rl/sim       the canonical legs on the bank (they already pass).

Weights/alpha are fit on split A only. The table comparator, bars and task list are unchanged.
Writes judge_leg_{supervised,unsupervised,rl,simulator}.json (stamped).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_battery_law import SEED, _paired_ci, controls_matched  # noqa: E402
from measure_views_law import _target_matrix  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from measure_judge_law import (  # noqa: E402
    COMPARATOR,
    LEGS,
    TARGETS,
    _alignment,
    _arms,
    _evidence,
    _fit_scale,
    _gbm_oos,
    _paired_stat_ci,
    _r2,
    _task,
    _z1,
)

OUT = ROOT / "docs" / "report_cards" / "experiments"
DEFAULT_PRODUCTS = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_bank"
DEFAULT_DEMO = ROOT / "looking_glass/artifacts/demo_v2"
ALPHA = 10.0

# The receipt's `arms` must describe the consumer THIS producer measured, not the flat-era
# donor: a merge that only adds keys lets the old description survive every bank-era re-stamp
# and contradict the receipt's own `protocol` (B-013/DEC-608). One definition per leg.
BANK_ARMS = {
    "supervised": {
        "brain": (
            "served block bank cfm_v2k_vf2_bank (179 dims, DEC-598) + standardized ridge "
            "alpha=10 (progress arm)"
        ),
        "brain_ledger": "bank + compact judge_ledger (PCA-39, fit on split A) — delivery arm",
        "pathA": "PATH A wide table, canonical GBM incumbent (frozen comparator), same rows",
    },
    "unsupervised": {
        "brain": (
            "served block bank (DEC-598); per-block distances z-scored on A, non-negative "
            "block weights fit on split-A pairs, scored on B; segment uses the served "
            "profile block"
        ),
        "brain_ledger": "bank + judge_ledger (PCA-39), same distance/clustering geometry",
        "pathA": "PATH A wide table, z-scored on split A only, same geometry",
    },
    "rl": {
        "brain": (
            "served block bank (DEC-598) + linear ridge head on the identified uplift "
            "(canonical leg)"
        ),
        "brain_ledger": "bank + judge_ledger, same head and rows",
        "pathA": "PATH A wide table + the DEC-428 GBM incumbent, same rows",
    },
    "simulator": {
        "brain": (
            "served block bank (DEC-598): dose-curve task on a single [bank; level] head; "
            "argmax task on per-level heads querying the bank at each action (canonical leg)"
        ),
        "brain_ledger": "bank + judge_ledger, same heads and cells",
        "pathA": "PATH A wide table + level one-hot -> the DEC-428 GBM incumbent, same cells",
    },
}


def _load(products: Path, demo: Path):
    import duckdb

    receipt = json.loads((products / "bank_receipt.json").read_text())
    con = duckdb.connect(str(products / "cfm_products.duckdb"), read_only=True)
    try:
        E = {
            str(k): np.asarray(v, float)
            for k, v in con.execute(
                "select customer_key, embedding from state_embeddings"
            ).fetchall()
        }
        ledger = {
            str(k): np.asarray(v, float)
            for k, v in con.execute(
                "select customer_key, embedding from view_embeddings where view='judge_ledger'"
            ).fetchall()
        }
    finally:
        con.close()
    arms = _arms(products / "cfm_products.duckdb", demo)
    if arms["problems"]:
        raise SystemExit("bank arms not aligned: " + "; ".join(arms["problems"][:3]))
    keys = sorted(E)
    blocks = {}
    for b in receipt["blocks"]:
        s, d = int(b["start"]), int(b["dim"])
        blocks[b["name"]] = {k: E[k][s : s + d] for k in keys}
    blocks["judge_ledger"] = ledger
    return receipt, blocks, arms, keys


def _ridge(alpha=ALPHA):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _supervised_tasks(arms, blocks, truth, akeys, bkeys, table):
    out = {}
    for col in ("brain", "brain_ledger"):
        A = np.vstack([arms["columns"][col]["A"][k] for k in akeys])
        B = np.vstack([arms["columns"][col]["B"][k] for k in bkeys])
        rng = np.random.default_rng(SEED)
        rand_b, dest_b = controls_matched(rng, A, B)
        rand_a, dest_a = controls_matched(rng, A, A)
        tasks = []
        usable, _excluded = _target_matrix(truth, akeys, bkeys)
        for name, (ya, yb) in usable.items():
            if float(np.std(ya)) < 1e-12 or float(np.std(yb)) < 1e-12:
                tasks.append({"name": name, "degenerate": True, "degenerate_reason": "no variance"})
                continue
            head = _ridge()
            pb = head.fit(A, ya).predict(B)
            mt = _gbm_oos(
                np.vstack([table[k] for k in akeys]), ya, np.vstack([table[k] for k in bkeys])
            )
            pr = head.fit(rand_a, ya).predict(rand_b)
            pd = head.fit(dest_a, ya).predict(dest_b)
            _m, lo, hi = _paired_ci((yb - mt) ** 2 - (yb - pb) ** 2)
            tasks.append(
                _task(
                    name,
                    "held-out R2 on split B",
                    yb,
                    pb,
                    mt,
                    _r2(yb, pr),
                    _r2(yb, pd),
                    (lo, hi),
                )
            )
        out[col] = {"tasks": tasks}
    return out


def _block_dists(blocks, names, akeys, bkeys):
    da, db = {}, {}
    for n in names:
        mu, sd = _fit_scale({k: blocks[n][k] for k in akeys})
        da[n] = pdist(np.vstack([_z1(blocks[n][k], mu, sd) for k in akeys]))
        db[n] = pdist(np.vstack([_z1(blocks[n][k], mu, sd) for k in bkeys]))
    return da, db


def _geometry_tasks(blocks, truth, akeys, bkeys, table, flat_tasks, names):
    num = tuple(
        sorted(k for k, v in next(iter(truth.values())).items() if isinstance(v, (int, float)))
    )
    persona = [f for f in num if f not in set(TARGETS) | {"offer_opt"}]
    future = [f for f in TARGETS if f in num]
    da, db = _block_dists(blocks, names, akeys, bkeys)
    X = np.column_stack([da[n] for n in names])
    rng = np.random.default_rng(SEED)
    tasks = []
    for task, fields in (("persona_geometry", persona), ("future_geometry", future)):
        vec = lambda k: np.asarray([truth[k][f] for f in fields], float)  # noqa: E731
        mu, sd = _fit_scale({k: vec(k) for k in akeys})
        dt_a = pdist(np.vstack([_z1(vec(k), mu, sd) for k in akeys]))
        dt_b = pdist(np.vstack([_z1(vec(k), mu, sd) for k in bkeys]))
        idx = rng.choice(len(dt_a), size=min(200_000, len(dt_a)), replace=False)
        w, _ = nnls(X[idx], dt_a[idx])
        pred = sum(w[i] * db[n] for i, n in enumerate(names))
        score = float(_alignment(pred, dt_b))
        cap = min(dt_b.size, 200_000)
        pidx = (
            rng.choice(dt_b.size, size=cap, replace=False)
            if dt_b.size > cap
            else np.arange(dt_b.size)
        )
        Db, Dt = pred[pidx], dt_b[pidx]
        muw, sdw = _fit_scale({k: table[k] for k in akeys})
        d_table = pdist(np.vstack([_z1(table[k], muw, sdw) for k in bkeys]))
        Dw = d_table[pidx]
        lo, hi = _paired_stat_ci(
            lambda q: (_alignment(Db[q], Dt[q]), _alignment(Dw[q], Dt[q])), cap
        )
        base = next((t for t in flat_tasks if t.get("name") == task), {})
        tasks.append(
            {
                "name": task,
                "metric": "squared correlation between the block-weighted brain distance and "
                f"the lab's true distance ({len(fields)} fields)",
                "brain": round(score, 6),
                "pathA": round(float(_alignment(d_table, dt_b)), 6),
                "ci_lo": round(float(lo), 6),
                "ci_hi": round(float(hi), 6),
                "n": int(dt_b.size),
                "n_bootstrap_pairs": int(cap),
                "random": base.get("random", 0.0),
                "destroyed": base.get("destroyed", 0.0),
                "comparator": COMPARATOR,
                "block_weights": {n: round(float(w[i]), 6) for i, n in enumerate(names)},
            }
        )
    seg = {k: truth[k].get("segment") for k in bkeys if truth[k].get("segment") is not None}
    sk = [k for k in bkeys if k in seg]
    klab = len({seg[k] for k in sk})
    if klab >= 2 and len(sk) >= 30:
        prof = blocks["profile"]
        mu, sd = _fit_scale({k: prof[k] for k in akeys})
        Xb = np.vstack([_z1(prof[k], mu, sd) for k in sk])
        muw, sdw = _fit_scale({k: table[k] for k in akeys})
        tb = np.vstack([_z1(table[k], muw, sdw) for k in sk])
        yb = np.asarray([seg[k] for k in sk])
        blab = KMeans(n_clusters=klab, random_state=SEED, n_init=10).fit_predict(Xb)
        tlab = KMeans(n_clusters=klab, random_state=SEED, n_init=10).fit_predict(tb)
        lo, hi = _paired_stat_ci(
            lambda q: (adjusted_rand_score(yb[q], blab[q]), adjusted_rand_score(yb[q], tlab[q])),
            len(sk),
        )
        base = next((t for t in flat_tasks if t.get("name") == "segment_structure"), {})
        tasks.append(
            {
                "name": "segment_structure",
                "metric": "adjusted Rand index against the lab's declared segment "
                "(k from the label cardinality); brain uses the served profile block",
                "brain": round(float(adjusted_rand_score(yb, blab)), 6),
                "pathA": round(float(adjusted_rand_score(yb, tlab)), 6),
                "ci_lo": round(float(lo), 6),
                "ci_hi": round(float(hi), 6),
                "n": int(len(sk)),
                "random": base.get("random", 0.0),
                "destroyed": base.get("destroyed", 0.0),
                "comparator": COMPARATOR,
            }
        )
    return tasks


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    ap.add_argument("--demo", default=str(DEFAULT_DEMO))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--legs", default="supervised,unsupervised,rl,simulator")
    a = ap.parse_args()

    t0 = time.perf_counter()
    products, demo = Path(a.products), Path(a.demo)
    receipt, blocks, arms, keys = _load(products, demo)
    truth, akeys, bkeys = arms["truth"], arms["akeys"], arms["bkeys"]
    table = arms["table"]
    bank_names = [b["name"] for b in receipt["blocks"]]
    ledger_names = bank_names + ["judge_ledger"]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for leg in [s.strip() for s in a.legs.split(",") if s.strip()]:
        if leg in ("rl", "simulator"):
            body = LEGS[leg](products / "cfm_products.duckdb", demo)
            body["protocol"] = "bank v2 (DEC-598): canonical leg on the served block bank"
        elif leg == "supervised":
            body = {
                "columns": _supervised_tasks(arms, blocks, truth, akeys, bkeys, table),
                "excluded": [],
                "protocol": (
                    "bank v2 (DEC-598): standardized ridge alpha=10 on the bank (progress) and "
                    "bank + compact judge_ledger (delivery); table = canonical GBM. The GBM v1 "
                    "flat numbers are in the canonical receipt."
                ),
            }
        elif leg == "unsupervised":
            flat = LEGS["unsupervised"](products / "cfm_products.duckdb", demo)
            ft = (flat.get("columns", {}).get("brain", {}) or {}).get("tasks", [])
            body = {
                "columns": {
                    "brain": {
                        "tasks": _geometry_tasks(blocks, truth, akeys, bkeys, table, ft, bank_names)
                    },
                    "brain_ledger": {
                        "tasks": _geometry_tasks(
                            blocks, truth, akeys, bkeys, table, ft, ledger_names
                        )
                    },
                },
                "excluded": [],
                "protocol": (
                    "bank v2 (DEC-598): per-block distances z-scored on A, non-negative block "
                    "weights fit on split-A pairs, scored on split B; segment uses the served "
                    "profile block. The matched random/destroyed controls are from the flat leg."
                ),
            }
        else:
            continue
        body["products"] = products_identity(str(products))
        body["splits"] = {"n_fit_A": len(akeys), "n_score_B": len(bkeys)}
        # producer-owned provenance blocks, written fresh on every run: the frozen
        # comparator's metadata, the bank-v2 consumer description, and a `checks` rollup
        # that is this run's own working state (the flat-era residue is dropped below so a
        # re-stamp can never contradict `protocol` again).
        body["path_a"] = arms["path_a"]
        body["arms"] = BANK_ARMS[leg]
        working, evidence = _evidence(leg, body)
        body.update({"working": working, "evidence": evidence})
        body["checks"] = {leg: {"working": working, "evidence": evidence}}
        # the canonical LEG bodies carry a flat-era provenance block this producer does not
        # own (an empty `diagnostics`; the supervised receipt's `incumbent` record): strip it
        # so the bank receipt never re-adds what `drop` removes from the previous file.
        body.pop("diagnostics", None)
        body.pop("incumbent", None)
        body["kind"] = "judge_leg"
        body["complete"] = True
        stamp_merge_write(
            out_dir / f"judge_leg_{leg}.json",
            body,
            " ".join(sys.argv),
            stamp_blocks=False,
            drop=("diagnostics", "incumbent"),
        )
        print(f"bc-bank {leg}: working={working} {evidence[:180]}")
    print(f"WROTE judge_leg_*.json ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
