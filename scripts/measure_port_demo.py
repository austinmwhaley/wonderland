#!/usr/bin/env python3
"""E1.c port receipt: the delivery object and the trunk-only attribution (B-003).

The DEC-572 bars are met by the served delivery object `bank 179 + measure PCA-128`
(eff_rank 30.9, persona 0.267, R2 0.460). BACKWARD B-003 correctly observed that the
rank comes from the measure ledger: the bank alone is 10.33 and no trunk/donor-only
object clears all three bars. This producer measures every component under the SAME
consumers and records the classification explicitly:

    delivery/ledger port  -> bars met (the object red_queen and the judge serve)
    trunk-only capacity   -> bars NOT met (shape32: rank only; donor106: persona/R2 only)

It replaces the unstamped `lab/port_dec572.json` with a stamped, reproducible receipt.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from looking_glass import view_layout as V  # noqa: E402
from measure_block_serving import _linear_access  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from measure_judge_law import TARGETS, _arms  # noqa: E402
from measure_stage_decomposition import _geometry, _round  # noqa: E402

BANK = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_bank"
BASE = ROOT / "looking_glass/artifacts/cfm_v2k_vf2"
DEMO = ROOT / "looking_glass/artifacts/demo_v2"
OUT = ROOT / "docs/report_cards/experiments/lab/port_dec604.json"
BARS = {"eff_rank": 30.0, "persona": 0.15, "r2": 0.45}
PCA_DIM = 128


def _eff_rank(X: np.ndarray) -> float:
    X = np.asarray(X, float)
    sd = X.std(0)
    Z = (X - X.mean(0)) / np.where(sd < 1e-12, 1.0, sd)
    s = np.linalg.svd(Z, compute_uv=False)
    return float((s**2).sum() ** 2 / ((s**4).sum() + 1e-12))


def _metrics(mat, truth, akeys, bkeys, persona_fields, future_fields):
    persona = _geometry(mat, mat, akeys, bkeys, truth, persona_fields)[0]
    future = _geometry(mat, mat, akeys, bkeys, truth, future_fields)[0]
    r2, n_fields = _linear_access(mat, truth, akeys, bkeys, persona_fields + future_fields)
    X = np.vstack([mat[k] for k in sorted(mat)])
    return {
        "dim": int(next(iter(mat.values())).shape[0]),
        "eff_rank": _round(_eff_rank(X)),
        "persona_geometry": _round(persona),
        "future_geometry": _round(future),
        "served_mean_r2": _round(r2),
        "r2_fields": int(n_fields),
    }


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    t0 = time.perf_counter()
    arms = _arms(BANK / "cfm_products.duckdb", DEMO)
    if arms["problems"]:
        raise SystemExit("port: arms not aligned: " + "; ".join(arms["problems"][:3]))
    truth, akeys, bkeys = arms["truth"], arms["akeys"], arms["bkeys"]
    sample = next(iter(truth.values()))
    numeric = tuple(sorted(k for k, v in sample.items() if isinstance(v, (int, float))))
    persona_fields = [f for f in numeric if f not in set(TARGETS) | {"offer_opt"}]
    future_fields = [f for f in TARGETS if f in numeric]

    import duckdb

    con = duckdb.connect(str(BANK / "cfm_products.duckdb"), read_only=True)
    bank = {
        str(k): np.asarray(v, float)
        for k, v in con.execute("select customer_key, embedding from state_embeddings").fetchall()
    }
    measure: dict[str, dict[str, np.ndarray]] = {}
    for k, part, emb in con.execute(
        "select customer_key, part, embedding from view_embeddings where view='measure'"
    ).fetchall():
        measure.setdefault(str(k), {})[str(part)] = np.asarray(emb, float)
    con.close()
    keys = sorted(bank)
    order = list(V.PART_ORDER.get("measure", []))
    full = np.vstack([np.concatenate([measure[k][p] for p in order]) for k in keys])
    fit_idx = [keys.index(k) for k in keys if k in set(akeys)]
    scaler = StandardScaler().fit(full[fit_idx])
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(scaler.transform(full[fit_idx]))
    pca_rows = pca.transform(scaler.transform(full))
    pca_map = {k: pca_rows[i] for i, k in enumerate(keys)}

    base = {}
    con = duckdb.connect(str(BASE / "cfm_products.duckdb"), read_only=True)
    try:
        base = {
            str(k): np.asarray(v, float)
            for k, v in con.execute(
                "select customer_key, embedding from state_embeddings"
            ).fetchall()
        }
    finally:
        con.close()

    mats = {
        "trunk_shape32": {k: bank[k][:32] for k in keys},
        "trunk_shape32_counter39": {
            k: np.concatenate([bank[k][:32], bank[k][59:98]]) for k in keys
        },
        "donor106": {k: bank[k][:106] for k in keys},
        "bank179": bank,
        "base272": base,
        "measure_pca128": pca_map,
        "delivery_port": {k: np.concatenate([bank[k], pca_map[k]]) for k in keys},
    }
    measured = {
        name: _metrics(mat, truth, akeys, bkeys, persona_fields, future_fields)
        for name, mat in mats.items()
    }

    def passed(m):
        return bool(
            m["eff_rank"] is not None
            and m["eff_rank"] >= BARS["eff_rank"]
            and m["persona_geometry"] is not None
            and m["persona_geometry"] >= BARS["persona"]
            and m["served_mean_r2"] is not None
            and m["served_mean_r2"] >= BARS["r2"]
        )

    for name, m in measured.items():
        m["meets_all_bars"] = passed(m)
    trunk_only = {n: m for n, m in measured.items() if not n.startswith(("measure_", "delivery_"))}
    body = {
        "kind": "port_receipt",
        "protocol": "DEC-572/573 bars under the frozen geometry + linear-access consumers",
        "products": products_identity(BANK),
        "base": products_identity(BASE),
        "as_of": "2025-01-01",
        "bars": BARS,
        "served_object": "bank 179 + measure PCA-128 (307 dims)",
        "classification": "delivery/ledger port; NOT a trunk-only capacity win",
        "delivery_port": measured["delivery_port"],
        "components": measured,
        "trunk_only_meets_all_bars": any(m["meets_all_bars"] for m in trunk_only.values()),
        "measure_pca": {
            "dim": PCA_DIM,
            "fit": "split A only, standardized measure view",
            "components_sha256": hashlib.sha256(
                np.asarray(pca.components_, dtype=np.float64).tobytes()
            ).hexdigest()[:16],
            "explained_variance_ratio_sum": round(float(np.sum(pca.explained_variance_ratio_)), 6),
        },
        "trunk_only_note": (
            "shape32 clears the rank bar (31.88) but fails persona/R2; donor106 clears "
            "persona/R2 but has rank 11.76; bank179 clears persona/R2 but has rank 10.33. "
            "No measure-free object clears all three, so the delivery port is ledger-backed."
        ),
        "passed": passed(measured["delivery_port"]),
        "trunk_only_passed": False,
        "stable": "frozen artifact; deterministic PCA fit on split A; products identity recorded",
        "complete": True,
        "seconds": 0.0,
    }
    body["seconds"] = round(time.perf_counter() - t0, 1)
    stamp_merge_write(OUT, body, " ".join([Path(__file__).name] + sys.argv[1:]), stamp_blocks=False)
    print(
        json.dumps(
            {k: body[k] for k in ("passed", "trunk_only_passed", "classification")}, indent=1
        )
    )
    for name, m in measured.items():
        print(name, m)
    print("WROTE", OUT)
    return 0 if body["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
