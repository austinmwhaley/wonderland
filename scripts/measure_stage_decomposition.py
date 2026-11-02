#!/usr/bin/env python3
"""B3.2 stage decomposition: where is the fine structure lost?

M0 exact counter (39-d) -> M1 fold (w_counter) -> M2 state+fold -> M3 post-proj -> M4 served.

The rung decomposition (DEC-555/556) found the MASS-metric drop at M0->M1. This producer asks the
same question for the FINE structure the B+C legs need: persona geometry, future geometry and
segment structure, at every stage. It writes one receipt; the one card renders it.

The point is attribution, not a new capability gate: if the metric is already low at M0, the
trunk/objective is the wall; if it is high at M0/M1 and collapses by M3/M4, the serving contract
is the wall. No retraining is implied by this receipt alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from measure_judge_law import (  # noqa: E402
    SEED,
    TARGETS,
    _alignment,
    _arms,
    _fit_scale,
    _z1,
)
from safe_run import ensure_boundary  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_stage_decomposition.json"
STAGES = ("M0_exact_counter", "M1_fold_only", "M2_state_plus_fold", "M3_post_proj", "M4_served")
METRICS = ("persona_geometry", "future_geometry", "segment_ari")


def _load_stage_arrays(products: Path) -> dict[str, dict[str, np.ndarray]]:
    """The stage inputs, read from the products DB and the frozen checkpoint.

    `customer_state.state` is the 256-d state h; `customer_state.embedding` is the 39-d exact
    counter readout c; `state_embeddings.embedding` is the served E. The checkpoint supplies the
    frozen `w_counter` fold and the shared `proj`.
    """
    import duckdb

    db = products / "cfm_products.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    rows = con.sql("select customer_key, state, embedding from customer_state").fetchall()
    h = {k: np.asarray(s, float) for k, s, _e in rows}
    c = {k: np.asarray(e, float) for k, _s, e in rows}
    served = {
        k: np.asarray(e, float)
        for k, e in con.sql("select customer_key, embedding from state_embeddings").fetchall()
    }
    con.close()

    regs = sorted(products.glob("registry_*.json"), key=lambda q: q.stat().st_mtime)
    if not regs:
        raise SystemExit(f"stage decomposition: no registry_*.json under {products}")
    tag = json.loads(regs[-1].read_text()).get("tag")
    if not tag:
        raise SystemExit(f"stage decomposition: registry under {products} has no tag")
    ckpt = products / f"cfm_{tag.replace('.', '_')}.pt"
    if not ckpt.exists():
        raise SystemExit(f"stage decomposition: checkpoint {ckpt} missing")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = blob["state"]
    wc = sd["w_counter.weight"].detach().cpu().numpy().astype(np.float64)
    bc = sd["w_counter.bias"].detach().cpu().numpy().astype(np.float64)
    wp = sd["proj.weight"].detach().cpu().numpy().astype(np.float64)
    bp = sd["proj.bias"].detach().cpu().numpy().astype(np.float64)

    keys = [k for k in h if k in c and k in served]
    m0 = {k: c[k] for k in keys}
    m1 = {k: c[k] @ wc.T + bc for k in keys}
    m2 = {k: h[k] + m1[k] for k in keys}
    m3 = {k: m2[k] @ wp.T + bp for k in keys}
    m4 = {k: served[k] for k in keys}
    return {
        "M0_exact_counter": m0,
        "M1_fold_only": m1,
        "M2_state_plus_fold": m2,
        "M3_post_proj": m3,
        "M4_served": m4,
    }


def _geometry(stage_a, stage_b, ka, kb, truth, fields) -> tuple[float | None, int, int]:
    """Squared correlation between the stage's pairwise distances and the lab's true distances."""
    from scipy.spatial.distance import pdist

    vec = lambda k: np.asarray([truth[k][f] for f in fields], float)  # noqa: E731
    mu_t, sd_t = _fit_scale({k: vec(k) for k in ka})
    dt = pdist(np.vstack([_z1(vec(k), mu_t, sd_t) for k in kb]))
    mu, sd = _fit_scale({k: stage_a[k] for k in ka})
    d = pdist(np.vstack([_z1(stage_b[k], mu, sd) for k in kb]))
    return _alignment(d, dt), len(kb), int(dt.size)


def _segment_ari(stage_a, stage_b, ka, kb, seg, klab) -> float | None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    mu, sd = _fit_scale({k: stage_a[k] for k in ka})
    xa = np.vstack([_z1(stage_a[k], mu, sd) for k in ka])
    xb = np.vstack([_z1(stage_b[k], mu, sd) for k in kb])
    assign = KMeans(n_clusters=klab, n_init=4, random_state=SEED).fit(xa).predict(xb)
    return float(adjusted_rand_score([seg[k] for k in kb], assign))


def _round(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def main() -> int:
    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="/home/austin_whaley/wonderland_work/cfm_v2k")
    ap.add_argument("--demo", default="/home/austin_whaley/wonderland_work/demo_v2")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.perf_counter()
    products = Path(a.products)
    demo = Path(a.demo)
    stages = _load_stage_arrays(products)
    arms = _arms(products / "cfm_products.duckdb", demo)
    if arms["problems"]:
        raise SystemExit(
            "stage decomposition: arms not aligned: " + "; ".join(arms["problems"][:3])
        )
    truth, akeys, bkeys = arms["truth"], arms["akeys"], arms["bkeys"]

    sample = next(iter(truth.values()))
    numeric = tuple(sorted(k for k, v in sample.items() if isinstance(v, (int, float))))
    future_fields = [f for f in TARGETS if f in numeric]
    persona_fields = [f for f in numeric if f not in set(TARGETS) | {"offer_opt"}]
    seg = {k: truth[k].get("segment") for k in truth if truth[k].get("segment") is not None}
    klab = len({seg[k] for k in bkeys if k in seg})

    table: dict = {}
    for name in STAGES:
        mat = stages[name]
        ka = [k for k in akeys if k in truth and k in mat]
        kb = [k for k in bkeys if k in truth and k in mat]
        row: dict = {"dim": int(next(iter(mat.values())).shape[0]), "n": len(kb)}
        for metric, fields in (
            ("persona_geometry", persona_fields),
            ("future_geometry", future_fields),
        ):
            if not fields or len(ka) < 30 or len(kb) < 30:
                row[metric] = None
                continue
            score, n, pairs = _geometry(mat, mat, ka, kb, truth, fields)
            row[metric] = _round(score)
            row[f"{metric}_n"] = n
            row[f"{metric}_pairs"] = pairs
        ka_s = [k for k in ka if k in seg]
        kb_s = [k for k in kb if k in seg]
        row["segment_ari"] = (
            _round(_segment_ari(mat, mat, ka_s, kb_s, seg, klab))
            if klab >= 2 and len(ka_s) >= 30 and len(kb_s) >= 30
            else None
        )
        table[name] = row

    # Validate the reconstruction against the stored artifact: the served stage must be the
    # stored served embedding, and M3 vs M4 must be distance-identical (isometric boundary).
    from scipy.spatial.distance import pdist

    kb = [k for k in bkeys if k in stages["M3_post_proj"] and k in stages["M4_served"]]
    m3_m4 = _alignment(
        pdist(np.vstack([stages["M3_post_proj"][k] for k in kb])),
        pdist(np.vstack([stages["M4_served"][k] for k in kb])),
    )

    # Attribution: the largest adjacent drop for each metric (positive = structure lost).
    attribution: dict = {}
    for metric in METRICS:  # principle-ok: 3 metric names, not rows/time/examples
        vals = [(s, table[s].get(metric)) for s in STAGES if table[s].get(metric) is not None]
        drops = [
            (f"{vals[i][0]}->{vals[i + 1][0]}", float(vals[i][1]) - float(vals[i + 1][1]))
            for i in range(len(vals) - 1)  # principle-ok: adjacent stage pairs, not rows/time
        ]
        attribution[metric] = (
            max(drops, key=lambda kv: kv[1])[0]
            if drops and max(d for _, d in drops) > 0
            else "none"
        )

    complete = bool(
        table
        and all(table[s].get("persona_geometry") is not None for s in STAGES)
        and m3_m4 is not None
        and len(kb) >= 30
    )
    body = {
        "kind": "stage_decomposition",
        "complete": complete,
        "products": products_identity(products),
        "n_fit_A": len(akeys),
        "n_score_B": len(bkeys),
        "n_pairs": int(len(kb) * (len(kb) - 1) // 2),
        "stages": table,
        "attribution": attribution,
        "validation": {
            "m3_m4_distance_alignment": _round(m3_m4),
            "served_is_stored_state_embeddings": True,
            "persona_fields": persona_fields,
            "future_fields": future_fields,
            "segment_k": int(klab),
            "definition": "M0 exact counter -> M1 w_counter fold -> M2 state+fold -> "
            "M3 shared proj -> M4 served; distance fidelity is squared corr vs the lab's "
            "true pairwise distances, z-scored on A and scored on B",
        },
        "seconds": 0.0,
    }
    body["seconds"] = round(time.perf_counter() - t0, 1)
    stamp_merge_write(
        Path(a.out),
        body,
        " ".join([Path(__file__).name] + sys.argv[1:]),
        stamp_blocks=False,
    )
    print(f"stage decomposition -> {a.out} ({body['seconds']}s)")
    for s in STAGES:
        r = table[s]
        print(
            f"  {s:20s} dim={r['dim']:3d} persona={r['persona_geometry']} "
            f"future={r['future_geometry']} segment_ari={r['segment_ari']}"
        )
    print(f"  attribution: {attribution}")
    print(f"  M3 vs M4 distance alignment: {body['validation']['m3_m4_distance_alignment']}")
    print(f"  complete={complete}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
