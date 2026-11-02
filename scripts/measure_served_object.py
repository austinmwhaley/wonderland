#!/usr/bin/env python3
"""B3.6 served-object X-ray: does the ledger (B's companion statistic) carry fine structure?

DEC-567's L1 layer said the judge's delivery column is a naive 256+595 concat that never read the
595-d ledger alone; DEC-569 pre-registered this producer before the run. It scores, under the
FROZEN B3.2/B3.4 consumers (imported, not redefined), the ledger full, each of its named parts,
and the ledger block-PCA (fit on split A only) at the two pre-registered widths, against the three
references the B rows already carry: the exact counter (39), the served shape (256), and PATH A
(the wide table encoded once over A union B by the judge's own matrix path — the fair additive
family). The pre-registered branch (>= 0.30 exposure / .15-.30 partial / < .15 insufficient) is
COMPUTED BY THE CARD from these arm numbers (DEC-569); this receipt reports measurements only and
carries no criterion of its own (certification rule 3).

Honesty (rule 4): the ledger is a SERVED STATISTIC, not a learned-shape claim. A gain from any
ledger arm is served-statistic sufficiency; `shape_served` and `exact_counter` travel in the same
receipt so the learned-only attribution is one table away, never a separate narrative.

CPU only; no retraining; current artifacts. One job at a time inside the 32 GB slice.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_block_serving import _linear_access  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from measure_judge_law import TARGETS, _arms  # noqa: E402
from measure_stage_decomposition import _geometry, _load_stage_arrays, _round, _segment_ari  # noqa: E402
from safe_run import ensure_boundary  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_served_object.json"

# DEC-569 pre-registered arm set: the two block-PCA widths are the B3.5 pre-registered pair
# (39 = the counter's own width, 128 = the operator's equal-dimension idea). Evaluation-contract
# constants, frozen before this run; the card recomputes nothing they feed.
PCA_SIZES = (39, 128)


def _faceplate_slices(products: Path) -> dict:
    """The ledger's published part slices (the single source of truth for the parts).

    Returned as {part: [(lo, hi), ...]} in the faceplate's own order, so the parts are read from
    the artifact's stamp rather than retyped: a layout change re-slices this probe automatically.
    """
    import duckdb

    con = duckdb.connect(str(products / "cfm_products.duckdb"), read_only=True)
    stamp = con.execute("select stamp from view_faceplate limit 1").fetchone()
    con.close()
    if not stamp:
        raise SystemExit(
            "served object: no view_faceplate row — the ledger has no published layout"
        )
    slices = (json.loads(stamp[0]).get("ledger") or {}).get("slices")
    if not slices:
        raise SystemExit(
            "served object: faceplate carries no ledger slices — the parts are unaddressable"
        )
    return {p: [tuple(r) for r in runs] for p, runs in slices.items()}


def _tile_check(slices: dict, width: int) -> tuple[bool, str, np.ndarray | None]:
    """The G1 exact-recount property, read-only: the part runs must tile [0, width) with no gaps
    or overlaps. Returns (ok, note, perm) where perm is the permutation that reassembles the
    parts in the faceplate's publication order, so the reassembly can be checked numerically
    against the stored vector instead of trusted from the stamp."""
    seen = np.zeros(width, dtype=bool)
    for runs in slices.values():
        for lo, hi in runs:
            if lo < 0 or hi > width or hi <= lo:
                return False, f"run [{lo}, {hi}) outside [0, {width})", None
            if seen[lo:hi].any():
                return False, f"overlapping runs at [{lo}, {hi})", None
            seen[lo:hi] = True
    if not seen.all():
        return False, f"{int((~seen).sum())} column(s) not covered by any part", None
    perm = np.empty(width, dtype=int)
    idx = 0
    for runs in slices.values():
        for lo, hi in runs:
            perm[idx : idx + (hi - lo)] = np.arange(lo, hi)
            idx += hi - lo
    return True, "parts tile the published width exactly", perm


def _part_maps(ledger: dict, slices: dict, akeys, bkeys) -> dict[str, dict[str, np.ndarray]]:
    """Each named part as its own {key: vector} map, sliced from the published concatenation.

    The slice comes from the part's own runs in the full view — never from a global offset — so a
    reordered faceplate can never silently shift a part."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for part, runs in slices.items():
        part_cols: list[int] = []
        for lo, hi in runs:
            part_cols.extend(range(lo, hi))
        out[f"part_{part}"] = {
            k: np.asarray(ledger[side][k], float)[part_cols]
            for side in ("A", "B")
            for k in (akeys if side == "A" else bkeys)
            if k in ledger[side]
        }
    return out


def _pca_maps(ledger: dict, akeys, bkeys, size: int) -> dict[str, np.ndarray]:
    """Block PCA of the ledger fit on split A ONLY, applied to A and B (the B3.4 mechanism).

    One fit, two whole-matrix transforms: no per-item loop, and the A-fit/B-score split is the
    same frozen geometry every other arm uses."""
    ka = [k for k in akeys if k in ledger["A"]]
    kb = [k for k in bkeys if k in ledger["B"]]
    xa = np.vstack([np.asarray(ledger["A"][k], float) for k in ka])
    xb = np.vstack([np.asarray(ledger["B"][k], float) for k in kb])
    pca = PCA(n_components=size, random_state=0).fit(xa)
    pa, pb = pca.transform(xa), pca.transform(xb)
    out = {k: pa[i] for i, k in enumerate(ka)}
    out.update({k: pb[i] for i, k in enumerate(kb)})
    return out


def main() -> int:
    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="/home/austin_whaley/wonderland_work/cfm_v2k")
    ap.add_argument("--demo", default="/home/austin_whaley/wonderland_work/demo_v2")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.perf_counter()
    products = Path(a.products)
    arms = _arms(products / "cfm_products.duckdb", Path(a.demo))
    if arms["problems"]:
        raise SystemExit("served object: arms not aligned: " + "; ".join(arms["problems"][:3]))
    truth, akeys, bkeys = arms["truth"], arms["akeys"], arms["bkeys"]

    sample = next(iter(truth.values()))
    numeric = tuple(sorted(k for k, v in sample.items() if isinstance(v, (int, float))))
    future_fields = [f for f in TARGETS if f in numeric]
    persona_fields = [f for f in numeric if f not in set(TARGETS) | {"offer_opt"}]
    seg = {k: truth[k].get("segment") for k in truth if truth[k].get("segment") is not None}
    klab = len({seg[k] for k in bkeys if k in seg})

    ledger = arms["ledger"]
    if not (ledger["A"] and ledger["B"]):
        raise SystemExit(
            "served object: the measure view is not served — there is no ledger to X-ray"
        )
    width = int(next(iter(ledger["A"].values())).shape[0])

    slices = _faceplate_slices(products)
    tile_ok, tile_note, perm = _tile_check(slices, width)
    # numeric re-derivation check (G1 applied read-only): reassemble the parts in the faceplate's
    # publication order and demand equality with the stored vector on every scored row. One
    # vectorized comparison per side — the allclose-then-localize form keeps it loop-free.
    recount_ok, recount_note = True, "no check"
    if tile_ok and perm is not None:
        for side, keys in (("A", akeys), ("B", bkeys)):
            common = [k for k in keys if k in ledger[side]]
            if not common:
                continue
            m = np.vstack([np.asarray(ledger[side][k], float) for k in common])
            if not np.allclose(m[:, perm], m, rtol=1e-9, atol=1e-12):
                bad = int(np.argmax(~np.isclose(m[:, perm], m, rtol=1e-9, atol=1e-12).all(axis=1)))
                recount_ok = False
                recount_note = f"recount mismatch on {side} customer {common[bad]!r}"
                break

    stages = _load_stage_arrays(products)
    counter = stages["M0_exact_counter"]
    shape = stages["M4_served"]

    mats: dict[str, dict[str, np.ndarray]] = {"ledger_full": {**ledger["A"], **ledger["B"]}}
    mats.update(_part_maps(ledger, slices, akeys, bkeys))
    for size in PCA_SIZES:
        mats[f"ledger_pca_{size}"] = _pca_maps(ledger, akeys, bkeys, size)
    mats["exact_counter"] = counter
    mats["shape_served"] = shape
    mats["path_a"] = arms["table"]
    ledger_arm_names = (
        ["ledger_full"]
        + [n for n in mats if n.startswith("part_")]
        + [f"ledger_pca_{s}" for s in PCA_SIZES]
    )

    table: dict = {}
    for name, mat in mats.items():
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

    access: dict = {}
    access_arms = [n for n in ledger_arm_names] + ["exact_counter", "shape_served", "path_a"]
    for name in access_arms:
        mean_r2, n_fields = _linear_access(
            mats[name], truth, akeys, bkeys, persona_fields + future_fields
        )
        access[name] = {"mean_r2": _round(mean_r2), "n_fields": int(n_fields)}

    complete = bool(
        tile_ok
        and recount_ok
        and all(
            table[n].get("persona_geometry") is not None
            for n in ledger_arm_names + ["exact_counter", "shape_served", "path_a"]
        )
        and access["ledger_full"]["mean_r2"] is not None
        and access["path_a"]["mean_r2"] is not None
    )
    body = {
        "kind": "served_object",
        "complete": complete,
        "products": products_identity(products),
        "n_fit_A": len(akeys),
        "n_score_B": len(bkeys),
        "ledger_arms": ledger_arm_names,
        "reference_arms": ["exact_counter", "shape_served", "path_a"],
        "arms": table,
        "linear_access": {
            "complete": bool(
                access["ledger_full"]["mean_r2"] is not None
                and access["path_a"]["mean_r2"] is not None
            ),
            "protocol": "frozen B3.3 consumer: standardized RidgeCV fit on A, mean R2 on B over the "
            "lab's persona+future numeric fields; no B fit, no metric tuning",
            "arms": access,
        },
        "validation": {
            "parts_tiles_published_width": tile_ok,
            "tile_note": tile_note,
            "parts_recount_matches_stored_view": recount_ok,
            "recount_note": recount_note,
            "part_widths": {p: sum(hi - lo for lo, hi in runs) for p, runs in slices.items()},
            "ledger_width": width,
            "persona_fields": persona_fields,
            "future_fields": future_fields,
            "segment_k": int(klab),
        },
        "honesty": "the ledger and its parts are SERVED STATISTICS, not a learned-shape claim "
        "(certification rule 4): a gain from any ledger arm is served-statistic sufficiency; "
        "shape_served and exact_counter are in this same receipt as the learned-only attribution",
        "seconds": 0.0,
    }
    body["seconds"] = round(time.perf_counter() - t0, 1)
    stamp_merge_write(
        Path(a.out),
        body,
        " ".join([Path(__file__).name] + sys.argv[1:]),
        stamp_blocks=False,
    )
    print(f"served object -> {a.out} ({body['seconds']}s)")
    for name in ledger_arm_names + body["reference_arms"]:
        r = table[name]
        print(
            f"  {name:22s} dim={r['dim']:3d} persona={r['persona_geometry']} "
            f"future={r['future_geometry']} segment_ari={r['segment_ari']} "
            f"access_r2={access[name]['mean_r2']}"
        )
    print(f"  validation: tile={tile_ok} recount={recount_ok} ({recount_note})")
    print(f"  complete={complete}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
