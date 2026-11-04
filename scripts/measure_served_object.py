#!/usr/bin/env python3
"""B3.6 served-object X-ray: does the ledger (B's companion statistic) carry fine structure?

DEC-567's L1 layer said the judge's delivery column is a naive 256+595 concat that never read the
595-d ledger alone; DEC-569 pre-registered this producer before the run. It scores, under the
FROZEN B3.2/B3.4 consumers (imported, not redefined), the ledger full, each of its named parts,
and the ledger block-PCA (fit on split A only) at the two pre-registered widths, against the three
references the B rows already carry: the exact counter (39), the served shape (256), and PATH A
(the wide table encoded once over A union B by the judge's own matrix path — the fair additive
family). The parts are read from their PUBLISHED rows (the storage contract: one block per part,
in view_layout's PART_ORDER) and cross-checked byte-for-byte against the judge's flat ledger arm
plus the faceplate's declared widths (DEC-571: attempt 1 sliced the flat view at the faceplate's
NATIVE interleaved offsets — two different layouts — and the pre-registered G1 read-only recount
refused the run for exactly that; the arm set, consumers and branches are unchanged by the fix).
The pre-registered branch (>= 0.30 exposure / .15-.30 partial / < .15 insufficient) is
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

from looking_glass import view_layout as V  # noqa: E402
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


def _faceplate_ledger(products: Path) -> tuple[dict, dict, int]:
    """The ledger's published stamp: (native slices, declared part widths, declared view width).

    Read from the artifact's own stamp, never retyped: a layout change re-slices this probe
    automatically, and a stored row that disagrees with its declared width is a lying artifact
    this run refuses. The native slices are the ledger's INTERLEAVED per-field layout — they tile
    the native width, they do NOT address the stored part blocks (see `_part_rows`).
    """
    import duckdb

    con = duckdb.connect(str(products / "cfm_products.duckdb"), read_only=True)
    stamp = con.execute("select stamp from view_faceplate limit 1").fetchone()
    con.close()
    if not stamp:
        raise SystemExit(
            "served object: no view_faceplate row — the ledger has no published layout"
        )
    stamp = json.loads(stamp[0])
    slices = (stamp.get("ledger") or {}).get("slices")
    if not slices:
        raise SystemExit(
            "served object: faceplate carries no ledger slices — the parts are unaddressable"
        )
    part_widths = ((stamp.get("parts") or {}).get("measure")) or {}
    view_width = int((stamp.get("views") or {}).get("measure", 0))
    return (
        {p: [tuple(r) for r in runs] for p, runs in slices.items()},
        {p: int(d) for p, d in part_widths.items()},
        view_width,
    )


def _tile_check(slices: dict, width: int) -> tuple[bool, str]:
    """Stamp self-consistency, read-only: the native part runs must tile [0, width) with no gaps
    or overlaps. (This validates the STAMP against itself; the check against the STORED bytes is
    `_parts_match_flat`, which the stamp's interleaved layout cannot answer.)"""
    seen = np.zeros(width, dtype=bool)
    for runs in slices.values():
        for lo, hi in runs:
            if lo < 0 or hi > width or hi <= lo:
                return False, f"run [{lo}, {hi}) outside [0, {width})"
            if seen[lo:hi].any():
                return False, f"overlapping runs at [{lo}, {hi})"
            seen[lo:hi] = True
    if not seen.all():
        return False, f"{int((~seen).sum())} column(s) not covered by any part"
    return True, "the native slices tile the published ledger width exactly"


def _part_rows(db: Path) -> dict[str, dict[str, np.ndarray]]:
    """Each named part as its published {key: vector} map, read from its own rows.

    The storage contract (`view_layout`, DEC-388 O1): a view is the DOCUMENTED concatenation of
    its parts, in `PART_ORDER` — so the outlet publishes one block per part, and that block IS
    the part. The faceplate's `ledger.slices` describe the ledger's NATIVE layout (the
    interleaved per-field object the builder recounts), which is not the storage concatenation
    order: slicing the flat view at the native offsets yields jumbled columns across parts, and
    the pre-registered G1 read-only recount (DEC-569) is exactly the check that refuses that.
    (DEC-571: attempt 1 failed this check for that reason; the parts are therefore read from
    their published rows and cross-checked byte-for-byte against the judge's flat arm below.)"""
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute(
        "select customer_key, part, embedding from view_embeddings where view = 'measure'"
    ).fetchall()
    con.close()
    out: dict[str, dict[str, np.ndarray]] = {}
    for k, part, vec in rows:
        out.setdefault(part, {})[k] = np.asarray(vec, float)
    return out


def _parts_match_flat(part_rows: dict, order: list[str], ledger: dict) -> tuple[bool, str]:
    """The published part blocks concatenated in the contract order must be byte-identical to
    the judge's flat ledger arm on every key it carries: O0 has to score the same object the
    judge scores, and a drifting reader is exactly the defect this repo has already paid for
    twice. One comparison per side over the stacked rows (no per-item loop)."""
    widths = {p: next(iter(rows.values())).shape[0] for p, rows in part_rows.items()}
    missing = [p for p in order if p not in part_rows]
    if missing:
        return False, f"parts not published: {missing}"
    for side in ("A", "B"):
        keys = [k for k in ledger[side] if all(p in part_rows and k in part_rows[p] for p in order)]
        if not keys:
            return False, f"no {side} key carries all parts"
        flat = np.vstack([np.asarray(ledger[side][k], float) for k in keys])
        rebuilt = np.concatenate(
            [np.vstack([part_rows[p][k] for k in keys]) for p in order], axis=1
        )
        if flat.shape[1] != rebuilt.shape[1]:
            return False, f"{side} width mismatch: flat {flat.shape[1]} vs parts {rebuilt.shape[1]}"
        if not np.array_equal(flat, rebuilt):
            bad = int(np.argmax((flat != rebuilt).any(axis=1)))
            return (
                False,
                f"part blocks disagree with the flat ledger on {side} customer {keys[bad]!r}",
            )
    return (
        True,
        f"part blocks ({' + '.join(str(widths[p]) for p in order)} = {sum(widths.values())}) "
        "byte-identical to the flat ledger arm on every scored A and B row",
    )


def _widths_match(
    part_rows: dict, declared: dict, view_width: int, ledger: dict
) -> tuple[bool, str]:
    """Stored part widths against the faceplate's declared widths (stamp vs bytes; the
    outlet_schema dimension guard, applied read-only to every part at once)."""
    bad = []
    for p, rows in part_rows.items():
        w = next(iter(rows.values())).shape[0]
        if w != declared.get(p, -1):
            bad.append(f"{p}: stored {w} != declared {declared.get(p)}")
    flat_w = next(iter(ledger["A"].values())).shape[0]
    if flat_w != view_width:
        bad.append(f"flat ledger: stored {flat_w} != declared {view_width}")
    return (not bad), "; ".join(bad) if bad else "every stored part width matches the faceplate"


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
    ap.add_argument("--products", default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/cfm_v2k")
    ap.add_argument("--demo", default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/demo_v2")
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

    slices, declared_widths, view_width = _faceplate_ledger(products)
    tile_ok, tile_note = _tile_check(slices, view_width or width)

    # The parts, read from their PUBLISHED rows (the storage contract: one block per part, in
    # PART_ORDER), then two read-only guardrails against the stored bytes: every part width
    # matches the faceplate, and the part blocks concatenated in the contract order are
    # byte-identical to the judge's flat ledger arm (O0 must score the same object the judge
    # scores — DEC-571, attempt 1's perm-recount refused the run for conflating the native and
    # stored layouts, and the parts were jumbled; this reader is the fix).
    part_rows = _part_rows(products / "cfm_products.duckdb")
    order = list(V.PART_ORDER.get("measure", []))
    widths_ok, widths_note = _widths_match(part_rows, declared_widths, view_width, ledger)
    match_ok, match_note = _parts_match_flat(part_rows, order, ledger)

    stages = _load_stage_arrays(products)
    counter = stages["M0_exact_counter"]
    shape = stages["M4_served"]

    mats: dict[str, dict[str, np.ndarray]] = {"ledger_full": {**ledger["A"], **ledger["B"]}}
    for part in order:
        if part in part_rows:
            mats[f"part_{part}"] = part_rows[part]
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
        and widths_ok
        and match_ok
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
            "native_slices_tile_published_width": tile_ok,
            "tile_note": tile_note,
            "stored_part_widths_match_faceplate": widths_ok,
            "widths_note": widths_note,
            "part_blocks_byte_identical_to_flat_ledger": match_ok,
            "match_note": match_note,
            "native_part_widths": {
                p: sum(hi - lo for lo, hi in runs) for p, runs in slices.items()
            },
            "stored_part_widths": {
                p: next(iter(r.values())).shape[0] for p, r in part_rows.items()
            },
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
    print(f"  validation: tile={tile_ok} widths={widths_ok} match={match_ok} ({match_note})")
    print(f"  complete={complete}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
