#!/usr/bin/env python3
"""S4 — the outlets are certified by evidence, not by having names (DEC-388, DEC-397, DEC-412).

    python3 scripts/measure_views_law.py [--products DIR] [--demo DIR] [--out PATH]

SPINE §4 makes S4 the stage that turns a state into *usable* views: "the encoder must
pre-adapt to consumers it will never meet". Its working-state proof is
**per-view accessibility parity + per-view negative controls**, and the failure it exists to
catch is "present but unreadable sold as sufficiency". So this script asks, of every
published outlet separately:

1. **Does it decode anything at all?** A linear head (the only shipping supervised head,
   DEC-407) fitted on split A must predict at least one generative truth target out of
   sample, beating its destroyed and random controls with a paired CI excluding zero.
2. **Is it ACCESSIBLE, not merely present?** The diagnostic MLP (DEC-407: diagnoses, never
   ships) is fitted on the same rows and the same splits. If the MLP beats the linear head
   beyond the measured noise floor, that is an **accessibility gap charged to the encoder**
   — never permission to ship the MLP.
3. **Does it fail when it should?** Random and destroyed (column-permuted) controls per
   view: a control scoring above its real view means the "signal" was an artifact of the
   head, not of the outlet.

Targets are the lab's **generative latent traits** (`reference_truth.json`), which the
encoder never saw and which no consumer could derive from the published columns — the
anti-circularity requirement of certification-integrity rule 4.

The verdict is computed. Nothing here can be talked into a PASS by prose: an empty outlet,
a view missing from the table, or a faceplate whose dims disagree with the stored arrays is
UNKNOWN/FAIL, never a pass by absence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_battery_law import _load_all, _paired_ci, _ridge_oos  # noqa: E402  (one definition)
from measure_input_law import _rollup, stamp_merge_write  # noqa: E402

from looking_glass import view_layout as V  # noqa: E402

SEED = 65537  # principle-ok: evaluation-contract seed, frozen with the gate (DEC-410)

#: The generative traits the lab drew for each customer. Numeric, so a linear head can be
#: scored with R^2 and a paired per-customer squared-error delta. `regime_switches` /
#: `final_regime_days` are DERIVED from the lab's `regime_runs` trajectory (a list of
#: [regime, days] pairs, not a scalar) — the derivation is deterministic, recorded here, and
#: the encoder never saw it.
TARGETS = (
    "x0",
    "h0",
    "price_sensitivity",
    "return_proneness",
    "churn_day",
    "regime_switches",
    "final_regime_days",
)


def _target(truth: dict, keys: list[str], name: str):
    """One target as a finite float vector, or None when it is not evaluable on these rows.

    Non-evaluable is EXCLUDED AND REPORTED (AGENTS.md: never force a degenerate row to PASS
    or FAIL): a missing key, a non-scalar field, a constant, or a non-finite value all
    return None and the caller records why.
    """
    try:
        if name == "regime_switches":
            vals = [float(len(truth[k]["regime_runs"])) for k in keys]
        elif name == "final_regime_days":
            vals = [float(truth[k]["regime_runs"][-1][1]) for k in keys]
        else:
            vals = [float(truth[k][name]) for k in keys]
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or float(np.std(arr)) < 1e-12:
        return None
    return arr


def _target_matrix(truth: dict, akeys, bkeys):
    """{target: (yA, yB)} over the targets that are genuinely evaluable, plus exclusions."""
    usable, excluded = {}, []
    for name in TARGETS:
        ya, yb = _target(truth, akeys, name), _target(truth, bkeys, name)
        if ya is None or yb is None:
            excluded.append(name)
        else:
            usable[name] = (ya, yb)
    return usable, excluded


#: principle-ok: evaluation-contract tolerance. A parity gap must clear this multiple of the
#: MEASURED paired SE before it is charged as an accessibility gap — the floor is computed
#: from the bootstrap, this multiplier only says how many of them count (DEC-388).
PARITY_SIGMAS = 3.0


def _load_views(db: Path) -> tuple[dict, dict]:
    """{view: {customer_key: vector}} plus the faceplate stamp the artifact carries."""
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    views: dict = {}
    for key, view, part, vec in con.execute(
        "select customer_key, view, part, embedding from view_embeddings"
    ).fetchall():
        views.setdefault(view, {}).setdefault(key, {})[part] = np.asarray(vec, dtype=np.float64)
    # A view is the documented concat of its parts (view_layout owns that order); rebuild
    # it from the contract rather than from whatever order the scan happened to return —
    # alphabetically sorting the parts would silently build a DIFFERENT vector than the
    # producer published, and every score below would be measuring that instead.
    flat = {}
    for view, by_key in views.items():
        published = [
            p for p in V.PART_ORDER.get(view, []) if p in {q for r in by_key.values() for q in r}
        ]
        extra = sorted({p for row in by_key.values() for p in row} - set(published))
        parts = published + extra  # an uncontracted part still counts, and is reported
        flat[view] = {
            k: np.concatenate([row[p] for p in parts]) if all(p in row for p in parts) else None
            for k, row in by_key.items()
        }
        if extra:
            print(f"[S4] {view}: parts not in the published contract: {extra}")
    stamp = con.execute("select stamp from view_faceplate limit 1").fetchone()
    con.close()
    return flat, (json.loads(stamp[0]) if stamp else {})


def _matrix(view_map: dict, keys: list[str]):
    rows = [view_map.get(k) for k in keys]
    if any(r is None for r in rows) or not rows:
        return None
    if len({r.shape[0] for r in rows}) != 1:
        # Ragged: the producer contradicted its own stamp. `outlet_schema` names it and
        # FAILS the stage; here it is simply not measurable, so return None (UNKNOWN).
        return None
    return np.vstack(rows)


def _mlp_oos(Ea, ya, Eb, yb, seed: int = SEED):
    """The DIAGNOSTIC tier (DEC-407): a small MLP on the same rows/splits, used only to
    distinguish "information absent" from "information present but not linearly readable"."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(Ea) < 8 or float(np.std(ya)) < 1e-12 or float(np.std(yb)) < 1e-12:
        return None, None
    # principle-ok: capacity DERIVED from the outlet's own width (no typed layer size); the
    # diagnostic never ships, so it is capped for runtime, not for performance.
    width = int(min(64, max(8, 2 * Ea.shape[1])))
    m = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(width,), max_iter=400, random_state=seed, early_stopping=True
        ),
    )
    m.fit(Ea, ya)
    p = m.predict(Eb)
    ss = float(np.sum((yb - yb.mean()) ** 2))
    return p, float(1.0 - np.sum((yb - p) ** 2) / ss)


def _controls(rng: np.random.Generator, Eb):
    """Random (fresh gaussian) and destroyed (column-permuted: same marginals, no customer)
    embeddings for one view. H4 demands both, and the destruction must keep the geometry the
    head could otherwise exploit."""
    rand = rng.normal(loc=Eb.mean(), scale=Eb.std() + 1e-12, size=Eb.shape)
    destroyed = np.column_stack([rng.permutation(Eb[:, j]) for j in range(Eb.shape[1])])
    return rand, destroyed


def check_outlet_schema(views: dict, faceplate: dict, keys: list[str]) -> dict:
    """The socket exists and the faceplate tells the truth about it (O1 + O4).

    A published column whose width disagrees with the self-description is worse than a
    missing one: consumers validated against the stamp and then read something else."""
    t0 = time.perf_counter()
    missing = [v for v in V.FULL_ORDER if v not in views or not views[v]]
    problems = []
    for view, by_key in views.items():
        got = {int(vec.shape[0]) for vec in by_key.values() if vec is not None}
        declared = (faceplate.get("views") or {}).get(view)
        if len(got) != 1:
            problems.append(f"{view}: ragged widths {sorted(got)}")
        elif declared is not None and got != {declared}:
            problems.append(f"{view}: stored {next(iter(got))} != faceplate {declared}")
        if any(vec is None for vec in by_key.values()):
            problems.append(f"{view}: some customers are missing a part")
    n_rows = sum(len(v) for v in views.values())
    ev = (
        f"{len(views)} views / {n_rows} rows; declared-not-served "
        f"{faceplate.get('declared_not_emitted', [])} advertised"
    )
    return {
        "working": not missing and not problems,
        "evidence": (ev + (" | MISSING " + ",".join(missing) if missing else ""))
        + ((" | " + "; ".join(problems)) if problems else ""),
        "views": sorted(views),
        "missing": missing,
        "seconds": round(time.perf_counter() - t0, 3),
    }


def _evaluate(views: dict, truth: dict, akeys, bkeys):
    """Fit every head ONCE per (view, target) and return the records both gates read.

    Both S4 gates consume the same fits: computing them separately would fit the ridge twice
    and could report two different numbers for one pair (single source of truth).
    """
    usable, excluded = _target_matrix(truth, akeys, bkeys)
    rng = np.random.default_rng(SEED)
    recs, unreadable = [], []
    for view, by_key in views.items():
        Ea, Eb = _matrix(by_key, akeys), _matrix(by_key, bkeys)
        if Ea is None or Eb is None:
            unreadable.append(view)
            continue
        rand, destroyed = _controls(rng, Eb)
        for name, (ya, yb) in usable.items():
            p_lin, r2_lin = _ridge_oos(Ea, ya, Eb, yb)
            # The controls go through the SAME head as the real view — a control beaten by a
            # weaker head proves nothing, and this is where a fake "signal" would hide.
            p_rand, _ = _ridge_oos(Ea, ya, rand, yb)
            p_dest, _ = _ridge_oos(Ea, ya, destroyed, yb)
            if p_lin is None or p_rand is None or p_dest is None:
                continue
            _, lo_r, _ = _paired_ci((yb - p_rand) ** 2 - (yb - p_lin) ** 2)
            _, lo_d, _ = _paired_ci((yb - p_dest) ** 2 - (yb - p_lin) ** 2)
            p_mlp, r2_mlp = _mlp_oos(Ea, ya, Eb, yb)
            gap = None
            if p_mlp is not None:
                # per-customer squared-error advantage of the MLP over the linear head
                mean, lo, hi = _paired_ci((yb - p_lin) ** 2 - (yb - p_mlp) ** 2)
                if lo > 0 and mean > PARITY_SIGMAS * (hi - lo) / 2.0:
                    gap = f"{view}/{name}: MLP R2={r2_mlp:.3f} > linear R2={r2_lin:.3f}"
            recs.append(
                {
                    "view": view,
                    "target": name,
                    "r2_linear": float(r2_lin),
                    "r2_mlp": None if r2_mlp is None else float(r2_mlp),
                    "beats_controls": bool(lo_r > 0 and lo_d > 0 and r2_lin > 0),
                    "ci_lo_vs_random": float(lo_r),
                    "ci_lo_vs_destroyed": float(lo_d),
                    "gap": gap,
                }
            )
    return recs, unreadable, excluded


def check_decodability(recs: list, unreadable: list, excluded: list, n_views: int) -> dict:
    """Per-view: a linear head must decode at least one generative trait above BOTH controls,
    with the paired CI excluding zero. No PASS from a point estimate (certification rule 5)."""
    decoded = {}
    for r in recs:
        if r["beats_controls"] and r["r2_linear"] > decoded.get(r["view"], {}).get("r2", -np.inf):
            decoded[r["view"]] = {"r2": round(r["r2_linear"], 4), "target": r["target"]}
    dead = sorted(set(r["view"] for r in recs) - set(decoded)) + [
        f"{v}(unreadable)" for v in unreadable
    ]
    # Which trait each view passed on, printed: three views "passing" on the same easy target
    # is a different claim from three views each decoding something of their own, and the
    # card must not be able to blur them.
    carried = dict(Counter(v["target"] for v in decoded.values()))
    return {
        "working": n_views > 0 and not dead,
        "evidence": f"linear-head decoders beat random+destroyed (paired CI lo>0) for "
        f"{len(decoded)}/{n_views} views; undecodable: {dead or 'none'}; "
        f"carried by {carried}"
        + (f" | targets excluded (non-scalar/constant/non-finite): {excluded}" if excluded else ""),
        "best_by_view": decoded,
        "undecodable": dead,
        "targets_excluded": excluded,
    }


def check_accessibility_parity(recs: list, excluded: list) -> dict:
    """DEC-388's guarantee, per outlet: the MLP beating the linear head beyond the measured
    noise floor is an ACCESSIBILITY GAP charged to the encoder (the trunk failed to make it
    readable), never a licence to ship the MLP (DEC-407)."""
    gaps = sorted(r["gap"] for r in recs if r["gap"])
    return {
        "working": bool(recs) and not gaps,
        "evidence": (
            f"{len(recs)} view x target pairs at parity within {PARITY_SIGMAS:g}x the paired "
            f"SE; accessibility gaps: {gaps or 'none'}"
            if recs
            else "no evaluable view x target pair — UNKNOWN is the honest reading"
        )
        + (f" | targets excluded: {excluded}" if excluded else ""),
        "pairs": len(recs),
        "gaps": gaps,
    }


def _products_db(products: str | Path) -> Path:
    """Resolve `--products` to the products DB, accepting a directory OR a file path.

    The sibling receipt scripts disagree on this (`measure_battery_law.py` takes the .duckdb
    path, this script originally took only a directory); accepting both means one more script
    never dead-ends on a path shape.
    """
    p = Path(products)
    return p if p.is_file() else p / "cfm_products.duckdb"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--products",
        default=str(ROOT / "looking_glass" / "artifacts" / "cfm"),
        help="dir holding cfm_products.duckdb, or its path",
    )
    ap.add_argument("--demo", default=str(ROOT / "rabbit_hole" / "data"))
    ap.add_argument("--out", default=str(ROOT / "docs/report_cards/experiments/judge_stages.json"))
    a = ap.parse_args()

    db = _products_db(a.products)
    if not db.exists():
        print(
            f"S4: no products DB at {db} — produce it with\n"
            f"  python -m looking_glass.customer_foundation_model products "
            f"--out-dir {a.products}",
            file=sys.stderr,
        )
        return 2
    demo = Path(a.demo)
    _reg, truth, _emb, _raw, _stream, bkeys, akeys, _as_of = _load_all(str(db), demo)
    views, faceplate = _load_views(db)
    if not views:
        print(
            "S4: view_embeddings is EMPTY — the outlets were never published "
            "(DEC-424). This is a gap, not a pass.",
            file=sys.stderr,
        )

    recs, unreadable, excluded = _evaluate(views, truth, akeys, bkeys)
    checks = {
        "outlet_schema": check_outlet_schema(views, faceplate, akeys + bkeys),
        "decodability": check_decodability(recs, unreadable, excluded, len(views)),
        "accessibility_parity": check_accessibility_parity(recs, excluded),
    }
    stamp_merge_write(
        Path(a.out),
        {"S4": {**_rollup(checks), "pairs": recs}},
        command=" ".join([Path(__file__).name] + sys.argv[1:]),
    )
    for name, c in checks.items():
        print(f"[{'OK' if c['working'] else 'FAIL'}] {name}: {c['evidence']}")
    print(f"S4 -> {a.out}")
    return 0 if all(c["working"] for c in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
