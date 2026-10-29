#!/usr/bin/env python3
"""The JUDGE card's top line: PATH B vs PATH A, reported twice (DEC-390, DEC-383, DEC-428).

    python3 scripts/measure_judge_law.py [--products DIR] [--demo DIR] [--legs supervised]

The project's definition of progress is one question asked per leg —

    PATH B (event stream -> frozen encoder -> simple head)  >  PATH A (wide table -> GBM)

— and answered TWICE, because the ledger is a companion and not a win: `brain` is PROGRESS and
`brain + ledger` is DELIVERY. Until DEC-427 built the `measure` view the second column had
nothing to deliver; until this script existed neither column had a receipt, so all eight top-line
rows rendered UNKNOWN (never PASS by absence).

What makes the comparison worth the paper:

* **Symmetric geometry.** Both arms fit on `encoder_samples` split A and score split B. No
  cross-validation crossing the population boundary, no OOF on one side and holdout on the other.
* **Row-exact alignment by key, not by position.** The brain arm and the table arm are indexed by
  `customer_key`, so a customer missing from the wide table is missing from BOTH arms instead of
  shifting one matrix by a row — which would look like a model difference and be a bug.
* **One incumbent for every leg** (DEC-428), cited rather than reinvented, because AGENTS.md fixes
  "the table baseline" while nothing in the repo pinned the model.
* **The table is encoded once over A ∪ B.** One-hot encoding observes values, so encoding the two
  splits separately can yield two different column sets and silently mis-align the arms.
* **Controls through the brain's own head** (a control beaten by a weaker head proves nothing), and
  degenerate traits excluded-and-reported rather than dropped.

This script MEASURES. It never writes a verdict, a `criterion`, or a `rule`: the card computes
those from the receipt (certification rule 3), so the verdict cannot be improved by narration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_battery_law import SEED, _load_all, _paired_ci, _ridge_oos  # noqa: E402  (one definition)
from measure_input_law import _rollup, stamp_merge_write  # noqa: E402
from measure_views_law import (  # noqa: E402  (one definition)
    _controls,
    _load_views,
    _matrix,
    _products_db,
    _target_matrix,
)

#: THE canonical PATH A incumbent for every leg (DEC-428). Chosen by citation from the configs
#: already in the repo (`measure_causal.py`) rather than invented here, and pinned in one place so
#: the fixed baseline stays fixed. CPU by policy: the GPU is a contested shared resource and
#: adopting a GPU path without a measured benchmark is what "measure before adopting" forbids.
INCUMBENT = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.06,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": SEED,
}

COMPARATOR = "PATH A (GBM incumbent)"
COLUMNS = ("brain", "brain_ledger")


def _r2(y, p) -> float:
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def _empty_columns(note: str) -> dict:
    return {
        c: ({"tasks": [], "note": note} if c == "brain_ledger" else {"tasks": []}) for c in COLUMNS
    }


def _table_matrix(demo: Path, keys: list[str]):
    """PATH A's wide table as a matrix, encoded ONCE over A ∪ B, returned keyed by customer.

    Reuses the repo's canonical `table_features` rather than adding a fifth inline copy of the
    drop-and-dummy rule; encoding the two splits separately is the trap this exists to avoid
    (one-hot encoding observes values, so the two splits can come back with different columns).
    """
    import polars as pl

    from demo.auto_encoder import table_features

    path = demo / "path_a" / "path_a_wide_table.parquet"
    if not path.exists():
        return None, {"error": f"no wide table at {path}"}
    A = pl.read_parquet(path)
    have = set(A["customer_key"].to_list())
    kept = [k for k in keys if k in have]
    if not kept:
        return None, {"error": "no judged customer appears in the PATH A wide table"}
    X, _labels, _seg = table_features(A, kept)
    return {k: np.asarray(row, dtype=np.float64) for k, row in zip(kept, X, strict=True)}, {
        "source": str(path),
        "columns": int(np.asarray(X).shape[1]),
        "n_requested": len(keys),
        "n_present": len(kept),
    }


def _rows(table: dict, keys: list[str]) -> np.ndarray:
    return np.vstack([table[k] for k in keys])


def _gbm_oos(Xa, ya, Xb) -> np.ndarray:
    """The incumbent, fit on exactly the rows the brain's head was fit on, scored on B."""
    from xgboost import XGBRegressor

    model = XGBRegressor(**INCUMBENT)
    model.fit(Xa, ya)
    return np.asarray(model.predict(Xb), dtype=np.float64)


def _tasks_for(arm_a, arm_b, table_a, table_b, usable, rng) -> list[dict]:
    """Score one column's tasks: PATH B's linear head on `arm_*` vs the incumbent on `table_*`."""
    tasks = []
    for name, (ya, yb) in usable.items():
        p_brain, r2_brain = _ridge_oos(arm_a, ya, arm_b, yb)
        if p_brain is None:
            tasks.append(
                {
                    "name": name,
                    "degenerate": True,
                    "degenerate_reason": "too few rows, or the target has no variance to fit",
                }
            )
            continue
        rand, destroyed = _controls(rng, arm_b)
        _, r2_rand = _ridge_oos(arm_a, ya, rand, yb)
        _, r2_dest = _ridge_oos(arm_a, ya, destroyed, yb)
        p_tab = _gbm_oos(table_a, ya, table_b)
        # paired per-customer squared-error advantage of the brain over the incumbent:
        # a positive CI means the brain is better ON THE SAME customers, not on average-of-averages
        _mean, lo, hi = _paired_ci((yb - p_tab) ** 2 - (yb - p_brain) ** 2)
        # A negative control is only evidence if a reader can read it. SSE > 2x SST (R2 < -1)
        # means the control's predictions are wilder than doing nothing at all by more than the
        # target's own variance, which on a wide design (the 591-d ledger) is a conditioning
        # artifact of rare indicator columns, not a demonstration that the real arm has signal.
        # Reported as a flag beside the number: the card's H4 gate ("control <= brain") passes
        # either way, and a gate that passes because the control was absurd is not a gate.
        unreadable = [
            ctl
            for ctl, r2 in (("random", r2_rand), ("destroyed", r2_dest))
            if r2 is not None and r2 < -1.0
        ]
        tasks.append(
            {
                "name": name,
                "metric": "held-out R2 on split B",
                "brain": round(float(r2_brain), 6),
                "pathA": round(float(_r2(yb, p_tab)), 6),
                "ci_lo": round(float(lo), 6),
                "ci_hi": round(float(hi), 6),
                "n": int(len(yb)),
                "random": None if r2_rand is None else round(float(r2_rand), 6),
                "destroyed": None if r2_dest is None else round(float(r2_dest), 6),
                "comparator": COMPARATOR,
                "controls_unreadable": unreadable,
            }
        )
    return tasks


def leg_supervised(db: Path, demo: Path) -> dict:
    """The supervised leg: the frozen donor + a linear head vs the wide table + the GBM incumbent."""
    _reg, truth, emb, _raw, _stream, bkeys, akeys, _as_of = _load_all(str(db), demo)
    usable, excluded = _target_matrix(truth, akeys, bkeys)
    if not usable:
        return {
            "columns": _empty_columns("no usable generative trait on these splits"),
            "excluded": excluded,
        }
    brain_a, brain_b = _matrix(emb, akeys), _matrix(emb, bkeys)
    if brain_a is None or brain_b is None:
        return {
            "columns": _empty_columns(
                "state_embeddings are ragged or missing judged customers (see S4 outlet_schema)"
            ),
            "excluded": excluded,
        }
    table, tmeta = _table_matrix(demo, list(akeys) + list(bkeys))
    if table is None:
        return {
            "columns": _empty_columns(
                f"PATH A is unavailable ({tmeta.get('error')}) — a leg with no baseline is UNKNOWN, never a pass"
            ),
            "excluded": excluded,
        }
    try:
        table_a, table_b = _rows(table, akeys), _rows(table, bkeys)
    except KeyError as exc:
        return {
            "columns": _empty_columns(
                f"a judged customer is absent from the wide table ({exc}) — dropping them from one arm "
                f"only would compare two different populations, so the leg refuses instead"
            ),
            "excluded": excluded,
        }

    views, _stamp = _load_views(db)
    led_a, led_b = (
        _matrix(views.get("measure", {}), akeys),
        _matrix(views.get("measure", {}), bkeys),
    )
    rng = np.random.default_rng(SEED)
    columns = {"brain": {"tasks": _tasks_for(brain_a, brain_b, table_a, table_b, usable, rng)}}
    diagnostics: dict = {}
    if led_a is not None and led_b is not None:
        # DELIVERY: the brain plus the mechanical ledger, same head, same rows, same splits
        columns["brain_ledger"] = {
            "tasks": _tasks_for(
                np.hstack([brain_a, led_a]),
                np.hstack([brain_b, led_b]),
                table_a,
                table_b,
                usable,
                rng,
            )
        }
    else:
        columns["brain_ledger"] = {
            "tasks": [],
            "note": "the measure view is not served (DEC-427) — there is no ledger to deliver",
        }
    if led_a is not None and led_b is not None:
        # DIAGNOSTIC (measured, never gated — the card's two columns are the court): SPINE §5's
        # corollary says the ledger is "mechanically what PATH A's table already is", so a
        # ledger-backed number may not be booked as a brain win. That PREMISE is testable, and on
        # the first build it did not hold: the ledger scored 0.98 on `h0` where the 48-column
        # table scored -0.12. Either the ledger is a strictly richer object than the current
        # table, or PATH A's baseline is under-built — and a brain that beats a straw man is
        # worth nothing. Surfaced here for an operator ruling; this producer does not get to
        # widen the baseline, because what the comparator contains is a frozen definition.
        diagnostics["ledger_only"] = {
            "tasks": _tasks_for(led_a, led_b, table_a, table_b, usable, rng),
            "purpose": "tests SPINE §5's premise that the ledger is what PATH A's table already is",
        }
    return {
        "columns": columns,
        "diagnostics": diagnostics,
        "excluded": excluded,
        "splits": {
            "n_fit_A": len(akeys),
            "n_score_B": len(bkeys),
            "source": "encoder_samples (the frozen population split)",
        },
        "path_a": tmeta,
        "incumbent": {"library": "xgboost", "model": "XGBRegressor", **INCUMBENT},
        "arms": {
            "brain": "state_embeddings donor + linear ridge head (DEC-407: linear is the shipping head)",
            "brain_ledger": "brain (state_embeddings) + measure view, same linear head and rows",
        },
    }


LEGS = {"supervised": leg_supervised}


def _evidence(leg: str, body: dict) -> tuple[bool, str]:
    """The producer's own working-state is about the MEASUREMENT being complete — the leg's
    PASS/FAIL belongs to the card, never to this file (certification rule 3)."""
    tasks = body["columns"]["brain"]["tasks"]
    live = [t for t in tasks if not t.get("degenerate")]
    complete = bool(live) and all(
        t.get("random") is not None
        and t.get("destroyed") is not None
        and math_ok(t)
        and t["random"] <= t["brain"]
        and t["destroyed"] <= t["brain"]
        for t in live
    )
    parts = [
        f"{len(live)} live task(s) (+{len(tasks) - len(live)} degenerate) on n="
        f"{body.get('splits', {}).get('n_score_B', 0)} B-rows scored against "
        f"{body.get('splits', {}).get('n_fit_A', 0)} A-rows",
        f"comparator {COMPARATOR}",
    ]
    if "path_a" in body:
        parts.append(
            f"table {body['path_a']['columns']} cols / {body['path_a']['n_present']} customers"
        )
    parts.append(
        f"brain+ledger {len(body['columns']['brain_ledger']['tasks'])} task(s)"
        + (
            f" ({body['columns']['brain_ledger']['note']})"
            if body["columns"]["brain_ledger"].get("note")
            else ""
        )
    )
    unreadable = [t["name"] for t in live if t.get("controls_unreadable")]
    if unreadable:
        # say it out loud: the card's H4 gate ("control <= brain") is satisfied by a control at
        # R2 = -1e10, so on those traits the negative control is not evidence of anything
        parts.append(
            f"WARNING: negative control uninformative on {len(unreadable)}/{len(live)} trait(s) "
            f"({', '.join(unreadable[:4])}) — R2 < -1 is a conditioning artifact of the wide "
            f"design, so 'control below brain' proves nothing there"
        )
    diag = [t for t in (body.get("diagnostics") or {}).get("ledger_only", {}).get("tasks", [])]
    dlive = [t for t in diag if not t.get("degenerate")]
    if dlive:
        best = max(dlive, key=lambda t: t["brain"])
        beats = sum(1 for t in dlive if t["ci_lo"] > 0)
        parts.append(
            f"ledger-only diagnostic: beats the table on {beats}/{len(dlive)} trait(s), best "
            f"R2={best['brain']} on '{best['name']}' vs the table's {best['pathA']} "
            + (
                " — SPINE §5's 'the ledger is what the table already is' does NOT hold here, "
                "so the baseline or the corollary needs an operator ruling"
                if beats
                else " — consistent with SPINE §5"
            )
        )
    if not live:
        return False, "; ".join(parts) + " | " + body["columns"]["brain"].get("note", "no tasks")
    return complete, "; ".join(parts)


def math_ok(t: dict) -> bool:
    vals = [t.get(k) for k in ("brain", "pathA", "ci_lo", "ci_hi", "n")]
    return all(v is not None and np.isfinite(float(v)) for v in vals) and t["ci_lo"] <= t["ci_hi"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--products",
        default=str(ROOT / "looking_glass" / "artifacts" / "cfm"),
        help="dir holding cfm_products.duckdb, or its path",
    )
    ap.add_argument("--demo", default=str(ROOT / "rabbit_hole" / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "docs/report_cards/experiments"))
    ap.add_argument(
        "--legs", default="supervised", help="comma-separated subset of " + ",".join(LEGS)
    )
    a = ap.parse_args()

    db = _products_db(a.products)
    if not db.exists():
        print(
            f"judge: no products DB at {db} — produce it with\n"
            f"  python -m looking_glass.customer_foundation_model products --out-dir {a.products}",
            file=sys.stderr,
        )
        return 2
    demo = Path(a.demo)
    if not (demo / "reference_truth.json").exists():
        print(
            f"judge: no reference_truth.json under {demo} — produce it with\n"
            f"  python3 scripts/make_demo_data.py --out {demo}",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(a.out_dir)
    rc = 0
    for leg in [s.strip() for s in a.legs.split(",") if s.strip()]:
        if leg not in LEGS:
            print(f"judge: unknown leg {leg!r} (have {sorted(LEGS)})", file=sys.stderr)
            rc = 2
            continue
        t0 = time.perf_counter()
        body = LEGS[leg](db, demo)
        working, evidence = _evidence(leg, body)
        body.update(_rollup({leg: {"working": working, "evidence": evidence}}))
        path = out_dir / f"judge_leg_{leg}.json"
        stamp_merge_write(
            path, body, " ".join([Path(__file__).name] + sys.argv[1:]), stamp_blocks=False
        )
        print(
            f"judge_leg_{leg} -> {path} ({time.perf_counter() - t0:.1f}s): {evidence}", flush=True
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
