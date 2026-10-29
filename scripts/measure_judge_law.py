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

from measure_battery_law import (  # noqa: E402  (one definition for every arm, head and CI)
    MIN_ARM_DAYS,
    N_BOOT,
    SEED,
    _load_all,
    _paired_ci,
    _ridge_head,
    _ridge_oos,
    controls_matched,
)
from measure_input_law import _rollup, stamp_merge_write  # noqa: E402
from measure_views_law import (  # noqa: E402  (one definition)
    TARGETS,
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


def _paired_ci_many(deltas, n_boot: int = N_BOOT, seed: int = SEED, cap: int | None = None):
    """Paired percentile CI for a LARGE-n paired statistic (customer PAIRS, not customers).

    `measure_battery_law._paired_ci` builds the whole (n_boot x n) index matrix at once, which is
    correct for ~700 customers and ~4 GB of RAM for ~260k pairs. This variant draws one replicate
    at a time, so memory is O(cap) rather than O(n_boot x n). `cap` bounds the pair count by a
    budget derived from the pool the run actually has (the pair count is >> the customer count, so
    subsampling pairs bounds cost without shrinking the population); it is recorded in the receipt.
    """
    d = np.asarray(deltas, dtype=np.float64)
    n_all = len(d)
    rng = np.random.default_rng(seed)
    if cap is None:
        from looking_glass import gpu_budget

        # one float64 per (replicate x pair) at 5% of available memory, floored at 20k pairs
        cap = max(20_000, int(0.05 * gpu_budget.available_gb() * 2**30 / (8 * n_boot)))
    if n_all > cap:
        d = d[rng.choice(n_all, size=cap, replace=False)]
    n = len(d)
    means = np.fromiter(
        (d[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)), float, count=n_boot
    )
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_stat_ci(stat_fn, n: int, n_boot: int = N_BOOT, seed: int = SEED):
    """Paired CI of (brain statistic - PATH A statistic) for a statistic that is not a per-row mean.

    Cluster agreement and policy value are computed on the SAMPLE, so the bootstrap must recompute
    them inside each replicate — averaging a per-unit surrogate and calling it paired would be
    exactly the fake CI rule 5 forbids. `stat_fn(idx)` returns the two scores on the resampled rows.
    """
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        brain, tab = stat_fn(idx)
        diffs[b] = brain - tab
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _task(name, metric, yb, p_brain, p_tab, r2_rand, r2_dest, ci=None) -> dict:
    """One task row of a leg receipt: score, comparator, paired CI, controls. Measurements only —
    the card owns verdicts, and a receipt may not carry a criterion (rule 3)."""
    if ci is None:
        _mean, lo, hi = _paired_ci((np.asarray(yb) - p_tab) ** 2 - (np.asarray(yb) - p_brain) ** 2)
    else:
        lo, hi = ci
    unreadable = [
        ctl
        for ctl, r2 in (("random", r2_rand), ("destroyed", r2_dest))
        if r2 is not None and r2 < -1.0
    ]
    return {
        "name": name,
        "metric": metric,
        "brain": None if p_brain is None else round(float(_r2(yb, p_brain)), 6),
        "pathA": None if p_tab is None else round(float(_r2(yb, p_tab)), 6),
        "ci_lo": round(float(lo), 6),
        "ci_hi": round(float(hi), 6),
        "n": int(len(yb)),
        "random": None if r2_rand is None else round(float(r2_rand), 6),
        "destroyed": None if r2_dest is None else round(float(r2_dest), 6),
        "comparator": COMPARATOR,
        "controls_unreadable": unreadable,
    }


def _degenerate(name, reason) -> dict:
    return {"name": name, "degenerate": True, "degenerate_reason": reason}


def _rows_by_key(source: dict, keys, name: str, problems: list) -> dict:
    """key -> vector, with the ragged/missing check done where the artifact is first read.

    A column whose width disagrees with the producer's own stamp is not a measurement problem to
    average through: it means the reader is reading something other than what was published.
    """
    out: dict = {}
    for k in keys:
        v = source.get(k)
        if v is None:
            problems.append(f"{name}: customer {k!r} is not served")
            continue
        out[k] = np.asarray(v, dtype=np.float64)
    widths = {v.shape[0] for v in out.values()}
    if len(widths) > 1:
        problems.append(f"{name}: ragged served widths {sorted(widths)}")
    return out


def _arms(db: Path, demo: Path) -> dict:
    """One population, one split, one table encoding, one ledger — the DEC-428 geometry, shared.

    Extracted from `leg_supervised` so every later leg (DEC-434) measures against the SAME frozen
    protocol rather than a near-copy: the same `encoder_samples` split, the same PATH A matrix
    encoded once over A ∪ B, the same two column definitions (`brain`, `brain + ledger`). A leg
    that quietly re-derived any of those would be grading a different judge than the card claims.
    """
    _reg, truth, emb, _raw, stream, bkeys, akeys, _as_of = _load_all(str(db), demo)
    problems: list = []
    table, tmeta = _table_matrix(demo, list(akeys) + list(bkeys))
    views, _stamp = _load_views(db)
    ledger = views.get("measure", {})

    brain_a = _rows_by_key(emb, akeys, "state_embeddings(A)", problems)
    brain_b = _rows_by_key(emb, bkeys, "state_embeddings(B)", problems)
    led_a = _rows_by_key(ledger, akeys, "measure view(A)", problems)
    led_b = _rows_by_key(ledger, bkeys, "measure view(B)", problems)
    tab = _rows_by_key(table or {}, list(akeys) + list(bkeys), "PATH A table", problems)

    columns = {"brain": {"A": brain_a, "B": brain_b}}
    columns["brain_ledger"] = {
        "A": {k: np.concatenate([brain_a[k], led_a[k]]) for k in brain_a if k in led_a},
        "B": {k: np.concatenate([brain_b[k], led_b[k]]) for k in brain_b if k in led_b},
    }
    return {
        "truth": truth,
        "stream": stream,
        "akeys": akeys,
        "bkeys": bkeys,
        "columns": columns,
        "table": tab,
        "path_a": tmeta,
        "problems": problems,
        "ledger_served": bool(led_a and led_b),
    }


def _common_populations(a: dict, col: str, want: set) -> tuple[list, list]:
    """(A keys, B keys) that exist in EVERY arm this task needs, in split order.

    Arms are intersected, never zipped: a customer missing from one arm would otherwise shift a
    matrix by one row and show up as a model difference.
    """
    ak = [k for k in a["akeys"] if k in a["columns"][col]["A"] and k in a["table"]]
    bk = [k for k in a["bkeys"] if k in a["columns"][col]["B"] and k in a["table"]]
    if want:
        ak = [k for k in ak if k in want]
        bk = [k for k in bk if k in want]
    return ak, bk


def _fit_scale(rows: dict) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) fitted on split A ONLY — shared by every arm of every leg."""
    A = np.vstack(list(rows.values()))
    sd = A.std(0)
    return A.mean(0), np.where(sd < 1e-12, 1.0, sd)


def _z(rows: dict, mu, sd) -> dict:
    """z-score B by A-fitted statistics (both arms get the same treatment: distance and
    clustering are scale-sensitive, and the wide table mixes event counts with fractions, so
    letting each arm choose its own scaling would be a free hyperparameter for PATH A)."""
    return {k: (v - mu) / sd for k, v in rows.items()}


def _z1(v, mu, sd) -> np.ndarray:
    return (v - mu) / sd


#: principle-ok: compute bound on the pair bootstrap. The POPULATION is all pairs (reported in
#: `n`); the resample size is a cost knob, and it travels in the receipt as `n_bootstrap_pairs`.
BOOT_PAIRS = 100_000


def _alignment(D, Dt):
    """Squared correlation between an arm's pairwise distances and the lab's true distances.

    Scale-free on purpose. The first run of this leg scored MSE-R2 of two raw distance vectors
    and reported -649 for the brain and -4.6e15 for a control: numbers dominated by the arms'
    UNITS, not by whether the geometry agrees. "Distance equals shared futures" is a claim about
    agreement, so the statistic is the one that measures agreement and lands in [0, 1] for both
    arms and both controls alike.
    """
    D, Dt = np.asarray(D, float), np.asarray(Dt, float)
    if D.size < 3 or D.std() < 1e-12 or Dt.std() < 1e-12:
        return None
    return float(np.corrcoef(D, Dt)[0, 1] ** 2)


def _round(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


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
        # ONE control generator for the whole judge (DEC-434): the noise is drawn at the ARM's own
        # per-column scale, so a control can only fail for lack of information. `_controls`' single
        # global sigma made the `brain + ledger` column's control score R2 = -4.6e15, which is an
        # arithmetic artifact, and DEC-433 gives an unreadable control the power to block a PASS.
        rand, destroyed = controls_matched(rng, arm_a, arm_b)
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


def leg_unsupervised(db: Path, demo: Path) -> dict:
    """Does the donor put customers near each other BECAUSE they are alike? (DEC-434)

    The leg's definition is "distance equals shared futures; structure beats table/random/destroyed".
    Three tasks, each scored the same way for both arms and against PATH A on the same rows: two
    distance-fidelity tasks (the lab's persona latents, then its future outcomes) and one cluster
    task against the lab's declared `segment`. Nothing here is fitted to the labels — the lab's
    truth is used only to score, never to train (anti-overfit clause).
    """
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    a = _arms(db, demo)
    truth, akeys, bkeys = a["truth"], a["akeys"], a["bkeys"]
    if a["problems"] or a["table"] is None:
        return {
            "columns": _empty_columns(
                "arms not aligned: "
                + "; ".join(a["problems"][:3] or [a["path_a"].get("error", "?")])
            ),
            "excluded": [],
        }

    # trait sets are read out of the truth file's own schema, not hand-picked: `future` is the
    # frozen supervised TARGETS, `persona` is every other numeric field the lab declares
    sample = next(iter(truth.values()))
    numeric = tuple(sorted(k for k, v in sample.items() if isinstance(v, (int, float))))
    future_fields = [f for f in TARGETS if f in numeric]
    persona_fields = [f for f in numeric if f not in set(TARGETS) | {"offer_opt"}]

    columns: dict = {}
    for col in COLUMNS:
        colA, colB = a["columns"][col]["A"], a["columns"][col]["B"]
        tasks = []
        for tname, fields in (
            ("persona_geometry", persona_fields),
            ("future_geometry", future_fields),
        ):
            ka = [k for k in akeys if k in truth and k in colA and k in a["table"]]
            kb = [k for k in bkeys if k in truth and k in colB and k in a["table"]]
            if not fields:
                tasks.append(
                    _degenerate(tname, "the lab declares no numeric fields for this geometry")
                )
                continue
            if len(ka) < 30 or len(kb) < 30:
                tasks.append(
                    _degenerate(tname, f"too few customers in both arms (A={len(ka)}, B={len(kb)})")
                )
                continue
            vec = lambda k: np.asarray([truth[k][f] for f in fields], float)  # noqa: E731
            muT, sdT = _fit_scale({k: vec(k) for k in ka})
            muB, sdB = _fit_scale({k: colA[k] for k in ka})
            muW, sdW = _fit_scale({k: a["table"][k] for k in ka})
            Dt = pdist(np.vstack([_z1(vec(k), muT, sdT) for k in kb]))
            EA = np.vstack([colA[k] for k in ka])
            rng = np.random.default_rng(SEED)
            rand, dest = controls_matched(rng, EA, np.vstack([colB[k] for k in kb]))
            D = {
                "brain": pdist(np.vstack([_z1(colB[k], muB, sdB) for k in kb])),
                "pathA": pdist(np.vstack([_z1(a["table"][k], muW, sdW) for k in kb])),
                "random": pdist(_z1(rand, muB, sdB)),
                "destroyed": pdist(_z1(dest, muB, sdB)),
            }
            score = {arm: _alignment(d, Dt) for arm, d in D.items()}
            # the statistic is a correlation, so the bootstrap re-correlates inside each
            # replicate over a capped subsample of the SAME pairs (a cost knob, recorded)
            cap = min(Dt.size, BOOT_PAIRS)
            idx = (
                rng.choice(Dt.size, size=cap, replace=False)
                if Dt.size > cap
                else np.arange(Dt.size)
            )
            Db, Dw, Dt_c = D["brain"][idx], D["pathA"][idx], Dt[idx]
            lo, hi = _paired_stat_ci(
                lambda q: (_alignment(Db[q], Dt_c[q]), _alignment(Dw[q], Dt_c[q])), cap
            )
            tasks.append(
                {
                    "name": tname,
                    "metric": "squared correlation between the arm's pairwise distances and the "
                    f"lab's true distances ({len(fields)} fields: {', '.join(fields)})",
                    "brain": _round(score["brain"]),
                    "pathA": _round(score["pathA"]),
                    "ci_lo": round(float(lo), 6),
                    "ci_hi": round(float(hi), 6),
                    "n": int(Dt.size),
                    "n_bootstrap_pairs": int(cap),
                    "random": _round(score["random"]),
                    "destroyed": _round(score["destroyed"]),
                    "comparator": COMPARATOR,
                    "controls_unreadable": [c for c in ("random", "destroyed") if score[c] is None],
                }
            )
        # ---- cluster structure against the lab's declared segment -------------------
        seg = {k: truth[k].get("segment") for k in truth if truth[k].get("segment") is not None}
        ka = [k for k in akeys if k in seg and k in colA and k in a["table"]]
        kb = [k for k in bkeys if k in seg and k in colB and k in a["table"]]
        klab = len({seg[k] for k in kb})  # k derived from the label's own cardinality, not typed
        if klab < 2 or len(ka) < 30 or len(kb) < 30:
            tasks.append(
                _degenerate(
                    "segment_structure", "fewer than two declared segments, or too few customers"
                )
            )
        else:
            muB, sdB = _fit_scale({k: colA[k] for k in ka})
            muW, sdW = _fit_scale({k: a["table"][k] for k in ka})
            XA = np.vstack([_z1(colA[k], muB, sdB) for k in ka])
            XB = np.vstack([_z1(colB[k], muB, sdB) for k in kb])
            WA = np.vstack([_z1(a["table"][k], muW, sdW) for k in ka])
            WB = np.vstack([_z1(a["table"][k], muW, sdW) for k in kb])
            lab = np.array([seg[k] for k in kb])
            # fitted on A once, then the bootstrap recomputes the STATISTIC on each resample: the
            # model is not refit per replicate (A is never resampled), the score is
            n_init = 4  # principle-ok: k-means restarts, a compute-accuracy knob, not a model dim
            assign_b = KMeans(n_clusters=klab, n_init=n_init, random_state=SEED).fit(XA).predict(XB)
            assign_w = KMeans(n_clusters=klab, n_init=n_init, random_state=SEED).fit(WA).predict(WB)
            lo, hi = _paired_stat_ci(
                lambda idx: (
                    float(adjusted_rand_score(lab[idx], assign_b[idx])),
                    float(adjusted_rand_score(lab[idx], assign_w[idx])),
                ),
                len(kb),
            )
            ctl = {}
            rng = np.random.default_rng(SEED)
            # the controls go through the SAME fit-on-A / score-on-B path as the arms. A control
            # clustered on the very rows it is scored on is an easier experiment, and the gap
            # between it and the arm would measure the protocol instead of the representation.
            for cname, side in (("random", 0), ("destroyed", 1)):
                brk_a = controls_matched(rng, XA, XA)[side]
                brk_b = controls_matched(rng, XA, XB)[side]
                km = KMeans(n_clusters=klab, n_init=n_init, random_state=SEED).fit(brk_a)
                ctl[cname] = adjusted_rand_score(lab, km.predict(brk_b))
            tasks.append(
                {
                    "name": "segment_structure",
                    "metric": f"ARI of k-means (k={klab}, from the label's own cardinality) "
                    "against the lab's declared segment",
                    "brain": round(float(adjusted_rand_score(lab, assign_b)), 6),
                    "pathA": round(float(adjusted_rand_score(lab, assign_w)), 6),
                    "ci_lo": round(float(lo), 6),
                    "ci_hi": round(float(hi), 6),
                    "n": int(len(kb)),
                    "random": _round(ctl["random"]),
                    "destroyed": _round(ctl["destroyed"]),
                    "comparator": COMPARATOR,
                    # an ARI is not an MSE-R2, so the frozen vacuity threshold (DEC-433's -1)
                    # cannot fire here, and no new threshold is invented to replace it: only a
                    # control that produced NO number is called unreadable.
                    "controls_unreadable": [c for c in ("random", "destroyed") if ctl[c] is None],
                }
            )
        columns[col] = {"tasks": tasks}
    return {
        "columns": columns,
        "diagnostics": {},
        "excluded": [],
        "splits": {"n_fit_A": len(akeys), "n_score_B": len(bkeys), "source": "encoder_samples"},
        "path_a": a["path_a"],
        "arms": {
            "brain": "state_embeddings donor; pairwise-distance and k-means geometry",
            "brain_ledger": "brain + measure view (ledger), same distance/clustering geometry",
            "pathA": "PATH A wide table, z-scored on split A only, same geometry",
        },
    }


def leg_rl(db: Path, demo: Path) -> dict:
    """Randomized arms: does the donor know WHO responds, not just who buys? (DEC-434)

    Two tasks on the uplift the lab's one-factor-daily randomized offer arm IDENTIFIES (the same
    `arm_cells` definition the S5 battery uses — one definition, not two): R² on the per-customer
    uplift, and the policy value of ranking customers by predicted uplift. The second is the leg's
    actual question — a representation that predicts the average but cannot rank who responds is
    useless to a decision, and a mean-only model still scores respectably on the first.
    """
    from measure_battery_law import arm_cells, identified_uplift

    a = _arms(db, demo)
    if a["problems"] or a["table"] is None:
        return {
            "columns": _empty_columns(
                "arms not aligned: "
                + "; ".join(a["problems"][:3] or [a["path_a"].get("error", "?")])
            ),
            "excluded": [],
        }
    cells = arm_cells(a["stream"], a["truth"])
    up = identified_uplift(cells)
    n_min = 30  # principle-ok: evaluation-contract floor, same as the battery's probes

    columns: dict = {}
    for col in COLUMNS:
        ak, bk = _common_populations(a, col, set(up))
        if len(ak) < n_min or len(bk) < n_min:
            columns[col] = {
                "tasks": [
                    _degenerate(
                        "known_effect_uplift",
                        f"only {len(ak)} A / {len(bk)} B customers have >= the identifiability "
                        f"floor of arm-days on both sides",
                    )
                ]
            }
            continue
        ya = np.array([up[k] for k in ak])
        yb = np.array([up[k] for k in bk])
        Ea = np.vstack([a["columns"][col]["A"][k] for k in ak])
        Eb = np.vstack([a["columns"][col]["B"][k] for k in bk])
        TA = np.vstack([a["table"][k] for k in ak])
        TB = np.vstack([a["table"][k] for k in bk])
        p_brain, _r2b = _ridge_oos(Ea, ya, Eb, yb)
        if p_brain is None:
            columns[col] = {
                "tasks": [_degenerate("known_effect_uplift", "uplift has no variance to fit")]
            }
            continue
        p_tab = _gbm_oos(TA, ya, TB)
        rng = np.random.default_rng(SEED)
        rand, dest = controls_matched(rng, Ea, Eb)
        _, r2_rand = _ridge_oos(Ea, ya, rand, yb)
        _, r2_dest = _ridge_oos(Ea, ya, dest, yb)
        tasks = [
            _task(
                "known_effect_uplift",
                "R2 on the identified per-customer uplift "
                "(P(buy|offer=opt) - P(buy|offer off by >=2))",
                yb,
                p_brain,
                p_tab,
                r2_rand,
                r2_dest,
            )
        ]
        p_ctl = {
            "random": _ridge_oos(Ea, ya, rand, yb)[0],
            "destroyed": _ridge_oos(Ea, ya, dest, yb)[0],
        }

        def _policy_value(pred, idx):
            """Area under the top-k realized-uplift curve: the mean of the cumulative mean uplift
            as the campaign grows, under THIS arm's ranking. Random ranking gives the population
            mean, so a control cannot win by luck of the base rate."""
            order = idx[np.argsort(-pred[idx], kind="stable")]
            ys = yb[order]
            return float(np.mean(np.cumsum(ys) / np.arange(1, ys.size + 1)))

        full = np.arange(len(yb))
        lo, hi = _paired_stat_ci(
            lambda idx: (_policy_value(p_brain, idx), _policy_value(p_tab, idx)), len(yb)
        )
        tasks.append(
            {
                "name": "policy_value",
                "metric": "area under the top-k realized-uplift curve on randomized arm-days "
                "(bootstrap re-selects inside each replicate)",
                "brain": round(_policy_value(p_brain, full), 6),
                "pathA": round(_policy_value(p_tab, full), 6),
                "ci_lo": round(float(lo), 6),
                "ci_hi": round(float(hi), 6),
                "n": int(len(yb)),
                "random": None
                if p_ctl["random"] is None
                else round(_policy_value(p_ctl["random"], full), 6),
                "destroyed": None
                if p_ctl["destroyed"] is None
                else round(_policy_value(p_ctl["destroyed"], full), 6),
                "comparator": COMPARATOR,
                "controls_unreadable": [],
            }
        )
        columns[col] = {"tasks": tasks}
    return {
        "columns": columns,
        "diagnostics": {},
        "excluded": [],
        "splits": {
            "n_fit_A": len(a["akeys"]),
            "n_score_B": len(a["bkeys"]),
            "source": "encoder_samples",
        },
        "identified": {
            "population": len(up),
            "rule": f">= {MIN_ARM_DAYS} identified arm-days on each side of the contrast",
        },
        "path_a": a["path_a"],
        "arms": {
            "brain": "state_embeddings donor + linear ridge head on the identified uplift",
            "brain_ledger": "brain + measure view (ledger), same head and rows",
            "pathA": "PATH A wide table + the DEC-428 GBM incumbent, same rows",
        },
    }


def leg_simulator(db: Path, demo: Path) -> dict:
    """Can the donor answer "what if"? The red_king role, on identified counterfactuals (DEC-434)

    The lab bakes a known dose-response into the offer arm (q *= 1 - 0.22*|offer-offer_opt|) and
    randomizes it daily, so the response at each level is IDENTIFIED per customer from data alone.
    Two tasks: reproduce the curve, and NAME the customer's optimum level. A model can recover the
    mean response and still be useless here — which is exactly the difference between a predictor
    and a simulator.
    """
    from measure_battery_law import arm_cells, identified_dose_response

    a = _arms(db, demo)
    if a["problems"] or a["table"] is None:
        return {
            "columns": _empty_columns(
                "arms not aligned: "
                + "; ".join(a["problems"][:3] or [a["path_a"].get("error", "?")])
            ),
            "excluded": [],
        }
    dr = identified_dose_response(arm_cells(a["stream"], a["truth"]))
    levels = sorted({lvl for lv in dr.values() for lvl in lv})  # read off the stream, not typed
    onehot = lambda lvl: np.eye(len(levels))[levels.index(lvl)]  # noqa: E731

    columns: dict = {}
    for col in COLUMNS:
        ak, bk = _common_populations(a, col, set(dr))
        if len(ak) < 30 or len(bk) < 30 or len(levels) < 2:
            columns[col] = {
                "tasks": [
                    _degenerate(
                        "dose_response_curve",
                        f"only {len(ak)} A / {len(bk)} B customers have an identified response level "
                        f"(>= {MIN_ARM_DAYS} arm-days), or fewer than two levels appear",
                    )
                ]
            }
            continue

        def cells(keys, rows, broken=None):
            """(customer x identified level) cells; `broken` replaces the representation half only,
            so a control still knows which arm it is scoring — otherwise it fails for a reason that
            has nothing to do with the donor."""
            X, y, ks = [], [], []
            for i, k in enumerate(keys):
                rep = rows[k] if broken is None else broken[i]
                for lvl, (rate, _days) in sorted(dr[k].items()):
                    X.append(np.concatenate([rep, onehot(lvl)]))
                    y.append(rate)
                    ks.append(k)
            return np.vstack(X), np.asarray(y, float), ks

        def table_cells(keys):
            X, y, ks = [], [], []
            for k in keys:
                for lvl, (rate, _days) in sorted(dr[k].items()):
                    X.append(np.concatenate([a["table"][k], onehot(lvl)]))
                    y.append(rate)
                    ks.append(k)
            return np.vstack(X), np.asarray(y, float), ks

        Xa, ya, _ = cells(ak, a["columns"][col]["A"])
        Xb, yb, _ = cells(bk, a["columns"][col]["B"])
        Ta, _ty, _ = table_cells(ak)
        Tb, _sy, _ = table_cells(bk)
        rng = np.random.default_rng(SEED)
        Eb = np.vstack([a["columns"][col]["B"][k] for k in bk])
        Ea = np.vstack([a["columns"][col]["A"][k] for k in ak])
        rand_b, dest_b = controls_matched(rng, Ea, Eb)
        rand_a, dest_a = controls_matched(rng, Ea, Ea)
        Xr, _ry, _ = cells(bk, None, broken=rand_b)
        Xd, _dy, _ = cells(bk, None, broken=dest_b)

        p_brain, _r2b = _ridge_oos(Xa, ya, Xb, yb)
        if p_brain is None:
            columns[col] = {
                "tasks": [_degenerate("dose_response_curve", "identified curve has no variance")]
            }
            continue
        p_tab = _gbm_oos(Ta, ya, Tb)
        _, r2_rand = _ridge_oos(Xa, ya, Xr, yb)
        _, r2_dest = _ridge_oos(Xa, ya, Xd, yb)
        tasks = [
            _task(
                "dose_response_curve",
                f"R2 on the identified (customer x offer level) purchase rate "
                f"({len(levels)} levels: {levels})",
                yb,
                p_brain,
                p_tab,
                r2_rand,
                r2_dest,
            )
        ]

        # ---- does the recovered surface NAME the optimum? --------------------------
        # one fit per arm, then every (customer x level) row scored in ONE batch: fitting per
        # customer is the per-item anti-pattern the doctrine bans, and would let each customer's
        # fit see a different amount of data.
        def rows_for(keys, source, broken=None):
            X = []
            for i, k in enumerate(keys):
                rep = source[k] if broken is None else broken[i]
                for lvl in levels:
                    X.append(np.concatenate([rep, onehot(lvl)]))
            return np.vstack(X)

        def hits(pred, keys):
            """argmax of the recovered curve vs the lab's declared optimum (ties -> lowest level)."""
            curve = np.asarray(pred, float).reshape(len(keys), len(levels))
            want = np.array([int(a["truth"][k]["offer_opt"]) for k in keys], int)
            got = np.array([levels[i] for i in curve.argmax(1)], int)
            return (got == want).astype(float)

        brain_head = _ridge_head().fit(Xa, ya)
        hit_b = hits(brain_head.predict(rows_for(bk, a["columns"][col]["B"])), bk)
        import xgboost

        gbm = xgboost.XGBRegressor(**INCUMBENT).fit(Ta, ya)
        hit_t = hits(
            gbm.predict(rows_for(bk, {k: a["table"][k] for k in bk})),
            bk,
        )
        lo, hi = _paired_stat_ci(
            lambda idx: (float(hit_b[idx].mean()), float(hit_t[idx].mean())), len(bk)
        )
        ctl_hits = {}
        for cname, brk, brk_a in (("random", rand_b, rand_a), ("destroyed", dest_b, dest_a)):
            Xa_ctl, _ay, _ak = cells(ak, None, broken=brk_a)
            head = _ridge_head().fit(Xa_ctl, ya[: Xa_ctl.shape[0]])
            pred = head.predict(rows_for(bk, None, broken=brk))
            ctl_hits[cname] = float(hits(pred, bk).mean())
        tasks.append(
            {
                "name": "argmax_optimum",
                "metric": "share of customers whose declared offer optimum is the argmax of the "
                "recovered response surface (ties -> lowest level)",
                "brain": round(float(hit_b.mean()), 6),
                "pathA": round(float(hit_t.mean()), 6),
                "ci_lo": round(float(lo), 6),
                "ci_hi": round(float(hi), 6),
                "n": int(len(bk)),
                "random": round(float(ctl_hits["random"]), 6),
                "destroyed": round(float(ctl_hits["destroyed"]), 6),
                "comparator": COMPARATOR,
                "controls_unreadable": [],
            }
        )
        columns[col] = {"tasks": tasks}
    return {
        "columns": columns,
        "diagnostics": {},
        "excluded": [],
        "splits": {
            "n_fit_A": len(a["akeys"]),
            "n_score_B": len(a["bkeys"]),
            "source": "encoder_samples",
        },
        "identified": {
            "levels": levels,
            "customers": len(dr),
            "rule": f"an offer level is identified for a customer at >= {MIN_ARM_DAYS} arm-days",
        },
        "path_a": a["path_a"],
        "arms": {
            "brain": "state_embeddings donor + level one-hot -> linear head on the identified curve",
            "brain_ledger": "brain + measure view (ledger), same head and cells",
            "pathA": "PATH A wide table + level one-hot -> the DEC-428 GBM incumbent",
        },
    }


LEGS = {
    "supervised": leg_supervised,
    "unsupervised": leg_unsupervised,
    "rl": leg_rl,
    "simulator": leg_simulator,
}


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
