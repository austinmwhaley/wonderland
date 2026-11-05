#!/usr/bin/env python3
"""B2 sufficiency law (DEC-570, spec frozen 2026-11-02; D5 accepted with defaults as proposed).

`U` = the worst-case oracle gap of the SERVED OBJECT vs the FULL-HISTORY arm, at FIXED capacity,
over the published self-supervised probe family — the certification metric of AGENTS.md
(Layer 1): `U_tau = 1 - (loss_tau(E) - loss_tau(full)) / (loss_tau(full) - loss_tau(floor))`,
`U = min_tau`. `full` is the full event history per customer as the stream serves it (PATH A's
history-only wide table — leak-free per docs/report_cards/experiments/path_a_provenance.json:
all 10 fut-stamped columns dropped by the comparator's own rule); `floor` is the
population-constant predictor (the Bayes floor at this fixed capacity). A probe whose headroom
`loss_tau(full) - loss_tau(floor) <= 0` is EXCLUDED and reported (degenerate, never forced).

Arms (DEC-570/DEC-571; O0's EXPOSURE branch fixed the served arm by measurement):
  * served_object  = learned shape (M4_served) concatenated with the ledger's PUBLISHED part
    blocks in view_layout.PART_ORDER["measure"] — the concatenation is cross-checked
    byte-identical to the judge's flat ledger arm (the O0 readers, re-used, not re-typed)
  * learned_only   = the shape alone — the rule-4 attribution control, in the same receipt
  * full           = PATH A's wide table (the judge's own matrix path)
  * floor          = the population constant (1-d ones)

Every head is THE shipping head `_ridge_head` (standardized RidgeCV, one alpha grid) at fixed
capacity, fit on split A, scored on split B — the same frozen geometry the judge uses. Losses:
held-out B MSE (regression probes) and softmax NLL with a 1e-15 p-floor (next-type). U is
invariant to the R2-vs-MSE choice (per-probe constant B-total-SS scaling cancels in the ratio).
This receipt reports losses, not R2.

Probe family (pre-registered, self-supervised, no labels; BROADENED later, never narrowed):
  1. next-type            the type of the customer's next event after the boundary
  2. attr-next            a numeric attribute value of the next event (mean held-out loss across
                           the registry's included numeric fields is the probe's loss)
  3. numeric-next         the sum of the next event's present-field z (absent = 0), same per-field
                           A scales as attr-next
  4. dt                   time to the next event (log1p seconds), z on A-included
  5. horizon-count-value  event count + numeric mass over (as_of, as_of + H], H = one seeded draw
                           from the stream's own pooled positive within-customer gaps (recorded)

Arm consistency is by construction: every arm matrix is built over the SAME population
(shape ∩ ledger ∩ table ∩ every published part, per side), so each probe scores the same rows on
every arm. No per-arm customer drops after the fact.

Honesty (rule 4): the served object is shape-PLUS-ledger; learned_only travels in this same
receipt so a U the ledger alone carries is labeled served-statistic sufficiency, never
learned-shape sufficiency. This receipt carries no criterion of its own (rule 3): the bar
lives in b_spec.SUFFICIENCY_BAR and the card.

CPU only; no retraining; one job at a time inside the 32 GB slice.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from looking_glass import view_layout as V  # noqa: E402
from measure_battery_law import SEED, _load_all, _ridge_oos  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402
from measure_judge_law import _arms, _round  # noqa: E402
from measure_served_object import (  # noqa: E402
    _faceplate_ledger,
    _part_rows,
    _parts_match_flat,
    _widths_match,
)
from measure_stage_decomposition import _load_stage_arrays  # noqa: E402
from safe_run import ensure_boundary  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_sufficiency_law.json"
P_FLOOR = 1e-15  # frozen softmax p-floor (evaluation contract, recorded)
ARMS = ("served_object", "learned_only", "full", "floor")


def _field_source(f: str, reg: dict) -> str:
    """Where each registry numeric field physically lives (measured, not assumed): `value` is
    the stream's top-level column; the rest are JSON attributes. The parse is the frozen
    try_cast of the pinned expression."""
    return "column:value" if f == "value" else f"json:$.{f}"


def _next_rows(
    con: duckdb.DuckDBPyConnection, pq: str, as_of_s: str, fields: list[str]
) -> dict[str, dict]:
    """First next event per customer after the boundary: type, dt (seconds), each field's raw
    value (NULL-safe). Tie-break on (ts, event_type) is the frozen first-next rule.
    `value` reads the top-level column; the rest read JSON attributes (the pinned parse)."""
    json_fields = [f for f in fields if f != "value"]
    field_cols = ", ".join(
        [
            f"try_cast(json_extract_string(event_attributes, '$.{f}') as double) as f_{f}"
            for f in json_fields
        ]
        + ["try_cast(value as double) as f_value"]
    )
    q = f"""
    select customer_key, event_type, dt_s, {", ".join(f"f_{f}" for f in fields)}
    from (
      select customer_key,
             event_type,
             extract(epoch from (try_cast(event_ts as timestamp) - try_cast('{as_of_s}' as timestamp))) as dt_s,
             {field_cols},
             row_number() over (
               partition by customer_key order by try_cast(event_ts as timestamp), event_type) as rn
      from '{pq}'
      where try_cast(event_ts as timestamp) > try_cast('{as_of_s}' as timestamp)
    ) where rn = 1
    """
    out: dict[str, dict] = {}
    for row in con.execute(q).fetchall():
        rec = dict(
            zip(("customer_key", "event_type", "dt_s") + tuple(f"f_{f}" for f in fields), row)
        )
        out[rec["customer_key"]] = rec
    return out


def _gaps(con: duckdb.DuckDBPyConnection, pq: str) -> np.ndarray:
    """Pooled positive within-customer gaps (seconds), FULL stream, both sides of the boundary.
    One DuckDB pass over native timestamps (the autotune twin assumes a Polars frame; this is the
    same statistic, measured here against the stream the U probes actually score)."""
    q = f"""
    select extract(epoch from (cur - prev)) as g
    from (
      select customer_key,
             try_cast(event_ts as timestamp) as cur,
             lag(try_cast(event_ts as timestamp)) over (
               partition by customer_key order by try_cast(event_ts as timestamp)) as prev
      from '{pq}'
    ) where prev is not null and cur > prev
    """
    return np.asarray([r[0] for r in con.execute(q).fetchall()], float)


def _horizon_window(
    con: duckdb.DuckDBPyConnection, pq: str, as_of_s: str, horizon_s: float, fields: list[str]
) -> dict[str, dict]:
    """Event count + raw per-field sum over (as_of, as_of + H]. The z-mass is applied in numpy
    from the A-included scales (one source for the scales — never re-typed into SQL)."""
    json_fields = [f for f in fields if f != "value"]
    field_cols = ", ".join(
        [
            f"try_cast(json_extract_string(event_attributes, '$.{f}') as double) as f_{f}"
            for f in json_fields
        ]
        + ["try_cast(value as double) as f_value"]
    )
    mass_terms = ", ".join(f"sum(f_{f}) as sum_{f}" for f in fields)
    q = f"""
    select customer_key, count(*) as cnt, {mass_terms}, {", ".join(f"count(f_{f}) as npres_{f}" for f in fields)}
    from (
      select customer_key, {field_cols}
      from '{pq}'
      where try_cast(event_ts as timestamp) > try_cast('{as_of_s}' as timestamp)
        and try_cast(event_ts as timestamp) <= try_cast('{as_of_s}' as timestamp)
          + {horizon_s:.6f} * interval '1' second
    ) group by customer_key
    """
    cols = (
        ("customer_key", "cnt")
        + tuple(f"sum_{f}" for f in fields)
        + tuple(f"npres_{f}" for f in fields)
    )
    out: dict[str, dict] = {}
    for row in con.execute(q).fetchall():
        rec = dict(zip(cols, row))
        out[rec["customer_key"]] = rec
    return out


def main() -> int:
    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--products",
        default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/cfm_v2k",
    )
    ap.add_argument(
        "--demo",
        default="/home/austin_whaley/Documents/Repos/wonderland/looking_glass/artifacts/demo_v2",
    )
    ap.add_argument(
        "--stream",
        default=None,
        help="event stream parquet (default: <demo>/path_b/path_b_customer_event_stream.parquet)",
    )
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.perf_counter()
    products = Path(a.products)
    demo = Path(a.demo)
    db = products / "cfm_products.duckdb"
    pq = str(
        Path(a.stream) if a.stream else demo / "path_b" / "path_b_customer_event_stream.parquet"
    )

    reg, _truth, _emb, _raw, _hist, bkeys, akeys, as_of = _load_all(str(db), demo)
    fields = tuple(reg["fields"]["numeric_fields"])
    as_of_s = as_of.astype("datetime64[s]").astype(str)

    arms = _arms(db, demo)
    if arms["problems"]:
        raise SystemExit("sufficiency: arms not aligned: " + "; ".join(arms["problems"][:3]))
    table = arms["table"]
    ledger = arms["ledger"]

    shape = _load_stage_arrays(products)["M4_served"]
    part_rows = _part_rows(db)
    order = list(V.PART_ORDER.get("measure", []))
    _slices, declared, view_width = _faceplate_ledger(products)
    widths_ok, widths_note = _widths_match(part_rows, declared, view_width, ledger)
    match_ok, match_note = _parts_match_flat(part_rows, order, ledger)

    # ---- one population, one split: every arm scores the same rows (arm consistency) ----
    shape_dim = next(iter(shape.values())).shape[0]

    def _part_vec(p: str, k: str) -> np.ndarray:
        return np.asarray(part_rows[p][k], float)

    def _served_vec(k: str) -> np.ndarray:
        return np.concatenate([np.asarray(shape[k], float)] + [_part_vec(p, k) for p in order])

    popA = [k for k in akeys if all(k in part_rows[p] for p in order)]
    popB = [k for k in bkeys if all(k in part_rows[p] for p in order)]
    # akeys/bkeys come from encoder_samples; shape/ledger/table membership is the same split
    popA = [k for k in popA if k in shape and k in ledger["A"] and k in table]
    popB = [k for k in popB if k in shape and k in ledger["B"] and k in table]
    if not popA or not popB:
        raise SystemExit("sufficiency: empty A or B population after arm alignment — refusing")
    idxA = {k: i for i, k in enumerate(popA)}
    idxB = {k: i for i, k in enumerate(popB)}

    AM = {
        "served_object": {
            "A": np.vstack([_served_vec(k) for k in popA]),
            "B": np.vstack([_served_vec(k) for k in popB]),
        },
        "learned_only": {
            "A": np.vstack([np.asarray(shape[k], float) for k in popA]),
            "B": np.vstack([np.asarray(shape[k], float) for k in popB]),
        },
        "full": {
            "A": np.vstack([np.asarray(table[k], float) for k in popA]),
            "B": np.vstack([np.asarray(table[k], float) for k in popB]),
        },
        "floor": {"A": np.ones((len(popA), 1)), "B": np.ones((len(popB), 1))},
    }

    # ---- targets (one DuckDB pass each, vectorized) ----
    con = duckdb.connect()
    nxt = _next_rows(con, pq, as_of_s, list(fields))
    gaps = _gaps(con, pq)
    H = float(np.random.default_rng(SEED).choice(gaps)) if len(gaps) else None
    con.close()
    win = {}
    if H is not None:
        con = duckdb.connect()
        win = _horizon_window(con, pq, as_of_s, H, list(fields))
        con.close()

    # per-field A-included z scales (pinned: fit on A customers that HAVE the next-event value)
    scales: dict[str, tuple] = {}
    for f in fields:
        vals = [
            nxt[k][f"f_{f}"]
            for k in popA
            if k in nxt and nxt[k][f"f_{f}"] is not None and math.isfinite(nxt[k][f"f_{f}"])
        ]
        n_A = len(vals)
        if n_A < 8:
            scales[f] = (0.0, 1.0, n_A, False, f"n_A={n_A} < 8 — field excluded, reported")
            continue
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        if sd < 1e-12:
            scales[f] = (mu, 1.0, n_A, False, f"sd={sd:.3g} < 1e-12 — field excluded, reported")
            continue
        scales[f] = (mu, sd, n_A, True, "included")
    included_fields = [f for f in fields if scales[f][3]]

    probes: dict[str, dict] = {}

    def _finalize(name: str, losses: dict[str, float | None], extra: dict | None = None) -> None:
        lf, l0 = losses.get("full"), losses.get("floor")
        rec: dict = {
            "included": False,
            "reason": "",
            "loss": {k: _round(v) for k, v in losses.items()},
            "u": {},
        }
        if not all(_finite(v) for v in (lf, l0)):
            rec["reason"] = (
                f"full/floor head not defined ({lf!r}/{l0!r}) — probe excluded, reported"
            )
        else:
            headroom = lf - l0
            rec["headroom"] = _round(headroom)
            if headroom <= 0:
                rec["reason"] = (
                    f"headroom {headroom:.6f} <= 0 — probe excluded, reported (degenerate)"
                )
            else:
                rec["included"] = True
                rec["reason"] = "included"
                for arm in ARMS:
                    if _finite(losses.get(arm)):
                        rec["u"][arm] = _round(1.0 - (losses[arm] - lf) / headroom)
        if extra:
            rec.update(extra)
        probes[name] = rec

    def _finite(x) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(x)

    def _reg_losses(
        keys_A: list[str], keys_B: list[str], yA: np.ndarray, yB: np.ndarray
    ) -> dict[str, float | None]:
        """One regression target at fixed capacity across the four arms (A-fit/B-score, MSE)."""
        ia = np.array([idxA[k] for k in keys_A])
        ib = np.array([idxB[k] for k in keys_B])
        out: dict[str, float | None] = {}
        for arm in ARMS:
            p, _r2 = _ridge_oos(AM[arm]["A"][ia], yA, AM[arm]["B"][ib], yB)
            out[arm] = None if p is None else float(np.mean((yB - p) ** 2))
        return out

    # 1. next-type (softmax NLL; classes absent on A < 8 are excluded and reported)
    typesA = [nxt[k]["event_type"] for k in popA if k in nxt]
    typesB = [nxt[k]["event_type"] for k in popB if k in nxt]
    classes = sorted(set(typesA) | set(typesB))
    nA_cls = {c: typesA.count(c) for c in classes}
    kept = [c for c in classes if nA_cls[c] >= 8]
    excl = [c for c in classes if c not in kept]
    incA = [k for k in popA if k in nxt]
    incB = [k for k in popB if k in nxt and nxt[k]["event_type"] in kept]
    if kept and len(incB) >= 8:
        per_arm: dict[str, float | None] = {}
        # floor arm: the A-type frequencies CONDITIONAL on the kept class set — the head
        # normalizes over kept classes only, so the constant predictor must too (rule 5:
        # the comparator is fair, same output space)
        tot = sum(nA_cls[c] for c in kept)
        freq = {c: nA_cls[c] / tot for c in kept}
        nll_f = np.array([-math.log(max(freq[nxt[k]["event_type"]], P_FLOOR)) for k in incB])
        per_arm["floor"] = float(nll_f.mean())
        for arm in ARMS:
            if arm == "floor":
                continue
            ia = np.array([idxA[k] for k in incA])
            ib = np.array([idxB[k] for k in incB])
            tA = np.array([nxt[k]["event_type"] for k in incA])
            tB = np.array([nxt[k]["event_type"] for k in incB])
            scores, ok = [], True
            for c in kept:
                p, _r = _ridge_oos(
                    AM[arm]["A"][ia],
                    (tA == c).astype(float),
                    AM[arm]["B"][ib],
                    (tB == c).astype(float),
                )
                if p is None:
                    ok = False
                    break
                scores.append(p)
            per_arm[arm] = None if not ok else float(_softmax_nll(scores, incB, kept, nxt).mean())
        _finalize(
            "next-type",
            per_arm,
            {
                "classes": classes,
                "classes_included": kept,
                "classes_excluded": excl,
                "per_class_n_A": nA_cls,
                "p_floor": P_FLOOR,
                "n_A": len(incA),
                "n_B_scored": len(incB),
            },
        )
    else:
        _finalize(
            "next-type",
            {},
            {
                "classes": classes,
                "classes_included": kept,
                "classes_excluded": excl,
                "reason_override": "no kept class or <8 B rows — probe excluded, reported",
            },
        )

    # 2. attr-next (per included field; probe loss = mean of the field losses)
    field_losses: dict[str, dict[str, float | None]] = {}
    for f in included_fields:
        mu, sd, _n, _ok, _why = scales[f]

        def _okv(k: str) -> float | None:
            v = nxt.get(k, {}).get(f"f_{f}")
            return v if v is not None and math.isfinite(v) else None

        incA = [k for k in popA if _okv(k) is not None]
        incB = [k for k in popB if _okv(k) is not None]
        if len(incA) < 8 or len(incB) < 8:
            field_losses[f] = {arm: None for arm in ARMS}
            continue
        yA = np.array([(_okv(k) - mu) / sd for k in incA])
        yB = np.array([(_okv(k) - mu) / sd for k in incB])
        field_losses[f] = _reg_losses(incA, incB, yA, yB)
    fl_mean = {
        arm: (
            float(
                np.mean([v for v in (field_losses[f][arm] for f in included_fields) if _finite(v)])
            )
            if any(_finite(field_losses[f][arm]) for f in included_fields)
            else None
        )
        for arm in ARMS
    }
    _finalize(
        "attr-next",
        fl_mean,
        {
            "fields": {
                f: {
                    "included": scales[f][3],
                    "n_A": scales[f][2],
                    "mu": scales[f][0],
                    "sd": scales[f][1],
                    "reason": scales[f][4],
                }
                for f in fields
            },
            "field_losses": {
                f: {k: _round(v) for k, v in row.items()} for f, row in field_losses.items()
            },
        },
    )

    # 3. numeric-next (sum of present-field z, absent = 0; target z on A-included)
    incA = [k for k in popA if k in nxt]
    incB = [k for k in popB if k in nxt]
    if included_fields and len(incA) >= 8 and len(incB) >= 8:
        yA = np.array([_field_z_sum(k, nxt, scales) for k in incA])
        yB = np.array([_field_z_sum(k, nxt, scales) for k in incB])
        mu, sd = float(np.mean(yA)), float(np.std(yA))
        if sd < 1e-12:
            _finalize(
                "numeric-next",
                {},
                {"reason_override": f"no A variance (sd={sd:.3g}) — probe excluded, reported"},
            )
        else:
            zA, zB = (yA - mu) / sd, (yB - mu) / sd
            _finalize(
                "numeric-next",
                _reg_losses(incA, incB, zA, zB),
                {"n_A": len(incA), "n_B_scored": len(incB)},
            )
    else:
        _finalize(
            "numeric-next",
            {},
            {
                "reason_override": "no included field or <8 A/B rows with a next event — probe excluded, reported"
            },
        )

    # 4. dt (log1p seconds; target z on A-included)
    if len(incA) >= 8 and len(incB) >= 8:
        yA = np.array([math.log1p(nxt[k]["dt_s"]) for k in incA])
        yB = np.array([math.log1p(nxt[k]["dt_s"]) for k in incB])
        mu, sd = float(np.mean(yA)), float(np.std(yA))
        if sd < 1e-12:
            _finalize(
                "dt",
                {},
                {"reason_override": f"no A variance (sd={sd:.3g}) — probe excluded, reported"},
            )
        else:
            _finalize(
                "dt",
                _reg_losses(incA, incB, (yA - mu) / sd, (yB - mu) / sd),
                {"n_A": len(incA), "n_B_scored": len(incB)},
            )
    else:
        _finalize(
            "dt",
            {},
            {
                "reason_override": "fewer than 8 A or B rows with a next event — probe excluded, reported"
            },
        )

    # 5. horizon-count-value (H = one seeded draw from the pooled gaps; both z on ALL A)
    if H is not None:
        cntA = np.array([float(win.get(k, {}).get("cnt", 0)) for k in popA])
        cntB = np.array([float(win.get(k, {}).get("cnt", 0)) for k in popB])
        massA = np.array([_window_z_mass(k, win, scales) for k in popA])
        massB = np.array([_window_z_mass(k, win, scales) for k in popB])

        def _two(zA, zB):
            losses = {}
            for arm in ARMS:
                p, _r2 = _ridge_oos(AM[arm]["A"], zA, AM[arm]["B"], zB)
                losses[arm] = None if p is None else float(np.mean((zB - p) ** 2))
            return losses

        def _z(yA, yB):
            mu, sd = float(np.mean(yA)), float(np.std(yA))
            if sd < 1e-12:
                return {arm: None for arm in ARMS}, f"no A variance (sd={sd:.3g})"
            return _two((yA - mu) / sd, (yB - mu) / sd), None

        cnt_losses, cnt_note = _z(cntA, cntB)
        mass_losses, mass_note = _z(massA, massB)
        losses = {
            arm: (
                float(np.mean([v for v in (cnt_losses[arm], mass_losses[arm]) if _finite(v)]))
                if any(_finite(v) for v in (cnt_losses[arm], mass_losses[arm]))
                else None
            )
            for arm in ARMS
        }
        _finalize(
            "horizon-count-value",
            losses,
            {
                "H_seconds": _round(H),
                "n_gaps": int(len(gaps)),
                "gap_p50_s": _round(float(np.quantile(gaps, 0.5))),
                "gap_p95_s": _round(float(np.quantile(gaps, 0.95))),
                "count_losses": {k: _round(v) for k, v in cnt_losses.items() if _finite(v)},
                "mass_losses": {k: _round(v) for k, v in mass_losses.items() if _finite(v)},
                "variance_notes": [n for n in (cnt_note, mass_note) if n],
                "window": f"({as_of_s}, {as_of_s} + {H:.0f}s]",
            },
        )
    else:
        _finalize(
            "horizon-count-value",
            {},
            {"reason_override": "no positive gap in the stream — probe excluded, reported"},
        )

    # ---- U, worst case over the included probes ----
    u_arms: dict[str, float | None] = {}
    worst: dict[str, str | None] = {}
    for arm in ARMS:
        us = [
            (p, d["u"][arm])
            for p, d in probes.items()
            if d["included"] and _finite(d["u"].get(arm))
        ]
        u_arms[arm] = _round(min(v for _p, v in us)) if us else None
        worst[arm] = min(us, key=lambda t: t[1])[0] if us else None
    excluded = {p: d["reason"] for p, d in probes.items() if not d["included"]}

    complete = bool(
        widths_ok
        and match_ok
        and len(popB) > 0
        and all(
            p in probes and "included" in probes[p]
            for p in ("next-type", "attr-next", "numeric-next", "dt", "horizon-count-value")
        )
    )
    body = {
        "kind": "sufficiency",
        "complete": complete,
        "products": products_identity(db),
        "n_fit_A": len(popA),
        "n_score_B": len(popB),
        "arms": {
            arm: {
                "dim": int(AM[arm]["A"].shape[1]),
                "n_A": int(AM[arm]["A"].shape[0]),
                "n_B": int(AM[arm]["B"].shape[0]),
            }
            for arm in ARMS
        },
        "probes": probes,
        "u": {arm: u_arms[arm] for arm in ("served_object", "learned_only")},
        "u_all_arms": {arm: u_arms[arm] for arm in ARMS},
        "worst_probe": {arm: worst[arm] for arm in ("served_object", "learned_only")},
        "excluded_probes": excluded,
        "field_sources": {f: _field_source(f, reg) for f in fields},
        "numeric_fields": list(fields),
        "horizon": {
            "H_seconds": _round(H),
            "n_gaps": int(len(gaps)) if H is not None else 0,
            "gap_p50_s": _round(float(np.quantile(gaps, 0.5))) if H is not None else None,
            "gap_p95_s": _round(float(np.quantile(gaps, 0.95))) if H is not None else None,
            "draw": f"np.random.default_rng({SEED}).choice over the pooled positive within-customer gaps (full stream)",
        },
        "validation": {
            "split_counts": {
                "n_A": len(popA),
                "n_B": len(popB),
                "overlap": len(set(popA) & set(popB)),
            },
            "ledger_widths_ok": widths_ok,
            "ledger_widths_note": widths_note,
            "ledger_parts_byte_identical_to_flat": match_ok,
            "ledger_match_note": match_note,
            "served_dim": int(
                shape_dim + sum(next(iter(part_rows[p].values())).shape[0] for p in order)
            ),
            "arm_alignment_note": (
                "every arm is built over the SAME population (shape ∩ ledger ∩ table ∩ every "
                "published part, per side): served_object drops nothing after alignment, and each "
                "probe's included rows are a subset of that population on every arm"
            ),
            "full_arm_provenance": "docs/report_cards/experiments/path_a_provenance.json — PATH A's table_features drops all 10 fut-stamped columns by its own rule (leak-free)",
            "head_protocol": "the shipping _ridge_head (standardized RidgeCV, one alpha grid) at fixed capacity on every arm; fit on A, scored on B; the battery's honest-OOS guards",
        },
        "honesty": (
            "rule 4: the served object = learned shape ⊕ the ledger's named blocks; learned_only travels in "
            "this same receipt, so a U the ledger alone carries is served-statistic sufficiency, not "
            "learned-shape sufficiency. The full arm is history-only PATH A (leak-free by the provenance "
            "receipt). No criterion, rule or bar of its own: the bar lives in b_spec.SUFFICIENCY_BAR."
        ),
        "seconds": 0.0,
    }
    body["seconds"] = round(time.perf_counter() - t0, 1)
    stamp_merge_write(
        Path(a.out), body, " ".join([Path(__file__).name] + sys.argv[1:]), stamp_blocks=False
    )
    print(f"sufficiency law -> {a.out} ({body['seconds']}s)")
    for p, d in probes.items():
        ls = d.get("loss", {})
        print(
            f"  {p:20s} included={str(d['included']):5s} loss served {ls.get('served_object')} / "
            f"full {ls.get('full')} / floor {ls.get('floor')} / learned {ls.get('learned_only')}  "
            f"u served {d['u'].get('served_object')}  ({d['reason']})"
        )
    print(
        f"  U served_object={body['u']['served_object']}  learned_only={body['u']['learned_only']}  "
        f"worst_probe={body['worst_probe']}  complete={complete}"
    )
    return 0 if complete else 1


def _softmax_nll(
    scores: list[np.ndarray], incB: list[str], kept: list[str], nxt: dict
) -> np.ndarray:
    """Per-B-row NLL of the true next type under the one-vs-rest ridge scores (p-floor frozen)."""
    S = np.column_stack(scores)
    S = S - S.max(1, keepdims=True)
    Pm = np.clip(np.exp(S), P_FLOOR, None)
    Pm /= Pm.sum(1, keepdims=True)
    ki = np.array([kept.index(nxt[k]["event_type"]) for k in incB])
    return -np.log(Pm[np.arange(len(incB)), ki])


def _field_z_sum(k: str, nxt: dict, scales: dict) -> float:
    """numeric-next target for k: sum of present-field z over the included fields (absent = 0)."""
    rec = nxt.get(k)
    if rec is None:
        return 0.0
    out = 0.0
    for f, (mu, sd, _n, _ok, _why) in scales.items():
        if not _ok:
            continue
        v = rec[f"f_{f}"]
        if v is not None and math.isfinite(v):
            out += (v - mu) / sd
    return out


def _window_z_mass(k: str, win: dict, scales: dict) -> float:
    """horizon mass for k: per-event z-sum over the window (absent = 0), from the raw sums."""
    rec = win.get(k)
    if rec is None:
        return 0.0
    out = 0.0
    for f, (mu, sd, _n, _ok, _why) in scales.items():
        if not _ok:
            continue
        raw = rec.get(f"sum_{f}")
        npres = rec.get(f"npres_{f}") or 0
        if raw is not None and npres:
            out += (float(raw) - float(npres) * mu) / sd
    return out


if __name__ == "__main__":
    raise SystemExit(main())
