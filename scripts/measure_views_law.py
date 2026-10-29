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
import math
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


def _load_response(db: Path) -> dict | None:
    """The fitted `response` head + its calibration receipt, as the artifact carries them."""
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    try:
        row = con.execute("select stamp from response_head limit 1").fetchone()
    except Exception:  # noqa: BLE001 - a pre-DEC-426 artifact simply has no such table
        row = None
    con.close()
    return json.loads(row[0]) if row else None


def check_response_outlet(head: dict | None, faceplate: dict) -> dict:
    """DEC-426's three gates for the `response` outlet, frozen in DECISIONS.md before the run.

    The outlet is the only view whose content is a FITTED head rather than encoder geometry, so
    it owns its own contract on top of the shared per-view gates:

    1. **It predicts.** Its held-out predictions beat the two comparators a conditional-outlet
       claim must beat — the fitted population constant ("I know nothing about this customer")
       and a destroyed version of itself (same marginals, customer alignment permuted) — with
       the paired CI excluding zero.
    2. **It is actually action-conditioned.** The state+action head must beat the
       **action-blind** head (state only) and the **state-blind** head (the action's transition
       only). If either matches it, the outlet is a main effect wearing the name "response".
    3. **Its interval means what it says.** Empirical coverage of the nominal band within the
       derived tolerance (2x the binomial SE at the evaluated N), and coverage on rows whose arm
       was NOT randomized may not be better than on randomized rows unless σ was widened there.

    An outlet the producer never served is NOT a failure here: it is advertised in the
    faceplate's `not_served` (the `measure` precedent, DEC-383), and the row says which.
    """
    if not head:
        unserved = "response" in (faceplate.get("not_served") or {})
        return {
            "working": False,
            "evidence": "response outlet NOT served"
            + (
                " — advertised as not_served by the faceplate (the stream declared no action "
                "design, so there is nothing to condition on)"
                if unserved
                else " — and the faceplate does not advertise it either: the artifact is stale"
            ),
            "served": False,
        }
    problems = []
    ph = head.get("_pred_holdout") or {}
    need = ("state_action", "truth", "action_blind", "state_blind")
    if not all(k in ph for k in need):
        return {
            "working": False,
            "evidence": f"response head published without its predictions ({need}) — the gate "
            f"cannot be computed from the artifact, so the claim is UNKNOWN",
            "served": True,
        }
    y = np.asarray(ph["truth"], dtype=np.float64)
    full = np.asarray(ph["state_action"], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    comparators = {
        "population constant": np.full_like(y, float(head.get("target_mean_fit", y.mean()))),
        "destroyed": rng.permutation(full),
        "action-blind (state only)": np.asarray(ph["action_blind"], dtype=np.float64),
        "state-blind (action only)": np.asarray(ph["state_blind"], dtype=np.float64),
    }
    cis = {}
    for name, pred in comparators.items():
        _, lo, _ = _paired_ci((y - pred) ** 2 - (y - full) ** 2)
        cis[name] = float(lo)
        if not lo > 0:
            problems.append(f"{name} not beaten (paired CI lo={lo:+.4g})")
    # 3. calibration, at the N actually evaluated
    n = int(head.get("n_calibrate") or len(y))
    nominal = float(head.get("coverage_nominal", 0.80))
    got = float(head.get("coverage_empirical", 0.0))
    se = math.sqrt(max(nominal * (1 - nominal), 1e-12) / max(n, 1))
    band = 2.0 * se  # DEC-426: +/- the binomial SE at the evaluated N, x2 tolerance
    if got < nominal - band:
        problems.append(f"coverage {got:.3f} below nominal {nominal:.3f} - band {band:.3f}")
    by = head.get("coverage_by_randomization") or {}
    r, u = by.get("randomized"), by.get("unrandomized")
    # each group's OWN claim is checked against its OWN measured coverage: one global number can
    # hide a group that misses badly while the other over-covers
    for tag, grp in (("randomized", r), ("unrandomized", u)):
        if grp and float(grp["coverage"]) < nominal - band:
            problems.append(
                f"{tag} rows: coverage {float(grp['coverage']):.3f} below nominal "
                f"{nominal:.3f} - band {band:.3f} (n={grp.get('n')})"
            )
    if r and u:
        # DEC-384b is an ORDERING property of the published widths, not of a multiplier: the
        # rows whose arm was never logged as randomized may not carry the tighter interval.
        # Both widths are measured on their own rows, so a violation means the data said so and
        # the floor had to be applied — which the receipt is required to admit.
        s_r, s_u = float(head.get("sigma_randomized", 0)), float(head.get("sigma_unrandomized", 0))
        if s_u < s_r:
            problems.append(
                f"unrandomized sigma {s_u:g} narrower than randomized {s_r:g} (DEC-384b)"
            )
        if u["coverage"] > r["coverage"] + band and not head.get("sigma_forced_wider_unrandomized"):
            problems.append(
                f"unrandomized coverage {u['coverage']:.3f} > randomized {r['coverage']:.3f} "
                f"without widened sigma (DEC-384b)"
            )
    # a claim without its provenance is not a claim: what was treated as an action, and over
    # what derived horizon, must travel with the columns
    for k, what in (
        ("action_design", "which events were treated as actions"),
        ("gamma_derivation", "the derived discount/horizon"),
    ):
        if not head.get(k):
            problems.append(f"receipt missing {k} ({what})")
    return {
        "working": not problems,
        "evidence": (
            f"conditional outcome head: holdout R2={head.get('target_r2_holdout')} on "
            f"n={n} B-rows (LEAKY ceiling {head.get('target_r2_ceiling_leaky')} — the most any "
            f"head could predict from this target), coverage {got:.3f} (nominal {nominal:.2f}, "
            f"band +/-{band:.3f}), randomized fraction {head.get('randomized_frac')}, measured "
            f"sigma {head.get('sigma_randomized')}/{head.get('sigma_unrandomized')} "
            f"(rand/unrand, mean interval width {head.get('mean_interval_width')}"
            + (
                ", unrandomized floor forced wider"
                if head.get("sigma_forced_wider_unrandomized")
                else ""
            )
            + f"), delta incremental R2={head.get('delta_incremental_r2')} "
            f"(effective rank delta={head.get('delta_effective_rank')} "
            f"state={head.get('state_effective_rank')}), actions from "
            f"{head.get('action_design', {}).get('source', '?')}"
            + (" | " + "; ".join(problems) if problems else "")
        ),
        "served": True,
        "paired_ci_lo": {k: round(v, 6) for k, v in cis.items()},
        "coverage_by_randomization": by,
    }


def check_ledger_attribution(views: dict, truth: dict, akeys, bkeys, faceplate: dict) -> dict:
    """G5 (SPINE §5): a served ledger must come with brain / ledger / both + a destroyed control.

    The rule this enforces is the operator's 2026-10-27 corollary: **the brain earns the win,
    the ledger prevents the loss.** The ledger is mechanically what PATH A's table already is,
    so a number it carries can never be booked as a B-win. The row therefore does NOT gate on
    the ledger beating the brain — it gates on the four arms being MEASURED and reported, and
    on the ledger at least beating its own destroyed control (a ledger that cannot beat a
    column-permuted copy of itself is not carrying customer information at all).

    Frozen before the run (DEC-427): `working` = the measure view is served, every reported
    target carries all four finite arms, and `paired_ci_lo(ledger vs destroyed) > 0` on at least
    one target. Changing what this measures after seeing a number is a new protocol version.
    """
    if "measure" not in views:
        unserved = "measure" in (faceplate.get("not_served") or {})
        return {
            "working": False,
            "evidence": "ledger (measure view) NOT served"
            + (
                " — advertised as not_served by the faceplate"
                if unserved
                else " — and the faceplate does not advertise it either: the artifact is stale"
            )
            + "; produce it with the `products` command",
            "served": False,
        }
    usable, _excluded = _target_matrix(truth, akeys, bkeys)
    brain_parts = [v for v in V.FULL_ORDER if v in views]
    if not usable or not brain_parts:
        return {
            "working": False,
            "evidence": f"cannot attribute: targets={sorted(usable)} brain views={brain_parts}",
            "served": True,
        }

    # `_matrix` takes a {customer: vector} map, so the concatenations are built explicitly
    def _cat(view_names, keys):
        mats = []
        for vn in view_names:
            m = _matrix(views[vn], keys)
            if m is None:
                return None
            mats.append(m)
        return np.hstack(mats)

    brain_a, brain_b = _cat(brain_parts, akeys), _cat(brain_parts, bkeys)
    led_a, led_b = _matrix(views["measure"], akeys), _matrix(views["measure"], bkeys)
    if any(x is None for x in (brain_a, brain_b, led_a, led_b)):
        return {
            "working": False,
            "evidence": "attribution unmeasurable: a view is ragged or missing rows (see outlet_schema)",
            "served": True,
        }
    both_a, both_b = np.hstack([brain_a, led_a]), np.hstack([brain_b, led_b])
    rng = np.random.default_rng(SEED)
    _rand, destroyed_b = _controls(rng, led_b)
    arms, adds_any, real_any = {}, False, False
    for name, (ya, yb) in usable.items():
        p_brain, r2_brain = _ridge_oos(brain_a, ya, brain_b, yb)
        p_led, r2_led = _ridge_oos(led_a, ya, led_b, yb)
        p_both, r2_both = _ridge_oos(both_a, ya, both_b, yb)
        p_dest, r2_dest = _ridge_oos(led_a, ya, destroyed_b, yb)
        if any(p is None for p in (p_brain, p_led, p_both, p_dest)):
            continue
        _, lo_add, _ = _paired_ci((yb - p_brain) ** 2 - (yb - p_both) ** 2)
        _, lo_real, _ = _paired_ci((yb - p_dest) ** 2 - (yb - p_led) ** 2)
        arms[name] = {
            "learned_only": round(float(r2_brain), 4),
            "ledger_only": round(float(r2_led), 4),
            "both": round(float(r2_both), 4),
            "ledger_destroyed": round(float(r2_dest), 4),
            "ci_lo_ledger_adds": round(float(lo_add), 6),
            "ci_lo_ledger_vs_destroyed": round(float(lo_real), 6),
        }
        adds_any = adds_any or lo_add > 0
        real_any = real_any or (lo_real > 0 and r2_led > 0)
    if not arms:
        return {
            "working": False,
            "evidence": "no target could be attributed (all degenerate) — see the excluded list",
            "served": True,
        }
    best = max(arms.items(), key=lambda kv: kv[1]["ledger_only"])
    return {
        # the frozen rule: the arms must all be measured AND the ledger must beat its own
        # destroyed control somewhere. It is NOT required to beat the brain.
        "working": bool(real_any) and bool(arms),
        "evidence": (
            f"attribution measured on {len(arms)} trait(s); best ledger-only R2="
            f"{best[1]['ledger_only']} on '{best[0]}' (its destroyed control "
            f"{best[1]['ledger_destroyed']}, learned-only {best[1]['learned_only']}, both "
            f"{best[1]['both']}); ledger beats destroyed on "
            f"{sum(1 for v in arms.values() if v['ci_lo_ledger_vs_destroyed'] > 0)}/{len(arms)} "
            f"and adds beyond the brain on "
            f"{sum(1 for v in arms.values() if v['ci_lo_ledger_adds'] > 0)}/{len(arms)} "
            f"— a ledger-backed number is never booked as a brain win (SPINE §5)"
            + ("" if real_any else " | WARNING: the ledger never beats its destroyed control")
        ),
        "served": True,
        "arms": arms,
        "ledger_carries_signal": bool(real_any),
        "ledger_adds_beyond_brain": bool(adds_any),
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
    # every run of this producer happens inside the project's one memory ceiling (DEC-440):
    # a direct launch is re-entered into the slice, so the boundary does not depend on the
    # caller remembering to use the launcher.
    from safe_run import ensure_boundary

    ensure_boundary()

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
        # DEC-426: the response outlet owns a second contract because its content is a fitted
        # head, not encoder geometry — decodable traits alone cannot show it is conditional on
        # the action, nor that its published uncertainty covers.
        "response_outlet": check_response_outlet(_load_response(db), faceplate),
        # DEC-427 / SPINE §5 G5: a served ledger must be attributable. This is also the receipt
        # the products-time G1-G5 audit leaves pending, so the loop closes here.
        "ledger_attribution": check_ledger_attribution(views, truth, akeys, bkeys, faceplate),
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
