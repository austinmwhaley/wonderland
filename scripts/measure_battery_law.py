#!/usr/bin/env python3
"""S5 measurement: the objective battery on the frozen donor (PATH B consumers).

Three content-class probes were missing (SPINE §8 step 4, DEC-387); each is judged
brain-vs-naive with a paired bootstrap CI, on the HOLDOUT (split-B) customers only:

  1 ledger_reconstruction   ("how much")  — a LINEAR head on the served donor embedding
        reconstructs the exact stream ledger (spend, margin, purchases, returns);
        naive comparator = the same head on frequency-only features (bag size, recency,
        purchase/return counts). A state that lost level/mass must lose here.
  2 distance_regression     ("what it's like") — embedding distance must track the
        LAB TRUTH latent distance (generator x0/h0/d0/price_sens/return_prone/touch_pref);
        naive comparator = bag-of-event-types distance (pure frequency geometry).
        The raw (pre-boundary) state is reported as a diagnostic control.
  3 action_conditioned_outcome ("what if") — the lab randomizes one action key per day
        (one_factor_daily) with a KNOWN effect: purchase propensity scales with
        distance-to-optimum on the offer arm. Per-customer causal uplift
        P(buy|offer=opt) - P(buy|offer off by >=2) is identified from the stream; a
        linear head on E must predict that uplift better than the population constant.
  4 prune_pass              (deletion receipt) — per-objective weights are recorded with
        a trajectory; selection must be the frozen held-out-SSL rule; any objective whose
        weight decays to ~zero is listed for deletion (empty list today, disclosed).

Heads are fitted on split A and scored on split B (no same-customer games). Bootstrap
counts, seeds and identifiability floors are EVALUATION CONTRACT: fixed and recorded
(DEC-410 — the contract column, not tuned knobs).

Writes (merges) S5 into docs/report_cards/experiments/judge_stages.json.

    .venv/bin/python scripts/measure_battery_law.py \
        --products /tmp/opencode/cfm_s3b/cfm_products.duckdb \
        --demo /tmp/opencode/wonderland_demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_input_law import _rollup, stamp_merge_write  # noqa: E402

SEED = 65537  # principle-ok: evaluation-contract seed (frozen, recorded; DEC-410)
N_BOOT = 2000  # principle-ok: evaluation-contract bootstrap count
N_PAIRS = 20_000  # principle-ok: evaluation-contract pair sample
MIN_ARM_DAYS = 10  # principle-ok: evaluation-contract identifiability floor for uplift
PRUNE_EPS = 1e-2  # principle-ok: deletion threshold — weight ~0 means no gradient contribution
LATENT_COLS = ("x0", "h0", "d0", "price_sensitivity", "return_proneness")


# --------------------------------------------------------------------------- data


def _load_all(products: str, demo: Path):
    import duckdb

    reg_path = sorted(Path(products).parent.glob("registry_*.json"))[-1]
    reg = json.loads(reg_path.read_text())
    truth = {
        c["customer"]: c
        for c in json.loads((demo / "reference_truth.json").read_text())["customers"]
    }
    con = duckdb.connect(products, read_only=True)
    bkeys = [
        r[0]
        for r in con.execute("select customer_key from encoder_samples where split='B'").fetchall()
    ]
    akeys = [
        r[0]
        for r in con.execute("select customer_key from encoder_samples where split='A'").fetchall()
    ]
    emb = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in con.execute("select customer_key, embedding from state_embeddings").fetchall()
    }
    raw_state = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in con.execute("select customer_key, state from customer_state").fetchall()
    }
    con.close()
    manifest = json.loads((demo / "manifest.json").read_text())
    as_of = np.datetime64(manifest["as_of"])
    stream = pl.read_parquet(demo / "path_b" / "path_b_customer_event_stream.parquet").filter(
        pl.col("event_ts").cast(pl.Datetime) <= as_of
    )
    return reg, truth, emb, raw_state, stream, bkeys, akeys, as_of


def _ridge_head():
    """THE shipping head: standardized ridge with a self-selecting alpha grid.

    One definition on purpose (principle 6) — the battery probes, the view laws and every judge
    leg score through it, and the judge's whole claim is that a *dumb consumer* can read the donor
    (DEC-407). Callers that need the fitted object (e.g. to query the head at unobserved actions)
    build it here rather than re-typing the pipeline.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))


def _ridge_oos(Ea, ya, Eb, yb):
    """Fit ridge head on split A, score split B (honest out-of-sample). sklearn defaults."""
    if len(Ea) < 8 or float(np.std(ya)) < 1e-12 or float(np.std(yb)) < 1e-12:
        return None, None
    m = _ridge_head()
    m.fit(Ea, ya)
    p = m.predict(Eb)
    ss = float(np.sum((yb - yb.mean()) ** 2))
    return p, float(1.0 - np.sum((yb - p) ** 2) / ss)


def _paired_ci(deltas, n_boot=N_BOOT):
    """Percentile CI of the mean of paired per-unit advantages (positive = brain wins)."""
    rng = np.random.default_rng(SEED)
    d = np.asarray(deltas, dtype=np.float64)
    n = len(d)
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# --------------------------------------------------------------------------- probes


def check_ledger_reconstruction(stream, as_of, emb, akeys, bkeys):
    """'How much': the exact ledger, recomputed from the raw stream by DuckDB, must be
    linearly recoverable from the served donor — and beat the frequency-only head."""
    t0 = time.perf_counter()
    import duckdb

    con = duckdb.connect()
    con.register("s", stream)
    led = con.execute(
        """
        select customer_key,
               coalesce(sum(try_cast(json_extract_string(event_attributes,'$.price') as double)
                            * try_cast(json_extract_string(event_attributes,'$.qty') as double)), 0.0)::double as spend,
               coalesce(sum(try_cast(json_extract_string(event_attributes,'$.margin') as double)), 0.0)::double as margin,
               sum(case when event_type='purchase' then 1 else 0 end)::bigint as n_purch,
               sum(case when event_type='return' then 1 else 0 end)::bigint as n_ret,
               count(*)::bigint as n_events,
               date_diff('day', max(cast(event_ts as date)), cast(? as date))::bigint as days_since_last
        from s group by 1
        """,
        [as_of.astype(str)],
    ).pl()
    con.close()
    fmap = {r["customer_key"]: r for r in led.to_dicts()}

    def _y(keys, col, log=True):
        v = []
        for k in keys:
            r = fmap.get(k)
            x = 0.0 if r is None else float(r[col] or 0.0)
            v.append(np.log1p(max(x, 0.0)) if log else x)
        return np.asarray(v)

    ak = [k for k in akeys if k in emb]
    bk = [k for k in bkeys if k in emb]
    Ea, Eb = np.stack([emb[k] for k in ak]), np.stack([emb[k] for k in bk])

    def _naive_X(keys, exclude_type):
        """Frequency-only comparator: log1p counts of EVERY event type (minus the type
        that DEFINES the target — counting purchases to predict purchase count is the
        target itself, not a comparator) plus recency."""
        cols = [t for t in _BAG_TYPES if t != exclude_type]
        X = []
        for k in keys:
            bag = _BAGS.get(k)
            rec = float((fmap.get(k) or {}).get("days_since_last") or 0)
            if bag is None:
                X.append([0.0] * (len(cols) + 1))
            else:
                X.append([np.log1p(bag[_BAG_IDX[t]]) for t in cols] + [rec])
        return np.asarray(X)

    rows, excluded, fails = [], [], []
    ok = True
    scored = 0
    for col, excl in (
        ("spend", None),
        ("margin", None),
        ("n_purch", "purchase"),
        ("n_ret", "return"),
    ):
        ya, yb = _y(ak, col), _y(bk, col)
        if float(np.std(yb)) < 1e-9:
            excluded.append(col)
            continue
        Na, Nb = _naive_X(ak, excl), _naive_X(bk, excl)
        pb, r2b = _ridge_oos(Ea, ya, Eb, yb)
        pn, r2n = _ridge_oos(Na, ya, Nb, yb)
        if pb is None or pn is None:
            fails.append(col)
            ok = False
            continue
        _d, lo, hi = _paired_ci((yb - pn) ** 2 - (yb - pb) ** 2)
        scored += 1
        ok &= lo > 0 and r2b > 0
        rows.append(
            f"{col}: brain R2={r2b:.3f} naive R2={r2n:.3f} MSE-gain 95%CI[{lo:.3f},{hi:.3f}]"
        )
    ok = ok and scored > 0 and not fails
    return {
        "working": ok,
        "evidence": (
            f"ledger recomputed from the raw stream by DuckDB (not a table copy); LINEAR head on the "
            f"learned-shape donor (embedding = model.proj(h), no served measure blocks — the claim is "
            f"learned-only), fitted on {len(ak)} split-A customers and scored on the {len(bk)} holdout "
            f"B; naive comparator = same head on the full event-type bag + recency (target-defining type "
            f"excluded — no label leakage into the comparator); "
            + " | ".join(rows)
            + (f" | excluded(no variance): {excluded}" if excluded else "")
            + (f" | FIT FAILURES: {fails}" if fails else "")
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


_BAGS = None  # customer_key -> event-type count vector, filled by main()
_BAG_TYPES: list[str] = []
_BAG_IDX: dict[str, int] = {}


def check_distance_regression(truth, emb, raw_state, bkeys):
    """'What it's like': donor distance must track the lab-truth latent distance —
    and beat the bag-of-event-types distance (pure frequency geometry)."""
    t0 = time.perf_counter()
    keys = [k for k in bkeys if k in emb and k in truth]
    if len(keys) < 30:
        return {
            "working": False,
            "evidence": f"only {len(keys)} B customers with embeddings+truth",
            "seconds": 0.0,
        }
    if _BAGS is None or not all(k in _BAGS for k in keys):
        return {
            "working": False,
            "evidence": "bag counts unavailable (stream not loaded)",
            "seconds": 0.0,
        }
    Z = np.array([[truth[k][f] for f in LATENT_COLS] + list(truth[k]["touch_pref"]) for k in keys])
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    E = np.stack([emb[k] for k in keys])
    Es = (E - E.mean(0)) / (E.std(0) + 1e-12)
    Bg = np.stack([_BAGS[k] for k in keys])
    S = np.stack([raw_state[k] for k in keys]) if all(k in raw_state for k in keys) else None
    n = len(keys)
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(SEED)
    npairs = min(N_PAIRS, len(iu[0]))
    sel = rng.choice(len(iu[0]), size=npairs, replace=False)
    ii, jj = iu[0][sel], iu[1][sel]
    dt = np.sqrt(((Z[ii] - Z[jj]) ** 2).sum(1))
    de = np.sqrt(((Es[ii] - Es[jj]) ** 2).sum(1))
    dn = np.abs(Bg[ii] - Bg[jj]).sum(1)
    rho_b = float(spearmanr(de, dt).statistic)
    rho_n = float(spearmanr(dn, dt).statistic)
    rho_s = (
        float(spearmanr(np.sqrt(((S[ii] - S[jj]) ** 2).sum(1)), dt).statistic)
        if S is not None
        else float("nan")
    )
    # paired bootstrap over pairs: rho(brain) - rho(naive), common resample
    rb, rn = rankdata(de), rankdata(dn)
    rt = rankdata(dt)

    def _rho_from_ranks(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    deltas = []
    for _ in range(N_BOOT):
        s = rng.integers(0, npairs, size=npairs)
        ta, ea, na = rt[s], rb[s], rn[s]
        deltas.append(_rho_from_ranks(ea, ta) - _rho_from_ranks(na, ta))
    d, lo, hi = _paired_ci(np.asarray(deltas))
    ok = rho_b > rho_n and lo > 0 and rho_b > 0
    return {
        "working": ok,
        "evidence": (
            f"lab-truth latent distance (x0/h0/d0/price_sens/return_prone/touch_pref, z-scored) vs "
            f"distance over {npairs} customer pairs from {n} holdout B customers: donor Spearman "
            f"{rho_b:.3f} vs naive bag-of-event-types {rho_n:.3f}; paired-over-pairs gain 95%CI "
            f"[{lo:.3f},{hi:.3f}] (gate lo>0); diagnostic raw-state(pre-boundary) Spearman {rho_s:.3f}"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def arm_cells(stream, truth) -> "pl.DataFrame":
    """(customer_key, offer level, days, buys, declared optimum) for the randomized offer arm.

    ONE DuckDB pass, aggregated columnar to one row per (customer, level). The battery's uplift
    probe (S5) and the judge's `rl` and `simulator` legs (DEC-434) both consume this: the uplift is
    identified by RANDOMIZATION, so a second copy of this SQL would be a second chance for the two
    to disagree about what "identified" means — the drift principle 6 exists to prevent.
    """
    import duckdb

    con = duckdb.connect()
    con.register("s", stream)
    con.register(
        "o",
        pl.DataFrame(
            {"customer_key": list(truth.keys()), "opt": [truth[k]["offer_opt"] for k in truth]}
        ),
    )
    cells = con.execute(
        """
        with arms as (
            select customer_key, cast(event_ts as date) as d,
                   try_cast(json_extract_string(event_attributes,'$.offer') as integer) as off
            from s
            where event_type='email_send'
              and json_extract_string(event_attributes,'$.target_key')='offer'
        ),
        buys as (
            select distinct customer_key, cast(event_ts as date) as d
            from s where event_type='purchase'
        )
        select a.customer_key, a.off, o.opt, count(*) as days,
               sum(case when b.d is not null then 1 else 0 end) as buys
        from arms a join o using (customer_key) left join buys b
          on a.customer_key = b.customer_key and a.d = b.d
        group by 1, 2, 3
        """
    ).pl()
    con.close()
    return cells


def controls_matched(rng, X_fit, X_score):
    """Random and destroyed controls with the ARM's OWN per-column scale, fitted on split A.

    Why this exists next to `measure_views_law._controls` rather than replacing it: that one
    draws noise at a single GLOBAL sigma. For a donor whose columns share a scale that is fine;
    for the `brain + ledger` column, where the measure view puts raw event counts beside
    fractions, the same sigma makes some control columns 10^6x too wide — and the control then
    scores R2 = -4.6e15, failing for an arithmetic reason that has nothing to do with the
    information it was supposed to lack. A negative control has to fail for the RIGHT reason or
    it is not evidence (DEC-433 makes an unreadable control block a PASS, so the artifact has
    teeth). `_controls` stays untouched: the S1-S6 battery and the view laws publish committed
    evidence through it, and silently rewriting those numbers under a refactor banner is a
    re-measurement decision, not a cleanup.

    `destroyed` keeps the column-permutation definition (same marginals, no customer identity);
    `random` is drawn per column at A's mean and spread.
    """
    X_fit = np.asarray(X_fit, dtype=np.float64)
    mu, sd = X_fit.mean(0), X_fit.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    rand = rng.normal(loc=mu, scale=sd, size=np.asarray(X_score).shape)
    destroyed = np.column_stack([rng.permutation(X_score[:, j]) for j in range(X_score.shape[1])])
    return rand, destroyed


def identified_uplift(cells, min_days: int = MIN_ARM_DAYS) -> dict:
    """Per-customer causal uplift P(buy|offer=opt) − P(buy|offer off by ≥2), where identified.

    A customer appears only with `min_days` arm-days on BOTH sides: an under-sampled contrast is
    not a noisy estimate, it is an unidentified one, and reporting it as the former would let a
    handful of days carry a causal claim. Columnar aggregation, then a lookup dict — the event
    scan already happened in SQL, so this loop is over (customer × level) aggregates.
    """
    sides = (
        cells.with_columns(
            pl.col("days").cast(pl.Int64),
            pl.col("buys").cast(pl.Int64),
            (pl.col("off") == pl.col("opt")).cast(pl.Int64).alias("good"),
            ((pl.col("off") - pl.col("opt")).abs() >= 2).cast(pl.Int64).alias("bad"),
        )
        .group_by("customer_key")
        .agg(
            (pl.col("days") * pl.col("good")).sum().alias("good_days"),
            (pl.col("buys") * pl.col("good")).sum().alias("good_buys"),
            (pl.col("days") * pl.col("bad")).sum().alias("bad_days"),
            (pl.col("buys") * pl.col("bad")).sum().alias("bad_buys"),
        )
    )
    return {
        r["customer_key"]: r["good_buys"] / r["good_days"] - r["bad_buys"] / r["bad_days"]
        for r in sides.iter_rows(named=True)
        if r["good_days"] >= min_days and r["bad_days"] >= min_days
    }


def identified_dose_response(cells, min_days: int = MIN_ARM_DAYS) -> dict:
    """Per-customer identified purchase rate AT EACH offer level: {key: {level: (rate, days)}}.

    The counterfactual object the `simulator` leg grades (DEC-434): recovering the built-in effect
    needs the response at every arm level, not only the optimum-vs-bad contrast that
    `identified_uplift` collapses it to. Levels with fewer than `min_days` arm-days are absent —
    unidentified, not zero.
    """
    ok = cells.with_columns(pl.col("days").cast(pl.Int64), pl.col("buys").cast(pl.Int64)).filter(
        pl.col("days") >= min_days
    )
    out: dict = {}
    for r in ok.iter_rows(named=True):
        out.setdefault(r["customer_key"], {})[int(r["off"])] = (
            r["buys"] / r["days"],
            int(r["days"]),
        )
    return out


def check_action_conditioned_outcome(stream, truth, emb, akeys, bkeys):
    """'What if': the lab's one-factor-daily randomized offer arm has the KNOWN effect
    q *= (1 - 0.22*|offer-offer_opt|). Per-customer uplift P(buy|arm=opt) -
    P(buy|arm off by >=2) is causally identified (randomized arm); a linear head on the
    donor must decode it better than the population constant."""
    t0 = time.perf_counter()
    up = identified_uplift(arm_cells(stream, truth))
    bk = [k for k in bkeys if k in up and k in emb]
    ak = [k for k in akeys if k in up and k in emb]
    if len(bk) < 30 or len(ak) < 30:
        return {
            "working": False,
            "evidence": f"insufficient identified customers (A={len(ak)}, B={len(bk)}; >= {MIN_ARM_DAYS} arm-days/side required)",
            "seconds": round(time.perf_counter() - t0, 1),
        }
    y = np.array([up[k] for k in bk])
    ya = np.array([up[k] for k in ak])
    Eb = np.stack([emb[k] for k in bk])
    Ea = np.stack([emb[k] for k in ak])
    pb, r2b = _ridge_oos(Ea, ya, Eb, y)
    if pb is None:
        return {
            "working": False,
            "evidence": "degenerate uplift target (no variance)",
            "seconds": round(time.perf_counter() - t0, 1),
        }
    pn = np.full_like(y, float(ya.mean()))
    r2n = float(1.0 - np.sum((y - pn) ** 2) / np.sum((y - y.mean()) ** 2))
    _d, lo, hi = _paired_ci((y - pn) ** 2 - (y - pb) ** 2)
    rho = float(spearmanr(pb, y).statistic)
    ok = lo > 0 and r2b > max(r2n, 0.0)
    return {
        "working": ok,
        "evidence": (
            f"known-effect uplift from the one-factor-daily randomized offer arm "
            f"(P(buy|offer=opt) - P(buy|off>=2), >= {MIN_ARM_DAYS} arm-days/side): {len(bk)} holdout B "
            f"customers identified from {len(bkeys)}, head fitted on {len(ak)} A customers; brain OOS "
            f"R2={r2b:.3f} vs population-constant naive R2={r2n:.3f}; Spearman(brain,truth)={rho:.3f}; "
            f"paired MSE-gain 95%CI [{lo:.5f},{hi:.5f}] (gate lo>0)"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def check_prune_pass(reg):
    """Deletion receipt: per-objective weights recorded with a trajectory; selection is
    the frozen held-out-SSL rule; any weight decayed to ~0 is listed for deletion."""
    t0 = time.perf_counter()
    tw = reg.get("task_weights") or {}
    traj = reg.get("weight_trajectory") or []
    sel = str((reg.get("resolved") or {}).get("selection", ""))
    dead = sorted([k for k, v in tw.items() if v <= PRUNE_EPS])
    ok = bool(tw) and bool(traj) and "heldout" in sel
    if not tw:
        return {
            "working": False,
            "evidence": "no task_weights recorded in the registry",
            "seconds": 0.0,
        }
    deletion = (
        str(dead)
        if dead
        else "EMPTY — equal-weight mode never decays a task to zero (disclosed: vacuous today, "
        "enforced the day a learned mixture is adopted)"
    )
    return {
        "working": ok,
        "evidence": (
            f"per-objective weights recorded: {len(tw)} objectives, min={min(tw.values()):.4g}, "
            f"trajectory snapshots={len(traj)}, selection rule='{sel}' (frozen held-out-SSL); "
            f"weights <= {PRUNE_EPS:g} -> DELETE list: {deletion}"
        ),
        "seconds": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------- main


def _build_bags(stream, keys):
    """Bag-of-event-types counts per customer — the naive frequency-only geometry."""
    global _BAGS, _BAG_TYPES, _BAG_IDX
    kf = pl.DataFrame({"customer_key": list(dict.fromkeys(keys))})
    wide = (
        stream.join(kf, on="customer_key")
        .group_by(["customer_key", "event_type"])
        .len()
        .pivot(on="event_type", index="customer_key", values="len")
    )
    _BAG_TYPES = sorted(c for c in wide.columns if c != "customer_key")
    _BAG_IDX = {t: i for i, t in enumerate(_BAG_TYPES)}
    wide = kf.join(wide, on="customer_key", how="left").with_columns(
        pl.col(_BAG_TYPES).fill_null(0)
    )
    arr = wide.select(_BAG_TYPES).to_numpy().astype(np.float64)
    _BAGS = dict(zip(wide["customer_key"].to_list(), arr, strict=True))


def main() -> int:
    # every run of this producer happens inside the project's one memory ceiling (DEC-440):
    # a direct launch is re-entered into the slice, so the boundary does not depend on the
    # caller remembering to use the launcher.
    from safe_run import ensure_boundary

    ensure_boundary()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="/tmp/opencode/cfm_s3b/cfm_products.duckdb")
    ap.add_argument("--demo", default="/tmp/opencode/wonderland_demo")
    ap.add_argument("--out", default=str(ROOT / "docs/report_cards/experiments/judge_stages.json"))
    a = ap.parse_args()

    reg, truth, emb, raw_state, stream, bkeys, akeys, as_of = _load_all(a.products, Path(a.demo))
    _build_bags(stream, list(akeys) + list(bkeys))

    checks = {
        "ledger_reconstruction": check_ledger_reconstruction(stream, as_of, emb, akeys, bkeys),
        "distance_regression": check_distance_regression(truth, emb, raw_state, bkeys),
        "action_conditioned_outcome": check_action_conditioned_outcome(
            stream, truth, emb, akeys, bkeys
        ),
        "prune_pass": check_prune_pass(reg),
    }
    existing = stamp_merge_write(
        Path(a.out),
        {"S5": _rollup(checks)},
        " ".join([Path(__file__).name] + sys.argv[1:]),
    )
    print(f"judge stages -> {a.out}")
    for name, c in checks.items():
        print(f"  S5 :: {name}: {'PASS' if c['working'] else 'FAIL'}  ({c['seconds']}s)")
    print(f"S5 working={existing['S5']['working']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
