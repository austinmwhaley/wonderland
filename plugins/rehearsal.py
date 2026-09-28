"""Full production rehearsal: the monthly/weekly operating loop, end to end.

The runbook, executable (per calendar month):

  Day 1 (as-of D):
    (0) LADDER       if this as_of was never laddered: size Sample A first
                     (smallest rung within noise of best downstream AUC)
    (1) ENCODER      train_cfm(events <= D) on SAMPLE A (receipt-sized; the
                     split re-randomizes monthly; warm-starts from the previous
                     compatible checkpoint when one exists)
                     -> new tag vN.rM; build_products writes the sample-B
                        embedding tables + every customer's state at D
    (1b) DAILY STATE Layer B closes day D: absorbs day D's events into
         JOB        customer_state, then materializes state_embeddings (donor(h))
                     at D  — the ONLY writer of state/embeddings
    (2) PLUGIN       per supervised plugin: read the frozen sample-B table,
                     close labels at D, fit/gate/persist head pinned to vN.rM
    (3) INFERENCE    training day counts as a run; read-only score from
                     state_embeddings -> DuckDB
  For --days days from the 1st (default 8 = month-start + the next 7 days;
  pass 31 for the full month):
    (1b) daily state job (GPU absorbs) — states + embeddings advance one day
    (3)  on the plugin's weekday (Monday) + one mid-week day: read-only score
  Next 1st: new A/B rotation, new tag, refreshed tables, plugin re-pinned.

Emits a timeline receipt (plugins/artifacts/rehearsal_<start>.json).
Long-running (encoder training dominates) — run in the background.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from .inference import score_as_of
from .ladder import run as run_ladder
from .ladder_sample_a import chosen_rung as encoder_sample_rung
from .ladder_sample_a import run as run_encoder_ladder
from .targets import PURCHASE_PROPENSITY_30D

WORK = Path(__file__).resolve().parents[1]
OUT = WORK / "plugins" / "artifacts"
CFM_DIR = WORK / "looking_glass" / "artifacts" / "cfm"
STREAM_DB = "rabbit_hole/data/duckdb/customer_event_stream.duckdb"


def _mondays(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() == 0:
            days.append(d)
        d += timedelta(days=1)
    return days


def _inference_days(start: date, days: int, midweek: date | None = None) -> list[date]:
    """Training day + Mondays within the --days window (+ midweek if inside)."""
    end = start + timedelta(days=days - 1)
    out = {start, *_mondays(start, end)}
    if midweek and start <= midweek <= end:
        out.add(midweek)
    return sorted(out)


def _ab_signature(tag: str) -> dict:
    """Proof of monthly re-randomization: A/B counts + membership hash."""
    import duckdb

    con = duckdb.connect(str(CFM_DIR / "cfm_products.duckdb"), read_only=True)
    try:
        rows = con.execute(
            "SELECT split, customer_key FROM encoder_samples ORDER BY split, customer_key"
        ).fetchall()
    finally:
        con.close()
    counts = {"A": 0, "B": 0}
    for s, _ in rows:
        counts[s] = counts.get(s, 0) + 1
    h = hashlib.sha1("".join(f"{s}:{k}" for s, k in rows).encode()).hexdigest()[:12]
    return {"tag": tag, "counts": counts, "membership_sha": h}


def train_encoder(as_of: str, customers: int, anchors: int, log: Path) -> str:
    cmd = [
        sys.executable,
        "-m",
        "looking_glass.customer_foundation_model",
        "train",
        "--customers",
        str(customers),
        "--anchors",
        str(anchors),
        "--as-of",
        as_of,
        "--db",
        STREAM_DB,
        "--out-dir",
        str(CFM_DIR),
    ]
    t0 = time.perf_counter()
    print(
        f"[encoder] {as_of}: training on sample A (as-of cutoff, re-randomized split)", flush=True
    )
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORK))
    log.write_text((r.stdout or "") + "\n" + (r.stderr or ""))
    if r.returncode != 0:
        raise RuntimeError(f"encoder training failed (see {log}): {r.stderr[-500:]}")
    m = re.search(r"trained (\S+):", r.stdout or "")
    if not m:
        raise RuntimeError(f"could not parse encoder tag from output (see {log})")
    tag = m.group(1)
    print(f"[encoder] {as_of}: tag={tag} in {time.perf_counter() - t0:.1f}s", flush=True)
    return tag


def cycle(as_of: date, days: int, customers: int, anchors: int, seed: int, inference_days) -> dict:
    from looking_glass.daily_states import update_day

    from .head_template import run_target

    day = as_of.isoformat()
    # daily-job window: [as_of, window_end); inference days must fall inside it
    next_first = (as_of.replace(day=28) + timedelta(days=4)).replace(day=1)
    window_end = min(next_first, as_of + timedelta(days=days))
    inference_days = {d for d in inference_days if d < window_end}
    print(
        f"\n=== CYCLE as-of {day} (window {as_of}..{window_end - timedelta(days=1)}) ===",
        flush=True,
    )

    # (0) size Sample A first when this as_of has never been laddered —
    # the encoder trains on Sample A (chosen rung), never on all of population A
    if encoder_sample_rung(day) is None:
        print(f"[ladder] {day}: sizing Sample A first (no receipt for this as_of)", flush=True)
        run_encoder_ladder(day, customers=customers, anchors=anchors, skip_done=True)

    # (1) encoder (sample-sized via the receipt; warm-starts from the previous
    # compatible month when available)
    tag = train_encoder(day, customers, anchors, OUT / f"rehearsal_encoder_{day}.log")
    ab = _ab_signature(tag)

    print(f"[ladder] {day}: choosing the training-size rung", flush=True)
    ladder = run_ladder(PURCHASE_PROPENSITY_30D, as_of=day, seed=seed)

    # (2) per-plugin training, labels closed at D, pinned to the new tag
    print(f"[plugin] {day}: training {PURCHASE_PROPENSITY_30D.tag} pinned to {tag}", flush=True)
    ok, payload = run_target(PURCHASE_PROPENSITY_30D, seed=seed, as_of=day)
    primary = next(h for h in payload["heads"] if h["name"] == payload["head_name"])

    # (3) training day counts as an inference day (read-only) — day-1
    # states/embeddings were materialized by (1) itself
    inferences = [score_as_of(PURCHASE_PROPENSITY_30D, day)]
    n_daily = 0

    # daily chain over the --days window: layer B closes each day (GPU absorbs),
    # layer C reads on its weekday(s)
    d = as_of + timedelta(days=1)
    while d < window_end:
        update_day(d.isoformat())
        n_daily += 1
        if d in inference_days:
            inferences.append(score_as_of(PURCHASE_PROPENSITY_30D, d.isoformat()))
        d += timedelta(days=1)

    return {
        "as_of": day,
        "encoder_tag": tag,
        "ab": ab,
        "daily_jobs": n_daily,
        "day1_materialized_by_encoder": True,
        "ladder": {
            "chosen_n_train": ladder["chosen_n_train"],
            "best": ladder["best"],
            "tol": ladder["tol"],
            "falloff_n_train": ladder["falloff_n_train"],
        },
        "plugin": {
            "tag": PURCHASE_PROPENSITY_30D.tag,
            "verdict": bool(ok),
            "encoder_version": payload["encoder_version"],
            "primary_head": payload["head_name"],
            "metrics": primary["metrics"],
        },
        "inferences": inferences,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Monthly/weekly production rehearsal")
    ap.add_argument("--start", default="2025-11-01", help="first cycle date (the 1st)")
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--anchors", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--midweek", default="2025-11-05", help="extra non-Monday inference day")
    ap.add_argument(
        "--days",
        type=int,
        default=8,
        help="daily-state days per cycle from the 1st: 8 = month-start + the next "
        "7 days (default); 31 = full month",
    )
    a = ap.parse_args(argv)

    start = date.fromisoformat(a.start)
    nxt = (start.replace(day=28) + timedelta(days=4)).replace(day=1)  # next 1st
    mid = date.fromisoformat(a.midweek) if a.midweek else None
    c1_days = _inference_days(start, a.days, mid)
    c2_days = _inference_days(nxt, a.days, mid)

    t0 = time.perf_counter()
    print(
        f"REHEARSAL {start} + {a.days} days/cycle: daily state jobs daily; "
        f"inference days cycle1={c1_days} cycle2={c2_days}",
        flush=True,
    )
    c1 = cycle(start, a.days, a.customers, a.anchors, a.seed, set(c1_days))
    c2 = cycle(nxt, a.days, a.customers, a.anchors, a.seed, set(c2_days))

    receipt = {
        "start": start.isoformat(),
        "next_cycle": nxt.isoformat(),
        "customers": a.customers,
        "anchors": a.anchors,
        "architecture": {
            "state_tables": "customer_state + state_embeddings (written only by looking_glass.daily_states)",
            "plugin_training": "reads frozen sample_B tables; labels closed at as_of",
            "inference": "read-only over state_embeddings",
        },
        "cycles": [c1, c2],
        "split_rotated": c1["ab"]["membership_sha"] != c2["ab"]["membership_sha"],
        "daily_jobs_total": c1["daily_jobs"] + c2["daily_jobs"],
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"rehearsal_{start.isoformat()}.json"
    path.write_text(json.dumps(receipt, indent=1, default=str))

    def _line(c):
        return (
            f"  cycle {c['as_of']}: encoder={c['encoder_tag']} "
            f"plugin={'PASS' if c['plugin']['verdict'] else 'FAIL'} "
            f"auc={c['plugin']['metrics']['auc']:.3f} "
            f"chosen_rung={c['ladder']['chosen_n_train']} "
            f"daily_jobs={c['daily_jobs']} inferences={len(c['inferences'])}"
        )

    print("\n== REHEARSAL SUMMARY ==")
    print(_line(c1))
    print(_line(c2))
    print(f"  split rotated month-over-month: {receipt['split_rotated']}")
    print(f"  daily jobs total: {receipt['daily_jobs_total']}")
    print(f"  wall {receipt['wall_seconds']}s")
    print(f"receipt -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
