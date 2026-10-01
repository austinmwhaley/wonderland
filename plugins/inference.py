"""Plugin inference (3) — a PURE READ of Layer-B embeddings.

Contract:
  * Layer B materializes `state_embeddings` (donor(h) of every customer's live
    state) once per day via `python -m looking_glass.daily_states --as-of D`.
  * This module NEVER touches state or embeddings: it reads that day's rows,
    scores them with the persisted (encoder-pinned) head, and writes
    `plugin_scores` + an inference receipt into DuckDB.
  * Stale/missing embeddings are REJECTED with the exact fix — never advanced
    here (fail-safe: layer C reads, layer B writes).

  python -m plugins.inference supervised_purchase_propensity_30d --as-of 2025-11-03
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .base import CFM_PRODUCTS, OUT
from .head_template import HeadTemplate
from .targets import REGISTRY, Target

WORK = Path(__file__).resolve().parents[1]
SCORES_DB = WORK / "plugins" / "artifacts" / "scores.duckdb"


def score_as_of(
    target: Target, as_of: str, seed: int = 0, products=CFM_PRODUCTS, scores_db=SCORES_DB
):
    """Score all customers at ``as_of`` from the materialized embeddings (read-only)."""
    import duckdb
    import polars as pl

    from looking_glass.cfm_config import as_of_epoch  # same UTC boundary as the daily job

    t0 = time.perf_counter()
    manifest_p = OUT / f"{target.tag}.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"no artifact for {target.tag}; train the plugin first")
    payload = json.loads(manifest_p.read_text())["payload"]
    enc = payload.get("encoder_version")
    if not enc:
        raise ValueError("head artifact has no encoder pin; retrain to record it")
    head = HeadTemplate.load(payload["head_path"])
    table = target.feature_table
    if table != "donor_embeddings":
        raise NotImplementedError(
            f"live inference needs a state-consistent feature table "
            f"(donor_embeddings = donor(h)); target {target.name} uses {table!r}, whose "
            f"readout cannot be reproduced from a live advanced state"
        )

    day = as_of_epoch(as_of)
    con = duckdb.connect(str(products), read_only=True)
    try:
        # pin check: training embeddings must be the encoder the head learned on
        vers = [r[0] for r in con.execute(f"SELECT DISTINCT version FROM {table}").fetchall()]
        if vers != [enc]:
            raise ValueError(
                f"encoder pin mismatch: head={enc!r} products={vers!r} — retrain the head "
                f"(a plugin head never outlives its encoder tag)"
            )
        # the day's inference embeddings must already exist (Layer-B daily job)
        has_table = int(
            con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'state_embeddings'"
            ).fetchone()[0]
        )
        if not has_table:
            raise ValueError(
                f"state_embeddings table missing from products db — run the Layer-B daily "
                f"state job first: python -m looking_glass.daily_states --as-of {as_of}"
            )
        n_day = int(
            con.execute(
                "SELECT count(*) FROM state_embeddings WHERE as_of_epoch = ?",
                [float(day)],
            ).fetchone()[0]
            or 0
        )
        if n_day == 0:
            raise ValueError(
                f"state_embeddings has no rows for as_of={as_of} — run the Layer-B daily "
                f"state job first: python -m looking_glass.daily_states --as-of {as_of}"
            )
        day_versions = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT version FROM state_embeddings WHERE as_of_epoch = ?",
                [float(day)],
            ).fetchall()
        ]
        if day_versions != [enc]:
            raise ValueError(
                f"state_embeddings version for {as_of} = {day_versions!r} does not "
                f"match head pin {enc!r}"
            )
        n_expected = int(con.execute("SELECT count(*) FROM customer_state").fetchone()[0])
        if n_day < n_expected:
            raise ValueError(
                f"state_embeddings for {as_of} incomplete: {n_day}/{n_expected} customers — "
                f"re-run the daily state job (materialize covers every state)"
            )
        rows = con.execute(
            "SELECT customer_key, embedding FROM state_embeddings WHERE as_of_epoch = ?",
            [float(day)],
        ).fetchall()
    finally:
        con.close()

    keys = [r[0] for r in rows]
    X = np.stack([np.asarray(r[1], dtype=np.float32) for r in rows])
    scores = np.asarray(HeadTemplate.predict(head, X), dtype=np.float64)

    now = datetime.now(timezone.utc)
    day_date = datetime.fromisoformat(as_of).date()
    top = np.sort(scores)[-max(1, int(0.1 * len(scores))) :]

    # serving-calibration gate: compare the served mean to the head's own
    # held-out base rate with a DERIVED tolerance (3*binomial-SE + the head's
    # held-out calibration gap). Nothing used to read inference output back:
    # the Dec rehearsal served mean_score 0.516 vs base_rate 0.435 (+8.1pt,
    # >25 SE) while every training gate passed.
    heads = {h["name"]: h.get("metrics", {}) for h in payload.get("heads", [])}
    winner = heads.get(payload.get("head_name"), {})
    ref = winner.get("base_rate") or payload.get("dataset", {}).get("y_pos_rate")
    cal_gap = float(winner.get("calibration_gap") or 0.0)
    n = len(keys)
    score_gap = score_z = score_tol = None
    calib_ok = True
    if ref is not None and 0.0 < float(ref) < 1.0:
        ref = float(ref)
        se = math.sqrt(ref * (1.0 - ref) / max(n, 1))
        score_tol = 3.0 * se + cal_gap
        score_gap = float(scores.mean()) - ref
        score_z = score_gap / se if se > 0 else None
        calib_ok = abs(score_gap) <= score_tol
    head_id = f"{payload.get('head_name')}@{Path(payload['head_path']).name}"

    receipt = {
        "as_of": as_of,
        "plugin_tag": target.tag,
        "encoder_version": enc,
        "head_id": head_id,
        "source": "state_embeddings (read-only)",
        "n_scored": len(keys),
        "mean_score": float(scores.mean()),
        "top_decile_mean": float(top.mean()),
        "base_rate_ref": ref,
        "score_gap": score_gap,
        "score_gap_z": score_z,
        "score_tolerance": score_tol,
        "score_calib_ok": calib_ok,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "scored_at": now.isoformat(),
    }

    Path(scores_db).parent.mkdir(parents=True, exist_ok=True)
    sc = duckdb.connect(str(scores_db))
    try:
        sc.execute(
            "CREATE TABLE IF NOT EXISTS plugin_scores (customer_key TEXT, as_of DATE, "
            "plugin_tag TEXT, encoder_version TEXT, score DOUBLE, scored_at TIMESTAMP, "
            "head_id TEXT)"
        )
        sc.execute(
            "CREATE TABLE IF NOT EXISTS inference_receipts (as_of DATE, plugin_tag TEXT, "
            "encoder_version TEXT, n_scored BIGINT, n_new_events BIGINT, n_idle BIGINT, "
            "mean_score DOUBLE, top_decile_mean DOUBLE, wall_seconds DOUBLE, "
            "scored_at TIMESTAMP, head_id TEXT)"
        )
        # legacy DBs predate head_id — additive schema, idempotent
        sc.execute("ALTER TABLE plugin_scores ADD COLUMN IF NOT EXISTS head_id TEXT")
        sc.execute("ALTER TABLE inference_receipts ADD COLUMN IF NOT EXISTS head_id TEXT")
        # idempotent per (plugin, day): a rerun REPLACES the day instead of
        # stacking duplicate copies (200k duplicate key-groups existed)
        sc.execute(
            "DELETE FROM plugin_scores WHERE plugin_tag = ? AND as_of = ?",
            [target.tag, day_date],
        )
        sc.execute(
            "DELETE FROM inference_receipts WHERE plugin_tag = ? AND as_of = ?",
            [target.tag, day_date],
        )
        df_s = pl.DataFrame(
            {
                "customer_key": keys,
                "as_of": pl.Series([day_date] * len(keys), dtype=pl.Date),
                "plugin_tag": [target.tag] * len(keys),
                "encoder_version": [enc] * len(keys),
                "score": scores,
                "scored_at": [now] * len(keys),
                "head_id": [head_id] * len(keys),
            }
        )
        sc.register("_s", df_s)
        try:
            sc.execute("INSERT INTO plugin_scores SELECT * FROM _s")
        finally:
            sc.unregister("_s")
        # event/idle counts are the DAILY JOB's receipts now (kept NULL here)
        sc.execute(
            "INSERT INTO inference_receipts "
            "(as_of, plugin_tag, encoder_version, n_scored, mean_score, top_decile_mean, "
            "wall_seconds, scored_at, head_id) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                day_date,
                target.tag,
                enc,
                receipt["n_scored"],
                receipt["mean_score"],
                receipt["top_decile_mean"],
                receipt["wall_seconds"],
                now,
                head_id,
            ],
        )
    finally:
        sc.close()

    print(f"== INFERENCE {target.tag} as_of={as_of} (read-only) ==")
    for k, v in receipt.items():
        print(f"  {k:18s}: {v}")
    if not calib_ok:
        print(
            f"  WARNING: served mean is {score_gap:+.4f} from the held-out base rate "
            f"(z={score_z:.1f}, tol={score_tol:.4f}) — serving-calibration drift; "
            f"check population shift / retrain",
            flush=True,
        )
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score customers at a date (read-only)")
    ap.add_argument("target", choices=sorted(REGISTRY))
    ap.add_argument("--as-of", required=True, help="ISO date; Layer-B embeddings must exist for it")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    score_as_of(REGISTRY[a.target], a.as_of, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
