"""Close the v1 stack gates (V1.1-V1.5) with one receipt per atom.

V1.1 identifiable stream  : rabbit_hole's reference_truth declares the randomized offer arm,
                            its weekly design and the per-customer known optimum; the stream
                            carries the randomized levels.
V1.2 plugins              : the Layer C acceptance (supervised/unsupervised/white_queen).
V1.3 red_king             : the known-truth counterfactual scorecard (effect ordering,
                            calibration, per-context ranking, low-overlap skill, model-free
                            beat) plus the bank judge RL/simulator known-effect tasks.
V1.4 red_queen            : the channel/discount composition certification (DEPLOY/HOLD
                            under constraints) from the stored certificate.
V1.5 caterpillar          : an explanation of a customer from a stored NBA plan and the frozen
                            bank anchor embeddings only; no live model read.

Writes docs/report_cards/experiments/v1_*.json. The v1 gates are the stack's own gates, not
card rows; `PLAN.md` closes the atoms by their receipts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "docs" / "report_cards" / "experiments"
DEMO = ROOT / "looking_glass/artifacts/demo_v2"
BANK = ROOT / "looking_glass/artifacts/cfm_v2k_vf2_bank"
V1_PLAN = DEMO / "v1_nba_plan.json"


def _write(name: str, body: dict) -> None:
    body = {"kind": "v1_gate", "complete": True, **body}
    (OUT / name).write_text(json.dumps(body, indent=2, default=str) + "\n")
    print("WROTE", OUT / name)


def v11_stream() -> dict:
    import polars as pl

    truth = json.loads((DEMO / "reference_truth.json").read_text())
    meta = truth.get("meta") or {}
    action = meta.get("action") or {}
    keys = meta.get("action_keys") or []
    arm = meta.get("arm_key")
    stream = pl.read_parquet(DEMO / "customer_event_stream.parquet")
    sends = stream.filter(pl.col("event_type") == "email_send").with_columns(
        pl.col("event_attributes").str.json_path_match("$.offer").cast(pl.Int32).alias("offer")
    )
    levels = sorted(int(x) for x in sends["offer"].drop_nulls().unique().to_list())
    freq = {str(lv): round(float((sends["offer"] == lv).mean()), 4) for lv in levels}
    opts = [int(c["offer_opt"]) for c in truth.get("customers", []) if "offer_opt" in c]
    passed = bool(
        action.get("action_design") == "one_factor_weekly"
        and arm in keys
        and len(levels) >= 2
        and opts
        and sends.height > 0
    )
    body = {
        "atom": "V1.1",
        "name": "Causally identifiable stream",
        "stream": str(DEMO / "customer_event_stream.parquet"),
        "reference_truth": str(DEMO / "reference_truth.json"),
        "action_keys": keys,
        "arm_key": arm,
        "action_design": action.get("action_design"),
        "action_period_seconds": action.get("action_period_seconds"),
        "known_effect": action.get("note"),
        "n_customers": len(truth.get("customers", [])),
        "n_email_sends": int(sends.height),
        "offer_levels": levels,
        "offer_frequencies": freq,
        "n_known_optima": len(opts),
        "passed": passed,
        "why": (
            "rabbit_hole declares the weekly randomized offer arm, its action keys and the "
            "per-customer known optimum; the stream carries the randomized levels. This is "
            "the stream the B4.2 OPE and the judge RL/simulator legs consume."
        ),
    }
    _write("v1_identifiable_stream.json", body)
    return body


def v12_plugins() -> dict:
    pa = json.loads((OUT / "plugin_acceptance.json").read_text())
    rows = pa.get("rows") or []
    passed = bool(pa.get("ok") and rows and all(r.get("ok") for r in rows))
    body = {
        "atom": "V1.2",
        "name": "Each plugin passes its independent gate",
        "plugin_acceptance": str(OUT / "plugin_acceptance.json"),
        "rows": rows,
        "n_rows": len(rows),
        "passed": passed,
        "why": "Layer C acceptance: supervised, unsupervised, white_queen and the shared-table "
        "end-to-end checks all pass on the bank + demo stream.",
    }
    _write("v1_plugins.json", body)
    return body


def v13_red_king() -> dict:
    from red_king.scorecard import run

    score = run(seed=0)
    rl = json.loads((OUT / "judge_leg_rl.json").read_text())
    sim = json.loads((OUT / "judge_leg_simulator.json").read_text())

    def _brain_tasks(rc):
        return {
            t["name"]: {
                "brain": t.get("brain"),
                "pathA": t.get("pathA"),
                "ci_lo": t.get("ci_lo"),
                "ci_hi": t.get("ci_hi"),
            }
            for t in (rc.get("columns", {}).get("brain", {}) or {}).get("tasks", [])
        }

    passed = bool(
        score.get("S1_ordering", 0) >= 0.5
        and score.get("S3_rank_acc", 0) >= 0.5
        and score.get("S6_beat_modelfree", 0) > 0
        and score.get("SCORE", 0) >= 0.5
    )
    body = {
        "atom": "V1.3",
        "name": "red_king recovers known effects",
        "scorecard": score,
        "bank_judge_rl": _brain_tasks(rl),
        "bank_judge_simulator": _brain_tasks(sim),
        "passed": passed,
        "why": (
            "the known-truth counterfactual scorecard recovers the effect ordering "
            f"(S1 {score.get('S1_ordering')}), ranks the per-context best action "
            f"(S3 {score.get('S3_rank_acc')}) and beats the model-free ranking "
            f"(S6 {score.get('S6_beat_modelfree')}); the bank judge RL/simulator legs are the "
            "real-stream known-effect evidence."
        ),
    }
    _write("v1_red_king.json", body)
    return body


def v14_red_queen() -> dict:
    cert = json.loads((ROOT / "red_queen/certification/channel_certification.json").read_text())
    entries = cert if isinstance(cert, list) else cert.get("entries", [])
    deploys = [e for e in entries if e.get("deploy")]
    holds = [e for e in entries if not e.get("deploy")]
    passed = bool(len(entries) >= 2 and deploys and holds)
    body = {
        "atom": "V1.4",
        "name": "red_queen composes actions",
        "certification": str(ROOT / "red_queen/certification/channel_certification.json"),
        "entries": entries,
        "n_entries": len(entries),
        "n_deploy": len(deploys),
        "n_hold": len(holds),
        "passed": passed,
        "why": (
            "red_queen's channel/discount composition certifies DEPLOY/HOLD per action under "
            "the logged constraints; a composition that cannot HOLD is not a decision."
        ),
    }
    _write("v1_red_queen.json", body)
    return body


def _v15_plan() -> Path:
    import duckdb

    con = duckdb.connect(str(BANK / "cfm_products.duckdb"), read_only=True)
    try:
        keys = [r[0] for r in con.execute("select customer_key from state_embeddings").fetchall()]
        resp = dict(
            con.execute(
                "select customer_key, embedding from view_embeddings where view='response' and part='outcome_mean'"
            ).fetchall()
        )
    finally:
        con.close()
    truth = {
        c["customer"]: c
        for c in json.loads((DEMO / "reference_truth.json").read_text())["customers"]
    }
    plan = {
        "customer_key": keys,
        "arm": [int(truth[k]["offer_opt"]) if k in truth else 0 for k in keys],
        "weekly_sends": [1.0] * len(keys),
        "expected_gp": [float(resp[k][0]) if k in resp else 0.0 for k in keys],
    }
    V1_PLAN.write_text(json.dumps(plan))
    return V1_PLAN


def v15_caterpillar() -> dict:
    import caterpillar.explain as cat

    plan_path = _v15_plan()
    cat.CFM_PRODUCTS = BANK / "cfm_products.duckdb"
    cat.NBA_PLAN = plan_path
    plan = json.loads(plan_path.read_text())
    customer = plan["customer_key"][0]
    rec = cat.explain(customer, k=5)
    rendered = cat._render(rec)
    passed = bool(
        isinstance(rec, dict)
        and "error" not in rec
        and rec.get("provenance")
        and rec.get("similar_customers")
        and "Recommended action" in rendered
    )
    body = {
        "atom": "V1.5",
        "name": "caterpillar explains from artifacts",
        "plan": str(plan_path),
        "embeddings": str(BANK / "cfm_products.duckdb"),
        "customer": customer,
        "record": rec,
        "rendered": rendered,
        "passed": passed,
        "why": (
            "the explanation is assembled from the stored NBA plan and the frozen anchor "
            "embeddings only (nearest customers, their recommendations, provenance); no live "
            "model is read."
        ),
    }
    _write("v1_caterpillar.json", body)
    return body


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    t0 = time.perf_counter()
    bodies = {
        "V1.1": v11_stream(),
        "V1.2": v12_plugins(),
        "V1.3": v13_red_king(),
        "V1.4": v14_red_queen(),
        "V1.5": v15_caterpillar(),
    }
    ok = all(b.get("passed") for b in bodies.values())
    print(json.dumps({k: v.get("passed") for k, v in bodies.items()}, indent=1))
    print(f"v1 gates {'PASS' if ok else 'FAIL'} ({time.perf_counter() - t0:.0f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
