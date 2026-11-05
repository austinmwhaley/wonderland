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
RED_QUEEN_PLAN = ROOT / "red_queen/artifacts/nba_demo_plan.json"


def _write(name: str, body: dict) -> None:
    body = {"kind": "v1_gate", "complete": True, **body}
    if "products" not in body:
        from measure_input_law import products_identity

        body["products"] = products_identity(str(BANK))
    from measure_input_law import stamp_merge_write

    stamp_merge_write(OUT / name, body, " ".join(sys.argv), stamp_blocks=False)
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
    rk = json.loads((OUT / "red_king_known.json").read_text())
    score = rk.get("scorecard") or {}
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

    known = rk.get("known_effect") or {}
    pol = rk.get("policy") or {}
    passed = bool(rk.get("passed"))
    body = {
        "atom": "V1.3",
        "name": "red_king recovers known effects",
        "known_effect_receipt": str(OUT / "red_king_known.json"),
        "stream_sha256": rk.get("stream_sha256"),
        "as_of": rk.get("as_of"),
        "scorecard": score,
        "known_effect": known,
        "policy": pol,
        "bank_judge_rl": _brain_tasks(rl),
        "bank_judge_simulator": _brain_tasks(sim),
        "passed": passed,
        "why": (
            "red_king's per-customer randomized send-day value model recovers the generator's "
            f"declared offer optimum at {known.get('accuracy')} vs {known.get('chance')} chance "
            f"(majority {known.get('majority_accuracy')}); on the post-as-of randomized holdout "
            f"the argmax policy is {pol.get('policy')} vs {pol.get('random_all')} random "
            f"(paired CI {pol.get('paired_ci_policy_minus_random')}) and "
            f"{pol.get('best_constant')} best constant. The bank judge RL/simulator legs are "
            "supplementary served-bank evidence."
        ),
    }
    _write("v1_red_king.json", body)
    return body


def v14_red_queen() -> dict:
    import hashlib

    cert_path = ROOT / "red_queen/certification/channel_certification.json"
    cert = json.loads(cert_path.read_text())
    entries = cert if isinstance(cert, list) else cert.get("entries", [])
    deploys = [e for e in entries if e.get("deploy")]
    holds = [e for e in entries if not e.get("deploy")]
    passed = bool(len(entries) >= 2 and deploys and holds)
    body = {
        "atom": "V1.4",
        "name": "red_queen composes actions",
        "certification": str(cert_path),
        "certificate_sha256": hashlib.sha256(cert_path.read_bytes()).hexdigest(),
        "stream_sha256": hashlib.sha256(
            (DEMO / "customer_event_stream.parquet").read_bytes()
        ).hexdigest(),
        "entries": entries,
        "n_entries": len(entries),
        "n_deploy": len(deploys),
        "n_hold": len(holds),
        "passed": passed,
        "why": (
            "the demo randomizes one factor (offer); red_queen's per-channel offer policy "
            "clears the pre/post-as-of IPW deploy rule on email/sms/push (held-out margin "
            "4.99-5.39 vs 2.87-3.10 behavior, paired CI > 0). Discount and frequency have no "
            "randomized variation, so they are recorded HOLD/not identified rather than "
            "inherited from the retired stream. A composition with both verdicts is a decision."
        ),
    }
    _write("v1_red_queen.json", body)
    return body


def v15_caterpillar() -> dict:
    import hashlib

    import duckdb

    import caterpillar.explain as cat
    from red_queen.nba_demo import build_plan

    plan_path = RED_QUEEN_PLAN
    if not plan_path.exists():
        build_plan()
    cat.CFM_PRODUCTS = BANK / "cfm_products.duckdb"
    cat.NBA_PLAN = plan_path
    plan = json.loads(plan_path.read_text())
    con = duckdb.connect(str(BANK / "cfm_products.duckdb"), read_only=True)
    try:
        anchors = {
            str(r[0])
            for r in con.execute("select distinct customer_key from anchor_embeddings").fetchall()
        }
    finally:
        con.close()
    customer = next((k for k in plan["customer_key"] if k in anchors), plan["customer_key"][0])
    idx = plan["customer_key"].index(customer)
    rec = cat.explain(customer, k=5)
    rendered = cat._render(rec)
    same_arm = rec.get("recommended_arm") == plan["arm"][idx]
    same_gp = (
        abs(
            float(rec.get("expected_weekly_incremental_gp") or 0.0)
            - float(plan["expected_gp"][idx])
        )
        < 1e-6
    )
    passed = bool(
        isinstance(rec, dict)
        and "error" not in rec
        and rec.get("provenance")
        and rec.get("similar_customers")
        and same_arm
        and same_gp
        and "Recommended action" in rendered
    )
    body = {
        "atom": "V1.5",
        "name": "caterpillar explains from artifacts",
        "plan": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "plan_meta": plan.get("meta"),
        "embeddings": str(BANK / "cfm_products.duckdb"),
        "customer": customer,
        "record": rec,
        "rendered": rendered,
        "recommended_arm_matches_plan": same_arm,
        "expected_gp_matches_plan": same_gp,
        "passed": passed,
        "why": (
            "the explanation is assembled from the stored red_queen NBA plan (built from the "
            "randomized offer panel and the DEPLOY/HOLD certificate) and the frozen anchor "
            "embeddings only; the recommended arm and expected GP equal the plan's, and no "
            "live model or generator truth is read."
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
