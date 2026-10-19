"""Ground-truth OPE benchmark across environments.

Runs the whole-pool offline-RL + OPE survey per environment, scores each
environment's decisions against LIVE ground-truth returns through the bench
harness (FP/FN/precision/recall + Wilson intervals), and writes a report.

Usage: python benchmark_ope.py [env ...]
Writes /tmp/opencode/white_queen_matrix/benchmark.json and prints the summary.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from white_queen.tribunal.bench import Case, Cell, summarize

OUT = "/tmp/opencode/white_queen_matrix"
DISCRETE = ["cartpole", "mountaincar", "acrobot", "lunar"]
CONTINUOUS = ["mountaincar_continuous", "pendulum"]


def _case_from_matrix(r):
    cells = []
    rank = r.get("rank") or list(r["candidates"])
    order = {n: i + 1 for i, n in enumerate(rank)}
    for n, c in r["candidates"].items():
        _cert = (r.get("certificates") or {}).get(n) or {}
        cells.append(
            Cell(
                candidate=n,
                deploy=bool(c["deploy"]),
                truth=float(c["truth"]),
                anchor=float(r["behavior_mean"]),
                bar=float(r["bar"]),
                rank=order.get(n, 999),
                lo=_cert.get("lo"),
                hi=_cert.get("hi"),
            )
        )
    return Case(r["env"], cells, meta={"type": "discrete", "family": "discrete", "N": r["N"]})


def _case_from_continuous(r):
    return Case(
        r["env"],
        [
            Cell(
                candidate="iql_cont",
                deploy=bool(r["deploy"]),
                truth=float(r["iql_truth"]),
                anchor=float(r["behavior_mean"]),
                bar=float(r["bar"]),
                rank=1,
                lo=r.get("cert_lo"),
                hi=r.get("cert_hi"),
            )
        ],
        meta={"type": "continuous", "family": "continuous", "N": r["N"]},
    )


if __name__ == "__main__":
    import survey_matrix as SM
    import survey_continuous as SC

    want = sys.argv[1:]
    cases = []
    for e in want or DISCRETE:
        if e in CONTINUOUS:
            continue
        print(f"[bench] {e} ...", flush=True)
        cases.append(_case_from_matrix(SM.survey(e)))
    for e in want or CONTINUOUS:
        if e not in CONTINUOUS:
            continue
        print(f"[bench] {e} ...", flush=True)
        cases.append(_case_from_continuous(SC.survey_env(e)))
    if not want:
        from white_queen.tribunal.bench.scorecard import _bandit_case

        print("[bench] bandit_synth ...", flush=True)
        cases.append(_bandit_case(0))

    report = summarize(cases)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "benchmark.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    # ACCEPTANCE RECEIPT with provenance (the hard gate the card consumes).
    from white_queen.tribunal.bench.acceptance import (
        CONTRACT,
        REPORT_ONLY_DIMS,
        rows_from_cases,
    )

    rows, _agg = rows_from_cases(cases, seeds_used=CONTRACT["seeds_min"])
    hard = [r for r in rows if r[0] not in REPORT_ONLY_DIMS]
    report_only = {
        f"{r[0]}/{r[1]}": {"target": r[2], "achieved": r[3]}
        for r in rows
        if r[0] in REPORT_ONLY_DIMS
    }
    acc = {
        "passed": int(sum(1 for r in hard if r[4])),
        "total": int(len(hard)),
        "report_only": report_only,
        "baseline": "logging policy + model-free DR",
        "beats_traditional": bool(hard and all(r[4] for r in hard)),
        "overall": report["overall"],
        "rows": [
            {
                "dimension": r[0],
                "metric": r[1],
                "target": r[2],
                "achieved": r[3],
                "ok": bool(r[4]),
            }
            for r in rows
        ],
        "provenance": {
            "git_sha": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
            ).stdout.strip(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "envs": list(want) if want else DISCRETE + CONTINUOUS + ["bandit_synth"],
        },
    }
    with open(os.path.join(OUT, "acceptance.json"), "w") as f:
        json.dump(acc, f, indent=1, default=str)
    print("\n== OPE GROUND-TRUTH BENCHMARK ==")
    for p in report["per_case"]:
        print(
            f"  {p['case']:24s} tp={p['tp']} fp={p['fp']} fn={p['fn']} "
            f"tn={p['tn']} rank_rho={p['rank_rho']} "
            f"FP={p['false_positive']} FN={p['false_negative']}"
        )
    ov = report["overall"]
    print(
        f"  OVERALL precision={ov['precision']} {ov['precision_ci']} "
        f"recall={ov['recall']} {ov['recall_ci']} "
        f"wrong_deploys={ov['n_deploy_wrong']} missed={ov['n_missed']}"
    )
