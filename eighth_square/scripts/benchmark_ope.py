"""Ground-truth OPE benchmark across environments.

Runs the whole-pool offline-RL + OPE survey per environment, scores each
environment's decisions against LIVE ground-truth returns through the bench
harness (FP/FN/precision/recall + Wilson intervals), and writes a report.

Usage: python benchmark_ope.py [env ...]
Writes /tmp/opencode/wq_matrix/benchmark.json and prints the summary.
"""
import json
import os
import sys

sys.path.insert(0, "/home/austin-whaley/wq")
sys.path.insert(0, "/home/austin-whaley/wq/scripts")

from white_queen.tribunal.bench import Case, Cell, summarize

OUT = "/tmp/opencode/wq_matrix"
DISCRETE = ["cartpole", "mountaincar", "acrobot", "lunar"]
CONTINUOUS = ["mountaincar_continuous", "pendulum"]


def _case_from_matrix(r):
    cells = []
    rank = r.get("rank") or list(r["candidates"])
    order = {n: i + 1 for i, n in enumerate(rank)}
    for n, c in r["candidates"].items():
        cells.append(Cell(candidate=n, deploy=bool(c["deploy"]),
                          truth=float(c["truth"]),
                          anchor=float(r["behavior_mean"]), bar=float(r["bar"]),
                          rank=order.get(n, 999)))
    return Case(r["env"], cells, meta={"type": "discrete", "N": r["N"]})


def _case_from_continuous(r):
    return Case(r["env"], [Cell(candidate="iql_cont", deploy=bool(r["deploy"]),
                                truth=float(r["iql_truth"]),
                                anchor=float(r["behavior_mean"]),
                                bar=float(r["bar"]), rank=1)],
                meta={"type": "continuous", "N": r["N"]})


if __name__ == "__main__":
    import survey_matrix as SM
    import survey_continuous as SC

    want = sys.argv[1:]
    cases = []
    for e in (want or DISCRETE):
        if e in CONTINUOUS:
            continue
        print(f"[bench] {e} ...", flush=True)
        cases.append(_case_from_matrix(SM.survey(e)))
    for e in (want or CONTINUOUS):
        if e not in CONTINUOUS:
            continue
        print(f"[bench] {e} ...", flush=True)
        cases.append(_case_from_continuous(SC.survey_env(e)))

    report = summarize(cases)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "benchmark.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("\n== OPE GROUND-TRUTH BENCHMARK ==")
    for p in report["per_case"]:
        print(f"  {p['case']:24s} tp={p['tp']} fp={p['fp']} fn={p['fn']} "
              f"tn={p['tn']} rank_rho={p['rank_rho']} "
              f"FP={p['false_positive']} FN={p['false_negative']}")
    ov = report["overall"]
    print(f"  OVERALL precision={ov['precision']} {ov['precision_ci']} "
          f"recall={ov['recall']} {ov['recall_ci']} "
          f"wrong_deploys={ov['n_deploy_wrong']} missed={ov['n_missed']}")
