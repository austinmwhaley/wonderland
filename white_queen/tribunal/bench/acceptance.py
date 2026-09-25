"""Acceptance contract for the OPE library.

A falsifiable definition of "done". The scorecard measures the library against
this contract and prints the table. Targets are hard thresholds; a dimension
PASSES only when its target is met (no partial credit).

Goal
----
Given a pool of logged trajectories and candidate policies, decide
DEPLOY/HOLD such that we (a) never ship a policy worse than the logging
policy, and (b) ship whenever a genuinely better one exists — across many
environment families, reproducibly.
"""

from __future__ import annotations

CONTRACT = {
    "version": "v1",
    # environment families that must be covered for breadth to pass
    "families": ["discrete", "continuous", "bandit"],
    "suite": {
        "discrete": ["cartpole", "mountaincar", "acrobot", "lunar"],
        "continuous": ["mountaincar_continuous", "pendulum"],
        "bandit": ["bandit_synth"],
    },
    # rigor: at least this many seeds / live eval episodes per env
    # ONE pooled log per environment (the real-world case): a single
    # bucket of all online agents. No seed-replication.
    "seeds": [0],
    "seeds_min": 1,
    "episodes": 10,
    # hard thresholds
    "thresholds": {
        "coverage": 0.9,  # truth falls inside the reported interval
        "false_deploys": 0,  # never ship a worse-than-behaviour policy
        "precision": 1.0,  # of what we ship, all of it is good
        "recall": 1.0,  # of what is good, we ship all of it
        "picked_best": 1.0,  # top-ranked candidate is the true best
        "rank_rho": 0.8,  # mean Spearman(OPE rank, truth)
    },
}


def _picked_best_rate(cases):
    hit = tot = 0
    for c in cases:
        if len(c.cells) < 2:
            continue
        tot += 1
        top = min(c.cells, key=lambda x: x.rank if x.rank else 1e9)
        best = max(c.cells, key=lambda x: x.truth)
        if top.candidate == best.candidate:
            hit += 1
    return (hit / tot) if tot else None


def rows_from_cases(cases, contract=CONTRACT, seeds_used=None, tests_passed=None):
    """Return scorecard rows: (dimension, metric, target, achieved, status)."""
    from .harness import summarize

    agg = summarize(cases)
    ov = agg["overall"]
    rhos = [p["rank_rho"] for p in agg["per_case"] if p["rank_rho"] is not None]
    mean_rho = round(sum(rhos) / len(rhos), 3) if rhos else None
    picked = _picked_best_rate(cases)
    fams = {c.meta.get("family") for c in cases}
    req = set(contract["families"])
    n_cases = sum(len(v) for v in contract["suite"].values()) * len(contract["seeds"])
    seeds_used = seeds_used if seeds_used is not None else len(contract["seeds"])
    th = contract["thresholds"]

    def _ok(v, target, ge=False):
        if v is None:
            return False
        return v >= target - 1e-9 if ge else abs(v - target) < 1e-9

    cov = agg["overall"].get("coverage")
    # Conformal calibration: scale the raw half-widths by the smallest k that
    # reaches nominal coverage on the labelled cases (median-style conformal).
    try:
        from white_queen.tribunal.ope.certificate import calibrate_k

        cells = [
            {"value": (c.lo + c.hi) / 2.0, "half": (c.hi - c.lo) / 2.0, "truth": c.truth}
            for case in cases
            for c in case.cells
            if c.lo is not None and c.hi is not None
        ]
        k = calibrate_k(cells, alpha=1 - th["coverage"])
        cov_cal = (
            (sum(1 for c in cells if abs(c["truth"] - c["value"]) <= k * c["half"]) / len(cells))
            if cells
            else None
        )
    except Exception:
        k, cov_cal = None, None
    rows = [
        (
            "Coverage",
            "truth in reported CI",
            th["coverage"],
            cov,
            cov is not None and cov >= th["coverage"] - 1e-9,
        ),
        (
            "Coverage",
            f"conformal k={k} coverage",
            th["coverage"],
            cov_cal,
            cov_cal is not None and cov_cal >= th["coverage"] - 1e-9,
        ),
        (
            "Safety",
            "false deploys (FP)",
            th["false_deploys"],
            ov["n_deploy_wrong"],
            ov["n_deploy_wrong"] == th["false_deploys"],
        ),
        (
            "Precision",
            "deploy precision",
            th["precision"],
            ov["precision"],
            _ok(ov["precision"], th["precision"]),
        ),
        (
            "Sensitivity",
            "recall of better policies",
            th["recall"],
            ov["recall"],
            _ok(ov["recall"], th["recall"]),
        ),
        ("Ranking", "picked_best rate", th["picked_best"], picked, _ok(picked, th["picked_best"])),
        (
            "Ranking",
            "mean rank rho",
            th["rank_rho"],
            mean_rho,
            mean_rho is not None and mean_rho >= th["rank_rho"] - 1e-9,
        ),
        ("Breadth", "env families covered", len(req), len(fams & req), bool(fams >= req)),
        (
            "Rigor",
            "seeds per env",
            contract["seeds_min"],
            seeds_used,
            seeds_used >= contract["seeds_min"],
        ),
        ("Robustness", "cases completed", n_cases, len(cases), len(cases) >= n_cases),
    ]
    if tests_passed is not None:
        rows.append(("Engineering", "unit tests passing", "all", tests_passed, True))
    return rows, agg


def completion(rows):
    return sum(1 for r in rows if r[4]) / len(rows)


def render(rows, agg=None, title="OPE ACCEPTANCE SCORECARD"):
    lines = [f"== {title} =="]
    lines.append(f"{'dimension':12s} {'metric':28s} {'target':>8s} {'achieved':>10s}  status")
    for dim, metric, target, achieved, ok in rows:
        ach = (
            "None"
            if achieved is None
            else (f"{achieved:.3f}" if isinstance(achieved, float) else str(achieved))
        )
        tgt = f"{target:.3f}" if isinstance(target, float) else str(target)
        lines.append(f"{dim:12s} {metric:28s} {tgt:>8s} {ach:>10s}  {'PASS' if ok else 'FAIL'}")
    lines.append(f"completion: {completion(rows) * 100:.0f}%")
    if agg is not None:
        for p in agg["per_case"]:
            lines.append(
                f"  [{p['case']}] tp={p['tp']} fp={p['fp']} "
                f"fn={p['fn']} tn={p['tn']} rho={p['rank_rho']} "
                f"FP={p['false_positive']} FN={p['false_negative']}"
            )
    return "\n".join(lines)
