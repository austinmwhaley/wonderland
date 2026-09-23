"""Ground-truth benchmark harness.

The OPE library's whole job is a binary call: DEPLOY a candidate, or HOLD.
Without routine ground truth we cannot tell "appropriately cautious" from
"estimator is wrong". This harness makes that measurable and regression-proof:
given, per case (environment), the OPE decision for each candidate plus each
candidate's TRUE value and the behaviour anchor, it reports false positives,
false negatives, precision/recall (with Wilson intervals) and rank quality.

Wire a new environment in by producing a `Case`; the aggregation is generic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Cell:
    candidate: str
    deploy: bool
    truth: float
    anchor: float
    bar: float = 0.0
    rank: Optional[int] = None
    lo: Optional[float] = None
    hi: Optional[float] = None

    @property
    def worthy(self) -> bool:
        """Ground-truth target: the candidate beats the LOGGING policy (the
        stated goal is 'never ship worse than behaviour'). The statistical bar
        is a stricter decision threshold and is NOT the worthiness target."""
        return self.truth > self.anchor


@dataclass
class Case:
    name: str
    cells: List[Cell] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (no scipy)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _spearman(x, y):
    n = len(x)
    if n < 2:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) *
                    sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else None


def score_case(case: Case):
    tp = fp = fn = tn = 0
    for c in case.cells:
        if c.deploy and c.worthy:
            tp += 1
        elif c.deploy and not c.worthy:
            fp += 1
        elif (not c.deploy) and c.worthy:
            fn += 1
        else:
            tn += 1
    values = [c.truth for c in case.cells]
    scores = [-(c.rank or 1e9) for c in case.cells]  # rank 1 == best
    rho = _spearman(scores, values) if len(case.cells) > 1 else None
    return {"case": case.name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "deployed": [c.candidate for c in case.cells if c.deploy],
            "false_positive": [c.candidate for c in case.cells
                               if c.deploy and not c.worthy],
            "false_negative": [c.candidate for c in case.cells
                               if (not c.deploy) and c.worthy],
            "rank_rho": None if rho is None else round(rho, 3)}


def summarize(cases: List[Case]):
    per = [score_case(c) for c in cases]
    tp = sum(p["tp"] for p in per)
    fp = sum(p["fp"] for p in per)
    fn = sum(p["fn"] for p in per)
    tn = sum(p["tn"] for p in per)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    pi = wilson_interval(tp, tp + fp) if (tp + fp) else (float("nan"),) * 2
    ri = wilson_interval(tp, tp + fn) if (tp + fn) else (float("nan"),) * 2
    iv = [c for case in cases for c in case.cells
          if c.lo is not None and c.hi is not None]
    cover = (sum(1 for c in iv if c.lo <= c.truth <= c.hi) / len(iv)
             if iv else None)
    mean_width = (sum(c.hi - c.lo for c in iv) / len(iv)) if iv else None
    mean_width_rel = None
    if iv:
        rels = [(c.hi - c.lo) / max(abs(c.anchor), 1.0) for c in iv]
        mean_width_rel = sum(rels) / len(rels)
    return {
        "per_case": per,
        "overall": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": None if prec is None else round(prec, 3),
                    "precision_ci": [round(pi[0], 3), round(pi[1], 3)],
                    "recall": None if rec is None else round(rec, 3),
                    "recall_ci": [round(ri[0], 3), round(ri[1], 3)],
                    "n_deploy_wrong": fp,
                    "n_missed": fn,
                    "coverage": None if cover is None else round(cover, 3),
                    "n_intervals": len(iv),
                    "mean_width": (None if mean_width is None
                                   else round(mean_width, 3)),
                    "mean_width_rel": (None if mean_width_rel is None
                                       else round(mean_width_rel, 3))},
    }


def case_from_pipeline(env_name, decisions, estimates, truths, anchor, bar,
                       rank=None):
    """Adapter: build a Case from a pipeline report + live truths."""
    rank = rank or list(estimates)
    cells = []
    for i, cand in enumerate(rank):
        cells.append(Cell(candidate=cand,
                          deploy=bool(decisions[cand]["deploy"]),
                          truth=float(truths[cand]), anchor=float(anchor),
                          bar=float(bar), rank=i + 1))
    return Case(env_name, cells)
