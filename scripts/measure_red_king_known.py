#!/usr/bin/env python3
"""red_king V1.3 known-effect gate on the demo randomized offer stream (B-005).

The v1.3 atom is "red_king recovers known effects". The earlier receipt used the
synthetic `_bandit` scorecard; this receipt runs red_king's per-customer effect
estimator on the served demo stream's randomized weekly offer panel and validates
against the generator's declared `offer_opt`.

Pre-registered bars (before the run):
  * rank accuracy vs `offer_opt` >= 0.50 (2x the 4-arm chance), binomial p < 1e-6
  * held-out policy value > random, paired 95% CI lower bound > 0
  * held-out policy value >= the best constant arm
  * value relative error |estimated - held-out| / held-out <= 0.50

The fit uses only randomized send-day cells before the as-of; the evaluation uses the
post-as-of randomized cells. `offer_opt` is never a fit input.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from measure_input_law import stamp_merge_write  # noqa: E402
from red_king import demo_effect  # noqa: E402

DEMO = ROOT / "looking_glass/artifacts/demo_v2"
OUT = ROOT / "docs/report_cards/experiments"
AS_OF = datetime(2025, 1, 1, tzinfo=timezone.utc)
BAR_RANK = 0.50
BAR_P = 1e-6
BAR_REL_ERR = 0.50
SEED = 0
N_BOOT = 2000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _paired_ci(delta: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = np.asarray(
        [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(N_BOOT)]
    )
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _mean_spearman(V, V_eval) -> float | None:
    from scipy.stats import spearmanr

    vals = []
    for k in V:
        if k not in V_eval:
            continue
        a = np.asarray([V[k][x] for x in sorted(V[k])])
        b = np.asarray([V_eval[k][x] for x in sorted(V_eval[k])])
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        vals.append(float(spearmanr(a, b).statistic))
    return round(float(np.mean(vals)), 6) if vals else None


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    stream_path = DEMO / "customer_event_stream.parquet"
    truth_path = DEMO / "reference_truth.json"
    panel = demo_effect.panel(stream_path)
    pre = panel.filter(panel["day"] < AS_OF.date())
    post = panel.filter(panel["day"] >= AS_OF.date())
    V, pooled_fit = demo_effect.fit(pre, outcome="count", shrink=2.0)
    V_eval, _pooled_eval = demo_effect.fit(post, outcome="count", shrink=2.0)
    known = demo_effect.known_effect(V, truth_path)
    pol = demo_effect.policy(V, post, outcome="count")

    d_random = pol["policy_by_customer"] - pol["random_by_customer"]
    d_const = pol["policy_by_customer"] - pol["best_constant"]
    ci_random = _paired_ci(d_random)
    ci_const = _paired_ci(d_const)
    rel_err = (
        abs(pol["estimated_value"] - pol["policy"]) / abs(pol["policy"]) if pol["policy"] else None
    )
    score = float(
        np.mean(
            [
                min(known["accuracy"] / BAR_RANK, 1.0) if known["accuracy"] is not None else 0.0,
                1.0 if ci_random[0] > 0 else 0.0,
                1.0
                if pol["best_constant"] is not None and pol["policy"] >= pol["best_constant"]
                else 0.0,
                1.0 if rel_err is not None and rel_err <= BAR_REL_ERR else 0.0,
            ]
        )
    )
    passed = bool(
        known["accuracy"] is not None
        and known["accuracy"] >= BAR_RANK
        and known["binom_p"] < BAR_P
        and ci_random[0] > 0
        and pol["policy"] >= pol["best_constant"]
        and rel_err is not None
        and rel_err <= BAR_REL_ERR
    )
    body = {
        "kind": "red_king_known_effect",
        "atom": "V1.3",
        "stream": str(stream_path),
        "stream_sha256": _sha256(stream_path),
        "truth_sha256": _sha256(truth_path),
        "as_of": AS_OF.date().isoformat(),
        "protocol": (
            "per-customer randomized send-day value model (red_king.demo_effect); fit on "
            "pre-as-of cells, score the policy on post-as-of cells; truth is validation only"
        ),
        "n_fit_cells": int(pre.height),
        "n_eval_cells": int(post.height),
        "n_fit_customers": len(V),
        "n_eval_customers": pol["n_customers"],
        "outcome": "purchase count on the randomized offer day",
        "shrink": 2.0,
        "bars": {
            "rank_accuracy": BAR_RANK,
            "binom_p": BAR_P,
            "policy_ci_lo": 0.0,
            "best_constant": "policy >= best constant",
            "value_rel_err": BAR_REL_ERR,
        },
        "known_effect": known,
        "policy": {
            "policy": round(pol["policy"], 6),
            "random_all": round(pol["random_all"], 6),
            "random_customer_mean": round(pol["random"], 6),
            "best_constant": round(pol["best_constant"], 6),
            "best_constant_arm": pol["best_constant_arm"],
            "estimated_value": round(pol["estimated_value"], 6),
            "value_rel_err": round(rel_err, 6) if rel_err is not None else None,
            "paired_ci_policy_minus_random": list(ci_random),
            "paired_ci_policy_minus_best_constant": list(ci_const),
            "pooled_fit": {str(k): round(v, 6) for k, v in pooled_fit.items()},
            "pooled_eval": {str(k): round(v, 6) for k, v in pol["pooled"].items()},
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "scorecard": {
            "S1_ordering": _mean_spearman(V, V_eval),
            "S3_rank_acc": known["accuracy"],
            "S4_value_rel_err": round(rel_err, 6) if rel_err is not None else None,
            "S6_beat_modelfree": round(pol["policy"] - pol["random_all"], 6),
            "SCORE": round(score, 3),
        },
        "passed": passed,
        "why": (
            "the per-customer estimator recovers the generator's declared offer optimum at "
            f"{known['accuracy']:.3f} vs {known['chance']:.3f} chance "
            f"(majority {known['majority_accuracy']:.3f}); on the post-as-of randomized "
            f"holdout the argmax policy is {pol['policy']:.3f} vs {pol['random_all']:.3f} "
            f"random (paired CI {ci_random[0]:+.3f}, {ci_random[1]:+.3f}) and "
            f"{pol['best_constant']:.3f} best constant."
        ),
    }
    stamp_merge_write(
        OUT / "red_king_known.json",
        body,
        " ".join([Path(__file__).name] + sys.argv[1:]),
        stamp_blocks=False,
    )
    print(
        json.dumps(
            {k: body[k] for k in ("passed", "known_effect", "policy", "scorecard")}, indent=1
        )
    )
    print("WROTE", OUT / "red_king_known.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
