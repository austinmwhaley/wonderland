"""Top-level agnostic entry point: logs + candidate policy -> decision.

    from white_queen.tribunal.ope.api import evaluate
    report = evaluate(parquet_or_df_or_dict, candidate_policy)

Returns a decision (DEPLOY/HOLD), the estimate panel, robustness to hidden
confounding, and a plain-English rationale. Data source, action size, and
bandit-vs-sequential are all inferred by ope.data.to_canonical.
"""

from __future__ import annotations


def evaluate(
    source,
    candidate,
    *,
    gamma=0.99,
    columns=None,
    nA=None,
    estimate_propensity=True,
    behavior_cfg=None,
    behavior_seed=0,
    risk_aversion=0.5,
    fast=True,
    ensemble_K=None,
    fqe_cfg=None,
    cache_dir=None,
    weights_hash=None,
    candidate_name="candidate",
    source_name=None,
    n_episodes=None,
):
    """Evaluate one candidate against a logged dataset. Returns a report dict.

    This is the whole library in one call: ingest -> propensity -> panel ->
    gate -> judge. It never reads ground-truth outcomes (there are none in a
    real log); it decides from offline evidence only.
    """
    from . import data as _data
    from . import estimators as _E
    from . import gate as _gate
    from . import judge as _judge
    from .protocols import validate_diet
    from .receipts import behavior_stats
    from .drift import drift_report

    data = _data.to_canonical(
        source,
        columns=columns,
        nA=nA,
        estimate_propensity=estimate_propensity,
        behavior_cfg=behavior_cfg,
        behavior_seed=behavior_seed,
        source_name=source_name,
    )
    validate_diet(data)
    b = behavior_stats(data, gamma)
    if data.get("continuous"):
        from .continuous import panel_continuous

        panel = panel_continuous(
            data,
            candidate,
            gamma,
            fqe_cfg=fqe_cfg,
            fast=fast,
            cand_id=candidate_name,
            cache_dir=cache_dir,
            weights_hash=weights_hash,
        )
    else:
        panel = _E.panel(
            data,
            candidate,
            gamma,
            meta=None,
            fqe_cfg=fqe_cfg,
            cand_id=candidate_name,
            cache_dir=cache_dir,
            weights_hash=weights_hash,
            ensemble_K=ensemble_K,
            fast=fast,
        )
    rows = _gate.adjudicate(
        {candidate_name: panel}, b["mean"], b["std"], None, None, n_episodes=n_episodes
    )
    verdict = _judge.judge_diet(rows, b["mean"], b["std"], None, risk_aversion=risk_aversion)
    row = rows[candidate_name]
    dec = verdict["decisions"][candidate_name]
    rationale = _judge.explain_diet("log", b["mean"], verdict, rows)
    return {
        "candidate": candidate_name,
        "deploy": dec["deploy"],
        "decision": dec,
        "behavior_mean": round(b["mean"], 2),
        "behavior_std": round(b["std"], 2),
        "bar": verdict["bar"],
        "witnesses": dec["witnesses"],
        "estimates": row,
        "sensitivity": row.get("sensitivity"),
        "prescription": dec.get("prescription"),
        "provenance": data.get("provenance", {}),
        "drift": drift_report(data, gamma),
        "rationale": rationale,
    }
