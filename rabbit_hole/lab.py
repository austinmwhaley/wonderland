"""Layer-A use-case ontology and coverage matrix (the laboratory checklist).

Defines EVERY use case the encoder is claimed to serve and what each fixture must
ship for a verdict to be defensible:

  * stream      — a canonical Parquet stream + a `<id>_truth.json` receipt
  * oracle      — a known-truth predictor that reaches the ceiling (positive control)
  * baselines   — the strong model-free baseline(s) (the honest bar)
  * negative    — the same stream with signal removed (encoder MUST read NO_SIGNAL)
  * adversarial — a leakage/confound trap (the gate MUST NOT false-PASS)

`coverage_report()` is the checked artifact: it reports, per use case, which of the
required pieces exist under `rabbit_hole/data/lab/<id>/`. 100% coverage is the goal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LAB_DIR = Path("rabbit_hole/data/lab")

# fixture pieces every defensible use case must ship
REQUIRED = ("stream", "truth", "oracle", "baseline", "negative", "adversarial")


@dataclass(frozen=True)
class UseCase:
    id: str
    family: str  # binary | count | regression | multiclass | representation | decision | ssl
    kind: str  # clf | reg | multi | unsup | ope | ssl
    card: tuple = ("B",)  # which card(s) grade it: "B" (report_card) and/or "C" (head_battery)
    baseline: str = "mean"  # strong model-free baseline
    oracle: str = "closed-form"  # known-truth predictor


# ---------------------------------------------------------------------------
# THE ONTOLOGY — every head/target the encoder must serve
# ---------------------------------------------------------------------------
USE_CASES: tuple[UseCase, ...] = (
    # ---- B+C supervised heads (Layer C) ----
    UseCase("churn", "binary", "clf", ("C",), "RFM+GBM", "true-hazard"),
    UseCase("purchase_propensity", "binary", "clf", ("C",), "RFM+GBM", "true-hazard"),
    UseCase("funnel_conversion", "binary", "clf", ("C",), "margin-rate", "true-response"),
    UseCase("cart_abandonment", "binary", "clf", ("C",), "mean", "true-response"),
    UseCase("week_n_retention", "binary", "clf", ("C",), "RFM+GBM", "true-hazard"),
    UseCase("order_count", "count", "reg", ("C",), "RFM+GBM", "true-rate"),
    UseCase("basket_size", "count", "reg", ("C",), "mean", "true-rate"),
    UseCase("ltv", "regression", "reg", ("C",), "RFM+GBM", "true-rate"),
    UseCase("next_order_value", "regression", "reg", ("C",), "prev-value", "true-dist"),
    UseCase("time_to_purchase", "regression", "reg", ("C",), "mean", "true-rate"),
    UseCase("price_sensitivity", "regression", "reg", ("C",), "mean", "true-response"),
    UseCase("next_event", "multiclass", "multi", ("B", "C"), "1-gram", "true-transition"),
    UseCase("next_entity", "multiclass", "multi", ("B", "C"), "1-gram", "true-affinity"),
    UseCase("funnel_step", "multiclass", "multi", ("C",), "margin-rate", "true-transition"),
    UseCase("brand_affinity", "multiclass", "multi", ("C",), "popularity", "true-affinity"),
    UseCase("segmentation", "representation", "unsup", ("C",), "mean", "true-segments"),
    UseCase("anomaly", "representation", "unsup", ("B",), "mean", "true-injection"),
    UseCase("identity", "representation", "multi", ("B",), "1-gram", "true-id"),
    # ---- decision (white_queen) ----
    UseCase(
        "rl_ope_incremental_margin", "decision", "ope", ("C",), "logging-policy", "true-effect"
    ),
    UseCase("optimal_action", "decision", "ope", ("C",), "logging-policy", "true-argmax"),
    # ---- B self-supervised capabilities ----
    UseCase("next", "ssl", "ssl", ("B",), "1-gram", "true-transition"),
    UseCase("mask", "ssl", "ssl", ("B",), "1-gram", "true-transition"),
    UseCase("dt", "ssl", "ssl", ("B",), "mean", "true-rate"),
    UseCase("value", "ssl", "ssl", ("B",), "prev-value", "true-dist"),
    UseCase("agg", "ssl", "ssl", ("B",), "mean", "true-window"),
    UseCase("sf", "ssl", "ssl", ("B",), "mean", "true-successor"),
    UseCase("jepa", "ssl", "ssl", ("B",), "mean", "true-dynamics"),
    UseCase("order", "ssl", "ssl", ("B",), "1-gram", "true-transition"),
    UseCase("query", "ssl", "ssl", ("B",), "mean", "true-recency"),
    UseCase("pacing", "ssl", "ssl", ("B",), "mean", "true-velocity"),
    UseCase("boundary", "ssl", "ssl", ("B",), "mean", "true-reset"),
    UseCase("uncertainty", "ssl", "ssl", ("B",), "mean", "true-novelty"),
)


def _present(uc_dir: Path) -> dict:
    """Which required pieces physically exist for a use case."""
    return {
        "stream": (uc_dir / "customer_event_stream.parquet").exists(),
        "truth": any(uc_dir.glob("*_truth.json")),
        "oracle": (uc_dir / "oracle.json").exists(),
        "baseline": (uc_dir / "baseline.json").exists(),
        "negative": (uc_dir / "negative" / "customer_event_stream.parquet").exists(),
        "adversarial": (uc_dir / "adversarial" / "customer_event_stream.parquet").exists(),
    }


def coverage_report(lab_dir: Path = LAB_DIR) -> dict:
    rows = {}
    for uc in USE_CASES:
        present = _present(Path(lab_dir) / uc.id)
        rows[uc.id] = {
            "family": uc.family,
            "card": list(uc.card),
            "present": present,
            "complete": all(present.values()),
            "n_present": sum(present.values()),
        }
    complete = sum(1 for r in rows.values() if r["complete"])
    return {
        "n_use_cases": len(USE_CASES),
        "n_complete": complete,
        "coverage": round(complete / max(len(USE_CASES), 1), 4),
        "rows": rows,
    }


def main(argv=None):  # pragma: no cover - CLI convenience
    rep = coverage_report()
    print(f"coverage {rep['n_complete']}/{rep['n_use_cases']} ({rep['coverage']:.0%})")
    for uid, r in rep["rows"].items():
        miss = [k for k, v in r["present"].items() if not v]
        print(f"  {uid:28s} {r['n_present']}/6  missing={miss}")
    Path("rabbit_hole/lab_coverage.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
