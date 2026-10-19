"""One-command orchestration of the REFERENCE Layer-A -> Layer-B -> report-card pipeline.

Stages:
  1. generate the REFERENCE stream (real signal)
  2. generate the REFERENCE NULL control (snr=0)
  3. train the foundation model on the real stream
  4. train the foundation model on the null stream
  5. emit the B & B+C report card (white_queen + null controls)

The training tag is data-derived, so stage 5 carries `<TAG>` / `<NULLTAG>` placeholders
that the runner resolves from the newest registry in each checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TAG = "<TAG>"
NULL_TAG = "<NULLTAG>"


def build_commands(
    n_customers,
    window_days,
    seed,
    as_of,
    out_root,
    customers,
    sample_a,
    max_steps,
    full=False,
) -> list[list[str]]:
    reference = f"{out_root}/reference"
    reference_null = f"{out_root}/reference_null"
    cfm = f"{out_root}/reference_cfm"
    cfm_null = f"{out_root}/reference_null_cfm"
    return [
        [
            sys.executable,
            "-m",
            "rabbit_hole.generators.reference",
            "--n-customers",
            str(n_customers),
            "--window-days",
            str(window_days),
            "--seed",
            str(seed),
            "--out",
            reference,
        ],
        [
            sys.executable,
            "-m",
            "rabbit_hole.generators.reference",
            "--n-customers",
            str(n_customers),
            "--window-days",
            str(window_days),
            "--seed",
            str(seed),
            "--snr",
            "0",
            "--out",
            reference_null,
        ],
        [
            sys.executable,
            "-m",
            "looking_glass.customer_foundation_model",
            "train",
            "--db",
            f"{reference}/customer_event_stream.parquet",
            "--customers",
            str(customers),
            "--sample-a",
            str(sample_a),
            "--anchors",
            "3",
            "--as-of",
            as_of,
            "--out-dir",
            cfm,
            "--set",
            "use_measure=True",
            "--set",
            "measure_dim=128",
            "--set",
            "effect_actions=('freq','hour','dow','channel','offer')",
            "--set",
            "action_fields=('freq','hour','dow','channel','offer','discount_pct')",
            "--set",
            f"max_steps={max_steps}",
        ],
        [
            sys.executable,
            "-m",
            "looking_glass.customer_foundation_model",
            "train",
            "--db",
            f"{reference_null}/customer_event_stream.parquet",
            "--customers",
            str(customers),
            "--sample-a",
            str(sample_a),
            "--anchors",
            "3",
            "--as-of",
            as_of,
            "--out-dir",
            cfm_null,
            "--set",
            "use_measure=True",
            "--set",
            "measure_dim=128",
            "--set",
            "effect_actions=('freq','hour','dow','channel','offer')",
            "--set",
            "action_fields=('freq','hour','dow','channel','offer','discount_pct')",
            "--set",
            f"max_steps={max_steps}",
        ],
        [
            sys.executable,
            "-m",
            "looking_glass.foundation_report",
            "--db",
            f"{reference}/customer_event_stream.parquet",
            "--ckpt-dir",
            cfm,
            "--tag",
            TAG,
            "--as-of",
            as_of,
            "--stream",
            "reference",
            "--customers",
            str(customers),
            "--horizons",
            "30,90",
            "--fast",
            *(["--white-queen", "--supervised", "--unsupervised"] if full else []),
            "--truth",
            f"{reference}/reference_truth.json",
            "--null-db",
            f"{reference_null}/customer_event_stream.parquet",
            "--null-tag",
            NULL_TAG,
            "--null-ckpt-dir",
            cfm_null,
            "--out",
            f"{out_root}/card_reference.md",
        ],
    ]


def _resolve_tag(out_dir: str) -> str:
    regs = sorted(
        Path(out_dir).glob("registry_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not regs:
        raise SystemExit(f"no registry found in {out_dir}; training did not emit one")
    return json.loads(regs[-1].read_text())["tag"]


def _fill(cmd: list[str], token: str, value: str) -> list[str]:
    return [value if arg == token else arg for arg in cmd]


def main(argv=None):
    ap = argparse.ArgumentParser(description="REFERENCE end-to-end pipeline")
    ap.add_argument("--out-root", default="rabbit_hole/data/reference_pipeline")
    ap.add_argument("--n-customers", type=int, default=20000)
    ap.add_argument("--window-days", type=int, default=540)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--customers", type=int, default=12000)
    ap.add_argument("--sample-a", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument(
        "--full",
        action="store_true",
        help="include white_queen + supervised + unsupervised (slow)",
    )
    ap.add_argument("--shape-rank", type=int, default=48, dest="shape_rank")
    ap.add_argument("--seeds", default="0,1,2", help="consumer-gate seeds")
    ap.add_argument("--acceptance", default=None, help="OPE acceptance receipt (optional)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    train = build_commands(
        a.n_customers,
        a.window_days,
        a.seed,
        a.as_of,
        a.out_root,
        a.customers,
        a.sample_a,
        a.max_steps,
        full=a.full,
    )
    period = _period_from_truth(f"{a.out_root}/reference/reference_truth.json")
    post = cockpit_commands(
        a.out_root,
        a.as_of,
        a.customers,
        a.shape_rank,
        period,
        a.seeds,
        a.acceptance,
    )

    if a.dry_run:
        for cmd in train + post:
            print(" ".join(cmd))
        return 0

    for i, cmd in enumerate(train[:4], start=1):
        print(f"[reference-pipeline] stage {i}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    tag = _resolve_tag(train[2][train[2].index("--out-dir") + 1])
    null_tag = _resolve_tag(train[3][train[3].index("--out-dir") + 1])
    portfolio = _newest_portfolio(tag) or ""
    for i, cmd in enumerate(post, start=5):
        cmd = _fill(_fill(cmd, TAG, tag), NULL_TAG, null_tag)
        cmd = [portfolio if x == "<PORTFOLIO>" else x for x in cmd]
        print(f"[reference-pipeline] stage {i}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    return 0


def _period_from_truth(path: str) -> float:
    try:
        meta = json.loads(Path(path).read_text()).get("meta", {}) or {}
        act = meta.get("action") or {}
        return float(
            act.get("action_period_seconds") or meta.get("action_period_seconds") or 86400.0
        )
    except Exception:  # noqa: BLE001
        return 86400.0


def cockpit_commands(
    out_root: str,
    as_of: str,
    customers: int,
    shape_rank: int,
    period: float,
    seeds: str,
    acceptance: str | None,
    tag: str = TAG,
    null_tag: str = NULL_TAG,
) -> list[list[str]]:
    """The full steering chain: receipts -> objectives -> cockpit report card.

    Every consumer/decision receipt is produced by a real run; the cockpit is the
    final artifact. `tag`/`null_tag` are resolved from the training registries.
    """
    ref = f"{out_root}/reference"
    ref_null = f"{out_root}/reference_null"
    cfm = f"{out_root}/reference_cfm"
    cfm_null = f"{out_root}/reference_null_cfm"
    db = f"{ref}/customer_event_stream.parquet"
    ndb = f"{ref_null}/customer_event_stream.parquet"
    truth = f"{ref}/reference_truth.json"
    py = sys.executable
    common = [
        "--db",
        db,
        "--ckpt-dir",
        cfm,
        "--tag",
        tag,
        "--as-of",
        as_of,
        "--customers",
        str(customers),
    ]
    cmds = [
        [
            py,
            "scripts/refit_measure.py",
            *common,
            "--period-seconds",
            f"{period}",
            "--out",
            f"{cfm}/layer_b_{tag.replace('.', '_')}.pkl",
        ],
        [
            py,
            "scripts/refit_measure.py",
            "--db",
            ndb,
            "--ckpt-dir",
            cfm_null,
            "--tag",
            null_tag,
            "--as-of",
            as_of,
            "--customers",
            str(customers),
            "--period-seconds",
            f"{period}",
            "--out",
            f"{cfm_null}/layer_b_{null_tag.replace('.', '_')}.pkl",
        ],
        [
            py,
            "scripts/minimality_certificate.py",
            *common,
            "--horizons",
            "30,90",
            "--targets",
            "future",
            "--block",
            "shape",
            "--out",
            f"{out_root}/minimality_shape.json",
        ],
        [
            py,
            "scripts/fit_serve_projection.py",
            *common,
            "--k",
            str(shape_rank),
            "--mode",
            "blocks",
            "--period-seconds",
            f"{period}",
            "--out",
            f"{cfm}/serve_proj_{tag.replace('.', '_')}.npz",
            "--receipt",
            f"{out_root}/serve_real.json",
        ],
        [
            py,
            "scripts/fit_serve_projection.py",
            "--db",
            ndb,
            "--ckpt-dir",
            cfm_null,
            "--tag",
            null_tag,
            "--as-of",
            as_of,
            "--customers",
            str(customers),
            "--k",
            str(shape_rank),
            "--mode",
            "blocks",
            "--period-seconds",
            f"{period}",
            "--out",
            f"{cfm_null}/serve_proj_{null_tag.replace('.', '_')}.npz",
            "--receipt",
            f"{out_root}/serve_null.json",
        ],
        [
            py,
            "-m",
            "looking_glass.foundation_report",
            *common,
            "--horizons",
            "30,90",
            "--fast",
            "--white-queen",
            "--supervised",
            "--unsupervised",
            "--truth",
            truth,
            "--null-db",
            ndb,
            "--null-tag",
            null_tag,
            "--null-ckpt-dir",
            cfm_null,
            "--out",
            f"{out_root}/card.md",
            "--receipts",
            f"{out_root}/card_receipts.json",
        ],
        [
            py,
            "scripts/transfer_receipt.py",
            *common,
            "--horizon-days",
            "90",
            "--out",
            f"{out_root}/transfer.json",
        ],
        [
            py,
            "scripts/consumer_comparators.py",
            *common,
            "--horizon-days",
            "90",
            "--seeds",
            seeds,
            "--out",
            f"{out_root}/comparators.json",
        ],
        [
            py,
            "scripts/ablation_ladder.py",
            *common,
            "--horizon-days",
            "90",
            "--seed",
            "0",
            "--out",
            f"{out_root}/ablation.json",
        ],
        [
            py,
            "scripts/known_effect_receipt.py",
            *common,
            "--truth",
            truth,
            "--out",
            f"{out_root}/known_effect.json",
        ],
        [
            py,
            "scripts/minimality_certificate.py",
            *common,
            "--horizons",
            "30,90",
            "--targets",
            "joint",
            "--block",
            "full",
            "--out",
            f"{out_root}/minimality_joint.json",
        ],
        [
            py,
            "scripts/u_controls_receipt.py",
            *common,
            "--horizons",
            "30,90",
            "--out",
            f"{out_root}/u_controls.json",
        ],
        [py, "scripts/quality_receipts.py", "--out-dir", str(out_root)],
        [
            py,
            "scripts/objectives_receipts.py",
            "--card",
            f"{out_root}/card_receipts.json",
            "--portfolio",
            "<PORTFOLIO>",
            "--minimality",
            f"{out_root}/minimality_joint.json",
            "--known-effect",
            f"{out_root}/known_effect.json",
            "--schema-suite",
            f"{out_root}/schema_suite.json",
            "--transfer",
            f"{out_root}/transfer.json",
            "--accessibility",
            f"{out_root}/accessibility.json",
            "--encoder",
            f"{out_root}/encoder.json",
            "--comparators",
            f"{out_root}/comparators.json",
            *(["--acceptance", acceptance] if acceptance else []),
            "--out",
            f"{out_root}/objectives_inputs.json",
        ],
        [
            py,
            "-m",
            "looking_glass.objectives",
            "--artifacts",
            f"{out_root}/objectives_inputs.json",
            "--json",
            f"{out_root}/objectives.json",
            "--markdown",
            f"{out_root}/objectives.md",
        ],
        [
            py,
            "scripts/full_report_card.py",
            "--card",
            f"{out_root}/card_receipts.json",
            "--objectives",
            f"{out_root}/objectives.json",
            "--portfolio",
            "<PORTFOLIO>",
            "--minimality",
            f"{out_root}/minimality_joint.json",
            "--minimality-shape",
            f"{out_root}/minimality_shape.json",
            "--serve-projection",
            f"{out_root}/serve_real.json",
            "--acceptance",
            acceptance or "",
            "--schema-suite",
            f"{out_root}/schema_suite.json",
            "--encoder",
            f"{out_root}/encoder.json",
            "--accessibility",
            f"{out_root}/accessibility.json",
            "--transfer",
            f"{out_root}/transfer.json",
            "--known-effect",
            f"{out_root}/known_effect.json",
            "--comparators",
            f"{out_root}/comparators.json",
            "--ablation",
            f"{out_root}/ablation.json",
            "--u-controls",
            f"{out_root}/u_controls.json",
            "--card-md",
            f"{out_root}/card.md",
            "--truth",
            truth,
            "--stream",
            "reference",
            "--tag",
            tag,
            "--as-of",
            as_of,
            "--out-md",
            f"{out_root}/cockpit.md",
            "--out-json",
            f"{out_root}/cockpit.json",
        ],
    ]
    return cmds


def _newest_portfolio(tag: str) -> str | None:
    root = Path("looking_glass/artifacts/cfm/portfolio")
    hits = sorted(
        root.glob(f"portfolio_{tag.replace('.', '_')}_*.json"), key=lambda p: p.stat().st_mtime
    )
    return str(hits[-1]) if hits else None


if __name__ == "__main__":
    raise SystemExit(main())
