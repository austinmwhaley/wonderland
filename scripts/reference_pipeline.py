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
            "action_period_seconds=86400",
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
            "action_period_seconds=86400",
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
    return [str(arg).replace(token, value) for arg in cmd]


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
    ap.add_argument(
        "--shape-rank",
        type=int,
        default=None,
        dest="shape_rank",
        help="override; default reads the certified rank from the shape certificate",
    )
    ap.add_argument("--seeds", default="0,1,2", help="consumer-gate seeds")
    ap.add_argument(
        "--consumer-horizon-days",
        type=int,
        default=None,
        dest="consumer_horizon_days",
        help="override; default derives the shortest non-degenerate purchase horizon",
    )
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
    _db_path = f"{a.out_root}/reference/customer_event_stream.parquet"
    horizon = (
        int(a.consumer_horizon_days)
        if a.consumer_horizon_days is not None
        else (
            _consumer_horizon(_db_path, a.as_of)
            if Path(_db_path).exists()
            else 3  # dry-run with no generated stream: documented fallback
        )
    )
    post = cockpit_commands(
        a.out_root,
        a.as_of,
        a.customers,
        a.shape_rank if a.shape_rank is not None else "<SHAPE_RANK>",
        period,
        a.seeds,
        horizon,
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
    # Stage 5-7: refit real/null + certify the TRUNK shape rank.
    for i, cmd in enumerate(post[:3], start=5):
        cmd = _fill(_fill(cmd, TAG, tag), NULL_TAG, null_tag)
        cmd = [portfolio if x == "<PORTFOLIO>" else x for x in cmd]
        print(f"[reference-pipeline] stage {i}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    rank_path = Path(a.out_root) / "minimality_shape_unserved.json"
    rank = (
        int(a.shape_rank)
        if a.shape_rank is not None
        else int(json.loads(rank_path.read_text())["measure_dim"])
    )
    print(f"[reference-pipeline] certified shape rank = {rank}", flush=True)
    for i, cmd in enumerate(post[3:], start=8):
        cmd = _fill(_fill(cmd, TAG, tag), NULL_TAG, null_tag)
        cmd = _fill(cmd, "<SHAPE_RANK>", str(rank))
        cmd = [portfolio if x == "<PORTFOLIO>" else x for x in cmd]
        print(f"[reference-pipeline] stage {i}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    return 0


def _consumer_horizon(db: str, as_of: str, candidates=(3, 7, 14, 30, 90)) -> int:
    """Pick the shortest consumer horizon whose purchase label is not degenerate.

    A 90d purchase label on the reference is ~99.9% one class (daily sends); an AUC
    on 5 minority examples is noise. The horizon is DATA-DERIVED: the first
    candidate with min(class) >= max(20, 2% of n). Falls back to the last candidate.
    """
    import duckdb

    from looking_glass.cfm_config import as_of_epoch

    src = f"read_parquet('{db}')" if db.endswith((".parquet", ".pq")) else db
    tc = float(as_of_epoch(as_of))
    con = duckdb.connect()
    try:
        for h in candidates:
            row = con.execute(
                f"SELECT count(*) FILTER (WHERE has) purchasers, count(*) n FROM ("
                f"SELECT customer_key, max(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) has "
                f"FROM {src} WHERE epoch(CAST(event_ts AS TIMESTAMPTZ))>{tc} "
                f"AND epoch(CAST(event_ts AS TIMESTAMPTZ))<={tc + h * 86400.0} GROUP BY 1)"
            ).fetchone()
            pos, n = int(row[0] or 0), int(row[1] or 0)
            minority = min(pos, n - pos)
            if n and minority >= max(20, int(0.02 * n)):
                return int(h)
    finally:
        con.close()
    return int(candidates[-1])


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
    shape_rank,
    period: float,
    seeds: str,
    consumer_horizon: int,
    acceptance: str | None,
    tag: str = TAG,
    null_tag: str = NULL_TAG,
) -> list[list[str]]:
    """The full steering chain: receipts -> objectives -> cockpit report card.

    Every consumer/decision receipt is produced by a real run; the cockpit is the
    final artifact. `shape_rank` may be `<SHAPE_RANK>` (filled from the certified
    shape receipt after it runs). `tag`/`null_tag` are resolved from the registries.
    """
    ref = f"{out_root}/reference"
    ref_null = f"{out_root}/reference_null"
    cfm = f"{out_root}/reference_cfm"
    cfm_null = f"{out_root}/reference_null_cfm"
    db = f"{ref}/customer_event_stream.parquet"
    ndb = f"{ref_null}/customer_event_stream.parquet"
    truth = f"{ref}/reference_truth.json"
    py = sys.executable
    acc_path = acceptance or f"{out_root}/acceptance.json"
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
    null_common = [
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
            *null_common,
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
            "--unserved-shape",
            "--out",
            f"{out_root}/minimality_shape_unserved.json",
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
            *null_common,
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
            "scripts/minimality_certificate.py",
            *common,
            "--horizons",
            "30,90",
            "--targets",
            "future",
            "--block",
            "shape",
            "--out",
            f"{out_root}/minimality_shape_served.json",
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
            "scripts/minimality_blocks.py",
            "--shape",
            f"{out_root}/minimality_shape_served.json",
            "--joint",
            f"{out_root}/minimality_joint.json",
            "--layer-b-json",
            f"{cfm}/layer_b_{tag.replace('.', '_')}.pkl.json",
            "--out",
            f"{out_root}/minimality.json",
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
            str(consumer_horizon),
            "--out",
            f"{out_root}/transfer.json",
        ],
        [
            py,
            "scripts/consumer_comparators.py",
            *common,
            "--horizon-days",
            str(consumer_horizon),
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
            str(consumer_horizon),
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
            "scripts/u_controls_receipt.py",
            *common,
            "--horizons",
            "30,90",
            "--out",
            f"{out_root}/u_controls.json",
        ],
        [
            py,
            "scripts/ope_bridge_reconcile.py",
            *common,
            "--action-key",
            "channel",
            "--out",
            f"{out_root}/ope_reconcile.json",
        ],
        [
            py,
            "scripts/reproducibility_receipt.py",
            *common,
            "--out",
            f"{out_root}/reproducibility.json",
        ],
        [
            py,
            "scripts/quality_receipts.py",
            "--out-dir",
            str(out_root),
            "--ckpt-dir",
            cfm,
            "--tag",
            tag,
        ],
    ]
    if not acceptance:
        cmds += [
            [py, "scripts/benchmark_ope.py"],
            [
                py,
                "-c",
                "import shutil,sys; shutil.copy(sys.argv[1], sys.argv[2])",
                "/tmp/opencode/white_queen_matrix/acceptance.json",
                acc_path,
            ],
        ]
    cmds += [
        [
            py,
            "scripts/objectives_receipts.py",
            "--card",
            f"{out_root}/card_receipts.json",
            "--portfolio",
            "<PORTFOLIO>",
            "--minimality",
            f"{out_root}/minimality.json",
            "--acceptance",
            acc_path,
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
            "--reproducibility",
            f"{out_root}/reproducibility.json",
            "--efficiency",
            f"{out_root}/efficiency.json",
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
            f"{out_root}/minimality.json",
            "--minimality-shape",
            f"{out_root}/minimality_shape_served.json",
            "--serve-projection",
            f"{out_root}/serve_real.json",
            "--acceptance",
            acc_path,
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
            "--shape-attribution",
            f"{out_root}/shape_attribution.json",
            "--customers",
            str(customers),
            "--reproducibility",
            f"{out_root}/reproducibility.json",
            "--ope-reconcile",
            f"{out_root}/ope_reconcile.json",
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
