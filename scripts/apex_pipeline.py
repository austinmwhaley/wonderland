"""One-command orchestration of the APEX Layer-A -> Layer-B -> report-card pipeline.

Stages:
  1. generate the APEX stream (real signal)
  2. generate the APEX NULL control (snr=0)
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
) -> list[list[str]]:
    apex = f"{out_root}/apex"
    apex_null = f"{out_root}/apex_null"
    cfm = f"{out_root}/apex_cfm"
    cfm_null = f"{out_root}/apex_null_cfm"
    return [
        [
            sys.executable,
            "-m",
            "rabbit_hole.generators.apex",
            "--n-customers",
            str(n_customers),
            "--window-days",
            str(window_days),
            "--seed",
            str(seed),
            "--out",
            apex,
        ],
        [
            sys.executable,
            "-m",
            "rabbit_hole.generators.apex",
            "--n-customers",
            str(n_customers),
            "--window-days",
            str(window_days),
            "--seed",
            str(seed),
            "--snr",
            "0",
            "--out",
            apex_null,
        ],
        [
            sys.executable,
            "-m",
            "looking_glass.customer_foundation_model",
            "train",
            "--db",
            f"{apex}/customer_event_stream.parquet",
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
            f"{apex_null}/customer_event_stream.parquet",
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
            f"{apex}/customer_event_stream.parquet",
            "--ckpt-dir",
            cfm,
            "--tag",
            TAG,
            "--as-of",
            as_of,
            "--stream",
            "apex",
            "--customers",
            str(customers),
            "--horizons",
            "30,90",
            "--white-queen",
            "--supervised",
            "--unsupervised",
            "--truth",
            f"{apex}/apex_truth.json",
            "--null-db",
            f"{apex_null}/customer_event_stream.parquet",
            "--null-tag",
            NULL_TAG,
            "--null-ckpt-dir",
            cfm_null,
            "--out",
            f"{out_root}/card_apex.md",
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
    ap = argparse.ArgumentParser(description="APEX end-to-end pipeline")
    ap.add_argument("--out-root", default="rabbit_hole/data/apex_pipeline")
    ap.add_argument("--n-customers", type=int, default=20000)
    ap.add_argument("--window-days", type=int, default=540)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--customers", type=int, default=12000)
    ap.add_argument("--sample-a", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    commands = build_commands(
        a.n_customers,
        a.window_days,
        a.seed,
        a.as_of,
        a.out_root,
        a.customers,
        a.sample_a,
        a.max_steps,
    )

    if a.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        return 0

    for i, cmd in enumerate(commands[:4], start=1):
        print(f"[apex-pipeline] stage {i}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

    tag = _resolve_tag(commands[2][commands[2].index("--out-dir") + 1])
    null_tag = _resolve_tag(commands[3][commands[3].index("--out-dir") + 1])
    card = _fill(_fill(commands[4], TAG, tag), NULL_TAG, null_tag)
    print(f"[apex-pipeline] stage 5: {' '.join(card)}", flush=True)
    subprocess.run(card, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
