"""One-command REFERENCE certification at the FROZEN minimality standard.

Reproduces the served model and every cockpit receipt from a pre-built real/null
encoder pair:

  1. refit the Layer-B measure (real + null)
  2. certify the UNSERVED shape rank (reconstruction rule, DEC-220)
  3. fit the block-wise serve at that certified rank
  4. served + joint minimality certificates, then the block-wise receipt
  5. the full receipt chain + objectives + cockpit

Base training is deliberately separate (it is the expensive, documented recipe:
0.7.0-distill base + deterministic counter/schema-summary augment). Pass
--real-cfm/--null-cfm to the augmented models.

Usage:
  python3 scripts/reference_certify.py --out /tmp/ref_run \
      --real-cfm /tmp/.../reference_v4l_cfm --real-tag 0.11.0r776750s2000 \
      --null-cfm /tmp/.../reference_v4l_null_cfm --null-tag 0.11.0r703511s2000 \
      --db /tmp/.../reference_v4/customer_event_stream.parquet \
      --null-db /tmp/.../reference_v4_null/customer_event_stream.parquet \
      --truth /tmp/.../reference_v4/reference_truth.json --as-of 2025-01-01
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reference_pipeline import _newest_portfolio, cockpit_commands  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-cfm", required=True, dest="real_cfm")
    ap.add_argument("--real-tag", required=True, dest="real_tag")
    ap.add_argument("--null-cfm", required=True, dest="null_cfm")
    ap.add_argument("--null-tag", required=True, dest="null_tag")
    ap.add_argument("--db", required=True)
    ap.add_argument("--null-db", required=True, dest="null_db")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--period-seconds", type=float, default=86400.0, dest="period_seconds")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, target in {
        "reference": str(Path(a.db).parent),
        "reference_null": str(Path(a.null_db).parent),
        "reference_cfm": a.real_cfm,
        "reference_null_cfm": a.null_cfm,
    }.items():
        p = out / name
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            import shutil

            shutil.rmtree(p)
        p.symlink_to(target, target_is_directory=True)

    def _build(rank):
        return cockpit_commands(
            str(out),
            a.as_of,
            a.customers,
            rank,
            a.period_seconds,
            a.seeds,
            3,
            None,
            tag=a.real_tag,
            null_tag=a.null_tag,
        )

    # Stage 2 only first: the unserved reconstruction certificate fixes the rank.
    pre = _build("<RANK>")
    cert_cmd = pre[2]
    if a.dry_run:
        for cmd in pre:
            print(" ".join(cmd))
        return 0
    print("[certify] unserved shape certificate", flush=True)
    subprocess.run(cert_cmd, check=True, cwd=str(ROOT))
    rank_path = out / "minimality_shape_unserved.json"
    rank = int(json.loads(rank_path.read_text())["measure_dim"])
    print(f"[certify] frozen reconstruction rank = {rank}", flush=True)

    print("[certify] shape attribution receipt", flush=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/shape_attribution.py",
            "--db",
            a.db,
            "--ckpt-dir",
            a.real_cfm,
            "--tag",
            a.real_tag,
            "--as-of",
            a.as_of,
            "--customers",
            str(a.customers),
            "--out",
            str(out / "shape_attribution.json"),
        ],
        check=True,
        cwd=str(ROOT),
    )

    cmds = _build(rank)
    portfolio = _newest_portfolio(a.real_tag) or ""
    for i, cmd in enumerate(cmds):
        if i == 2:  # already run
            continue
        cmd = [portfolio if x == "<PORTFOLIO>" else x for x in cmd]
        print(f"[certify] stage {i}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    print("[certify] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
