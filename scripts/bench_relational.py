"""Benchmark: committed CPU-old generator vs the vectorized GPU-new generator at 2k.

Extracts the committed (HEAD) per-customer-loop generator to a temp file, times a 2k/540 run,
then runs the current generator to --out with GEN_CPU_OLD_SECONDS set, so the emitted
runtime_receipt.json carries the CPU-old vs GPU-new wall-clock and speedup.

Usage: bench_relational.py [--out DIR] [--n 2000] [--days 540] [--seed 17]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _time_old(n, days, seed):
    src = subprocess.run(
        ["git", "-C", str(ROOT), "show", "HEAD:rabbit_hole/generators/relational_reference.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with tempfile.TemporaryDirectory() as td:
        old = Path(td) / "old_relational_reference.py"
        old.write_text(src)
        t0 = time.perf_counter()
        rc = subprocess.run(
            [
                sys.executable,
                str(old),
                "--n-customers",
                str(n),
                "--window-days",
                str(days),
                "--seed",
                str(seed),
                "--as-of",
                "2025-01-01",
                "--out",
                str(Path(td) / "out"),
                "--no-validate",
            ],
            capture_output=True,
            text=True,
        )
        dt = time.perf_counter() - t0
        if rc.returncode != 0:
            raise SystemExit(f"old generator failed: {rc.stderr[-400:]}")
    return round(dt, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/opencode/t0_loop/relational_v6")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--days", type=int, default=540)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--old-seconds", type=float, default=None)
    a = ap.parse_args()
    old = a.old_seconds if a.old_seconds is not None else _time_old(a.n, a.days, a.seed)
    env = {**__import__("os").environ, "GEN_CPU_OLD_SECONDS": str(old)}
    t0 = time.perf_counter()
    rc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "rabbit_hole/generators/relational_reference.py"),
            "--n-customers",
            str(a.n),
            "--window-days",
            str(a.days),
            "--seed",
            str(a.seed),
            "--out",
            a.out,
            "--name",
            "relational_v6",
        ],
        env=env,
    )
    new = round(time.perf_counter() - t0, 2)
    if rc.returncode != 0:
        raise SystemExit("new generator failed")
    print(f"cpu_old={old}s gpu_new_total={new}s speedup_total={old / new:.2f}x")


if __name__ == "__main__":
    main()
