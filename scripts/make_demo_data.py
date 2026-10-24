"""Generate the synthetic demo data (Layer A) with the reference generator.

One command creates the event stream, the wide table, the known-truth receipt,
and the PATH A/B validation.  Deterministic for a fixed seed.

    python3 scripts/make_demo_data.py --out /tmp/opencode/wonderland_demo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/opencode/wonderland_demo")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--window-days", type=int, default=540, dest="window_days")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--as-of", default="2025-01-01", dest="as_of")
    ap.add_argument("--start-date", default="2024-01-01", dest="start_date")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "rabbit_hole.generators.relational_reference",
        "--n-customers",
        str(args.customers),
        "--window-days",
        str(args.window_days),
        "--seed",
        str(args.seed),
        "--start-date",
        args.start_date,
        "--as-of",
        args.as_of,
        "--out",
        str(out),
    ]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"\nDATA {out}")
    print("Next (one command; the demo is this dataset + the project card):")
    print(
        f"  python3 scripts/run_demo.py --data {out} --frozen /tmp/opencode/frozen_demo --build --refresh"
    )


if __name__ == "__main__":
    main()
