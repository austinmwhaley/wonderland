"""Run the wonderland demo: one dataset -> frozen encoder -> receipts -> REPORT_CARD.md.

The demo is not a separate artifact.  It is the project running on the synthetic
demo dataset (A), with the same B + B+C card the project always produces.

    python3 scripts/run_demo.py --generate --build --refresh   # full run from scratch
    python3 scripts/run_demo.py --refresh --research            # + pinned construction receipt
    python3 scripts/run_demo.py                                # re-render the saved card

Full refresh is compute-heavy (frozen encoder + GBM/XGBoost batteries).  Receipts
are cached under docs/report_cards/experiments/; `--refresh` recomputes them.

The construction receipt is pinned to the Layer-A reference build rather than the
demo `--data` directory, so it runs only with `--research`.  That keeps one
orchestrator instead of a separate refresh script.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs/report_cards/experiments"


def run(cmd, env=None):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/wonderland_demo")
    ap.add_argument("--frozen", default="/tmp/opencode/frozen_demo")
    ap.add_argument("--generate", action="store_true", help="generate the demo dataset first")
    ap.add_argument("--build", action="store_true", help="build the frozen encoder cache")
    ap.add_argument("--refresh", action="store_true", help="recompute receipts, then the card")
    ap.add_argument(
        "--research", action="store_true", help="also refresh the pinned construction receipt"
    )
    ap.add_argument("--skip-objectives", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    frozen = Path(args.frozen)
    py = sys.executable

    if args.generate:
        run([py, "scripts/make_demo_data.py", "--out", str(data)])
    if args.build or (args.refresh and not frozen.with_suffix(".npz").exists()):
        run([py, "scripts/build_frozen.py", str(data), "--out", str(frozen)])

    if args.refresh:
        env = dict(os.environ, FROZEN=str(frozen))
        run([py, "scripts/measure_data.py", "--data", str(data)])
        if args.research:
            run([py, "scripts/measure_construction.py"], env=env)
        run([py, "scripts/measure_donor.py", str(data)], env=env)
        run([py, "scripts/measure_controls.py", str(data)], env=env)
        run([py, "scripts/measure_identity.py", str(data)], env=env)
        run([py, "scripts/measure_ceiling.py", str(data)], env=env)
        if not args.skip_objectives:
            run([py, "scripts/measure_objectives.py", str(data)], env=env)
        run([py, "scripts/measure_demo_plugins.py", "--data", str(data), "--frozen", str(frozen)])
        run([py, "scripts/measure_causal.py", "--data", str(data)], env=env)
        run([py, "scripts/measure_response.py", "--data", str(data), "--frozen", str(frozen)])
        run([py, "scripts/audit_card.py", "--out", str(ROOT / "docs/AUDIT_CARD.md")])

    run([py, "scripts/report_card.py"])
    card = ROOT / "REPORT_CARD.md"
    print("\n" + card.read_text())
    print(f"\nCARD {card}")
    print(f"AUDIT {ROOT / 'docs/AUDIT_CARD.md'}")


if __name__ == "__main__":
    main()
