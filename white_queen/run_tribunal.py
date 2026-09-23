"""Stage 2 entry: run The Tribunal on a colony database. All paths resolve
inside white_queen/ unless given as absolute paths."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from white_queen.config import PRESETS, ROOT  # noqa: E402
from white_queen.tribunal.adjudicate import run  # noqa: E402


def _resolve(p, default):
    if p:
        return p if os.path.isabs(p) else str(ROOT / p)
    return default


def main():
    ap = argparse.ArgumentParser(description="White Queen Stage 2: The Tribunal")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    run(cfg, _resolve(args.db, cfg["db_path"]), _resolve(args.out, cfg["verdict_dir"]))


if __name__ == "__main__":
    main()
