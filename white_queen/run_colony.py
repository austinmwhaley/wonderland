"""Stage 1 entry: fill the colony database. All paths resolve inside
white_queen/ unless given as absolute paths."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from white_queen.config import PRESETS, ROOT  # noqa: E402
from white_queen import db  # noqa: E402
from white_queen.colony.collect import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="White Queen Stage 1: Behavior Colony")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    ap.add_argument("--db", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    args = ap.parse_args()
    cfg = PRESETS[args.preset]

    def _resolve(p, default):
        return p if p and os.path.isabs(p) else str(ROOT / p) if p else default

    db_path = _resolve(args.db, cfg["db_path"])
    run(cfg, db_path, _resolve(args.ckpt_dir, cfg["ckpt_dir"]))
    print("diet stats:", db.diet_stats(db_path))


if __name__ == "__main__":
    main()
