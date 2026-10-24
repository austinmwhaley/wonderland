"""Build the frozen-encoder cache so measurement scripts don't retrain.

Usage: python3 scripts/build_frozen.py <data> --seed 0 --out /tmp/opencode/frozen_2k
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=12)
a = ap.parse_args()
frozen.save(a.data, a.seed, a.out, epochs=a.epochs)
print("BUILT", a.out)
