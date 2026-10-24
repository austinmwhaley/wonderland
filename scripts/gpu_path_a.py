"""Run inside the dedicated `.venv-cudf`: PATH A aggregation on the polars-CUDF engine.

Usage: gpu_path_a.py --base base.parquet --customers customers.parquet --as-of ISO --out DIR
Writes features.parquet, labels.parquet, wide.parquet, engine.json (engine state + reason).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from rabbit_hole.generators.path_a_build import build_path_a  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--customers", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = pl.read_parquet(a.base)
    customers = pl.read_parquet(a.customers)
    features, labels, wide, state = build_path_a(
        base, customers, datetime.fromisoformat(a.as_of), engine="gpu"
    )
    out = Path(a.out)
    features.write_parquet(out / "features.parquet")
    labels.write_parquet(out / "labels.parquet")
    wide.write_parquet(out / "wide.parquet")
    (out / "engine.json").write_text(json.dumps(state, indent=2))
    print(json.dumps({"engine": state.get("used"), "rows": wide.height, "cols": wide.width}))


if __name__ == "__main__":
    main()
