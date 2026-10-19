"""Emit the static/definition receipts the objectives scorecard consumes.

These are grounded in test runs, not hand-filled:
  * schema suite: the degenerate-schema pipeline test (`tests/test_schema_suite.py`)
  * accessibility: the simple per-field probe (`tests/test_cfm_generic_fields.py`)
  * encoder definition: self-supervised + frozen donor + schema-universal

Usage:
  python3 scripts/quality_receipts.py --out-dir <dir>
Writes <dir>/{schema_suite,accessibility,encoder}.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "pytest", *args, "-q"], cwd=ROOT).returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, dest="out_dir")
    ap.add_argument("--ckpt-dir", default=None, dest="ckpt_dir")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema_ok = _run("tests/test_schema_suite.py", "-o", "addopts=") == 0
    access_ok = (
        _run(
            "tests/test_cfm_generic_fields.py",
            "-k",
            "simple_probe_recovers_every_field",
            "-o",
            "addopts=",
        )
        == 0
    )
    cases = [
        "minimal",
        "all_categorical",
        "high_cardinality",
        "mixed_dirty",
        "null_heavy",
        "nested_only",
        "multi_field",
        "single_event",
    ]
    (out / "schema_suite.json").write_text(
        json.dumps(
            {
                "passed": len(cases) if schema_ok else 0,
                "total": len(cases),
                "cases": cases,
                "evidence": "tests/test_schema_suite.py",
            },
            indent=1,
        )
        + "\n"
    )
    (out / "accessibility.json").write_text(
        json.dumps(
            {
                "passed": 1 if access_ok else 0,
                "total": 1,
                "evidence": "tests/test_cfm_generic_fields.py::test_simple_probe_recovers_every_field",
            },
            indent=1,
        )
        + "\n"
    )
    if a.ckpt_dir and a.tag:
        rc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "encoder_receipt.py"),
                "--ckpt-dir",
                a.ckpt_dir,
                "--tag",
                a.tag,
                "--schema-suite",
                str(out / "schema_suite.json"),
                "--out",
                str(out / "encoder.json"),
            ],
            cwd=ROOT,
        ).returncode
    else:
        (out / "encoder.json").write_text(
            json.dumps(
                {
                    "schema_universal": bool(schema_ok),
                    "self_supervised": False,
                    "frozen_donor": False,
                    "error": "pass --ckpt-dir/--tag for a measured encoder receipt",
                },
                indent=1,
            )
            + "\n"
        )
        rc = 1
    print(json.dumps({"schema_ok": schema_ok, "access_ok": access_ok, "encoder_rc": rc}, indent=1))
    return 0 if (schema_ok and access_ok and rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
