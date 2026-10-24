"""Axis sweep: build the frozen encoder under each axis config and run the battery.

Each config is one added axis (cumulative), matching the doctrine "one axis per run".
Runs are sequential (one GPU job at a time).  Results land in /tmp/opencode/sweep/.

    nohup python3 scripts/axis_sweep.py > /tmp/opencode/sweep.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = os.environ.get("SWEEP_DATA", "/tmp/opencode/t0_loop/ref_v3b_6k")
OUT = Path(os.environ.get("SWEEP_OUT", "/tmp/opencode/sweep"))
EPOCHS = os.environ.get("SWEEP_EPOCHS", "12")

# cumulative axis sets: each row adds one axis to the previous accepted stack
CONFIGS = [
    ("baseline", ""),
    ("measure", "measure"),
    ("query", "measure,query"),
    ("suffix", "measure,query,suffix"),
    ("slow", "measure,query,suffix,slow"),
    ("readout", "measure,query,suffix,slow,readout"),
    ("multicut", "measure,query,suffix,slow,readout,multicut"),
    ("tokbatch", "measure,query,suffix,slow,readout,multicut,tokbatch"),
]


def run(cmd, env):
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, env=env)
    return p.returncode


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for tag, axes in CONFIGS:
        t0 = time.time()
        cache = OUT / f"frozen_{tag}"
        env = dict(os.environ, WQ_AXES=axes)
        rc = run(
            [
                sys.executable,
                "scripts/build_frozen.py",
                DATA,
                "--out",
                str(cache),
                "--epochs",
                EPOCHS,
            ],
            env,
        )
        if rc != 0:
            results[tag] = {"axes": axes, "error": "build failed"}
            (OUT / "sweep_results.json").write_text(json.dumps(results, indent=1) + "\n")
            continue
        rc = run(
            [
                sys.executable,
                "scripts/measure_battery.py",
                "--frozen",
                str(cache),
                "--data",
                DATA,
                "--tag",
                tag,
                "--out",
                str(OUT),
            ],
            env,
        )
        rec = {"axes": axes, "build_s": round(time.time() - t0, 1)}
        bp = OUT / f"battery_{tag}.json"
        if rc == 0 and bp.exists():
            rec.update(json.loads(bp.read_text()))
        else:
            rec["error"] = "battery failed"
        results[tag] = rec
        (OUT / "sweep_results.json").write_text(json.dumps(results, indent=1) + "\n")
        print(f"=== {tag} ({axes}) {rec.get('build_s')}s ===", flush=True)
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
