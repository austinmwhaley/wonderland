"""Run the fast axis screen over all configs and print a compact table.

nohup python3 scripts/screen_axes.py > /tmp/opencode/screen.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = os.environ.get("SCREEN_DATA", "/tmp/opencode/t0_loop/ref_v3b")
EPOCHS = os.environ.get("SCREEN_EPOCHS", "4")

CONFIGS = [
    ("baseline", ""),
    ("measure", "measure"),
    ("suffix", "suffix"),
    ("slow", "slow"),
    ("readout", "readout"),
    ("query", "query"),
    ("multicut", "multicut"),
    ("tokbatch", "tokbatch"),
    ("all", "measure,query,suffix,slow,readout,multicut,tokbatch"),
]


def main():
    rows = []
    for tag, axes in CONFIGS:
        p = subprocess.run(
            [
                sys.executable,
                "scripts/screen_one.py",
                "--axes",
                axes,
                "--epochs",
                EPOCHS,
                "--data",
                DATA,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        line = [x for x in p.stdout.splitlines() if x.startswith("SCREEN ")]
        if not line:
            print(f"{tag:9s} FAILED\n{p.stderr[-500:]}", flush=True)
            continue
        r = json.loads(line[-1][len("SCREEN ") :])
        rows.append((tag, r))
        print(
            f"{tag:9s} v90 {r['value90']['margin']:+.4f}[{r['value90']['lo']:+.4f}] "
            f"o90 {r['orders90']['margin']:+.4f} churn {r['churn90']['margin']:+.4f} | "
            f"cat {r['id_category']:.3f}/{r['id_category_table']:.3f} ch {r['id_channel']:.3f} | "
            f"next {r['next_skill']:+.4f} | anom {r['anomaly_auc']:.3f} retr {r['retrieval']:.3f} "
            f"cnn {r['churn_nn']:.3f}",
            flush=True,
        )
    Path("/tmp/opencode/screen_results.json").write_text(json.dumps(rows, indent=1) + "\n")
    print("SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
