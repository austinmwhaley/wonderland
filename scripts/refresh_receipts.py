"""Refresh all card receipts FAST: train each data revision ONCE (cache), then probe in seconds.

Detached:  nohup python3 scripts/refresh_receipts.py > /tmp/opencode/refresh.log 2>&1 &
Poll:      tail -f /tmp/opencode/refresh.log
"""
import os
import subprocess
import sys

ROOT = "/home/austin-whaley/Documents/wonderland"
D2 = "/tmp/opencode/t0_loop/ref_v3b"
D6 = "/tmp/opencode/t0_loop/ref_v3b_6k"
C2 = "/tmp/opencode/frozen_2k"
C6 = "/tmp/opencode/frozen_6k"


def run(cmd, env=None):
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=False)
    print(f"  exit {p.returncode}", flush=True)


run(["scripts/build_frozen.py", D2, "--out", C2])
run(["scripts/build_frozen.py", D6, "--out", C6])
f2 = dict(os.environ, FROZEN=C2)
f6 = dict(os.environ, FROZEN=C6)
run(["scripts/measure_construction.py"])
run(["scripts/measure_controls.py", D2], env=f2)
run(["scripts/measure_objectives.py", D2], env=f2)
run(["scripts/measure_identity.py", D2], env=f2)
run(["scripts/measure_ceiling.py", D6], env=f6)
run(["scripts/measure_donor.py", D6], env=f6)
run(["scripts/report_card.py"])
print("DONE", flush=True)
