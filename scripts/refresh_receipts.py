"""Refresh every card receipt from one frozen encoder, then render both cards.

Train the 6k demo dataset ONCE (frozen cache), then run each measurement against
that cache.  All receipts use the SAME encoder config (WQ_AXES), so the card is
self-consistent.

    WQ_AXES="measure,query,..." nohup python3 scripts/refresh_receipts.py > /tmp/opencode/refresh.log 2>&1 &
"""

import os
import subprocess
import sys

ROOT = "/home/austin-whaley/Documents/wonderland"
D6 = os.environ.get("WONDERLAND_DEMO_DATA", "/tmp/opencode/t0_loop/ref_v3b_6k")
C6 = os.environ.get("WONDERLAND_DEMO_CACHE", "/tmp/opencode/frozen_6k")
AXES = os.environ.get("WQ_AXES", "")


def run(cmd, env=None):
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=False)
    print(f"  exit {p.returncode}", flush=True)


base = dict(os.environ, WQ_AXES=AXES)
run(["scripts/build_frozen.py", D6, "--out", C6], env=base)
f6 = dict(base, FROZEN=C6)

run(["scripts/measure_data.py", "--data", D6], env=base)
run(["scripts/measure_construction.py"], env=base)
run(["scripts/measure_controls.py", D6], env=f6)
run(["scripts/measure_objectives.py", D6], env=f6)
run(["scripts/measure_identity.py", D6], env=f6)
run(["scripts/measure_ceiling.py", D6], env=f6)
run(["scripts/measure_donor.py", D6], env=f6)
run(["scripts/measure_demo_plugins.py", "--data", D6, "--frozen", C6], env=base)
run(["scripts/measure_causal.py", "--data", D6], env=f6)
run(["scripts/measure_response.py", "--data", D6, "--frozen", C6], env=base)
run(["scripts/audit_card.py", "--out", "docs/AUDIT_CARD.md"], env=base)
run(["scripts/report_card.py"], env=base)
print("DONE", flush=True)
