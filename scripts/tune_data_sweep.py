"""Detached data-tune sweep: vary (propensity spread, sequence amplitude), regenerate,
validate, run the 3-seed demo, and record the E-vs-table margins + CIs to a results JSON.

Launch: nohup python3 scripts/tune_data_sweep.py > /tmp/opencode/tune_sweep.log 2>&1 &
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = "/home/austin-whaley/Documents/wonderland"
RES = Path("/tmp/opencode/t0_loop/tune_results.json")
COMBOS = [("0.40", "2.0"), ("0.80", "2.0"), ("0.80", "3.0"), ("1.20", "2.5")]


def run(cmd, env=None):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=False)


def main():
    out = []
    for i, (psd, amp) in enumerate(COMBOS):
        d = f"/tmp/opencode/t0_loop/sw_{i}"
        env = dict(os.environ, WQ_PROP_SD=psd, WQ_SEQ_AMP=amp)
        run([sys.executable, "-m", "rabbit_hole.generators.relational_reference",
             "--n-customers", "2000", "--out", d], env=env)
        run([sys.executable, "scripts/validate_layer_a.py", "--dir", d])
        run([sys.executable, "scripts/demo/demo.py", "--data", d,
             "--epochs", "12", "--seeds", "0,1,2", "--out", f"{d}_out"])
        rec = {"combo": i, "prop_sd": psd, "seq_amp": amp}
        try:
            r = json.load(open(f"{d}_out/demo_run.json"))
            for t in ("value90", "orders90", "churn90"):
                m = np.array([run_[t]["E"] - run_[t]["table"] for run_ in r["runs"]])
                lo, hi = stats.t.interval(0.95, len(m) - 1, loc=m.mean(), scale=stats.sem(m))
                tab = float(np.mean([run_[t]["table"] for run_ in r["runs"]]))
                rec[t] = {"margin": round(float(m.mean()), 4), "lo": round(float(lo), 4),
                          "hi": round(float(hi), 4), "table": round(tab, 4)}
            rec["win"] = bool(rec["value90"]["lo"] > 0 and rec["orders90"]["lo"] > 0)
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)
        out.append(rec)
        RES.write_text(json.dumps(out, indent=1))
        print("RESULT", json.dumps(rec), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
