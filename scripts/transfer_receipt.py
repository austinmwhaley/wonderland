"""T1.transfer receipt: does the frozen E add signal over raw/RFM+GBM? (DEC-212)"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="E vs raw/RFM+GBM transfer receipt")
    ap.add_argument("--db", required=True)
    ap.add_argument("--ckpt-dir", required=True, dest="ckpt_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--as-of", required=True, dest="as_of")
    ap.add_argument("--customers", type=int, default=6000)
    ap.add_argument("--horizon-days", type=int, default=90, dest="horizon_days")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from looking_glass.cfm_state import load_frozen_encoder
    from looking_glass.head_battery import run_head_battery

    model, cfg = load_frozen_encoder(a.tag, a.ckpt_dir)
    cfg.attr_fields = tuple(getattr(model, "attr_fields", ()) or ())
    cfg.numeric_fields = tuple(getattr(model, "numeric_fields", ()) or ())
    cfg.db, cfg.as_of, cfg.sample_customers = a.db, a.as_of, a.customers
    rows, _receipt = run_head_battery(model, cfg, horizon_days=a.horizon_days, seeds=(0, 1, 2))
    n_add = sum(1 for r in rows if r.get("verdict") == "PASS")
    n_fail = sum(1 for r in rows if r.get("verdict") == "FAIL")
    n_ns = sum(1 for r in rows if r.get("verdict") == "NO_SIGNAL")
    # Layer-0 "transferable" = E adds over raw on at least one task AND hurts none.
    beats = bool(n_add > 0 and n_fail == 0)
    rec = {
        "passed": int(beats),
        "total": 1,
        "baseline": "RFM/raw aggregates + GBM",
        "beats_traditional": beats,
        "heads_pass": n_add,
        "heads_fail": n_fail,
        "heads_no_signal": n_ns,
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, indent=2))
    return rec


if __name__ == "__main__":
    main()
