"""red_queen scheduler — materialize per-epoch counts into concrete touches.

For each (customer, weekly epoch) it:
  * sets the number of touches from the target frequency (derived from observed
    behaviour, capped),
  * SPREADS them within the epoch (within-epoch scheduling),
  * assigns the action MIX (arm composition) from the validated causal ranking.
Emits a concrete schedule: customer_key, ts, arm.  (Capability demo; the certified
path gates on white_queen DEPLOY.)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

LOG = Path(__file__).resolve().parents[0] / "artifacts" / "decision_log_weekly.npz"
OUT = Path(__file__).resolve().parents[0] / "artifacts" / "touch_schedule.json"
CADENCE = np.array([0.2, 0.6, 1.2, 2.0])
WEEK = 7 * 86400.0


def run(max_rows=20000, per_epoch_cap=8):
	from red_queen.engine import _validated_arm_effects
	eff = _validated_arm_effects()
	# action MIX: softmax over the validated causal effect -> proportional touches
	z = eff - eff.max()
	mix = np.exp(z); mix = mix / mix.sum()               # per-arm share of touches
	best = int(eff.argmax())
	z = np.load(LOG)
	CUST = z["customer"]; ET0 = z["epoch_ts"]; A = z["action"][:, 0]
	target = int(np.clip(round(float(A.mean())), 1, per_epoch_cap))
	rows = []
	seen_customers = set()
	for i in range(len(CUST)):
		k = CUST[i]
		if k in seen_customers:
			continue
		seen_customers.add(k)
		base = datetime.fromtimestamp(float(ET0[i]), tz=timezone.utc)
		cum = np.cumsum(mix)
		for j in range(target):
			off = (j + 0.5) / target * WEEK                 # spread within the epoch
			ts = (base + timedelta(seconds=off)).isoformat()
			arm = int(np.searchsorted(cum, (j + 0.5) / target))   # composition by mix
			rows.append({"customer_key": k, "ts": ts, "arm": int(arm)})
		if len(rows) >= max_rows:
			break
	import json
	OUT.parent.mkdir(parents=True, exist_ok=True)
	OUT.write_text(json.dumps({"rows": rows, "target_per_week": target,
							   "mix": [round(float(x), 3) for x in mix],
							   "best_arm": best}))
	by_arm = np.bincount([r["arm"] for r in rows], minlength=len(eff)).tolist()
	print("== RED_QUEEN SCHEDULER (within-epoch + composition) ==")
	print(f"  customers scheduled   : {len(seen_customers)}")
	print(f"  touches               : {len(rows)}  (target {target}/week/customer)")
	print(f"  touches by arm        : {by_arm}")
	print(f"  validated best arm    : {best}   mix {[round(float(x),3) for x in mix]}")
	print(f"  schedule -> {OUT}")
	return {"customers": len(seen_customers), "touches": len(rows),
			"by_arm": by_arm, "best_arm": best}


if __name__ == "__main__":
	run()
