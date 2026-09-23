"""Final integration: red_king as a witness for white_queen's decision.

WITHOUT red_king : white_queen offline-RL decision on the trajectory logs.
WITH    red_king : red_king multi-step value of the same logs as an added
                   certificate/gate. Combined rule:
                     DEPLOY iff white_queen deploys
                             AND red_king value beats model-free on held-out
                             (i.e., the world model corroborates the policy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[0] / "data" / "seq_email.npz"


def main(n_max=4000, seed=0):
	z = np.load(DATA)
	rng = np.random.default_rng(seed)
	idx = rng.choice(len(z["S"]), size=min(n_max, len(z["S"])), replace=False)
	src = {"obs": z["S"][idx], "act": z["A"][idx], "rew": z["R"][idx],
		   "next_obs": z["S2"][idx], "done": z["D"][idx]}
	nA = int(z["A"].max()) + 1
	from white_queen.tribunal.ope.pipeline import run
	rep = run(src, algorithms=("iql",), nA=nA, fast=True, ensemble_K=2,
			  offline_steps=1200, seed=seed)
	dep = rep.get("deployed")
	if isinstance(dep, dict):
		dep_summary = dep
	else:
		dep_summary = {k: rep[k] for k in ("deployed", "bar", "behavior_mean") if k in rep}
	from red_king.multistep import run as ms
	r = ms(seed=seed)
	corroborates = bool(r["rho_mb"] > r["rho_mf"])
	dep0 = dep_summary.get("deployed")
	deployed_bool = bool(dep0) if dep0 is not None else False
	combined = bool(deployed_bool and corroborates)
	print("== WHITE_QUEEN DECISION: WITHOUT vs WITH red_king ==")
	print(f"  WITHOUT red_king  : {dep_summary}")
	print(f"  WITH    red_king  : rho_mb={r['rho_mb']:+.3f} vs rho_mf={r['rho_mf']:+.3f} "
		  f"-> corroborates={corroborates}")
	print(f"  COMBINED DEPLOY   : {combined}")
	return {"without": dep_summary, "with_corroborates": corroborates, "combined": combined}


if __name__ == "__main__":
	main()
