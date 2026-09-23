"""white_queen on sequential trajectories, with vs without red_king.

WITHOUT red_king : white_queen's own model-free offline-RL OPE (`run`).
WITH    red_king : red_king multi-step rollout value as an MB witness.

Estimand: discounted INCREMENTAL gross margin (AGENTS.md reward standard).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "seq_email.npz"


def main(n_max=8000, seed=0):
	z = np.load(DATA)
	rng = np.random.default_rng(seed)
	idx = rng.choice(len(z["S"]), size=min(n_max, len(z["S"])), replace=False)
	src = {"obs": z["S"][idx], "act": z["A"][idx], "rew": z["R"][idx],
		   "next_obs": z["S2"][idx], "done": z["D"][idx]}
	nA = int(z["A"].max()) + 1
	# WITHOUT red_king: white_queen offline-RL pipeline (model-free-ish OPE)
	from white_queen.tribunal.ope.pipeline import run
	rep = run(src, algorithms=("iql",), nA=nA, fast=True, ensemble_K=2,
			  offline_steps=1500, seed=seed)
	print("== WHITE_QUEEN ON TRAJECTORIES ==")
	print("  without red_king -> keys:", sorted(list(rep.keys()))[:12] if isinstance(rep, dict) else type(rep))
	if isinstance(rep, dict):
		for k in ("deploy", "decision", "behavior_mean", "bar"):
			if k in rep:
				print(f"  without red_king {k}: {rep[k]}")
	# WITH red_king: multi-step rollout value of the greedy policy
	from red_king.red_king.multistep import run as ms
	r = ms(seed=seed)
	print(f"  WITH red_king multi-step value spearman: {r['rho_mb']:+.3f} (vs model-free {r['rho_mf']:+.3f})")
	return rep


if __name__ == "__main__":
	main()
