"""Run the Layer C plugin suite (independent plugins over the frozen CFM)."""
from __future__ import annotations

import argparse


def main(argv=None):
	ap = argparse.ArgumentParser(description="Layer C plugin suite")
	ap.add_argument("--window", type=int, default=365, help="label window in days")
	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--only", default="all",
					choices=["all", "supervised", "unsupervised", "white_queen"])
	a = ap.parse_args(argv)
	results = {}
	if a.only in ("all", "supervised"):
		from . import supervised
		results["supervised"] = supervised.run(a.window, a.seed)[0]
	if a.only in ("all", "unsupervised"):
		from . import segmentation
		results["unsupervised"] = segmentation.run(a.window, a.seed)[0]
	if a.only in ("all", "white_queen"):
		from . import white_queen_plugin
		results["white_queen"] = white_queen_plugin.run(a.window, a.seed)[0]
	print("\n== PLUGIN SUITE (%dd) ==" % a.window)
	for k, v in results.items():
		print(f"{k:16s} {'PASS' if v else 'FAIL'}")
	ok = all(results.values())
	print("SUITE VERDICT:", "PASS" if ok else "FAIL")
	return 0 if ok else 1


if __name__ == "__main__":
	raise SystemExit(main())
