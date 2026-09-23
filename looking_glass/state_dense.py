"""Dense state table — donor states at DAILY/WEEKLY cadence (Layer B).

Runs the frozen CFM over each sample-B customer's recent history once and emits
the per-step state at every weekly (and daily) epoch, so the decision log can be
built at the cadences red_queen must decide over.

Output: looking_glass/artifacts/cfm/state_dense.feather
  customer_key, epoch, cadence ('weekly'|'daily'), embedding
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
CFM_DIR = Path(__file__).resolve().parents[0] / "artifacts" / "cfm"
OUT = CFM_DIR / "state_dense.feather"


def build(cadence="weekly", max_customers=8000):
	import torch
	import polars as pl
	import looking_glass.customer_foundation_model as C
	ckpt = sorted(glob.glob(str(CFM_DIR / "cfm_v*.pt")))[-1]
	blob = torch.load(ckpt, map_location="cpu", weights_only=False)
	vocab = C.EventVocab(blob["vocab"]["et"], blob["vocab"]["brand"], blob["vocab"]["ent"])
	model = C.CFM(vocab, blob["dim"], n_experts=blob.get("n_experts", 1))
	model.load_state_dict(blob["state"]); model.eval()
	cfg = C.CFMConfig(sample_customers=max_customers)
	cfg.seq_len = 8192          # cover full history for dense epochs
	df = C._read_stream(cfg)
	keys = C._customer_keys(df, cfg)
	split = C.assign_split(keys, cfg)
	B = [k for k in keys if split[k] == "B"]
	seqs = C.build_sequences(df, B, cfg, split, with_anchors=False)
	step = 7 * 86400.0 if cadence == "weekly" else 86400.0
	rows_k, rows_e, rows_c, rows_v = [], [], [], []
	with torch.no_grad():
		for seq in seqs:
			if len(seq["ts"]) < 3:
				continue
			y, _ = model(seq)                      # (T, dim) per-step state readout
			ts = np.asarray(seq["ts"], dtype=np.float64)
			t = ts[0] + step
			while t <= ts[-1]:
				idx = int(np.searchsorted(ts, t, side="right")) - 1
				if idx >= 0:
					rows_k.append(seq["customer"]); rows_e.append(float(t))
					rows_c.append(cadence); rows_v.append(y[idx].tolist())
				t += step
	out = pl.DataFrame({"customer_key": rows_k, "epoch": rows_e,
						"cadence": rows_c, "embedding": rows_v})
	OUT.parent.mkdir(parents=True, exist_ok=True)
	out.write_ipc(str(OUT))
	return {"rows": out.height, "customers": len(set(rows_k)), "dim": len(rows_v[0]) if rows_v else 0,
			"cadence": cadence, "out": str(OUT)}


def main(argv=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("--cadence", default="weekly", choices=["weekly", "daily"])
	ap.add_argument("--customers", type=int, default=8000)
	a = ap.parse_args(argv)
	print("== DENSE STATE TABLE ==")
	for k, v in build(a.cadence, a.customers).items():
		print(f"  {k:12s}: {v}")


if __name__ == "__main__":
	main()
