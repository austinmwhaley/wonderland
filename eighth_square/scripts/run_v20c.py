import sys, time
sys.path.insert(0, "/home/austin-whaley/wq")
import torch
torch.set_num_threads(4)
from white_queen.config import PRESETS
from white_queen.tribunal import adjudicate
cfg = dict(PRESETS["quick"])
cfg["fast"] = True
cfg["ensemble_K"] = 3
cfg["ope_fqe_cfg"] = {"steps_max": 50000, "batch": 2048, "eval_every": 2500, "patience": 20}
t0 = time.perf_counter()
adjudicate.run_ope_only(cfg,
    "/home/austin-whaley/wq/white_queen/data/white_queen_quick.db",
    "/home/austin-whaley/wq/white_queen/verdicts/v20c",
    "/home/austin-whaley/wq/white_queen/verdicts/v15", mu_source="logged")
print("V20B ELAPSED %.1f min" % ((time.perf_counter()-t0)/60), flush=True)
print("V20B DONE", flush=True)
