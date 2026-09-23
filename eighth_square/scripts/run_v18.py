import sys
sys.path.insert(0, "/home/austin-whaley/wq")
import torch
print("cuda:", torch.cuda.is_available(), flush=True)
from white_queen.config import PRESETS
from white_queen.tribunal import adjudicate
cfg = PRESETS["quick"]
reps = adjudicate.run_ope_only(cfg, "/home/austin-whaley/wq/white_queen/data/white_queen_quick.db",
                               "/home/austin-whaley/wq/white_queen/verdicts/v18",
                               "/home/austin-whaley/wq/white_queen/verdicts/v15",
                               mu_source="logged")
print("V18 ALL DONE", flush=True)
