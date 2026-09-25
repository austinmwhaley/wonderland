import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sys

sys.path.insert(0, str(ROOT))
import numpy as np
import torch

print("cuda", torch.cuda.is_available(), flush=True)
from white_queen import db
from white_queen.tribunal.ope import estimators as E
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate
from white_queen.tribunal.ope.receipts import behavior_stats

cfg = dict(PRESETS["quick"])
env = make_env("cartpole", seed=999)


class U:
    def act(self, s, eval=True):
        return 0

    def action_probs(self, o, temperature=1.0):
        o = np.asarray(o)
        return np.full((len(o), 2), 0.5, dtype=np.float32)


for diet in ["mixed", "novice_only", "expert_only"]:
    data = db.load_diet(str(ROOT / "white_queen/data/white_queen_quick.duckdb"), diet)
    bs = behavior_stats(data, 0.99)
    bar = bs["mean"] + 0.2 * max(bs["std"], 0.05 * abs(bs["mean"]))
    print("== %s behavior=%.1f bar=%.1f ==" % (diet, bs["mean"], bar), flush=True)
    for name in ["uniform", "iql"]:
        cand = (
            U()
            if name == "uniform"
            else load_candidate(
                "iql", env, data, cfg, str(ROOT / "white_queen/verdicts/v15/iql_%s.pt") % diet
            )
        )
        torch.manual_seed(0)
        _, dm, info = E.fit_fqe(data, cand, 0.99, {"device": "cuda"}, temperature=1.0)
        print(
            "  %-8s FQE(Polyak) DM=%6.1f tau=%.3f  %s vs bar %.1f"
            % (name, dm, info["target_tau"], "PASS" if dm > bar else "fail", bar),
            flush=True,
        )
print("POLYAK REAL DONE", flush=True)
