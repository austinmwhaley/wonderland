import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sys

sys.path.insert(0, str(ROOT))
import numpy as np
import torch

torch.set_num_threads(1)
from white_queen import db
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate

cfg = dict(PRESETS["quick"])
cfg["device"] = "cpu"
env = make_env("cartpole", seed=999)

for diet in ["mixed", "novice_only", "expert_only"]:
    data = db.load_diet(str(ROOT / "white_queen/data/white_queen_quick.duckdb"), diet)
    eplen = {e: 0 for e in np.unique(data["episode"])}
    for e in np.unique(data["episode"]):
        eplen[e] = int((data["episode"] == e).sum())
    lens = np.array([eplen[e] for e in data["episode"]])
    long_ep = lens > 200
    for name in ["iql", "bc", "cql"]:
        cand = load_candidate(
            name, env, data, cfg, f"{ROOT}/white_queen/verdicts/v15/{name}_{diet}.pt"
        )
        p = cand.action_probs(data["obs"], temperature=1.0)  # argmax of softmax == argmax logits
        am = p.argmax(1)
        agree = (am == data["act"]).mean()
        agree_long = (am[long_ep] == data["act"][long_ep]).mean() if long_ep.any() else float("nan")
        print(
            f"{diet:12s} {name:4s}: argmax==logged act overall={agree:.3f} on long-exp episodes={agree_long:.3f} (n_long={long_ep.sum()})",
            flush=True,
        )
print("COVERAGE DONE", flush=True)
