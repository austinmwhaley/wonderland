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
from white_queen.tribunal.ope.model_based import learn_dynamics, rollout_estimate
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate

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
    step_fn, info = learn_dynamics(
        data,
        cfg={"steps_max": 8000, "eval_every": 1000, "patience": 3, "batch": 512},
        cache_dir=None,
    )
    iql = load_candidate("iql", env, data, cfg, f"{ROOT}/white_queen/verdicts/v15/iql_{diet}.pt")
    ru = rollout_estimate(data, U(), 0.99, step_fn, sim_min=100, sim_max=200)
    ri = rollout_estimate(data, iql, 0.99, step_fn, sim_min=100, sim_max=200)
    print(
        f"{diet}: uniform MB={ru['mb']:.1f} (done={ru['done_hits']}) | iql MB={ri['mb']:.1f} (done={ri['done_hits']}) | val_mse={info['val_mse']:.4f}",
        flush=True,
    )
print("MB VALIDATE DONE", flush=True)
