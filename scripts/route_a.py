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
from white_queen.tribunal.ope.model_based import learn_dynamics, rollout_estimate
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate
from white_queen.tribunal.ope.receipts import behavior_stats

cfg = dict(PRESETS["quick"])
env = make_env("cartpole", seed=999)
data = db.load_diet(str(ROOT / "white_queen/data/white_queen_quick.db"), "mixed")
bs = behavior_stats(data, 0.99)
print("behavior mean", round(bs["mean"], 1), "std", round(bs["std"], 1), flush=True)


class U:
    def act(self, s, eval=True):
        return 0

    def action_probs(self, o, temperature=1.0):
        o = np.asarray(o)
        return np.full((len(o), 2), 0.5, dtype=np.float32)


# 1) FQE with large budget for iql and uniform
for name, cand in [
    ("uniform", U()),
    (
        "iql",
        load_candidate("iql", env, data, cfg, str(ROOT / "white_queen/verdicts/v15/iql_mixed.pt")),
    ),
]:
    for steps in (300000,):
        torch.manual_seed(0)
        _, dm, info = E.fit_fqe(
            data,
            cand,
            0.99,
            {"steps_max": steps, "eval_every": 5000, "patience": 30, "batch": 512, "hidden": 128},
            temperature=1.0,
        )
        print(
            f"FQE {name} steps_max={steps}: DM={dm:.1f} stopped={info['stopped']} val={info['val_bellman']}",
            flush=True,
        )
# 2) MB with large dynamics budget
step_fn, info = learn_dynamics(
    data,
    cfg={"steps_max": 50000, "eval_every": 2500, "patience": 10, "batch": 512, "hidden": 128},
    cache_dir=None,
)
print("dyn val_mse", info["val_mse"], flush=True)
for name, cand in [
    ("uniform", U()),
    (
        "iql",
        load_candidate("iql", env, data, cfg, str(ROOT / "white_queen/verdicts/v15/iql_mixed.pt")),
    ),
]:
    r = rollout_estimate(data, cand, 0.99, step_fn, sim_min=200, sim_max=400)
    print(f"MB {name}: {r['mb']:.1f} done={r['done_hits']}", flush=True)
print("ROUTE A DONE", flush=True)
