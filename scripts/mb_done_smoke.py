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
from white_queen.tribunal.ope.model_based import learn_dynamics, rollout_estimate
from white_queen.config import PRESETS
from environments.registry import make_env
from white_queen.tribunal.candidates import load_candidate

cfg = dict(PRESETS["quick"])
cfg["device"] = "cpu"
data = db.load_diet(str(ROOT / "white_queen/data/white_queen_quick.db"), "novice_only")
env = make_env("cartpole", seed=999)
tiny = {
    "steps_max": 400,
    "eval_every": 100,
    "patience": 3,
    "batch": 128,
    "hidden": 48,
    "device": "cpu",
}
step_fn, info = learn_dynamics(data, cfg=tiny, cache_dir=None)
print("dyn info:", {k: info[k] for k in ("val_mse", "termination_head", "steps")}, flush=True)


class U:
    def act(self, s, eval=True):
        return 0

    def action_probs(self, o, temperature=1.0):
        o = np.asarray(o)
        return np.full((len(o), 2), 0.5, dtype=np.float32)


for name, cand in [
    ("uniform", U()),
    (
        "iql(expert)",
        load_candidate(
            "iql", env, data, cfg, str(ROOT / "white_queen/verdicts/v15/iql_novice_only.pt")
        ),
    ),
]:
    r = rollout_estimate(data, cand, 0.99, step_fn, temperature=1.0, sim_min=40, sim_max=80)
    print(f"{name}: MB={r['mb']:.1f} done_hits={r.get('done_hits')} sims={r['sims']}", flush=True)
