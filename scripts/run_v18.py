import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sys

sys.path.insert(0, str(ROOT))
import torch

print("cuda:", torch.cuda.is_available(), flush=True)
from white_queen.config import PRESETS
from white_queen.tribunal import adjudicate

cfg = PRESETS["quick"]
reps = adjudicate.run_ope_only(
    cfg,
    str(ROOT / "white_queen/data/white_queen_quick.duckdb"),
    str(ROOT / "white_queen/verdicts/v18"),
    str(ROOT / "white_queen/verdicts/v15"),
    mu_source="logged",
)
print("V18 ALL DONE", flush=True)
