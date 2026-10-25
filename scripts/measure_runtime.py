"""Runtime / memory receipt for the frozen foundation serving path.

Profiles, on the real demo data:
  * cached stream load,
  * lawful measure-state construction,
  * frozen encoder forward (state generation),
  * artifact save and load,
  * a supervised head fit on the published state.

Writes docs/report_cards/experiments/runtime.json. No training, no labels in B.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/demo"))

OUT = ROOT / "docs/report_cards/experiments/runtime.json"


def _rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/t0_loop/ref_v3b_6k")
    ap.add_argument("--frozen", default="/tmp/opencode/frozen_6k")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n", type=int, default=600, help="entities for the forward benchmark")
    args = ap.parse_args()

    import auto_encoder as AE
    from looking_glass.foundation import FrozenFoundation

    rec = {"data": args.data, "frozen": args.frozen, "n_entities": args.n}

    t0 = time.perf_counter()
    _b, _a, keys, seqs, vs, *_ = AE.load_data(args.data)
    rec["load_data_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    art = FrozenFoundation.from_cache(args.frozen)
    rec["load_artifact_s"] = round(time.perf_counter() - t0, 2)

    hold = sorted(keys)[: args.n]
    train = [k for k in keys if k not in set(hold)]

    t0 = time.perf_counter()
    art.fit_measure(seqs, train)
    rec["fit_measure_s"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    M = art.measure.transform(seqs, hold)
    rec["measure_transform_s"] = round(time.perf_counter() - t0, 2)
    rec["measure_dim"] = int(M.shape[1])
    rec["measure_rows_per_s"] = round(len(hold) / max(rec["measure_transform_s"], 1e-9), 1)

    views = ["shape", "learned_state", "forecast", "uncertainty", "heads"]
    t0 = time.perf_counter()
    out = art.state(seqs, hold, views=views)
    rec["state_forward_cold_s"] = round(time.perf_counter() - t0, 2)
    warm = []
    for _ in range(3):
        t0 = time.perf_counter()
        art.state(seqs, hold, views=views)
        warm.append(time.perf_counter() - t0)
    rec["state_forward_warm_s"] = round(min(warm), 3)
    rec["state_rows_per_s"] = round(len(hold) / max(min(warm), 1e-9), 1)
    rec["state_dims"] = {k: int(v.shape[1]) for k, v in out.items()}

    t0 = time.perf_counter()
    art.save("/tmp/opencode/artifact_profile")
    rec["artifact_save_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    FrozenFoundation.load("/tmp/opencode/artifact_profile")
    rec["artifact_load_s"] = round(time.perf_counter() - t0, 2)

    # supervised head on the published state (cold fit, 5-fold)
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    import frozen as F

    _E, Xtab, labels, hol, _seg, _m, _v, _B, _best, BLK = F.load(args.frozen)
    X = np.nan_to_num(np.hstack([_E, BLK]))
    y = np.asarray(labels["y_value_90"], float)
    t0 = time.perf_counter()
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(X[tr], y[tr])
        m.predict(X[te])
    rec["head_fit_5fold_s"] = round(time.perf_counter() - t0, 2)

    rec["peak_rss_mb"] = _rss_mb()
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
