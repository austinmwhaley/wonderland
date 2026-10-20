"""Report GPU/NVIDIA library availability and utilization (no heavy allocation).

Usage: python3 scripts/gpu_probe.py [--bench-xgb]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-xgb", action="store_true", help="tiny CPU-vs-GPU GBM benchmark")
    a = ap.parse_args(argv)
    out = {}
    try:
        import torch

        out["torch"] = {"version": torch.__version__, "cuda": torch.cuda.is_available()}
        if torch.cuda.is_available():
            out["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "allocated_gib": round(torch.cuda.memory_allocated() / 1e9, 2),
            }
    except Exception as e:  # noqa: BLE001
        out["torch_err"] = str(e)
    for mod in ("cupy", "cudf", "cuml", "xgboost", "pynvml"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "present")
        except Exception as e:  # noqa: BLE001
            out[mod] = f"missing/unavailable: {str(e)[:80]}"
    if a.bench_xgb:
        try:
            import xgboost as xgb
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split

            rng = np.random.default_rng(0)
            X = rng.normal(size=(6000, 200)).astype(np.float32)
            y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=6000) > 0).astype(int)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
            res = {}
            t = time.perf_counter()
            m = HistGradientBoostingClassifier(max_iter=200).fit(Xtr, ytr)
            res["sklearn_cpu_s"] = round(time.perf_counter() - t, 3)
            res["sklearn_auc"] = round(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])), 4)
            try:
                t = time.perf_counter()
                g = xgb.XGBClassifier(
                    n_estimators=200, tree_method="hist", device="cuda", verbosity=0
                ).fit(Xtr, ytr)
                res["xgboost_gpu_s"] = round(time.perf_counter() - t, 3)
                res["xgboost_auc"] = round(float(roc_auc_score(yte, g.predict_proba(Xte)[:, 1])), 4)
            except Exception as e:  # noqa: BLE001
                res["xgboost_err"] = str(e)[:200]
            out["xgb_bench"] = res
        except Exception as e:  # noqa: BLE001
            out["xgb_bench_err"] = str(e)[:200]
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
