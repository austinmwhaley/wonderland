"""Runs inside .venv-cudf: stream load on the GPU engine, writes the prep artifact.

Usage: gpu_prep.py <data-dir> <out.npz>  (env LOAD_ENGINE=gpu|cpu)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from stream_loader import load_data


def main():
    data, out = sys.argv[1], sys.argv[2]
    eng = os.environ.get("LOAD_ENGINE", "gpu")
    t0 = time.perf_counter()
    B, A, keys, seqs, vs, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons = load_data(data, engine=eng)
    dt = time.perf_counter() - t0
    ks = [seqs[k] for k in keys]
    spec = {
        "keys": keys,
        "vs": vs,
        "et_v": et_v,
        "catspec": catspec,
        "num": [list(x) for x in num],
        "taus": taus.tolist(),
        "periods": periods,
        "horizons": horizons.tolist(),
        "p_mask": p_mask,
        "as_of_ts": as_of_ts,
        "rev": __import__("stream_loader")._prep_rev(data),
        "engine": eng,
        "runtime_s": round(dt, 1),
    }
    np.savez(
        out,
        et=np.concatenate([x[0] for x in ks]) if ks else np.zeros(0, np.int64),
        gcat=np.concatenate([x[1] for x in ks]) if ks else np.zeros((0, 0), np.int64),
        gnum=np.concatenate([x[2] for x in ks]) if ks else np.zeros((0, 0), np.float32),
        time=np.concatenate([x[3] for x in ks]) if ks else np.zeros((0, 0), np.float32),
        gaps=np.concatenate([x[4] for x in ks]) if ks else np.zeros(0, np.float64),
        tss=np.concatenate([x[5] for x in ks]) if ks else np.zeros(0, np.float64),
        graw=np.concatenate([x[6] for x in ks]) if ks else np.zeros((0, 0), np.float32),
        lens=np.array([len(x[5]) for x in ks], np.int64),
        spec=np.frombuffer(json.dumps(spec).encode(), dtype=np.uint8),
    )
    print(f"gpu prep: {dt:.1f}s engine={eng} n={len(keys)} events={len(seqs[keys[0]][5]) if ks else 0}")


if __name__ == "__main__":
    main()
