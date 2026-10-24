"""Train-once cache for the frozen encoder.

Build once with `scripts/build_frozen.py <data> --seed S --out <cache>`.
Measurement scripts load it when the env var FROZEN is set -> no retraining.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "demo"))
sys.path.insert(0, str(_HERE.parent))
import auto_encoder as AE  # noqa: E402


def save(data, seed, path, epochs=12, d=128, layers=2):
    E, Xtab, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(
        data, seed=seed, epochs=epochs, d=d, layers=layers
    )
    labels_arr = {f"label_{k}": np.asarray(v) for k, v in labels.items()}
    np.savez(
        str(path) + ".npz",
        E=np.nan_to_num(E),
        Xtab=np.nan_to_num(Xtab),
        BLK=np.nan_to_num(BLK),
        hold=np.array(hold, dtype=object),
        seg=np.array(seg, dtype=object),
        best=np.array([float(best)]),
        **labels_arr,
    )
    torch.save(
        {"model": model, "vs": vs, "data": data, "seed": seed, "axes": sorted(AE.AXES)},
        str(path) + ".pt",
    )
    return str(path)


def load(path):
    ck = torch.load(str(path) + ".pt", map_location=AE.DEV, weights_only=False)
    model = ck["model"]
    model.eval()
    z = np.load(str(path) + ".npz", allow_pickle=True)
    labels = {k[len("label_") :]: z[k] for k in z.files if k.startswith("label_")}
    return (
        z["E"],
        z["Xtab"],
        labels,
        list(z["hold"]),
        list(z["seg"]),
        model,
        ck["vs"],
        None,
        float(z["best"][0]),
        z["BLK"],
    )
