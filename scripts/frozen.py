"""Train-once cache for the frozen encoder, with a run manifest.

Build once with `scripts/build_frozen.py <data> --seed S --out <cache>`.
Measurement scripts load it when the env var FROZEN is set -> no retraining.

Every cache writes `<path>.manifest.json` describing the exact run that produced it.
Measurement code must take its configuration from that manifest, never from whichever
process ran last; `assert_axes` refuses to mix a cache with a differently-configured
process.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "demo"))
sys.path.insert(0, str(_HERE.parent))
import auto_encoder as AE  # noqa: E402


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_HERE.parent, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def manifest_path(path) -> Path:
    return Path(str(path) + ".manifest.json")


def get_manifest(path):
    """Read the run manifest; fall back to the checkpoint for older caches."""
    p = manifest_path(path)
    if p.exists():
        return json.loads(p.read_text())
    ck = torch.load(str(path) + ".pt", map_location="cpu", weights_only=False)
    return {
        "data": ck.get("data"),
        "seed": ck.get("seed"),
        "axes": sorted(ck.get("axes") or []),
        "source": "checkpoint fallback (no manifest)",
    }


def assert_axes(path, expected):
    """Refuse to mix a cache built with different axes than the running process."""
    man = get_manifest(path)
    got = sorted(man.get("axes") or [])
    exp = sorted(expected or [])
    if got != exp:
        raise RuntimeError(
            f"cache axes {got} != process axes {exp} for {path}; "
            "refusing to mix runs (regenerate the cache or fix WQ_AXES)"
        )
    return man


def save(
    data,
    seed,
    path,
    epochs=12,
    d=128,
    layers=2,  # principle-ok: pinned fallback, receipted in manifest
):
    E, Xtab, labels, hold, seg, model, vs, B, best, BLK = AE.train_encoder(
        data, seed=seed, epochs=epochs, d=d, layers=layers
    )
    labels_arr = {f"label_{k}": np.asarray(v) for k, v in (labels or {}).items()}
    np.savez(
        str(path) + ".npz",
        E=np.asarray(E),
        Xtab=np.asarray(Xtab),
        BLK=np.asarray(BLK),
        hold=np.array(hold, dtype=object),
        seg=np.array(seg, dtype=object),
        best=np.array([float(best)]),
        **labels_arr,
    )
    # Persist the lawful measure vocabulary so the artifact is self-contained.
    from looking_glass import measure_state as MS

    _B2, _A2, keys2, seqs2, vs2, *_rest = AE.load_data(data)
    hold_set = set(hold)
    train_keys = [k for k in keys2 if k not in hold_set]
    builder = MS.MeasureBuilder(seqs2, train_keys, vs2)
    measure_spec = MS.measure_to_dict(builder)

    ckpt = {
        "model": model,
        "vs": vs,
        "data": data,
        "seed": seed,
        "axes": sorted(AE.AXES),
        "epochs": epochs,
        "d": d,
        "layers": layers,
        "measure": measure_spec,
    }
    torch.save(ckpt, str(path) + ".pt")

    from looking_glass import state_layout as SL

    layout = SL.describe(vs, model)
    man = {
        "data": str(data),
        "data_rev": Path(str(data)).name,
        "seed": int(seed),
        "axes": sorted(AE.AXES),
        "epochs": int(epochs),
        "d": int(model.d),
        "layers": int(len(model.ssm)),
        "e": int(model.ssm[0].e),
        "n_read": int(model.n_read),
        "E_dim": int(model.d),
        "BLK_dim": int(layout["blk_dim"]),
        "state_dim": int(layout["full_dim"]),
        "layout": layout,
        "as_of_ts": float(vs.get("as_of_ts", 0.0)),
        "id_idx": list(vs.get("id_idx") or []),
        "rel_idx": list(vs.get("rel_idx") or []),
        "rel_source": vs.get("rel_source"),
        "exo_idx": list(vs.get("exo_idx") or []),
        "exo_source": vs.get("exo_source"),
        "measure_dim": int(builder.dim),
        "measure_coverage": float(builder.coverage),
        "best_val": float(best),
        "config_source": "pinned baseline fallback (d/layers/epochs supplied by build_frozen)",
        "git_sha": _git_sha(),
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cache_sha256": _sha256(str(path) + ".pt"),
        "source": "build_frozen",
    }
    manifest_path(path).write_text(json.dumps(man, indent=1) + "\n")
    return str(path)


def load(path):
    ck = torch.load(str(path) + ".pt", map_location=AE.DEV, weights_only=False)
    model = ck["model"]
    model.eval()
    z = np.load(str(path) + ".npz", allow_pickle=True)
    # Fail safe at the boundary: a non-finite frozen state must never be sanitized
    # into a measurement (Robust; no silent masking).
    for name in ("E", "BLK"):
        arr = z[name]
        if not np.isfinite(arr).all():
            raise FloatingPointError(
                f"frozen cache {path}: non-finite {name}; refusing to score a broken artifact"
            )
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
