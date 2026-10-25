"""Frozen foundation artifact: one stream in -> one reusable, versioned model out.

The artifact is more than a weight file. It carries:
  * the frozen learned encoder,
  * the resolved schema/config (`vs`),
  * the state layout (named views),
  * the lawful measure vocabulary,
  * provenance (data, seed, axes, code version).

Consumers call `state(...)` and receive named views. They never mutate the encoder.

    art = FrozenFoundation.load("/path/to/artifact")
    z = art.state(seqs, keys, views=["shape", "measure"])
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# The training path lives under scripts/demo; make the artifact importable without
# requiring callers to configure sys.path.
_DEMO_DIR = Path(__file__).resolve().parents[1] / "scripts" / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from looking_glass import measure_state as MS  # noqa: E402
from looking_glass import state_layout as SL  # noqa: E402
from looking_glass import stream_adapter as SA  # noqa: E402

ARTIFACT_VERSION = "1.0.0"


@dataclass
class ArtifactMeta:
    version: str = ARTIFACT_VERSION
    data: str = ""
    seed: int = 0
    axes: list[str] = field(default_factory=list)
    as_of_ts: float = 0.0
    layout: dict = field(default_factory=dict)
    measure: dict | None = None
    schema: dict | None = None
    git_sha: str = ""

    def to_dict(self):
        return {
            "version": self.version,
            "data": self.data,
            "seed": self.seed,
            "axes": self.axes,
            "as_of_ts": self.as_of_ts,
            "layout": self.layout,
            "measure": self.measure,
            "schema": self.schema,
            "git_sha": self.git_sha,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            version=d.get("version", ARTIFACT_VERSION),
            data=d.get("data", ""),
            seed=int(d.get("seed", 0)),
            axes=list(d.get("axes") or []),
            as_of_ts=float(d.get("as_of_ts", 0.0)),
            layout=d.get("layout") or {},
            measure=d.get("measure"),
            schema=d.get("schema"),
            git_sha=d.get("git_sha", ""),
        )


def _git_sha():
    try:
        import auto_encoder  # noqa: F401  (ensures repo root on path)

        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class FrozenFoundation:
    """Frozen per-stream foundation model with named state views."""

    def __init__(self, model, vs, meta: ArtifactMeta, measure=None):
        self.model = model
        self.vs = vs
        self.meta = meta
        self.measure = measure

    # ---- construction -------------------------------------------------
    @classmethod
    def from_cache(cls, cache_path, data: str | None = None):
        """Load the train-once cache produced by `scripts/build_frozen.py`."""
        ck = torch.load(str(cache_path) + ".pt", map_location="cpu", weights_only=False)
        model = ck["model"]
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        vs = ck["vs"]
        src = data or ck.get("data", "")
        as_of = float(vs.get("as_of_ts", 0.0))
        if as_of <= 0.0 and src:
            import json as _json

            man = Path(src) / "manifest.json"
            if man.exists():
                from datetime import datetime as _dt

                as_of = _dt.fromisoformat(_json.loads(man.read_text())["as_of"]).timestamp()
        meta = ArtifactMeta(
            data=src,
            seed=int(ck.get("seed", 0)),
            axes=sorted(ck.get("axes") or []),
            as_of_ts=as_of,
            layout=SL.describe(vs, model),
            schema=(
                SA.Schema.from_prepared(
                    vs,
                    vs.get("et_v") or ck.get("et_v") or {},
                    vs.get("catspec") or ck.get("catspec") or [],
                    vs.get("num") or ck.get("num") or [],
                    vs.get("periods") or ck.get("periods") or [],
                    vs.get("as_of_ts", 0.0),
                    vs.get("action_keys") or ck.get("action_keys") or [],
                ).to_dict()
                if (vs.get("catspec") or ck.get("catspec"))
                else None
            ),
            git_sha=_git_sha(),
        )
        measure = MS.measure_from_dict(vs, ck["measure"]) if ck.get("measure") else None
        return cls(model, vs, meta, measure=measure)

    @classmethod
    def fit(
        cls,
        prepared,
        seed: int = 0,
        epochs: int = 12,
        d: int | None = 128,
        layers: int = 2,
    ):
        """Fit one foundation artifact from a prepared stream bundle.

        `prepared` is the tuple returned by `scripts/demo/stream_loader.load_data`, i.e.
        (B, A, keys, seqs, vs, et_v, catspec, num, taus, p_mask, as_of_ts, periods, horizons).
        Training never reads downstream labels.
        """
        import auto_encoder as AE

        (
            _B,
            _A,
            keys,
            seqs,
            vs,
            et_v,
            catspec,
            num,
            _taus,
            _p_mask,
            as_of_ts,
            periods,
            _horizons,
        ) = prepared
        E, Xtab, labels, hold, seg, model, vs2, _B2, best, BLK = AE.train_encoder(
            seed=seed, epochs=epochs, d=d, layers=layers, prepared=prepared
        )
        meta = ArtifactMeta(
            data=str(vs.get("data_path", "")),
            seed=int(seed),
            axes=sorted(AE.AXES),
            as_of_ts=float(as_of_ts),
            layout=SL.describe(vs2, model),
            schema=SA.Schema.from_prepared(vs2, et_v, catspec, num, periods, as_of_ts).to_dict(),
            git_sha=_git_sha(),
        )
        art = cls(model, vs2, meta)
        art._fit_outputs = (E, Xtab, labels, hold, seg, best, BLK)
        return art

    @classmethod
    def fit_stream(
        cls,
        frame,
        as_of_ts: float,
        action_keys=(),
        seed: int = 0,
        epochs: int = 12,
        d: int | None = 128,
        layers: int = 2,
    ):
        """Fit an artifact directly from a raw canonical event stream.

        No wide table and no pre-built bundle are required. Schema discovery and
        sequence construction share one implementation, so serving cannot drift.
        """
        import auto_encoder as AE

        found = SA.discover(frame, as_of_ts, action_keys=action_keys)
        prepared = (
            None,
            None,
            found["keys"],
            found["seqs"],
            found["vs"],
            found["et_v"],
            found["catspec"],
            found["num"],
            found["taus"],
            found["p_mask"],
            found["as_of_ts"],
            found["periods"],
            found["horizons"],
        )
        E, Xtab, labels, hold, seg, model, vs2, _B, best, BLK = AE.train_encoder(
            seed=seed, epochs=epochs, d=d, layers=layers, prepared=prepared
        )
        meta = ArtifactMeta(
            data="<stream>",
            seed=int(seed),
            axes=sorted(AE.AXES),
            as_of_ts=float(as_of_ts),
            layout=SL.describe(vs2, model),
            schema=found["schema"].to_dict(),
            git_sha=_git_sha(),
        )
        art = cls(model, vs2, meta)
        art._fit_outputs = (E, Xtab, labels, hold, seg, best, BLK)
        art._discovery_receipt = found["receipt"]
        return art

    def transform_events(self, frame, as_of=None):
        """Encode a raw canonical event stream with the frozen schema.

        Returns `(keys, state, receipt)`; `state` rows follow `keys` order.
        """
        if not self.meta.schema:
            raise ValueError("artifact has no frozen schema; cannot transform raw events")
        schema = SA.Schema.from_dict(self.meta.schema)
        keys, seqs, receipt = SA.transform(frame, schema, as_of or self.meta.as_of_ts)
        state = self.state(seqs, keys)
        return keys, state, receipt

    def fit_measure(self, seqs, train_keys, coverage: float = 0.99):
        """Resolve the lawful measure vocabulary from TRAINING entities only."""
        self.measure = MS.MeasureBuilder(seqs, train_keys, self.vs, coverage=coverage)
        self.meta.measure = MS.measure_to_dict(self.measure)
        return self.measure

    # ---- serving ------------------------------------------------------
    def state(self, seqs, keys, views=None):
        """Return named state views for `keys`. No future access is assumed:
        the caller passes each entity's already-truncated history."""
        if self.meta.as_of_ts <= 0.0:
            raise ValueError(
                "artifact has no as_of timestamp; refusing to emit a NaN state "
                "(fail safe, never sanitize)"
            )
        import auto_encoder as AE

        prev = AE.AS_OF_TS
        AE.AS_OF_TS = self.meta.as_of_ts
        try:
            E, BLK = AE.embed(self.model, seqs, list(keys), self.vs)
        finally:
            AE.AS_OF_TS = prev
        if not np.isfinite(E).all() or not np.isfinite(BLK).all():
            raise FloatingPointError(
                "frozen encoder produced non-finite state; refusing to sanitize (Robust)"
            )
        out = {"shape": E}
        for name in ("learned_state", "forecast", "uncertainty", "heads"):
            v = SL.slice_view(BLK, self.vs, self.model, name)
            if v.shape[1]:
                out[name] = v
        if views:
            out = {k: v for k, v in out.items() if k in views}
        if (views is None or "measure" in views) and self.measure is not None:
            out["measure"] = self.measure.transform(seqs, keys)
        if views is None:
            out["full"] = SL.compose_full(out["shape"], BLK, self.vs, self.model)
        return out

    def describe(self):
        d = self.meta.to_dict()
        d["views"] = {k: v["dim"] for k, v in SL.describe(self.vs, self.model)["views"].items()}
        d["shape_dim"] = int(self.model.d)
        d["blk_dim"] = int(SL.describe(self.vs, self.model)["blk_dim"])
        d["full_dim"] = int(SL.describe(self.vs, self.model)["full_dim"])
        if self.measure is not None:
            d["measure_dim"] = int(self.measure.dim)
        d["artifact_hash"] = self.hash()
        return d

    def hash(self):
        h = hashlib.sha256()
        h.update(json.dumps(self.meta.to_dict(), sort_keys=True, default=str).encode())
        for p in self.model.state_dict().values():
            h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
        return h.hexdigest()[:12]

    # ---- persistence --------------------------------------------------
    def save(self, path):
        """Save weights + metadata + measure vocabulary as one artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model,
                "vs": self.vs,
                "meta": self.meta.to_dict(),
                "measure": MS.measure_to_dict(self.measure) if self.measure is not None else None,
            },
            str(path) + ".pt",
        )
        return str(path)

    @classmethod
    def load(cls, path):
        ck = torch.load(str(path) + ".pt", map_location="cpu", weights_only=False)
        model = ck["model"]
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        vs = ck["vs"]
        meta = ArtifactMeta.from_dict(ck["meta"])
        measure = MS.measure_from_dict(vs, ck["measure"]) if ck.get("measure") else None
        return cls(model, vs, meta, measure=measure)
