"""Single source of truth for the frozen foundation state layout.

`E` is the learned shape. `BLK` is the legacy block bundle emitted by the encoder.
This module names every block, assigns it to a state view, and exposes slices so
every measurement and card row can state which view it used.

Layer-0 views:
  shape         learned sequence representation (`E`)
  learned_state per-timescale recurrent state + learned mass
  forecast      self-supervised past/future summary heads
  uncertainty   missingness/support + gap heads
  heads         identity/next-event/relational head outputs (predictions)
  measure       lawful typed measure state (built outside the network)

The block emission order here MUST match `Enc.forward` / `embed`.
"""

from __future__ import annotations

import numpy as np

# BLK emission order in Enc.forward -> embed(). One place only.
_ORDER = (
    "learned_state",  # MB
    "identity_probe",  # IB
    "gap",  # RB (head_gap)
    "uncertainty",  # UB
    "forecast_future",  # CB
    "next_event",  # OB
    "relational",  # RelB
    "measure_hist",  # HB (histogram head, NOT lawful measure)
    "suffix",  # SB
    "forecast_past",  # model.past(E)
)

# The permanent definition of the `full` view: concatenation order and membership.
# Changing this is a contract change and must be recorded in DECISIONS.md.
FULL_ORDER = ("shape", "learned_state", "forecast", "uncertainty", "heads")

_VIEW = {
    "learned_state": "learned_state",
    "suffix": "learned_state",
    "identity_probe": "heads",
    "gap": "uncertainty",
    "uncertainty": "uncertainty",
    "forecast_future": "forecast",
    "next_event": "heads",
    "relational": "heads",
    "measure_hist": "heads",
    "forecast_past": "forecast",
}


def block_dims(vs, model):
    """Dimensions of each emitted block, derived from the model/config."""
    e = int(model.ssm[0].e)
    d = int(model.d)
    rel = vs.get("rel_idx") or []
    has_hist = getattr(model, "head_hist", None) is not None
    return {
        "learned_state": 3 * e * d,
        "identity_probe": int(sum(vs.get("id_dims") or [])),
        "gap": 1,
        "uncertainty": int(vs["unc_dim"]),
        "forecast_future": int(len(vs["horizons"]) * vs["fut_dim"]),
        "next_event": int(vs["et"]),
        "relational": int(sum(vs["gcat"][j] for j in rel)),
        "measure_hist": int(vs.get("hist_dim", 0)) if has_hist else 0,
        "suffix": 2 * d if getattr(model, "use_suffix", False) else 0,
        "forecast_past": int(vs["past_dim"]),
    }


def describe(vs, model):
    """Return blocks, view composition, and dimensions for this artifact."""
    dims = block_dims(vs, model)
    blocks = []
    start = 0
    for name in _ORDER:
        dim = int(dims.get(name, 0))
        if dim <= 0:
            continue
        blocks.append(
            {
                "name": name,
                "view": _VIEW[name],
                "start": start,
                "stop": start + dim,
                "dim": dim,
            }
        )
        start += dim
    # Blocks for a view are NOT contiguous in BLK (they are interleaved), so a
    # view is defined by its block names, never by a [start:stop] range.
    views: dict[str, dict] = {}
    for b in blocks:
        v = views.setdefault(b["view"], {"blocks": [], "dim": 0})
        v["blocks"].append(b["name"])
        v["dim"] += b["dim"]
    shape_dim = int(model.d)
    return {
        "blocks": blocks,
        "views": views,
        "shape_dim": shape_dim,
        "blk_dim": start,
        "full_dim": shape_dim + start,
    }


def compose_full(shape, blk, vs, model):
    """Assemble the documented `full` view. `shape` first, then the BLK views in order."""
    parts = [shape]
    for name in FULL_ORDER[1:]:
        v = slice_view(blk, vs, model, name)
        if v.shape[1]:
            parts.append(v)
    return np.hstack(parts)


def slice_view(blk, vs, model, view):
    """Assemble one named view from its blocks (empty if absent)."""
    spec = describe(vs, model)
    v = spec["views"].get(view)
    if not v or not v["blocks"]:
        return blk[:, :0]
    by_name = {b["name"]: b for b in spec["blocks"]}
    return np.hstack([blk[:, by_name[n]["start"] : by_name[n]["stop"]] for n in v["blocks"]])
