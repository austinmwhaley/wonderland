"""Augment a frozen checkpoint with a DETERMINISTIC additive counter (mass axis).

The model-capability frontier needs the SHAPE block (not only the explicit
measure) to carry cumulative count/value. The trained counter was unstable; this
adds a FIXED additive counter to an existing frozen model instead:

  * gate = 1 (W_g.weight=0, bias=20), one register,
  * M_t = M_{t-1} + raw_t  (exact cumulative raw fields),
  * w_counter = 0 (the counter is appended to E, not injected into the trained
    state, so the frozen base is unchanged),
  * E shape = donor ‖ signed_log1p(M).

The base donor/measure (and therefore the decision/known-effect/OPE) are
unchanged; only the served shape gains the mass readout. No labels; no training.

Usage:
  python3 scripts/augment_counter.py --src-dir <dir> --src-tag <t> \
      --dst-dir <dir> --dst-tag <new> --version 0.8.2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, dest="src_dir")
    ap.add_argument("--src-tag", required=True, dest="src_tag")
    ap.add_argument("--dst-dir", required=True, dest="dst_dir")
    ap.add_argument("--dst-tag", required=True, dest="dst_tag")
    ap.add_argument("--version", default="0.8.2")
    ap.add_argument(
        "--type-values",
        action="store_true",
        dest="type_values",
        help="add per-event-type cumulative VALUE (one-hot(et) * value) to the compact counter",
    )
    ap.add_argument(
        "--event-types",
        action="store_true",
        dest="event_types",
        help="add the stream's own EVENT-TYPE counts (n_et+1) to the compact counter",
    )
    a = ap.parse_args(argv)

    from looking_glass.cfm_model import CounterBlock
    from looking_glass.cfm_state import load_frozen_encoder

    src = Path(a.src_dir)
    dst = Path(a.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    model, cfg = load_frozen_encoder(a.src_tag, src)
    device = next(model.parameters()).device
    # COMPACT MASS AXIS: count + numeric fields (x, x^2) + value/gap; no one-hots
    # (the full counter was 2566 dims and forced a 917-rank shape / OOM).
    raw_dim = 1 + 4 + 2 * len(model.numeric_fields)
    if a.event_types:
        raw_dim += int(model.vocab.n_et) + 1
    if a.type_values:
        raw_dim += int(model.vocab.n_et) + 1
    model.counter_registers = 1
    model.counter_compact = True
    model.counter_event_types = bool(a.event_types)
    model.counter_type_values = bool(a.type_values)
    model.counter_raw_dim = int(raw_dim)
    model.counter = CounterBlock(model.chan, raw_dim, n_registers=1).to(device)
    model.w_counter = torch.nn.Linear(raw_dim, model.dim).to(device)
    with torch.no_grad():
        model.counter.W_g.weight.zero_()
        model.counter.W_g.bias.fill_(20.0)  # sigma(20) ~ 1: exact cumulative sum
        model.w_counter.weight.zero_()
        model.w_counter.bias.zero_()  # do NOT alter the trained state
    model.counter_on = True
    model.eval()

    # save the augmented checkpoint (base weights + counter modules)
    _s = a.src_tag.replace(".", "_")
    _d = a.dst_tag.replace(".", "_")
    blob_path = src / f"cfm_{_s}.pt"
    blob = torch.load(blob_path, map_location="cpu", weights_only=False)
    blob["state"] = model.state_dict()
    blob["counter"] = True
    blob["counter_registers"] = 1
    blob["counter_compact"] = True
    blob["counter_event_types"] = bool(a.event_types)
    blob["counter_type_values"] = bool(a.type_values)
    torch.save(blob, dst / f"cfm_{_d}.pt")

    # Layer-B measure (unchanged) + registry (new tag/version)
    shutil.copy2(src / f"layer_b_{_s}.pkl", dst / f"layer_b_{_d}.pkl")
    if (src / f"layer_b_{_s}.pkl.json").exists():
        lb = json.loads((src / f"layer_b_{_s}.pkl.json").read_text())
        lb["tag"] = a.dst_tag
        (dst / f"layer_b_{_d}.pkl.json").write_text(json.dumps(lb, indent=2) + "\n")
    reg = json.loads((src / f"registry_{_s}.json").read_text())
    reg["tag"] = a.dst_tag
    reg["version"] = a.version
    reg["counter"] = True
    reg["counter_compact"] = True
    reg["counter_event_types"] = bool(a.event_types)
    reg["counter_type_values"] = bool(a.type_values)
    reg["counter_raw_dim"] = int(raw_dim)
    (dst / f"registry_{_d}.json").write_text(json.dumps(reg, indent=1) + "\n")
    print(
        json.dumps(
            {
                "src": str(src),
                "dst": str(dst),
                "src_tag": a.src_tag,
                "dst_tag": a.dst_tag,
                "counter_raw_dim": int(raw_dim),
                "counter_event_types": bool(a.event_types),
                "counter_type_values": bool(a.type_values),
                "shape_source_dim": int(model.dim + raw_dim),
                "w_counter_zero": True,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    main()
