"""Complexity-ladder gate (SPEC §20.5): score a frozen encoder per rung and emit
the capability-vs-complexity curve as a receipt.

Each rung is a stream from `rabbit_hole.generators.ladder`. Point `--ckpt-root` at
a directory whose subdirectories are named by rung and contain a trained CFM
checkpoint (cfm_*.pt + registry_*.json); this scores each with the §11.3 light
probe gate (strong baseline + bootstrap CI) plus the §20.4 capability card, and
prints the P/F/U/N grid.

Run:
  python3 -m looking_glass.ladder_curve --ladder-dir rabbit_hole/data/ladder \
      --ckpt-root /tmp/opencode --ckpt-prefix ladder_ --out /tmp/opencode/curve.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from looking_glass.cfm_data import (
    _customer_keys,
    _read_stream,
    assign_split,
    build_sequences,
    draw_sample,
)
from looking_glass.cfm_state import load_frozen_encoder
from looking_glass.probes import run_probe_report

RUNGS = ("L0_null", "L1_rate", "L2_markov", "L3_intent", "L4_trend", "L5_entity", "L6_action")


def _tag(ckpt_dir):
    reg = sorted(glob.glob(str(Path(ckpt_dir) / "registry_*.json")))
    return json.loads(Path(reg[-1]).read_text())["tag"] if reg else None


def score_rung(rung, ckpt_dir, ladder_dir, as_of, sample):
    tag = _tag(ckpt_dir)
    if tag is None:
        return None
    model, cfg = load_frozen_encoder(tag, ckpt_dir)
    cfg.db = str(Path(ladder_dir) / rung / "customer_event_stream.parquet")
    cfg.as_of, cfg.sample_customers = as_of, sample
    if rung == "L6_action":
        cfg.exogenous_events = ("action_send",)
    df = _read_stream(cfg)
    keys = _customer_keys(df, cfg)
    split = assign_split(keys, cfg)
    ak = draw_sample(keys, split, "A", sample, cfg.split_seed)
    seqs = build_sequences(df, ak, cfg, split, with_anchors=False)
    _md, rep = run_probe_report(model, model.vocab, cfg, seqs, tag, light=True)
    return {k: v.get("verdict_ci") for k, v in rep["objectives"].items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Complexity-ladder capability curve")
    ap.add_argument("--ladder-dir", default="rabbit_hole/data/ladder", dest="ladder_dir")
    ap.add_argument("--ckpt-root", default="/tmp/opencode", dest="ckpt_root")
    ap.add_argument("--ckpt-prefix", default="ladder_", dest="ckpt_prefix")
    ap.add_argument("--as-of", default="2024-12-31", dest="as_of")
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = {}
    for rung in RUNGS:
        ck = Path(a.ckpt_root) / f"{a.ckpt_prefix}{rung}"
        if not ck.exists():
            continue
        rows[rung] = score_rung(rung, ck, a.ladder_dir, a.as_of, a.sample)
        print(f"[{rung}] {rows[rung]}", flush=True)
    objs = sorted({o for r in rows.values() if r for o in r})
    print("\nrung       " + " ".join(f"{o[:4]:>4}" for o in objs))
    for rung, r in rows.items():
        print(f"{rung:10s}" + " ".join(f"{(r.get(o) or '-')[0]:>4}" for o in objs))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
