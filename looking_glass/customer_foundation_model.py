# ruff: noqa: F401
"""Looking Glass — the Customer Foundation Model (CFM).

A frozen, versioned, causal state function:

        state_c(t) = CFM_version( events of customer c with timestamp <= t )

Multi-objective self-supervised encoder (no labels), frozen after training.
Customers are partitioned into disjoint samples: A (encoder training) and B
(plugin training); A and B never overlap.

Products (DuckDB):
  * customer_state          -- the STATE for every customer, with as_of.
                                                           Holds the recurrence state H, the public
                                                           embedding S, and as_of so we know whether to
                                                           fade() (time passed, no events) or absorb()
                                                           (new events arrived).
  * anchor_embeddings -- point-in-time EMBEDDINGS S_c(anchor) for
                                                           sample-B customers at past anchor dates, used
                                                           as inputs to plugin training.

State ops:
  fade(H, dt)            -- decay the state when no events occurred for dt.
  absorb(model, H, evs)  -- push new events through the recurrence (O(#events)).
  StateStore.advance()   -- load state, fade to first new event, absorb, upsert.

Objectives (self-supervised): next-event type, next-entity, time-to-next-event,
contrastive, masked-event reconstruction, redundancy (VICReg). Plug in more.

CLI:
  python -m looking_glass.customer_foundation_model all --db <duckdb> --customers 500
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from looking_glass.cfm_config import (
    AT,
    CFMConfig,
    EMBED_DIM,
    GAMMA_MAX,
    LN2,
    SF_PHI,
    STREAM_TABLE,
    TIME_UNIT_SECONDS,
    _expert_biases,
    _f,
    _h,
    _seed_everything,
    _to_epoch,
    monthly_split_seed,
    sample_a,
)
from looking_glass.cfm_data import (
    _apply_data_revision,
    _covariates,
    _customer_keys,
    _random_anchor_epochs,
    _read_stream,
    assign_split,
    build_sequences,
)
from looking_glass.cfm_model import CFM, EventVocab, MultiScaleSSM, SelectiveSSM, _scan
from looking_glass.cfm_state import StateStore, absorb, build_products, fade
from looking_glass.cfm_training import (
    _collate,
    _combine,
    _jepa_loss,
    _loss,
    _mask_loss,
    _mask_loss_batch,
    _registry,
    _task_losses,
    _val_loss,
    train_cfm,
)
from looking_glass.cfm_validation import _causal, _next_event_acc, _objective_metrics, validate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print(rows, title):
    print(f"== {title} ==")
    print(f"{'check':46s} {'target':>12s} {'achieved':>16s}  status")
    for x in rows:
        print(
            f"{x['check']:46s} {str(x['target']):>12s} {str(x['achieved']):>16s}  "
            f"{'PASS' if x['ok'] else 'FAIL'}"
        )
    n = sum(1 for x in rows if x["ok"])
    print(f"completion: {n}/{len(rows)} ({100 * n / len(rows):.0f}%)")


def _rebuild_products(
    cfg: CFMConfig, tag: str | None = None, anchors: int | None = None, sample_b: int | None = None
):
    """Rebuild the products tables from an EXISTING checkpoint (no retraining).

    Makes the cheap tuning knobs actually cheap: anchors per sample-B customer
    and the sample-B size do NOT affect encoder training — only the products —
    so they can be swept in minutes instead of a ~40-min retrain per try.
    Populations/cutoff/samples are recovered from the run's registry, so the
    rebuild reproduces the same A/B; only the overrides change.
    NOTE: rebuilding resets states -> re-run the day-1 daily state job after.
    """
    import glob as _glob
    import json as _json

    from looking_glass.cfm_state import build_products, load_frozen_encoder

    out = Path(cfg.out_dir)
    regs = _glob.glob(str(out / "registry_*.json"))
    if not regs:
        raise SystemExit(f"no registry found in {out}; train first (or fix --out-dir)")
    if tag is None:
        reg_path = max(regs, key=lambda q: Path(q).stat().st_mtime)
        meta = _json.loads(Path(reg_path).read_text())
        tag = meta["tag"]
    else:
        reg_path = out / f"registry_{tag.replace('.', '_')}.json"
        if not reg_path.exists():
            raise SystemExit(f"registry for tag {tag} not found: {reg_path}")
        meta = _json.loads(reg_path.read_text())
    run_cfg = meta.get("config", {})
    model, rcfg = load_frozen_encoder(tag, out)
    rcfg.out_dir = str(out)  # write products next to the checkpoint
    rcfg.as_of = run_cfg.get("as_of")
    rcfg.split_seed = run_cfg.get("split_seed", rcfg.split_seed)
    rcfg.sample_customers = run_cfg.get("sample_customers")
    rcfg.sample_a_customers = run_cfg.get("sample_a_customers")
    rcfg.sample_b_customers = (
        sample_b if sample_b is not None else run_cfg.get("sample_b_customers")
    )
    rcfg.n_anchors = anchors if anchors is not None else run_cfg.get("n_anchors", rcfg.n_anchors)
    if rcfg.as_of is None:
        print(f"WARNING: registry for {tag} has no as_of; rebuilding on the FULL stream")
    df = _read_stream(rcfg)
    keys = _customer_keys(df, rcfg)
    split = assign_split(keys, rcfg)
    s, t = build_products(rcfg, model, None, df, keys, split)
    print(
        f"products rebuilt from {tag}: customer_state={s} training_embeddings={t} "
        f"(anchors/customer={rcfg.n_anchors}, sample_b={rcfg.sample_b_customers})",
        flush=True,
    )
    print(
        "NOTE: rebuilding resets states -> re-run the day-1 daily state job "
        "(python -m looking_glass.daily_states) after tuning.",
        flush=True,
    )
    return tag


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Looking Glass — Customer Foundation Model")
    ap.add_argument("cmd", choices=["train", "validate", "all", "products"])
    ap.add_argument(
        "--tag",
        default=None,
        help="products: encoder tag to rebuild from (default: most recent registry)",
    )
    ap.add_argument("--db", default=CFMConfig.db)
    ap.add_argument("--customers", type=int, default=CFMConfig.sample_customers)
    ap.add_argument(
        "--anchors",
        type=int,
        default=None,
        help="anchor dates per sample-B customer (default: 6 for train; registry value for products)",
    )
    ap.add_argument("--epochs", type=int, default=CFMConfig.epochs)
    ap.add_argument("--device", default=CFMConfig.device)
    ap.add_argument(
        "--as-of",
        default=None,
        help="ISO date/datetime: train on events <= this date (point-in-time). "
        "Also rotates the A/B split monthly (monthly re-randomization).",
    )
    ap.add_argument(
        "--sample-a",
        type=int,
        default=None,
        help="encoder training sample size drawn from population A (default: all of A)",
    )
    ap.add_argument(
        "--sample-b",
        type=int,
        default=None,
        help="plugin-training sample size drawn from population B (default: all of B)",
    )
    ap.add_argument(
        "--out-dir",
        default=CFMConfig.out_dir,
        help="artifact directory (default: artifacts/cfm relative to cwd; pass an "
        "explicit path to avoid cwd ambiguity, e.g. looking_glass/artifacts/cfm)",
    )
    a = ap.parse_args(argv)
    anchors = a.anchors if a.anchors is not None else CFMConfig.n_anchors
    cfg = sample_a(a.customers, anchors, db=a.db, epochs=a.epochs, device=a.device)
    cfg.out_dir = a.out_dir
    cfg.sample_a_customers = a.sample_a
    cfg.sample_b_customers = a.sample_b
    cfg.as_of = a.as_of
    if a.as_of:
        cfg.split_seed = monthly_split_seed(cfg.split_seed, a.as_of)
        print(f"point-in-time as_of={a.as_of}  split_seed={cfg.split_seed}")
    if a.cmd == "products":
        _rebuild_products(cfg, tag=a.tag, anchors=a.anchors, sample_b=a.sample_b)
        return 0
    if a.cmd in ("train", "all"):
        model, vocab, df, keys, split = train_cfm(cfg)
        s, t = build_products(cfg, model, vocab, df, keys, split)
        print(f"trained {cfg.tag}: customer_state={s} training_embeddings={t}")
    if a.cmd in ("validate", "all"):
        rows, verdict = validate(cfg)
        _print(rows, f"CFM VALIDATION {cfg.tag}")
        print("VERDICT:", "PASS" if verdict else "FAIL")
        return 0 if verdict else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
