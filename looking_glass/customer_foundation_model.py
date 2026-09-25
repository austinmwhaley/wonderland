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


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Looking Glass — Customer Foundation Model")
    ap.add_argument("cmd", choices=["train", "validate", "all"])
    ap.add_argument("--db", default=CFMConfig.db)
    ap.add_argument("--customers", type=int, default=CFMConfig.sample_customers)
    ap.add_argument("--anchors", type=int, default=CFMConfig.n_anchors)
    ap.add_argument("--epochs", type=int, default=CFMConfig.epochs)
    ap.add_argument("--device", default=CFMConfig.device)
    a = ap.parse_args(argv)
    cfg = sample_a(a.customers, a.anchors, db=a.db, epochs=a.epochs, device=a.device)
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
