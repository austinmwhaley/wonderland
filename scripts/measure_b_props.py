"""B7.1 calibration law + B8.1 identity law, stamped on the cfm_v2k products (DEC-585).

calibration (B7.1, GPU cuda:0 — refuse + record if unavailable):
    multiclass ECE (10 equal-width max-confidence bins, sum over bins of
    (n_b/N) * |acc_b - conf_b|) of the SHIPPED next-type head:
    `model.head_next(y_last)` softmax, y_last = the per-step readout at the customer's
    last pre-as_of tick (the head was trained on the per-step readout y, not on h —
    verified shape-only against the build objective). Population = split-B customers
    with a next event (first FULL-stream event with ts > the last pre-as_of tick);
    truth = the vocab event-type index, OOV -> the n_et bucket (OOVs recorded).
    OOD retention: next events split at as_of + OOD_HORIZON_DAYS (30) — both in-window
    and OOD ECE/accuracy recorded, no OOD bar (evaluation-contract literal).

identity (B8.1): the served-version registry resolution rule (exactly one registry
    next to the products for the tag the products' state_embeddings actually carry) +
    registry sha256 + data_signature + as_of + n_customers + config sha256 (canonical
    JSON of resolved + overrides) + checkpoint present + filename tag == registry tag.

Writes docs/report_cards/experiments/encoder_b_props.json (re-stamps the legacy file;
its legacy numbers are quarantined in DEC-585).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import b_spec as B  # noqa: E402
from cfm_products import close_at, load_products, resolve_served_tag, stream_sequences  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "encoder_b_props.json"


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def identity_block(products: str) -> dict:
    """B8.1: the registry identity of the served products (one resolution rule, DEC-448)."""
    t0 = time.perf_counter()
    try:
        tag, reg_path = resolve_served_tag(products)
    except SystemExit as exc:
        return {
            "complete": False,
            "present": False,
            "why": f"served-version resolution refused: {exc}",
            "seconds": round(time.perf_counter() - t0, 1),
        }
    meta = json.loads(reg_path.read_text())
    ckpt = reg_path.parent / f"cfm_{tag.replace('.', '_')}.pt"
    present = ckpt.exists()
    config_canon = json.dumps(
        {"resolved": meta.get("resolved"), "overrides": meta.get("overrides")},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "complete": True,
        "present": bool(present),
        "tag": tag,
        "registry_sha256": _sha16(reg_path),
        "data_signature": meta.get("data_signature"),
        "as_of": meta.get("as_of"),
        "n_customers": meta.get("n_customers"),
        "config_sha256": hashlib.sha256(config_canon.encode()).hexdigest()[:16],
        "checkpoint_sha256": _sha16(ckpt) if present else None,
        "why": (
            f"served tag {tag} resolves to exactly one registry next to the products; "
            f"config sha over canonical JSON (resolved + overrides); checkpoint "
            f"{'present' if present else 'MISSING'} (filename tag == registry tag)"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _ece(conf: list[float], correct: list[float], n_bins: int = 10) -> float:
    """Multiclass expected calibration error over equal-width max-confidence bins."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    idx = np.clip(np.digitize(conf, edges), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += float(m.sum()) / conf.size * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def calibration_block(model, cfg, demo: Path, products: str) -> dict:
    """B7.1: the shipped next-type head on split-B customers, in-window vs OOD."""
    import torch

    from looking_glass.cfm_config import TIME_UNIT_SECONDS, _to_epoch

    t0 = time.perf_counter()
    dev = str(next(model.parameters()).device)
    vocab = model.vocab
    n_et = int(vocab.n_et)
    _df, _keys, split, seqs = stream_sequences(cfg, min_events=1)
    seqs_by_key = {s["customer"]: s for s in seqs}
    bkeys = [k for k, v in split.items() if v == "B" and k in seqs_by_key]

    # next events: first FULL-stream event strictly after the last pre-as_of tick
    full = pl.read_parquet(demo / "customer_event_stream.parquet")
    wanted = set(bkeys)
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for c, t, e in zip(
        full["customer_key"].to_list(), full["event_ts"].to_list(), full["event_type"].to_list()
    ):
        if c in wanted:
            grouped[c].append((float(_to_epoch(t)), str(e)))
    next_ev: dict[str, tuple[float, str]] = {}
    for k in bkeys:
        evs = grouped.get(k)
        if not evs:
            continue
        evs.sort(key=lambda p: p[0])
        ts_list = [p[0] for p in evs]
        i = bisect.bisect_right(ts_list, float(seqs_by_key[k]["ts"][-1]))
        if i < len(evs):
            next_ev[k] = evs[i]

    as_of_epoch = close_at(products)
    ood_cut = as_of_epoch + B.OOD_HORIZON_DAYS * TIME_UNIT_SECONDS

    conf_in, corr_in, oov_in = [], [], 0
    conf_ood, corr_ood = [], []
    n_pop = len(next_ev)
    with torch.no_grad():
        for k, (t_next, e_next) in next_ev.items():
            seq = seqs_by_key[k]
            y, _h = model(seq)
            probs = torch.softmax(model.head_next(y[:, -1]), dim=-1)
            conf = float(probs.max())
            pred = int(probs.argmax())
            truth = vocab.et.get(e_next, n_et)
            correct = 1.0 if pred == truth else 0.0
            if t_next <= ood_cut:
                conf_in.append(conf)
                corr_in.append(correct)
                oov_in += int(e_next not in vocab.et)
            else:
                conf_ood.append(conf)
                corr_ood.append(correct)

    ece_in = _ece(conf_in, corr_in)
    acc_in = float(np.mean(corr_in)) if corr_in else float("nan")
    ece_ood = _ece(conf_ood, corr_ood)
    acc_ood = float(np.mean(corr_ood)) if corr_ood else float("nan")
    return {
        "complete": True,
        "ece_next_event": ece_in,
        "acc": acc_in,
        "n_next": len(conf_in),
        "n_pop": n_pop,
        "n_oov": oov_in,
        "ood": {
            "ece": ece_ood,
            "acc": acc_ood,
            "n": len(conf_ood),
            "horizon_days": B.OOD_HORIZON_DAYS,
        },
        "device": dev,
        "as_of_epoch": as_of_epoch,
        "why": (
            f"multiclass ECE (10 equal-width max-confidence bins) of the shipped "
            f"head_next softmax on the per-step readout y_last at the last pre-as_of tick; "
            f"population = {len(bkeys)} split-B customers, {n_pop} with a next full-stream "
            f"event; in-window = next event within as_of+{B.OOD_HORIZON_DAYS}d, OOD beyond "
            f"(recorded, no OOD bar); OOV event types -> the n_et bucket"
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    from safe_run import ensure_boundary

    ensure_boundary()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="/home/austin_whaley/wonderland_work/cfm_v2k")
    ap.add_argument("--demo", default="/home/austin_whaley/wonderland_work/demo_v2")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.perf_counter()
    demo = Path(a.demo)
    id_blk = identity_block(a.products)
    print(f"[b_props] identity: present={id_blk.get('present')} {id_blk.get('why')}", flush=True)

    try:
        tag, _reg = resolve_served_tag(a.products)
        model, cfg, _meta = load_products(tag, a.products, require_gpu=True)
    except (RuntimeError, SystemExit) as exc:
        cal_blk = {
            "complete": False,
            "ece_next_event": None,
            "acc": None,
            "n_next": 0,
            "n_pop": 0,
            "n_oov": 0,
            "ood": {"ece": None, "acc": None, "n": 0, "horizon_days": B.OOD_HORIZON_DAYS},
            "device": "cpu?",
            "why": f"refused: {exc} (the build device is required; CPU vs GPU forward "
            "maxdiff 2.03e-06 exceeds every bar)",
        }
    else:
        cal_blk = calibration_block(model, cfg, demo, a.products)
    print(
        f"[b_props] calibration: ece={cal_blk.get('ece_next_event')} "
        f"acc={cal_blk.get('acc')} n={cal_blk.get('n_next')}/{cal_blk.get('n_pop')}",
        flush=True,
    )

    body = {
        "complete": bool(id_blk.get("complete") and cal_blk.get("complete")),
        "tag": id_blk.get("tag"),
        "products": products_identity(a.products),
        "calibration": cal_blk,
        "identity": id_blk,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stamp_merge_write(Path(a.out), body, " ".join(sys.argv), stamp_blocks=False)
    print("WROTE", a.out)
    return 0 if body["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
