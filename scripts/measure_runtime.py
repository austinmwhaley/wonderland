"""B5.2 runtime law: the build_products state-loop cost on the cfm_v2k products (DEC-585).

N=5000 forward benchmark doing EXACTLY the state-loop work per customer
(`model(seq)` -> h + the per-customer counter readout the build published), then the
batched E replica + the exact day-1-close fade replica (the close work). Cold/warm wall
split (cold = first 500 customers: loader + first forwards), rows/s on the warm part.
Peak RSS vs the project slice cap (recorded; cap None renders as a CI no-op on the card).
The incremental path is the RECORDED state_job_receipts row (the close ran incrementally
on the live store at build time), not a re-measurement.

CPU producer: the state loop is device-invariant (the build loop itself is per-customer
`model(seq)`; this measures that work on the machine the card certifies, and the build
receipt records its own device). Re-stamps docs/report_cards/experiments/runtime.json;
the legacy numbers are quarantined in DEC-585.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cfm_products import (
    close_at,
    load_products,
    resolve_served_tag,
    stored_tables,
    stream_sequences,
)  # noqa: E402
from measure_input_law import products_identity, stamp_merge_write  # noqa: E402

OUT = ROOT / "docs" / "report_cards" / "experiments" / "runtime.json"
COLD = 500  # first customers carry the loader/warm-up; rows/s is measured on the rest


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main() -> int:
    from safe_run import ensure_boundary, slice_cap_gb

    ensure_boundary()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--products", default="/home/austin_whaley/wonderland_work/cfm_v2k")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    import torch

    from looking_glass.cfm_config import LN2, TIME_UNIT_SECONDS
    from looking_glass.cfm_state import donor_states
    from looking_glass.cfm_training import _collate

    t0 = time.perf_counter()
    rss_before = _rss_mb()
    tag, _reg = resolve_served_tag(a.products)
    model, cfg, meta = load_products(tag, a.products, cpu=True)
    _df, _keys, _split, seqs = stream_sequences(cfg, min_events=1)
    n = len(seqs)
    n_expected = int(meta.get("n_customers") or 0)
    stored = stored_tables(a.products)
    ctr_by_key = {
        k: np.asarray(v[1], dtype=np.float32) for k, v in stored["customer_state"].items()
    }
    ca = close_at(a.products)
    hl = float(cfg.state_half_life_days)

    H, S = [], []
    cold_s = 0.0
    warm_s = 0.0
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            t1 = time.perf_counter()
            _y, h = model(seq)
            hp = h.cpu().numpy().astype(np.float32)
            # the build loop's per-customer counter readout (published into customer_state)
            tcol = _collate([seq], model.vocab, model._dev())
            tok = model.tokens_batch(tcol)
            _cre, M = model.counter(tok, model.raw_fields(tcol), mask=tcol.get("mask"))
            elapsed = time.perf_counter() - t1
            if i < COLD:
                cold_s += elapsed
            else:
                warm_s += elapsed
            H.append(hp)
            # the exact day-1 close fade, op-for-op with fade_idle (CPU float32)
            last_ts = float(seq["ts"][-1])
            dt = torch.tensor([float(ca) - last_ts], dtype=torch.float32).unsqueeze(1)
            decay = torch.exp(-(LN2 * dt) / max(hl * TIME_UNIT_SECONDS, 1.0)).clamp(max=1.0)
            S.append((torch.as_tensor(hp, dtype=torch.float32) * decay).numpy()[0])

    H_all = np.vstack(H)
    close_t1 = time.perf_counter()
    ctr_all = np.vstack([ctr_by_key[s["customer"]] for s in seqs])
    E_all = donor_states(model, H_all, ctr_all)
    close_s = time.perf_counter() - close_t1

    cap = slice_cap_gb()
    body = {
        "complete": True,
        "n": int(n),
        "n_expected": n_expected,
        "cold_seconds": round(cold_s, 2),
        "warm_seconds": round(warm_s, 2),
        "rows_per_s": round((n - COLD) / warm_s, 1) if warm_s > 0 else None,
        "close_seconds": round(close_s, 2),
        "dims_state": int(H_all.shape[1]),
        "dims_served": int(E_all.shape[1]),
        "peak_rss_mb": _rss_mb(),
        "rss_mb_at_load": rss_before,
        "slice_cap_gb": cap,
        "incremental": {
            "as_of_epoch": ca,
            "n_absorbed": int(stored["close_receipts"][0][1]),
            "n_idle": int(stored["close_receipts"][0][2]),
            "n_embeddings": int(stored["close_receipts"][0][3]),
            "device": stored["close_receipts"][0][4],
            "wall_seconds": float(stored["close_receipts"][0][5]),
        },
        "products": products_identity(a.products),
        "device": "cpu",
        "tag": tag,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    stamp_merge_write(Path(a.out), body, " ".join(sys.argv), stamp_blocks=False)
    print(json.dumps(body, indent=1, default=str))
    print("WROTE", a.out)
    return 0 if body["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
