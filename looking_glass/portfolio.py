"""Layer-B portfolio grade: is the encoder good at ALL its self-supervised
objectives, with geometry? (DEC-008 / DEC-009)

Measurement design:
  * **Held-out only**: the same 85/15 row split training uses (`_val_split`).
  * **Structure skill**: every objective's loss is measured twice per fold —
    on the real validation sequences and on **destroyed** sequences (events
    permuted within each customer; marginals preserved, structure removed) —
    under identical seeded stochastic draws. skill = loss_destroyed -
    loss_real; positive means the model uses temporal structure, and a purely
    marginal predictor scores 0 by construction (one yardstick for every
    objective — no hand-built baselines).
  * **SE**: grouped folds over held-out customers; skill ± std/sqrt(folds).
  * **Geometry** (report + one gate): effective rank (participation ratio),
    mean pairwise cosine, and |off-diagonal correlation| — each against a
    column-permuted null of the same states; the gate is the collapse guard
    (effective rank >= 0.3 x null, documented conservative floor).

Grade rows follow the gate-row contract; the receipt is written to
`portfolio/<tag>_<stamp>.json` so every training run is graded, not just
claimed (doctrine #9/#11).

    python3 -m looking_glass.customer_foundation_model portfolio --tag <t>
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path(__file__).resolve().parent / "artifacts" / "cfm" / "portfolio"

# contrast = instance discrimination: its capability is agreement, and a
# destroyed-input null is the wrong yardstick for it (states may still
# separate instances without order). redundancy is graded in geometry.
GATED_EXCLUSIONS = {"contrast", "redundancy"}


def _destroyed(seqs: list[dict], seed: int) -> list[dict]:
    """Destroy temporal structure per customer: permute the event content
    jointly (type/brand/entity/value/timestamps move together). Customer
    identity and sequence length are preserved; marginals are preserved."""
    rng = np.random.default_rng(seed)
    fields = (
        "event_type",
        "brand",
        "entity_type",
        "entity_id",
        "value",
        "event_ts",
        "ts",
        "co",
    )
    out = []
    for s in seqs:
        L = len(s["event_type"])
        perm = rng.permutation(L)
        d = dict(s)
        for f in fields:
            if f in s and s[f] is not None and len(s[f]) == L:
                d[f] = [s[f][int(i)] for i in perm]
        out.append(d)
    return out


def _geometry(states: np.ndarray, seed: int) -> dict:
    """Representation geometry vs a column-permuted null of the SAME states."""
    n, dim = states.shape
    z = states - states.mean(0, keepdims=True)
    rng = np.random.default_rng(seed)
    null = states[:, rng.permutation(dim)]
    zn = null - null.mean(0, keepdims=True)

    def eff_rank(cov):
        ev = np.linalg.eigvalsh(cov)
        ev = np.clip(ev, 0.0, None)
        s1, s2 = ev.sum(), (ev**2).sum()
        return float(s1**2 / s2) if s2 > 0 else 0.0

    cov = (z.T @ z) / max(n - 1, 1)
    cov_n = (zn.T @ zn) / max(n - 1, 1)
    # |off-diagonal correlation| (redundancy) + its permuted-dim null
    sd = np.sqrt(np.diag(cov)).clip(min=1e-12)
    corr = cov / np.outer(sd, sd)
    off = corr - np.eye(dim)
    sd_n = np.sqrt(np.diag(cov_n)).clip(min=1e-12)
    corr_n = cov_n / np.outer(sd_n, sd_n)
    off_n = corr_n - np.eye(dim)
    # mean pairwise cosine of L2-normalized states (anisotropy signal)
    nz = states / np.linalg.norm(states, axis=1, keepdims=True).clip(min=1e-12)
    pairs = min(2000, n * (n - 1) // 2)
    if pairs > 0:
        i = rng.integers(0, n, pairs)
        j = rng.integers(0, n, pairs)
        keep = i != j
        mean_cos = float((nz[i[keep]] * nz[j[keep]]).sum(-1).mean()) if keep.any() else float("nan")
    else:
        mean_cos = float("nan")
    return {
        "n_states": int(n),
        "dim": int(dim),
        "eff_rank": round(eff_rank(cov), 3),
        "eff_rank_null": round(eff_rank(cov_n), 3),
        "eff_rank_ratio": round(eff_rank(cov) / max(eff_rank(cov_n), 1e-9), 3),
        "mean_pairwise_cos": round(mean_cos, 4),
        "redundancy_offdiag": round(float(np.abs(off).mean()), 5),
        "redundancy_offdiag_null": round(float(np.abs(off_n).mean()), 5),
    }


def evaluate(
    model,
    vocab,
    cfg,
    seqs: list[dict],
    seed: int = 0,
    folds: int = 5,
    out_dir=None,
    tag: str = "unknown",
) -> dict:
    """Grade the encoder on its whole self-supervised portfolio (held out)."""
    import torch

    from looking_glass.cfm_training import (
        _task_losses,
        _val_split,
        forward_states,
    )

    t0 = time.perf_counter()
    _tr_idx, val_idx = _val_split(len(seqs), seed)
    val = [seqs[int(i)] for i in val_idx]
    if len(val) < 4:
        raise SystemExit(f"portfolio needs >=4 held-out rows, got {len(val)}")
    customers = sorted({s["customer"] for s in val})
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(customers)
    fold_ids = np.array_split(np.array(customers, dtype=object), min(folds, len(customers)))

    gated = [o for o in cfg.objectives if o not in GATED_EXCLUSIONS]
    real_folds: dict[str, list[float]] = {o: [] for o in gated}
    shuf_folds: dict[str, list[float]] = {o: [] for o in gated}
    seen: set[str] = set()

    was_training = model.training
    model.eval()
    try:
        for gi, g in enumerate(fold_ids):
            gset = set(g.tolist())
            batch = [s for s in val if s["customer"] in gset]
            if len(batch) < 2:
                continue
            torch.manual_seed(seed + gi)  # identical stochastic draws (real vs destroyed)
            with torch.no_grad():
                real = _task_losses(model, vocab, batch, cfg)
            shuf_seqs = _destroyed(batch, seed * 1000 + gi)
            torch.manual_seed(seed + gi)
            with torch.no_grad():
                shuf = _task_losses(model, vocab, shuf_seqs, cfg)
            seen.update(k for k in real if k != "redundancy")
            for o in gated:
                if o in real and o in shuf:
                    real_folds[o].append(float(real[o]))
                    shuf_folds[o].append(float(shuf[o]))

        # geometry on held-out final states (chunked)
        states = []
        for i in range(0, len(val), 256):
            with torch.no_grad():
                h = forward_states(model, val[i : i + 256])
            states.append(h.detach().cpu().numpy())
        z = np.concatenate(states, axis=0) if states else np.zeros((0, 1))
    finally:
        model.train(was_training)

    objectives = []
    rows = []
    for o in gated:
        if o not in real_folds or not real_folds[o]:
            continue
        r = np.asarray(real_folds[o])
        s = np.asarray(shuf_folds[o])
        skill = s - r  # positive = real beats destroyed = uses structure
        se = float(np.std(skill) / np.sqrt(max(len(skill), 1)))
        objectives.append(
            {
                "objective": o,
                "real": round(float(r.mean()), 5),
                "destroyed": round(float(s.mean()), 5),
                "skill": round(float(skill.mean()), 5),
                "skill_se": round(se, 5),
                "folds": int(len(skill)),
            }
        )
        rows.append(
            {
                "check": f"portfolio: {o} structure-skill > 0",
                "achieved": f"{skill.mean():+.4f} ± {se:.4f} ({len(skill)} folds)",
                "ok": bool(skill.mean() > 0),
            }
        )
    geometry = _geometry(z, seed + 2) if len(z) >= 4 else {}
    if geometry:
        rows.append(
            {
                # collapse guard: effective rank vs column-permuted null of the
                # same states (0.3 = documented conservative floor)
                "check": "geometry: effective rank >= 0.3 x null",
                "achieved": f"{geometry['eff_rank']} / {geometry['eff_rank_null']}"
                f" (ratio {geometry['eff_rank_ratio']})",
                "ok": bool(geometry["eff_rank_ratio"] >= 0.3),
            }
        )
    missing = sorted(set(cfg.objectives) - seen - GATED_EXCLUSIONS)
    ok = all(r["ok"] for r in rows) if rows else False
    receipt = {
        "tag": tag,
        "seed": seed,
        "folds": folds,
        "n_val": len(val),
        "n_seqs": len(seqs),
        "sf_mode": getattr(cfg, "sf_mode", "purchase"),
        "objectives": objectives,
        "objectives_missing": missing,
        "geometry": geometry,
        "rows": rows,
        "ok": bool(ok),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(out_dir) if out_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    stamp = receipt["ran_at"].replace(":", "").replace("-", "").split(".")[0]
    path = out / f"portfolio_{tag.replace('.', '_')}_{stamp}Z.json"
    path.write_text(json.dumps(receipt, indent=1, default=float))

    print("== LAYER-B PORTFOLIO GRADE (held-out, destroyed-data null) ==")
    for o, row in zip(objectives, rows):
        status = "PASS" if row["ok"] else "FAIL"
        print(f"  {o['objective']:12s} skill {o['skill']:+.4f} ± {o['skill_se']:.4f}  {status}")
    if geometry:
        print(
            f"  geometry: eff_rank {geometry['eff_rank']}/{geometry['eff_rank_null']}"
            f"  mean_cos {geometry['mean_pairwise_cos']}"
            f"  redundancy {geometry['redundancy_offdiag']}"
            f" (null {geometry['redundancy_offdiag_null']})"
        )
    if missing:
        print(f"  WARNING: objectives not evaluated (missing inputs): {missing}")
    print(f"PORTFOLIO: {'PASS' if ok else 'FAIL'}   receipt -> {path}")
    receipt["receipt_path"] = str(path)
    return receipt
