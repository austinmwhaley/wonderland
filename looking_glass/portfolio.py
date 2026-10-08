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
GATED_EXCLUSIONS = {"contrast", "redundancy", "variance", "spectrum", "volume", "iso", "trajectory"}
# sf's destroyed-null is scale-broken (permutation smooths discounted-sum
# targets) — FIXED via the target-variance R² yardstick (DEC-018): skill =
# R²_real - R²_destroyed = L_shuf/V_shuf - L_real/V_real, dimensionless.
# sf is GATED again. The mechanism stays for future broken yardsticks.
YARDSTICK_PENDING: set = set()


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


def _mp_pr_frac(n, dim, seed):
    """Participation-ratio fraction of a finite-sample isotropic-gaussian null
    (Marchenko-Pastur): the effective rank a PURE-NOISE n x dim matrix shows by
    chance. The scale-free geometry score is measured against this, so the bar
    is valid at any dimension / sample size (unlike a raw 0.30 fraction)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, dim))
    a = a - a.mean(0, keepdims=True)
    cov = (a.T @ a) / max(n - 1, 1)
    ev = np.linalg.eigvalsh(cov).clip(0.0, None)
    s2 = (ev**2).sum()
    return float(ev.sum() ** 2 / s2) / dim if s2 > 0 else 0.0


def _geometry(states: np.ndarray, seed: int) -> dict:
    """Representation geometry vs a column-permuted null of the SAME states."""
    n, dim = states.shape
    z = states - states.mean(0, keepdims=True)
    rng = np.random.default_rng(seed)
    # null: independently shuffle rows WITHIN each column — destroys joint
    # structure while keeping every marginal. (Whole-matrix permutation would
    # preserve the covariance spectrum exactly and make every null vacuous.)
    null = np.column_stack([states[:, c][rng.permutation(n)] for c in range(dim)])
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
    # Scale-free geometry (v4.0, DEC-029): MP-normalized participation ratio.
    # pr_frac = observed PR/dim; mp_frac = the pure-noise (Marchenko-Pastur)
    # PR/dim at this n, dim. score 1.0 = perfectly isotropic, 0.0 = pure noise,
    # <0.0 = LESS isotropic than random noise. Independent of dimension/sample.
    pr_frac = eff_rank(cov) / dim
    mp_frac = _mp_pr_frac(n, dim, seed + 7)
    pr_score = float((pr_frac - mp_frac) / max(1.0 - mp_frac, 1e-9))
    return {
        "n_states": int(n),
        "dim": int(dim),
        "eff_rank": round(eff_rank(cov), 3),
        "eff_rank_null": round(eff_rank(cov_n), 3),
        "eff_rank_ratio": round(eff_rank(cov) / max(eff_rank(cov_n), 1e-9), 3),
        "pr_frac": round(pr_frac, 4),
        "mp_null_frac": round(mp_frac, 4),
        "pr_score": round(pr_score, 4),
        "mean_pairwise_cos": round(mean_cos, 4),
        "redundancy_offdiag": round(float(np.abs(off).mean()), 5),
        "redundancy_offdiag_null": round(float(np.abs(off_n).mean()), 5),
    }


def _canaries(states, seqs, seed: int) -> dict:
    """Leak detector: a frozen state should carry BEHAVIOR, not assignment
    metadata. Logistic probes on held-out states; PASS = the state cannot
    predict the A/B arm or the period bucket."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.array([str(s.get("group", "A")) for s in seqs])
    months = np.array([int(str(s["event_ts"][-1])[:7].replace("-", "")) for s in seqs])
    # split by CUSTOMER (never by the label — grouping by the label would put
    # a whole class on one side and the probe would be meaningless)
    cust = np.array([str(s.get("customer", f"r{i}")) for i, s in enumerate(seqs)])
    out = {}
    gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    tr, te = next(gss.split(states, groups=cust))
    # A/B membership: AUC must be near-chance
    uniq_g = np.unique(groups)
    if len(uniq_g) == 2:
        clf = LogisticRegression(max_iter=1000).fit(states[tr], groups[tr])
        auc = float(roc_auc_score(groups[te], clf.predict_proba(states[te])[:, 1]))
        out["group_auc"] = round(auc, 4)
    # period bucket: accuracy must be near-chance (multi-class)
    if len(np.unique(months)) > 1:
        clf = LogisticRegression(max_iter=1000).fit(states[tr], months[tr])
        acc = float(accuracy_score(months[te], clf.predict(states[te])))
        chance = float(max(np.bincount(months[tr] - months[tr].min()).max() / len(tr), 1e-9))
        out["period_acc"] = round(acc, 4)
        out["period_chance"] = round(chance, 4)
    return out


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
        _val_split,
        compute_whitening,
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
    measured = sorted(set(gated) | YARDSTICK_PENDING)
    real_folds: dict[str, list[float]] = {o: [] for o in measured}
    shuf_folds: dict[str, list[float]] = {o: [] for o in measured}
    sf_var_real: list[float] = []
    sf_var_shuf: list[float] = []
    seen: set[str] = set()

    was_training = model.training
    model.eval()
    # Grade the FROZEN donor-boundary transform (DEC-022) applied to held-out
    # states — the honest holdout number, no refit-on-the-graded-split. Only a
    # checkpoint that ships no transform at all gets one derived here.
    if not getattr(model, "whiten_on", False):
        compute_whitening(model, vocab, cfg, val)
    try:
        for gi, g in enumerate(fold_ids):
            gset = set(g.tolist())
            batch = [s for s in val if s["customer"] in gset]
            if len(batch) < 2:
                continue
            from looking_glass.cfm_training import task_losses_chunked

            torch.manual_seed(seed + gi)  # identical stochastic draws (real vs destroyed)
            aux_r: dict = {}
            real = task_losses_chunked(model, vocab, batch, cfg, batch=128, aux=aux_r)
            shuf_seqs = _destroyed(batch, seed * 1000 + gi)
            torch.manual_seed(seed + gi)
            aux_s: dict = {}
            shuf = task_losses_chunked(model, vocab, shuf_seqs, cfg, batch=128, aux=aux_s)
            if "sf_target_var" in aux_r and "sf_target_var" in aux_s:
                sf_var_real.append(aux_r["sf_target_var"])
                sf_var_shuf.append(aux_s["sf_target_var"])
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
            # grade the CONSUMED representation (donor boundary, DEC-022):
            # whitened when the transform is set — this is what every
            # downstream head actually reads
            zc = model.donor_batch(h)
            states.append(zc.detach().float().cpu().numpy())
        z = np.concatenate(states, axis=0) if states else np.zeros((0, 1))
    finally:
        model.train(was_training)

    objectives = []
    rows = []
    report_only = []
    for o in measured:
        if o not in real_folds or not real_folds[o]:
            continue
        gated_o = o not in YARDSTICK_PENDING
        r = np.asarray(real_folds[o])
        s = np.asarray(shuf_folds[o])
        if o == "sf" and sf_var_real and sf_var_shuf:
            # R² yardstick (DEC-018): dimensionless, immune to the smoothing
            # artifact — R²_real - R²_shuf = L_shuf/V_shuf - L_real/V_real
            vr = np.maximum(np.asarray(sf_var_real[: len(r)]), 1e-12)
            vs = np.maximum(np.asarray(sf_var_shuf[: len(s)]), 1e-12)
            skill = s[: len(r)] / vs - r / vr
        else:
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
        row = {
            "check": f"portfolio: {o} structure-skill > 0",
            "achieved": f"{skill.mean():+.4f} ± {se:.4f} ({len(skill)} folds)",
            "ok": bool(skill.mean() > 0),
        }
        if not gated_o:
            row["ok"] = True  # report-only: yardstick pending (see receipt note)
            row["check"] = f"portfolio: {o} structure-skill (REPORT-ONLY, yardstick pending)"
            report_only.append(o)
        rows.append(row)
    geometry = _geometry(z, seed + 2) if len(z) >= 4 else {}
    canaries = _canaries(z, val, seed + 3) if len(z) >= 4 else {}
    if geometry:
        # PRIMARY geometry gate (v4.0, DEC-029): SCALE-FREE, self-calibrating.
        # The score normalizes the consumed representation's participation ratio
        # against the finite-sample Marchenko-Pastur (pure-noise) null, so the
        # bar is valid at any dimension / sample size (the old static 0.30
        # fraction was calibrated only for D=256, B~2.6k). Score > 0 means the
        # representation is more isotropic than random noise.
        rows.append(
            {
                "check": "geometry: isotropy score vs Marchenko-Pastur null > 0",
                "achieved": (
                    f"score {geometry['pr_score']} (PR/dim {geometry['pr_frac']} "
                    f"vs noise {geometry['mp_null_frac']})"
                ),
                "ok": bool(geometry["pr_score"] > 0.0),
            }
        )
    if canaries:
        if "group_auc" in canaries:
            rows.append(
                {
                    "check": "canary: state cannot predict A/B arm (AUC < 0.65)",
                    "achieved": f"AUC {canaries['group_auc']}",
                    "ok": bool(canaries["group_auc"] < 0.65),
                }
            )
        if "period_acc" in canaries:
            rows.append(
                {
                    "check": "canary: state cannot predict period bucket (< 1.5x chance)",
                    "achieved": f"acc {canaries['period_acc']} (chance {canaries['period_chance']})",
                    "ok": bool(canaries["period_acc"] < 1.5 * canaries["period_chance"]),
                }
            )
    missing = sorted(set(cfg.objectives) - seen - GATED_EXCLUSIONS - YARDSTICK_PENDING)
    # per-task contribution at the kept state, in the CURRENT balance mode
    # (DEC-014): DWA -> w_k*L_k with the final learned rate-weights; Kendall
    # (uncertainty mode) -> 0.5*exp(-s_k)*L_k. Recorded either way.
    mode = getattr(cfg, "weight_mode", "uncertainty")
    ftw = getattr(cfg, "final_task_weights", None) or {}
    contributions = {}
    lv = {k: float(v.detach().reshape(-1)[0]) for k, v in model.log_var.items()}
    for o in objectives:
        k = o["objective"]
        if mode == "dwa" and k in ftw:
            contributions[k] = float(ftw[k]) * o["real"]
        elif k in lv:
            contributions[k] = 0.5 * float(np.exp(-lv[k])) * o["real"]
    tot = sum(abs(v) for v in contributions.values()) or 1.0
    contributions = {k: round(v / tot, 4) for k, v in contributions.items()}
    ok = all(r["ok"] for r in rows) if rows else False
    receipt = {
        "tag": tag,
        "seed": seed,
        "folds": folds,
        "n_val": len(val),
        "n_seqs": len(seqs),
        "sf_mode": getattr(cfg, "sf_mode", "purchase"),
        "agg_horizons_days": list(getattr(cfg, "agg_horizons_days", []) or []),
        "task_log_var": {
            k: round(float(v.detach().reshape(-1)[0]), 4) for k, v in model.log_var.items()
        },
        "objectives": objectives,
        "objectives_missing": missing,
        "yardstick_pending": sorted(YARDSTICK_PENDING),
        "report_only": report_only,
        "canaries": canaries,
        "task_contributions": contributions,
        "sf_target_var_real": [round(float(v), 4) for v in sf_var_real],
        "sf_target_var_shuf": [round(float(v), 4) for v in sf_var_shuf],
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
    obj_rows = [r for r in rows if r["check"].startswith("portfolio:")]
    for o, row in zip(objectives, obj_rows):
        status = "PASS" if row["ok"] else "FAIL"
        if "REPORT-ONLY" in row["check"]:
            status = "REPORT"
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
