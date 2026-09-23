"""Single training governor for every white_queen-owned model.

Before: five hand-rolled loops (FQE, FVE, dynamics, BC, MIS/DICE) with
subtly different bugs — BC/MIS/DICE didn't early-stop at all, tolerances
differed, LR was hardcoded, best weights weren't restored, NaN losses
silently poisoned weights.

After: one function. Cosine-annealed LR (fast early, fine late, no decay
tuning) + validation early-stop with best-restore + NaN guards. Same math
per model, no tuning dials: budgets come from autotune.resolve_*_cfg.

Two modes (honest about minimax):
- supervised: val_fn provided -> early-stop on held-out loss, restore best.
  Used by FQE, FVE, dynamics, BC (BC gains a real train/val split here).
- saddle: val_fn=None -> no fake early-stop on the minimax objective
  (that selects for collapsed ratios). Cosine schedule + budget + NaN/div
  guards only. Used by MIS/DualDICE and GradientDICE. Diagnostics
  (mean_w, ESS) are reported, not optimized.

Determinism: governor owns no RNG; trainers pass step closures over their
seeded rngs. Device/dtype: trainers' closures handle tensors; governor only
touches optimizer LRs and module state_dicts.
"""

import copy
import math


def seed_all(seed):
    """Seed torch (+numpy global legacy) BEFORE net construction. govern()
    seeds again at start for the training draws; this call fixes the init
    lottery (v9-vs-v12 FQE gap: identical configs, different inits). None = opt out."""
    if seed is None:
        return
    try:
        import torch as _torch
        _torch.manual_seed(int(seed))
    except Exception:
        pass
    try:
        import numpy as _np
        _np.random.seed(int(seed) % (2 ** 32))
    except Exception:
        pass


def cosine_lr(step, steps_max, lr_base, lr_min_ratio=0.01, warmup_frac=0.05):
    """LR at step: linear warmup then cosine decay to lr_min.

    Warmup avoids early divergence with large autotuned batches; cosine gives
    fast early progress and fine late convergence with no decay schedule to
    tune. Pure function of (step, budget) — fully adaptive via steps_max.
    """
    if steps_max <= 0:
        return lr_base * lr_min_ratio
    warmup = max(int(steps_max * warmup_frac), 1)
    lr_min = lr_base * lr_min_ratio
    if step < warmup:
        return lr_base * (float(step + 1) / warmup)
    t = (step - warmup) / max(steps_max - warmup, 1)
    t = min(max(t, 0.0), 1.0)
    return lr_min + 0.5 * (lr_base - lr_min) * (1.0 + math.cos(math.pi * t))


def _set_lr(optims, lr):
    for o in optims:
        for g in o.param_groups:
            g["lr"] = lr


def _snapshot(modules):
    return {k: copy.deepcopy(m.state_dict()) for k, m in modules.items()}


def _restore(modules, snap):
    for k, m in modules.items():
        m.load_state_dict(snap[k])


def govern(modules, optims, step_fn, val_fn, cfg):
    """Run training to completion. Returns info receipt.

    modules: {name: nn.Module} snapshotted at best (supervised) for restore.
    optims: list of optimizers sharing one cosine schedule.
    step_fn(n_steps, lr): run n_steps gradient steps at lr; return train loss
      (float, may be None/inf for saddle) — NaN train loss stops the run.
    val_fn(): held-out scalar, lower better (supervised only; None = saddle).
      CONTRACT: val_fn must measure generalization against targets that do
      not contain the modules' own outputs. A self-referential val (e.g. a
      tree-backup error mixing (1-Rho)*V(O)) scores untrained constant nets
      best and best-restore will crown them (observed: FVE DM -0.0). Use
      Monte-Carlo or otherwise grounded targets for selection.
    cfg keys (all autotuned upstream, overridable): steps_max, eval_every,
      patience, lr, lr_min_ratio, warmup_frac, tol_rel, tol_abs.

    Returns {"steps", "n_evals", "best_val" (or None), "stopped":
      "budget" | "patience" | "nan" | "saddle_budget",
      "lr_final", "lr_base"}.
    """
    steps_max = int(cfg.get("steps_max", 20000))
    eval_every = int(cfg.get("eval_every", 500))
    patience = int(cfg.get("patience", 5))
    lr_base = float(cfg.get("lr", 1e-3))
    lr_min_ratio = float(cfg.get("lr_min_ratio", 0.01))
    warmup_frac = float(cfg.get("warmup_frac", 0.05))
    tol_rel, tol_abs = float(cfg.get("tol_rel", 1e-4)), float(cfg.get("tol_abs", 1e-6))
    seed = cfg.get("seed", None)
    # Determinism (reliability fix): trainers seeded numpy but never torch, so
    # identical configs gave different models run-to-run (v9 vs v12 FQE gap).
    # Seeding here covers every governed model at once; pass seed=None to opt
    # out (e.g. deliberate diversity outside ensemble strides).
    if seed is not None:
        try:
            import torch as _torch
            _torch.manual_seed(int(seed))
        except Exception:
            pass

    saddle = val_fn is None
    best, best_snap, bad, steps, n_evals = None, None, 0, 0, 0
    stopped, train_loss = "budget", None

    def _tol(best):
        return max(tol_abs, abs(best) * tol_rel)

    if not saddle:
        best = val_fn()
        if not math.isfinite(best):
            best = float("inf")
        best_snap = _snapshot(modules)

    while steps < steps_max:
        if not saddle and bad >= patience:
            stopped = "patience"
            break
        lr = cosine_lr(steps, steps_max, lr_base, lr_min_ratio, warmup_frac)
        _set_lr(optims, lr)
        chunk = min(eval_every, steps_max - steps)
        train_loss = step_fn(chunk, lr)
        steps += chunk
        if train_loss is not None and not math.isfinite(float(train_loss)):
            stopped = "nan"
            break
        if saddle:
            continue
        n_evals += 1
        ve = val_fn()
        if not math.isfinite(ve):
            bad += 1
            continue
        if ve < best - _tol(best):
            best, bad = ve, 0
            best_snap = _snapshot(modules)
        else:
            bad += 1

    if saddle:
        stopped = "nan" if stopped == "nan" else "saddle_budget"
    elif stopped == "budget":
        pass  # ran out of budget with patience unexhausted
    if not saddle and best_snap is not None:
        _restore(modules, best_snap)
    return {"steps": steps, "n_evals": n_evals,
            "best_val": None if saddle else round(float(best), 5),
            "stopped": stopped, "lr_final": round(cosine_lr(
                min(steps, steps_max), steps_max, lr_base,
                lr_min_ratio, warmup_frac), 8),
            "lr_base": lr_base}
