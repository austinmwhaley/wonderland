"""Adjudication: per diet, train (or load) candidates -> OPE panel -> live
ground truth -> verdict table + OPE-vs-truth rank correlation. The rank
correlation is the headline Tribunal metric: it scores whether our doctrine
would have picked the winner before spending a step of live traffic.

Pure-OPE reruns: run_diet_ope_only loads saved candidate weights (no
retraining) and re-runs panels + gate only. Data and agent weights untouched.
"""

import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environments.registry import make_env  # noqa: E402
from white_queen import db  # noqa: E402
from white_queen.tribunal.candidates import CANDIDATES, train_candidate  # noqa: E402
from white_queen.tribunal.ope import estimators as ope  # noqa: E402
from white_queen.tribunal.ope import gate as gate_mod  # noqa: E402
from white_queen.tribunal.ope.receipts import behavior_stats, spearman  # noqa: E402

log = logging.getLogger("white_queen.tribunal")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def env_receipt():
    """Runtime environment versions. Determinism claims ("seeded reruns match")
    are only valid on matching stacks — torch 2.2-vs-2.11 drift is exactly the
    class of silent change this pins down. Recorded in every verdict."""
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            try:
                out["cuda_device"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception as e:
        out["torch"] = f"unavailable ({e!r})"
    return out


def _discounted_rollout(cand, env, episodes, gamma, proxy_temp=None, seed=0, max_steps=None):
    """Live rollout in discounted units. proxy_temp=None -> argmax policy
    (what ships); else the OPE proxy (what was judged). Their gap quantifies
    exactly what ESS-forced softness costs — reported, never hidden.
    max_steps=None -> 10_000 safety cap (infinite-loop guard, not tuning)."""
    import numpy as _np

    rng = _np.random.default_rng(seed)
    env_seed = getattr(env, "_wq_seed", 0)
    max_steps = int(max_steps or 10_000)
    rets = []
    for e in range(episodes):
        try:
            state, _ = env.reset(seed=env_seed + e)
        except TypeError:
            state, _ = env.reset()
        disc, g, t, done = 1.0, 0.0, 0, False
        while not done and t < max_steps:
            if proxy_temp is None:
                a = cand.act(state, eval=True)
            else:
                from white_queen.tribunal.ope.protocols import sample_actions

                p = cand.action_probs(state[None, :], temperature=proxy_temp)
                a = int(sample_actions(rng, p)[0])
            state, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            g += disc * float(r)
            disc *= gamma
            t += 1
        rets.append(g)
    return float(_np.mean(rets))


def run_diet(cfg, db_path, diet, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Tribunal env seed is reproducibility only (fixed live-truth stream),
    # not a tuning dial. Override via cfg["tribunal_env_seed"] if needed.
    env = make_env(cfg["env"], seed=int(cfg.get("tribunal_env_seed", 999)))
    data = db.load_diet(db_path, diet)
    bstat = behavior_stats(data, cfg["gamma"])
    n_ep = len(np.unique(np.asarray(data["episode"])))
    # Ground-truth budget autotunes from data unless pinned: enough live
    # episodes to match half the logged count, [20, 100].
    gt_eps = cfg.get("ground_truth_episodes")
    if gt_eps is None:
        gt_eps = int(min(100, max(20, n_ep // 2)))
    log.info(f"tribunal [{diet}]: N={data['N']} behavior_mean={bstat['mean']:.1f}")
    panels = {}
    for name in CANDIDATES:
        ckpt = os.path.join(out_dir, f"{name}_{diet}.pt")
        cand = train_candidate(name, env, data, cfg, ckpt)
        # ope_meta=None / fqe base None both mean autotune (recommended).
        # Budgets scale with data AND horizon (see autotune ledger).
        # Pin via cfg["ope_fqe_cfg"] if needed.
        fqe_base = dict(cfg.get("ope_fqe_cfg") or {})
        from white_queen.tribunal.ope.cache import file_hash

        _wh, _cdir = None, cfg.get("ope_cache_dir")
        try:
            _wh = file_hash(ckpt)
        except Exception:
            pass
        p = ope.panel(
            data,
            cand,
            cfg["gamma"],
            cfg.get("ope_meta"),
            fqe_base,
            cand_id=f"{name}_{diet}",
            cache_dir=_cdir,
            weights_hash=_wh,
            fast=cfg.get("fast", False),
            ensemble_K=cfg.get("ensemble_K"),
        )
        p["truth"] = _discounted_rollout(cand, env, gt_eps, cfg["gamma"])
        p["proxy_truth"] = _discounted_rollout(
            cand, env, gt_eps, cfg["gamma"], proxy_temp=p["temperature"]
        )
        panels[name] = p
        log.info(
            f"  {name:4s} blend={p['blended']:7.1f} DR={p['dr']:7.1f} "
            f"WDR={p['wdr']:7.1f} MAGIC={p['magic']:7.1f} "
            f"MIS={p['mis']:7.1f}(e={p['mis_info']['mis_ess_frac']:.3f}) "
            f"EFQE={p['efqe']['mean']:6.1f}±{p['efqe']['disagreement']:.1f} "
            f"LSTDQ={p['lstdq']['dm']:6.1f} FVE={p['fve_dm']:6.1f} "
            f"MB={p['mb']['mb']:6.1f}±{p['mb']['se']:.1f} "
            f"GD={p['gdice_mis']:6.1f} SLOPE={p['slope_pick']}:{p['slope_val']:.1f}"
        )
        log.info(
            f"         WIS={p['wis']:7.1f} FQE={p['fqe_dm']:7.1f} "
            f"lam={p['lambda_dr']:.2f} "
            f"ess={p['ess_frac']:.3f} T={p['temperature']} "
            f"TRUTH={p['truth']:7.1f} PROXY={p['proxy_truth']:7.1f} "
            f"below_anchor={p['below_anchor']}"
        )
    rows = gate_mod.adjudicate(
        panels, bstat["mean"], bstat["std"], cfg.get("gate"), cfg.get("ope_meta"), n_episodes=n_ep
    )
    order = list(panels)
    rho = spearman([panels[c]["blended"] for c in order], [panels[c]["truth"] for c in order])
    gate_deployed = [c for c, r in rows.items() if r["deploy"]]
    best_true = max(order, key=lambda c: panels[c]["truth"])
    # Smart judge (truth-blind, deterministic): report card in, decision out.
    from white_queen.tribunal.ope import judge as judge_mod

    judge_v = judge_mod.judge_diet(
        rows,
        bstat["mean"],
        bstat["std"],
        cfg.get("gate"),
        cfg.get("risk_aversion", judge_mod.DEFAULT_RISK_AVERSION),
        allow_deploy=cfg.get("allow_deploy", True),
        use_rank=(cfg.get("tier") != "screen"),
    )
    deployed = judge_v["deployed"]
    # Receipt: what was configured (None=auto) + what was actually used.
    example_gate = next(iter(rows.values())).get("gate") if rows else None
    report = {
        "diet": diet,
        "behavior_mean": round(bstat["mean"], 1),
        "gate_cfg": {
            "configured": cfg.get("gate"),
            "resolved": example_gate,
            "ope_meta_configured": cfg.get("ope_meta"),
            "ground_truth_episodes": gt_eps,
        },
        "rows": rows,
        "env": env_receipt(),
        "spearman_blended_truth": round(rho, 3),
        "deployed": deployed,
        "true_best": best_true,
        "gate_picked_best": best_true in deployed,
        "gate_deployed": gate_deployed,
        "decided_by": "judge",
        "judge": judge_v,
        "judge_rationale": judge_mod.explain_diet(diet, bstat["mean"], judge_v, rows),
    }
    with open(os.path.join(out_dir, f"verdict_{diet}.json"), "w") as f:
        json.dump(report, f, indent=2)
    log.info(
        f"  verdict: deployed={deployed} true_best={best_true} "
        f"picked_best={best_true in deployed} spearman={rho:.3f}"
    )
    return report


def run(cfg, db_path, out_dir="white_queen_verdicts"):
    reports = {d: run_diet(cfg, db_path, d, out_dir) for d in cfg["diets"]}
    log.info("\n== TRIBUNAL SUMMARY ==")
    for d, r in reports.items():
        marks = {c: ("DEPLOY" if c in r.get("deployed", []) else "hold") for c in r["rows"]}
        log.info(
            f"[{d}] {marks} spearman={r['spearman_blended_truth']:.3f} "
            f"picked_best={r['gate_picked_best']}"
        )
    return reports


def run_diet_ope_only(cfg, db_path, diet, out_dir, ckpt_dir, mu_source="logged"):
    """Pure-OPE rerun: load saved candidate weights, re-run panels + gate.

    No candidate training, no new data. ckpt_dir holds {name}_{diet}.pt files
    (e.g. v8 verdicts). OPE training (FQE/MIS/dynamics) still runs — that's
    estimator fitting, not agent training. Full-autotune FQE budget (no legacy
    hint cap) unless cfg["ope_fqe_cfg"] pins steps_max.
    mu_source="estimated": industry track — behavior probs re-estimated from
    the logs (behavior.with_estimated_propensities) instead of exact lab
    propensities. Academic track ("logged") is the default.
    """
    import os as _os

    _os.makedirs(out_dir, exist_ok=True)
    from white_queen.tribunal.candidates import CANDIDATES, load_candidate

    env = make_env(cfg["env"], seed=int(cfg.get("tribunal_env_seed", 999)))
    data = db.load_diet(db_path, diet)
    mu_info = {"mu_source": "logged"}
    if mu_source == "estimated":
        from white_queen.tribunal.ope.behavior import with_estimated_propensities

        data, mu_info = with_estimated_propensities(data)
        log.info(
            f"industry mu-hat: acc={mu_info['behavior']['accuracy']} "
            f"floor={mu_info['floor']} clipped={mu_info['clipped_frac']}"
        )
    bstat = behavior_stats(data, cfg["gamma"])
    n_ep = len(np.unique(np.asarray(data["episode"])))
    gt_eps = cfg.get("ground_truth_episodes")
    if gt_eps is None:
        gt_eps = int(min(100, max(20, n_ep // 2)))
    log.info(f"tribunal-ope-only [{diet}]: N={data['N']} behavior_mean={bstat['mean']:.1f}")
    from white_queen.tribunal.ope.cache import file_hash

    _names = list(cfg.get("candidates", CANDIDATES))
    _tier = cfg.get("tier")  # None = certify/full, "screen" = cheap+no-deploy
    _cache_dir = cfg.get("ope_cache_dir")
    panels = {}
    for name in _names:
        ckpt = _os.path.join(ckpt_dir, f"{name}_{diet}.pt")
        cand = load_candidate(name, env, data, cfg, ckpt)
        # Full autotune: no legacy ope_fqe_steps hint cap.
        fqe_base = dict(cfg.get("ope_fqe_cfg") or {})
        _wh = None
        try:
            _wh = file_hash(ckpt)
        except Exception:
            pass
        _pk = {"cand_id": f"{name}_{diet}", "cache_dir": _cache_dir, "weights_hash": _wh}
        if _tier == "screen":
            # Screen tier: small explicit budgets; may over-hold, never deploy.
            fqe_base = dict(fqe_base, steps_max=2000, eval_every=200, patience=2)
            p = ope.panel(
                data,
                cand,
                cfg["gamma"],
                {"bootstrap_B": 200},
                fqe_base,
                ensemble_K=2,
                dice_steps=1000,
                magic_B=200,
                fast=cfg.get("fast", False),
                **_pk,
            )
        else:
            p = ope.panel(
                data,
                cand,
                cfg["gamma"],
                cfg.get("ope_meta"),
                fqe_base,
                fast=cfg.get("fast", False),
                ensemble_K=cfg.get("ensemble_K"),
                **_pk,
            )
        p["truth"] = _discounted_rollout(cand, env, gt_eps, cfg["gamma"])
        p["proxy_truth"] = _discounted_rollout(
            cand, env, gt_eps, cfg["gamma"], proxy_temp=p["temperature"]
        )
        panels[name] = p
        log.info(
            f"  {name:4s} blend={p['blended']:7.1f} DR={p['dr']:7.1f} "
            f"FQE={p['fqe_dm']:7.1f} ess={p['ess_frac']:.3f} T={p['temperature']} "
            f"TRUTH={p['truth']:7.1f} PROXY={p['proxy_truth']:7.1f}"
        )
    rows = gate_mod.adjudicate(
        panels, bstat["mean"], bstat["std"], cfg.get("gate"), cfg.get("ope_meta"), n_episodes=n_ep
    )
    order = list(panels)
    rho = spearman([panels[c]["blended"] for c in order], [panels[c]["truth"] for c in order])
    gate_deployed = [c for c, r in rows.items() if r["deploy"]]
    best_true = max(order, key=lambda c: panels[c]["truth"])
    from white_queen.tribunal.ope import judge as judge_mod

    judge_v = judge_mod.judge_diet(
        rows,
        bstat["mean"],
        bstat["std"],
        cfg.get("gate"),
        cfg.get("risk_aversion", judge_mod.DEFAULT_RISK_AVERSION),
        allow_deploy=cfg.get("allow_deploy", True),
        use_rank=(cfg.get("tier") != "screen"),
    )
    deployed = judge_v["deployed"]
    example_gate = next(iter(rows.values())).get("gate") if rows else None
    report = {
        "diet": diet,
        "behavior_mean": round(bstat["mean"], 1),
        "gate_cfg": {
            "configured": cfg.get("gate"),
            "resolved": example_gate,
            "ope_meta_configured": cfg.get("ope_meta"),
            "ground_truth_episodes": gt_eps,
            "mode": "ope-only (loaded candidates, no retraining)",
            "mu_source": mu_info.get("mu_source", "logged"),
            "mu_info": mu_info.get("behavior", {}),
        },
        "rows": rows,
        "env": env_receipt(),
        "spearman_blended_truth": round(rho, 3),
        "deployed": deployed,
        "true_best": best_true,
        "gate_picked_best": best_true in deployed,
        "gate_deployed": gate_deployed,
        "decided_by": "judge",
        "judge": judge_v,
        "judge_rationale": judge_mod.explain_diet(diet, bstat["mean"], judge_v, rows),
    }
    with open(_os.path.join(out_dir, f"verdict_{diet}.json"), "w") as f:
        json.dump(report, f, indent=2)
    log.info(
        f"  verdict: deployed={deployed} true_best={best_true} "
        f"picked_best={best_true in deployed} spearman={rho:.3f}"
    )
    return report


def run_ope_only(cfg, db_path, out_dir, ckpt_dir, mu_source="logged"):
    reports = {
        d: run_diet_ope_only(cfg, db_path, d, out_dir, ckpt_dir, mu_source=mu_source)
        for d in cfg["diets"]
    }
    log.info(f"\n== TRIBUNAL OPE-ONLY SUMMARY (mu_source={mu_source}) ==")
    for d, r in reports.items():
        marks = {c: ("DEPLOY" if c in r.get("deployed", []) else "hold") for c in r["rows"]}
        log.info(
            f"[{d}] {marks} spearman={r['spearman_blended_truth']:.3f} "
            f"picked_best={r['gate_picked_best']}"
        )
    return reports
