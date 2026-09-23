"""The verdict, v4 (adaptive): ensemble in, go/no-go out. Strictness is
computed from the diet — bootstrap lower bounds against behavior-relative
bars plus an ESS floor. WIS direction is advisory-only (point estimate, no
CI, worst rank correlation in the panel). Disagreement among estimators is
reported, never averaged away.

gate_cfg=None and meta=None both mean autotune (recommended). Explicit keys
override per-rule.
"""
from .autotune import resolve_bar, resolve_bootstrap, resolve_gate
from .receipts import bootstrap_ci
import numpy as _np


def _json_num(x):
    """Round for JSON receipts; NaN/Inf/None -> None (strict JSON has no NaN;
    json.dump emits NaN by default — a corrupt artifact waiting to happen)."""
    if not isinstance(x, (int, float)):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return round(float(x), 1)


def adjudicate(panels, behavior_mean, behavior_std, gate_cfg=None, meta=None,
               n_episodes=None):
    """panels: {candidate: panel-dict from estimators.panel}.
    Returns {candidate: row} with deploy flags + reasoning.

    n_episodes=None -> inferred from the first panel's dr_vals length.
    """
    if n_episodes is None:
        try:
            n_episodes = len(next(iter(panels.values()))["dr_vals"])
        except Exception:
            n_episodes = 30
    g = resolve_gate(n_episodes, behavior_std, behavior_mean, gate_cfg)
    bar = resolve_bar(behavior_mean, behavior_std, g["rel_edge_std"])
    # Bootstrap resolution autotuned from n unless caller pins B/alpha.
    B = (meta or {}).get("bootstrap_B") if isinstance(meta, dict) else None
    al = (meta or {}).get("ci_alpha") if isinstance(meta, dict) else None
    B, al = resolve_bootstrap(n_episodes, B, al)
    rows = {}
    for name, p in panels.items():
        lo, hi = bootstrap_ci(p["dr_vals"], B=B, alpha=al)
        # IS CI kept as a diagnostic receipt only (deploy decision uses DR).
        # Tolerate panels without per-episode IS values (e.g. unit tests).
        try:
            blo, _ = bootstrap_ci(p["is_vals"], B=B, alpha=al)
        except KeyError:
            blo = None
        vetoes = []
        ess_ok = p["ess_frac"] >= g["min_ess_frac"]
        if not ess_ok:
            vetoes.append(f"ESS {p['ess_frac']:.3f} < {g['min_ess_frac']:.3f}")
        if not lo > bar:
            vetoes.append(f"DR lower-CI {lo:.1f} <= bar {bar:.1f}")
        # WIS is advisory only (v4+): point estimate with no CI and worst rank
        # correlation in the panel (often inverse). Recorded, never vetoes.
        wis_gap = float(p["wis"] - behavior_mean)
        advisories = []
        if not p["wis"] > behavior_mean:
            advisories.append(f"WIS disagrees ({wis_gap:+.1f}, advisory only)")
        # Soft-vs-sharp gap: offline estimate of the proxy problem — how far
        # the shippable-policy value (sharp DM) sits from the judged mush.
        # Advisory only (sharp path unvalidated at scale); never vetoes.
        try:
            _soft, _sharp = float(p["fqe_dm"]), float(p.get("sharp_dm"))
            import math as _math
            if _math.isfinite(_soft) and _math.isfinite(_sharp):
                advisories.append(
                    f"soft-vs-sharp gap {abs(_sharp - _soft):.1f} "
                    f"(soft {_soft:.1f}, sharp {_sharp:.1f})")
        except (TypeError, ValueError):
            pass
        # Level-vs-FQE gap: the median composite disagreeing with the ranker
        # flags single-estimator dominance (collapsed FQE or ecstatic MB).
        try:
            import math as _math2
            _lv, _fq = float(p.get("level_est")), float(p["fqe_dm"])
            _sc = max(float(behavior_std), 0.05 * abs(float(behavior_mean)) + 1e-9)
            if _math2.isfinite(_lv) and _math2.isfinite(_fq) and abs(_lv - _fq) > _sc:
                advisories.append(
                    f"level/FQE disagree by {abs(_lv - _fq):.1f} "
                    f"(level {_lv:.1f}, FQE {_fq:.1f})")
        except (TypeError, ValueError):
            pass
        # Hidden-confounding sensitivity (Rosenbaum/MSM): smallest odds-
        # distortion Gamma that could pull the decision statistic (DR, the
        # witness that actually gates) down to the bar. Large Gamma* = robust;
        # near 1 = fragile. For human-log behavior this is the one number that
        # says whether unobserved context could flip the call.
        sens = None
        try:
            from .sensitivity import gamma_star as _gstar
            _dv = p.get("dr_vals")
            if _dv:
                _w = p.get("ep_weights")
                if _w is None:
                    _w = [1.0] * len(_dv)
                gs, frontier = _gstar(_dv, _w, bar)
                _kw = ("already_below_bar" if gs is None else
                       ("inf" if gs == float("inf") else round(float(gs), 2)))
                sens = {"gamma_star": _kw,
                        "frontier": [(float(g), _json_num(m)) for g, m in frontier[:5]],
                        "target": "dr", "bar": round(bar, 2)}
                if isinstance(gs, float) and gs < 1.25:
                    advisories.append(
                        f"fragile to hidden confounding (Gamma*={gs:.2f}); "
                        f"unobserved context could flip the call")
        except Exception:
            sens = None
        rows[name] = {
            "blended": round(p["blended"], 1), "dr": round(p["dr"], 1),
            "dr_ci": [round(lo, 1), round(hi, 1)],
            "wis": round(p["wis"], 1), "is": round(p["is"], 1),
            "wdr": _json_num(p.get("wdr")), "magic": _json_num(p.get("magic")),
            "magic_w": p.get("magic_w", []),
            "fqe_dm": _json_num(p["fqe_dm"]),
            "level_est": _json_num(p.get("level_est")),
            "sharp_dm": _json_num(p.get("sharp_dm")),
            "efqe": {k: (round(v, 1) if isinstance(v, float) else v)
                     for k, v in p["efqe"].items() if k != "nets"},
            "lstdq": _json_num(p["lstdq"]["dm"]),
            "lstdq_cond": p["lstdq"]["cond"],
            "fve_dm": _json_num(p.get("fve_dm")),
            "mb": _json_num(p["mb"]["mb"]),
            "mb_se": p["mb"]["se"], "mb_sims": p["mb"]["sims"],
            "mb_sharp": _json_num((p.get("mb_sharp") or {}).get("mb")),
            "mb_sharp_se": (p.get("mb_sharp") or {}).get("se"),
            "gdice_mis": _json_num(p.get("gdice_mis")),
            "anchor": p["anchor"], "slope_pick": p["slope_pick"],
            "slope_val": p["slope_val"], "below_anchor": p["below_anchor"],
            "mis": _json_num(p.get("mis", float("nan"))),
            "support": p.get("support", {}),
            "sharp_info": p.get("sharp_info", {}),
            "fqe_info": p.get("fqe_info", {}),
            "fqe_behavior": _json_num(p.get("fqe_behavior")),
            "mis_ess_frac": round(p.get("mis_info", {}).get("mis_ess_frac", 0.0), 3),
            "lambda_dr": round(p["lambda_dr"], 2),
            "ess_frac": round(p["ess_frac"], 3),
            "temperature": p["temperature"], "rho_cap": round(p["rho_cap"], 2),
            "truth": _json_num(p.get("truth")),
            "deploy": not vetoes, "vetoes": vetoes, "advisories": advisories,
            # Evidence passthrough: per-episode DR contributions let the judge
            # compute exact bootstrap p-values + Holm (v11+). ~10KB/candidate.
            "dr_vals": [float(x) for x in p.get("dr_vals", [])],
            "ep_returns": [float(x) for x in p.get("ep_returns", [])],
            "dr_step": _json_num(p.get("dr_step")),
            "step_ess_frac": _json_num(p.get("step_ess_frac")),
            "dr_step_adv": _json_num(p.get("dr_step_adv")),
            "dr_step_num": p.get("dr_step_num", []),
            "dr_step_den": p.get("dr_step_den", []),
            "dr_step_adv_num": p.get("dr_step_adv_num", []),
            "dr_step_adv_den": p.get("dr_step_adv_den", []),
            "sensitivity": sens,
            "timing": dict(p.get("timing", {})),
            "gate": {"bar": round(bar, 2), **{k: round(v, 4) if isinstance(v, float) else v
                                             for k, v in g.items()}},
            "bootstrap": {"B": B, "alpha": al},
        }
    return rows
