"""The `response` outlet (SPINE §5 / DEC-384, built by DEC-426): what happens if we DO something.

The simulator is the fourth consumer (DEC-384), and it asks counterfactual questions of the
frozen object. DEC-384 ratified this content class as a published bucket with three BINDING
disciplines, all enforced here rather than in prose:

  (a) **conditional, never causal** — the head predicts the observed outcome given the action
      that was actually logged. Nothing in this module claims an uplift; causality is attached
      downstream, and only where the arm was logged as randomized;
  (b) **uncertainty widens where randomization is absent** — the scale factor is measured from
      held-out coverage, not typed;
  (c) **pre-action cuts** — the conditioning state is the readout at the step strictly BEFORE
      the action, so the outcome can never leak into its own prediction.

The encoder is NEVER trained here: it is frozen, read, and the head is fitted on population A
afterwards. The action enters as the recurrence's OWN transition, `Δ_t = y_t − fade(y_{t−1})`
— schema-agnostic (no business event names, no per-domain action code), and the same object the
state update consumed. The target is the stream's own discounted future value, the SF φ of
DEC-009 (`[log1p|value|, one-hot(event_type)]`), reached through the single reverse-scan
definition in `cfm_training.discounted_future` so the outlet and the objective cannot drift.

Everything is vectorized per batch (P16): one padded forward per chunk, no Python loop over
events.
"""

from __future__ import annotations

import math

PARTS = ("outcome_mean", "outcome_sigma", "support", "randomized")

#: principle-ok: evaluation-contract target for the conformal band. The scale that meets it is
#: MEASURED; this number only states the coverage the published interval claims, and the receipt
#: reports the empirical coverage actually achieved against it.
NOMINAL_COVERAGE = 0.80


def _z(nominal: float) -> float:
    """Two-sided normal quantile for the nominal coverage (stdlib only; no scipy round-trip)."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(0.5 + nominal / 2.0)


def derive_gamma(gaps_days) -> tuple[float, dict]:
    """Discount rate DERIVED from the stream's own cadence: γ = exp(-1/τ), τ = median gap.

    The horizon the response is measured over is therefore "one typical inter-event interval"
    of THIS stream — the same horizon-free reasoning as the SF objective (DEC-009), and it is
    reported so no reader has to guess what future was predicted.
    """
    import numpy as np

    g = np.asarray(list(gaps_days), dtype=np.float64)
    g = g[np.isfinite(g) & (g > 0)]
    if g.size == 0:
        raise ValueError(
            "response outlet: the stream has no positive inter-event gaps, so no horizon can "
            "be derived — the outlet needs a timed stream (declare as_of/cadence upstream)"
        )
    tau = float(np.median(g))
    gamma = float(math.exp(-1.0 / max(tau, 1e-6)))
    return gamma, {"tau_days": round(tau, 6), "gamma": round(gamma, 6), "n_gaps": int(g.size)}


def _fade(y_prev, dt_seconds, half_life_days):
    """The serving fade law, imported by reference so train/serve/simulate share ONE definition."""
    from looking_glass.cfm_state import fade

    return fade(y_prev, dt_seconds, half_life_days)


def declare_action_design(db) -> dict:
    """Read the ACTION declaration from the place that actually knows it: the stream's truth file.

    `CFMConfig.exogenous_events` / `action_fields` default to empty, and nothing ever derived
    them — the frozen demo artifact therefore trained with `exogenous_events=[]`, so EVERY event
    (including the generator's randomized arm) entered as endogenous, the self-supervised mask
    never excluded an action target, and this outlet had nothing to condition on. The generator
    writes the design into `<data dir>/reference_truth.json` → `meta`:
    `action_keys`, `arm_key`, `action.action_period_seconds`. Reading it is what "exogenous
    events are data/schema-declared" (v6 zero-domain-literals) requires; typing
    `("email_send",)` in source would be the banned alternative.

    Raises (never defaults) when the stream declares nothing: a guessed action family would let
    the outlet publish a causal-looking column with no randomization behind it.
    """
    import json
    from pathlib import Path

    root = (
        Path(db).parent
        if str(db).endswith((".parquet", ".arrow", ".feather", ".csv"))
        else Path(db)
    )
    cand = []
    manifest = root / "manifest.json"
    if manifest.exists():
        try:
            ref = json.loads(manifest.read_text()).get("truth")
            if ref:
                cand.append(root / ref)
        except (ValueError, OSError):
            pass  # an unreadable manifest is reported below, with the paths that were tried
    cand += [root / "reference_truth.json", *sorted(root.glob("*_truth.json"))]
    for path in cand:
        if not path.exists():
            continue
        try:
            meta = json.loads(path.read_text()).get("meta", {})
        except (ValueError, OSError):
            continue
        keys = [str(k) for k in (meta.get("action_keys") or ())]
        if not keys:
            continue
        action = meta.get("action") or {}
        return {
            "source": str(path),
            "action_keys": tuple(keys),
            "arm_key": str(meta.get("arm_key") or ""),
            "action_period_seconds": action.get("action_period_seconds"),
            "action_design": action.get("action_design"),
        }
    raise LookupError(
        f"no action declaration found for stream {db}: looked for "
        f"{', '.join(str(p.name) for p in cand) or 'nothing'} in {root}. The `response` outlet "
        f"predicts outcomes GIVEN the logged action, so it needs the stream to declare which "
        f"events are actions (the generator's truth file `meta.action_keys`). Declaring them by "
        f"guessing an event name would publish a causal-looking column with no randomization "
        f"behind it — refuse instead (Principle 7)."
    )


def _action_masks(group, design: dict):
    """(B,T) action / randomized masks for a chunk, from the declared design — vectorized.

    A row is an ACTION iff it carries one of the declared `action_keys` (non-null), which is how
    the generator marks the arm; it is RANDOMIZED iff the key it varied (`target_key`) is the
    declared `arm_key` — the one factor the design actually randomized (DEC-384b: causality
    attaches only there). `target_key` is used ONLY as this label, never as a head feature, so
    the published columns cannot leak it into a prediction.
    """
    import numpy as np

    keys = [k for k in design.get("action_keys", ()) if k]
    arm = design.get("arm_key") or ""
    B = len(group)
    T = max((len(s["event_type"]) for s in group), default=0)
    act = np.zeros((B, max(T, 1)), dtype=bool)
    rnd = np.zeros((B, max(T, 1)), dtype=bool)
    for b, s in enumerate(group):
        n = len(s["event_type"])
        if not n:
            continue
        mask = np.zeros(n, dtype=bool)
        numeric_mask = s.get("numeric_mask") or {}
        attrs = s.get("attrs") or {}
        for k in keys:
            for src in (numeric_mask, attrs):
                v = src.get(k)
                if v is None:
                    continue
                arr = np.asarray(v)
                if arr.dtype == bool:
                    mask |= arr
                else:
                    # principle-ok: object-dtype nulls are `None`, so the comparison is
                    # elementwise inside numpy — no Python loop over events.
                    mask |= np.not_equal(arr, None, dtype=bool)  # noqa: E711
        act[b, :n] = mask
        if arm:
            tk = attrs.get("target_key")
            if tk is not None:
                same = np.asarray(tk, dtype=object) == arm
                rnd[b, :n] = same & mask
        # a run that already declared actions (cfg.exogenous_events) keeps them: union
        pre = s.get("is_action")
        if pre is not None:
            act[b, :n] |= np.asarray(pre, dtype=bool)
    return act, rnd


def collect(model, cfg, seqs, *, gamma: float, action_design: dict, chunk: int | None = None):
    """Vectorized sample rows over logged ACTION events: (pre-state, Δ, discounted future).

    Returns column arrays (numpy) ready to fit, or None when the stream has no action events.
    `pre` is the readout strictly before the action (DEC-384c); `delta` is what the action did
    to the state; `target` is the SF value from the action step onward at the derived γ.
    `action_design` (from `declare_action_design`) says which rows are actions and which were a
    LOGGED randomized arm (DEC-384b) — the outlet takes its action semantics from the stream's
    declaration, never from a name in source.
    """
    import numpy as np
    import torch

    from looking_glass.cfm_config import TIME_UNIT_SECONDS
    from looking_glass.cfm_training import _collate, discounted_future, embed_batch_size

    vocab = model.vocab
    dev = model._dev()
    batch = chunk or embed_batch_size(cfg)
    cols = {"pre": [], "delta": [], "target": [], "customer": [], "randomized": [], "post": []}
    n_actions = 0
    for start in range(0, len(seqs), batch):
        group = seqs[start : start + batch]
        t = _collate(group, vocab, dev)
        act_np, rnd_np = _action_masks(group, action_design)
        act = torch.as_tensor(act_np, device=dev) & (t["mask"] > 0)
        rand = torch.as_tensor(rnd_np, device=dev)
        with torch.no_grad():
            y, _h = model.encode(t)  # (B,T,D) per-step readout — the same call the objective uses
            # --- target: the stream's own discounted future value (DEC-009's agnostic φ) ----
            v = torch.log1p(t["val"].abs())
            oh = torch.nn.functional.one_hot(
                t["et"].clamp(max=vocab.n_et - 1), num_classes=vocab.n_et
            ).float()
            phi = torch.cat([v, oh], dim=-1)
            dt_days = torch.expm1(t["dt"]) / TIME_UNIT_SECONDS
            decay = torch.full_like(dt_days, math.log(max(gamma, 1e-9)))
            decay = torch.exp(decay * dt_days)
            R = discounted_future(phi, decay)  # (B,T,PD)
            # --- conditioning: strictly pre-action state, and the action's own transition ----
            if not bool(act.any()):
                continue
            y_prev = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)  # shift by one
            dt_sec = torch.expm1(t["dt"])
            faded = torch.cat(
                [
                    torch.zeros_like(y[:, :1]),
                    # `t["dt"]` is already (B,T,1), so it broadcasts against y=(B,T,D) as-is.
                    _fade(y[:, :-1], dt_sec[:, 1:], cfg.state_half_life_days),
                ],
                dim=1,
            )
            delta = y - faded  # what THIS event did to the state
        idx = act.nonzero(as_tuple=True)
        if idx[0].numel() == 0:
            continue
        n_actions += int(idx[0].numel())
        # Every selection is a tensor gather — no Python loop over events (P16). The customer
        # labels are gathered by the SAME indices, so a row can cannot mismatch its state.
        labels = np.asarray([str(s.get("customer")) for s in group], dtype=object)[
            idx[0].cpu().numpy()
        ]
        rand_row = rand[idx].cpu().numpy().astype(np.float64)
        cols["pre"].append(y_prev[idx].cpu().numpy())
        cols["delta"].append(delta[idx].cpu().numpy())
        # the readout AT the action step: it has already absorbed the action and sits one step
        # from the outcome, so it is a LEAKY ceiling. It is never a feature and never a claim —
        # it exists so the receipt can say how much signal there was to find, which is the only
        # way a reader can tell "the outlet is weak" from "there was nothing to decode".
        cols["post"].append(y[idx].float().cpu().numpy())
        cols["target"].append(R[idx].float().mean(dim=-1).cpu().numpy())
        cols["customer"].append(labels)
        cols["randomized"].append(rand_row)
    if not cols["pre"]:
        return None
    return {
        "pre": np.concatenate(cols["pre"], axis=0),
        "delta": np.concatenate(cols["delta"], axis=0),
        "post": np.concatenate(cols["post"], axis=0),
        "target": np.concatenate(cols["target"], axis=0),
        "customer": [c for group in cols["customer"] for c in group],
        "randomized": np.concatenate(cols["randomized"], axis=0),
        "n_actions": int(n_actions),
    }


def _standardize(Xa, Xb):
    import numpy as np

    mu = Xa.mean(0)
    sd = Xa.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (Xa - mu) / sd, (Xb - mu) / sd, mu, sd


def _measured_sigma(resid, nominal=None):
    """The interval half-width for a group of held-out residuals: split conformal, MEASURED.

    The published sigma is the `(n+1)*nominal/n` quantile of `|residual|` divided by the nominal
    normal quantile, so `[mu +/- z*sigma]` covers the nominal fraction of the calibration rows BY
    CONSTRUCTION, and the coverage actually achieved is reported beside the claim. Nothing is
    typed: the width comes out of the held-out residuals, and a group whose data is genuinely
    noisier is MEASURED wider (DEC-384b), not multiplied by a fudge factor.

    This replaces a fitted heteroscedastic scale head, which the real artifact showed to be
    unmeasurably bad: it predicted sigma ~ 1e-6 for a large share of rows while missing them
    badly, so the single multiplier that reached nominal coverage came out at **22932x** —
    +-3000 intervals for the well-predicted customers and tight ones for the over-confident
    rows, exactly backwards. Calibrated-homoscedastic beats uncalibrated-heteroscedastic:
    sharpness only counts AFTER calibration (P8), and `mean_interval_width` is receipted so a
    reader can see the sharpness that was traded for honesty rather than mistake width for
    virtue.
    """
    import numpy as np

    nominal = NOMINAL_COVERAGE if nominal is None else nominal
    resid = np.abs(np.asarray(resid, dtype=np.float64))
    n = resid.size
    if n == 0:
        raise ValueError("response outlet: cannot calibrate an interval on zero rows")
    z = _z(nominal)
    # finite-sample split-conformal level; clamps to 1.0 when n is too small to certify
    level = min(1.0, (n + 1) * nominal / n)
    sigma = float(np.quantile(resid, level) / z)
    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError(
            f"response outlet: calibrated sigma is {sigma} — refuse to publish an interval "
            f"that is not a positive number"
        )
    return sigma, float(np.mean(resid <= z * sigma))


def fit(rows_a, rows_b, *, seed: int = 65537):
    """Fit the conditional-outlet head on A rows and calibrate its interval on B rows.

    ONE ridge on the design `[pre | delta | 1]` predicts the conditional mean; the interval comes
    from the held-out residuals (`_measured_sigma`), measured separately for the rows whose arm
    was logged as randomized and those whose was not (DEC-384b). The encoder is never touched.
    """
    import numpy as np
    from sklearn.linear_model import Ridge

    Xa = np.hstack([rows_a["pre"], rows_a["delta"], np.ones((len(rows_a["pre"]), 1))])
    Xb = np.hstack([rows_b["pre"], rows_b["delta"], np.ones((len(rows_b["pre"]), 1))])
    Xa, Xb, mu, sd = _standardize(Xa, Xb)
    ya, yb = rows_a["target"], rows_b["target"]
    # principle-ok: the ridge penalty is the sklearn default on standardized columns and the
    # head never ships as a decision policy — the acceptance gate is the paired-CI receipt, not
    # this fit's own loss. A learned penalty would select on the same rows the gate measures.
    mean_head = Ridge(random_state=seed).fit(Xa, ya)
    mu_b = mean_head.predict(Xb)
    resid_b = yb - mu_b
    z = _z(NOMINAL_COVERAGE)
    sigma, cover = _measured_sigma(resid_b)

    # The ceiling (disclosure, not a comparator): fit the SAME head on the leaky readout at the
    # action step. If the outlet's R2 lands near this, the target is nearly irreducible and the
    # row must be read as "little signal existed", not "the head is weak". Never a feature.
    ceiling = None
    if "post" in rows_a and len(rows_a["post"]):
        Pa = np.hstack([rows_a["post"], np.ones((len(rows_a["post"]), 1))])
        Pb = np.hstack([rows_b["post"], np.ones((len(rows_b["post"]), 1))])
        Pa, Pb, _mu, _sd = _standardize(Pa, Pb)
        p_post = Ridge(random_state=seed).fit(Pa, ya).predict(Pb)
        ceiling = round(float(1.0 - np.sum((yb - p_post) ** 2) / np.sum((yb - yb.mean()) ** 2)), 4)

    # Attribution controls, fitted on the SAME rows and head class: state-blind (delta only) and
    # action-blind (state only). Without them a "response" that only knows the action's main
    # effect — or only the customer's baseline — would still look like signal.
    controls = {}
    for name, which in (("action_blind", "pre"), ("state_blind", "delta")):
        if which == "pre":
            Ca, Cb = Xa[:, : rows_a["pre"].shape[1]], Xb[:, : rows_b["pre"].shape[1]]
        else:
            d0 = rows_a["pre"].shape[1]
            Ca = Xa[:, d0 : d0 + rows_a["delta"].shape[1]]
            Cb = Xb[:, d0 : d0 + rows_b["delta"].shape[1]]
        controls[name] = Ridge(random_state=seed).fit(Ca, ya).predict(Cb)

    # DIAGNOSTIC (not a gate criterion — the frozen three above decide the verdict): how much of
    # the action's transition is INCREMENTAL to the state? `delta` is the recurrence's own step,
    # so most of it is the state relaxing plus the event's input content rather than the action's
    # identity. Projecting delta onto the pre-action state and scoring what is LEFT separates
    # "the head is weak" from "this representation cannot carry the claim". The first real build
    # measured ~0.0 here while the state-blind gate failed at lo=-0.0001, which is what told us
    # the action representation, not the fit, is the bottleneck axis.
    d_pre = rows_a["pre"].shape[1]
    d_del = rows_a["delta"].shape[1]
    Pa_inc = np.hstack([Xa[:, :d_pre], np.ones((len(Xa), 1))])
    Da_inc = Xa[:, d_pre : d_pre + d_del]
    beta_inc = np.linalg.lstsq(Pa_inc, Da_inc, rcond=None)[0]
    Perp_a = np.hstack([Da_inc - Pa_inc @ beta_inc, np.ones((len(Xa), 1))])
    Pb_inc = np.hstack([Xb[:, :d_pre], np.ones((len(Xb), 1))])
    Perp_b = np.hstack([Xb[:, d_pre : d_pre + d_del] - Pb_inc @ beta_inc, np.ones((len(Xb), 1))])
    p_perp = Ridge(random_state=seed).fit(Perp_a, ya).predict(Perp_b)
    ss = float(np.sum((yb - yb.mean()) ** 2))
    r2_perp = 1.0 - float(np.sum((yb - p_perp) ** 2)) / ss if ss > 0 else None

    # effective rank (participation ratio) of the standardized block: 1.0 means one direction
    def _part(C):
        cov = np.cov(C, rowvar=False)
        return float(np.trace(cov) / max(np.linalg.norm(cov, "fro"), 1e-12))

    # DEC-384b: causality attaches only where the arm was logged as randomized, so the
    # UNRANDOMIZED interval must be AT LEAST as wide as the randomized one. Both widths are
    # MEASURED on their own rows; the floor is applied (and receipted) only if the data said
    # otherwise — a forced floor is a disclosure, not a calibration.
    rand = np.asarray(rows_b["randomized"], dtype=np.float64) > 0.5
    groups = {}
    for tag, sel in (("randomized", rand), ("unrandomized", ~rand)):
        if not bool(np.any(sel)):
            continue
        s_g, c_g = _measured_sigma(resid_b[sel])
        groups[tag] = {"sigma": round(s_g, 6), "coverage": round(c_g, 4), "n": int(sel.sum())}
    forced_wider = False
    if (
        "randomized" in groups
        and "unrandomized" in groups
        and groups["unrandomized"]["sigma"] < groups["randomized"]["sigma"]
    ):
        groups["unrandomized"]["sigma"] = groups["randomized"]["sigma"]
        groups["unrandomized"]["coverage"] = round(
            float(np.mean(np.abs(resid_b[~rand]) <= z * groups["randomized"]["sigma"])), 4
        )
        forced_wider = True
    sigma_r = float(groups.get("randomized", {}).get("sigma", sigma))
    sigma_u = float(groups.get("unrandomized", {}).get("sigma", sigma))
    return {
        "kind": "response_head",
        "mean_coef": mean_head.coef_.astype("float32").tolist(),
        "mean_intercept": float(mean_head.intercept_),
        "mu": mu.astype("float32").tolist(),
        "sd": sd.astype("float32").tolist(),
        "z": round(float(z), 4),
        "coverage_nominal": NOMINAL_COVERAGE,
        "coverage_empirical": round(cover, 4),
        "sigma_randomized": round(sigma_r, 6),
        "sigma_unrandomized": round(sigma_u, 6),
        "sigma_forced_wider_unrandomized": forced_wider,
        # the sharpness that was traded for calibration — printed so nobody mistakes a wide
        # interval for a conservative virtue
        "mean_interval_width": round(float(2 * z * (rand * sigma_r + (~rand) * sigma_u).mean()), 6),
        "n_fit": int(len(ya)),
        "n_calibrate": int(len(yb)),
        "target_r2_holdout": round(
            float(1.0 - np.sum((yb - mu_b) ** 2) / np.sum((yb - yb.mean()) ** 2)), 4
        ),
        # LEAKY by construction (the readout at the action step) — a disclosure of how much of
        # this target was predictable at all, never a claim about the served head.
        "target_r2_ceiling_leaky": ceiling,
        # axis attribution for a failed state-blind gate (disclosure, not a criterion): the R2
        # of the target on the part of `delta` that the pre-action state CANNOT explain, and the
        # effective rank of each block. ~0 + rank~1 means the action representation carries no
        # incremental signal, which is a different finding from "the head did not learn".
        "delta_incremental_r2": round(float(r2_perp), 4) if r2_perp is not None else None,
        "delta_effective_rank": round(_part(Da_inc), 4),
        "state_effective_rank": round(_part(Xa[:, :d_pre]), 4),
        "randomized_frac": round(float(np.mean(rand)) if rand.size else 0.0, 4),
        # The fitted population constant: the outlet's own DEC-384 comparator ("better than
        # predicting nothing about this customer"), so the S4 receipt can compute the paired CI
        # from the artifact without refitting anything.
        "target_mean_fit": float(np.mean(ya)),
        "coverage_by_randomization": groups,
        "_pred_holdout": {
            "state_action": mu_b.tolist(),
            "truth": yb.tolist(),
            "action_blind": controls["action_blind"].tolist(),
            "state_blind": controls["state_blind"].tolist(),
        },
    }


def apply_head(rows, head):
    """Score action rows with the published weights (pure numpy — the artifact stays readable
    without sklearn, and a consumer must be able to reproduce the column from the stamp)."""
    import numpy as np

    X = np.hstack([rows["pre"], rows["delta"], np.ones((len(rows["pre"]), 1))])
    X = (X - np.asarray(head["mu"], dtype=np.float64)) / np.asarray(head["sd"], dtype=np.float64)
    mu_hat = X @ np.asarray(head["mean_coef"], dtype=np.float64) + float(head["mean_intercept"])
    if not np.all(np.isfinite(mu_hat)):
        raise RuntimeError("response outlet: the published mean head produced a non-finite value")
    return mu_hat


def publish_rows(rows, head, *, version: str) -> list[dict]:
    """One row per customer x response PART — the shape `view_embeddings` stores.

    The published column is the customer's OWN logged actions aggregated: mean conditional
    outcome, the MEASURED interval half-width blended by that customer's own randomized fraction
    (a customer whose actions were never randomized therefore carries the wider interval —
    DEC-384b as arithmetic rather than as a promise), the support it rests on, and that
    randomized fraction itself.
    """
    import numpy as np

    mu_hat = apply_head(rows, head)
    cust = np.asarray(rows["customer"], dtype=object)
    rand = np.asarray(rows["randomized"], dtype=np.float64)
    s_r, s_u = float(head["sigma_randomized"]), float(head["sigma_unrandomized"])
    out = []
    # argsort once and walk the boundaries: one pass over the rows, not one scan per customer
    order = np.argsort(cust, kind="stable")
    labels, starts = np.unique(cust[order], return_index=True)
    bounds = np.append(starts, len(order))
    for i, label in enumerate(labels):
        sel = order[bounds[i] : bounds[i + 1]]
        f = float(rand[sel].mean())
        sigma = f * s_r + (1.0 - f) * s_u  # the customer's own measured width
        if not np.isfinite(sigma) or sigma <= 0:
            raise RuntimeError(
                f"response outlet: customer {label!r} got sigma {sigma} — refuse to publish an "
                f"interval that is not a positive number"
            )
        values = {
            "outcome_mean": [float(mu_hat[sel].mean())],
            "outcome_sigma": [sigma],
            "support": [float(len(sel))],
            "randomized": [f],
        }
        for part, embedding in values.items():
            out.append(
                {
                    "customer_key": str(label),
                    "as_of_epoch": None,  # filled by the producer: the customer's own last event
                    "version": version,
                    "view": "response",
                    "part": part,
                    "dim": 1,
                    "embedding": embedding,
                }
            )
    return out
