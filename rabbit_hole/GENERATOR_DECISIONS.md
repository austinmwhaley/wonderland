# rabbit_hole — Synthetic Stream Generator Decisions (review log)

Layer-A design decisions for the synthetic event stream, recorded so they can be
reviewed, added, or removed. Status values: **accepted (operator)** = decided by
the operator; **proposed** = made by the agent, open to review; **superseded**.

Scope note (layering): a *logged treatment* (what the system did) is faithful raw
data and may live in the stream. *Labels, rewards, and ground-truth counterfactuals*
must **not** live in the canonical stream; they stay in separate ground-truth
artifacts.

| ID | Decision | Status |
|---|---|---|
| GD-001 | Remove email open events — unobservable/unreliable under Apple MPP. | accepted (operator) |
| GD-002 | Keep SMS/push opens (MPP is email-specific). | proposed |
| GD-003 | No logged propensity. All RL is offline/observational (`white_queen`); marketing owns the policy, so propensity is estimated, never logged. | accepted (operator) |
| GD-004 | Model the treatment (per-channel cadence band + discount) as the observed marketing *policy* — confounded by latent intent, not randomized — and surface the action in the stream. | proposed |
| GD-005 | Keep a small randomized **holdout** (no-send) **for ground-truth validation only** (never as a required propensity). | proposed |
| GD-006 | Surface the action into the canonical stream: `{channel}_send` carries `campaign_id`, `arm` (cadence band), `discount_pct`; outcomes (`open`/`click`) are separate events. | proposed |
| GD-007 | Ground truth (`optimal_arm`, response surface, true incremental effects) lives in **separate artifacts**, never in the canonical stream. | proposed |
| GD-008 | Replace the fixed "CONV ladder" with a **measured, receipted** dose–response (the documented `0.01/0.03/0.06/0.10` does not reproduce). | proposed |
| GD-009 | Enforce the observation-window bound and regenerate the standing artifact once, with a build receipt + git SHA. | proposed |
| GD-010 | Acceptance: assert no email opens; action attributes present; no propensity anywhere stream-facing; dose–response receipted; keep window/determinism checks. | proposed |

## Rationale and alternatives

- **GD-001 (email opens out).** Apple MPP proxies/prefetches opens, so the event is
  noise. *Alternative:* keep opens with an `unreliable` flag — rejected: invites
  their misuse. Clicks (deliberate) and conversions remain the honest email signal.
- **GD-002 (SMS/push opens in).** No MPP equivalent. *Alternative:* drop all opens
  for uniformity — open to review; left in because they are genuinely observed.
- **GD-003 (no propensity).** The email program is owned by marketing; we observe it,
  we do not assign it. `white_queen` already supports observational OPE via
  `estimate_propensity=True` + sensitivity bounds. *Alternative:* keep logged
  propensity as a validation luxury — rejected: it would let the architecture depend
  on something production will never have.
- **GD-004 (policy, not trial).** Treatment assignment already depends on activity +
  latent intent (confounded). Keep that; do not present it as randomized. *Alternative:*
  fully deterministic policy — rejected: no overlap/positivity at all, so OPE is
  undefined even with sensitivity bounds.
- **GD-005 (holdout for validation).** AGENTS permits "a small randomized holdout" as
  the ground-truth source for counterfactual validation. It is a *measurement* control,
  not a propensity we log per user. *Alternative:* no holdout — rejected: removes the
  only ground-truth anchor for incrementality.
- **GD-006 (action in the stream).** Downstream RL needs to see what action was taken.
  Today `arm`/`discount` sit in side tables and the send event exposes outcomes
  (`opened`/`clicked`) instead. Swap: send = action; click = outcome. *Alternative:*
  leave as-is — rejected: makes the "embedding powers RL" objective untestable.
- **GD-007 (GT out of the stream).** Keeps Layer A faithful (no labels/rewards).
  GT tables are the fixture `red_king`/`white_queen` validate against.
- **GD-008 (dose–response).** The response is per-customer unimodal
  (`p_conv = peak·exp(1 − x/x0)·exp(0.5·intent)·season·(1+0.04·disc)`); the old
  increasing ladder is not in code and does not reproduce. Pin the *measured* response
  in a receipt.
- **GD-009 (window bound).** Code already right-censors post-window events; the standing
  artifact predates the fix. One regeneration + receipt makes the artifact match the code.
- **GD-010 (acceptance).** Turn each decision into a checked invariant.

## Open questions for review

1. GD-002: drop **all** opens, or keep SMS/push?
2. GD-005: is a randomized holdout acceptable in the synthetic fixture even though
   production has none?
3. GD-006: is `arm` (cadence band) the treatment we want surfaced, or a coarser
   frequency bucket?
