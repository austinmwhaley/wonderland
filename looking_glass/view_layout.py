"""The published outlet schema for the production donor (SPINE §5, DEC-388 / DEC-397).

SPINE §5 promises a frozen artifact that exposes **named views** so a simple consumer can
select what it needs (the Outlet Standard O1: "the socket is a standard, not a bespoke
harness"). Until now the production producer published exactly ONE fused vector
(`state_embeddings.embedding`), and the only code describing named views
(`state_layout.py`) was wired to the deleted demo `Enc` stack — it reads
`model.ssm[0].e`, which `CFM` does not have. So the outlets were declared, not built, and
stage S4 could only render UNKNOWN.

This module is the single source of truth for what the production encoder actually
publishes. Three rules it enforces:

* **Derived, never typed** — dimensions come from running the head, not from a number in
  this file. A head that changes width changes the receipt automatically (Principle 1).
* **Every part is named and addressable.** A view is a concat of named parts, so a card
  row can attribute a score to a part and the conformance kit can zero/permute exactly
  one of them (DEC-405 rule 6's knob-turn receipt).
* **Absent is declared, not silently missing.** `response` (DEC-384) and `measure`
  (DEC-383) are approved outlets this producer does not emit yet. They stay in
  `DECLARED`, so the card renders them UNKNOWN and the faceplate advertises them as
  uncertified — the alternative (omitting them) is how a gap becomes a narrative.

The content classes here are the ones SPINE §5 lists; `full` is a bundle, not content, and
is never stored — it is the documented concatenation of `FULL_ORDER`.
"""

from __future__ import annotations

# Permanent member order of the `full` bundle (SPINE §5 contract: the ledger is excluded
# by definition, so "full + measure" is a separately-named arm). Changing this is a
# contract change and must be recorded in DECISIONS.md.
FULL_ORDER = ("shape", "learned_state", "forecast", "uncertainty", "heads")

#: Every outlet the project has approved. `response`/`measure` are declared-but-not-emitted
#: by this producer; `DECLARED` is what the faceplate and the card must account for.
DECLARED = ("shape", "learned_state", "forecast", "uncertainty", "heads", "response", "measure")

#: What each approved outlet is FOR (SPINE §1 consumer trace). The card prints this; it is
#: not a place to assert a capability.
PURPOSE = {
    "shape": "behavioral profile — who is this customer like (distance / all supervised)",
    "learned_state": "what the recurrence held: per-timescale contents + learned mass",
    "forecast": "learned past/future summaries (self-supervised summary heads)",
    "uncertainty": "what I actually saw: timing/support signal for abstaining heads",
    "heads": "the network's own predictions (provisional bucket, DEC-397 earned-keep)",
    "response": "action-conditioned outcome with uncertainty (RL + simulator, DEC-384)",
    "measure": "exact lawful tallies computed outside the network (ledger, DEC-383)",
}


def _donor(m, y, h):
    """The public readout, batched or not, from ONE definition.

    `CFM.donor` is documented for a single state and `CFM.donor_batch` for (B, D) batches;
    picking by rank here keeps the published `shape` column a single concept. Their
    agreement is asserted in the tests, because the two paths drifting apart is precisely
    the train/serve skew this repo has already paid for twice.
    """
    return m.donor_batch(h) if h.dim() == 2 else m.donor(h)


#: view -> {part_name: fn(model, y, h) -> Tensor | None}, IN PUBLICATION ORDER.
#: This is the outlet contract: the order here is the order the columns are concatenated
#: in, and it is recorded on the faceplate so a consumer (or the S4 receipt) rebuilds the
#: exact same vector instead of guessing one. A part whose head does not exist on this
#: model returns None, is dropped, and says so — the receipt records the absence rather
#: than the code pretending the content is there.
_PARTS = {
    "shape": (("donor", _donor),),
    "learned_state": (
        ("recurrence", lambda m, y, h: h),
        ("ledger_totals", lambda m, y, h: m.head_ledger(y)),
        ("rfm_recon", lambda m, y, h: m.head_recon(y)),
    ),
    "forecast": (
        ("future_state", lambda m, y, h: m.head_foresight(y)),
        ("psr", lambda m, y, h: m.head_psr(y)),
        ("recency_level", lambda m, y, h: m.head_recency(y)),
    ),
    "uncertainty": (
        ("next_time", lambda m, y, h: m.head_dt(y)),
        ("occurrence", lambda m, y, h: m.head_occ(y)),
    ),
    "heads": (("next_event", lambda m, y, h: m.head_next(y)),),
}

#: {view: [part names in publication order]} — the contract, readable without a model.
PART_ORDER = {view: [name for name, _ in spec] for view, spec in _PARTS.items()}


def _parts_for(model):
    return _PARTS


def parts(model, y, h) -> dict:
    """{view: {part_name: 1-D tensor}} — the attribution-level view of the outlet.

    Inputs may be a single state (`[dim]`) or a batch (`[B, dim]`); the shape is preserved
    per part so the caller can write rows per customer without re-running the encoder.
    """
    if y.dim() == 2 and h.dim() == 1:
        # `CFM.forward` returns the PER-STEP readout y=[T,dim] and the FINAL state h=[dim].
        # Passing those here is the bug this guard exists to catch: every head would emit
        # one row per event, so a published column's width would silently depend on the
        # customer's sequence length. The outlet is the readout AT the last event.
        raise ValueError(
            f"parts() got y=[{y.shape[0]},{y.shape[1]}] with h=[{h.shape[0]}] — that is the "
            "per-step readout against a final state. Pass the last tick and a batch: "
            "y[-1].unsqueeze(0), h.unsqueeze(0)"
        )
    out = {}
    for view, spec in _parts_for(model).items():
        got = {}
        for name, fn in spec:
            try:
                t = fn(model, y, h)
            except Exception as exc:  # an absent head is a fact about the artifact, not a crash
                print(f"[view-layout] {view}.{name} unavailable: {type(exc).__name__}: {exc}")
                continue
            if t is None:
                continue
            got[name] = t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(1, -1)
        if got:
            out[view] = got
    return out


def emit(model, y, h) -> dict:
    """{view: tensor} — the published vectors (concat of the view's parts, in part order).

    Part order is fixed by `_parts_for`, so a column's meaning does not drift between
    releases; adding a part is an additive change (DEC-412's compatibility contract).
    """
    import torch

    return {
        view: torch.cat([p for p in group.values()], dim=-1)
        for view, group in parts(model, y, h).items()
    }


def describe(model) -> dict:
    """{view: dim} plus the part breakdown, DERIVED by running the heads on a zero state.

    Zeroes are used rather than real data because only the shapes are wanted: this must not
    depend on having a stream open, and a zero input cannot make a Linear/GELU head return
    a different width.
    """
    import torch

    with torch.no_grad():
        h = torch.zeros(1, int(model.dim), device=model._dev())
        y = torch.zeros(1, int(model.dim), device=model._dev())
        group = parts(model, y, h)
    return {
        "views": {v: int(t.shape[-1]) for v, t in _concat(group).items()},
        "parts": {v: {k: int(t.shape[-1]) for k, t in g.items()} for v, g in group.items()},
        "emitted": sorted(group),
        "declared_not_emitted": [v for v in DECLARED if v not in group],
        "full_order": list(FULL_ORDER),
        "full_dim": int(
            sum(int(t.shape[-1]) for v, t in _concat(group).items() if v in FULL_ORDER)
        ),
    }


def _concat(group: dict) -> dict:
    import torch

    return {v: torch.cat([p for p in g.values()], dim=-1) for v, g in group.items()}


def faceplate(model, cfg) -> dict:
    """The self-describing stamp a consumer validates against (Outlet Standard O4).

    Fan-out to thousands of uncoordinated consumers is why this exists: each one
    self-checks the columns it was told to read and refuses locally, rather than asking the
    encoder a question it cannot answer (DEC-405 rule 4).
    """
    d = describe(model)
    return {
        "kind": "view_faceplate",
        "version": str(cfg.tag),
        "encoder_dim": int(model.dim),
        **d,
        # An outlet that is approved but not published here is advertised, not hidden: a
        # consumer that selects it gets a refusal with a reason, never a zero-filled column.
        "not_served": {
            v: "declared (SPINE §5) but not emitted by this producer"
            for v in d["declared_not_emitted"]
        },
    }
