"""The ONE rebuild-from-blob contract: `cfm_state.build_cfm_from_blob`.

The trainer (`cfm_training`) saves the architecture flags beside the weights; a rebuild
that reads a subset constructs a DIFFERENT network, and the `strict=True` load that
validation and the dense-state job depend on can never succeed. That was the live bug:
`cfm_validation` and `state_dense` each carried an inline copy that omitted `unified`,
`isometric_boundary`, the counter/head dims and the DEC-613 trunk keys — on a production
blob (`unified_ssm=True`, `isometric_boundary=True`) their rebuild was a lookalike.

These tests save blobs exactly as `cfm_training` does (the `torch.save` dict), then prove:
  (a) `build_cfm_from_blob` reconstructs the trained architecture,
  (b) the saved weights load with `strict=True`,
  (c) a forward on both models is identical for the same input,
  (d) the looped-trunk keys (`trunk`, `n_loops`, `seq_len`) round-trip,
  (e) a pre-DEC-613 blob (no trunk keys) still rebuilds as the SSM trunk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from looking_glass.cfm_config import _to_epoch
from looking_glass.cfm_looped_trunk import LoopedGPTTrunk
from looking_glass.cfm_model import CFM, EventVocab, MultiScaleSSM
from looking_glass.cfm_state import build_cfm_from_blob

DIM = 16
SEQ_LEN = 32


def _seq(key="c1", n=8):
    """The adapter's sequence shape (`build_sequences`), small enough for a unit test."""
    ets = ["view", "view", "order", "view", "order", "view", "view", "order"][:n]
    stamps = [f"2025-01-01T{10 + i:02d}:00:00+00:00" for i in range(n)]
    return {
        "customer": key,
        "group": "A",
        "anchor_epoch": None,
        "event_type": np.array(ets, dtype=object),
        "brand": np.array([None] * n, dtype=object),
        "entity_type": np.array(["customer"] * n, dtype=object),
        "entity_id": np.array([key] * n, dtype=object),
        "value": np.ones(n, dtype=np.float32),
        "event_ts": np.array(stamps, dtype=object),
        "ts": [_to_epoch(s) for s in stamps],
        "co": [[0.0, 0.0]] * n,
        "attrs": {"channel": np.array(["web"] * n, dtype=object)},
        "numeric": {"amount": np.arange(n, dtype=np.float64)},
        "numeric_mask": {"amount": np.ones(n, dtype=bool)},
    }


def _trainer_blob(model, seq_len=SEQ_LEN):
    """The dict `cfm_training`'s save writes — key for key (the rebuild contract)."""
    return {
        "state": model.state_dict(),
        "vocab": model.vocab.dumps(),
        "dim": model.dim,
        "n_experts": model.n_experts,
        "sf_mode": model.sf_mode,
        "slow_intent": False,
        "unified": model.unified,
        "zca": model.zca,
        "isometric_boundary": model.isometric_boundary,
        "bilinear_recurrence": model.bilinear,
        "looped_recurrence": model.looped,
        "drive_looped_recurrence": model.drive_looped,
        # DEC-613: trunk identity.
        "trunk": model.trunk,
        "n_loops": model.n_loops,
        "seq_len": int(seq_len),
        "readout_norm": model.readout_norm_on,
        "input_norm": model.input_norm_on,
        "readout_skip": model.readout_skip_on,
        "event_residual": model.event_residual_on,
        "readout_scale": model.readout_scale_on,
        "stats_pathway": model.stats_pathway_on,
        "continuous_time": model.continuous_time_on,
        "counter": model.counter_on,
        "counter_registers": model.counter_registers,
        "counter_compact": model.counter_compact,
        "counter_event_types": model.counter_event_types,
        "counter_type_values": model.counter_type_values,
        "schema_summary": model.schema_summary,
        "numeric_fields": list(model.numeric_fields),
        "attr_fields": list(model.attr_fields),
        "numeric_stats": {f: dict(v) for f, v in model.numeric_stats.items()},
        "field_receipt": {},
        "action_receipt": {},
        "effect_dim": model.effect_dim,
        "effect_actions": [],
        "effect_head": model.effect_head,
        "survive_dim": model.survive_dim,
        "rate_dim": model.rate_dim,
        "action_profile_dim": model.profile_dim,
        "action_profile_fields": [],
        "profile_served": False,
        "profile_loss": "mse",
        "profile_state": "y",
        "profile_weight": 1.0,
        "serve_unfaded": False,
        "ledger_recency": False,
        "action_fields": list(model.action_fields),
        "primary_numeric": model.primary_numeric,
        "whiten_mean": (model.whiten_mean if getattr(model, "whiten_on", False) else None),
        "whiten_W": (model.whiten_W if getattr(model, "whiten_on", False) else None),
    }


def _save_blob(path, model, seq_len=SEQ_LEN):
    """Persist a blob exactly as the trainer does, returning it for inspection."""
    blob = _trainer_blob(model, seq_len=seq_len)
    torch.save(blob, path)
    return blob


def _production_like_model(seed=0):
    """Small dim, SSM trunk, production architecture flags ON.

    `unified` + `isometric_boundary` are the two the old inline copies omitted, so this
    is the exact combination whose `strict=True` load they could never satisfy.
    """
    torch.manual_seed(seed)
    vocab = EventVocab.build([_seq()])
    return CFM(
        vocab,
        DIM,
        n_experts=1,
        unified=True,
        isometric_boundary=True,
        readout_norm=True,
        input_norm=True,
        continuous_time=True,
        schema_summary=True,
        numeric_fields=("amount",),
    )


def test_builder_reconstructs_trained_arch_and_strict_load_succeeds(tmp_path):
    model = _production_like_model()
    blob = _save_blob(tmp_path / "cfm_t.pt", model)

    rebuilt = build_cfm_from_blob(blob, tag="t")

    # (a) it is the trained architecture, not a lookalike
    assert rebuilt.unified is True
    assert rebuilt.isometric_boundary is True
    assert rebuilt.trunk == "ssm"
    assert rebuilt.numeric_fields == ("amount",)
    assert set(rebuilt.state_dict()) == set(blob["state"])
    # (b) the saved weights load with strict=True (the check the old copies failed) —
    # both directly and through the builder's verification mode.
    rebuilt.load_state_dict(blob["state"], strict=True)
    rebuild_checked = build_cfm_from_blob(blob, tag="t", strict=True)
    assert rebuild_checked.trunk == "ssm"


def test_rebuild_without_the_architecture_flags_cannot_strict_load():
    """The bug this builder prevents, pinned: the old copies rebuilt without `unified` /
    `isometric_boundary`, so a production blob could never strict-load into them."""
    model = _production_like_model()
    blob = _trainer_blob(model)
    lookalike = CFM(model.vocab, DIM, n_experts=1)  # the old inline construction
    with pytest.raises(RuntimeError):
        lookalike.load_state_dict(blob["state"], strict=True)


def test_rebuilt_forward_is_identical_to_the_trained_model(tmp_path):
    model = _production_like_model()
    blob = _save_blob(tmp_path / "cfm_t.pt", model)
    rebuilt = build_cfm_from_blob(blob, tag="t", strict=True)
    model.eval()
    seq = _seq()
    with torch.no_grad():
        # (c) same input -> identical output, on the state and on the consumed readout
        y0, h0 = model(seq)
        y1, h1 = rebuilt(seq)
        e0 = model.donor_batch(h0.unsqueeze(0))[0]
        e1 = rebuilt.donor_batch(h1.unsqueeze(0))[0]
    assert torch.equal(y0, y1)
    assert torch.equal(h0, h1)
    assert torch.equal(e0, e1)


def test_looped_trunk_keys_round_trip(tmp_path):
    """(d) DEC-613: `trunk` / `n_loops` / `seq_len` survive the save -> rebuild path."""
    torch.manual_seed(0)
    vocab = EventVocab.build([_seq()])
    model = CFM(
        vocab,
        DIM,
        n_experts=1,
        trunk="looped_gpt",
        n_loops=3,
        readout_norm=True,
        input_norm=True,
        max_pos=SEQ_LEN,
    )
    blob = _save_blob(tmp_path / "cfm_lg.pt", model)
    assert blob["trunk"] == "looped_gpt" and blob["n_loops"] == 3

    rebuilt = build_cfm_from_blob(blob, tag="lg", strict=True)
    assert rebuilt.trunk == "looped_gpt"
    assert rebuilt.n_loops == 3
    assert isinstance(rebuilt.ssm, LoopedGPTTrunk)
    assert rebuilt.ssm.max_pos == SEQ_LEN  # seq_len is the trunk's position budget
    model.eval()
    seq = _seq()
    with torch.no_grad():
        y0, h0 = model(seq)
        y1, h1 = rebuilt(seq)
    assert torch.equal(y0, y1)
    assert torch.equal(h0, h1)


def test_old_blob_without_trunk_keys_rebuilds_as_ssm(tmp_path):
    """(e) Back-compat: pre-DEC-613 blobs carry no trunk keys -> the SSM trunk.

    Also covers the whitened boundary restore (`whiten_W` in the blob, not the
    state_dict), which the forward-only comparison would never exercise.
    """
    torch.manual_seed(0)
    vocab = EventVocab.build([_seq()])
    model = CFM(vocab, DIM, n_experts=1)  # pre-DEC-613 flag set: legacy SSM, no trunk
    model.set_whitening(torch.linspace(-1.0, 1.0, DIM), torch.eye(DIM))
    blob = _save_blob(tmp_path / "cfm_old.pt", model)
    for key in ("trunk", "n_loops", "seq_len"):
        blob.pop(key)

    rebuilt = build_cfm_from_blob(blob, tag="old", strict=True)
    assert rebuilt.trunk == "ssm"
    assert rebuilt.n_loops == 2  # the constructor's documented default
    assert isinstance(rebuilt.ssm, MultiScaleSSM)
    assert rebuilt.whiten_on is True
    assert torch.equal(rebuilt.whiten_W, model.whiten_W)
    model.eval()
    seq = _seq()
    with torch.no_grad():
        y0, h0 = model(seq)
        y1, h1 = rebuilt(seq)
        # the whitened donor boundary is the served readout: same state, same E
        e0 = model.donor_batch(h0.unsqueeze(0))[0]
        e1 = rebuilt.donor_batch(h1.unsqueeze(0))[0]
    assert torch.equal(y0, y1)
    assert torch.equal(h0, h1)
    assert torch.equal(e0, e1)
