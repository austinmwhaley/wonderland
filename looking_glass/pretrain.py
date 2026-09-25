"""State management utilities for the batch-training cadence.

Once the Mamba core is pretrained, customer hidden states evolve with new
events without weight changes.  This module provides the three primitives
that make the operational cadence operable:

1.  **State extraction** — ``extract_final_states`` runs the core on a
    padded batch and returns the final valid hidden state per sequence.
    Used to snapshot customer representations after full-history training.

2.  **Incremental forward** — ``incremental_forward`` prepends a saved
    hidden state as a seed token before a window of new events, then
    forward-passes only the delta to obtain an updated state.  This is the
    daily-inference operation: weight-stable, deterministic, low-latency.

3.  **LoRA retrofitting** — ``wrap_core_with_qdora`` walks the frozen
    core's ``FeedForward`` layers and swaps each ``nn.Linear`` for a
    ``QDoRALinear`` initialised from the pretrained weights.  This is
    called exactly once after pretraining; downstream tasks then deep-copy
    the retrofitted core and train only the fresh LoRA adapters.

4.  **Persistence** — ``save_history_to_store`` extracts final states,
    wraps them as ``StateRecord`` s stamped with ``backbone_version``, and
    writes them to a ``PointInTimeStateStore`` for as-of queries.
"""

from __future__ import annotations

from datetime import datetime

import torch
from torch import Tensor

from .state_store import PointInTimeStateStore, StateRecord


def _parse_ts(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def extract_final_states(
    core,
    hidden_states: Tensor,
    delta_t: Tensor,
    valid_mask: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Run the core on a padded batch and return the final valid state per sequence.

    Args:
        core: :class:`EntityCore` (or any module with the same forward contract).
        hidden_states: ``(batch, seq_len, hidden_dim)`` padded input.
        delta_t: ``(batch, seq_len, 1)`` inter-event deltas.
        valid_mask: ``(batch, seq_len)`` boolean mask, True for real positions.
        attention_mask: Optional causal/padding mask passed through.

    Returns:
        ``(batch, hidden_dim)`` — the hidden state at the last valid position
        of each sequence.
    """

    encoded = core(hidden_states, delta_t, attention_mask=attention_mask).encoded_states
    lengths = valid_mask.sum(dim=1).long().clamp(min=1)
    return encoded[torch.arange(encoded.size(0), device=encoded.device), lengths - 1]


def incremental_forward(
    core,
    seed_states: Tensor,
    new_hidden_states: Tensor,
    new_delta_t: Tensor,
    new_valid_mask: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Forward-pass a window of new events seeded with a saved prior state.

    The seed is prepended as position 0 with delta_t=0. The core sees
    ``[seed, ev_1, ev_2, ...]`` and returns the state at the last valid
    position (which may be the seed only when there are no new events).

    Args:
        core: :class:`EntityCore`.
        seed_states: ``(batch, hidden_dim)`` — saved $h_{t-1}$ per sequence.
        new_hidden_states: ``(batch, new_seq_len, hidden_dim)``.
        new_delta_t: ``(batch, new_seq_len, 1)``.
        new_valid_mask: ``(batch, new_seq_len)`` boolean.
        attention_mask: Optional.

    Returns:
        ``(batch, hidden_dim)`` — $h_t$ after processing the seed + new events.
    """

    batch, _, dim = new_hidden_states.shape
    device = new_hidden_states.device

    seed = seed_states.unsqueeze(1)  # (batch, 1, hidden_dim)
    seed_dt = torch.zeros(batch, 1, 1, dtype=new_hidden_states.dtype, device=device)
    seed_mask = torch.ones(batch, 1, dtype=torch.bool, device=device)

    combined_states = torch.cat([seed, new_hidden_states], dim=1)
    combined_dt = torch.cat([seed_dt, new_delta_t], dim=1)
    combined_mask = torch.cat([seed_mask, new_valid_mask], dim=1)

    if attention_mask is not None:
        extra = attention_mask.size(-1) - new_valid_mask.size(1)
        if extra > 0:
            pad = torch.zeros(batch, 1, extra, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([pad, attention_mask], dim=2)
        prefix = torch.ones(
            batch, 1, attention_mask.size(-1), dtype=attention_mask.dtype, device=device
        )
        attention_mask = torch.cat([prefix, attention_mask], dim=1)

    return extract_final_states(core, combined_states, combined_dt, combined_mask, attention_mask)


def save_history_to_store(
    core,
    hidden_states: Tensor,
    delta_t: Tensor,
    valid_mask: Tensor,
    entity_ids: list[str],
    as_of_ts_list: list[datetime],
    store: PointInTimeStateStore,
    backbone_version: str = "v0",
    attention_mask: Tensor | None = None,
) -> int:
    """Extract final states per entity and persist them to the store."""

    vectors = (
        extract_final_states(core, hidden_states, delta_t, valid_mask, attention_mask)
        .cpu()
        .tolist()
    )
    records = [
        StateRecord(
            entity_id=eid,
            as_of_ts=ts,
            vector=vec,
            backbone_version=backbone_version,
        )
        for eid, ts, vec in zip(entity_ids, as_of_ts_list, vectors)
    ]
    return store.write_states(records)


def wrap_core_with_qdora(core, qdora_config) -> None:
    """Retrofit every :class:`FeedForward` inside *core* with QDoRA adapters.

    Walks ``core.sequence_engine.shared_block`` (both MambaBlock and
    SambaBlock have ``self.ff`` of type ``FeedForward``) and calls
    :meth:`FeedForward.upgrade_to_qdora` on each.  This is a one-way
    mutation in place; the original linear weights become the frozen base
    and fresh low-rank adapters are appended.
    """

    engine = core.sequence_engine
    block = engine.shared_block
    if hasattr(block, "ff"):
        block.ff.upgrade_to_qdora(qdora_config)
    if hasattr(block, "shared_block") and hasattr(block.shared_block, "ff"):
        block.shared_block.ff.upgrade_to_qdora(qdora_config)


def incremental_state_update(
    core,
    store: PointInTimeStateStore,
    entity_ids: list[str],
    new_hidden_states: Tensor,
    new_delta_t: Tensor,
    new_valid_mask: Tensor,
    seed_as_of: datetime,
    as_of_ts_list: list[datetime],
    backbone_version: str = "v0",
    attention_mask: Tensor | None = None,
) -> int:
    """Daily-cadence primitive: load prior states, roll forward, persist.

    For each entity in *entity_ids* the function queries the store for the
    latest state at or before *seed_as_of* (returned as a zero vector when
    no prior state exists — cold start), then calls
    :func:`incremental_forward` to process the new event window
    (*new_hidden_states*, *new_delta_t*, *new_valid_mask*) seeded with
    that prior state.  The updated per-entity state is written back to the
    store stamped with a new ``as_of_ts`` from *as_of_ts_list*.

    This is the core operation for a daily (or hourly) update: the
    pretrained backbone is frozen, the prior state is a lookup, and the
    forward pass is weight-stable and deterministic.

    Args:
        core: :class:`EntityCore` (pretrained, frosted or not).
        store: A :class:`PointInTimeStateStore` for read and write.
        entity_ids: Per-row entity identifiers, one per batch position.
        new_hidden_states: ``(batch, new_seq_len, hidden_dim)`` tokenized
            new events.
        new_delta_t: ``(batch, new_seq_len, 1)``.
        new_valid_mask: ``(batch, new_seq_len)`` boolean.
        seed_as_of: Query the store with this datetime for prior states.
        as_of_ts_list: One ``datetime`` per entity — the new cut-off stored
            with the updated state.
        backbone_version: Version key for store read and write.
        attention_mask: Optional causal/padding mask (see
            :func:`incremental_forward`).

    Returns:
        Number of state records written (equals ``len(entity_ids)``).
    """

    dim = new_hidden_states.size(-1)
    device = new_hidden_states.device

    prior = store.get_states_as_of(
        list(entity_ids),
        seed_as_of,
        backbone_version=backbone_version,
    )
    seed = torch.stack(
        [
            torch.tensor(prior[eid], dtype=new_hidden_states.dtype, device=device)
            if prior.get(eid) is not None
            else torch.zeros(dim, dtype=new_hidden_states.dtype, device=device)
            for eid in entity_ids
        ],
        dim=0,
    )

    updated = incremental_forward(
        core,
        seed,
        new_hidden_states,
        new_delta_t,
        new_valid_mask,
        attention_mask,
    )

    records = [
        StateRecord(
            entity_id=eid,
            as_of_ts=ts,
            vector=vec.cpu().tolist(),
            backbone_version=backbone_version,
        )
        for eid, ts, vec in zip(entity_ids, as_of_ts_list, updated)
    ]
    return store.write_states(records)
