from datetime import datetime, timezone

import torch

from looking_glass import (
    EntityCore,
    InMemoryStateStore,
    SequenceEngine,
    StateRecord,
    TemporalStack,
    incremental_state_update,
)
from looking_glass.interfaces import TaskHeadBase


HIDDEN = 16
T1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
T3 = datetime(2024, 1, 3, tzinfo=timezone.utc)


class _NoopHead(TaskHeadBase):
    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}


def _core():
    return EntityCore(
        temporal_encoder=TemporalStack(hidden_dim=HIDDEN),
        sequence_engine=SequenceEngine(
            hidden_dim=HIDDEN, recurrent_steps=1, num_heads=4, backend="mamba2"
        ),
        task_head=_NoopHead(),
    )


def test_incremental_state_update():
    core = _core()
    store = InMemoryStateStore(default_version="v1")

    # Write initial states for two entities at T1.
    store.write_states(
        [
            StateRecord("a", T1, [1.0] * HIDDEN, "v1"),
            StateRecord("b", T1, [5.0] * HIDDEN, "v1"),
        ]
    )

    # New event window: one event per entity with dummy hidden states.
    new_hidden = torch.randn(2, 1, HIDDEN)
    new_dt = torch.full((2, 1, 1), 1.0)
    new_mask = torch.ones(2, 1, dtype=torch.bool)

    written = incremental_state_update(
        core,
        store,
        entity_ids=["a", "b"],
        new_hidden_states=new_hidden,
        new_delta_t=new_dt,
        new_valid_mask=new_mask,
        seed_as_of=T2,
        as_of_ts_list=[T2, T2],
        backbone_version="v1",
    )
    assert written == 2

    # After the update, querying at T2 should return the new state.
    a_state = store.get_state_as_of("a", T2, backbone_version="v1")
    assert a_state is not None
    assert len(a_state) == HIDDEN
    assert a_state != [1.0] * HIDDEN  # state was rolled forward

    # Querying at T1 should still return the old state.
    a_old = store.get_state_as_of("a", T1, backbone_version="v1")
    assert a_old == [1.0] * HIDDEN


def test_cold_start_entity_gets_zero_seed():
    core = _core()
    store = InMemoryStateStore(default_version="v1")

    new_hidden = torch.randn(1, 1, HIDDEN)
    new_dt = torch.full((1, 1, 1), 1.0)
    new_mask = torch.ones(1, 1, dtype=torch.bool)

    incremental_state_update(
        core,
        store,
        entity_ids=["new_entity"],
        new_hidden_states=new_hidden,
        new_delta_t=new_dt,
        new_valid_mask=new_mask,
        seed_as_of=T1,
        as_of_ts_list=[T1],
        backbone_version="v1",
    )
    state = store.get_state_as_of("new_entity", T2, backbone_version="v1")
    assert state is not None
    assert any(v != 0.0 for v in state)  # not all-zero anymore
