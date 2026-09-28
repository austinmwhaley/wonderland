"""looking_glass — Layer B: the frozen customer-foundation donor.

The ONLY supported surface is the CFM production stack; import it by submodule:

    cfm_config   — config + the ONE day-boundary helper (``as_of_epoch``)
    cfm_data     — point-in-time stream cuts, splits, samples
    cfm_model    — ``CFM`` encoder (selective multi-scale SSM) + vocab
    cfm_state    — ``StateStore`` (only writer of state/embeddings) + products
    cfm_training — governed training (governor, warm-start, ladder receipts)
    daily_states — the daily state job (days 2..N)
    customer_foundation_model — CLI entry (train / products / rebuild)
    sufficiency_battery, layer_b_proof, state_dense, autotune — measurement

Plugins consume the frozen tables (``donor_embeddings`` / ``state_embeddings``),
never model internals. There is no other public API — the former
factory-function stack (create_embedding_model / create_temporal_core_model /
create_supervised_model) was deleted as dead code.
"""

from __future__ import annotations

__all__: list[str] = []
