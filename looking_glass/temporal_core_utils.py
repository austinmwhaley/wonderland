"""Temporal core tensor helpers: causal masks, normalization, and batch sizing."""

from __future__ import annotations

import logging
from datetime import datetime

import torch
from torch import Tensor

# Logger name preserved from the original temporal_core module.
logger = logging.getLogger("looking_glass.temporal_core")


def _build_causal_attention_mask(valid_mask: Tensor) -> Tensor:
    """Build a boolean mask that enforces both padding and causal constraints."""

    seq_len = valid_mask.size(1)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=valid_mask.device))
    query_mask = valid_mask[:, None, :, None]
    key_mask = valid_mask[:, None, None, :]
    return query_mask & key_mask & causal.unsqueeze(0).unsqueeze(0)


def _normalize_masked(values: Tensor, valid_mask: Tensor) -> Tensor:
    """Normalize feature tensor using only valid (non-padding) positions."""

    if values.size(-1) == 0:
        return values

    flat_values = values[valid_mask]
    if flat_values.numel() == 0:
        return values

    means = flat_values.mean(dim=0, keepdim=True)
    stds = flat_values.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    normalized = values.clone()
    normalized[valid_mask] = (flat_values - means) / stds
    normalized[~valid_mask] = 0.0
    return normalized


def _parse_iso_timestamp(value: object) -> datetime:
    """Parse an ISO timestamp and normalize trailing ``Z`` into UTC offset."""

    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _resolve_train_batch_size(
    requested: int,
    total_sequences: int,
    max_seq_len: int,
    device: torch.device,
    progress_label: str,
) -> int:
    """Resolve a safe train batch size, especially for CUDA attention memory.

    Attention memory grows approximately with ``seq_len^2``; this helper caps
    batch sizes on CUDA for longer sequences to reduce OOM risk.
    """

    batch = max(1, min(int(requested), max(1, total_sequences)))
    if device.type != "cuda":
        return batch

    # Attention memory scales roughly with sequence length squared.
    if max_seq_len >= 128:
        safe_cap = 64
    elif max_seq_len >= 96:
        safe_cap = 128
    elif max_seq_len >= 64:
        safe_cap = 192
    else:
        safe_cap = 256

    resolved = min(batch, safe_cap)
    if resolved < batch:
        logger.info(
            "%s: reducing train_batch_size from %d to %d for seq_len=%d on CUDA",
            progress_label,
            batch,
            resolved,
            max_seq_len,
        )
    return resolved
