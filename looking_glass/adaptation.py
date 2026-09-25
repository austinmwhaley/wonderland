"""Adaptation modules: QDoRA and adapter fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import bitsandbytes as bnb  # type: ignore

    _HAS_BNB = True
except ImportError:
    bnb = None  # type: ignore
    _HAS_BNB = False


@dataclass(frozen=True)
class QDoRAConfig:
    """Configuration for QDoRA adaptation."""

    rank: int = 8
    alpha: float = 16.0
    quantize_base: bool = True
    dropout: float = 0.05


class QDoRALinear(nn.Module):
    """Quantized Weight-Decomposed Adaptation linear layer.

    Forward semantics::

            out = base(x) + m * normalize(lora_b(lora_a(dropout(x)))) * (alpha / rank)

    The base weight is frozen (optionally 4-bit quantized) and never
    normalized; the DoRA-style decomposition is applied to the *adaptation
    update*: its direction is normalized per sample and rescaled by a
    trainable per-output-dimension magnitude ``m``.  At initialization
    ``lora_b`` is zero, so the layer starts exactly at the base behaviour.

    Input:
            x: Tensor (..., in_features)

    Output:
            Tensor (..., out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: QDoRAConfig,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if config.rank < 1:
            raise ValueError("QDoRA rank must be >= 1")

        self.alpha = config.alpha
        self.rank = config.rank
        self.scaling = config.alpha / float(config.rank)
        self.dropout = nn.Dropout(config.dropout)

        if config.quantize_base and _HAS_BNB:
            self.base = bnb.nn.Linear4bit(  # type: ignore[attr-defined]
                in_features,
                out_features,
                bias=bias,
                compute_dtype=torch.float16,
            )
        else:
            self.base = nn.Linear(in_features, out_features, bias=bias)

        # The base weight is frozen by construction: only the low-rank
        # adapters (lora_a, lora_b) and the magnitude vector are trainable.
        # This is what makes QDoRA cheap — the backbone never moves.
        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(in_features, config.rank, bias=False)
        self.lora_b = nn.Linear(config.rank, out_features, bias=False)
        self.magnitude = nn.Parameter(torch.ones(out_features))

        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        update = self.lora_b(self.lora_a(self.dropout(x)))

        update_norm = update.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        normalized_update = update / update_norm
        scaled_update = normalized_update * self.magnitude * self.scaling
        return base_out + scaled_update


class AdapterFusion(nn.Module):
    """Fuse multiple adapter/task outputs with learned soft weighting.

    Input:
            adapter_outputs: list of tensors, each (batch, seq_len, hidden_dim)

    Output:
            Tensor (batch, seq_len, hidden_dim)
    """

    def __init__(self, hidden_dim: int, num_adapters: int) -> None:
        super().__init__()
        if num_adapters < 1:
            raise ValueError("num_adapters must be >= 1")

        self.num_adapters = num_adapters
        self.score = nn.Linear(hidden_dim, num_adapters)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, adapter_outputs: list[Tensor]) -> Tensor:
        if len(adapter_outputs) != self.num_adapters:
            raise ValueError(
                f"Expected {self.num_adapters} adapter outputs, got {len(adapter_outputs)}"
            )

        stacked = torch.stack(adapter_outputs, dim=2)
        pooled = stacked.mean(dim=2)
        logits = self.score(pooled)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        fused = (stacked * weights).sum(dim=2)
        return self.proj(fused)
