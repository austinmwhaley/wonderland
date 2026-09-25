"""Interface definitions for swappable Universal Entity Engine modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class TemporalEncoderBase(nn.Module, ABC):
    """Standardized interface for temporal encoding modules.

    Input:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
            delta_t: Tensor of shape (batch, seq_len, 1) or (batch, seq_len)

    Output:
            Tensor of shape (batch, seq_len, hidden_dim)
    """

    @abstractmethod
    def forward(self, hidden_states: Tensor, delta_t: Tensor) -> Tensor:
        """Apply temporal features to hydrated entity states."""


class SequenceEngineBase(nn.Module, ABC):
    """Standardized interface for sequence-processing backbones.

    Input:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Optional tensor broadcastable to attention scores

    Output:
            Tensor of shape (batch, seq_len, hidden_dim)
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Run sequence modeling over hydrated and time-aware states."""


class TaskHeadBase(nn.Module, ABC):
    """Standardized interface for task-specific business heads.

    Input:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim)

    Output:
            Dict[str, Tensor], where each tensor is task-specific.
    """

    @abstractmethod
    def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
        """Map sequence states into one or more task outputs."""
