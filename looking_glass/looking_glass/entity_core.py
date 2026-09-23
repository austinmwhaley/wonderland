"""Top-level compositional model for Universal Entity Engine."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .interfaces import SequenceEngineBase, TaskHeadBase, TemporalEncoderBase


@dataclass
class EntityCoreOutput:
	"""Structured model output.

	Attributes:
		encoded_states: Tensor (batch, seq_len, hidden_dim)
		task_outputs: Dict[str, Tensor], each tensor shape depends on task
	"""

	encoded_states: Tensor
	task_outputs: dict[str, Tensor]


class EntityCore(nn.Module):
	"""Universal Entity Engine core composition.

	Hydration input contract:
		hidden_states: Tensor (batch, seq_len, hidden_dim)
		delta_t: Tensor (batch, seq_len, 1) or (batch, seq_len)

	The `TemporalEncoderBase`, `SequenceEngineBase`, and `TaskHeadBase`
	modules are fully swappable under a strict interface contract.
	"""

	def __init__(
		self,
		temporal_encoder: TemporalEncoderBase,
		sequence_engine: SequenceEngineBase,
		task_head: TaskHeadBase,
	) -> None:
		super().__init__()
		self.temporal_encoder = temporal_encoder
		self.sequence_engine = sequence_engine
		self.task_head = task_head

	def forward(
		self,
		hidden_states: Tensor,
		delta_t: Tensor,
		attention_mask: Tensor | None = None,
	) -> EntityCoreOutput:
		temporal_states = self.temporal_encoder(hidden_states, delta_t)
		encoded_states = self.sequence_engine(temporal_states, attention_mask)
		task_outputs = self.task_head(encoded_states)
		return EntityCoreOutput(
			encoded_states=encoded_states,
			task_outputs=task_outputs,
		)
