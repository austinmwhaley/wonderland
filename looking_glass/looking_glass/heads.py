"""Business task heads for the Universal Entity Engine."""

from __future__ import annotations

from collections import OrderedDict

from torch import Tensor, nn

from .adaptation import AdapterFusion, QDoRAConfig, QDoRALinear
from .interfaces import TaskHeadBase


class BusinessHead(TaskHeadBase):
	"""Single-task business head.

	Input:
		hidden_states: Tensor (batch, seq_len, hidden_dim)

	Output:
		{task_name: Tensor (batch, seq_len, output_dim)}
	"""

	def __init__(
		self,
		hidden_dim: int,
		output_dim: int,
		task_name: str,
		qdora_config: QDoRAConfig | None = None,
	) -> None:
		super().__init__()
		self.task_name = task_name
		if qdora_config is None:
			qdora_config = QDoRAConfig()

		self.proj = QDoRALinear(
			in_features=hidden_dim,
			out_features=output_dim,
			config=qdora_config,
		)

	def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
		return {self.task_name: self.proj(hidden_states)}


class MultiTaskBusinessHead(TaskHeadBase):
	"""Multi-task head with adapter fusion.

	Input:
		hidden_states: Tensor (batch, seq_len, hidden_dim)

	Output:
		Dict[str, Tensor] where each value is (batch, seq_len, output_dim)
	"""

	def __init__(
		self,
		hidden_dim: int,
		task_output_dims: dict[str, int],
		qdora_config: QDoRAConfig | None = None,
		use_fusion: bool = True,
	) -> None:
		super().__init__()
		if not task_output_dims:
			raise ValueError("task_output_dims must contain at least one task")
		if qdora_config is None:
			qdora_config = QDoRAConfig()

		self.task_names = list(task_output_dims.keys())
		self.shared_adapters = nn.ModuleDict(
			OrderedDict(
				(
					task,
					QDoRALinear(
						in_features=hidden_dim,
						out_features=hidden_dim,
						config=qdora_config,
					),
				)
				for task in self.task_names
			)
		)
		self.heads = nn.ModuleDict(
			OrderedDict(
				(task, nn.Linear(hidden_dim, out_dim))
				for task, out_dim in task_output_dims.items()
			)
		)
		self.use_fusion = use_fusion
		self.fusion = (
			AdapterFusion(hidden_dim=hidden_dim, num_adapters=len(self.task_names))
			if use_fusion
			else None
		)

	def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
		adapted = [self.shared_adapters[task](hidden_states) for task in self.task_names]

		if self.use_fusion and self.fusion is not None:
			fused = self.fusion(adapted)
			return {task: self.heads[task](fused) for task in self.task_names}

		outputs: dict[str, Tensor] = {}
		for task, task_states in zip(self.task_names, adapted):
			outputs[task] = self.heads[task](task_states)
		return outputs


class ClassificationHead(nn.Module):
	"""Generic classification head for binary or multi-class tasks on pooled representations.

	Takes a pooled representation (single vector per batch element) and produces class logits.
	This is separate from TaskHeadBase heads which operate on sequences.

	Args:
		hidden_dim: Dimension of input pooled representation.
		num_classes: Number of output classes. Default 2 for binary classification.
		intermediate_dim: Optional dimension for hidden layer. If None, uses hidden_dim.
	"""

	def __init__(
		self,
		hidden_dim: int,
		num_classes: int = 2,
		intermediate_dim: int | None = None,
	) -> None:
		super().__init__()
		if intermediate_dim is None:
			intermediate_dim = hidden_dim

		self.net = nn.Sequential(
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, intermediate_dim),
			nn.SiLU(),
			nn.Linear(intermediate_dim, num_classes),
		)
		self._num_classes = num_classes

	def forward(self, pooled_states: Tensor) -> Tensor:
		"""Map pooled states to class logits.

		Args:
			pooled_states: Tensor of shape (batch_size, hidden_dim).

		Returns:
			Tensor of shape (batch_size, num_classes). If num_classes==1, returns shape (batch_size,).
		"""
		output = self.net(pooled_states)
		if self._num_classes == 1:
			return output.squeeze(-1)
		return output


class RegressionHead(nn.Module):
	"""Generic regression head for continuous value prediction on pooled representations.

	Takes a pooled representation (single vector per batch element) and produces continuous predictions.
	This is separate from TaskHeadBase heads which operate on sequences.

	Args:
		hidden_dim: Dimension of input pooled representation.
		output_dim: Dimension of output predictions. Default 1 for scalar regression.
		intermediate_dim: Optional dimension for hidden layer. If None, uses hidden_dim.
	"""

	def __init__(
		self,
		hidden_dim: int,
		output_dim: int = 1,
		intermediate_dim: int | None = None,
	) -> None:
		super().__init__()
		if intermediate_dim is None:
			intermediate_dim = hidden_dim

		self.net = nn.Sequential(
			nn.LayerNorm(hidden_dim),
			nn.Linear(hidden_dim, intermediate_dim),
			nn.SiLU(),
			nn.Linear(intermediate_dim, output_dim),
		)
		self._output_dim = output_dim

	def forward(self, pooled_states: Tensor) -> Tensor:
		"""Map pooled states to continuous predictions.

		Args:
			pooled_states: Tensor of shape (batch_size, hidden_dim).

		Returns:
			Tensor of shape (batch_size,) if output_dim==1, else (batch_size, output_dim).
		"""
		output = self.net(pooled_states)
		if self._output_dim == 1:
			return output.squeeze(-1)
		return output
