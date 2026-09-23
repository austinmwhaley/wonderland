"""Checkpoint persistence for trained backbones.

The "train the backbone once, attach many heads" paradigm only works
operationally if the backbone can be saved and reloaded.  This module
provides that keystone:

* :func:`save_core_checkpoint` / :func:`load_core_checkpoint` — serialize a
  trained :class:`EntityCore` (weights plus the construction config needed
  to rebuild the exact module tree) to a single file.
* :func:`derive_backbone_version` — content-hash the weights so a
  ``backbone_version`` can be *derived* from the actual parameters rather
  than being a free-form string that can drift out of sync.

The higher-level ``save_pretrained`` / ``load_pretrained`` methods on
``EmbeddingModel``, ``TemporalCoreModel``, and ``SupervisedModel`` build on
these primitives.

Note: checkpoints saved with 4-bit quantized QDoRA bases require
``bitsandbytes`` at load time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .adaptation import QDoRAConfig, QDoRALinear
from .entity_core import EntityCore
from .interfaces import TaskHeadBase
from .sequence import SequenceEngine
from .temporal import TemporalStack

_CHECKPOINT_FORMAT = 1


class _PassThroughTaskHead(TaskHeadBase):
	"""No-op task head used when reconstructing a backbone-only core."""

	def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
		return {}


def derive_backbone_version(core: EntityCore) -> str:
	"""Content-hash a core's weights into a stable version identifier.

	The same weights always produce the same version; any weight change
	(retraining, adapter updates) produces a different one.  Use this as
	``backbone_version`` when writing point-in-time state records so the
	store can never silently mix incompatible representations.
	"""

	hasher = hashlib.sha256()
	for name, tensor in sorted(core.state_dict().items()):
		hasher.update(name.encode("utf-8"))
		hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
	return "w-" + hasher.hexdigest()[:12]


def _qdora_config_of(module: torch.nn.Module) -> dict[str, Any] | None:
	"""Return the QDoRA config of a FeedForward's inner layer, if present."""

	linear = getattr(module, "linear_in", None)
	if not isinstance(linear, QDoRALinear):
		return None
	return {
		"rank": linear.rank,
		"alpha": linear.alpha,
		# 4-bit bases require bitsandbytes at load time.
		"quantize_base": "Linear4bit" in type(linear.base).__name__,
		"dropout": float(linear.dropout.p),
	}


def core_config_from_core(core: EntityCore) -> dict[str, Any]:
	"""Extract the construction config needed to rebuild *core* exactly."""

	engine = core.sequence_engine
	if not isinstance(engine, SequenceEngine):
		raise TypeError("checkpointing requires a SequenceEngine-based core")
	temporal = core.temporal_encoder
	if not isinstance(temporal, TemporalStack):
		raise TypeError("checkpointing requires a TemporalStack-based encoder")

	block = engine.shared_block
	ssm = block.ssm  # type: ignore[attr-defined]
	num_heads = int(getattr(getattr(block, "attn", None), "num_heads", 0)) or 8
	dropout = float(getattr(getattr(block, "ff", None), "dropout", torch.nn.Dropout(0.1)).p)
	return {
		"hidden_dim": int(temporal.hidden_dim),
		"backend": str(engine.backend_name),
		"recurrent_steps": int(engine.recurrent_steps),
		"num_heads": num_heads,
		"state_dim": int(ssm.state_dim),
		"conv_kernel": int(ssm.conv_kernel),
		"dropout": dropout,
		"qdora": _qdora_config_of(block.ff),  # type: ignore[attr-defined]
	}


def build_core_from_config(config: dict[str, Any]) -> EntityCore:
	"""Rebuild an :class:`EntityCore` (with passthrough head) from a config."""

	qdora = config.get("qdora")
	qdora_config = QDoRAConfig(**qdora) if qdora else None
	return EntityCore(
		temporal_encoder=TemporalStack(
			hidden_dim=int(config["hidden_dim"]),
			dropout=float(config.get("dropout", 0.1)),
		),
		sequence_engine=SequenceEngine(
			hidden_dim=int(config["hidden_dim"]),
			num_heads=int(config.get("num_heads", 8)),
			state_dim=int(config.get("state_dim", 64)),
			conv_kernel=int(config.get("conv_kernel", 4)),
			recurrent_steps=int(config.get("recurrent_steps", 2)),
			dropout=float(config.get("dropout", 0.1)),
			backend=str(config.get("backend", "samba")),
			qdora_config=qdora_config,
		),
		task_head=_PassThroughTaskHead(),
	)


def save_core_checkpoint(
	core: EntityCore,
	path: str | Path,
	backbone_version: str | None = None,
	extra: dict[str, Any] | None = None,
) -> str:
	"""Serialize *core* (weights + rebuild config) to *path*.

	Args:
		core: The trained :class:`EntityCore`.
		path: Destination file (parent directories are created).
		backbone_version: Version tag stored in the checkpoint.  When omitted,
			the content-derived :func:`derive_backbone_version` is used so the
			version always reflects the actual weights.
		extra: Optional metadata (field mappings, training info) stored alongside.

	Returns:
		The ``backbone_version`` written into the checkpoint.
	"""

	version = backbone_version or derive_backbone_version(core)
	payload = {
		"format": _CHECKPOINT_FORMAT,
		"config": core_config_from_core(core),
		"state_dict": {k: v.detach().cpu() for k, v in core.state_dict().items()},
		"backbone_version": version,
		"weights_version": derive_backbone_version(core),
		"extra": dict(extra or {}),
	}
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, path)
	return version


def load_core_checkpoint(
	path: str | Path,
	map_location: str | torch.device = "cpu",
) -> tuple[EntityCore, str, dict[str, Any]]:
	"""Load a checkpoint written by :func:`save_core_checkpoint`.

	Returns:
		``(core, backbone_version, extra)`` — the rebuilt core with weights
		loaded, the stored version tag, and any metadata saved with it.
	"""

	try:
		payload = torch.load(path, map_location=map_location, weights_only=True)
	except TypeError:  # older torch without weights_only
		payload = torch.load(path, map_location=map_location)
	core = build_core_from_config(payload["config"])
	core.load_state_dict(payload["state_dict"])
	return core, str(payload["backbone_version"]), dict(payload.get("extra") or {})
