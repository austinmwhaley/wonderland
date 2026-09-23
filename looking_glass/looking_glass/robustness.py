"""Robustness utilities for pretraining data quality control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

ValidatorFn = Callable[[dict[str, Tensor]], Tensor]


@dataclass(frozen=True)
class SamplingReport:
	"""Summary of rejection sampling decisions."""

	total: int
	kept: int
	rejected: int
	keep_ratio: float


class RejectionSampler:
	"""Reject inconsistent events from a batch.

	Validator contract:
		Input: batch dict where each value has first dimension = batch size
		Output: boolean tensor of shape (batch,) where True means keep
	"""

	def __init__(self, validators: list[ValidatorFn]) -> None:
		if not validators:
			raise ValueError("At least one validator is required")
		self.validators = validators

	def _combined_mask(self, batch: dict[str, Tensor]) -> Tensor:
		masks = [validator(batch).bool() for validator in self.validators]
		base = masks[0]
		for mask in masks[1:]:
			base = torch.logical_and(base, mask)
		return base

	def filter_batch(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], SamplingReport]:
		if not batch:
			raise ValueError("batch must not be empty")

		keep_mask = self._combined_mask(batch)
		total = int(keep_mask.shape[0])
		kept = int(keep_mask.sum().item())
		rejected = total - kept
		keep_ratio = float(kept / max(total, 1))

		filtered = {name: tensor[keep_mask] for name, tensor in batch.items()}
		report = SamplingReport(
			total=total,
			kept=kept,
			rejected=rejected,
			keep_ratio=keep_ratio,
		)
		return filtered, report
