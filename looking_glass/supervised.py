# ruff: noqa: F401
"""Supervised task heads — the final stage of the pipeline.

This module supports two training modes:

1.  **From-scratch** (original behaviour) — builds a fresh ``EntityCore`` per
    task.  Suitable for standalone experiments and small datasets.

2.  **Frozen-core LoRA** (the production paradigm) — accepts a pretrained
    ``EntityCore`` via ``pretrained_core``, freezes all non-adapter
    parameters, and trains only the lightweight LoRA adapters (injected into
    the ``FeedForward`` layers of the sequence engine) plus the task-specific
    classification or regression head.  Each call to ``fit_predict``
    deep-copies the core, so multiple tasks train independently on the same
    backbone without interference.

Pipeline summary:
    1. Build categorical vocabularies from training rows only.
    2. Normalize numeric/vector features using training statistics.
    3. Encode rows with the shared temporal + sequence core (seq_len=1).
    4. Train either a binary classification head (BCE-with-logits) or a
       regression head (MSE).
    5. Report validation metrics and full-dataset id-level predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import logging
import math
from pathlib import Path
import random
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from looking_glass.embeddings import _best_available_device
from looking_glass.checkpoint import (
    build_core_from_config,
    core_config_from_core,
    derive_backbone_version,
)
from looking_glass.entity_core import EntityCore
from looking_glass.heads import ClassificationHead, RegressionHead
from looking_glass.interfaces import TaskHeadBase
from looking_glass.metrics import pr_auc, roc_auc
from looking_glass.sequence import SequenceEngine
from looking_glass.temporal import TemporalStack
from looking_glass.supervised_inference import _SupervisedInferenceMixin
from looking_glass.supervised_training import (
    _best_classification_threshold,
    _classification_metrics,
    _freeze_core_except_adapters,
    _regression_metrics,
    _PassThroughTaskHead,
    _SupervisedTrainingMixin,
    logger,
)
from looking_glass.supervised_types import (
    PredictionResults,
    SupervisedModelConfig,
    ValidationReport,
)
from looking_glass.supervised_validation import (
    validate_classification_success,
    validate_embeddings,
    validate_prediction_report,
    validate_regression_success,
)


class SupervisedModel(_SupervisedTrainingMixin, _SupervisedInferenceMixin):
    """Supervised task model supporting from-scratch and frozen-core LoRA training.

    Modes
        * **From-scratch** — builds a new ``EntityCore`` and trains
          everything end-to-end.  Default when ``pretrained_core=None``.
        * **Frozen-core LoRA** — accepts a pretrained ``EntityCore`` (with
          ``QDoRA`` adapters already injected via
          :func:`wrap_core_with_qdora`), freezes all non-adapter parameters,
          and trains only the LoRA adapters and the task head.  Each
          ``fit_predict`` call deep-copies the core so multiple tasks share
          the same backbone without weight interference.

    Tasks
        * ``classification`` — binary label prediction with
          BCE-with-logits.  A positive-class weight compensates for
          imbalance and the decision threshold is optimised for F1 on
          the training split.
        * ``regression`` — continuous value prediction with MSE loss.
          Targets are normalised during training and restored to original
          scale at inference time.
    """

    def __init__(
        self,
        task: str,
        id_field: str,
        target_field: str,
        categorical_fields: list[str],
        numeric_fields: list[str],
        vector_fields: list[str] | None = None,
        config: SupervisedModelConfig | None = None,
    ) -> None:
        """Initialize supervised model metadata and choose execution device."""

        task_value = task.lower().strip()
        if task_value not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")

        self.task = task_value
        self.id_field = id_field
        self.target_field = target_field
        self.categorical_fields = list(categorical_fields)
        self.numeric_fields = list(numeric_fields)
        self.vector_fields = list(vector_fields or [])
        self.config = config or SupervisedModelConfig()
        self.device_ = (
            _best_available_device()
            if self.config.device == "auto"
            else torch.device(self.config.device)
        )
        self.loss_: float | None = None
        self.report_: ValidationReport | None = None
        self._artifacts: dict[str, object] | None = None


def create_supervised_model(
    task: str,
    id_field: str,
    target_field: str,
    categorical_fields: list[str],
    numeric_fields: list[str],
    vector_fields: list[str] | None = None,
    hidden_dim: int = 128,
    epochs: int = 80,
    seed: int = 17,
    learning_rate: float = 1e-2,
    validation_fraction: float = 0.2,
    device: str = "auto",
    sequence_backend: str = "samba",
    train_batch_size: int = 8192,
    show_progress: bool = False,
    progress_label: str | None = None,
    pretrained_core: object | None = None,
    qdora_config: object | None = None,
    warmup_fraction: float = 0.1,
    use_cosine_schedule: bool = True,
    grad_clip_norm: float = 1.0,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 1e-4,
    auto_retry: bool = True,
    max_retries: int = 2,
    retry_lr_factor: float = 0.1,
    collapse_auc_threshold: float = 0.55,
) -> SupervisedModel:
    """Factory helper that mirrors ``SupervisedModel`` constructor options."""

    return SupervisedModel(
        task=task,
        id_field=id_field,
        target_field=target_field,
        categorical_fields=categorical_fields,
        numeric_fields=numeric_fields,
        vector_fields=vector_fields,
        config=SupervisedModelConfig(
            hidden_dim=hidden_dim,
            epochs=epochs,
            seed=seed,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            device=device,
            sequence_backend=sequence_backend,
            train_batch_size=train_batch_size,
            show_progress=show_progress,
            progress_label=progress_label,
            pretrained_core=pretrained_core,
            qdora_config=qdora_config,
            warmup_fraction=warmup_fraction,
            use_cosine_schedule=use_cosine_schedule,
            grad_clip_norm=grad_clip_norm,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            auto_retry=auto_retry,
            max_retries=max_retries,
            retry_lr_factor=retry_lr_factor,
            collapse_auc_threshold=collapse_auc_threshold,
        ),
    )
