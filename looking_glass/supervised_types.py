"""Shared config and result types for supervised task heads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupervisedModelConfig:
    """Hyperparameters and runtime options for supervised training."""

    hidden_dim: int = 128
    epochs: int = 80
    seed: int = 17
    learning_rate: float = 1e-2
    validation_fraction: float = 0.2
    device: str = "auto"
    sequence_backend: str = "samba"
    # When provided the entity core is shared across tasks; only adapter
    # params and the task head are trained.  Set to None for the original
    # per-task-from-scratch behavior.
    pretrained_core: object | None = None  # EntityCore | None
    # Optional QDoRA config for adaptation layers inside the sequence engine.
    # Ignored when pretrained_core is provided (the core already carries it).
    qdora_config: object | None = None  # QDoRAConfig | None
    # Same FlashAttention ceiling guard as EmbeddingModelConfig.
    train_batch_size: int = 8192
    # Automatic head-training policy: linear warmup + cosine LR schedule,
    # gradient clipping, early stopping on validation loss (restore best),
    # and a collapse guard that retries at a reduced LR when the head
    # learns nothing (constant outputs or near-chance ranking).
    warmup_fraction: float = 0.1
    use_cosine_schedule: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4
    auto_retry: bool = True
    max_retries: int = 2
    retry_lr_factor: float = 0.1
    collapse_auc_threshold: float = 0.55
    show_progress: bool = False
    progress_label: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Model-agnostic validation summary emitted after training."""

    task: str
    target_field: str
    train_rows: int
    validation_rows: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class PredictionResults:
    """Container for id-level predictions and their validation report."""

    id_field: str
    target_field: str
    predictions: dict[str, float]
    report: ValidationReport
