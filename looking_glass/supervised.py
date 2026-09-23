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

logger = logging.getLogger(__name__)

from .embeddings import _best_available_device
from .checkpoint import build_core_from_config, core_config_from_core, derive_backbone_version
from .entity_core import EntityCore
from .heads import ClassificationHead, RegressionHead
from .interfaces import TaskHeadBase
from .metrics import pr_auc, roc_auc
from .sequence import SequenceEngine
from .temporal import TemporalStack


class _PassThroughTaskHead(TaskHeadBase):
    """No-op task head used when the model only needs encoded states."""

    def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
        return {}


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


class SupervisedModel:
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

    def fit_predict(self, records: list[dict[str, object]]) -> PredictionResults:
        """Train on a train/validation split and return id-level predictions.

        The split is randomized but reproducible from ``config.seed``.
        For regression, targets are normalized during optimization and restored
        to original scale at inference time.

        The automatic training policy applies: warmup + cosine schedule,
        gradient clipping, and early stopping inside each attempt, plus a
        collapse guard that retries at ``retry_lr_factor`` times the LR
        (up to ``max_retries`` times) when the head learns nothing. The
        best attempt by validation ranking metric is returned.
        """

        attempt_lr = float(self.config.learning_rate)
        max_attempts = max(0, int(self.config.max_retries)) + 1
        best: tuple[float, PredictionResults, dict[str, object], float | None, ValidationReport | None] | None = None
        for attempt in range(max_attempts):
            results, collapsed, key = self._fit_once(records, attempt_lr)
            snapshot = (dict(self._artifacts or {}), self.loss_, self.report_)
            if best is None or key > best[0]:
                best = (key, results) + snapshot  # type: ignore[assignment]
            if not collapsed:
                break
            if not self.config.auto_retry or attempt + 1 >= max_attempts:
                logger.warning(
                    "Supervised %s head collapsed after %d attempt(s); keeping best (metric=%.4f)",
                    self.task, attempt + 1, best[0],
                )
                break
            attempt_lr = attempt_lr * float(self.config.retry_lr_factor)
            logger.warning(
                "Supervised %s head collapsed (metric=%.4f); retry %d/%d at lr=%.2e",
                self.task, key, attempt + 1, max_attempts - 1, attempt_lr,
            )
        assert best is not None  # _fit_once raises on empty input
        _, results, artifacts, loss_value, report = best
        self._artifacts = artifacts
        self.loss_ = loss_value
        self.report_ = report
        return results

    def _fit_once(
        self, records: list[dict[str, object]], learning_rate: float
    ) -> tuple[PredictionResults, bool, float]:
        """Run one training attempt; return (results, collapsed, val_metric)."""

        if len(records) < 2:
            raise ValueError("Need at least 2 records for supervised training/validation")

        torch.manual_seed(self.config.seed)
        rng = random.Random(self.config.seed)

        # Build a deterministic shuffled split from the configured seed.
        indices = list(range(len(records)))
        rng.shuffle(indices)
        val_count = max(1, int(len(records) * self.config.validation_fraction))
        val_indices = sorted(indices[:val_count])
        train_indices = sorted(indices[val_count:])
        if not train_indices:
            raise ValueError("Training split is empty; reduce validation_fraction")

        train_records = [records[idx] for idx in train_indices]

        # Prevent leakage by deriving categorical vocab from training rows only.
        cat_vocab: dict[str, dict[str, int]] = {}
        for field in self.categorical_fields:
            values = sorted({str(r.get(field, "")) for r in train_records})
            cat_vocab[field] = {value: idx for idx, value in enumerate(values)}

        cat_idx_tensors: dict[str, Tensor] = {
            field: torch.tensor(
                [
                    cat_vocab[field].get(str(r.get(field, "")), 0)
                    for r in records
                ],
                dtype=torch.long,
                device=self.device_,
            )
            for field in self.categorical_fields
        }

        numeric_values, feature_means, feature_stds = self._build_numeric_matrix(records, train_indices=train_indices)

        cat_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(max(1, len(vocab)), self.config.hidden_dim)
                for field, vocab in cat_vocab.items()
            }
        ).to(self.device_)
        numeric_projection = nn.Linear(int(numeric_values.size(1)), self.config.hidden_dim).to(self.device_)

        pretrained_core: EntityCore | None = self.config.pretrained_core  # type: ignore[assignment]
        if pretrained_core is not None:
            core = copy.deepcopy(pretrained_core).to(self.device_)
            _freeze_core_except_adapters(core)
        else:
            qdora_cfg = self.config.qdora_config  # type: ignore[attr-defined]
            core = EntityCore(
                temporal_encoder=TemporalStack(hidden_dim=self.config.hidden_dim),
                sequence_engine=SequenceEngine(
                    hidden_dim=self.config.hidden_dim,
                    recurrent_steps=2,
                    num_heads=8,
                    backend=self.config.sequence_backend,
                    qdora_config=qdora_cfg,
                ),
                task_head=_PassThroughTaskHead(),
            ).to(self.device_)
        if self.task == "classification":
            task_head: nn.Module = ClassificationHead(hidden_dim=self.config.hidden_dim, num_classes=1).to(self.device_)
        else:
            task_head = RegressionHead(hidden_dim=self.config.hidden_dim, output_dim=1).to(self.device_)

        trainable_params: list[torch.nn.Parameter] = (
            list(cat_embeddings.parameters())
            + list(numeric_projection.parameters())
            + [p for p in core.parameters() if p.requires_grad]
            + list(task_head.parameters())
        )
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

        # Linear warmup + cosine decay so one default LR works across scales.
        total_epochs = max(int(self.config.epochs), 0)
        warmup_epochs = max(1, int(total_epochs * float(self.config.warmup_fraction))) if total_epochs > 0 else 0
        use_cosine = bool(self.config.use_cosine_schedule)

        def _lr_multiplier(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            if not use_cosine or total_epochs <= warmup_epochs:
                return 1.0
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_multiplier)

        targets = torch.tensor(
            [float(r.get(self.target_field, 0.0) or 0.0) for r in records],
            dtype=torch.float32,
            device=self.device_,
        )

        train_idx_t = torch.tensor(train_indices, dtype=torch.long, device=self.device_)
        val_idx_t = torch.tensor(val_indices, dtype=torch.long, device=self.device_)

        normalized_targets = targets
        regression_target_mean = torch.tensor(0.0, dtype=torch.float32, device=self.device_)
        regression_target_std = torch.tensor(1.0, dtype=torch.float32, device=self.device_)
        if self.task == "regression":
            # Normalize regression targets so learning rate works across scales.
            regression_target_mean = targets[train_idx_t].mean()
            regression_target_std = targets[train_idx_t].std(unbiased=False).clamp(min=1e-6)
            normalized_targets = (targets - regression_target_mean) / regression_target_std

        pos_weight: Tensor | None = None
        if self.task == "classification":
            # Compensate for class imbalance so minority positives are not ignored.
            target_train_full = targets[train_idx_t]
            positive_count = float((target_train_full == 1.0).sum().item())
            negative_count = float((target_train_full == 0.0).sum().item())
            if positive_count > 0.0 and negative_count > 0.0:
                pos_weight = torch.tensor(
                    [max(negative_count / positive_count, 1.0)],
                    dtype=torch.float32,
                    device=self.device_,
                )

        batch_size = min(self.config.train_batch_size, len(train_indices))
        _label = self.config.progress_label or f"Supervised {self.task}"
        last_loss = 0.0

        epoch_range = range(self.config.epochs)
        epoch_iter: Iterable = (
            tqdm(epoch_range, desc=_label, unit="epoch", leave=False)
            if self.config.show_progress and self.config.epochs > 0
            else epoch_range
        )

        patience = max(0, int(self.config.early_stop_patience))
        min_delta = float(self.config.early_stop_min_delta)
        use_early_stop = patience > 0 and total_epochs > 1

        def _snapshot_state() -> dict[str, dict[str, Tensor]]:
            return {
                "cat_embeddings": copy.deepcopy(cat_embeddings.state_dict()),
                "numeric_projection": copy.deepcopy(numeric_projection.state_dict()),
                "core": copy.deepcopy(core.state_dict()),
                "task_head": copy.deepcopy(task_head.state_dict()),
            }

        def _restore_state(state: dict[str, dict[str, Tensor]]) -> None:
            cat_embeddings.load_state_dict(state["cat_embeddings"])
            numeric_projection.load_state_dict(state["numeric_projection"])
            core.load_state_dict(state["core"])
            task_head.load_state_dict(state["task_head"])

        def _validation_loss() -> float:
            core.eval()
            task_head.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for start in range(0, len(val_indices), batch_size):
                    chunk = val_idx_t[start : start + batch_size]
                    enc = self._encode(cat_embeddings, cat_idx_tensors, numeric_projection, numeric_values, core, idx=chunk)
                    pred = task_head(enc)
                    tgt = normalized_targets[chunk]
                    if self.task == "classification":
                        batch_loss = F.binary_cross_entropy_with_logits(pred, tgt, pos_weight=pos_weight)
                    else:
                        batch_loss = F.mse_loss(pred, tgt)
                    total += float(batch_loss.item())
                    count += 1
            core.train()
            task_head.train()
            return total / max(count, 1)

        best_val_loss = float("inf")
        best_state: dict[str, dict[str, Tensor]] | None = None
        stagnant = 0

        for ep_idx, _ in enumerate(epoch_iter):
            # Shuffle training indices so each mini-batch sees different samples.
            shuffled = train_idx_t[torch.randperm(len(train_idx_t), device=self.device_)]
            ep_loss = 0.0
            num_batches = 0
            for start in range(0, len(shuffled), batch_size):
                batch_idx = shuffled[start : start + batch_size]
                enc = self._encode(cat_embeddings, cat_idx_tensors, numeric_projection, numeric_values, core, idx=batch_idx)
                pred = task_head(enc)
                tgt = normalized_targets[batch_idx]
                if self.task == "classification":
                    loss = F.binary_cross_entropy_with_logits(pred, tgt, pos_weight=pos_weight)
                else:
                    loss = F.mse_loss(pred, tgt)
                optimizer.zero_grad()
                loss.backward()
                if self.config.grad_clip_norm and float(self.config.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(self.config.grad_clip_norm))
                optimizer.step()
                ep_loss += float(loss.item())
                num_batches += 1
            scheduler.step()
            last_loss = ep_loss / max(num_batches, 1)
            if (ep_idx + 1) % 10 == 0 or ep_idx == self.config.epochs - 1:
                logger.debug("%s  epoch %d/%d  loss=%.6f", _label, ep_idx + 1, self.config.epochs, last_loss)
            if use_early_stop:
                val_loss = _validation_loss()
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_state = _snapshot_state()
                    stagnant = 0
                else:
                    stagnant += 1
                    if stagnant >= patience:
                        logger.info(
                            "%s  early stop at epoch %d/%d (best val loss=%.6f)",
                            _label, ep_idx + 1, self.config.epochs, best_val_loss,
                        )
                        break
        if best_state is not None:
            _restore_state(best_state)

        with torch.no_grad():
            core.eval()
            task_head.eval()
            best_threshold: float | None = None
            # Validation set is at most 25% of records — safe to encode in one shot.
            enc_val = self._encode(cat_embeddings, cat_idx_tensors, numeric_projection, numeric_values, core, idx=val_idx_t)
            pred_val = task_head(enc_val)
            target_val = targets[val_idx_t]

            # Full-dataset predictions chunked to stay under the FlashAttention limit.
            n_all = numeric_values.size(0)
            pred_chunks: list[Tensor] = []
            for start in range(0, n_all, batch_size):
                chunk_idx = torch.arange(start, min(start + batch_size, n_all), device=self.device_)
                pred_chunks.append(task_head(self._encode(cat_embeddings, cat_idx_tensors, numeric_projection, numeric_values, core, idx=chunk_idx)))
            pred_all = torch.cat(pred_chunks, dim=0)

            if self.task == "classification":
                # Choose threshold on training predictions, then evaluate on held-out rows.
                enc_train = self._encode(
                    cat_embeddings,
                    cat_idx_tensors,
                    numeric_projection,
                    numeric_values,
                    core,
                    idx=train_idx_t,
                )
                pred_train = task_head(enc_train)
                probs_train = torch.sigmoid(pred_train)
                probs = torch.sigmoid(pred_val)
                best_threshold = _best_classification_threshold(probs_train, targets[train_idx_t])
                metrics = _classification_metrics(probs, target_val, threshold=best_threshold)
                metrics["threshold"] = best_threshold
                # Held-out ranking quality (threshold-free, honest).
                val_probs_list = probs.detach().cpu().tolist()
                val_labels_list = target_val.detach().cpu().tolist()
                metrics["roc_auc"] = roc_auc(val_probs_list, val_labels_list)
                metrics["pr_auc"] = pr_auc(val_probs_list, val_labels_list)
                all_outputs = torch.sigmoid(pred_all).detach().cpu().tolist()
            else:
                # Convert regression outputs back to natural target units.
                pred_val = (pred_val * regression_target_std) + regression_target_mean
                pred_all = (pred_all * regression_target_std) + regression_target_mean
                metrics = _regression_metrics(pred_val, target_val)
                all_outputs = pred_all.detach().cpu().tolist()

        self.loss_ = last_loss
        report = ValidationReport(
            task=self.task,
            target_field=self.target_field,
            train_rows=len(train_indices),
            validation_rows=len(val_indices),
            metrics=metrics,
        )
        self.report_ = report
        self._artifacts = {
            "core": core,
            "task_head": task_head,
            "cat_embeddings": cat_embeddings,
            "numeric_projection": numeric_projection,
            "cat_vocab": cat_vocab,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "regression_target_mean": regression_target_mean,
            "regression_target_std": regression_target_std,
            "threshold": best_threshold,
        }

        # Aggregate multiple rows per id by mean prediction.
        by_id: dict[str, list[float]] = {}
        for row, value in zip(records, all_outputs):
            by_id.setdefault(str(row[self.id_field]), []).append(float(value))
        predictions = {
            key: float(sum(values) / len(values))
            for key, values in by_id.items()
        }

        # Collapse guard: a head that emits (near-)constant outputs or ranks
        # at chance learned nothing — typically a too-hot learning rate.
        with torch.no_grad():
            spread = pred_val.detach().float()
            if self.task == "classification":
                spread = torch.sigmoid(spread)
            out_std = float(spread.std(unbiased=False).item()) if spread.numel() > 1 else 0.0
        try:
            finite = math.isfinite(float(last_loss)) and all(
                math.isfinite(float(v)) for v in all_outputs
            )
        except (TypeError, ValueError):
            finite = False
        if self.task == "classification":
            key = float(metrics.get("roc_auc", float("nan")))
            collapsed = (
                (not finite)
                or out_std < 1e-6
                or (not math.isfinite(key))
                or key < float(self.config.collapse_auc_threshold)
            )
        else:
            key = float(metrics.get("r2", float("nan")))
            collapsed = (
                (not finite) or out_std < 1e-6 or (math.isfinite(key) and key < -1.0)
            )
        if not math.isfinite(key):
            key = float("-inf")

        return (
            PredictionResults(
                id_field=self.id_field,
                target_field=self.target_field,
                predictions=predictions,
                report=report,
            ),
            collapsed,
            key,
        )

    def _build_numeric_matrix(
        self,
        records: list[dict[str, object]],
        train_indices: list[int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build normalized continuous feature matrix (always returns stats)."""

        numeric_base = torch.tensor(
            [
                [float(r.get(field, 0.0) or 0.0) for field in self.numeric_fields]
                for r in records
            ],
            dtype=torch.float32,
            device=self.device_,
        ) if self.numeric_fields else torch.zeros(len(records), 0, dtype=torch.float32, device=self.device_)

        vector_tensors: list[Tensor] = []
        for field in self.vector_fields:
            sample = next((r.get(field) for r in records if isinstance(r.get(field), list)), None)
            dim = len(sample) if isinstance(sample, list) else 0
            if dim == 0:
                continue
            vector_tensors.append(
                torch.tensor(
                    [
                        (r.get(field) if isinstance(r.get(field), list) and len(r.get(field)) == dim else [0.0] * dim)
                        for r in records
                    ],
                    dtype=torch.float32,
                    device=self.device_,
                )
            )

        if vector_tensors:
            vector_concat = torch.cat(vector_tensors, dim=1)
            combined = torch.cat([numeric_base, vector_concat], dim=1)
        else:
            combined = numeric_base

        if combined.size(1) == 0:
            # Keep one channel so downstream projection always has valid input size.
            z = torch.zeros(len(records), 1, dtype=torch.float32, device=self.device_)
            return z, torch.tensor([[0.0]], device=self.device_), torch.tensor([[1.0]], device=self.device_)

        if train_indices:
            train_idx_t = torch.tensor(train_indices, dtype=torch.long, device=self.device_)
            reference = combined[train_idx_t]
        else:
            reference = combined

        means = reference.mean(dim=0, keepdim=True)
        stds = reference.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
        return (combined - means) / stds, means, stds

    def _encode(
        self,
        cat_embeddings: nn.ModuleDict,
        cat_idx_tensors: dict[str, Tensor],
        numeric_projection: nn.Linear,
        numeric_values: Tensor,
        core: EntityCore,
        idx: Tensor | None = None,
    ) -> Tensor:
        """Encode rows with the shared backbone and return pooled states.

        If ``idx`` is provided, only the selected rows are encoded.
        """
        if idx is not None:
            nv = numeric_values[idx]
            ci: dict[str, Tensor] = {f: t[idx] for f, t in cat_idx_tensors.items()}
        else:
            nv = numeric_values
            ci = cat_idx_tensors
        n = nv.size(0)
        cat_sum = torch.zeros(n, self.config.hidden_dim, dtype=torch.float32, device=self.device_)
        for field, embedding in cat_embeddings.items():
            cat_sum = cat_sum + embedding(ci[field])
        x = (cat_sum + numeric_projection(nv)).unsqueeze(1)
        delta_t = torch.zeros(n, 1, 1, dtype=torch.float32, device=self.device_)
        return core(x, delta_t).encoded_states[:, 0, :]

    def predict(self, records: list[dict[str, object]]) -> dict[str, float]:
        """Score records with the trained head (no further training).

        Returns a mapping from ``id_field`` values to prediction scores.
        For classification tasks these are probabilities (0-1); for
        regression they are values in the original target units.

        Requires that :meth:`fit_predict` has already been called or the
        model was loaded via :meth:`load_pretrained`.
        """

        if self._artifacts is None:
            raise RuntimeError("Model must be fit or loaded before predict")
        if not records:
            return {}

        art = self._artifacts
        device = self.device_
        cat_vocab = art["cat_vocab"]

        cat_idx_tensors: dict[str, Tensor] = {
            field: torch.tensor(
                [vocab.get(str(r.get(field, "")), 0) for r in records],
                dtype=torch.long,
                device=device,
            )
            for field, vocab in cat_vocab.items()
        }

        # Build and normalize the numeric + vector feature matrix.
        numeric_base = torch.tensor(
            [
                [float(r.get(field, 0.0) or 0.0) for field in self.numeric_fields]
                for r in records
            ],
            dtype=torch.float32,
            device=device,
        ) if self.numeric_fields else torch.zeros(len(records), 0, dtype=torch.float32, device=device)

        vector_tensors: list[Tensor] = []
        for field in self.vector_fields:
            sample = next((r.get(field) for r in records if isinstance(r.get(field), list)), None)
            dim = len(sample) if isinstance(sample, list) else 0
            if dim == 0:
                continue
            vector_tensors.append(
                torch.tensor(
                    [
                        (r.get(field) if isinstance(r.get(field), list) and len(r.get(field)) == dim else [0.0] * dim)
                        for r in records
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            )
        combined = torch.cat([numeric_base] + vector_tensors, dim=1) if vector_tensors else numeric_base
        if combined.size(1) == 0:
            combined = torch.zeros(len(records), 1, dtype=torch.float32, device=device)

        f_mean = art["feature_means"].to(device)
        f_std = art["feature_stds"].to(device)
        numeric_values = (combined - f_mean) / f_std

        cat_embs = art["cat_embeddings"]
        num_proj = art["numeric_projection"]
        core = art["core"]
        task_head = art["task_head"]
        core.eval()
        task_head.eval()

        batch_size = min(self.config.train_batch_size, len(records))
        n = len(records)

        with torch.no_grad():
            pred_chunks: list[Tensor] = []
            for start in range(0, n, batch_size):
                chunk_idx = torch.arange(start, min(start + batch_size, n), device=device)
                enc = self._encode(cat_embs, cat_idx_tensors, num_proj, numeric_values, core, idx=chunk_idx)
                pred_chunks.append(task_head(enc))
            pred_all = torch.cat(pred_chunks, dim=0)

        if self.task == "classification":
            outputs = torch.sigmoid(pred_all).detach().cpu().tolist()
        else:
            t_mean = art["regression_target_mean"].to(device)
            t_std = art["regression_target_std"].to(device)
            outputs = ((pred_all * t_std) + t_mean).detach().cpu().tolist()

        by_id: dict[str, list[float]] = {}
        for row, value in zip(records, outputs):
            by_id.setdefault(str(row[self.id_field]), []).append(float(value))
        return {key: float(sum(vals) / len(vals)) for key, vals in by_id.items()}

    def save_pretrained(self, path: str | Path) -> str:
        """Persist trained weights and metadata to a single checkpoint file.

        The checkpoint contains the model architecture, vocabularies,
        normalization statistics, and all learned parameters, so it can be
        fully restored with :meth:`load_pretrained`.

        Returns a content-derived version hash.
        """

        if self._artifacts is None:
            raise RuntimeError("No trained artefacts to save; call fit_predict first")

        art = self._artifacts
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _cpu(t: Tensor | None) -> object:
            return t.detach().cpu() if t is not None else None

        payload: dict[str, object] = {
            "format": 1,
            "task": self.task,
            "id_field": self.id_field,
            "target_field": self.target_field,
            "categorical_fields": list(self.categorical_fields),
            "numeric_fields": list(self.numeric_fields),
            "vector_fields": list(self.vector_fields),
            "hidden_dim": self.config.hidden_dim,
            "sequence_backend": self.config.sequence_backend,
            "cat_vocab": art["cat_vocab"],
            "feature_means": _cpu(art["feature_means"]),
            "feature_stds": _cpu(art["feature_stds"]),
            "regression_target_mean": _cpu(art["regression_target_mean"]),
            "regression_target_std": _cpu(art["regression_target_std"]),
            "threshold": float(art["threshold"]) if art["threshold"] is not None else None,
            "core_config": core_config_from_core(art["core"]),
            "core_state": {k: v.detach().cpu() for k, v in art["core"].state_dict().items()},
            "head_state": art["task_head"].state_dict(),
            "cat_embeddings_state": {k: v.detach().cpu() for k, v in art["cat_embeddings"].state_dict().items()},
            "numeric_projection_state": art["numeric_projection"].state_dict(),
        }
        torch.save(payload, path)
        return derive_backbone_version(art["core"])

    @classmethod
    def load_pretrained(cls, path: str | Path, device: str = "auto") -> "SupervisedModel":
        """Reload a checkpoint saved by :meth:`save_pretrained`.

        The restored model is ready for :meth:`predict` without retraining.
        """

        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        task = str(payload["task"])
        hidden_dim = int(payload["hidden_dim"])
        config = SupervisedModelConfig(
            hidden_dim=hidden_dim,
            sequence_backend=str(payload["sequence_backend"]),
            device=device,
        )
        model = cls(
            task=task,
            id_field=str(payload["id_field"]),
            target_field=str(payload["target_field"]),
            categorical_fields=list(payload["categorical_fields"]),
            numeric_fields=list(payload["numeric_fields"]),
            vector_fields=list(payload["vector_fields"]),
            config=config,
        )

        cat_vocab = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in payload["cat_vocab"].items()}

        cat_embs = nn.ModuleDict({
            field: nn.Embedding(max(1, len(vocab)), hidden_dim)
            for field, vocab in cat_vocab.items()
        }).to(model.device_)
        cat_embs.load_state_dict(payload["cat_embeddings_state"])

        # Infer projection input dim from saved weight shape.
        np_state = dict(payload["numeric_projection_state"])
        np_weight = np_state["weight"]
        in_dim = int(np_weight.shape[1] if len(np_weight.shape) == 2 else np_weight.shape[0])
        num_proj = nn.Linear(in_dim, hidden_dim).to(model.device_)
        num_proj.load_state_dict(np_state)

        core = build_core_from_config(payload["core_config"]).to(model.device_)
        core.load_state_dict(payload["core_state"])

        if task == "classification":
            task_head: nn.Module = ClassificationHead(hidden_dim=hidden_dim, num_classes=1).to(model.device_)
        else:
            task_head = RegressionHead(hidden_dim=hidden_dim, output_dim=1).to(model.device_)
        task_head.load_state_dict(payload["head_state"])

        model._artifacts = {
            "core": core,
            "task_head": task_head,
            "cat_embeddings": cat_embs,
            "numeric_projection": num_proj,
            "cat_vocab": cat_vocab,
            "feature_means": None if payload["feature_means"] is None else torch.as_tensor(payload["feature_means"]).to(model.device_),
            "feature_stds": None if payload["feature_stds"] is None else torch.as_tensor(payload["feature_stds"]).to(model.device_),
            "regression_target_mean": torch.as_tensor(payload["regression_target_mean"]) if payload["regression_target_mean"] is not None else torch.tensor(0.0, device=model.device_),
            "regression_target_std": torch.as_tensor(payload["regression_target_std"]) if payload["regression_target_std"] is not None else torch.tensor(1.0, device=model.device_),
            "threshold": float(payload["threshold"]) if payload["threshold"] is not None else None,
        }
        return model

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


def validate_embeddings(
    vectors: dict[str, list[float]],
    min_vectors: int = 1,
    min_norm_std: float = 0.0,
    min_nonzero_fraction: float = 0.0,
    max_mean_abs_cosine: float = 1.0,
    cosine_sample_size: int = 512,
    seed: int = 17,
) -> dict[str, float | int | bool]:
    """Check vector-map quality for dimensional or temporal embeddings.

    Validation combines structural checks (count/dimension/finite values) with
    geometric checks (norm spread, non-zero rate, and pairwise cosine collapse).
    """

    count = len(vectors)
    dims = {len(v) for v in vectors.values()}
    vector_dim = next(iter(dims)) if len(dims) == 1 else -1
    finite = True
    mean_norm = 0.0
    norm_std = 0.0
    nonzero_fraction = 0.0
    mean_abs_cosine = 0.0

    if count > 0 and vector_dim > 0:
        matrix = torch.tensor(list(vectors.values()), dtype=torch.float32)
        finite = bool(torch.isfinite(matrix).all().item())
        if not finite:
            matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        norms = torch.linalg.vector_norm(matrix, dim=1)
        mean_norm = float(norms.mean().item())
        norm_std = float(norms.std(unbiased=False).item())
        nonzero_fraction = float((norms > 1e-8).float().mean().item())

        if matrix.size(0) > 1:
            sample_count = min(int(cosine_sample_size), int(matrix.size(0)))
            generator = torch.Generator(device=matrix.device)
            generator.manual_seed(seed)
            sample_idx = torch.randperm(matrix.size(0), generator=generator, device=matrix.device)[:sample_count]
            sample = F.normalize(matrix[sample_idx], dim=1, eps=1e-12)
            cosine = sample @ sample.T
            off_diag_mask = ~torch.eye(sample_count, dtype=torch.bool, device=matrix.device)
            off_diag = cosine[off_diag_mask]
            if off_diag.numel() > 0:
                mean_abs_cosine = float(off_diag.abs().mean().item())
    else:
        for vector in vectors.values():
            for value in vector:
                if not math.isfinite(float(value)):
                    finite = False
                    break
            if not finite:
                break

    norm_std_ok = norm_std >= float(min_norm_std)
    nonzero_ok = nonzero_fraction >= float(min_nonzero_fraction)
    cosine_ok = mean_abs_cosine <= float(max_mean_abs_cosine)

    return {
        "ok": (
            count >= min_vectors
            and len(dims) == 1
            and finite
            and norm_std_ok
            and nonzero_ok
            and cosine_ok
        ),
        "vector_count": count,
        "vector_dim": vector_dim,
        "finite": finite,
        "mean_norm": mean_norm,
        "norm_std": norm_std,
        "nonzero_fraction": nonzero_fraction,
        "mean_abs_cosine": mean_abs_cosine,
        "norm_std_ok": norm_std_ok,
        "nonzero_ok": nonzero_ok,
        "cosine_ok": cosine_ok,
    }


def validate_prediction_report(report: ValidationReport) -> dict[str, bool]:
    """Validate that report row counts are non-zero and metric values are finite."""

    has_rows = report.train_rows > 0 and report.validation_rows > 0
    metrics_finite = all(math.isfinite(float(v)) for v in report.metrics.values())
    return {
        "ok": has_rows and metrics_finite,
        "has_rows": has_rows,
        "metrics_finite": metrics_finite,
    }


def validate_classification_success(
    report: ValidationReport,
    min_f1: float = 0.05,
    min_recall: float = 0.05,
    min_precision: float = 0.0,
    min_accuracy: float = 0.0,
) -> dict[str, bool | float]:
    """Evaluate classification report against configurable minimum thresholds."""

    accuracy = float(report.metrics.get("accuracy", 0.0))
    precision = float(report.metrics.get("precision", 0.0))
    f1 = float(report.metrics.get("f1", 0.0))
    recall = float(report.metrics.get("recall", 0.0))
    accuracy_ok = accuracy >= min_accuracy
    precision_ok = precision >= min_precision
    f1_ok = f1 >= min_f1
    recall_ok = recall >= min_recall
    return {
        "ok": accuracy_ok and precision_ok and f1_ok and recall_ok,
        "accuracy_ok": accuracy_ok,
        "precision_ok": precision_ok,
        "f1_ok": f1_ok,
        "recall_ok": recall_ok,
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "recall": recall,
    }


def validate_regression_success(
    report: ValidationReport,
    min_r2: float = 0.0,
    max_rmse: float | None = None,
) -> dict[str, bool | float]:
    """Evaluate regression report against configurable R^2 and optional RMSE bounds."""

    r2 = float(report.metrics.get("r2", float("-inf")))
    rmse = float(report.metrics.get("rmse", float("inf")))
    r2_ok = r2 >= min_r2
    rmse_ok = True if max_rmse is None else rmse <= max_rmse
    return {
        "ok": r2_ok and rmse_ok,
        "r2_ok": r2_ok,
        "rmse_ok": rmse_ok,
        "r2": r2,
        "rmse": rmse,
    }


def _freeze_core_except_adapters(core: nn.Module) -> None:
    """Freeze all parameters of *core* except those belonging to QDoRA adapters.

    QDoRA adapter parameters are identified by the substring ``lora_a``,
    ``lora_b``, or ``magnitude`` in their name. All other parameters
    (SSM, attention, norms, etc.) are frozen so they survive downstream
    task-specific training unchanged.
    """

    adapter_keys = {"lora_a", "lora_b", "magnitude"}
    for name, param in core.named_parameters():
        requires = any(keyword in name for keyword in adapter_keys)
        param.requires_grad = requires


def _classification_metrics(
    probs: Tensor,
    targets: Tensor,
    threshold: float,
) -> dict[str, float]:
    """Compute binary classification diagnostics for a fixed threshold."""

    preds = (probs >= threshold).float()
    correct = float((preds == targets).float().mean().item())
    tp = float(((preds == 1.0) & (targets == 1.0)).sum().item())
    fp = float(((preds == 1.0) & (targets == 0.0)).sum().item())
    fn = float(((preds == 0.0) & (targets == 1.0)).sum().item())
    tn = float(((preds == 0.0) & (targets == 0.0)).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    specificity = tn / max(tn + fp, 1.0)
    brier = float(((probs - targets) ** 2).mean().item())
    positive_rate = float(targets.mean().item()) if targets.numel() > 0 else 0.0
    predicted_positive_rate = float(preds.mean().item()) if preds.numel() > 0 else 0.0
    return {
        "accuracy": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "brier": brier,
        "positive_rate": positive_rate,
        "predicted_positive_rate": predicted_positive_rate,
    }


def _best_classification_threshold(probs: Tensor, targets: Tensor) -> float:
    """Pick threshold maximizing validation-train F1 over a fixed candidate grid."""

    if probs.numel() == 0:
        return 0.5

    positive_count = float((targets == 1.0).sum().item())
    negative_count = float((targets == 0.0).sum().item())
    if positive_count == 0.0 or negative_count == 0.0:
        return 0.5

    candidate_thresholds: Iterable[float] = [i / 100.0 for i in range(10, 91, 2)]
    best_threshold = 0.5
    best_score = -1.0

    for threshold in candidate_thresholds:
        metrics = _classification_metrics(probs, targets, threshold=threshold)
        score = float(metrics["f1"])
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def _regression_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Compute common regression diagnostics used by smoke quality gates."""

    mae = float(torch.mean(torch.abs(pred - target)).item())
    mse = float(torch.mean((pred - target) ** 2).item())
    rmse = float(torch.sqrt(torch.tensor(mse)).item())
    target_mean = torch.mean(target)
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2).clamp(min=1e-6)
    r2 = float((1.0 - (ss_res / ss_tot)).item())
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
