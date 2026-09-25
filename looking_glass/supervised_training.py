"""Supervised head training: the fit loop, collapse guard, and metrics."""

from __future__ import annotations

import copy
import logging
import math
import random
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from looking_glass.entity_core import EntityCore
from looking_glass.heads import ClassificationHead, RegressionHead
from looking_glass.interfaces import TaskHeadBase
from looking_glass.metrics import pr_auc, roc_auc
from looking_glass.sequence import SequenceEngine
from looking_glass.supervised_types import PredictionResults, ValidationReport
from looking_glass.temporal import TemporalStack

# Logger name preserved from the original supervised module.
logger = logging.getLogger("looking_glass.supervised")


class _PassThroughTaskHead(TaskHeadBase):
    """No-op task head used when the model only needs encoded states."""

    def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
        return {}


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


class _SupervisedTrainingMixin:
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
        best: (
            tuple[
                float, PredictionResults, dict[str, object], float | None, ValidationReport | None
            ]
            | None
        ) = None
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
                    self.task,
                    attempt + 1,
                    best[0],
                )
                break
            attempt_lr = attempt_lr * float(self.config.retry_lr_factor)
            logger.warning(
                "Supervised %s head collapsed (metric=%.4f); retry %d/%d at lr=%.2e",
                self.task,
                key,
                attempt + 1,
                max_attempts - 1,
                attempt_lr,
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
                [cat_vocab[field].get(str(r.get(field, "")), 0) for r in records],
                dtype=torch.long,
                device=self.device_,
            )
            for field in self.categorical_fields
        }

        numeric_values, feature_means, feature_stds = self._build_numeric_matrix(
            records, train_indices=train_indices
        )

        cat_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(max(1, len(vocab)), self.config.hidden_dim)
                for field, vocab in cat_vocab.items()
            }
        ).to(self.device_)
        numeric_projection = nn.Linear(int(numeric_values.size(1)), self.config.hidden_dim).to(
            self.device_
        )

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
            task_head: nn.Module = ClassificationHead(
                hidden_dim=self.config.hidden_dim, num_classes=1
            ).to(self.device_)
        else:
            task_head = RegressionHead(hidden_dim=self.config.hidden_dim, output_dim=1).to(
                self.device_
            )

        trainable_params: list[torch.nn.Parameter] = (
            list(cat_embeddings.parameters())
            + list(numeric_projection.parameters())
            + [p for p in core.parameters() if p.requires_grad]
            + list(task_head.parameters())
        )
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

        # Linear warmup + cosine decay so one default LR works across scales.
        total_epochs = max(int(self.config.epochs), 0)
        warmup_epochs = (
            max(1, int(total_epochs * float(self.config.warmup_fraction)))
            if total_epochs > 0
            else 0
        )
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
                    enc = self._encode(
                        cat_embeddings,
                        cat_idx_tensors,
                        numeric_projection,
                        numeric_values,
                        core,
                        idx=chunk,
                    )
                    pred = task_head(enc)
                    tgt = normalized_targets[chunk]
                    if self.task == "classification":
                        batch_loss = F.binary_cross_entropy_with_logits(
                            pred, tgt, pos_weight=pos_weight
                        )
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
                enc = self._encode(
                    cat_embeddings,
                    cat_idx_tensors,
                    numeric_projection,
                    numeric_values,
                    core,
                    idx=batch_idx,
                )
                pred = task_head(enc)
                tgt = normalized_targets[batch_idx]
                if self.task == "classification":
                    loss = F.binary_cross_entropy_with_logits(pred, tgt, pos_weight=pos_weight)
                else:
                    loss = F.mse_loss(pred, tgt)
                optimizer.zero_grad()
                loss.backward()
                if self.config.grad_clip_norm and float(self.config.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, float(self.config.grad_clip_norm)
                    )
                optimizer.step()
                ep_loss += float(loss.item())
                num_batches += 1
            scheduler.step()
            last_loss = ep_loss / max(num_batches, 1)
            if (ep_idx + 1) % 10 == 0 or ep_idx == self.config.epochs - 1:
                logger.debug(
                    "%s  epoch %d/%d  loss=%.6f", _label, ep_idx + 1, self.config.epochs, last_loss
                )
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
                            _label,
                            ep_idx + 1,
                            self.config.epochs,
                            best_val_loss,
                        )
                        break
        if best_state is not None:
            _restore_state(best_state)

        with torch.no_grad():
            core.eval()
            task_head.eval()
            best_threshold: float | None = None
            # Validation set is at most 25% of records — safe to encode in one shot.
            enc_val = self._encode(
                cat_embeddings,
                cat_idx_tensors,
                numeric_projection,
                numeric_values,
                core,
                idx=val_idx_t,
            )
            pred_val = task_head(enc_val)
            target_val = targets[val_idx_t]

            # Full-dataset predictions chunked to stay under the FlashAttention limit.
            n_all = numeric_values.size(0)
            pred_chunks: list[Tensor] = []
            for start in range(0, n_all, batch_size):
                chunk_idx = torch.arange(start, min(start + batch_size, n_all), device=self.device_)
                pred_chunks.append(
                    task_head(
                        self._encode(
                            cat_embeddings,
                            cat_idx_tensors,
                            numeric_projection,
                            numeric_values,
                            core,
                            idx=chunk_idx,
                        )
                    )
                )
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
        predictions = {key: float(sum(values) / len(values)) for key, values in by_id.items()}

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
            collapsed = (not finite) or out_std < 1e-6 or (math.isfinite(key) and key < -1.0)
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
