"""Supervised inference: feature encoding, predict, and checkpoint IO."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from looking_glass.checkpoint import (
    build_core_from_config,
    core_config_from_core,
    derive_backbone_version,
)
from looking_glass.entity_core import EntityCore
from looking_glass.heads import ClassificationHead, RegressionHead
from looking_glass.supervised_types import SupervisedModelConfig

if TYPE_CHECKING:
    from looking_glass.supervised import SupervisedModel


class _SupervisedInferenceMixin:
    def _build_numeric_matrix(
        self,
        records: list[dict[str, object]],
        train_indices: list[int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build normalized continuous feature matrix (always returns stats)."""

        numeric_base = (
            torch.tensor(
                [
                    [float(r.get(field, 0.0) or 0.0) for field in self.numeric_fields]
                    for r in records
                ],
                dtype=torch.float32,
                device=self.device_,
            )
            if self.numeric_fields
            else torch.zeros(len(records), 0, dtype=torch.float32, device=self.device_)
        )

        vector_tensors: list[Tensor] = []
        for field in self.vector_fields:
            sample = next((r.get(field) for r in records if isinstance(r.get(field), list)), None)
            dim = len(sample) if isinstance(sample, list) else 0
            if dim == 0:
                continue
            vector_tensors.append(
                torch.tensor(
                    [
                        (
                            r.get(field)
                            if isinstance(r.get(field), list) and len(r.get(field)) == dim
                            else [0.0] * dim
                        )
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
            return (
                z,
                torch.tensor([[0.0]], device=self.device_),
                torch.tensor([[1.0]], device=self.device_),
            )

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
        numeric_base = (
            torch.tensor(
                [
                    [float(r.get(field, 0.0) or 0.0) for field in self.numeric_fields]
                    for r in records
                ],
                dtype=torch.float32,
                device=device,
            )
            if self.numeric_fields
            else torch.zeros(len(records), 0, dtype=torch.float32, device=device)
        )

        vector_tensors: list[Tensor] = []
        for field in self.vector_fields:
            sample = next((r.get(field) for r in records if isinstance(r.get(field), list)), None)
            dim = len(sample) if isinstance(sample, list) else 0
            if dim == 0:
                continue
            vector_tensors.append(
                torch.tensor(
                    [
                        (
                            r.get(field)
                            if isinstance(r.get(field), list) and len(r.get(field)) == dim
                            else [0.0] * dim
                        )
                        for r in records
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            )
        combined = (
            torch.cat([numeric_base] + vector_tensors, dim=1) if vector_tensors else numeric_base
        )
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
                enc = self._encode(
                    cat_embs, cat_idx_tensors, num_proj, numeric_values, core, idx=chunk_idx
                )
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
            "cat_embeddings_state": {
                k: v.detach().cpu() for k, v in art["cat_embeddings"].state_dict().items()
            },
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

        cat_vocab = {
            k: {kk: int(vv) for kk, vv in v.items()} for k, v in payload["cat_vocab"].items()
        }

        cat_embs = nn.ModuleDict(
            {
                field: nn.Embedding(max(1, len(vocab)), hidden_dim)
                for field, vocab in cat_vocab.items()
            }
        ).to(model.device_)
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
            task_head: nn.Module = ClassificationHead(hidden_dim=hidden_dim, num_classes=1).to(
                model.device_
            )
        else:
            task_head = RegressionHead(hidden_dim=hidden_dim, output_dim=1).to(model.device_)
        task_head.load_state_dict(payload["head_state"])

        model._artifacts = {
            "core": core,
            "task_head": task_head,
            "cat_embeddings": cat_embs,
            "numeric_projection": num_proj,
            "cat_vocab": cat_vocab,
            "feature_means": None
            if payload["feature_means"] is None
            else torch.as_tensor(payload["feature_means"]).to(model.device_),
            "feature_stds": None
            if payload["feature_stds"] is None
            else torch.as_tensor(payload["feature_stds"]).to(model.device_),
            "regression_target_mean": torch.as_tensor(payload["regression_target_mean"])
            if payload["regression_target_mean"] is not None
            else torch.tensor(0.0, device=model.device_),
            "regression_target_std": torch.as_tensor(payload["regression_target_std"])
            if payload["regression_target_std"] is not None
            else torch.tensor(1.0, device=model.device_),
            "threshold": float(payload["threshold"]) if payload["threshold"] is not None else None,
        }
        return model
