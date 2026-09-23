"""Entity embedding training and vector persistence utilities.

This module is the dimensional-representation layer for the project.

High-level idea:
1. Take tabular rows for one entity type (for example products or customers).
2. Encode categorical fields with learned embeddings.
3. Project numeric and optional precomputed vector fields into the same hidden space.
4. Train a shared backbone with self-supervised reconstruction losses.
5. Export one stable vector per entity id and persist it to LanceDB.

Why this exists:
- It gives the temporal model compact, reusable context vectors.
- It centralizes feature normalization and persistence contracts.
- It keeps dimensional models consistent across entity types.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sqlite3

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

from .entity_core import EntityCore
from .checkpoint import build_core_from_config, core_config_from_core, derive_backbone_version
from .heads import ClassificationHead, RegressionHead
from .interfaces import TaskHeadBase
from .sequence import SequenceEngine
from .temporal import TemporalStack


class _PassThroughTaskHead(TaskHeadBase):
    """No-op task head used when only backbone encodings are required."""

    def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
        return {}


def _best_available_device() -> torch.device:
    """Pick the fastest available accelerator with a deterministic fallback.

    Resolution order is CUDA, then Apple MPS, then CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Runtime and optimization settings for :class:`EmbeddingModel`."""

    hidden_dim: int = 128
    epochs: int = 120
    seed: int = 17
    learning_rate: float = 1e-2
    device: str = "auto"
    sequence_backend: str = "samba"
    # Maximum records per gradient-update step. Keeps CUDA FlashAttention
    # under its 65 535 batch-size ceiling and limits GPU memory usage.
    train_batch_size: int = 8192
    show_progress: bool = False
    progress_label: str | None = None


@dataclass(frozen=True)
class Embeddings:
    """Container for learned vectors with convenience persistence methods."""

    id_field: str
    vectors: dict[str, list[float]]
    records: list[dict[str, object]]

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, entity_id: str) -> list[float]:
        return self.vectors[entity_id]

    def save(
        self,
        lancedb_dir: Path,
        output_table: str,
        records: list[dict[str, object]] | None = None,
        keep_fields: list[str] | None = None,
    ) -> int:
        return save_embeddings_to_lancedb(
            lancedb_dir=lancedb_dir,
            output_table=output_table,
            records=(records if records is not None else self.records),
            id_field=self.id_field,
            embeddings=self,
            keep_fields=keep_fields,
        )


class EmbeddingModel:
    """Generic entity embedding model for tabular records."""

    def __init__(
        self,
        id_field: str,
        categorical_fields: list[str],
        numeric_fields: list[str],
        vector_fields: list[str] | None = None,
        config: EmbeddingModelConfig | None = None,
    ) -> None:
        """Store model configuration and select the execution device.

        Args:
            id_field: Unique identifier column used to aggregate vectors.
            categorical_fields: Discrete feature names.
            numeric_fields: Continuous feature names.
            vector_fields: Optional existing vector features to concatenate.
            config: Optional training configuration.
        """

        self.id_field = id_field
        self.categorical_fields = list(categorical_fields)
        self.numeric_fields = list(numeric_fields)
        self.vector_fields = list(vector_fields or [])
        self.config = config or EmbeddingModelConfig()
        self.device_ = (
            _best_available_device()
            if self.config.device == "auto"
            else torch.device(self.config.device)
        )
        self.loss_: float | None = None
        self._cat_vocab: dict[str, dict[str, int]] = {}
        # Trained artefacts retained after fit_transform so the model can be
        # checkpointed (save_pretrained) and reused for inference (transform).
        self._artifacts: dict[str, object] | None = None

    def fit_transform(self, records: list[dict[str, object]]) -> Embeddings:
        """Train the embedding model and return per-entity vectors.

        Training uses reconstruction-style objectives:
        - Categorical heads reconstruct category indices.
        - A regression head reconstructs normalized continuous inputs.

        This encourages latent states to preserve broad feature information
        while staying architecture-compatible with the temporal core stack.
        """

        if not records:
            self.loss_ = 0.0
            return Embeddings(id_field=self.id_field, vectors={}, records=[])

        torch.manual_seed(self.config.seed)

        # Build deterministic vocabularies so repeated runs with same seed are stable.
        for field in self.categorical_fields:
            values = sorted({str(r.get(field, "")) for r in records})
            self._cat_vocab[field] = {value: idx for idx, value in enumerate(values)}

        cat_idx_tensors: dict[str, Tensor] = {
            field: torch.tensor(
                [self._cat_vocab[field][str(r.get(field, ""))] for r in records],
                dtype=torch.long,
                device=self.device_,
            )
            for field in self.categorical_fields
        }

        # Normalize numeric values to zero-mean / unit-scale for stable optimization.
        means: Tensor | None = None
        stds: Tensor | None = None
        if self.numeric_fields:
            numeric_values = torch.tensor(
                [
                    [float(r.get(field, 0.0) or 0.0) for field in self.numeric_fields]
                    for r in records
                ],
                dtype=torch.float32,
                device=self.device_,
            )
            means = numeric_values.mean(dim=0, keepdim=True)
            stds = numeric_values.std(dim=0, keepdim=True).clamp(min=1e-6)
            numeric_norm = (numeric_values - means) / stds
        else:
            numeric_norm = torch.zeros(len(records), 0, dtype=torch.float32, device=self.device_)

        vector_tensors: list[Tensor] = []
        vector_dims: dict[str, int] = {}
        for field in self.vector_fields:
            sample = next((r.get(field) for r in records if isinstance(r.get(field), list)), None)
            dim = len(sample) if isinstance(sample, list) else 0
            if dim == 0:
                continue
            vector_dims[field] = dim
            # Missing or mismatched vectors become zeros so batch shapes remain valid.
            vec = torch.tensor(
                [
                    (r.get(field) if isinstance(r.get(field), list) and len(r.get(field)) == dim else [0.0] * dim)
                    for r in records
                ],
                dtype=torch.float32,
                device=self.device_,
            )
            vector_tensors.append(vec)

        if vector_tensors:
            vector_features = torch.cat(vector_tensors, dim=1)
        else:
            vector_features = torch.zeros(len(records), 0, dtype=torch.float32, device=self.device_)

        combined_numeric = torch.cat([numeric_norm, vector_features], dim=1)
        if combined_numeric.size(1) == 0:
            # Keep at least one input channel so the projection layer is always well-defined.
            combined_numeric = torch.zeros(len(records), 1, dtype=torch.float32, device=self.device_)

        cat_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(len(vocab), self.config.hidden_dim)
                for field, vocab in self._cat_vocab.items()
            }
        ).to(self.device_)
        in_dim = int(combined_numeric.size(1))
        numeric_projection = nn.Linear(in_dim, self.config.hidden_dim).to(self.device_)
        core = EntityCore(
            temporal_encoder=TemporalStack(hidden_dim=self.config.hidden_dim),
            sequence_engine=SequenceEngine(
                hidden_dim=self.config.hidden_dim,
                recurrent_steps=2,
                num_heads=8,
                backend=self.config.sequence_backend,
            ),
            task_head=_PassThroughTaskHead(),
        ).to(self.device_)

        cat_heads = nn.ModuleDict(
            {
                field: ClassificationHead(
                    hidden_dim=self.config.hidden_dim,
                    num_classes=len(self._cat_vocab[field]),
                )
                for field in self.categorical_fields
            }
        ).to(self.device_)
        numeric_head = RegressionHead(
            hidden_dim=self.config.hidden_dim,
            output_dim=int(combined_numeric.size(1)),
        ).to(self.device_)

        optimizer = torch.optim.Adam(
            list(cat_embeddings.parameters())
            + list(numeric_projection.parameters())
            + list(core.parameters())
            + list(cat_heads.parameters())
            + list(numeric_head.parameters()),
            lr=self.config.learning_rate,
        )

        ids = [str(r[self.id_field]) for r in records]
        n = len(records)
        batch_size = min(self.config.train_batch_size, n)
        last_loss = 0.0
        _label = self.config.progress_label or f"Embedding {self.id_field}"

        epoch_range = range(self.config.epochs)
        epoch_iter = (
            tqdm(epoch_range, desc=_label, unit="epoch", leave=False)
            if self.config.show_progress and self.config.epochs > 0
            else epoch_range
        )

        for ep_idx, _ in enumerate(epoch_iter):
            # Shuffle once per epoch so each mini-batch sees a different slice.
            perm = torch.randperm(n, device=self.device_)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]

                cat_sum = torch.zeros(len(idx), self.config.hidden_dim, dtype=torch.float32, device=self.device_)
                for field, embedding in cat_embeddings.items():
                    cat_sum = cat_sum + embedding(cat_idx_tensors[field][idx])
                x = (cat_sum + numeric_projection(combined_numeric[idx])).unsqueeze(1)
                delta_t = torch.zeros(x.size(0), 1, 1, dtype=torch.float32, device=self.device_)
                encoded = core(x, delta_t).encoded_states[:, 0, :]

                losses: list[Tensor] = []
                for field in self.categorical_fields:
                    losses.append(F.cross_entropy(cat_heads[field](encoded), cat_idx_tensors[field][idx]))
                numeric_pred = numeric_head(encoded)
                if numeric_pred.ndim == 1:
                    losses.append(F.mse_loss(numeric_pred, combined_numeric[idx].squeeze(-1)))
                else:
                    losses.append(F.mse_loss(numeric_pred, combined_numeric[idx]))

                batch_loss: Tensor = sum(losses)  # type: ignore[assignment]
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss += float(batch_loss.item())
                num_batches += 1
            last_loss = epoch_loss / max(num_batches, 1)
            if (ep_idx + 1) % 10 == 0 or ep_idx == self.config.epochs - 1:
                logger.debug("%s  epoch %d/%d  loss=%.6f", _label, ep_idx + 1, self.config.epochs, last_loss)

        # Inference pass: also chunked to stay under the FlashAttention limit.
        core.eval()
        with torch.no_grad():
            vector_chunks: list[Tensor] = []
            for start in range(0, n, batch_size):
                idx = slice(start, start + batch_size)
                cat_sum = torch.zeros(
                    combined_numeric[idx].size(0),
                    self.config.hidden_dim,
                    dtype=torch.float32,
                    device=self.device_,
                )
                for field, embedding in cat_embeddings.items():
                    cat_sum = cat_sum + embedding(cat_idx_tensors[field][idx])
                x = (cat_sum + numeric_projection(combined_numeric[idx])).unsqueeze(1)
                delta_t = torch.zeros(x.size(0), 1, 1, dtype=torch.float32, device=self.device_)
                vector_chunks.append(core(x, delta_t).encoded_states[:, 0, :].detach().cpu())
            vectors = torch.cat(vector_chunks, dim=0).tolist()

        self.loss_ = last_loss
        # Retain trained modules and input statistics so the model can be
        # checkpointed and reused for inference without retraining.
        self._artifacts = {
            "cat_embeddings": cat_embeddings,
            "numeric_projection": numeric_projection,
            "core": core,
            "numeric_mean": means,
            "numeric_std": stds,
            "vector_dims": vector_dims,
            "in_dim": in_dim,
        }
        # Multiple rows can map to one entity id; average them into a single stable vector.
        grouped: dict[str, list[list[float]]] = {}
        for entity_id, vector in zip(ids, vectors):
            grouped.setdefault(entity_id, []).append(vector)
        agg_vectors: dict[str, list[float]] = {}
        for entity_id, values in grouped.items():
            if len(values) == 1:
                agg_vectors[entity_id] = values[0]
            else:
                agg_vectors[entity_id] = torch.tensor(values, dtype=torch.float32).mean(dim=0).tolist()

        return Embeddings(
            id_field=self.id_field,
            vectors=agg_vectors,
            records=records,
        )

    def transform(self, records: list[dict[str, object]]) -> Embeddings:
        """Encode records with the trained model (no further training).

        Unknown categorical values map to vocabulary index 0; numeric
        features are normalised with the statistics learned during
        :meth:`fit_transform`; vector features missing or with a mismatched
        dimension become zero vectors.  Multiple rows mapping to the same
        ``id_field`` are averaged into a single vector.

        Requires that :meth:`fit_transform` has already been called (the
        training artefacts must exist).
        """

        if self._artifacts is None:
            raise RuntimeError("Model must be fit or loaded before transform")
        if not records:
            return Embeddings(id_field=self.id_field, vectors={}, records=[])

        art = self._artifacts
        device = self.device_
        cat_vocab = self._cat_vocab

        cat_idx_tensors: dict[str, Tensor] = {
            field: torch.tensor(
                [vocab.get(str(r.get(field, "")), 0) for r in records],
                dtype=torch.long,
                device=device,
            )
            for field, vocab in cat_vocab.items()
        }

        if self.numeric_fields:
            vals = torch.tensor(
                [
                    [float(r.get(field, 0.0) or 0.0) for field in self.numeric_fields]
                    for r in records
                ],
                dtype=torch.float32,
                device=device,
            )
            numeric_norm = (vals - art["numeric_mean"].to(device)) / art["numeric_std"].to(device)
        else:
            numeric_norm = torch.zeros(len(records), 0, dtype=torch.float32, device=device)

        vec_tensors: list[Tensor] = []
        for field in self.vector_fields:
            dim = int(art["vector_dims"].get(field, 0))
            if dim == 0:
                continue
            vec_tensors.append(
                torch.tensor(
                    [
                        (r.get(field) if isinstance(r.get(field), list) and len(r.get(field)) == dim else [0.0] * dim)
                        for r in records
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            )

        vec_features = torch.cat(vec_tensors, dim=1) if vec_tensors else torch.zeros(len(records), 0, dtype=torch.float32, device=device)
        combined_numeric = torch.cat([numeric_norm, vec_features], dim=1)
        if combined_numeric.size(1) == 0:
            combined_numeric = torch.zeros(len(records), 1, dtype=torch.float32, device=device)

        cat_embs = art["cat_embeddings"]
        num_proj = art["numeric_projection"]
        core = art["core"]
        core.eval()
        batch_size = min(self.config.train_batch_size, len(records))
        n = len(records)
        ids = [str(r[self.id_field]) for r in records]

        with torch.no_grad():
            vector_chunks: list[Tensor] = []
            for start in range(0, n, batch_size):
                idx = slice(start, start + batch_size)
                cat_sum = torch.zeros(combined_numeric[idx].size(0), self.config.hidden_dim, dtype=torch.float32, device=device)
                for field, emb in cat_embs.items():
                    cat_sum = cat_sum + emb(cat_idx_tensors[field][idx])
                x = (cat_sum + num_proj(combined_numeric[idx])).unsqueeze(1)
                delta_t = torch.zeros(x.size(0), 1, 1, dtype=torch.float32, device=device)
                vector_chunks.append(core(x, delta_t).encoded_states[:, 0, :].detach().cpu())
            vectors = torch.cat(vector_chunks, dim=0).tolist()

        grouped: dict[str, list[list[float]]] = {}
        for entity_id, vector in zip(ids, vectors):
            grouped.setdefault(entity_id, []).append(vector)
        agg_vectors: dict[str, list[float]] = {}
        for entity_id, values in grouped.items():
            if len(values) == 1:
                agg_vectors[entity_id] = values[0]
            else:
                agg_vectors[entity_id] = torch.tensor(values, dtype=torch.float32).mean(dim=0).tolist()

        return Embeddings(id_field=self.id_field, vectors=agg_vectors, records=records)

    def save_pretrained(self, path: str | Path) -> str:
        """Persist trained weights, vocabulary, and normalisation statistics.

        The checkpoint is a single ``.pt`` file that can be reloaded with
        :meth:`load_pretrained`.  Returns a content-derived version hash.
        """

        if self._artifacts is None:
            raise RuntimeError("No trained artefacts to save; call fit_transform first")

        art = self._artifacts
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, object] = {
            "format": 1,
            "id_field": self.id_field,
            "categorical_fields": list(self.categorical_fields),
            "numeric_fields": list(self.numeric_fields),
            "vector_fields": list(self.vector_fields),
            "hidden_dim": self.config.hidden_dim,
            "sequence_backend": self.config.sequence_backend,
            "cat_vocab": self._cat_vocab,
            "numeric_mean": art["numeric_mean"].detach().cpu() if art["numeric_mean"] is not None else None,
            "numeric_std": art["numeric_std"].detach().cpu() if art["numeric_std"] is not None else None,
            "vector_dims": dict(art["vector_dims"]),
            "in_dim": int(art["in_dim"]),
            "core_config": core_config_from_core(art["core"]),
            "core_state": {k: v.detach().cpu() for k, v in art["core"].state_dict().items()},
            "cat_embeddings_state": {k: v.detach().cpu() for k, v in art["cat_embeddings"].state_dict().items()},  # type: ignore[union-attr]
            "numeric_projection_state": art["numeric_projection"].state_dict(),  # type: ignore[union-attr]
        }
        for k, v in list(payload.items()):
            if isinstance(v, Tensor):
                payload[k] = v.detach().cpu()

        torch.save(payload, path)
        return derive_backbone_version(art["core"])  # type: ignore[arg-type]

    @classmethod
    def load_pretrained(cls, path: str | Path, device: str = "auto") -> "EmbeddingModel":
        """Reload a checkpoint saved by :meth:`save_pretrained`.

        The restored model is ready for :meth:`transform` without retraining.
        """

        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")

        config = EmbeddingModelConfig(
            hidden_dim=int(payload["hidden_dim"]),
            sequence_backend=str(payload["sequence_backend"]),
            device=device,
        )
        model = cls(
            id_field=str(payload["id_field"]),
            categorical_fields=list(payload["categorical_fields"]),
            numeric_fields=list(payload["numeric_fields"]),
            vector_fields=list(payload["vector_fields"]),
            config=config,
        )
        model._cat_vocab = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in payload["cat_vocab"].items()}

        core = build_core_from_config(payload["core_config"])
        core.load_state_dict(payload["core_state"])
        core = core.to(model.device_)

        cat_embs = nn.ModuleDict({
            field: nn.Embedding(max(1, len(vocab)), config.hidden_dim)
            for field, vocab in model._cat_vocab.items()
        }).to(model.device_)
        cat_embs.load_state_dict(payload["cat_embeddings_state"])

        in_dim = int(payload["in_dim"])
        num_proj = nn.Linear(in_dim, config.hidden_dim).to(model.device_)
        num_proj.load_state_dict(payload["numeric_projection_state"])

        model._artifacts = {
            "cat_embeddings": cat_embs,
            "numeric_projection": num_proj,
            "core": core,
            "numeric_mean": payload["numeric_mean"].to(model.device_) if payload["numeric_mean"] is not None else None,
            "numeric_std": payload["numeric_std"].to(model.device_) if payload["numeric_std"] is not None else None,
            "vector_dims": {k: int(v) for k, v in payload["vector_dims"].items()},
            "in_dim": in_dim,
        }
        return model


def create_embedding_model(
    id_field: str,
    categorical_fields: list[str],
    numeric_fields: list[str],
    vector_fields: list[str] | None = None,
    hidden_dim: int = 128,
    epochs: int = 120,
    seed: int = 17,
    device: str = "auto",
    sequence_backend: str = "samba",
    train_batch_size: int = 8192,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> EmbeddingModel:
    """Construct an :class:`EmbeddingModel` from flat keyword arguments."""

    return EmbeddingModel(
        id_field=id_field,
        categorical_fields=categorical_fields,
        numeric_fields=numeric_fields,
        vector_fields=vector_fields,
        config=EmbeddingModelConfig(
            hidden_dim=hidden_dim,
            epochs=epochs,
            seed=seed,
            device=device,
            sequence_backend=sequence_backend,
            train_batch_size=train_batch_size,
            show_progress=show_progress,
            progress_label=progress_label,
        ),
    )


def load_records_from_sqlite(
    sqlite_path: Path,
    source_table: str,
    columns: list[str],
    where: str | None = None,
    params: tuple[object, ...] | None = None,
    order_by: str | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Load rows from SQLite into dictionary records.

    The query is intentionally simple and explicit so callers control filtering,
    ordering, and limits from one place in pipeline code.
    """

    select_cols = ", ".join(columns)
    query = f"SELECT {select_cols} FROM {source_table}"
    if where:
        query += f" WHERE {where}"
    if order_by:
        query += f" ORDER BY {order_by}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(query, params or ()).fetchall()
    return [dict(zip(columns, row)) for row in rows]


def save_embeddings_to_lancedb(
    lancedb_dir: Path,
    output_table: str,
    records: list[dict[str, object]],
    id_field: str,
    embeddings: Embeddings | dict[str, list[float]],
    keep_fields: list[str] | None = None,
) -> int:
    """Persist vectors to a LanceDB table and return written row count.

    Only records with ids present in the vector map are written. Duplicated ids
    are collapsed to one row in the output table.
    """

    keep_fields = keep_fields or []
    vectors = embeddings.vectors if isinstance(embeddings, Embeddings) else embeddings
    by_id: dict[str, dict[str, object]] = {}
    for row in records:
        entity_id = str(row[id_field])
        if entity_id not in vectors:
            continue
        output = {id_field: entity_id, "vector": vectors[entity_id]}
        for field in keep_fields:
            output[field] = row.get(field)
        by_id.setdefault(entity_id, output)
    output_rows = list(by_id.values())

    try:
        import lancedb
    except ImportError as exc:
        raise RuntimeError(
            "lancedb is not installed. Install with: pip install lancedb pyarrow"
        ) from exc

    lancedb_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lancedb_dir))
    db.create_table(output_table, data=output_rows, mode="overwrite")
    return len(output_rows)


def load_vectors_from_lancedb(
    lancedb_dir: Path,
    table_name: str,
    id_field: str,
    vector_field: str = "vector",
) -> dict[str, list[float]]:
    """Load vector rows from LanceDB into a plain ``id -> vector`` mapping."""

    try:
        import lancedb
    except ImportError as exc:
        raise RuntimeError(
            "lancedb is not installed. Install with: pip install lancedb pyarrow"
        ) from exc

    db = lancedb.connect(str(lancedb_dir))
    table = db.open_table(table_name)
    if hasattr(table, "to_pylist"):
        rows = table.to_pylist()
    else:
        rows = table.to_arrow().to_pylist()
    return {str(row[id_field]): row[vector_field] for row in rows}


def attach_vector_feature(
    records: list[dict[str, object]],
    lookup_key: str,
    vector_lookup: dict[str, list[float]],
    output_field: str,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> list[dict[str, object]]:
    """Add a looked-up vector field to every record, mutating in-place.

    Mutating instead of copying avoids O(n * num_fields) dict allocations for
    large event tables (e.g., 10M rows × 8 fields = 80M operations saved).
    """
    # Infer default vector width so missing ids get shape-compatible zero vectors.
    default_dim = len(next(iter(vector_lookup.values()))) if vector_lookup else 0
    default_vec = [0.0] * default_dim
    row_iter = (
        tqdm(
            records,
            desc=(progress_label or f"Attach {output_field}"),
            unit="row",
            leave=False,
        )
        if show_progress
        else records
    )
    for row in row_iter:
        key = str(row.get(lookup_key, ""))
        row[output_field] = vector_lookup.get(key, default_vec)  # type: ignore[index]
    return records
