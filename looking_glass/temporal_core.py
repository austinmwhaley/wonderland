"""Temporal core model — the central pretraining stage.

This module is the bridge between static entity representations (product,
store, and campaign vectors resolved at tokenize time from
``event_payload_json``) and downstream supervised task heads.

It trains on **full customer event timelines** (not isolated rows) with
causal masking, so temporal order and inter-event gaps directly influence
the learned representations.

Architecture
    The tokenizer parses each event's payload JSON and projects every
    signal source — categorical embeddings, numeric projections, and
    resolved entity vectors (product, store, campaign) — independently to
    ``hidden_dim``, then sums them element-wise::

        E_event = Σ E_cat + Σ E_num + Σ E_vector

    TemporalStack enriches the result with time-aware features
    (Time2Vec, TAPE, NeuralODE), then a SequenceEngine (Mamba-2 or Samba)
    processes the causally-masked sequence to produce per-position encoded
    states.

Training objective
    Causal next-event prediction with a dedicated reconstruction head for
    each categorical field and a joint regression head for numeric fields
    PLUS all resolved entity vectors.  The model learns to predict
    "what comes next" from everything that came before.

Outputs
    * Per-event embeddings keyed by ``event_id``
    * Per-customer summaries: ``core_last_vector``, ``core_mean_vector``,
      ``core_event_count``, ``core_as_of_ts``, and ``backbone_version``
    * The trained ``EntityCore`` (exposed via ``trained_core``) ready for
      downstream frozen-core LoRA training
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from .embeddings import _best_available_device
from .checkpoint import load_core_checkpoint, save_core_checkpoint
from .entity_core import EntityCore
from .heads import ClassificationHead, RegressionHead
from .interfaces import TaskHeadBase
from .sequence import SequenceEngine
from .state_store import StateRecord
from .tokenizer import PayloadSchema, parse_event_payload, collect_payload_vocabularies
from .temporal import TemporalStack

logger = logging.getLogger(__name__)


class _PassThroughTaskHead(TaskHeadBase):
    """No-op task head used to expose encoded sequence states directly."""

    def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
        return {}


@dataclass(frozen=True)
class TemporalCoreConfig:
    """Hyperparameters and runtime options for temporal core training."""

    hidden_dim: int = 128
    epochs: int = 120
    seed: int = 17
    learning_rate: float = 1e-2
    device: str = "auto"
    sequence_backend: str = "samba"
    # Stamps every emitted point-in-time state so retraining the backbone does
    # not silently mix incompatible representations (see state_store.invalidate).
    backbone_version: str = "v0"
    # Payload schema for the new event-stream contract. When provided the
    # tokenizer extracts categorical / numeric / product-vector signals from
    # ``event_payload_json`` instead of top-level columns.  Leave as None
    # for the original flat-field behaviour.
    payload_schema: PayloadSchema | None = None
    # Optional entity-vector lookup tables for in-flight payload parsing.
    # Keys match ``payload_schema.vector_id_keys`` (e.g. ``"product_id"``,
    # ``"store_id"``, ``"campaign_id"``).  Each value is a ``{entity_id:
    # vector}`` dict.  All resolved vectors are summed element-wise.
    vector_lookups: dict[str, dict[str, list[float]]] | None = None
    # Same role as in embedding/supervised models; lower default because
    # temporal attention memory grows with sequence length squared.
    train_batch_size: int = 512
    input_is_time_sorted: bool = False
    show_progress: bool = False
    progress_label: str | None = None


@dataclass(frozen=True)
class TemporalCoreOutputs:
    """Structured artifacts produced by :class:`TemporalCoreModel`."""

    sequence_id_field: str
    event_id_field: str
    event_embeddings: dict[str, list[float]]
    customer_records: list[dict[str, object]]
    records: list[dict[str, object]]


class TemporalCoreModel:
    """Temporal sequence model trained via causal next-event prediction.

    Records are grouped by ``sequence_id_field`` and time-ordered per group,
    then encoded with causal attention so each position only attends to
    prior context in its customer's timeline.

    The model supports two input contracts:

    * **Legacy flat columns** — ``categorical_fields``, ``numeric_fields``,
      and ``vector_fields`` accessed directly from top-level dict keys.
    * **Canonical event-stream schema** — a ``PayloadSchema`` that declares
      how the ``event_payload_json`` field maps to features, plus optional
      ``vector_lookups`` for resolving entity IDs (product_id, store_id,
      campaign_id) against pre-computed static embedding tables.

    When a ``PayloadSchema`` is provided, the legacy field lists are
    ignored and the tokenizer parses every event's payload JSON at tensorize
    time.  All resolved entity vectors are summed element-wise before the
    sequence engine processes the token.
    """

    def __init__(
        self,
        sequence_id_field: str,
        event_id_field: str,
        timestamp_field: str,
        categorical_fields: list[str],
        numeric_fields: list[str],
        vector_fields: list[str] | None = None,
        config: TemporalCoreConfig | None = None,
    ) -> None:
        """Store field mappings and choose execution device."""

        self.sequence_id_field = sequence_id_field
        self.event_id_field = event_id_field
        self.timestamp_field = timestamp_field
        self.categorical_fields = list(categorical_fields)
        self.numeric_fields = list(numeric_fields)
        self.vector_fields = list(vector_fields or [])
        self.config = config or TemporalCoreConfig()
        self.device_ = (
            _best_available_device()
            if self.config.device == "auto"
            else torch.device(self.config.device)
        )
        self.loss_: float | None = None
        self._trained_core: EntityCore | None = None

    def fit_transform(self, records: list[dict[str, object]]) -> TemporalCoreOutputs:
        """Train temporal core and emit event/customer embeddings.

        Implementation notes:
        - Sequence tensors are assembled on CPU first to avoid GPU OOM spikes.
        - Mini-batches are moved to the configured device for forward/backward.
        - Causal masks enforce next-event prediction semantics.
        """

        if not records:
            self.loss_ = 0.0
            return TemporalCoreOutputs(
                sequence_id_field=self.sequence_id_field,
                event_id_field=self.event_id_field,
                event_embeddings={},
                customer_records=[],
                records=[],
            )

        torch.manual_seed(self.config.seed)
        progress_prefix = self.config.progress_label or "Temporal core"
        logger.info("%s: start fit records=%d", progress_prefix, len(records))

        sequences = self._group_records(
            records,
            show_progress=self.config.show_progress,
            progress_label=progress_prefix,
            input_is_time_sorted=self.config.input_is_time_sorted,
        )
        if not sequences:
            raise ValueError("No sequences available for temporal core training")
        logger.info("%s: prepared sequences=%d", progress_prefix, len(sequences))

        for _, sequence_rows in sequences:
            if len(sequence_rows) > 1:
                break
        else:
            raise ValueError("Need at least one sequence with 2 or more events")

        # Global categorical vocab so all sequences share one index space.
        cat_vocab: dict[str, dict[str, int]] = {}
        if self.config.payload_schema is not None:
            cat_vocab = collect_payload_vocabularies(records, self.config.payload_schema)
        else:
            cat_field_iter = (
                tqdm(
                    self.categorical_fields,
                    desc=f"{progress_prefix}: build vocab",
                    unit="field",
                    leave=False,
                )
                if self.config.show_progress and self.categorical_fields
                else self.categorical_fields
            )
            for field in cat_field_iter:
                values = sorted({str(row.get(field, "")) for row in records})
                cat_vocab[field] = {value: idx for idx, value in enumerate(values)}
        logger.debug("%s: built vocab fields=%d", progress_prefix, len(cat_vocab))

        batch_size = len(sequences)
        max_seq_len = max(len(sequence_rows) for _, sequence_rows in sequences)
        train_batch_size = _resolve_train_batch_size(
            requested=self.config.train_batch_size,
            total_sequences=batch_size,
            max_seq_len=max_seq_len,
            device=self.device_,
            progress_label=progress_prefix,
        )
        logger.info(
            "%s: tensorize sequences=%d max_seq_len=%d",
            progress_prefix,
            batch_size,
            max_seq_len,
        )

        # Build all large tensors on CPU to avoid GPU OOM. Mini-batches are
        # moved to self.device_ during training and inference passes below.
        _cpu = torch.device("cpu")
        valid_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=_cpu)
        delta_t = torch.zeros(batch_size, max_seq_len, 1, dtype=torch.float32, device=_cpu)
        event_ids = [["" for _ in range(max_seq_len)] for _ in range(batch_size)]

        cat_idx_tensors: dict[str, Tensor] = {}
        if self.config.payload_schema is not None:
            _cat_fields = list(self.config.payload_schema.categorical_fields) + ["event_type"]
            _num_fields = list(self.config.payload_schema.numeric_fields)
            cat_idx_tensors = {
                field: torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=_cpu)
                for field in _cat_fields
            }
        else:
            cat_idx_tensors = {
                field: torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=_cpu)
                for field in self.categorical_fields
            }

        numeric_tensor = torch.zeros(
            batch_size,
            max_seq_len,
            len(self.config.payload_schema.numeric_fields) if self.config.payload_schema is not None else len(self.numeric_fields),
            dtype=torch.float32,
            device=_cpu,
        )

        vector_dims: dict[str, int] = {}
        if self.config.payload_schema is not None and self.config.vector_lookups is not None:
            sample = next((v for table in self.config.vector_lookups.values() for v in table.values() if v), None)
            prod_dim = len(sample) if sample else 0
            if prod_dim > 0:
                vector_dims["combined_vector"] = prod_dim
        else:
            for field in self.vector_fields:
                sample = next((row.get(field) for row in records if isinstance(row.get(field), list)), None)
                vector_dims[field] = len(sample) if isinstance(sample, list) else 0
        vector_tensors: dict[str, Tensor] = {
            field: torch.zeros(batch_size, max_seq_len, dim, dtype=torch.float32, device=_cpu)
            for field, dim in vector_dims.items()
            if dim > 0
        }

        sequence_ids: list[str] = []
        sequence_lengths: list[int] = []
        sequence_last_ts: list[datetime | None] = []

        sequence_iter = (
            tqdm(
                sequences,
                desc=f"{progress_prefix}: tensorize",
                unit="seq",
                leave=False,
            )
            if self.config.show_progress
            else sequences
        )
        total_sequences = len(sequences)
        for batch_idx, (sequence_id, sequence_rows) in enumerate(sequence_iter):
            sequence_ids.append(sequence_id)
            sequence_lengths.append(len(sequence_rows))
            previous_ts: datetime | None = None

            for seq_idx, row in enumerate(sequence_rows):
                valid_mask[batch_idx, seq_idx] = True
                event_ids[batch_idx][seq_idx] = str(row[self.event_id_field])

                current_ts = _parse_iso_timestamp(row[self.timestamp_field])
                if previous_ts is not None:
                    # delta_t is measured in days and clamped non-negative.
                    delta_days = max((current_ts - previous_ts).total_seconds() / 86400.0, 0.0)
                    delta_t[batch_idx, seq_idx, 0] = float(delta_days)
                previous_ts = current_ts

                if self.config.payload_schema is not None:
                    payload = row.get("event_payload_json", {})
                    if isinstance(payload, dict):
                        _build_lookups = {}
                        if self.config.vector_lookups is not None:
                            for key, table in self.config.vector_lookups.items():
                                _build_lookups[key] = lambda eid, tbl=table: tbl.get(eid)
                        parsed = parse_event_payload(
                            payload,
                            self.config.payload_schema,
                            vector_lookups=_build_lookups if _build_lookups else None,
                            default_vector_width=len(next(iter(vector_tensors.values()), torch.zeros(1))),
                        )
                    else:
                        parsed = parse_event_payload({}, self.config.payload_schema)
                    for fld in cat_idx_tensors:
                        if fld == "event_type":
                            cat_idx_tensors[fld][batch_idx, seq_idx] = cat_vocab[fld].get(str(row.get("event_type", "")), 0)
                        else:
                            cat_idx_tensors[fld][batch_idx, seq_idx] = cat_vocab[fld].get(parsed.cat_values.get(fld, ""), 0)
                    for field_idx, fld in enumerate(self.config.payload_schema.numeric_fields):
                        numeric_tensor[batch_idx, seq_idx, field_idx] = parsed.num_values.get(fld, 0.0)
                    if "combined_vector" in vector_tensors:
                        cv = parsed.combined_vector
                        if len(cv) == vector_tensors["combined_vector"].size(-1):
                            vector_tensors["combined_vector"][batch_idx, seq_idx] = torch.tensor(cv, dtype=torch.float32, device=_cpu)
                else:
                    for field in self.categorical_fields:
                        cat_idx_tensors[field][batch_idx, seq_idx] = cat_vocab[field][str(row.get(field, ""))]
                    for field_idx, field in enumerate(self.numeric_fields):
                        numeric_tensor[batch_idx, seq_idx, field_idx] = float(row.get(field, 0.0) or 0.0)
                    for field, tensor in vector_tensors.items():
                        dim = tensor.size(-1)
                        value = row.get(field)
                        if isinstance(value, list) and len(value) == dim:
                            tensor[batch_idx, seq_idx] = torch.tensor(value, dtype=torch.float32, device=_cpu)

            sequence_last_ts.append(previous_ts)
            done_sequences = batch_idx + 1
            if done_sequences % 5_000 == 0 or done_sequences == total_sequences:
                logger.debug(
                    "%s: tensorize %d/%d (%.1f%%)",
                    progress_prefix,
                    done_sequences,
                    total_sequences,
                    100.0 * done_sequences / max(total_sequences, 1),
                )

        # Element-wise token construction: each signal source is independently
        # projected to hidden_dim, then summed, so a missing source (zero
        # vector) acts as the identity element.
        numeric_normalized = _normalize_masked(numeric_tensor, valid_mask) if numeric_tensor.size(-1) > 0 else numeric_tensor
        numeric_targets = _normalize_masked(numeric_tensor, valid_mask) if numeric_tensor.size(-1) > 0 else None

        cat_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(max(1, len(vocab)), self.config.hidden_dim)
                for field, vocab in cat_vocab.items()
            }
        ).to(self.device_)
        numeric_projection = nn.Linear(int(max(numeric_normalized.size(-1), 1)), self.config.hidden_dim).to(self.device_)
        vector_projections = nn.ModuleDict({
            field: nn.Linear(dim, self.config.hidden_dim).to(self.device_)
            for field, tensor in vector_tensors.items()
            if (dim := tensor.size(-1)) > 0
        }) if vector_tensors else nn.ModuleDict()
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
        self._trained_core = core
        cat_heads = nn.ModuleDict(
            {
                field: ClassificationHead(
                    hidden_dim=self.config.hidden_dim,
                    num_classes=len(cat_vocab[field]),
                )
                for field in list(cat_vocab.keys())
            }
        ).to(self.device_)
        _eff_num_dim = int(numeric_tensor.size(-1)) if numeric_tensor.size(-1) > 0 else 0
        _vec_dims = sum(t.size(-1) for t in vector_tensors.values())
        _reg_dim = max(_eff_num_dim + _vec_dims, 1)
        numeric_head = (
            RegressionHead(hidden_dim=self.config.hidden_dim, output_dim=_reg_dim).to(self.device_)
            if _reg_dim > 0
            else None
        )
        # Build regression targets: numeric fields plus all vector tensor (product)
        # concatenated along the feature axis so the next-event head reconstructs both.
        _reg_targets: list[Tensor] = []
        if numeric_targets is not None and numeric_targets.size(-1) > 0:
            _reg_targets.append(numeric_targets)
        for _, vt in vector_tensors.items():
            _reg_targets.append(vt)
        combined_targets = torch.cat(_reg_targets, dim=-1) if len(_reg_targets) > 1 else (_reg_targets[0] if _reg_targets else torch.zeros(batch_size, max_seq_len, 1, device=_cpu))

        optimizer = torch.optim.Adam(
            list(cat_embeddings.parameters())
            + list(numeric_projection.parameters())
            + list(vector_projections.parameters())
            + list(core.parameters())
            + list(cat_heads.parameters())
            + ([] if numeric_head is None else list(numeric_head.parameters())),
            lr=self.config.learning_rate,
        )

        # Need at least one "next step" target; first token in each sequence
        # cannot be used as a target for next-event prediction.
        if int(valid_mask[:, 1:].sum().item()) == 0:
            raise ValueError("No next-event targets available for temporal core training")

        last_loss = 0.0
        logger.info(
            "%s: train epochs=%d train_batch_size=%d",
            progress_prefix,
            self.config.epochs,
            train_batch_size,
        )
        epoch_iter = range(self.config.epochs)
        if self.config.show_progress and self.config.epochs > 0:
            epoch_iter = tqdm(
                epoch_iter,
                desc=(self.config.progress_label or "Temporal core epochs"),
                unit="epoch",
                leave=False,
            )

        try:
            for ep_idx, _ in enumerate(epoch_iter):
                total_loss = 0.0
                num_batches = 0
                for start in range(0, batch_size, train_batch_size):
                    end = min(start + train_batch_size, batch_size)
                    mb_valid = valid_mask[start:end].to(self.device_)
                    mb_delta_t = delta_t[start:end].to(self.device_)
                    mb_numeric = numeric_normalized[start:end].to(self.device_)
                    mb_vectors = {f: t[start:end].to(self.device_) for f, t in vector_tensors.items()}
                    mb_cat = {f: t[start:end].to(self.device_) for f, t in cat_idx_tensors.items()}

                    encoded = self._encode(
                        cat_embeddings=cat_embeddings,
                        cat_idx_tensors=mb_cat,
                        numeric_projection=numeric_projection,
                        numeric_tensor=mb_numeric,
                        vector_tensors=mb_vectors,
                        vector_projections=vector_projections,
                        delta_t=mb_delta_t,
                        valid_mask=mb_valid,
                        core=core,
                    )

                    mb_next_mask = mb_valid[:, 1:]
                    flat_mask = mb_next_mask.reshape(-1)
                    if flat_mask.sum().item() == 0:
                        continue

                    # States up to t predict labels at t+1.
                    context_states = encoded[:, :-1, :]
                    flat_states = context_states.reshape(-1, self.config.hidden_dim)[flat_mask]

                    losses: list[Tensor] = []
                    for field in list(cat_vocab.keys()):
                        targets = mb_cat[field][:, 1:].reshape(-1)[flat_mask]
                        losses.append(F.cross_entropy(cat_heads[field](flat_states), targets))

                    if numeric_head is not None:
                        mb_reg_tgt = combined_targets[start:end].to(self.device_)
                        numeric_pred = numeric_head(flat_states)
                        reg_target = mb_reg_tgt[:, 1:, :].reshape(-1, _reg_dim)[flat_mask]
                        if numeric_pred.ndim == 1:
                            losses.append(F.mse_loss(numeric_pred, reg_target.squeeze(-1)))
                        else:
                            losses.append(F.mse_loss(numeric_pred, reg_target))

                    loss = sum(losses)
                    if not torch.isfinite(loss):
                        logger.warning(
                            "%s: non-finite loss at epoch %d batch_start=%d; skipping update",
                            progress_prefix,
                            ep_idx + 1,
                            start,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                    num_batches += 1

                last_loss = total_loss / max(1, num_batches)
                if (ep_idx + 1) % 10 == 0:
                    _label = self.config.progress_label or "Temporal core"
                    logger.debug("%s  epoch %d/%d  loss=%.6f", _label, ep_idx + 1, self.config.epochs, last_loss)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.exception(
                    "%s: out of memory during temporal training (batch=%d seq_len=%d)",
                    progress_prefix,
                    train_batch_size,
                    max_seq_len,
                )
                if self.device_.type == "cuda":
                    torch.cuda.empty_cache()
            raise

        with torch.no_grad():
            encoded_parts: list[Tensor] = []
            inference_starts = range(0, batch_size, train_batch_size)
            total_inference_batches = (
                batch_size + train_batch_size - 1
            ) // train_batch_size
            logger.info(
                "%s: final encode batches=%d",
                progress_prefix,
                total_inference_batches,
            )
            inference_iter = (
                tqdm(
                    inference_starts,
                    desc=f"{progress_prefix}: final encode",
                    unit="batch",
                    leave=False,
                )
                if self.config.show_progress
                else inference_starts
            )
            for batch_idx, start in enumerate(inference_iter):
                end = min(start + train_batch_size, batch_size)
                mb_valid = valid_mask[start:end].to(self.device_)
                mb_delta_t = delta_t[start:end].to(self.device_)
                mb_numeric = numeric_normalized[start:end].to(self.device_)
                mb_vectors = {f: t[start:end].to(self.device_) for f, t in vector_tensors.items()}
                mb_cat = {f: t[start:end].to(self.device_) for f, t in cat_idx_tensors.items()}
                mb_encoded = self._encode(
                    cat_embeddings=cat_embeddings,
                    cat_idx_tensors=mb_cat,
                    numeric_projection=numeric_projection,
                    numeric_tensor=mb_numeric,
                    vector_tensors=mb_vectors,
                    vector_projections=vector_projections,
                    delta_t=mb_delta_t,
                    valid_mask=mb_valid,
                    core=core,
                )
                encoded_parts.append(mb_encoded.cpu())

                done_batches = batch_idx + 1
                if done_batches % 25 == 0 or done_batches == total_inference_batches:
                    logger.debug(
                        "%s: final encode %d/%d (%.1f%%)",
                        progress_prefix,
                        done_batches,
                        total_inference_batches,
                        100.0 * done_batches / max(total_inference_batches, 1),
                    )
            encoded_cpu = torch.cat(encoded_parts, dim=0)

        non_finite_mask = ~torch.isfinite(encoded_cpu)
        if bool(non_finite_mask.any().item()):
            logger.warning(
                "%s: replacing %d non-finite encoded values",
                progress_prefix,
                int(non_finite_mask.sum().item()),
            )
            encoded_cpu = torch.nan_to_num(encoded_cpu, nan=0.0, posinf=0.0, neginf=0.0)

        self.loss_ = last_loss

        event_embeddings: dict[str, list[float]] = {}
        customer_records: list[dict[str, object]] = []

        for batch_idx, sequence_id in enumerate(sequence_ids):
            length = sequence_lengths[batch_idx]
            valid_states = encoded_cpu[batch_idx, :length, :]
            for seq_idx in range(length):
                event_embeddings[event_ids[batch_idx][seq_idx]] = valid_states[seq_idx].tolist()

            last_ts = sequence_last_ts[batch_idx]
            customer_records.append(
                {
                    self.sequence_id_field: sequence_id,
                    # Keep both "last" and "mean" summaries because some heads
                    # perform better with recency while others prefer stability.
                    "core_last_vector": valid_states[-1].tolist(),
                    "core_mean_vector": valid_states.mean(dim=0).tolist(),
                    "core_event_count": float(length),
                    # Point-in-time stamp: this state is valid as of the last
                    # event included, under this backbone version.
                    "core_as_of_ts": last_ts.isoformat() if last_ts is not None else "",
                    "backbone_version": self.config.backbone_version,
                }
            )

        logger.info("%s: completed fit loss=%.6f", progress_prefix, last_loss)

        return TemporalCoreOutputs(
            sequence_id_field=self.sequence_id_field,
            event_id_field=self.event_id_field,
            event_embeddings=event_embeddings,
            customer_records=customer_records,
            records=records,
        )

    @property
    def trained_core(self) -> EntityCore | None:
        """The EntityCore trained during the last ``fit_transform`` call, or None."""
        return self._trained_core

    def save_pretrained(self, path, backbone_version: str | None = None) -> str:
        """Persist the trained backbone (weights + rebuild config) to *path*.

        The version tag resolution order is: explicit ``backbone_version``
        argument, then ``config.backbone_version`` when it was set to a
        non-default value, otherwise a content hash of the actual weights
        (so the version can never drift from the parameters).  Returns the
        version written into the checkpoint.
        """

        if self._trained_core is None:
            raise RuntimeError("No trained backbone to save; call fit_transform first")
        version = (
            backbone_version
            or (None if self.config.backbone_version == "v0" else self.config.backbone_version)
        )
        return save_core_checkpoint(
            self._trained_core,
            path,
            backbone_version=version,
            extra={
                "sequence_id_field": self.sequence_id_field,
                "event_id_field": self.event_id_field,
                "timestamp_field": self.timestamp_field,
                "hidden_dim": self.config.hidden_dim,
                "sequence_backend": self.config.sequence_backend,
            },
        )

    @classmethod
    def load_pretrained(cls, path, device: str = "auto") -> "TemporalCoreModel":
        """Reload a backbone saved by :meth:`save_pretrained`.

        The returned model has ``trained_core`` set and is ready to be used
        as the ``pretrained_core`` for downstream supervised heads.  Field
        mappings are restored from checkpoint metadata when present.
        """

        from pathlib import Path as _Path

        core, version, extra = load_core_checkpoint(_Path(path), map_location="cpu")
        config = TemporalCoreConfig(
            hidden_dim=int(extra.get("hidden_dim", 128)),
            sequence_backend=str(extra.get("sequence_backend", "samba")),
            backbone_version=version,
            device=device,
        )
        model = cls(
            sequence_id_field=str(extra.get("sequence_id_field", "customer_id")),
            event_id_field=str(extra.get("event_id_field", "event_id")),
            timestamp_field=str(extra.get("timestamp_field", "event_ts")),
            categorical_fields=[],
            numeric_fields=[],
            vector_fields=[],
            config=config,
        )
        model._trained_core = core.to(model.device_)
        return model

    def _encode(
        self,
        cat_embeddings: nn.ModuleDict,
        cat_idx_tensors: dict[str, Tensor],
        numeric_projection: nn.Linear,
        numeric_tensor: Tensor,
        vector_tensors: dict[str, Tensor],
        vector_projections: nn.ModuleDict,
        delta_t: Tensor,
        valid_mask: Tensor,
        core: EntityCore,
    ) -> Tensor:
        """Encode one mini-batch with element-wise token construction.

        Categorical embeddings, a numeric projection, and every vector
        projection are independently mapped to ``hidden_dim`` then summed,
        following the contract::

            E_event = Σ E_cat + E_num + Σ E_product
        """

        batch_size, seq_len = valid_mask.shape
        cat_sum = torch.zeros(
            batch_size,
            seq_len,
            self.config.hidden_dim,
            dtype=torch.float32,
            device=numeric_tensor.device,
        )
        for field, embedding in cat_embeddings.items():
            cat_sum = cat_sum + embedding(cat_idx_tensors[field])

        hidden_states = cat_sum + numeric_projection(numeric_tensor)
        for field, tensor in vector_tensors.items():
            if field in vector_projections:
                hidden_states = hidden_states + vector_projections[field](tensor)
        hidden_states = hidden_states * valid_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        attention_mask = _build_causal_attention_mask(valid_mask)
        encoded = core(hidden_states, delta_t, attention_mask=attention_mask).encoded_states
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        return encoded * valid_mask.unsqueeze(-1).to(dtype=encoded.dtype)

    def _group_records(
        self,
        records: list[dict[str, object]],
        show_progress: bool = False,
        progress_label: str | None = None,
        input_is_time_sorted: bool = False,
    ) -> list[tuple[str, list[dict[str, object]]]]:
        """Group raw rows by sequence id and optionally time-sort each sequence."""

        label = progress_label or "Temporal core"
        total_rows = len(records)
        logger.info("%s: group rows total=%d", label, total_rows)

        grouped: dict[str, list[dict[str, object]]] = {}
        group_iter = (
            tqdm(
                records,
                desc=f"{label}: group rows",
                unit="row",
                leave=False,
            )
            if show_progress
            else records
        )
        for row_idx, row in enumerate(group_iter, start=1):
            grouped.setdefault(str(row[self.sequence_id_field]), []).append(row)
            if row_idx % 1_000_000 == 0 or row_idx == total_rows:
                logger.debug(
                    "%s: group rows %d/%d (%.1f%%)",
                    label,
                    row_idx,
                    total_rows,
                    100.0 * row_idx / max(total_rows, 1),
                )

        total_sequences = len(grouped)
        logger.info("%s: grouped sequences=%d", label, total_sequences)

        if input_is_time_sorted:
            logger.info("%s: skip sequence sort (input marked time-sorted)", label)
            return sorted(grouped.items(), key=lambda pair: pair[0])

        sort_iter = grouped.items()
        if show_progress:
            sort_iter = tqdm(
                sort_iter,
                total=len(grouped),
                desc=f"{label}: sort sequences",
                unit="seq",
                leave=False,
            )

        ordered: list[tuple[str, list[dict[str, object]]]] = []
        for seq_idx, (sequence_id, rows) in enumerate(sort_iter, start=1):
            rows.sort(
                key=lambda row: (
                    str(row[self.timestamp_field]),
                    str(row[self.event_id_field]),
                )
            )
            ordered.append((sequence_id, rows))
            if seq_idx % 10_000 == 0 or seq_idx == total_sequences:
                logger.debug(
                    "%s: sort sequences %d/%d (%.1f%%)",
                    label,
                    seq_idx,
                    total_sequences,
                    100.0 * seq_idx / max(total_sequences, 1),
                )
        ordered.sort(key=lambda pair: pair[0])
        return ordered


def create_temporal_core_model(
    sequence_id_field: str,
    event_id_field: str,
    timestamp_field: str,
    categorical_fields: list[str],
    numeric_fields: list[str],
    vector_fields: list[str] | None = None,
    hidden_dim: int = 128,
    epochs: int = 120,
    seed: int = 17,
    learning_rate: float = 1e-2,
    device: str = "auto",
    sequence_backend: str = "samba",
    backbone_version: str = "v0",
    train_batch_size: int = 512,
    input_is_time_sorted: bool = False,
    show_progress: bool = False,
    progress_label: str | None = None,
    payload_schema: PayloadSchema | None = None,
    vector_lookups: dict[str, dict[str, list[float]]] | None = None,
) -> TemporalCoreModel:
    """Factory mirroring other model creators with a configured core model."""
    return TemporalCoreModel(
        sequence_id_field=sequence_id_field,
        event_id_field=event_id_field,
        timestamp_field=timestamp_field,
        categorical_fields=categorical_fields,
        numeric_fields=numeric_fields,
        vector_fields=vector_fields,
        config=TemporalCoreConfig(
            hidden_dim=hidden_dim,
            epochs=epochs,
            seed=seed,
            learning_rate=learning_rate,
            device=device,
            sequence_backend=sequence_backend,
            backbone_version=backbone_version,
            payload_schema=payload_schema,
            vector_lookups=vector_lookups,
            train_batch_size=train_batch_size,
            input_is_time_sorted=input_is_time_sorted,
            show_progress=show_progress,
            progress_label=progress_label,
        ),
    )


def to_state_records(
    outputs: TemporalCoreOutputs,
    vector_field: str = "core_last_vector",
) -> list[StateRecord]:
    """Convert temporal-core customer summaries into point-in-time StateRecords.

    Each record carries the per-customer ``core_as_of_ts`` (time of the last
    event encoded) and the backbone version, so it can be written to a
    PointInTimeStateStore and later queried as-of any future prediction date.
    Customers without a usable timestamp are skipped.
    """

    records: list[StateRecord] = []
    for row in outputs.customer_records:
        as_of_raw = str(row.get("core_as_of_ts", ""))
        if not as_of_raw:
            continue
        records.append(
            StateRecord(
                entity_id=str(row[outputs.sequence_id_field]),
                as_of_ts=datetime.fromisoformat(as_of_raw),
                vector=list(row[vector_field]),  # type: ignore[arg-type]
                backbone_version=str(row.get("backbone_version", "v0")),
            )
        )
    return records


def _build_causal_attention_mask(valid_mask: Tensor) -> Tensor:
    """Build a boolean mask that enforces both padding and causal constraints."""

    seq_len = valid_mask.size(1)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=valid_mask.device))
    query_mask = valid_mask[:, None, :, None]
    key_mask = valid_mask[:, None, None, :]
    return query_mask & key_mask & causal.unsqueeze(0).unsqueeze(0)


def _normalize_masked(values: Tensor, valid_mask: Tensor) -> Tensor:
    """Normalize feature tensor using only valid (non-padding) positions."""

    if values.size(-1) == 0:
        return values

    flat_values = values[valid_mask]
    if flat_values.numel() == 0:
        return values

    means = flat_values.mean(dim=0, keepdim=True)
    stds = flat_values.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    normalized = values.clone()
    normalized[valid_mask] = (flat_values - means) / stds
    normalized[~valid_mask] = 0.0
    return normalized


def _parse_iso_timestamp(value: object) -> datetime:
    """Parse an ISO timestamp and normalize trailing ``Z`` into UTC offset."""

    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _resolve_train_batch_size(
    requested: int,
    total_sequences: int,
    max_seq_len: int,
    device: torch.device,
    progress_label: str,
) -> int:
    """Resolve a safe train batch size, especially for CUDA attention memory.

    Attention memory grows approximately with ``seq_len^2``; this helper caps
    batch sizes on CUDA for longer sequences to reduce OOM risk.
    """

    batch = max(1, min(int(requested), max(1, total_sequences)))
    if device.type != "cuda":
        return batch

    # Attention memory scales roughly with sequence length squared.
    if max_seq_len >= 128:
        safe_cap = 64
    elif max_seq_len >= 96:
        safe_cap = 128
    elif max_seq_len >= 64:
        safe_cap = 192
    else:
        safe_cap = 256

    resolved = min(batch, safe_cap)
    if resolved < batch:
        logger.info(
            "%s: reducing train_batch_size from %d to %d for seq_len=%d on CUDA",
            progress_label,
            batch,
            resolved,
            max_seq_len,
        )
    return resolved
