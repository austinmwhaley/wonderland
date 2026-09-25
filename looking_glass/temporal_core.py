# ruff: noqa: F401
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

from looking_glass.embeddings import _best_available_device
from looking_glass.checkpoint import load_core_checkpoint, save_core_checkpoint
from looking_glass.entity_core import EntityCore
from looking_glass.heads import ClassificationHead, RegressionHead
from looking_glass.interfaces import TaskHeadBase
from looking_glass.sequence import SequenceEngine
from looking_glass.state_store import StateRecord
from looking_glass.tokenizer import PayloadSchema, parse_event_payload, collect_payload_vocabularies
from looking_glass.temporal import TemporalStack
from looking_glass.temporal_core_training import (
    TemporalCoreOutputs,
    _TemporalCoreTrainingMixin,
    _PassThroughTaskHead,
)
from looking_glass.temporal_core_training import logger
from looking_glass.temporal_core_utils import (
    _build_causal_attention_mask,
    _normalize_masked,
    _parse_iso_timestamp,
    _resolve_train_batch_size,
)


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


class _TemporalCoreRecordsMixin:
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


class _TemporalCorePersistenceMixin:
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
        version = backbone_version or (
            None if self.config.backbone_version == "v0" else self.config.backbone_version
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


class TemporalCoreModel(
    _TemporalCoreTrainingMixin, _TemporalCoreRecordsMixin, _TemporalCorePersistenceMixin
):
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
