"""Temporal core training: fit_transform, sequence grouping, and encoding."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from looking_glass.entity_core import EntityCore
from looking_glass.heads import ClassificationHead, RegressionHead
from looking_glass.interfaces import TaskHeadBase
from looking_glass.sequence import SequenceEngine
from looking_glass.temporal import TemporalStack
from looking_glass.temporal_core_utils import (
    _build_causal_attention_mask,
    _normalize_masked,
    _parse_iso_timestamp,
    _resolve_train_batch_size,
    logger,
)
from looking_glass.tokenizer import collect_payload_vocabularies, parse_event_payload


class _PassThroughTaskHead(TaskHeadBase):
    """No-op task head used to expose encoded sequence states directly."""

    def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
        return {}


@dataclass(frozen=True)
class TemporalCoreOutputs:
    """Structured artifacts produced by :class:`TemporalCoreModel`."""

    sequence_id_field: str
    event_id_field: str
    event_embeddings: dict[str, list[float]]
    customer_records: list[dict[str, object]]
    records: list[dict[str, object]]


class _TemporalCoreTrainingMixin:
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
            len(self.config.payload_schema.numeric_fields)
            if self.config.payload_schema is not None
            else len(self.numeric_fields),
            dtype=torch.float32,
            device=_cpu,
        )

        vector_dims: dict[str, int] = {}
        if self.config.payload_schema is not None and self.config.vector_lookups is not None:
            sample = next(
                (v for table in self.config.vector_lookups.values() for v in table.values() if v),
                None,
            )
            prod_dim = len(sample) if sample else 0
            if prod_dim > 0:
                vector_dims["combined_vector"] = prod_dim
        else:
            for field in self.vector_fields:
                sample = next(
                    (row.get(field) for row in records if isinstance(row.get(field), list)), None
                )
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
                            default_vector_width=len(
                                next(iter(vector_tensors.values()), torch.zeros(1))
                            ),
                        )
                    else:
                        parsed = parse_event_payload({}, self.config.payload_schema)
                    for fld in cat_idx_tensors:
                        if fld == "event_type":
                            cat_idx_tensors[fld][batch_idx, seq_idx] = cat_vocab[fld].get(
                                str(row.get("event_type", "")), 0
                            )
                        else:
                            cat_idx_tensors[fld][batch_idx, seq_idx] = cat_vocab[fld].get(
                                parsed.cat_values.get(fld, ""), 0
                            )
                    for field_idx, fld in enumerate(self.config.payload_schema.numeric_fields):
                        numeric_tensor[batch_idx, seq_idx, field_idx] = parsed.num_values.get(
                            fld, 0.0
                        )
                    if "combined_vector" in vector_tensors:
                        cv = parsed.combined_vector
                        if len(cv) == vector_tensors["combined_vector"].size(-1):
                            vector_tensors["combined_vector"][batch_idx, seq_idx] = torch.tensor(
                                cv, dtype=torch.float32, device=_cpu
                            )
                else:
                    for field in self.categorical_fields:
                        cat_idx_tensors[field][batch_idx, seq_idx] = cat_vocab[field][
                            str(row.get(field, ""))
                        ]
                    for field_idx, field in enumerate(self.numeric_fields):
                        numeric_tensor[batch_idx, seq_idx, field_idx] = float(
                            row.get(field, 0.0) or 0.0
                        )
                    for field, tensor in vector_tensors.items():
                        dim = tensor.size(-1)
                        value = row.get(field)
                        if isinstance(value, list) and len(value) == dim:
                            tensor[batch_idx, seq_idx] = torch.tensor(
                                value, dtype=torch.float32, device=_cpu
                            )

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
        numeric_normalized = (
            _normalize_masked(numeric_tensor, valid_mask)
            if numeric_tensor.size(-1) > 0
            else numeric_tensor
        )
        numeric_targets = (
            _normalize_masked(numeric_tensor, valid_mask) if numeric_tensor.size(-1) > 0 else None
        )

        cat_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(max(1, len(vocab)), self.config.hidden_dim)
                for field, vocab in cat_vocab.items()
            }
        ).to(self.device_)
        numeric_projection = nn.Linear(
            int(max(numeric_normalized.size(-1), 1)), self.config.hidden_dim
        ).to(self.device_)
        vector_projections = (
            nn.ModuleDict(
                {
                    field: nn.Linear(dim, self.config.hidden_dim).to(self.device_)
                    for field, tensor in vector_tensors.items()
                    if (dim := tensor.size(-1)) > 0
                }
            )
            if vector_tensors
            else nn.ModuleDict()
        )
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
        combined_targets = (
            torch.cat(_reg_targets, dim=-1)
            if len(_reg_targets) > 1
            else (
                _reg_targets[0]
                if _reg_targets
                else torch.zeros(batch_size, max_seq_len, 1, device=_cpu)
            )
        )

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
                    mb_vectors = {
                        f: t[start:end].to(self.device_) for f, t in vector_tensors.items()
                    }
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
                    logger.debug(
                        "%s  epoch %d/%d  loss=%.6f",
                        _label,
                        ep_idx + 1,
                        self.config.epochs,
                        last_loss,
                    )
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
            total_inference_batches = (batch_size + train_batch_size - 1) // train_batch_size
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
