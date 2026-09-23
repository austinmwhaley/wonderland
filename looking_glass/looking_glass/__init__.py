"""Public package surface for the looking_glass project.

looking_glass is a reusable toolkit for building neural entity models.
It does NOT contain application-specific logic like data generation or
smoke test workflows—those live in scripts/.

Users import from this package to build their own applications.
"""

from .adaptation import AdapterFusion, QDoRAConfig, QDoRALinear
from .checkpoint import (
    build_core_from_config,
    core_config_from_core,
    derive_backbone_version,
    load_core_checkpoint,
    save_core_checkpoint,
)
from .embeddings import (
	attach_vector_feature,
	create_embedding_model,
	Embeddings,
	EmbeddingModel,
	EmbeddingModelConfig,
	load_vectors_from_lancedb,
	load_records_from_sqlite,
	save_embeddings_to_lancedb,
)
from .entity_core import EntityCore, EntityCoreOutput
from .heads import BusinessHead, ClassificationHead, MultiTaskBusinessHead, RegressionHead
from .interfaces import SequenceEngineBase, TaskHeadBase, TemporalEncoderBase
from .robustness import RejectionSampler, SamplingReport
from .sequence import (
	FlashSelfAttention,
	get_sequence_backend_name,
	get_sequence_implementation_name,
	get_supported_sequence_backends,
	MambaBlock,
	Mamba2SequenceLayer,
	SambaBlock,
	SequenceEngine,
	UniversalTransformerBlock,
)
from .supervised import (
	create_supervised_model,
	PredictionResults,
	SupervisedModel,
	SupervisedModelConfig,
	validate_classification_success,
	validate_embeddings,
	validate_regression_success,
	validate_prediction_report,
	ValidationReport,
)
from .temporal_core import (
	create_temporal_core_model,
	TemporalCoreConfig,
	TemporalCoreModel,
	TemporalCoreOutputs,
	to_state_records,
)
from .temporal import (
	NeuralODELatentDrift,
	ODEDriftField,
	TAPE,
	TemporalStack,
	Time2Vec,
	cumulative_time,
	log_scaled_delta_t,
)
from .tokenizer import (
	collect_payload_vocabularies,
	parse_event_payload,
	ParsedPayload,
	PayloadSchema,
)
from .metrics import (
	LiftBucket,
	lift_table,
	pr_auc,
	ranking_metrics,
	roc_auc,
)
from .outcomes import (
	attach_state_vectors,
	assert_no_leakage,
	build_outcomes,
	FEATURE_FIELDS,
	LabelSpec,
	OutcomeFrame,
)
from .pretrain import (
	extract_final_states,
	incremental_forward,
	incremental_state_update,
	save_history_to_store,
	wrap_core_with_qdora,
)
from .state_store import (
	InMemoryStateStore,
	LanceDBStateStore,
	PointInTimeStateStore,
	StateRecord,
)
from .baselines import (
	baseline_implementation_name,
	BaselineResult,
	compare,
	GBTBaseline,
)

__all__ = [
	"AdapterFusion",
	"attach_state_vectors",
	"attach_vector_feature",
	"assert_no_leakage",
	"baseline_implementation_name",
	"BaselineResult",
	"build_core_from_config",
	"build_outcomes",
	"BusinessHead",
	"ClassificationHead",
	"collect_payload_vocabularies",
	"compare",
	"core_config_from_core",
	"create_embedding_model",
	"create_supervised_model",
	"create_temporal_core_model",
	"cumulative_time",
	"derive_backbone_version",
	"Embeddings",
	"EmbeddingModel",
	"EmbeddingModelConfig",
	"EntityCore",
	"EntityCoreOutput",
	"extract_final_states",
	"FEATURE_FIELDS",
	"FlashSelfAttention",
	"get_sequence_backend_name",
	"get_sequence_implementation_name",
	"get_supported_sequence_backends",
	"GBTBaseline",
	"incremental_forward",
	"incremental_state_update",
	"InMemoryStateStore",
	"LabelSpec",
	"LanceDBStateStore",
	"LiftBucket",
	"lift_table",
	"load_core_checkpoint",
	"load_records_from_sqlite",
	"load_vectors_from_lancedb",
	"log_scaled_delta_t",
	"Mamba2SequenceLayer",
	"MambaBlock",
	"MultiTaskBusinessHead",
	"NeuralODELatentDrift",
	"ODEDriftField",
	"OutcomeFrame",
	"parse_event_payload",
	"ParsedPayload",
	"PayloadSchema",
	"PointInTimeStateStore",
	"pr_auc",
	"PredictionResults",
	"QDoRAConfig",
	"QDoRALinear",
	"ranking_metrics",
	"RegressionHead",
	"RejectionSampler",
	"roc_auc",
	"SambaBlock",
	"SamplingReport",
	"save_core_checkpoint",
	"save_embeddings_to_lancedb",
	"save_history_to_store",
	"SequenceEngine",
	"SequenceEngineBase",
	"StateRecord",
	"SupervisedModel",
	"SupervisedModelConfig",
	"TAPE",
	"TaskHeadBase",
	"TemporalCoreConfig",
	"TemporalCoreModel",
	"TemporalCoreOutputs",
	"TemporalEncoderBase",
	"TemporalStack",
	"Time2Vec",
	"to_state_records",
	"UniversalTransformerBlock",
	"validate_classification_success",
	"validate_embeddings",
	"validate_prediction_report",
	"validate_regression_success",
	"ValidationReport",
	"wrap_core_with_qdora",
]
