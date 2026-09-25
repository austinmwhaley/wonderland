# ruff: noqa: F401
# Standard library imports used for timestamp math and path bootstrapping.
from contextlib import contextmanager
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

import torch
from tqdm import tqdm

# Ensure the repository root is importable when this file is run directly.
# This makes `python scripts/smoke_test.py` work without installing the package.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from looking_glass import (
    attach_vector_feature,
    create_embedding_model,
    create_supervised_model,
    create_temporal_core_model,
    get_sequence_backend_name,
    get_sequence_implementation_name,
    load_vectors_from_lancedb,
    load_records_from_sqlite,
    QDoRAConfig,
    save_embeddings_to_lancedb,
    validate_classification_success,
    validate_embeddings,
    validate_regression_success,
    validate_prediction_report,
    wrap_core_with_qdora,
    GBTBaseline,
    ranking_metrics,
)

from looking_glass.scripts.smoke_config import (
    SQLITE_PATH,
    LANCEDB_DIR,
    HIDDEN_DIM,
    SEED,
    DEFAULT_PRODUCT_EPOCHS,
    DEFAULT_CUSTOMER_EPOCHS,
    DEFAULT_CORE_EPOCHS,
    DEFAULT_OUTCOME_EPOCHS,
    _int_env,
    _optional_int_env,
    _float_env,
    _optional_float_env,
    PRODUCT_EPOCHS,
    CUSTOMER_EPOCHS,
    CORE_EPOCHS,
    CORE_TRAIN_BATCH_SIZE,
    OUTCOME_EPOCHS,
    SMOKE_EVENT_LIMIT,
    OUTCOME_HISTORY_DAYS,
    OUTCOME_LABEL_DAYS,
    OUTCOME_MIN_PAST_ORDERS,
    OUTCOME_ACTIVE_LOOKBACK_DAYS,
    OUTCOME_MIN_RECENT_ORDERS,
    OUTCOME_TARGET_MODE,
    OUTCOME_MIN_CHURN_POSITIVE_RATE,
    OUTCOME_MAX_CHURN_POSITIVE_RATE,
    OUTCOME_MIN_NONZERO_LTV_RATE,
    SMOKE_SEQUENCE_BACKEND,
    PRODUCT_EMBED_MIN_NORM_STD,
    CUSTOMER_EMBED_MIN_NORM_STD,
    EVENT_EMBED_MIN_NORM_STD,
    EMBED_MAX_MEAN_ABS_COSINE,
    CHURN_MIN_F1,
    CHURN_MIN_RECALL,
    CHURN_MIN_PRECISION,
    LTV_MIN_R2,
    PRODUCT_GOOD_MIN_NORM_STD,
    PRODUCT_GOOD_MAX_MEAN_ABS_COSINE,
    CUSTOMER_GOOD_MIN_NORM_STD,
    CUSTOMER_GOOD_MAX_MEAN_ABS_COSINE,
    EVENT_GOOD_MIN_NORM_STD,
    EVENT_GOOD_MAX_MEAN_ABS_COSINE,
    CORE_GOOD_MAX_LOSS,
    CORE_GOOD_MIN_CUSTOMERS,
    CORE_GOOD_MIN_MEAN_EVENTS,
    CHURN_GOOD_MIN_ACCURACY,
    CHURN_GOOD_MIN_PRECISION,
    CHURN_GOOD_MIN_RECALL,
    CHURN_GOOD_MIN_F1,
    LTV_GOOD_MIN_R2,
    LTV_GOOD_MAX_RMSE,
    CUSTOMER_DEVICE,
    CORE_DEVICE,
    GOOD_QUALITY_MODE,
    REDUCED_SMOKE_BUDGET,
    ENFORCE_GOOD_QUALITY,
)
from looking_glass.scripts.smoke_support import (
    _StageProgress,
    _resolve_device_label,
    _build_customer_outcome_records,
    _build_event_outcome_records,
    _build_customer_dimension_records,
    _attach_customer_lookup_fields,
    _assert_ok,
    _embedding_good_quality,
    _temporal_core_good_quality,
    _LOG_PATH,
    _setup_logging,
)
from looking_glass.scripts.smoke_pipeline import (
    run_customer_embedding_stage,
    run_event_enrichment_stage,
    run_outcome_stage,
    run_product_embedding_stage,
    run_temporal_core_stage,
)

if __name__ == "__main__":
    logger = _setup_logging()
    sequence_backend = get_sequence_backend_name(SMOKE_SEQUENCE_BACKEND)
    sequence_implementation = get_sequence_implementation_name()
    logger.info("Log: %s", _LOG_PATH.resolve())
    logger.info(
        (
            "Smoke config: product_epochs=%d customer_epochs=%d core_epochs=%d "
            "outcome_epochs=%d core_batch=%d event_limit=%s "
            "history_days=%d label_days=%d min_past_orders=%d "
            "active_lookback_days=%d min_recent_orders=%d target_mode=%s "
            "good_quality_mode=%s good_quality_enforced=%s sequence_backend=%s sequence_impl=%s "
            "churn_rate_range=[%.2f, %.2f] min_nonzero_ltv_rate=%.2f "
            "good_gate: core_max_loss=%.2f core_min_customers=%d core_min_mean_events=%.2f "
            "churn_good(f1>=%.2f,precision>=%.2f,recall>=%.2f,accuracy>=%.2f) "
            "ltv_good(r2>=%.2f,rmse<=%s)"
        ),
        PRODUCT_EPOCHS,
        CUSTOMER_EPOCHS,
        CORE_EPOCHS,
        OUTCOME_EPOCHS,
        CORE_TRAIN_BATCH_SIZE,
        "all" if SMOKE_EVENT_LIMIT is None else str(SMOKE_EVENT_LIMIT),
        OUTCOME_HISTORY_DAYS,
        OUTCOME_LABEL_DAYS,
        OUTCOME_MIN_PAST_ORDERS,
        OUTCOME_ACTIVE_LOOKBACK_DAYS,
        OUTCOME_MIN_RECENT_ORDERS,
        OUTCOME_TARGET_MODE,
        GOOD_QUALITY_MODE,
        ENFORCE_GOOD_QUALITY,
        sequence_backend,
        sequence_implementation,
        OUTCOME_MIN_CHURN_POSITIVE_RATE,
        OUTCOME_MAX_CHURN_POSITIVE_RATE,
        OUTCOME_MIN_NONZERO_LTV_RATE,
        CORE_GOOD_MAX_LOSS,
        CORE_GOOD_MIN_CUSTOMERS,
        CORE_GOOD_MIN_MEAN_EVENTS,
        CHURN_GOOD_MIN_F1,
        CHURN_GOOD_MIN_PRECISION,
        CHURN_GOOD_MIN_RECALL,
        CHURN_GOOD_MIN_ACCURACY,
        LTV_GOOD_MIN_R2,
        "none" if LTV_GOOD_MAX_RMSE is None else f"{LTV_GOOD_MAX_RMSE:.2f}",
    )

    run_progress = _StageProgress(total_stages=8)
    # ---------------------------------------------------------------------
    # Stage 0: (Optional) synthetic data generation/bootstrap.
    # ---------------------------------------------------------------------
    # Optional bootstrap: regenerate source data if needed.
    # Keep this off for normal iterative runs because it's expensive.
    # generate_data_main()

    product_device_label = _resolve_device_label("auto")
    customer_device_label = _resolve_device_label(CUSTOMER_DEVICE)
    core_device_label = _resolve_device_label(CORE_DEVICE)

    try:
        # -----------------------------------------------------------------
        # Stage 1: Product embedding model (entity silo #1).
        # -----------------------------------------------------------------
        with run_progress.stage(f"product embedding model ({product_device_label})"):
            model, product_rows, product_validation, product_quality = run_product_embedding_stage(
                sequence_backend
            )

        # -----------------------------------------------------------------
        # Stage 2: Customer embedding model (entity silo #2).
        # -----------------------------------------------------------------
        with run_progress.stage(f"customer embedding model ({customer_device_label})"):
            (
                customer_model,
                customer_rows,
                customer_validation,
                customer_quality,
                customer_embeddings,
            ) = run_customer_embedding_stage(sequence_backend)

        # -----------------------------------------------------------------
        # Stage 3: Load event stream + attach siloed vectors.
        # -----------------------------------------------------------------
        with run_progress.stage("event enrichment"):
            customer_event_records, outcome_cutoff_ts = run_event_enrichment_stage(
                customer_embeddings, logger
            )

        # -----------------------------------------------------------------
        # Stage 4: Temporal core model over event sequences.
        # -----------------------------------------------------------------
        with run_progress.stage(f"temporal core model ({core_device_label})"):
            (
                core_model,
                core_outputs,
                core_validation,
                core_embedding_quality,
                core_quality,
                event_rows,
                qdora_core,
            ) = run_temporal_core_stage(customer_event_records, sequence_backend, logger)

        # -----------------------------------------------------------------
        # Stage 5: Supervised outcomes from transactional history.
        # -----------------------------------------------------------------
        with run_progress.stage("build supervised outcomes"):
            outcome_records = run_outcome_stage(
                core_outputs, outcome_cutoff_ts, customer_embeddings, logger
            )

        # -----------------------------------------------------------------
        # Stage 6: Churn classification head (75/25 train/validation split).
        # -----------------------------------------------------------------
        with run_progress.stage(f"churn head ({core_device_label})"):
            churn_model = create_supervised_model(
                task="classification",
                id_field="customer_id",
                target_field="churn_label",
                categorical_fields=[],
                numeric_fields=[
                    "order_count",
                    "completed_count",
                    "cancelled_count",
                    "refunded_count",
                    "return_count",
                    "total_spend",
                    "avg_order_total",
                    "max_order_total",
                    "recency_days",
                    "recent_order_count_30d",
                    "recent_spend_30d",
                    "active_order_count",
                    "return_rate",
                    "cancel_rate",
                    "complete_rate",
                    "product_event_ratio",
                    "core_event_count",
                ],
                vector_fields=["customer_vector", "core_last_vector"],
                hidden_dim=HIDDEN_DIM,
                epochs=OUTCOME_EPOCHS,
                seed=SEED,
                learning_rate=1e-3,
                validation_fraction=0.25,
                device=CORE_DEVICE,
                sequence_backend=sequence_backend,
                show_progress=True,
                progress_label="Churn head epochs",
                pretrained_core=qdora_core,
            )
            churn_results = churn_model.fit_predict(outcome_records)
            churn_validation = validate_prediction_report(churn_results.report)
            _assert_ok("churn_model", churn_validation)

            # Require churn to do more than predict the majority class.
            churn_quality = validate_classification_success(
                churn_results.report,
                min_f1=CHURN_MIN_F1,
                min_recall=CHURN_MIN_RECALL,
                min_precision=CHURN_MIN_PRECISION,
            )
            _assert_ok("churn_quality", churn_quality)
            churn_good_quality = validate_classification_success(
                churn_results.report,
                min_accuracy=CHURN_GOOD_MIN_ACCURACY,
                min_precision=CHURN_GOOD_MIN_PRECISION,
                min_recall=CHURN_GOOD_MIN_RECALL,
                min_f1=CHURN_GOOD_MIN_F1,
            )
            _assert_ok("churn_good_quality", churn_good_quality, enforce=ENFORCE_GOOD_QUALITY)

            churn_scores = [
                float(churn_results.predictions[str(r["customer_id"])])
                for r in outcome_records
                if str(r["customer_id"]) in churn_results.predictions
            ]
            churn_labels = [
                float(r.get("churn_label", 0.0) or 0.0)
                for r in outcome_records
                if str(r["customer_id"]) in churn_results.predictions
            ]
            churn_rank = ranking_metrics(churn_scores, churn_labels)
            churn_results.report.metrics.update(churn_rank)
            logger.info("Churn ranking: %s", {k: round(float(v), 4) for k, v in churn_rank.items()})

            churn_baseline = GBTBaseline(
                task="classification",
                id_field="customer_id",
                target_field="churn_label",
                feature_fields=[
                    "order_count",
                    "completed_count",
                    "cancelled_count",
                    "refunded_count",
                    "return_count",
                    "total_spend",
                    "avg_order_total",
                    "max_order_total",
                    "recency_days",
                    "recent_order_count_30d",
                    "recent_spend_30d",
                    "active_order_count",
                    "return_rate",
                    "cancel_rate",
                    "complete_rate",
                    "core_event_count",
                ],
                seed=SEED,
            )
            churn_baseline_result = churn_baseline.fit_predict(outcome_records)
            logger.info(
                "Churn baseline (%s): f1=%.4f auc=%.4f",
                churn_baseline_result.implementation,
                churn_baseline_result.metrics.get("f1", 0),
                churn_baseline_result.metrics.get("roc_auc", 0),
            )

        # -----------------------------------------------------------------
        # Stage 7: LTV regression head (75/25 train/validation split).
        # -----------------------------------------------------------------
        with run_progress.stage(f"ltv head ({core_device_label})"):
            ltv_model = create_supervised_model(
                task="regression",
                id_field="customer_id",
                target_field="ltv_value",
                categorical_fields=[],
                numeric_fields=[
                    "order_count",
                    "completed_count",
                    "cancelled_count",
                    "refunded_count",
                    "return_count",
                    "total_spend",
                    "avg_order_total",
                    "max_order_total",
                    "recency_days",
                    "recent_order_count_30d",
                    "recent_spend_30d",
                    "active_order_count",
                    "return_rate",
                    "cancel_rate",
                    "complete_rate",
                    "product_event_ratio",
                    "core_event_count",
                ],
                vector_fields=["customer_vector", "core_mean_vector"],
                hidden_dim=HIDDEN_DIM,
                epochs=OUTCOME_EPOCHS,
                seed=SEED,
                learning_rate=1e-3,
                validation_fraction=0.25,
                device=CORE_DEVICE,
                sequence_backend=sequence_backend,
                show_progress=True,
                progress_label="LTV head epochs",
                pretrained_core=qdora_core,
            )
            ltv_results = ltv_model.fit_predict(outcome_records)
            ltv_validation = validate_prediction_report(ltv_results.report)
            _assert_ok("ltv_model", ltv_validation)
            ltv_quality = validate_regression_success(
                ltv_results.report,
                min_r2=LTV_MIN_R2,
            )
            _assert_ok("ltv_quality", ltv_quality)
            ltv_good_quality = validate_regression_success(
                ltv_results.report,
                min_r2=LTV_GOOD_MIN_R2,
                max_rmse=LTV_GOOD_MAX_RMSE,
            )
            _assert_ok("ltv_good_quality", ltv_good_quality, enforce=ENFORCE_GOOD_QUALITY)

            ltv_baseline = GBTBaseline(
                task="regression",
                id_field="customer_id",
                target_field="ltv_value",
                feature_fields=[
                    "order_count",
                    "completed_count",
                    "total_spend",
                    "avg_order_total",
                    "max_order_total",
                    "recency_days",
                    "recent_order_count_30d",
                    "active_order_count",
                    "return_rate",
                    "complete_rate",
                    "core_event_count",
                ],
                seed=SEED,
            )
            ltv_baseline_result = ltv_baseline.fit_predict(outcome_records)
            logger.info(
                "LTV baseline (%s): r2=%.4f rmse=%.2f",
                ltv_baseline_result.implementation,
                ltv_baseline_result.metrics.get("r2", 0),
                ltv_baseline_result.metrics.get("rmse", float("inf")),
            )

        # -----------------------------------------------------------------
        # Stage 8: Final run report (human-readable smoke-test summary).
        # -----------------------------------------------------------------
        with run_progress.stage("final report"):
            logger.info(
                "Product embeddings on %s | rows=%d | loss=%.6f | %s",
                model.device_,
                product_rows,
                model.loss_,
                product_validation,
            )
            logger.info("Product quality determination: %s", product_quality)
            logger.info(
                "Customer embeddings on %s | rows=%d | loss=%.6f | %s",
                customer_model.device_,
                customer_rows,
                customer_model.loss_,
                customer_validation,
            )
            logger.info("Customer quality determination: %s", customer_quality)
            logger.info(
                "Temporal core on %s | event rows=%d | loss=%.6f | %s",
                core_model.device_,
                event_rows,
                core_model.loss_,
                core_validation,
            )
            logger.info("Temporal core embedding quality determination: %s", core_embedding_quality)
            logger.info("Temporal core quality determination: %s", core_quality)
            logger.info("Churn metrics: %s", churn_results.report.metrics)
            logger.info("Churn validation: %s | quality: %s", churn_validation, churn_quality)
            logger.info("Churn good-quality determination: %s", churn_good_quality)
            logger.info("LTV metrics: %s", ltv_results.report.metrics)
            logger.info("LTV validation: %s | quality: %s", ltv_validation, ltv_quality)
            logger.info("LTV good-quality determination: %s", ltv_good_quality)

            baseline_quality = {
                "ok": all(
                    [
                        bool(product_validation.get("ok", False)),
                        bool(customer_validation.get("ok", False)),
                        bool(core_validation.get("ok", False)),
                        bool(churn_validation.get("ok", False)),
                        bool(churn_quality.get("ok", False)),
                        bool(ltv_validation.get("ok", False)),
                        bool(ltv_quality.get("ok", False)),
                    ]
                ),
                "product_embeddings": bool(product_validation.get("ok", False)),
                "customer_embeddings": bool(customer_validation.get("ok", False)),
                "temporal_event_embeddings": bool(core_validation.get("ok", False)),
                "churn_head": bool(churn_validation.get("ok", False))
                and bool(churn_quality.get("ok", False)),
                "ltv_head": bool(ltv_validation.get("ok", False))
                and bool(ltv_quality.get("ok", False)),
            }
            logger.info("Baseline quality determination: %s", baseline_quality)

            good_quality = {
                "ok": all(
                    [
                        bool(product_quality.get("ok", False)),
                        bool(customer_quality.get("ok", False)),
                        bool(core_embedding_quality.get("ok", False)),
                        bool(core_quality.get("ok", False)),
                        bool(churn_good_quality.get("ok", False)),
                        bool(ltv_good_quality.get("ok", False)),
                    ]
                ),
                "product_embeddings": bool(product_quality.get("ok", False)),
                "customer_embeddings": bool(customer_quality.get("ok", False)),
                "temporal_event_embeddings": bool(core_embedding_quality.get("ok", False)),
                "temporal_core": bool(core_quality.get("ok", False)),
                "churn_head": bool(churn_good_quality.get("ok", False)),
                "ltv_head": bool(ltv_good_quality.get("ok", False)),
            }
            logger.info("Good-quality determination: %s", good_quality)

            overall_quality = {
                "ok": bool(baseline_quality["ok"])
                and (bool(good_quality["ok"]) if ENFORCE_GOOD_QUALITY else True),
                "good_quality_enforced": ENFORCE_GOOD_QUALITY,
                "baseline_ok": bool(baseline_quality["ok"]),
                "good_quality_ok": bool(good_quality["ok"]),
                "product_embeddings": bool(good_quality.get("product_embeddings", False)),
                "customer_embeddings": bool(good_quality.get("customer_embeddings", False)),
                "temporal_event_embeddings": bool(
                    good_quality.get("temporal_event_embeddings", False)
                ),
                "temporal_core": bool(good_quality.get("temporal_core", False)),
                "churn_head": bool(good_quality.get("churn_head", False)),
                "ltv_head": bool(good_quality.get("ltv_head", False)),
            }
            logger.info("Overall quality determination: %s", overall_quality)

            logger.info(
                "Paradigm: frozen-core LoRA | core shared=%s | heads trained with LoRA adapters",
                qdora_core is not None,
            )
            logger.info(
                "Churn model vs baseline: f1=%.4f vs %.4f | auc=%.4f vs %.4f",
                churn_results.report.metrics.get("f1", 0),
                churn_baseline_result.metrics.get("f1", 0),
                churn_results.report.metrics.get("roc_auc", 0),
                churn_baseline_result.metrics.get("roc_auc", 0),
            )
            logger.info(
                "LTV model vs baseline: r2=%.4f vs %.4f | rmse=%.2f vs %.2f",
                ltv_results.report.metrics.get("r2", 0),
                ltv_baseline_result.metrics.get("r2", 0),
                ltv_results.report.metrics.get("rmse", float("inf")),
                ltv_baseline_result.metrics.get("rmse", float("inf")),
            )
    except Exception:
        logger.exception("Smoke test failed")
        raise
    finally:
        run_progress.close()
        logging.info("Log written to: %s", _LOG_PATH.resolve())
