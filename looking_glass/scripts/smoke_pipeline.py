"""Smoke pipeline stage bodies: embeddings, enrichment, and temporal core."""

from datetime import datetime, timedelta

from looking_glass import (
    attach_vector_feature,
    create_embedding_model,
    create_temporal_core_model,
    load_records_from_duckdb,
    load_vectors_from_lancedb,
    QDoRAConfig,
    save_embeddings_to_lancedb,
    validate_embeddings,
    wrap_core_with_qdora,
)
from looking_glass.scripts.smoke_config import (
    CORE_EPOCHS,
    OUTCOME_ACTIVE_LOOKBACK_DAYS,
    OUTCOME_HISTORY_DAYS,
    OUTCOME_LABEL_DAYS,
    OUTCOME_MAX_CHURN_POSITIVE_RATE,
    OUTCOME_MIN_CHURN_POSITIVE_RATE,
    OUTCOME_MIN_NONZERO_LTV_RATE,
    OUTCOME_MIN_PAST_ORDERS,
    OUTCOME_MIN_RECENT_ORDERS,
    OUTCOME_TARGET_MODE,
    CORE_DEVICE,
    CORE_TRAIN_BATCH_SIZE,
    CUSTOMER_DEVICE,
    CUSTOMER_EMBED_MIN_NORM_STD,
    CUSTOMER_EPOCHS,
    CUSTOMER_GOOD_MAX_MEAN_ABS_COSINE,
    CUSTOMER_GOOD_MIN_NORM_STD,
    EMBED_MAX_MEAN_ABS_COSINE,
    ENFORCE_GOOD_QUALITY,
    EVENT_EMBED_MIN_NORM_STD,
    EVENT_GOOD_MAX_MEAN_ABS_COSINE,
    EVENT_GOOD_MIN_NORM_STD,
    HIDDEN_DIM,
    LANCEDB_DIR,
    PRODUCT_EMBED_MIN_NORM_STD,
    PRODUCT_EPOCHS,
    PRODUCT_GOOD_MAX_MEAN_ABS_COSINE,
    PRODUCT_GOOD_MIN_NORM_STD,
    SEED,
    SMOKE_EVENT_LIMIT,
    DUCKDB_PATH,
)
from looking_glass.scripts.smoke_support import (
    _assert_ok,
    _attach_customer_lookup_fields,
    _build_customer_dimension_records,
    _build_customer_outcome_records,
    _build_event_outcome_records,
    _embedding_good_quality,
    _temporal_core_good_quality,
)


def run_product_embedding_stage(sequence_backend: str):
    # 1) Load raw entity records from DuckDB.
    records = load_records_from_duckdb(
        db_path=DUCKDB_PATH,
        source_table="products",
        columns=["product_id", "category", "base_price"],
    )

    # 2) Train product siloed embedding model.
    model = create_embedding_model(
        id_field="product_id",
        categorical_fields=["category"],
        numeric_fields=["base_price"],
        hidden_dim=HIDDEN_DIM,
        epochs=PRODUCT_EPOCHS,
        seed=SEED,
        sequence_backend=sequence_backend,
        show_progress=True,
        progress_label="Product embedding epochs",
    )
    embeddings = model.fit_transform(records)

    # 3) Persist vectors + selected metadata for retrieval/debugging.
    product_rows = embeddings.save(
        lancedb_dir=LANCEDB_DIR,
        output_table="product_embeddings",
        keep_fields=["category"],
    )
    product_validation = validate_embeddings(
        embeddings.vectors,
        min_norm_std=PRODUCT_EMBED_MIN_NORM_STD,
        min_nonzero_fraction=0.99,
        max_mean_abs_cosine=EMBED_MAX_MEAN_ABS_COSINE,
    )
    _assert_ok("product_embeddings", product_validation)
    product_quality = _embedding_good_quality(
        product_validation,
        min_norm_std=PRODUCT_GOOD_MIN_NORM_STD,
        max_mean_abs_cosine=PRODUCT_GOOD_MAX_MEAN_ABS_COSINE,
    )
    _assert_ok("product_embedding_quality", product_quality, enforce=ENFORCE_GOOD_QUALITY)
    return (
        model,
        product_rows,
        product_validation,
        product_quality,
    )


def run_customer_embedding_stage(sequence_backend: str):
    # Use all customers (no row cap) so the customer silo sees full coverage.
    customer_dimension_rows = load_records_from_duckdb(
        db_path=DUCKDB_PATH,
        source_table="customers",
        columns=[
            "customer_id",
            "signup_ts",
            "birth_year",
            "gender",
            "country",
            "state_region",
            "city",
            "postal_code",
            "loyalty_tier",
            "cardholder_status",
            "income_band",
            "acquisition_channel",
            "lifecycle_stage",
        ],
    )
    if not customer_dimension_rows:
        raise RuntimeError("No customer rows loaded for customer embedding run")

    # Anchor tenure to latest observed event time in the dataset.
    latest_event_row = load_records_from_duckdb(
        db_path=DUCKDB_PATH,
        source_table="customer_events",
        columns=["event_ts"],
        order_by="event_ts DESC",
        limit=1,
    )
    if not latest_event_row:
        raise RuntimeError("No customer events found for tenure reference timestamp")
    latest_event_ts = datetime.fromisoformat(
        str(latest_event_row[0]["event_ts"]).replace("Z", "+00:00")
    )

    customer_dimension_records = _build_customer_dimension_records(
        customer_records=customer_dimension_rows,
        reference_ts=latest_event_ts,
    )

    customer_model = create_embedding_model(
        id_field="customer_id",
        categorical_fields=[
            "gender",
            "country",
            "state_region",
            "city",
            "postal_code",
            "loyalty_tier",
            "income_band",
            "acquisition_channel",
            "lifecycle_stage",
        ],
        numeric_fields=["birth_year", "cardholder_status", "tenure_days"],
        hidden_dim=HIDDEN_DIM,
        epochs=CUSTOMER_EPOCHS,
        seed=SEED,
        device=CUSTOMER_DEVICE,
        sequence_backend=sequence_backend,
        show_progress=True,
        progress_label="Customer embedding epochs",
    )
    customer_embeddings = customer_model.fit_transform(customer_dimension_records)
    customer_rows = customer_embeddings.save(
        lancedb_dir=LANCEDB_DIR,
        output_table="customer_embeddings",
        keep_fields=["country", "loyalty_tier", "lifecycle_stage"],
    )
    customer_validation = validate_embeddings(
        customer_embeddings.vectors,
        min_norm_std=CUSTOMER_EMBED_MIN_NORM_STD,
        min_nonzero_fraction=0.99,
        max_mean_abs_cosine=EMBED_MAX_MEAN_ABS_COSINE,
    )
    _assert_ok("customer_embeddings", customer_validation)
    customer_quality = _embedding_good_quality(
        customer_validation,
        min_norm_std=CUSTOMER_GOOD_MIN_NORM_STD,
        max_mean_abs_cosine=CUSTOMER_GOOD_MAX_MEAN_ABS_COSINE,
    )
    _assert_ok("customer_embedding_quality", customer_quality, enforce=ENFORCE_GOOD_QUALITY)
    return (
        customer_model,
        customer_rows,
        customer_validation,
        customer_quality,
        customer_embeddings,
    )


def run_event_enrichment_stage(customer_embeddings, logger):
    latest_order_row = load_records_from_duckdb(
        db_path=DUCKDB_PATH,
        source_table="orders",
        columns=["order_ts"],
        order_by="order_ts DESC",
        limit=1,
    )
    if not latest_order_row:
        raise RuntimeError("No order rows found to derive leakage cutoff")

    latest_order_ts = datetime.fromisoformat(
        str(latest_order_row[0]["order_ts"]).replace("Z", "+00:00")
    )
    outcome_cutoff_ts = latest_order_ts - timedelta(days=OUTCOME_LABEL_DAYS)
    outcome_cutoff_iso = outcome_cutoff_ts.isoformat()

    logger.info("event enrichment: load customer events")
    customer_event_records = load_records_from_duckdb(
        db_path=DUCKDB_PATH,
        source_table="customer_events",
        where="event_ts <= ?",
        params=(outcome_cutoff_iso,),
        order_by="event_ts DESC" if SMOKE_EVENT_LIMIT is not None else "event_ts",
        limit=SMOKE_EVENT_LIMIT,
        columns=[
            "event_id",
            "customer_id",
            "event_ts",
            "event_type",
            "entity_type",
            "source_table",
            "entity_id",
            "value",
        ],
    )
    if SMOKE_EVENT_LIMIT is not None:
        customer_event_records.reverse()
        logger.info(
            "event enrichment: using latest %d historical customer events before cutoff=%s",
            len(customer_event_records),
            outcome_cutoff_iso,
        )
    else:
        logger.info(
            "event enrichment: using full historical event stream before cutoff=%s",
            outcome_cutoff_iso,
        )
    if len(customer_event_records) < 2:
        raise RuntimeError("Not enough customer event rows for temporal core training")

    # Attach static customer vectors to every event row by customer_id.
    logger.info("event enrichment: attach customer vectors")
    customer_event_records = attach_vector_feature(
        records=customer_event_records,
        lookup_key="customer_id",
        vector_lookup=customer_embeddings.vectors,
        output_field="customer_vector",
        show_progress=True,
        progress_label="Enrichment: customer vectors",
    )
    customer_hits = sum(
        1
        for row in customer_event_records
        if str(row.get("customer_id", "")) in customer_embeddings.vectors
    )
    customer_hit_rate = customer_hits / max(len(customer_event_records), 1)
    logger.info(
        "event enrichment: customer vector coverage %.2f%%",
        100.0 * customer_hit_rate,
    )
    if customer_hit_rate < 0.99:
        raise RuntimeError(f"Customer vector coverage too low: {customer_hit_rate:.3f}")

    logger.info("event enrichment: load product vectors")
    product_vectors = load_vectors_from_lancedb(
        lancedb_dir=LANCEDB_DIR,
        table_name="product_embeddings",
        id_field="product_id",
    )

    # Attach product vectors to event rows by entity_id when applicable.
    logger.info("event enrichment: attach product vectors")
    customer_event_records = attach_vector_feature(
        records=customer_event_records,
        lookup_key="entity_id",
        vector_lookup=product_vectors,
        output_field="product_vector",
        show_progress=True,
        progress_label="Enrichment: product vectors",
    )
    product_entity_rows = [
        row
        for row in customer_event_records
        if str(row.get("entity_type", "")).lower() == "product"
    ]
    if product_entity_rows:
        product_hits = sum(
            1 for row in product_entity_rows if str(row.get("entity_id", "")) in product_vectors
        )
        product_hit_rate = product_hits / max(len(product_entity_rows), 1)
        logger.info(
            "event enrichment: product vector coverage %.2f%% on product events",
            100.0 * product_hit_rate,
        )
        if product_hit_rate < 0.85:
            raise RuntimeError(
                f"Product vector coverage too low on product events: {product_hit_rate:.3f}"
            )
    return (
        customer_event_records,
        outcome_cutoff_ts,
    )


def run_temporal_core_stage(customer_event_records, sequence_backend, logger):
    core_model = create_temporal_core_model(
        sequence_id_field="customer_id",
        event_id_field="event_id",
        timestamp_field="event_ts",
        categorical_fields=["event_type", "entity_type", "source_table"],
        numeric_fields=["value"],
        vector_fields=["customer_vector", "product_vector"],
        hidden_dim=HIDDEN_DIM,
        epochs=CORE_EPOCHS,
        seed=SEED,
        learning_rate=1e-3,
        device=CORE_DEVICE,
        sequence_backend=sequence_backend,
        train_batch_size=CORE_TRAIN_BATCH_SIZE,
        input_is_time_sorted=True,
        show_progress=True,
        progress_label="Temporal core",
    )
    core_outputs = core_model.fit_transform(customer_event_records)
    core_validation = validate_embeddings(
        core_outputs.event_embeddings,
        min_norm_std=EVENT_EMBED_MIN_NORM_STD,
        min_nonzero_fraction=0.99,
        max_mean_abs_cosine=EMBED_MAX_MEAN_ABS_COSINE,
    )
    _assert_ok("event_embeddings", core_validation)
    core_embedding_quality = _embedding_good_quality(
        core_validation,
        min_norm_std=EVENT_GOOD_MIN_NORM_STD,
        max_mean_abs_cosine=EVENT_GOOD_MAX_MEAN_ABS_COSINE,
    )
    _assert_ok("event_embedding_quality", core_embedding_quality, enforce=ENFORCE_GOOD_QUALITY)
    core_quality = _temporal_core_good_quality(
        core_validation=core_validation,
        core_loss=float(core_model.loss_ or 0.0),
        customer_records=core_outputs.customer_records,
    )
    _assert_ok("temporal_core_quality", core_quality, enforce=ENFORCE_GOOD_QUALITY)
    event_rows = save_embeddings_to_lancedb(
        lancedb_dir=LANCEDB_DIR,
        output_table="event_embeddings",
        records=customer_event_records,
        id_field="event_id",
        embeddings=core_outputs.event_embeddings,
        keep_fields=["customer_id", "event_type", "entity_type", "source_table"],
    )

    # Retrofit the trained core with QDoRA adapters so downstream
    # task heads train only lightweight LoRA params on a frozen backbone.
    qdora_core = core_model.trained_core
    if qdora_core is not None:
        wrap_core_with_qdora(qdora_core, QDoRAConfig(rank=8, quantize_base=False))
        logger.info("temporal core retrofitted with QDoRA for frozen-core downstream heads")
    return (
        core_model,
        core_outputs,
        core_validation,
        core_embedding_quality,
        core_quality,
        event_rows,
        qdora_core,
    )


def run_outcome_stage(core_outputs, outcome_cutoff_ts, customer_embeddings, logger):
    # Pull per-customer temporal summaries emitted by the core model.
    core_customer_lookup = {
        str(row["customer_id"]): {
            "core_last_vector": row["core_last_vector"],
            "core_mean_vector": row["core_mean_vector"],
            "core_event_count": row["core_event_count"],
        }
        for row in core_outputs.customer_records
    }
    if not core_customer_lookup:
        raise RuntimeError("No temporal customer summaries available for supervised outcomes")

    def _build_outcomes_for_mode(mode: str) -> list[dict[str, object]]:
        if mode == "order":
            order_records = load_records_from_duckdb(
                db_path=DUCKDB_PATH,
                source_table="orders",
                columns=[
                    "customer_id",
                    "order_ts",
                    "order_status",
                    "order_total",
                    "return_flag",
                    "return_amount",
                ],
            )
            return _build_customer_outcome_records(
                order_records=order_records,
                cutoff_ts=outcome_cutoff_ts,
                history_days=OUTCOME_HISTORY_DAYS,
                label_days=OUTCOME_LABEL_DAYS,
                min_past_orders=OUTCOME_MIN_PAST_ORDERS,
                active_lookback_days=OUTCOME_ACTIVE_LOOKBACK_DAYS,
                min_recent_orders=OUTCOME_MIN_RECENT_ORDERS,
            )

        event_value_records = load_records_from_duckdb(
            db_path=DUCKDB_PATH,
            source_table="customer_events",
            columns=[
                "customer_id",
                "event_ts",
                "source_table",
                "entity_type",
                "value",
            ],
        )
        return _build_event_outcome_records(
            event_records=event_value_records,
            cutoff_ts=outcome_cutoff_ts,
            history_days=OUTCOME_HISTORY_DAYS,
            label_days=OUTCOME_LABEL_DAYS,
            min_past_events=OUTCOME_MIN_PAST_ORDERS,
            active_lookback_days=OUTCOME_ACTIVE_LOOKBACK_DAYS,
            min_recent_events=OUTCOME_MIN_RECENT_ORDERS,
        )

    primary_mode = "order" if OUTCOME_TARGET_MODE == "auto" else OUTCOME_TARGET_MODE
    fallback_mode = "event" if primary_mode == "order" else "order"
    mode_candidates = [primary_mode, fallback_mode]
    selection_issues: list[str] = []
    selected_mode: str | None = None
    selected_records: list[dict[str, object]] | None = None

    for mode in mode_candidates:
        outcome_records = _build_outcomes_for_mode(mode)
        if len(outcome_records) < 2:
            issue = f"mode={mode}: not enough raw outcome rows ({len(outcome_records)})"
            logger.warning("build supervised outcomes: %s", issue)
            selection_issues.append(issue)
            continue

        pre_filter_positive_rate = sum(
            float(row.get("churn_label", 0.0) or 0.0) for row in outcome_records
        ) / len(outcome_records)
        logger.info(
            "build supervised outcomes: mode=%s pre-filter rows=%d churn_positive_rate=%.4f",
            mode,
            len(outcome_records),
            pre_filter_positive_rate,
        )

        core_feature_coverage = sum(
            1 for row in outcome_records if str(row["customer_id"]) in core_customer_lookup
        ) / len(outcome_records)
        logger.info(
            "build supervised outcomes: mode=%s temporal core coverage %.2f%%",
            mode,
            100.0 * core_feature_coverage,
        )
        retained_records = [
            row for row in outcome_records if str(row["customer_id"]) in core_customer_lookup
        ]
        if len(retained_records) < 2:
            issue = (
                f"mode={mode}: not enough rows after temporal-coverage filter "
                f"({len(retained_records)})"
            )
            logger.warning("build supervised outcomes: %s", issue)
            selection_issues.append(issue)
            continue

        churn_positive_rate = sum(
            float(row.get("churn_label", 0.0) or 0.0) for row in retained_records
        ) / len(retained_records)
        nonzero_ltv_rate = sum(
            1 for row in retained_records if float(row.get("ltv_value", 0.0) or 0.0) > 0.0
        ) / len(retained_records)
        ltv_mean = sum(float(row.get("ltv_value", 0.0) or 0.0) for row in retained_records) / len(
            retained_records
        )
        logger.info(
            (
                "build supervised outcomes: mode=%s retained rows=%d "
                "churn_positive_rate=%.4f nonzero_ltv_rate=%.4f ltv_mean=%.2f"
            ),
            mode,
            len(retained_records),
            churn_positive_rate,
            nonzero_ltv_rate,
            ltv_mean,
        )

        if (
            churn_positive_rate <= OUTCOME_MIN_CHURN_POSITIVE_RATE
            or churn_positive_rate >= OUTCOME_MAX_CHURN_POSITIVE_RATE
        ):
            issue = (
                f"mode={mode}: degenerate churn label balance "
                f"({churn_positive_rate:.4f}) outside "
                f"[{OUTCOME_MIN_CHURN_POSITIVE_RATE:.4f}, {OUTCOME_MAX_CHURN_POSITIVE_RATE:.4f}]"
            )
            logger.warning("build supervised outcomes: %s", issue)
            selection_issues.append(issue)
            continue

        if nonzero_ltv_rate < OUTCOME_MIN_NONZERO_LTV_RATE:
            issue = (
                f"mode={mode}: sparse nonzero LTV coverage "
                f"({nonzero_ltv_rate:.4f}) below {OUTCOME_MIN_NONZERO_LTV_RATE:.4f}"
            )
            logger.warning("build supervised outcomes: %s", issue)
            selection_issues.append(issue)
            continue

        selected_mode = mode
        selected_records = retained_records
        break

    if selected_records is None or selected_mode is None:
        issue_summary = "; ".join(selection_issues) if selection_issues else "unknown"
        raise RuntimeError(
            f"Unable to build non-degenerate supervised outcomes in any mode: {issue_summary}"
        )

    if selected_mode != primary_mode:
        logger.warning(
            (
                "build supervised outcomes: switched target mode from %s to %s "
                "due to label degeneracy checks"
            ),
            primary_mode,
            selected_mode,
        )
    logger.info("build supervised outcomes: selected target mode=%s", selected_mode)

    outcome_records = selected_records

    outcome_records = _attach_customer_lookup_fields(
        records=outcome_records,
        lookup=core_customer_lookup,
        id_field="customer_id",
    )
    outcome_records = attach_vector_feature(
        records=outcome_records,
        lookup_key="customer_id",
        vector_lookup=customer_embeddings.vectors,
        output_field="customer_vector",
    )
    return outcome_records
