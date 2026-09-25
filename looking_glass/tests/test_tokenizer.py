from looking_glass import (
    PayloadSchema,
    collect_payload_vocabularies,
    parse_event_payload,
)


_TEST_SCHEMA = PayloadSchema(
    categorical_fields=["payment_method", "device"],
    numeric_fields=["order_value", "rating"],
    vector_id_keys=["product_id"],
)


def test_parse_categorical_and_numeric():
    parsed = parse_event_payload(
        {"payment_method": "card", "device": "ios", "order_value": 99.99, "rating": 4},
        _TEST_SCHEMA,
    )
    assert parsed.cat_values["payment_method"] == "card"
    assert parsed.cat_values["device"] == "ios"
    assert parsed.num_values["order_value"] == 99.99
    assert parsed.num_values["rating"] == 4.0


def test_missing_keys_get_defaults():
    parsed = parse_event_payload({}, _TEST_SCHEMA)
    assert parsed.cat_values["payment_method"] == ""
    assert parsed.cat_values["device"] == ""
    assert parsed.num_values["order_value"] == 0.0
    assert parsed.num_values["rating"] == 0.0


def test_non_numeric_becomes_zero():
    parsed = parse_event_payload({"order_value": "not-a-number"}, _TEST_SCHEMA)
    assert parsed.num_values["order_value"] == 0.0


def test_vector_sum_with_lookups():
    lookup = {"product_id": lambda eid: [1.0, 2.0, 3.0] if eid == "p01" else None}
    parsed = parse_event_payload(
        {"product_id": "p01"},
        _TEST_SCHEMA,
        vector_lookups=lookup,
        default_vector_width=3,
    )
    assert parsed.combined_vector == [1.0, 2.0, 3.0]


def test_vector_sum_adds_multiple():
    lookup = {
        "product_id": lambda eid: [1.0, 1.0] if eid == "a" else None,
    }
    schema = PayloadSchema(categorical_fields=[], numeric_fields=[], vector_id_keys=["product_id"])
    parsed = parse_event_payload(
        {"product_id": "a"}, schema, vector_lookups=lookup, default_vector_width=2
    )
    assert parsed.combined_vector == [1.0, 1.0]


def test_missing_entity_id_is_zero_vector():
    lookup = {"product_id": lambda eid: None}
    parsed = parse_event_payload({}, _TEST_SCHEMA, vector_lookups=lookup, default_vector_width=4)
    assert parsed.combined_vector == [0.0, 0.0, 0.0, 0.0]


def test_collect_payload_vocabularies():
    rows = [
        {
            "event_type": "purchase",
            "event_payload_json": {"payment_method": "card", "device": "ios"},
        },
        {
            "event_type": "view",
            "event_payload_json": {"payment_method": "paypal", "device": "android"},
        },
    ]
    schema = PayloadSchema(categorical_fields=["payment_method", "device"], numeric_fields=[])
    vocabs = collect_payload_vocabularies(rows, schema)
    assert set(vocabs.keys()) == {"payment_method", "device", "event_type"}
    assert vocabs["payment_method"] == {"card": 0, "paypal": 1}
    assert vocabs["event_type"] == {"purchase": 0, "view": 1}


def test_collect_payload_vocabularies_deterministic():
    rows = [{"event_type": "b"}, {"event_type": "a"}, {"event_type": "a"}]
    schema = PayloadSchema(categorical_fields=[], numeric_fields=[])
    vocabs = collect_payload_vocabularies(rows, schema)
    assert vocabs["event_type"] == {"a": 0, "b": 1}
