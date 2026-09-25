from looking_glass import GBTBaseline, baseline_implementation_name, compare


def _classification_rows(n=200):
    rows = []
    for i in range(n):
        positive = i % 2 == 0
        x = 5.0 + i * 0.01 if positive else -5.0 - i * 0.01
        rows.append(
            {
                "customer_id": f"c{i}",
                "x": x,
                "noise": (i % 7),
                "churn_label": 1.0 if positive else 0.0,
            }
        )
    return rows


def _regression_rows(n=200):
    return [{"customer_id": f"c{i}", "x": float(i), "value_label": 3.0 * i + 2.0} for i in range(n)]


def test_baseline_implementation_name():
    assert baseline_implementation_name() in {"xgboost", "linear_fallback"}


def test_classification_baseline_separates():
    model = GBTBaseline(
        task="classification",
        id_field="customer_id",
        target_field="churn_label",
        feature_fields=["x", "noise"],
    )
    result = model.fit_predict(_classification_rows())
    assert result.task == "classification"
    assert len(result.predictions) == 200
    assert result.metrics["roc_auc"] > 0.9
    assert 0.0 <= result.metrics["threshold"] <= 1.0


def test_regression_baseline_fits_linear():
    model = GBTBaseline(
        task="regression",
        id_field="customer_id",
        target_field="value_label",
        feature_fields=["x"],
    )
    result = model.fit_predict(_regression_rows())
    assert result.task == "regression"
    assert result.metrics["r2"] > 0.9


def test_compare_table():
    table = compare({"roc_auc": 0.85}, {"roc_auc": 0.80}, keys=["roc_auc"])
    assert abs(table["roc_auc"]["delta_model_minus_baseline"] - 0.05) < 1e-9
