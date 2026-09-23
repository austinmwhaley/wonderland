import math

from looking_glass import lift_table, pr_auc, ranking_metrics, roc_auc


def test_roc_auc_separable():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
    labels = [0, 0, 0, 1, 1, 1]
    assert roc_auc(scores, labels) == 1.0
    assert roc_auc(scores[::-1], labels) == 0.0


def test_roc_auc_single_class_is_half():
    assert roc_auc([0.1, 0.2, 0.3], [1, 1, 1]) == 0.5
    assert roc_auc([0.1, 0.2, 0.3], [0, 0, 0]) == 0.5


def test_roc_auc_handles_ties():
    auc = roc_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
    assert abs(auc - 0.5) < 1e-9


def test_pr_auc_perfect_ranking():
    ap = pr_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert abs(ap - 1.0) < 1e-9


def test_pr_auc_no_positives_is_zero():
    assert pr_auc([0.9, 0.1], [0, 0]) == 0.0


def test_lift_table_structure_and_top_bucket():
    scores = [i / 100.0 for i in range(100)]
    labels = [1 if i >= 90 else 0 for i in range(100)]
    table = lift_table(scores, labels, n_bins=10)
    assert len(table) == 10
    assert sum(b.count for b in table) == 100
    assert sum(b.positives for b in table) == 10
    # Top bucket should concentrate positives -> high lift.
    assert table[0].lift > 5.0
    assert table[0].cumulative_lift > 5.0


def test_ranking_metrics_bundle():
    m = ranking_metrics([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert set(m) == {"roc_auc", "pr_auc", "top_decile_lift"}
    assert all(math.isfinite(v) for v in m.values())
