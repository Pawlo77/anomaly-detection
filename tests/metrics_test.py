"""Tests for unified anomaly metrics protocol."""

import numpy as np
import pytest

from anomaly_detection.metrics import (
    DEFAULT_METRICS_PROTOCOL,
    MetricThresholdConfig,
    contamination_slice_prediction_stability,
    evaluate_metrics,
    labels_from_scores,
    majority_inlier_accuracy,
    mean_absolute_pairwise_spearman,
    metric_accuracy,
    metric_f1,
    metric_mcc,
    metric_pr_auc,
    metric_precision,
    metric_recall,
    metric_roc_auc,
    smtp_extreme_skew_metrics,
)


def _fixture_labels_scores() -> tuple[np.ndarray, np.ndarray]:
    """Create stable labels/scores pair for deterministic assertions."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.array([0.05, 0.10, 0.40, 0.55, 0.70, 0.95], dtype=np.float64)
    return y_true, scores


def test_labels_from_scores_respects_contamination() -> None:
    """Thresholding marks exact ceil(contamination*n) outlier count."""
    _, scores = _fixture_labels_scores()
    labels = labels_from_scores(scores, MetricThresholdConfig(contamination=1 / 3))
    assert labels.sum() == 2
    assert set(np.unique(labels)).issubset({0, 1})


def test_labels_from_scores_zero_contamination_all_inliers() -> None:
    """Zero contamination clears the outlier budget (plan datasets with purely inlier probes)."""
    scores = np.array([0.1, 3.9, 0.2], dtype=np.float64)
    labels = labels_from_scores(scores, MetricThresholdConfig(contamination=0.0))
    assert int(labels.sum()) == 0


def test_labels_from_scores_uses_deterministic_tie_breaking() -> None:
    """Tied scores still produce exact outlier count deterministically."""
    scores = np.array([0.9, 0.9, 0.1, 0.1], dtype=np.float64)
    labels = labels_from_scores(scores, MetricThresholdConfig(contamination=0.25))
    assert labels.sum() == 1
    assert labels[0] == 1
    assert labels[1] == 0


def test_binary_metrics_match_expected_values() -> None:
    """Binary wrappers return consistent scalar values."""
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 0, 1], dtype=np.int64)

    assert metric_accuracy(y_true, y_pred) == pytest.approx(0.5)
    assert metric_precision(y_true, y_pred) == pytest.approx(0.5)
    assert metric_recall(y_true, y_pred) == pytest.approx(0.5)
    assert metric_f1(y_true, y_pred) == pytest.approx(0.5)
    assert metric_mcc(y_true, y_pred) == pytest.approx(0.0)


def test_score_metrics_single_class_fallbacks_are_stable() -> None:
    """Ranking metrics handle degenerate class vectors safely."""
    y_true = np.array([0, 0, 0], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    assert metric_roc_auc(y_true, scores) == pytest.approx(0.5)
    assert metric_pr_auc(y_true, scores) == pytest.approx(0.0)


def test_evaluate_metrics_returns_full_report() -> None:
    """Protocol evaluation returns complete report fields."""
    y_true, scores = _fixture_labels_scores()
    report = evaluate_metrics(
        y_true,
        scores,
        threshold_config=MetricThresholdConfig(contamination=0.5),
    )

    assert set(report.model_dump()) == {
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "mcc",
        "f1",
    }
    assert report.roc_auc == pytest.approx(1.0)
    assert report.pr_auc == pytest.approx(1.0)


def test_protocol_uses_custom_metric_composition() -> None:
    """Composable protocol can be swapped for custom behavior."""

    def fake_accuracy(_: np.ndarray, __: np.ndarray) -> float:
        return 0.123

    custom_protocol = DEFAULT_METRICS_PROTOCOL.__class__(
        binary_metrics={
            "accuracy": fake_accuracy,
            "precision": metric_precision,
            "recall": metric_recall,
            "mcc": metric_mcc,
            "f1": metric_f1,
        },
        score_metrics={
            "roc_auc": metric_roc_auc,
            "pr_auc": metric_pr_auc,
        },
    )

    y_true, scores = _fixture_labels_scores()
    report = evaluate_metrics(y_true, scores, protocol=custom_protocol)
    assert report.accuracy == pytest.approx(0.123)


def test_majority_inlier_accuracy_matches_extreme_skew_story() -> None:
    """§3.1 majority baseline approximates ``1 - c`` for rare positives."""
    smtp_c = 0.0003
    assert majority_inlier_accuracy(smtp_c) == pytest.approx(1.0 - smtp_c, rel=0, abs=1e-6)


def test_smtp_extreme_skew_flags_accuracy_gap() -> None:
    """Poor detector can trail always-normal strat."""
    contour = smtp_extreme_skew_metrics(
        contamination=0.0003,
        model_accuracy=0.995,
    )
    assert contour["accuracy_deficit_vs_always_normal"] < 0


def test_mean_absolute_pairwise_spearman_perfect_corr() -> None:
    preds = np.array([[1, 1], [1, 1], [0, 0]], dtype=np.float64)
    assert mean_absolute_pairwise_spearman(preds) == pytest.approx(1.0)


def test_mean_pairwise_spearman_on_duplicated_predictions() -> None:
    ys = np.array([9.8, 0.01, 0.5, -1.5, -2.8], dtype=np.float64)
    pred = labels_from_scores(ys, MetricThresholdConfig(contamination=0.2))
    dup = np.column_stack([pred, pred]).astype(np.float64)
    assert mean_absolute_pairwise_spearman(dup) == pytest.approx(1.0)


def test_contamination_slice_matches_rank_stability_gadget() -> None:
    ys = np.linspace(-1.0, 10.0, num=96, dtype=np.float64)
    cont_levels = np.array([0.05, 0.10], dtype=np.float64)
    out = contamination_slice_prediction_stability(ys, cont_levels)
    manual = np.column_stack(
        tuple(
            labels_from_scores(
                ys, MetricThresholdConfig(contamination=float(cont_levels[j]))
            ).astype(np.float64)
            for j in range(cont_levels.shape[0])
        )
    )
    expected = mean_absolute_pairwise_spearman(manual)
    assert out["mean_abs_pairwise_spearman_preds"] == pytest.approx(expected)
