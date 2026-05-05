"""Unified metric protocol for anomaly detection evaluation.

This module wraps ``sklearn.metrics`` with a stable project-level interface.
All metric functions consume binary labels, while thresholding from continuous
scores is centralized to ensure reproducible behavior across models.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, computed_field
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type MetricFn = Callable[[IntArray, IntArray], float]
type ScoreMetricFn = Callable[[IntArray, FloatArray], float]

OUTLIER_LABEL: int = 1
"""Integer label representing anomaly class in this project."""

INLIER_LABEL: int = 0
"""Integer label representing normal class in this project."""


class MetricThresholdConfig(BaseModel):
    """Configuration for score-to-label thresholding.

    Attributes:
        contamination: Fraction of points labeled as anomalies.
        outlier_label: Label used for predicted anomalies.
        inlier_label: Label used for predicted normal observations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    contamination: float = Field(default=0.1, ge=0.0, le=1.0)
    outlier_label: int = Field(default=OUTLIER_LABEL)
    inlier_label: int = Field(default=INLIER_LABEL)


class BinaryEvaluationInput(BaseModel):
    """Validated input for binary metric computation.

    Attributes:
        y_true: Ground-truth labels where anomaly is ``outlier_label``.
        y_pred: Predicted labels where anomaly is ``outlier_label``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    y_true: IntArray
    y_pred: IntArray

    @computed_field(return_type=int)
    @property
    def n_samples(self) -> int:
        """Return total number of evaluated samples."""
        return int(self.y_true.shape[0])


class ScoreEvaluationInput(BaseModel):
    """Validated input for ranking metric computation.

    Attributes:
        y_true: Ground-truth binary labels.
        scores: Continuous anomaly scores (higher means more anomalous).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    y_true: IntArray
    scores: FloatArray


class MetricsReport(BaseModel):
    """Structured evaluation report for anomaly detection.

    Attributes:
        accuracy: Classification accuracy.
        precision: Precision for anomaly class.
        recall: Recall for anomaly class.
        roc_auc: Area under ROC curve.
        pr_auc: Average precision (area under PR curve).
        mcc: Matthews correlation coefficient.
        f1: F1 score for anomaly class.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)
    pr_auc: float = Field(ge=0.0, le=1.0)
    mcc: float = Field(ge=-1.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


def _as_int_vector(values: npt.ArrayLike) -> IntArray:
    """Convert input to contiguous ``int64`` vector.

    Args:
        values: Array-like labels.

    Returns:
        Contiguous one-dimensional ``int64`` vector.

    Raises:
        ValueError: If input cannot be interpreted as one-dimensional.
    """
    vector = np.asarray(values, dtype=np.int64)
    if vector.ndim != 1:
        raise ValueError("Expected one-dimensional label vector.")
    return np.ascontiguousarray(vector)


def _as_float_vector(values: npt.ArrayLike) -> FloatArray:
    """Convert input to contiguous ``float64`` vector.

    Args:
        values: Array-like score values.

    Returns:
        Contiguous one-dimensional ``float64`` vector.

    Raises:
        ValueError: If input cannot be interpreted as one-dimensional.
    """
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("Expected one-dimensional score vector.")
    return np.ascontiguousarray(vector)


def _validate_equal_length(lhs: npt.NDArray[np.generic], rhs: npt.NDArray[np.generic]) -> None:
    """Validate vectors have matching sample counts.

    Args:
        lhs: Left-hand vector.
        rhs: Right-hand vector.

    Raises:
        ValueError: If vector lengths differ.
    """
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError("Input vectors must have identical length.")


def labels_from_scores(
    scores: npt.ArrayLike,
    config: MetricThresholdConfig | None = None,
) -> IntArray:
    """Convert anomaly scores to binary labels via fixed outlier budget.

    Args:
        scores: Continuous anomaly scores where larger means more anomalous.
        config: Optional thresholding configuration.

    Returns:
        Vector with ``outlier_label`` and ``inlier_label`` values.
        Exactly ``ceil(contamination * n_samples)`` points are labeled outliers.

    Raises:
        ValueError: If scores are empty.
    """
    cfg = config or MetricThresholdConfig()
    score_vector = _as_float_vector(scores)
    if score_vector.size == 0:
        raise ValueError("Cannot threshold empty score vector.")

    n_samples = int(score_vector.size)
    n_outliers = int(np.ceil(cfg.contamination * n_samples))
    n_outliers = int(np.clip(n_outliers, 0, n_samples))
    if n_outliers == 0:
        return np.full(score_vector.shape, fill_value=cfg.inlier_label, dtype=np.int64)
    labels = np.full(score_vector.shape, fill_value=cfg.inlier_label, dtype=np.int64)
    sorted_idx = np.lexsort((np.arange(n_samples, dtype=np.int64), -score_vector))
    outlier_idx = sorted_idx[:n_outliers]
    labels[outlier_idx] = cfg.outlier_label
    return labels


def metric_accuracy(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Compute binary accuracy for anomaly labels.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Thresholded predictions aligned sample-wise.

    Returns:
        Fraction of agreeing labels within ``[0, 1]``.
    """
    yt = _as_int_vector(y_true)
    yp = _as_int_vector(y_pred)
    _validate_equal_length(yt, yp)
    return float(accuracy_score(yt, yp))


def metric_precision(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Compute anomaly precision with deterministic zero-handling safeguards.

    Args:
        y_true: Ground-truth labels for positives (outliers coded as ``OUTLIER_LABEL``).
        y_pred: Discrete predictions referencing the same label encoding.

    Returns:
        Precision for the anomaly class respecting sklearn ``zero_division`` policy.
    """
    yt = _as_int_vector(y_true)
    yp = _as_int_vector(y_pred)
    _validate_equal_length(yt, yp)
    return float(precision_score(yt, yp, pos_label=OUTLIER_LABEL, zero_division=0.0))


def metric_recall(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Compute anomaly recall with deterministic zero-handling safeguards.

    Args:
        y_true: Ground-truth anomalies versus inliers per project constants.
        y_pred: Detector predictions subjected to identical encoding.

    Returns:
        Recall for detecting labeled anomalies.
    """
    yt = _as_int_vector(y_true)
    yp = _as_int_vector(y_pred)
    _validate_equal_length(yt, yp)
    return float(recall_score(yt, yp, pos_label=OUTLIER_LABEL, zero_division=0.0))


def metric_f1(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Harmonic mean of anomaly precision/recall with stable zero denominators.

    Args:
        y_true: Validation labels.
        y_pred: Detector outputs snapped to discrete classes.

    Returns:
        Macro-stable F1 for the anomaly class.
    """
    yt = _as_int_vector(y_true)
    yp = _as_int_vector(y_pred)
    _validate_equal_length(yt, yp)
    return float(f1_score(yt, yp, pos_label=OUTLIER_LABEL, zero_division=0.0))


def metric_mcc(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Compute Matthews correlation coefficient balancing class skew.

    Args:
        y_true: Ground-truth binary vector.
        y_pred: Comparable prediction vector identical length/shape rules.

    Returns:
        Signed correlation statistic within ``[-1, 1]``.
    """
    yt = _as_int_vector(y_true)
    yp = _as_int_vector(y_pred)
    _validate_equal_length(yt, yp)
    return float(matthews_corrcoef(yt, yp))


def metric_roc_auc(y_true: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Compute ROC-AUC ranking quality for anomaly scores.

    Args:
        y_true: Binary labels aligning with anomaly scores dimensionality.
        scores: Monotone anomaly scores emitted by detectors.

    Returns:
        ROC-AUC in ``[0, 1]``, defaulting ``0.5`` when positives/negatives are singletons.
    """
    yt = _as_int_vector(y_true)
    ys = _as_float_vector(scores)
    _validate_equal_length(yt, ys)
    if np.unique(yt).size < 2:
        return 0.5
    return float(roc_auc_score(yt, ys))


def metric_pr_auc(y_true: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    """Compute PR-AUC (average precision) for imbalanced anomalies.

    Args:
        y_true: Binary labels aligning with anomaly scores dimensionality.
        scores: Ranking scores prioritized for recall-oriented evaluation.

    Returns:
        Average precision in ``[0, 1]``, or ``0.0`` if positives are absent.
    """
    yt = _as_int_vector(y_true)
    ys = _as_float_vector(scores)
    _validate_equal_length(yt, ys)
    if np.sum(yt == OUTLIER_LABEL) == 0:
        return 0.0
    return float(average_precision_score(yt, ys))


def majority_inlier_accuracy(contamination: float) -> float:
    """Accuracy when labeling every point as normal (plan §3.1 SMTP-style baseline).

    Args:
        contamination: Fraction of positives in the population.

    Returns:
        Limiting accuracy roughly ``1 - contamination`` for large ``n``.
    """
    c = float(contamination)
    return float(max(0.0, min(1.0, 1.0 - c)))


def smtp_extreme_skew_metrics(
    contamination: float,
    model_accuracy: float,
) -> dict[str, float]:
    """Demonstrate why accuracy is misleading when positives are extremely rare (§3.1).

    Compares the evaluated ``model_accuracy`` against the always-normal majority baseline.
    """
    baseline = majority_inlier_accuracy(contamination)
    return {
        "always_normal_accuracy_ceiling": baseline,
        "accuracy_deficit_vs_always_normal": float(model_accuracy) - baseline,
    }


def mean_absolute_pairwise_spearman(predictions: npt.ArrayLike) -> float:
    """Average absolute Spearman rho across unique column pairs (plan §2.5 ECOD stability).

    Args:
        predictions: Integer or float matrix ``(n_samples, n_settings)`` such as binary preds.

    Returns:
        Mean absolute Spearman correlation across column pairs, ``nan`` when ``< 2`` columns.
    """
    matrix = np.asarray(predictions)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return float("nan")
    cols = matrix.shape[1]
    accum: list[float] = []
    for left in range(cols):
        for right in range(left + 1, cols):
            rho, _ = scipy_stats.spearmanr(matrix[:, left], matrix[:, right])
            if np.isnan(rho):
                continue
            accum.append(abs(float(rho)))
    return float(np.mean(np.asarray(accum, dtype=np.float64))) if accum else float("nan")


def contamination_slice_prediction_stability(
    scores: npt.ArrayLike,
    contamination_values: npt.ArrayLike,
) -> dict[str, float]:
    """Spearman stability between binary preds sliced at many contamination cutoffs (§2.5 ECOD)."""
    ys = _as_float_vector(scores)
    cont = np.asarray(contamination_values, dtype=np.float64).ravel()
    stacked = np.column_stack(
        tuple(
            labels_from_scores(ys, MetricThresholdConfig(contamination=float(c))).astype(np.float64)
            for c in cont
        )
    )
    return {"mean_abs_pairwise_spearman_preds": mean_absolute_pairwise_spearman(stacked)}


@dataclass(slots=True, frozen=True)
class MetricsProtocol:
    """Composable protocol for metric execution.

    Attributes:
        binary_metrics: Mapping of metric names to binary metric callables.
        score_metrics: Mapping of metric names to score-based metric callables.
    """

    binary_metrics: dict[str, MetricFn]
    score_metrics: dict[str, ScoreMetricFn]

    def evaluate(self, y_true: IntArray, y_pred: IntArray, scores: FloatArray) -> MetricsReport:
        """Compute full report from validated vectors.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            scores: Continuous anomaly scores.

        Returns:
            Immutable metrics report.
        """
        values: dict[str, float] = {}
        for name, fn in self.binary_metrics.items():
            values[name] = fn(y_true, y_pred)
        for name, fn in self.score_metrics.items():
            values[name] = fn(y_true, scores)
        return MetricsReport(**values)


DEFAULT_METRICS_PROTOCOL = MetricsProtocol(
    binary_metrics={
        "accuracy": metric_accuracy,
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
"""Default protocol implementing all planned project metrics."""


def evaluate_metrics(
    y_true: npt.ArrayLike,
    scores: npt.ArrayLike,
    threshold_config: MetricThresholdConfig | None = None,
    protocol: MetricsProtocol = DEFAULT_METRICS_PROTOCOL,
) -> MetricsReport:
    """Evaluate all anomaly detection metrics under unified protocol.

    Args:
        y_true: Ground-truth binary labels.
        scores: Continuous anomaly scores where higher is more anomalous.
        threshold_config: Optional thresholding configuration.
        protocol: Metric protocol implementation.

    Returns:
        Immutable report containing all configured metrics.
    """
    yt = _as_int_vector(y_true)
    ys = _as_float_vector(scores)
    _validate_equal_length(yt, ys)
    yp = labels_from_scores(ys, config=threshold_config)
    return protocol.evaluate(y_true=yt, y_pred=yp, scores=ys)
