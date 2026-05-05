"""Descriptive statistics for canonical datasets feeding reports and loaders."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..geometry import distance_concentration_ratio

DISTANCE_SAMPLE_LIMIT = 300
"""Maximum leading rows passed to ``distance_concentration_ratio`` after ``dropna``."""


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Summary statistics payload.

    Attributes:
        n_rows: Number of rows in dataset.
        n_features: Number of features in dataset (excluding label).
        contamination: Fraction of samples labeled as outliers.
        missing_fraction: Fraction of missing values in dataset.
        max_feature_missing_fraction: Maximum fraction of missing values in any single feature.
        outlier_count: Number of samples labeled as outliers.
        inlier_count: Number of samples labeled as inliers.
        mean_abs_correlation: Mean absolute correlation between numeric features.
        distance_concentration_ratio: Ratio of (max distance - min distance) to mean distance,
            as a measure of high-dimensionality and distance concentration effects.
    """

    n_rows: int
    n_features: int
    contamination: float
    missing_fraction: float
    max_feature_missing_fraction: float
    outlier_count: int
    inlier_count: int
    mean_abs_correlation: float
    distance_concentration_ratio: float


def compute_descriptive_stats(frame: pd.DataFrame, label_column: str) -> DatasetStats:
    """Compute core descriptive statistics.

    Args:
        frame: Input table with labels.
        label_column: Binary label column name.

    Returns:
        Structured descriptive stats object.
    """
    labels = frame[label_column]
    contamination = float(labels.mean())
    missing_fraction = float(frame.isna().sum().sum()) / float(frame.shape[0] * frame.shape[1])
    feature_missing = frame.drop(columns=[label_column]).isna().mean()
    max_feature_missing_fraction = (
        float(feature_missing.max()) if not feature_missing.empty else 0.0
    )
    outlier_count = int(labels.sum())
    inlier_count = int(frame.shape[0] - outlier_count)
    numeric = frame.drop(columns=[label_column]).select_dtypes(include=["number"])
    mean_abs_correlation = _mean_abs_correlation(numeric)
    dcr = _dataset_distance_concentration_ratio(numeric)
    return DatasetStats(
        n_rows=frame.shape[0],
        n_features=frame.shape[1] - 1,
        contamination=contamination,
        missing_fraction=missing_fraction,
        max_feature_missing_fraction=max_feature_missing_fraction,
        outlier_count=outlier_count,
        inlier_count=inlier_count,
        mean_abs_correlation=mean_abs_correlation,
        distance_concentration_ratio=dcr,
    )


def _mean_abs_correlation(features: pd.DataFrame) -> float:
    """Mean absolute pairwise correlation excluding the diagonal entries.

    Args:
        features: Numeric-only dataframe possibly high-dimensional/sparse correlations.

    Returns:
        Aggregate correlation magnitude in ``[0, 1]`` averaged over upper triangle.
    """
    if features.shape[1] < 2:
        return 0.0
    corr = features.corr(numeric_only=True).abs().values
    if corr.size == 0:
        return 0.0
    upper = corr[np.triu_indices_from(corr, k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.nanmean(upper))


def _dataset_distance_concentration_ratio(features: pd.DataFrame) -> float:
    """Compute distance concentration on complete numeric rows (leading slice)."""
    if features.empty or features.shape[0] < 2:
        return 0.0
    complete = features.dropna()
    if complete.shape[0] < 2:
        return 0.0
    matrix = complete.to_numpy(dtype=np.float64, copy=False)[
        : min(complete.shape[0], DISTANCE_SAMPLE_LIMIT)
    ]
    return float(distance_concentration_ratio(matrix))
