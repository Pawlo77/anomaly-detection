"""Descriptive statistics generation for report inputs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DISTANCE_SAMPLE_LIMIT = 300
"""Maximum rows used when estimating distance concentration."""

DISTANCE_PAIR_COUNT = 500
"""Number of random pairwise distances used for concentration estimate."""

DISTANCE_RANDOM_SEED = 42
"""Deterministic seed for distance concentration sampling."""


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
    distance_concentration_ratio = _distance_concentration_ratio(numeric)
    return DatasetStats(
        n_rows=frame.shape[0],
        n_features=frame.shape[1] - 1,
        contamination=contamination,
        missing_fraction=missing_fraction,
        max_feature_missing_fraction=max_feature_missing_fraction,
        outlier_count=outlier_count,
        inlier_count=inlier_count,
        mean_abs_correlation=mean_abs_correlation,
        distance_concentration_ratio=distance_concentration_ratio,
    )


def _mean_abs_correlation(features: pd.DataFrame) -> float:
    """Compute mean absolute correlation between numeric features."""
    if features.shape[1] < 2:
        return 0.0
    corr = features.corr(numeric_only=True).abs().values
    if corr.size == 0:
        return 0.0
    upper = corr[np.triu_indices_from(corr, k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.nanmean(upper))


def _distance_concentration_ratio(features: pd.DataFrame) -> float:
    """Compute distance concentration ratio from random pairwise distances."""
    if features.empty or features.shape[0] < 3:
        return 0.0
    sample = features.dropna().to_numpy(dtype=float, copy=False)
    if sample.shape[0] < 3:
        return 0.0
    sample = sample[: min(sample.shape[0], DISTANCE_SAMPLE_LIMIT)]
    n_rows = sample.shape[0]
    if n_rows < 3:
        return 0.0

    rng = np.random.default_rng(DISTANCE_RANDOM_SEED)
    row_idx_a = rng.integers(0, n_rows, size=DISTANCE_PAIR_COUNT * 2)
    row_idx_b = rng.integers(0, n_rows, size=DISTANCE_PAIR_COUNT * 2)
    mask = row_idx_a != row_idx_b
    row_idx_a = row_idx_a[mask][:DISTANCE_PAIR_COUNT]
    row_idx_b = row_idx_b[mask][:DISTANCE_PAIR_COUNT]
    if row_idx_a.size == 0:
        return 0.0

    distances = np.linalg.norm(sample[row_idx_a] - sample[row_idx_b], axis=1)
    d_mean = float(np.mean(distances))
    if d_mean == 0.0:
        return 0.0
    return float((np.max(distances) - np.min(distances)) / d_mean)
