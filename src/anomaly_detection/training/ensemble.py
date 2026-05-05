"""Rank-based ensemble primitives for blind-test scoring (experimental plan §5)."""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


def normalize_to_rank(scores: npt.ArrayLike) -> FloatArray:
    """Convert anomaly scores to fractional rank in ``[0, 1]``.

    Args:
        scores: One-dimensional anomaly score vector.

    Returns:
        Fractional rank vector where larger means more anomalous.

    Raises:
        ValueError: When ``scores`` is empty or not one-dimensional.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if values.size == 0:
        raise ValueError("scores must not be empty")
    return np.asarray(rankdata(values) / values.size, dtype=np.float64)


def weighted_borda(
    ranked_scores: Mapping[str, FloatArray],
    weights: Mapping[str, float] | None = None,
) -> FloatArray:
    """Aggregate ranked scores with weighted Borda averaging.

    Args:
        ranked_scores: Mapping from algorithm name to rank vectors.
        weights: Optional algorithm weights defaulting to unit mass.

    Returns:
        Ensemble score vector averaged across contributors.

    Raises:
        ValueError: When mappings are inconsistent or weights are invalid.
    """
    if not ranked_scores:
        raise ValueError("ranked_scores must not be empty")
    size = next(iter(ranked_scores.values())).shape[0]
    total_weight = 0.0
    agg = np.zeros(shape=(size,), dtype=np.float64)
    for algorithm, values in ranked_scores.items():
        if values.shape != (size,):
            raise ValueError("all rank vectors must have equal shape")
        weight = float(weights.get(algorithm, 1.0) if weights is not None else 1.0)
        if weight <= 0.0:
            raise ValueError("weights must be > 0")
        agg += weight * values
        total_weight += weight
    return agg / total_weight


def consensus_labels(
    ranked_scores: Mapping[str, FloatArray],
    ensemble_score: FloatArray,
    contamination_estimate: float,
) -> IntArray:
    """Compute final labels using agreement-first then ensemble threshold.

    Args:
        ranked_scores: Mapping from algorithm to normalized ranks.
        ensemble_score: Aggregated diagnostic score vector aligned with ranks.
        contamination_estimate: Target contamination controlling quantile cutoff.

    Returns:
        Binary labels where ``1`` marks outliers.

    Raises:
        ValueError: When ``contamination_estimate`` falls outside ``(0, 1)``.
    """
    if not 0.0 < contamination_estimate < 1.0:
        raise ValueError("contamination_estimate must be in (0, 1)")
    high_confidence_threshold = 2.0 * contamination_estimate
    if high_confidence_threshold >= 1.0:
        high_confidence_threshold = 0.999

    n_samples = ensemble_score.shape[0]
    agreement = np.zeros(shape=(n_samples,), dtype=np.int64)
    cutoff = 1.0 - high_confidence_threshold
    for values in ranked_scores.values():
        agreement += (values > cutoff).astype(np.int64)

    percentile = np.percentile(ensemble_score, 100.0 * (1.0 - contamination_estimate))
    fallback = (ensemble_score > percentile).astype(np.int64)

    labels = np.where(agreement >= 4, 1, np.where(agreement <= 1, 0, fallback))
    return np.asarray(labels, dtype=np.int64)
